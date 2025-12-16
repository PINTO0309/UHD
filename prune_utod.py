import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Dict, List, Sequence, Tuple
from types import SimpleNamespace
from copy import deepcopy
import subprocess
import sys

import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning.utils import count_ops_and_params

from uhd.models import build_model
from uhd.utils import default_device
from uhd.resize import normalize_resize_mode
from uhd.data import detection_collate
from uhd.metrics import decode_anchor, evaluate_map
from train import parse_classes, load_aug_config, make_datasets


def parse_img_size(arg: str) -> Tuple[int, int]:
    s = str(arg).lower().replace(" ", "")
    if "x" in s:
        h, w = s.split("x")
        return int(h), int(w)
    val = int(float(s))
    return val, val


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _extract_utod_channels(model: nn.Module) -> Dict:
    """Extract actual channel sizes from a pruned UltraTinyOD backbone."""
    bb = getattr(model, "backbone", None)
    if bb is None:
        return {}
    ch = {}
    try:
        ch["stem"] = bb.stem.conv.out_channels
        ch["block1_dw"] = bb.block1.dw.conv.out_channels
        ch["block1_pw"] = bb.block1.pw.conv.out_channels
        ch["block2_dw"] = bb.block2.dw.conv.out_channels
        ch["block2_pw"] = bb.block2.pw.conv.out_channels
        ch["block3_dw"] = bb.block3.dw.conv.out_channels
        ch["block3_pw"] = bb.block3.pw.conv.out_channels
        ch["block4_dw"] = bb.block4.dw.conv.out_channels
        ch["block4_pw"] = bb.block4.pw.conv.out_channels
        ch["sppf_out"] = bb.sppf.cv2.conv.out_channels if hasattr(bb, "sppf") else getattr(bb, "out_channels", None)
    except Exception:
        pass
    return ch


def load_ultratinyod_from_ckpt(ckpt_path: str, device: torch.device, use_ema: bool = False) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise ValueError(f"{ckpt_path} does not look like a training checkpoint (missing 'model').")
    arch = str(ckpt.get("arch", ""))
    if arch.lower() != "ultratinyod":
        raise ValueError(f"Checkpoint arch={arch} is not UltraTinyOD.")
    state = ckpt.get("ema") if use_ema and ckpt.get("ema") is not None else ckpt["model"]
    anchors = ckpt.get("anchors", [])
    num_anchors = int(ckpt.get("num_anchors", len(anchors) if anchors else 3))
    classes = ckpt.get("classes", [0])
    num_classes = len(classes)
    model = build_model(
        "ultratinyod",
        width=int(ckpt.get("cnn_width", ckpt.get("c_stem", 16))),
        num_classes=num_classes,
        anchors=anchors,
        num_anchors=num_anchors,
        output_stride=int(ckpt.get("output_stride", 8)),
        utod_use_residual=bool(ckpt.get("utod_residual", False)),
        use_improved_head=bool(ckpt.get("use_improved_head", False)),
        use_head_ese=bool(ckpt.get("utod_head_ese", False)),
        use_iou_aware_head=bool(ckpt.get("use_iou_aware_head", False)),
        quality_power=float(ckpt.get("quality_power", 1.0)),
        utod_context_rfb=bool(ckpt.get("utod_context_rfb", False)),
        utod_context_dilation=int(ckpt.get("utod_context_dilation", 2)),
        utod_large_obj_branch=bool(ckpt.get("utod_large_obj_branch", False)),
        utod_large_obj_depth=int(ckpt.get("utod_large_obj_depth", 1)),
        utod_large_obj_ch_scale=float(ckpt.get("utod_large_obj_ch_scale", 1.0)),
        activation=ckpt.get("activation", "swish"),
        use_batchnorm=bool(ckpt.get("use_batchnorm", True)),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys while loading: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys while loading: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
    if anchors and hasattr(model, "set_anchors"):
        model.set_anchors(torch.as_tensor(anchors, dtype=torch.float32))
    model.to(device)
    return model, ckpt


def collect_prunable_modules(model: nn.Module, protect_head: bool = True, prune_depthwise: bool = False) -> List[Tuple[str, nn.Module]]:
    prunable: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.groups != 1 and not prune_depthwise:
            continue  # skip depthwise unless explicitly requested
        if protect_head and ("head" in name):
            continue
        if "backbone.sppf.cv2" in name:
            continue  # keep the final feature conv intact
        if "backbone.block4" in name:
            continue
        prunable.append((name, module))
    return prunable


def prune_model(
    model: nn.Module,
    example_inputs: torch.Tensor,
    prune_ratio: float,
    min_channels: int,
    protect_head: bool,
    prune_step: float = 0.0,
    prune_depthwise: bool = False,
) -> List[Dict]:
    model.eval()
    pruned_layers: List[Dict] = []
    orig_out = {
        name: m.out_channels for name, m in collect_prunable_modules(model, protect_head=protect_head, prune_depthwise=prune_depthwise)
    }
    targets = {
        name: max(min_channels, int(round(ch * (1.0 - prune_ratio)))) for name, ch in orig_out.items()
    }
    steps = {name: max(1, int(round(ch * prune_step))) if prune_step > 0 else None for name, ch in orig_out.items()}

    for name, _ in collect_prunable_modules(model, protect_head=protect_head, prune_depthwise=prune_depthwise):
        total_removed = 0
        while True:
            module = dict(model.named_modules()).get(name)
            if module is None or not isinstance(module, nn.Conv2d):
                break
            ch = module.out_channels
            target = targets[name]
            if ch <= target:
                break
            step_remove = ch - target if prune_step <= 0 else min(ch - target, steps[name])
            scores = module.weight.detach().abs().mean(dim=(1, 2, 3))
            idxs = scores.argsort()[:step_remove].tolist()
            with torch.enable_grad():
                dg = tp.DependencyGraph().build_dependency(
                    model,
                    example_inputs=example_inputs,
                    ignored_layers=None,
                    verbose=False,
                )
            group = dg.get_pruning_group(module, tp.prune_conv_out_channels, idxs)
            if not dg.check_pruning_group(group):
                print(f"[skip] pruning group for {name} is not valid; stopping this layer.")
                break

            ops_before, params_before = count_ops_and_params(model, example_inputs=example_inputs)
            group.prune()
            ops_after, params_after = count_ops_and_params(model, example_inputs=example_inputs)
            print(
                f"[step] {name} prune {step_remove} -> {module.out_channels} | "
                f"FLOPs {ops_before/1e9:.2f}G -> {ops_after/1e9:.2f}G, "
                f"Params {params_before/1e6:.2f}M -> {params_after/1e6:.2f}M"
            )
            total_removed += step_remove
        if total_removed > 0:
            module = dict(model.named_modules()).get(name)
            after_ch = module.out_channels if module is not None else None
            pruned_layers.append(
                {"layer": name, "removed": total_removed, "before": orig_out[name], "after": after_ch}
            )
    return pruned_layers


def build_pruned_ckpt(model: nn.Module, meta: Dict, pruning_info: Dict) -> Dict:
    keys_to_keep = [
        "arch",
        "classes",
        "anchors",
        "num_anchors",
        "cnn_width",
        "output_stride",
        "utod_residual",
        "use_improved_head",
        "use_iou_aware_head",
        "utod_head_ese",
        "quality_power",
        "utod_context_rfb",
        "utod_context_dilation",
        "utod_large_obj_branch",
        "utod_large_obj_depth",
        "utod_large_obj_ch_scale",
        "activation",
        "use_batchnorm",
        "resize_mode",
    ]
    new_ckpt = {k: meta[k] for k in keys_to_keep if k in meta}
    new_ckpt["arch"] = "ultratinyod"
    new_ckpt["model"] = model.state_dict()
    new_ckpt["pruning"] = pruning_info
    ch = _extract_utod_channels(model)
    if ch:
        new_ckpt["pruned_channels"] = ch
    return new_ckpt


def run_train_val(ckpt_path: str, args, meta: Dict, ckpt_non_strict: bool = False):
    if not args.image_dir:
        raise ValueError("--image-dir is required for validation via train.py")
    class_arg = args.classes if args.classes is not None else ",".join([str(c) for c in meta.get("classes", [0])])
    arch_arg = meta.get("arch", "ultratinyod")
    cmd = [
        sys.executable,
        "train.py",
        "--arch",
        arch_arg,
        "--val-only",
        "--ckpt",
        ckpt_path,
        "--img-size",
        args.img_size,
        "--image-dir",
        args.image_dir,
        "--classes",
        class_arg,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--conf-thresh",
        str(args.conf_thresh),
        "--train-split",
        str(args.train_split),
        "--val-split",
        str(args.val_split),
    ]
    if args.resize_mode:
        cmd += ["--resize-mode", args.resize_mode]
    if args.aug_config:
        cmd += ["--aug-config", args.aug_config]
    if args.val_count is not None:
        cmd += ["--val-count", str(args.val_count)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.use_ema:
        cmd.append("--use-ema")
    if ckpt_non_strict:
        cmd.append("--ckpt-non-strict")
    print(f"[val/train.py] running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"train.py val-only failed with code {res.returncode}")
    return res.stdout


def run_pruned_validation(
    model: nn.Module,
    meta: Dict,
    device: torch.device,
    img_size: Tuple[int, int],
    args,
):
    class_ids = parse_classes(args.classes) if args.classes is not None else meta.get("classes", [0])
    num_classes = len(class_ids)
    resize_mode = normalize_resize_mode(args.resize_mode or meta.get("resize_mode", "torch_bilinear"))
    try:
        aug_cfg = load_aug_config(args.aug_config)
    except Exception as e:
        print(f"[warn] failed to load aug-config {args.aug_config}: {e}; using empty augment config.")
        aug_cfg = {}
    ds_args = SimpleNamespace(
        image_dir=args.image_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        img_size=f"{img_size[0]}x{img_size[1]}",
        resize_mode=resize_mode,
        aug_config=args.aug_config,
        aug_cfg=aug_cfg,
        val_only=True,
        val_count=args.val_count,
    )
    _, val_ds = make_datasets(ds_args, class_ids, aug_cfg, resize_mode=resize_mode)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
        persistent_workers=False,
    )
    head = getattr(model, "head", None)
    anchors = head.anchors if head is not None else torch.tensor(meta.get("anchors", []), device=device)
    has_quality = bool(getattr(head, "has_quality", False)) if head is not None else False
    wh_scale = head.wh_scale if (head is not None and getattr(head, "use_improved_head", False)) else None
    score_mode = getattr(head, "score_mode", "obj_quality_cls") if head is not None else "obj_quality_cls"
    quality_power = float(getattr(head, "quality_power", 1.0)) if head is not None else 1.0

    model.eval()
    all_preds = []
    all_targets = []
    total_gt = 0
    total_pred = 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            raw = model(imgs)
            decoded = decode_anchor(
                raw,
                anchors=anchors,
                num_classes=num_classes,
                conf_thresh=args.conf_thresh,
                nms_thresh=args.nms_iou,
                has_quality=has_quality,
                wh_scale=wh_scale,
                score_mode=score_mode,
                quality_power=quality_power,
            )
            total_pred += sum(len(p) for p in decoded)
            all_preds.extend([[(s, c, b.detach().cpu()) for (s, c, b) in p] for p in decoded])
            all_targets.extend(targets)
            for tgt in targets:
                total_gt += len(tgt["boxes"])
    metrics = evaluate_map(all_preds, all_targets, num_classes=num_classes, iou_thresh=args.iou_thresh)
    metrics["_val_samples"] = len(val_ds)
    metrics["_val_gt_boxes"] = total_gt
    metrics["_val_pred_boxes"] = total_pred
    print(f"[val(pruned)] gt boxes: {total_gt}, pred boxes: {total_pred}")
    print("[val(pruned)] mAP@0.5: {:.4f}".format(metrics.get("mAP@0.5", 0.0)))
    for k, v in metrics.items():
        if k == "mAP@0.5":
            continue
        if isinstance(v, (int, float)):
            print(f"[val(pruned)] {k}: {v:.4f}")
        else:
            print(f"[val(pruned)] {k}: {v}")
    return metrics


def default_out_path(ckpt_path: str, ratio: float) -> str:
    base, ext = os.path.splitext(ckpt_path)
    tag = f"_pruned_r{int(ratio * 100):02d}"
    return base + tag + (ext if ext else ".pt")


def main():
    parser = argparse.ArgumentParser(description="Prune UltraTinyOD checkpoints with Torch-Pruning.")
    parser.add_argument("--ckpt", required=True, help="Path to UltraTinyOD checkpoint (.pt).")
    parser.add_argument("--out", default=None, help="Output path for pruned checkpoint.")
    parser.add_argument("--img-size", default="64x64", help="Dummy input size HxW for graph tracing.")
    parser.add_argument("--device", default=None, help="cpu or cuda (defaults to cuda if available).")
    parser.add_argument("--prune-ratio", type=float, default=0.2, help="Per-layer output-channel prune ratio (target).")
    parser.add_argument(
        "--prune-step",
        type=float,
        default=0.0,
        help="Optional staged pruning step (e.g., 0.05 prunes ~5% of original channels per pass until reaching --prune-ratio).",
    )
    parser.add_argument("--min-channels", type=int, default=8, help="Minimum output channels to keep per conv.")
    parser.add_argument("--prune-depthwise", action="store_true", help="Also prune depthwise convs (default: skip).")
    parser.add_argument(
        "--no-protect-head",
        dest="protect_head",
        action="store_false",
        default=True,
        help="Allow pruning on head blocks.",
    )
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights from the checkpoint when available.")
    parser.add_argument("--dry-run", action="store_true", help="Do not save; only report pruning results.")
    parser.add_argument("--validate", action="store_true", help="Run a quick validation after pruning.")
    parser.add_argument("--image-dir", default=None, help="Image directory for validation (same as training image-dir).")
    parser.add_argument("--classes", default=None, help="Comma-separated class ids (defaults to checkpoint classes).")
    parser.add_argument("--aug-config", default="uhd/aug.yaml", help="Augmentation config for dataset loading.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split used to form val set.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split.")
    parser.add_argument("--val-count", type=int, default=None, help="Limit number of validation samples.")
    parser.add_argument("--batch-size", type=int, default=64, help="Validation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset split.")
    parser.add_argument("--resize-mode", default=None, help="Resize mode override for validation.")
    parser.add_argument("--conf-thresh", type=float, default=0.15, help="Confidence threshold for decoding/validation.")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold for validation decoding.")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for mAP@0.5.")
    parser.add_argument("--eval-before", action="store_true", help="Run validation on the original checkpoint before pruning.")
    args = parser.parse_args()

    img_h, img_w = parse_img_size(args.img_size)
    device = default_device(args.device)
    print(f"[info] loading {args.ckpt} on {device}")
    model, meta = load_ultratinyod_from_ckpt(args.ckpt, device=device, use_ema=args.use_ema)
    example_inputs = torch.randn(1, 3, img_h, img_w, device=device)
    params_before = count_parameters(model)

    if args.validate and args.eval_before:
        run_train_val(args.ckpt, args, meta, ckpt_non_strict=False)

    pruned_layers = prune_model(
        model,
        example_inputs=example_inputs,
        prune_ratio=float(args.prune_ratio),
        min_channels=int(args.min_channels),
        protect_head=bool(args.protect_head),
        prune_step=float(args.prune_step),
        prune_depthwise=bool(args.prune_depthwise),
    )
    params_after = count_parameters(model)

    model.eval()
    with torch.no_grad():
        _ = model(example_inputs)

    print(f"[done] pruned layers: {len(pruned_layers)}")
    for info in pruned_layers:
        print(f"  {info['layer']}: {info['before']} -> {info['after']} (-{info['removed']})")
    print(f"[stats] params {params_before:,} -> {params_after:,} (delta {-params_before + params_after:,})")

    if args.dry_run:
        return

    out_path = args.out or default_out_path(args.ckpt, ratio=float(args.prune_ratio))
    pruning_info = {
        "source": os.path.abspath(args.ckpt),
        "use_ema": bool(args.use_ema),
        "protect_head": bool(args.protect_head),
        "prune_ratio": float(args.prune_ratio),
        "prune_step": float(args.prune_step),
        "prune_depthwise": bool(args.prune_depthwise),
        "min_channels": int(args.min_channels),
        "img_size": (img_h, img_w),
        "params_before": int(params_before),
        "params_after": int(params_after),
        "layers": pruned_layers,
    }
    val_metrics = None
    if args.validate:
        if not args.image_dir:
            raise ValueError("--image-dir is required when --validate is set.")
        val_metrics = run_pruned_validation(model, meta, device, (img_h, img_w), args)
        pruning_info["val_metrics"] = val_metrics

    pruned_ckpt = build_pruned_ckpt(model, meta, pruning_info)
    torch.save(pruned_ckpt, out_path)
    print(f"[save] wrote pruned checkpoint to {out_path}")
    # Post-prune train.py eval is skipped because channel shapes no longer match the original architecture.


if __name__ == "__main__":
    main()
