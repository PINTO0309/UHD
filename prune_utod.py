import argparse
import os
from typing import Dict, List, Sequence, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning.utils import count_ops_and_params
from torch.utils.data import DataLoader

from uhd.models import build_model
from uhd.utils import default_device
from uhd.data import detection_collate
from uhd.metrics import evaluate_map
from train import parse_classes, load_aug_config, normalize_resize_mode, make_datasets


def parse_img_size(arg: str) -> Tuple[int, int]:
    s = str(arg).lower().replace(" ", "")
    if "x" in s:
        h, w = s.split("x")
        return int(h), int(w)
    val = int(float(s))
    return val, val


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


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


def collect_prunable_modules(model: nn.Module, protect_head: bool = True) -> List[Tuple[str, nn.Module]]:
    prunable: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.groups != 1:
            continue  # skip depthwise
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
) -> List[Dict]:
    model.eval()
    pruned_layers: List[Dict] = []
    orig_out = {
        name: m.out_channels for name, m in collect_prunable_modules(model, protect_head=protect_head)
    }
    targets = {
        name: max(min_channels, int(round(ch * (1.0 - prune_ratio)))) for name, ch in orig_out.items()
    }
    steps = {name: max(1, int(round(ch * prune_step))) if prune_step > 0 else None for name, ch in orig_out.items()}

    for name, _ in collect_prunable_modules(model, protect_head=protect_head):
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
    return new_ckpt


def run_validation(
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
    # make_datasets expects aug_cfg separately
    train_ds, val_ds = make_datasets(ds_args, class_ids, aug_cfg, resize_mode=resize_mode)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
        persistent_workers=False,
    )
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            raw, decoded = model(imgs, decode=True)
            # Optional post-filter by confidence (forward decode uses default 0.3).
            if decoded is None:
                decoded = [[] for _ in range(imgs.size(0))]
            if args.conf_thresh is not None:
                decoded = [
                    [(score, cls, box) for (score, cls, box) in pred if score >= args.conf_thresh]
                    for pred in decoded
                ]
            preds_cpu = []
            for p_img in decoded:
                preds_cpu.append([(score, cls, box.detach().cpu()) for score, cls, box in p_img])
            all_preds.extend(preds_cpu)
            all_targets.extend(targets)
    metrics = evaluate_map(all_preds, all_targets, num_classes=num_classes, iou_thresh=args.iou_thresh)
    print("[val] mAP@0.5: {:.4f}".format(metrics.get("mAP@0.5", 0.0)))
    for k, v in metrics.items():
        if k == "mAP@0.5":
            continue
        print(f"[val] {k}: {v:.4f}")
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
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for mAP@0.5.")
    parser.add_argument("--conf-thresh", type=float, default=0.15, help="Confidence threshold for decoding.")
    args = parser.parse_args()

    img_h, img_w = parse_img_size(args.img_size)
    device = default_device(args.device)
    print(f"[info] loading {args.ckpt} on {device}")
    model, meta = load_ultratinyod_from_ckpt(args.ckpt, device=device, use_ema=args.use_ema)
    example_inputs = torch.randn(1, 3, img_h, img_w, device=device)
    params_before = count_parameters(model)

    pruned_layers = prune_model(
        model,
        example_inputs=example_inputs,
        prune_ratio=float(args.prune_ratio),
        min_channels=int(args.min_channels),
        protect_head=bool(args.protect_head),
        prune_step=float(args.prune_step),
    )
    params_after = count_parameters(model)

    model.eval()
    with torch.no_grad():
        _ = model(example_inputs)

    print(f"[done] pruned layers: {len(pruned_layers)}")
    for info in pruned_layers:
        print(f"  {info['layer']}: {info['before']} -> {info['after']} (-{info['removed']})")
    print(f"[stats] params {params_before:,} -> {params_after:,} (delta {-params_before + params_after:,})")

    val_metrics = None
    if args.validate:
        if not args.image_dir:
            raise ValueError("--image-dir is required when --validate is set.")
        val_metrics = run_validation(model, meta, device, (img_h, img_w), args)

    if args.dry_run:
        return

    out_path = args.out or default_out_path(args.ckpt, ratio=float(args.prune_ratio))
    pruning_info = {
        "source": os.path.abspath(args.ckpt),
        "use_ema": bool(args.use_ema),
        "protect_head": bool(args.protect_head),
        "prune_ratio": float(args.prune_ratio),
        "prune_step": float(args.prune_step),
        "min_channels": int(args.min_channels),
        "img_size": (img_h, img_w),
        "params_before": int(params_before),
        "params_after": int(params_after),
        "layers": pruned_layers,
    }
    if val_metrics is not None:
        pruning_info["val_metrics"] = val_metrics
    pruned_ckpt = build_pruned_ckpt(model, meta, pruning_info)
    torch.save(pruned_ckpt, out_path)
    print(f"[save] wrote pruned checkpoint to {out_path}")


if __name__ == "__main__":
    main()
