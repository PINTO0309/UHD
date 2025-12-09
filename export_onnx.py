#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from uhd.ultratinyod import UltraTinyOD, UltraTinyODConfig


def parse_img_size(arg: str) -> Tuple[int, int]:
    arg = str(arg).lower().replace(" ", "")
    if "x" in arg:
        h, w = arg.split("x")
        return int(float(h)), int(float(w))
    v = int(float(arg))
    return v, v


def _is_state_dict(obj) -> bool:
    return isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values())


def load_checkpoint(path: str, use_ema: bool) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Returns (state_dict, meta)."""
    ckpt = torch.load(path, map_location="cpu")
    meta: Dict = ckpt if isinstance(ckpt, dict) else {}
    state: Optional[Dict[str, torch.Tensor]] = None
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["ema"] if use_ema and "ema" in ckpt else ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif _is_state_dict(ckpt):
            state = ckpt
    if state is None:
        raise ValueError(f"Unsupported checkpoint format for {path}")
    return state, meta


def infer_utod_config(state: Dict[str, torch.Tensor], meta: Dict, args) -> Tuple[UltraTinyODConfig, Dict]:
    anchors_tensor = state.get("head.anchors", None)
    anchors_from_meta = meta.get("anchors") if isinstance(meta, dict) else None
    anchors: Optional[Sequence[Tuple[float, float]]] = None
    num_anchors = anchors_tensor.shape[0] if anchors_tensor is not None else None
    anchors_source = "default"
    anchors_np = anchors_tensor.cpu().numpy() if anchors_tensor is not None else None
    if anchors_np is not None:
        anchors = anchors_np.tolist()
        num_anchors = anchors_tensor.shape[0]
        anchors_source = "state"
    elif anchors_from_meta:
        anchors = anchors_from_meta
        num_anchors = len(anchors_from_meta)
        anchors_source = "meta"

    # num_classes: meta > CLI > derive from cls_out shape
    if "classes" in meta and meta["classes"]:
        num_classes = len(meta["classes"])
    else:
        cls_weight = state.get("head.cls_out.weight")
        if cls_weight is not None and num_anchors:
            num_classes = int(cls_weight.shape[0] // num_anchors)
        else:
            num_classes = 1

    # backbone width from meta > weight shape
    if "cnn_width" in meta:
        c_stem = int(meta["cnn_width"])
    else:
        stem_w = state.get("backbone.stem.conv.weight")
        c_stem = int(stem_w.shape[0]) if stem_w is not None else 16

    stride = int(meta.get("output_stride") or 8)

    use_improved_head = bool(
        meta.get("use_improved_head")
        or any(k.startswith("head.quality") for k in state.keys())
    )
    use_iou_aware_head = bool(
        meta.get("use_iou_aware_head")
        or any(k.startswith("head.box_tower") or k.startswith("head.quality_tower") or k.startswith("head.cls_tower") for k in state.keys())
    )
    quality_power = max(0.0, float(meta.get("quality_power", 1.0)))
    activation = str(meta.get("activation", "swish") or "swish")
    has_ese_weights = ("head.head_se.fc.weight" in state) and ("head.head_se.fc.bias" in state)
    if meta.get("utod_head_ese") and not has_ese_weights:
        print("[WARN] eSE requested in checkpoint meta but head.head_se weights missing; disabling eSE for export.")
    use_head_ese = bool(has_ese_weights)
    use_residual = bool(meta.get("utod_residual") or "backbone.block3_skip.conv.weight" in state)
    use_context_rfb = bool(meta.get("utod_context_rfb") or any(k.startswith("head.head_rfb") for k in state.keys()))
    context_dilation = int(meta.get("utod_context_dilation", 2))
    use_large_obj_branch = bool(meta.get("utod_large_obj_branch") or any(k.startswith("head.large_obj_") for k in state.keys()))
    large_obj_depth = int(meta.get("utod_large_obj_depth", 1))
    large_obj_ch_scale = float(meta.get("utod_large_obj_ch_scale", 1.0))

    cfg = UltraTinyODConfig(
        num_classes=num_classes,
        anchors=anchors,
        stride=stride,
        use_improved_head=use_improved_head,
        use_head_ese=use_head_ese,
        use_iou_aware_head=use_iou_aware_head,
        quality_power=quality_power,
        activation=activation,
        use_context_rfb=use_context_rfb,
        context_dilation=context_dilation,
        use_large_obj_branch=use_large_obj_branch,
        large_obj_branch_depth=large_obj_depth,
        large_obj_branch_expansion=large_obj_ch_scale,
    )
    overrides = {
        "num_classes": num_classes,
        "anchors": anchors,
        "stride": stride,
        "c_stem": c_stem,
        "use_improved_head": use_improved_head,
        "use_head_ese": use_head_ese,
        "use_residual": use_residual,
        "use_iou_aware_head": use_iou_aware_head,
        "quality_power": quality_power,
        "anchors_source": anchors_source,
        "activation": activation,
        "use_context_rfb": use_context_rfb,
        "context_dilation": context_dilation,
        "use_large_obj_branch": use_large_obj_branch,
        "large_obj_branch_depth": large_obj_depth,
        "large_obj_branch_expansion": large_obj_ch_scale,
    }
    return cfg, overrides


class UltraTinyODWithPost(nn.Module):
    def __init__(self, model: UltraTinyOD, topk: int = 100, conf_thresh: float = 0.0) -> None:
        super().__init__()
        self.model = model
        self.topk = int(topk)
        self.conf_thresh = float(conf_thresh)
        self.has_quality = bool(getattr(model.head, "has_quality", False))
        self.score_mode = getattr(model, "score_mode", getattr(model.head, "score_mode", "obj_quality_cls"))
        self.quality_power = float(getattr(model, "quality_power", getattr(model.head, "quality_power", 1.0)))

    def forward(self, x: torch.Tensor):
        raw = self.model(x, decode=False)  # [B, na*no, H, W]
        if isinstance(raw, tuple):
            raw = raw[0]
        b, _, h, w = raw.shape
        na = self.model.num_anchors
        no = self.model.head.no
        pred = raw.view(b, na, no, h, w).permute(0, 1, 3, 4, 2)

        tx = pred[..., 0]
        ty = pred[..., 1]
        tw = pred[..., 2]
        th = pred[..., 3]
        obj = pred[..., 4].sigmoid()
        cls_start = 6 if self.has_quality else 5
        quality = pred[..., 5].sigmoid() if self.has_quality else None
        cls_logits = pred[..., cls_start:]
        cls_scores = cls_logits.sigmoid()

        quality_use = quality
        if quality_use is not None and self.quality_power != 1.0:
            quality_use = torch.pow(quality_use, self.quality_power)
        smode = (self.score_mode or "obj_quality_cls").lower()
        if smode == "quality_cls" and quality_use is not None:
            score_base = quality_use
        elif smode == "obj_cls":
            score_base = obj
        else:
            score_base = obj
            if quality_use is not None:
                score_base = score_base * quality_use

        # grid
        gy, gx = torch.meshgrid(
            torch.arange(h, device=pred.device),
            torch.arange(w, device=pred.device),
            indexing="ij",
        )
        gx = gx.view(1, 1, h, w)
        gy = gy.view(1, 1, h, w)

        anchors = self.model.head.anchors.to(pred.device)
        if self.model.use_improved_head:
            anchors = anchors * self.model.head.wh_scale.to(pred.device)
        pw = anchors[:, 0].view(1, na, 1, 1)
        ph = anchors[:, 1].view(1, na, 1, 1)

        cx = (tx.sigmoid() + gx) / float(w)
        cy = (ty.sigmoid() + gy) / float(h)
        bw = pw * F.softplus(tw)
        bh = ph * F.softplus(th)

        scores = score_base.unsqueeze(-1) * cls_scores  # [B, A, H, W, C]
        if self.conf_thresh > 0:
            scores = torch.where(scores >= self.conf_thresh, scores, torch.zeros_like(scores))
        best_scores, best_cls = scores.max(dim=-1)  # [B, A, H, W]

        best_scores_flat = best_scores.view(b, -1)
        best_cls_flat = best_cls.view(b, -1)
        cx_flat = cx.view(b, -1)
        cy_flat = cy.view(b, -1)
        bw_flat = bw.view(b, -1)
        bh_flat = bh.view(b, -1)

        k = min(self.topk, best_scores_flat.shape[1])
        top_scores, top_idx = torch.topk(best_scores_flat, k=k, dim=1)
        top_cls = torch.gather(best_cls_flat, 1, top_idx)
        top_cx = torch.gather(cx_flat, 1, top_idx)
        top_cy = torch.gather(cy_flat, 1, top_idx)
        top_bw = torch.gather(bw_flat, 1, top_idx)
        top_bh = torch.gather(bh_flat, 1, top_idx)

        detections = torch.stack(
            [top_scores, top_cls.float(), top_cx, top_cy, top_bw, top_bh],
            dim=-1,
        )
        return detections


class UltraTinyODRawWithAnchors(nn.Module):
    """
    Wrap UltraTinyOD to export raw logits together with anchors/wh_scale so post-process can be done externally.
    """

    def __init__(self, model: UltraTinyOD) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x, decode=False)
        raw = out[0] if isinstance(out, (tuple, list)) else out
        anchors = self.model.head.anchors
        wh_scale = self.model.head.wh_scale
        return raw, anchors, wh_scale


def export_onnx(
    model: nn.Module,
    output_path: str,
    img_size: Tuple[int, int],
    opset: int,
    simplify: bool = True,
    output_names: Optional[Sequence[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> None:
    model.eval()
    h, w = img_size
    dummy = torch.zeros(1, 3, h, w, device=next(model.parameters()).device)
    input_names = ["images"]
    if output_names is None:
        output_names = ["detections"]
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[WARN] Failed to import onnx/onnxsim for simplification: {exc}")
            return
        model_onnx = onnx.load(output_path)
        model_simplified, check = onnx_simplify(model_onnx)
        if not check:
            print("[WARN] onnx-simplifier check failed; keeping original export.")
            return
        onnx.save(model_simplified, output_path)


def verify_outputs(model: UltraTinyODWithPost, onnx_path: str, img_size: Tuple[int, int]) -> Dict[str, float]:
    import numpy as np
    import onnxruntime as ort

    h, w = img_size
    torch.manual_seed(0)
    sample = torch.randn(1, 3, h, w)
    with torch.no_grad():
        torch_out = model(sample)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {"images": sample.numpy()})

    deltas = {}
    ref = torch_out.detach().cpu().numpy()
    o_out = ort_outs[0]
    scores_ref = ref[..., 0]
    scores_onnx = o_out[..., 0]
    active_mask = (scores_ref > 1e-3) & (scores_onnx > 1e-3)
    full_delta = float(np.max(np.abs(ref - o_out)))
    if active_mask.any():
        active_delta = float(np.max(np.abs(ref[active_mask] - o_out[active_mask])))
    else:
        active_delta = full_delta
    deltas["detections"] = full_delta
    deltas["detections_active"] = active_delta
    return deltas


def build_argparser():
    parser = argparse.ArgumentParser(description="Export UltraTinyOD to ONNX with merged postprocess.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Training checkpoint (.pt) containing optimizer/ema/etc.")
    src.add_argument("--weights", help="Weights/state_dict only (.pt).")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--img-size", default="64x64", help="Input size as HxW (e.g., 64x64).")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--topk", type=int, default=100, help="Number of detections to keep (Top-K).")
    parser.add_argument("--conf-thresh", type=float, default=0.0, help="Optional confidence threshold before Top-K.")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false", help="Use raw model weights instead of EMA (defaults to EMA when available).")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--no-merge-postprocess", dest="merge_postprocess", action="store_false", help="Export raw model only.")
    parser.set_defaults(merge_postprocess=True)
    parser.add_argument("--no-simplify", action="store_true", help="Skip onnx-simplifier.")
    parser.add_argument("--non-strict", action="store_true", help="Load weights with strict=False.")
    parser.add_argument("--verify", action="store_true", help="Run a quick ONNXRuntime vs PyTorch diff check.")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    ckpt_path = args.checkpoint or args.weights
    state, meta = load_checkpoint(ckpt_path, use_ema=bool(args.use_ema))
    cfg, inferred = infer_utod_config(state, meta, args)

    model = UltraTinyOD(
        num_classes=inferred["num_classes"],
        config=cfg,
        c_stem=inferred["c_stem"],
        use_residual=inferred["use_residual"],
        use_improved_head=inferred["use_improved_head"],
        use_head_ese=inferred["use_head_ese"],
        use_iou_aware_head=inferred.get("use_iou_aware_head", False),
        quality_power=inferred.get("quality_power", 1.0),
        activation=inferred.get("activation", "swish"),
    )
    try:
        model.load_state_dict(state, strict=not args.non_strict)
    except RuntimeError as e:
        if "head.head_se" in str(e):
            print("[WARN] head_se weights missing in checkpoint; loading non-strict for eSE and keeping initializer.")
            model.load_state_dict(state, strict=False)
        else:
            raise
    model.eval()

    # Ensure anchors in the model match the chosen source (state/override/meta)
    if inferred.get("anchors") is not None:
        anchors_tensor = torch.as_tensor(inferred["anchors"], dtype=torch.float32)
        if anchors_tensor.ndim == 1:
            anchors_tensor = anchors_tensor.view(-1, 2)
        if anchors_tensor.ndim == 2 and anchors_tensor.shape[1] == 2:
            model.head.set_anchors(anchors_tensor)

    if not args.merge_postprocess:
        export_module = UltraTinyODRawWithAnchors(model)
        output_names = ["pred", "anchors", "wh_scale"]
    else:
        export_module = UltraTinyODWithPost(model, topk=args.topk, conf_thresh=args.conf_thresh)
        output_names = ["detections"]

    device = torch.device("cpu")
    export_module.to(device)

    img_size = parse_img_size(args.img_size)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    export_onnx(
        export_module,
        args.output,
        img_size=img_size,
        opset=int(args.opset),
        simplify=not args.no_simplify,
        output_names=output_names,
        dynamic_axes=None,
    )
    print(f"Exported ONNX to {args.output}")

    if args.verify and args.merge_postprocess:
        deltas = verify_outputs(export_module, args.output, img_size=img_size)
        print("Verification max abs diff:", deltas)


if __name__ == "__main__":
    main()
