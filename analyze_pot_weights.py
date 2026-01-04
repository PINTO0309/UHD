#!/usr/bin/env python3
import argparse
import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from export_cpp import fold_bn_state_dict
from export_onnx import infer_utod_config, load_checkpoint
from uhd.resize import Y_BIN_RESIZE_MODE, Y_ONLY_RESIZE_MODE, Y_TRI_RESIZE_MODE, YUV422_RESIZE_MODE, normalize_resize_mode
from uhd.ultratinyod import ConvBNAct, UltraTinyOD


def _infer_input_channels(meta: Dict) -> int:
    try:
        resize_mode = normalize_resize_mode(meta.get("resize_mode", "opencv_inter_nearest"))
    except Exception:
        resize_mode = "opencv_inter_nearest"
    if resize_mode == YUV422_RESIZE_MODE:
        return 2
    if resize_mode in (Y_ONLY_RESIZE_MODE, Y_BIN_RESIZE_MODE, Y_TRI_RESIZE_MODE):
        return 1
    return 3


def _load_utod_model(state: Dict[str, torch.Tensor], meta: Dict, args: argparse.Namespace) -> UltraTinyOD:
    cfg, inferred = infer_utod_config(state, meta, args)
    model = UltraTinyOD(
        num_classes=inferred["num_classes"],
        config=cfg,
        c_stem=inferred["c_stem"],
        in_channels=_infer_input_channels(meta),
        use_residual=inferred["use_residual"],
        use_improved_head=inferred["use_improved_head"],
        use_head_ese=inferred["use_head_ese"],
        use_iou_aware_head=inferred.get("use_iou_aware_head", False),
        quality_power=inferred.get("quality_power", 1.0),
        activation=inferred.get("activation", "swish"),
    )
    try:
        model.load_state_dict(state, strict=not args.non_strict)
    except RuntimeError as exc:
        if "head.head_se" in str(exc):
            print("[WARN] head_se weights missing in checkpoint; loading non-strict for eSE and keeping initializer.")
            model.load_state_dict(state, strict=False)
        else:
            raise
    model.eval()
    if inferred.get("anchors") is not None:
        anchors_tensor = torch.as_tensor(inferred["anchors"], dtype=torch.float32)
        if anchors_tensor.ndim == 1:
            anchors_tensor = anchors_tensor.view(-1, 2)
        if anchors_tensor.ndim == 2 and anchors_tensor.shape[1] == 2:
            model.head.set_anchors(anchors_tensor)
    return model


def _iter_convbnact_1x1(model: torch.nn.Module) -> Iterable[Tuple[str, ConvBNAct]]:
    for name, module in model.named_modules():
        if not isinstance(module, ConvBNAct):
            continue
        k = module.conv.kernel_size
        if isinstance(k, tuple):
            k_h, k_w = k
        else:
            k_h = k_w = int(k)
        if k_h == 1 and k_w == 1:
            yield name, module


def _weights_from_state(state: Dict[str, torch.Tensor], name: str) -> Optional[np.ndarray]:
    key = f"{name}.conv.weight" if name else "conv.weight"
    weight = state.get(key)
    if weight is None:
        return None
    return weight.detach().cpu().to(torch.float32).numpy()


def _flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    if not weights:
        return np.array([], dtype=np.float32)
    flat = np.concatenate([w.reshape(-1) for w in weights], axis=0)
    return flat.astype(np.float32, copy=False)


def _percentiles(arr: np.ndarray, pts: Iterable[float]) -> List[float]:
    if arr.size == 0:
        return [float("nan")] * len(list(pts))
    return [float(v) for v in np.percentile(arr, list(pts))]


def _format_pct(label: str, pts: List[float], vals: List[float], fmt: str) -> str:
    items = []
    for p, v in zip(pts, vals):
        if math.isnan(v):
            items.append(f"p{p:g}=nan")
        else:
            items.append(f"p{p:g}=" + fmt.format(v))
    return f"{label}: " + ", ".join(items)


def _summarize_weights(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    abs_w = np.abs(weights)
    nz = abs_w > 0
    log2_abs = np.log2(abs_w[nz]) if np.any(nz) else np.array([], dtype=np.float32)
    return abs_w, log2_abs, int(np.sum(nz))


def _print_summary(weights: np.ndarray, k_min: Optional[int], k_max: Optional[int]) -> None:
    if weights.size == 0:
        print("No ConvBNAct(1x1) weights found.")
        return
    abs_w, log2_abs, nz_count = _summarize_weights(weights)
    zero_count = int(weights.size - nz_count)
    print(f"Collected {weights.size} weights ({zero_count} zeros).")

    abs_pts = [0, 1, 5, 50, 95, 99, 100]
    abs_vals = _percentiles(abs_w, abs_pts)
    print(_format_pct("abs|w|", abs_pts, abs_vals, "{:.6e}"))

    log_pts = [0, 1, 5, 50, 95, 99, 99.9, 100]
    log_vals = _percentiles(log2_abs, log_pts)
    print(_format_pct("log2|w|", log_pts, log_vals, "{:.3f}"))

    if log2_abs.size > 0:
        p01, p999 = np.percentile(log2_abs, [0.1, 99.9])
        sugg_min = int(math.floor(p01))
        sugg_max = int(math.ceil(p999))
        print(f"suggested k range (p0.1..p99.9): [{sugg_min}, {sugg_max}]")

    if k_min is not None and k_max is not None and log2_abs.size > 0:
        below = int(np.sum(log2_abs < k_min))
        above = int(np.sum(log2_abs > k_max))
        total = int(log2_abs.size)
        below_pct = 100.0 * below / total
        above_pct = 100.0 * above / total
        print(f"coverage for k in [{k_min}, {k_max}]: below={below} ({below_pct:.2f}%), above={above} ({above_pct:.2f}%)")


def _print_per_layer(layer_stats: List[Tuple[str, np.ndarray]]) -> None:
    pts = [1, 50, 99]
    for name, weights in layer_stats:
        abs_w, log2_abs, nz_count = _summarize_weights(weights)
        if abs_w.size == 0:
            continue
        abs_vals = _percentiles(abs_w, pts)
        log_vals = _percentiles(log2_abs, pts)
        zero_count = int(weights.size - nz_count)
        print(
            f"{name}: n={weights.size}, zero={zero_count}, "
            f"abs(p1/p50/p99)={abs_vals[0]:.3e}/{abs_vals[1]:.3e}/{abs_vals[2]:.3e}, "
            f"log2(p1/p50/p99)={log_vals[0]:.2f}/{log_vals[1]:.2f}/{log_vals[2]:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ConvBNAct(1x1) weights for PoT/SPoT range selection.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if present.")
    parser.add_argument("--non-strict", action="store_true", help="Allow missing/unexpected keys when loading.")
    parser.add_argument("--no-fold-bn", action="store_true", help="Disable Conv+BN folding (default: fold BN).")
    parser.add_argument("--k-min", type=int, default=None, help="Optional k_min to report coverage (log2|w| < k_min).")
    parser.add_argument("--k-max", type=int, default=None, help="Optional k_max to report coverage (log2|w| > k_max).")
    parser.add_argument("--per-layer", action="store_true", help="Print per-layer stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state, meta = load_checkpoint(args.ckpt, use_ema=bool(args.use_ema))
    model = _load_utod_model(state, meta, args)
    if not args.no_fold_bn:
        state = fold_bn_state_dict(model, state)

    layer_stats: List[Tuple[str, np.ndarray]] = []
    for name, _module in _iter_convbnact_1x1(model):
        weights = _weights_from_state(state, name)
        if weights is None:
            continue
        layer_stats.append((name, weights))

    print(f"ConvBNAct(1x1) layers found: {len(layer_stats)}")
    all_weights = _flatten_weights([w for _, w in layer_stats])
    _print_summary(all_weights, args.k_min, args.k_max)
    if args.per_layer:
        _print_per_layer(layer_stats)


if __name__ == "__main__":
    main()
