#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from export_onnx import infer_utod_config, load_checkpoint
from uhd.resize import (
    Y_BIN_RESIZE_MODE,
    Y_ONLY_RESIZE_MODE,
    Y_TRI_RESIZE_MODE,
    YUV422_RESIZE_MODE,
    normalize_resize_mode,
    resize_image_numpy,
)
from uhd.ultratinyod import ConvBNAct, UltraTinyOD


def _sanitize_identifier(name: str, used: Dict[str, int]) -> str:
    ident = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"t_{ident}"
    count = used.get(ident, 0)
    used[ident] = count + 1
    if count:
        ident = f"{ident}_{count}"
    return ident


def _tensor_to_numpy(t: torch.Tensor) -> Tuple[np.ndarray, str]:
    if t.is_floating_point():
        arr = t.detach().cpu().to(torch.float32).numpy()
        return arr, "UHD_F32"
    if t.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
        arr = t.detach().cpu().to(torch.int32).numpy()
        return arr, "UHD_I32"
    arr = t.detach().cpu().to(torch.float32).numpy()
    return arr, "UHD_F32"


def _write_array(
    f,
    c_name: str,
    arr: np.ndarray,
    c_type: str,
    max_per_line: int = 8,
    float_fmt: str = "{:.8e}f",
) -> None:
    f.write(f"static const {c_type} {c_name}[] = {{\n")
    flat = arr.reshape(-1)
    if flat.size == 0:
        f.write("};\n\n")
        return
    for i in range(0, flat.size, max_per_line):
        chunk = flat[i : i + max_per_line]
        if c_type == "float":
            items = [float_fmt.format(float(v)) for v in chunk]
        else:
            items = [str(int(v)) for v in chunk]
        f.write("    " + ", ".join(items) + ",\n")
    f.write("};\n\n")


def _write_shape(f, c_name: str, shape: Sequence[int]) -> None:
    dims = ", ".join(str(int(d)) for d in shape)
    f.write(f"static const int32_t {c_name}[] = {{{dims}}};\n")


def _infer_input_channels(meta: Dict) -> int:
    resize_mode = normalize_resize_mode(meta.get("resize_mode") or "torch_bilinear")
    if resize_mode == YUV422_RESIZE_MODE:
        return 2
    if resize_mode in (Y_ONLY_RESIZE_MODE, Y_BIN_RESIZE_MODE, Y_TRI_RESIZE_MODE):
        return 1
    return 3


def _parse_img_size(arg: str) -> Tuple[int, int]:
    arg = str(arg).lower().replace(" ", "")
    if "x" in arg:
        h, w = arg.split("x")
        return int(float(h)), int(float(w))
    v = int(float(arg))
    return v, v


def _list_calib_images(calib_dir: Optional[str], calib_list: Optional[str], limit: int) -> List[str]:
    paths: List[str] = []
    if calib_list:
        with open(calib_list, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                paths.append(line)
    if calib_dir:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for root, _, files in os.walk(calib_dir):
            for name in files:
                if name.lower().endswith(exts):
                    paths.append(os.path.join(root, name))
    paths = sorted(dict.fromkeys(paths))
    if limit > 0:
        paths = paths[:limit]
    return paths


def _load_calib_tensor(path: str, size: Tuple[int, int], resize_mode: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    np_img = np.asarray(img).astype(np.float32) / 255.0
    np_img = resize_image_numpy(np_img, size=size, mode=resize_mode)
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
    return tensor


def _collect_activation_scales(
    model: nn.Module,
    image_paths: Sequence[str],
    size: Tuple[int, int],
    resize_mode: str,
    batch_size: int,
    lowbit_quant_target: str,
    lowbit_a_bits: int,
    highbit_quant_target: str,
    highbit_a_bits: int,
) -> List[Dict[str, object]]:
    max_abs: Dict[str, float] = {}
    hooks = []

    def _hook(name: str):
        def _fn(_, __, output):
            if isinstance(output, (tuple, list)):
                out = output[0]
            else:
                out = output
            if not isinstance(out, torch.Tensor):
                return
            val = float(out.detach().abs().max().cpu())
            prev = max_abs.get(name, 0.0)
            if val > prev:
                max_abs[name] = val
        return _fn

    bits_map: Dict[str, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, ConvBNAct):
            continue
        bit_val = _quant_bits_for_name(
            name,
            lowbit_quant_target,
            highbit_quant_target,
            lowbit_a_bits,
            highbit_a_bits,
        )
        if bit_val >= 2:
            bits_map[name] = int(bit_val)
            hooks.append(module.register_forward_hook(_hook(name)))

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            if not batch_paths:
                continue
            imgs = []
            for path in batch_paths:
                try:
                    imgs.append(_load_calib_tensor(path, size, resize_mode))
                except Exception:
                    continue
            if not imgs:
                continue
            batch = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)
            model(batch)

    for handle in hooks:
        handle.remove()

    scales: List[Dict[str, object]] = []
    for name, amax in sorted(max_abs.items()):
        bit_val = int(bits_map.get(name, 0))
        qmax = (1 << (bit_val - 1)) - 1 if bit_val >= 2 else 0
        if qmax > 0:
            scale = float(amax) / float(qmax) if amax > 0 else 1.0
        else:
            scale = float(amax)
        scales.append({"name": name, "amax": float(amax), "scale": float(scale), "bits": bit_val})
    return scales


def _match_quant_target(name: str, quant_target: str) -> bool:
    quant_target = str(quant_target or "both").lower()
    if quant_target == "both":
        return name.startswith("backbone.") or name.startswith("head.")
    if quant_target == "backbone":
        return name.startswith("backbone.")
    if quant_target == "head":
        return name.startswith("head.")
    if quant_target == "none":
        return False
    return name.startswith("backbone.") or name.startswith("head.")


def _quant_bits_for_name(
    name: str,
    lowbit_target: str,
    highbit_target: str,
    low_bits: int,
    high_bits: int,
) -> int:
    in_low = _match_quant_target(name, lowbit_target)
    in_high = _match_quant_target(name, highbit_target)
    if in_low and in_high:
        raise ValueError("lowbit-quant-target and highbit-quant-target overlap for module: " + name)
    if in_high:
        return int(high_bits)
    if in_low:
        return int(low_bits)
    return 0


def _quantize_weights(
    arr: np.ndarray,
    bits: int,
    per_channel: bool = True,
    ch_axis: int = 0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, str]:
    bits = int(bits)
    if bits < 2:
        raise ValueError("bits must be >= 2 for quantization.")
    if bits <= 8:
        q_dtype = np.int8
        dtype_str = "UHD_I8"
    elif bits <= 16:
        q_dtype = np.int16
        dtype_str = "UHD_I16"
    else:
        raise ValueError("bits > 16 not supported for weight quantization.")
    qmax = (1 << (bits - 1)) - 1
    if per_channel:
        max_val = np.max(np.abs(arr), axis=tuple(i for i in range(arr.ndim) if i != ch_axis), keepdims=True)
    else:
        max_val = np.max(np.abs(arr))
    scale = max_val / float(qmax)
    scale = np.maximum(scale, eps)
    q = np.round(arr / scale)
    q = np.clip(q, -qmax, qmax).astype(q_dtype)
    if per_channel:
        scale = np.squeeze(scale, axis=tuple(i for i in range(scale.ndim) if i != ch_axis))
    return q, scale.astype(np.float32), dtype_str


def _quantize_bias(
    arr: np.ndarray,
    bits: int = 16,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    bits = int(bits)
    if bits < 2:
        raise ValueError("bits must be >= 2 for bias quantization.")
    qmax = (1 << (bits - 1)) - 1
    max_val = np.max(np.abs(arr))
    scale = max(max_val / float(qmax), eps)
    q = np.round(arr / scale)
    q = np.clip(q, -qmax, qmax).astype(np.int16)
    return q, np.array([scale], dtype=np.float32)


def _pack_quantized(q_arr: np.ndarray, bits: int) -> Tuple[np.ndarray, int]:
    bits = int(bits)
    if bits not in (2, 4):
        return q_arr, 0
    qmax = (1 << (bits - 1)) - 1
    offset = qmax
    flat = q_arr.reshape(-1).astype(np.int16) + offset
    if bits == 4:
        if flat.size % 2:
            flat = np.concatenate([flat, np.zeros(1, dtype=flat.dtype)])
        lo = flat[0::2] & 0x0F
        hi = (flat[1::2] & 0x0F) << 4
        packed = (lo | hi).astype(np.uint8)
        return packed, bits
    # bits == 2
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    v0 = (flat[0::4] & 0x03)
    v1 = (flat[1::4] & 0x03) << 2
    v2 = (flat[2::4] & 0x03) << 4
    v3 = (flat[3::4] & 0x03) << 6
    packed = (v0 | v1 | v2 | v3).astype(np.uint8)
    return packed, bits


def _fold_conv_bn(
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_std = torch.rsqrt(running_var + eps)
    scale = bn_weight * inv_std
    shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
    weight_fused = weight * scale.reshape(shape)
    bias_fused = (bias - running_mean) * scale + bn_bias
    return weight_fused, bias_fused


def _collect_bn_pairs(model: nn.Module) -> List[Tuple[str, str, float]]:
    name_to_module = dict(model.named_modules())
    pairs: List[Tuple[str, str, float]] = []
    for name, module in model.named_modules():
        if isinstance(module, ConvBNAct):
            conv_name = f"{name}.conv" if name else "conv"
            bn_name = f"{name}.bn" if name else "bn"
            bn_mod = name_to_module.get(bn_name)
            eps = float(getattr(bn_mod, "eps", 1e-5))
            pairs.append((conv_name, bn_name, eps))
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.named_children())
        for idx in range(len(children) - 1):
            child_name, child = children[idx]
            next_name, next_mod = children[idx + 1]
            if isinstance(child, nn.Conv2d) and isinstance(next_mod, nn.BatchNorm2d):
                conv_name = f"{name}.{child_name}" if name else child_name
                bn_name = f"{name}.{next_name}" if name else next_name
                eps = float(getattr(next_mod, "eps", 1e-5))
                pairs.append((conv_name, bn_name, eps))
    return pairs


def fold_bn_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    export_state = dict(state_dict)
    pairs = _collect_bn_pairs(model)
    for conv_name, bn_name, eps in pairs:
        conv_w_key = f"{conv_name}.weight"
        bn_rm_key = f"{bn_name}.running_mean"
        bn_rv_key = f"{bn_name}.running_var"
        if conv_w_key not in export_state:
            continue
        if bn_rm_key not in export_state or bn_rv_key not in export_state:
            continue
        conv_b_key = f"{conv_name}.bias"
        bn_w_key = f"{bn_name}.weight"
        bn_b_key = f"{bn_name}.bias"
        bn_nt_key = f"{bn_name}.num_batches_tracked"

        weight = export_state[conv_w_key]
        bias = export_state.get(
            conv_b_key,
            torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device),
        )
        running_mean = export_state[bn_rm_key].to(dtype=weight.dtype, device=weight.device)
        running_var = export_state[bn_rv_key].to(dtype=weight.dtype, device=weight.device)
        bn_weight = export_state.get(
            bn_w_key,
            torch.ones(weight.shape[0], dtype=weight.dtype, device=weight.device),
        )
        bn_bias = export_state.get(
            bn_b_key,
            torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device),
        )

        fused_w, fused_b = _fold_conv_bn(weight, bias, running_mean, running_var, bn_weight, bn_bias, eps)
        export_state[conv_w_key] = fused_w
        export_state[conv_b_key] = fused_b

        for key in (bn_w_key, bn_b_key, bn_rm_key, bn_rv_key, bn_nt_key):
            if key in export_state:
                del export_state[key]
    return export_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export UltraTinyOD checkpoint weights to C++ source.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--output-dir", default=".", help="Directory to write C++ sources.")
    parser.add_argument("--prefix", default=None, help="Output prefix (default: ckpt basename).")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if present.")
    parser.add_argument("--non-strict", action="store_true", help="Allow missing/unexpected keys when loading.")
    parser.add_argument("--no-fold-bn", action="store_true", help="Disable Conv+BN folding (default: fold BN).")
    parser.add_argument("--no-quant-bias", action="store_true", help="Disable bias int16 quantization.")
    parser.add_argument("--bias-bits", type=int, default=16, help="Bias quantization bits (default: 16).")
    parser.add_argument(
        "--quant-only",
        dest="quant_only",
        action="store_true",
        help="Skip float weights when quantized weights are emitted (default).",
    )
    parser.add_argument(
        "--with-float",
        dest="quant_only",
        action="store_false",
        help="Also export float weights (disable quant-only).",
    )
    parser.set_defaults(quant_only=True)
    parser.add_argument(
        "--bin-output",
        dest="bin_output",
        action="store_true",
        help="Write tensor data to a binary file (default).",
    )
    parser.add_argument(
        "--no-bin-output",
        dest="bin_output",
        action="store_false",
        help="Embed tensor data directly into the C++ source.",
    )
    parser.set_defaults(bin_output=True)
    parser.add_argument(
        "--no-pack-lowbit",
        action="store_true",
        help="Disable bit-packing for low-bit weights (default: pack 2/4-bit).",
    )
    parser.add_argument("--bin-path", default=None, help="Output binary file path (default: <prefix>.bin).")
    parser.add_argument("--export-act-scale", action="store_true", help="Export activation scale estimates.")
    parser.add_argument("--calib-image-dir", default=None, help="Directory of images for activation calibration.")
    parser.add_argument("--calib-list", default=None, help="Text file listing calibration images (one per line).")
    parser.add_argument("--calib-num", type=int, default=128, help="Max number of calibration images to use.")
    parser.add_argument("--calib-batch-size", type=int, default=16, help="Batch size for activation calibration.")
    parser.add_argument("--img-size", default="64x64", help="Input size for calibration (HxW or single int).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state, meta = load_checkpoint(args.ckpt, use_ema=args.use_ema)
    meta = meta or {}
    arch = str(meta.get("arch", "ultratinyod")).lower()
    if arch != "ultratinyod":
        raise ValueError(f"Only ultratinyod is supported for export_cpp.py (got arch={arch}).")

    cfg, overrides = infer_utod_config(state, meta, args)
    if "w_bits" in meta:
        cfg.w_bits = int(meta["w_bits"])
    if "a_bits" in meta:
        cfg.a_bits = int(meta["a_bits"])
    lowbit_quant_target = str(meta.get("lowbit_quant_target", meta.get("quant_target", "both")) or "both").lower()
    if lowbit_quant_target not in ("backbone", "head", "both", "none"):
        lowbit_quant_target = "both"
    highbit_quant_target = str(meta.get("highbit_quant_target", "none") or "none").lower()
    if highbit_quant_target not in ("backbone", "head", "both", "none"):
        highbit_quant_target = "none"
    lowbit_w_bits = int(meta.get("lowbit_w_bits", meta.get("w_bits", 0) or 0))
    lowbit_a_bits = int(meta.get("lowbit_a_bits", meta.get("a_bits", 0) or 0))
    highbit_w_bits = int(meta.get("highbit_w_bits", 8))
    highbit_a_bits = int(meta.get("highbit_a_bits", 8))
    cfg.lowbit_quant_target = lowbit_quant_target
    cfg.lowbit_w_bits = lowbit_w_bits
    cfg.lowbit_a_bits = lowbit_a_bits
    cfg.highbit_quant_target = highbit_quant_target
    cfg.highbit_w_bits = highbit_w_bits
    cfg.highbit_a_bits = highbit_a_bits
    if lowbit_quant_target != "none" and highbit_quant_target != "none":
        if (lowbit_quant_target in ("backbone", "both") and highbit_quant_target in ("backbone", "both")) or (
            lowbit_quant_target in ("head", "both") and highbit_quant_target in ("head", "both")
        ):
            raise ValueError("lowbit-quant-target and highbit-quant-target overlap in checkpoint metadata.")
    if highbit_quant_target != "none":
        quant_target = "mixed" if lowbit_quant_target != "none" else highbit_quant_target
    else:
        quant_target = lowbit_quant_target

    input_channels = _infer_input_channels(meta)
    model = UltraTinyOD(
        num_classes=overrides["num_classes"],
        config=cfg,
        c_stem=int(overrides["c_stem"]),
        in_channels=input_channels,
        use_residual=bool(overrides["use_residual"]),
        use_improved_head=bool(overrides["use_improved_head"]),
        use_head_ese=bool(overrides["use_head_ese"]),
        use_iou_aware_head=bool(overrides["use_iou_aware_head"]),
        quality_power=float(overrides["quality_power"]),
        activation=str(overrides["activation"] or "swish"),
    )

    missing, unexpected = model.load_state_dict(state, strict=not args.non_strict)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    if (missing or unexpected) and not args.non_strict:
        raise RuntimeError("State dict mismatch; re-run with --non-strict to force export.")

    tensors = model.state_dict()
    fold_bn = not args.no_fold_bn
    if fold_bn:
        tensors = fold_bn_state_dict(model, tensors)

    export_act_scale = bool(args.export_act_scale or args.calib_image_dir or args.calib_list)
    act_scales: List[Dict[str, object]] = []
    if export_act_scale:
        img_size = _parse_img_size(args.img_size)
        calib_paths = _list_calib_images(args.calib_image_dir, args.calib_list, int(args.calib_num))
        if not calib_paths:
            raise ValueError("Activation scale export requested but no calibration images found.")
        act_scales = _collect_activation_scales(
            model,
            calib_paths,
            size=img_size,
            resize_mode=normalize_resize_mode(meta.get("resize_mode") or "opencv_inter_nearest"),
            batch_size=int(args.calib_batch_size),
            lowbit_quant_target=lowbit_quant_target,
            lowbit_a_bits=lowbit_a_bits,
            highbit_quant_target=highbit_quant_target,
            highbit_a_bits=highbit_a_bits,
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.ckpt))[0]
    header_path = os.path.join(output_dir, f"{prefix}.h")
    source_path = os.path.join(output_dir, f"{prefix}.cpp")

    used_names: Dict[str, int] = {}
    entries: List[Dict[str, object]] = []
    q_entries: List[Dict[str, object]] = []
    w_bits = int(getattr(cfg, "w_bits", 0))
    quant_only = bool(args.quant_only)
    quant_bias = not bool(args.no_quant_bias)
    bias_bits = int(args.bias_bits)
    pack_lowbit = not bool(args.no_pack_lowbit)
    if quant_only and max(lowbit_w_bits, highbit_w_bits) < 2:
        print("[WARN] --quant-only specified but w_bits < 2; float weights will be exported.")
    if quant_bias and bias_bits < 2:
        print("[WARN] bias-bits < 2 disables bias quantization.")
        quant_bias = False

    skip_float = set()
    if max(lowbit_w_bits, highbit_w_bits) >= 2 and (lowbit_quant_target != "none" or highbit_quant_target != "none"):
        for name, tensor in tensors.items():
            arr, dtype = _tensor_to_numpy(tensor)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            is_weight = dtype == "UHD_F32" and arr.ndim == 4 and name.endswith(".weight")
            if not is_weight:
                continue
            bits = _quant_bits_for_name(
                name,
                lowbit_quant_target,
                highbit_quant_target,
                lowbit_w_bits,
                highbit_w_bits,
            )
            if bits < 2:
                continue
            q_arr, scale, q_dtype = _quantize_weights(arr.astype(np.float32), bits, per_channel=True, ch_axis=0)
            orig_shape = arr.shape
            orig_ndim = arr.ndim
            packed_bits = 0
            if pack_lowbit:
                q_arr_packed, packed_bits = _pack_quantized(q_arr, bits)
                if packed_bits:
                    q_arr = q_arr_packed
                    q_dtype = "UHD_U8"
            q_data_name = _sanitize_identifier(f"qdata_{name}", used_names)
            q_scale_name = _sanitize_identifier(f"qscale_{name}", used_names)
            q_shape_name = _sanitize_identifier(f"qshape_{name}", used_names)
            q_entries.append(
                {
                    "name": name,
                    "bits": bits,
                    "dtype": q_dtype,
                    "packed_bits": packed_bits,
                    "data_name": q_data_name,
                    "scale_name": q_scale_name,
                    "shape_name": q_shape_name,
                    "shape": orig_shape,
                    "q_arr": q_arr,
                    "scale": scale,
                    "ndim": orig_ndim,
                    "ch_axis": 0,
                }
            )
            if quant_only:
                skip_float.add(name)
            if quant_bias:
                bias_name = name[:-7] + ".bias"
                bias_tensor = tensors.get(bias_name)
                if bias_tensor is not None:
                    bias_arr, bias_dtype = _tensor_to_numpy(bias_tensor)
                    if bias_arr.ndim == 0:
                        bias_arr = bias_arr.reshape(1)
                    if bias_dtype == "UHD_F32" and bias_arr.ndim == 1:
                        bq_arr, bq_scale = _quantize_bias(bias_arr.astype(np.float32), bits=bias_bits)
                        bq_data_name = _sanitize_identifier(f"qdata_{bias_name}", used_names)
                        bq_scale_name = _sanitize_identifier(f"qscale_{bias_name}", used_names)
                        bq_shape_name = _sanitize_identifier(f"qshape_{bias_name}", used_names)
                        q_entries.append(
                            {
                                "name": bias_name,
                                "bits": bias_bits,
                                "dtype": "UHD_I16",
                                "packed_bits": 0,
                                "data_name": bq_data_name,
                                "scale_name": bq_scale_name,
                                "shape_name": bq_shape_name,
                                "shape": bq_arr.shape,
                                "q_arr": bq_arr,
                                "scale": bq_scale,
                                "ndim": bq_arr.ndim,
                                "ch_axis": -1,
                            }
                        )
                        if quant_only:
                            skip_float.add(bias_name)

    for name, tensor in tensors.items():
        if name in skip_float:
            continue
        arr, dtype = _tensor_to_numpy(tensor)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        data_name = _sanitize_identifier(f"data_{name}", used_names)
        shape_name = _sanitize_identifier(f"shape_{name}", used_names)
        entries.append(
            {
                "name": name,
                "dtype": dtype,
                "data_name": data_name,
                "shape_name": shape_name,
                "shape": arr.shape,
                "arr": arr,
                "ndim": int(arr.ndim),
            }
        )

    anchors = cfg.anchors or []
    anchors_np = np.array(anchors, dtype=np.float32).reshape(-1)

    resize_mode = normalize_resize_mode(meta.get("resize_mode") or "torch_bilinear")
    activation = str(overrides.get("activation", "swish") or "swish")

    with open(header_path, "w") as h:
        h.write("#pragma once\n")
        h.write("#include <cstddef>\n")
        h.write("#include <cstdint>\n\n")
        h.write("enum UhdDType : int32_t { UHD_F32 = 0, UHD_I8 = 1, UHD_I16 = 2, UHD_I32 = 3, UHD_U8 = 4 };\n\n")
        h.write("struct UhdTensor {\n")
        h.write("    const char* name;\n")
        h.write("    const void* data;\n")
        h.write("    const int32_t* shape;\n")
        h.write("    size_t ndim;\n")
        h.write("    UhdDType dtype;\n")
        h.write("};\n\n")
        h.write("struct UhdTensorBin {\n")
        h.write("    const char* name;\n")
        h.write("    uint64_t offset;\n")
        h.write("    uint64_t nbytes;\n")
        h.write("    const int32_t* shape;\n")
        h.write("    size_t ndim;\n")
        h.write("    UhdDType dtype;\n")
        h.write("};\n\n")
        h.write("struct UhdConfig {\n")
        h.write("    int32_t input_channels;\n")
        h.write("    int32_t num_classes;\n")
        h.write("    int32_t stride;\n")
        h.write("    int32_t num_anchors;\n")
        h.write("    int32_t c_stem;\n")
        h.write("    int32_t out_stride;\n")
        h.write("    int32_t use_improved_head;\n")
        h.write("    int32_t use_head_ese;\n")
        h.write("    int32_t use_iou_aware_head;\n")
        h.write("    int32_t use_context_rfb;\n")
        h.write("    int32_t use_large_obj_branch;\n")
        h.write("    int32_t large_obj_branch_depth;\n")
        h.write("    float large_obj_branch_expansion;\n")
        h.write("    float quality_power;\n")
        h.write("    int32_t context_dilation;\n")
        h.write("    int32_t w_bits;\n")
        h.write("    int32_t a_bits;\n")
        h.write("    int32_t fused_bn;\n")
        h.write("    int32_t quantized_weights;\n")
        h.write("    int32_t has_act_scales;\n")
        h.write("    int32_t bin_output;\n")
        h.write("    const char* bin_path;\n")
        h.write("    const char* quant_target;\n")
        h.write("    const char* lowbit_quant_target;\n")
        h.write("    int32_t lowbit_w_bits;\n")
        h.write("    int32_t lowbit_a_bits;\n")
        h.write("    const char* highbit_quant_target;\n")
        h.write("    int32_t highbit_w_bits;\n")
        h.write("    int32_t highbit_a_bits;\n")
        h.write("    const char* activation;\n")
        h.write("    const char* resize_mode;\n")
        h.write("    const float* anchors;\n")
        h.write("    size_t anchors_count;\n")
        h.write("};\n\n")
        h.write("extern const UhdConfig kUhdConfig;\n")
        h.write("extern const UhdTensor kUhdTensors[];\n")
        h.write("extern const size_t kUhdTensorCount;\n")
        h.write("extern const UhdTensorBin kUhdTensorsBin[];\n")
        h.write("extern const size_t kUhdTensorBinCount;\n")
        h.write("\n")
        h.write("struct UhdQuantizedTensor {\n")
        h.write("    const char* name;\n")
        h.write("    const void* data;\n")
        h.write("    const float* scale;\n")
        h.write("    const int32_t* shape;\n")
        h.write("    size_t ndim;\n")
        h.write("    size_t scale_len;\n")
        h.write("    int32_t bits;\n")
        h.write("    int32_t ch_axis;\n")
        h.write("    UhdDType dtype;\n")
        h.write("    int32_t packed_bits;\n")
        h.write("};\n\n")
        h.write("struct UhdQuantizedTensorBin {\n")
        h.write("    const char* name;\n")
        h.write("    uint64_t data_offset;\n")
        h.write("    uint64_t data_bytes;\n")
        h.write("    uint64_t scale_offset;\n")
        h.write("    uint64_t scale_bytes;\n")
        h.write("    const int32_t* shape;\n")
        h.write("    size_t ndim;\n")
        h.write("    size_t scale_len;\n")
        h.write("    int32_t bits;\n")
        h.write("    int32_t ch_axis;\n")
        h.write("    UhdDType dtype;\n")
        h.write("    int32_t packed_bits;\n")
        h.write("};\n\n")
        h.write("extern const UhdQuantizedTensor kUhdQuantizedTensors[];\n")
        h.write("extern const size_t kUhdQuantizedTensorCount;\n")
        h.write("extern const UhdQuantizedTensorBin kUhdQuantizedTensorsBin[];\n")
        h.write("extern const size_t kUhdQuantizedTensorBinCount;\n")
        h.write("\n")
        h.write("struct UhdActivationScale {\n")
        h.write("    const char* name;\n")
        h.write("    float scale;\n")
        h.write("    float amax;\n")
        h.write("    int32_t bits;\n")
        h.write("};\n\n")
        h.write("struct UhdActivationScaleBin {\n")
        h.write("    const char* name;\n")
        h.write("    uint64_t scale_offset;\n")
        h.write("    uint64_t scale_bytes;\n")
        h.write("    uint64_t amax_offset;\n")
        h.write("    uint64_t amax_bytes;\n")
        h.write("    uint64_t bits_offset;\n")
        h.write("    uint64_t bits_bytes;\n")
        h.write("};\n\n")
        h.write("extern const UhdActivationScale kUhdActivationScales[];\n")
        h.write("extern const size_t kUhdActivationScaleCount;\n")
        h.write("extern const UhdActivationScaleBin kUhdActivationScalesBin[];\n")
        h.write("extern const size_t kUhdActivationScaleBinCount;\n")

    bin_output = bool(args.bin_output)
    bin_path = args.bin_path or os.path.join(output_dir, f"{prefix}.bin")
    bin_path_cfg = args.bin_path if args.bin_path else os.path.basename(bin_path)
    if not bin_output:
        bin_path_cfg = ""
    bin_entries: List[Dict[str, object]] = []
    bin_q_entries: List[Dict[str, object]] = []
    bin_act_entries: List[Dict[str, object]] = []
    if bin_output:
        bin_offset = 0

        def _write_block(data: bytes, align: int = 4) -> Tuple[int, int]:
            nonlocal bin_offset
            offset = bin_offset
            nbytes = len(data)
            bf.write(data)
            pad = (-nbytes) % align
            if pad:
                bf.write(b"\x00" * pad)
            bin_offset += nbytes + pad
            return offset, nbytes

        with open(bin_path, "wb") as bf:
            for entry in entries:
                arr = entry["arr"]
                data = arr.tobytes(order="C")
                offset, nbytes = _write_block(data, align=4)
                bin_entries.append(
                    {
                        "name": entry["name"],
                        "shape_name": entry["shape_name"],
                        "shape": entry["shape"],
                        "ndim": entry["ndim"],
                        "dtype": entry["dtype"],
                        "offset": offset,
                        "nbytes": nbytes,
                    }
                )
            for entry in q_entries:
                q_data = entry["q_arr"].tobytes(order="C")
                data_offset, data_bytes = _write_block(q_data, align=4)
                scale_data = entry["scale"].tobytes(order="C")
                scale_offset, scale_bytes = _write_block(scale_data, align=4)
                bin_q_entries.append(
                    {
                        "name": entry["name"],
                        "shape_name": entry["shape_name"],
                        "shape": entry["shape"],
                        "ndim": entry["ndim"],
                        "bits": entry["bits"],
                        "ch_axis": entry["ch_axis"],
                        "dtype": entry["dtype"],
                        "packed_bits": entry.get("packed_bits", 0),
                        "scale_len": int(entry["scale"].size),
                        "data_offset": data_offset,
                        "data_bytes": data_bytes,
                        "scale_offset": scale_offset,
                        "scale_bytes": scale_bytes,
                    }
                )
            for entry in act_scales:
                scale_data = np.array([entry["scale"]], dtype=np.float32).tobytes(order="C")
                amax_data = np.array([entry["amax"]], dtype=np.float32).tobytes(order="C")
                bits_data = np.array([entry["bits"]], dtype=np.int32).tobytes(order="C")
                scale_offset, scale_bytes = _write_block(scale_data, align=4)
                amax_offset, amax_bytes = _write_block(amax_data, align=4)
                bits_offset, bits_bytes = _write_block(bits_data, align=4)
                bin_act_entries.append(
                    {
                        "name": entry["name"],
                        "scale_offset": scale_offset,
                        "scale_bytes": scale_bytes,
                        "amax_offset": amax_offset,
                        "amax_bytes": amax_bytes,
                        "bits_offset": bits_offset,
                        "bits_bytes": bits_bytes,
                    }
                )
        print(f"Wrote binary weights: {bin_path}")

    with open(source_path, "w") as c:
        c.write(f'#include "{os.path.basename(header_path)}"\n\n')
        c.write("#include <cstddef>\n\n")

        if anchors_np.size:
            _write_array(c, "kUhdAnchors", anchors_np.astype(np.float32), "float")
        else:
            c.write("static const float kUhdAnchors[] = {};\n\n")

        for entry in entries:
            _write_shape(c, entry["shape_name"], entry["shape"])
            c.write("\n")
            if not bin_output:
                arr = entry["arr"]
                dtype = entry["dtype"]
                if dtype == "UHD_I32":
                    _write_array(c, entry["data_name"], arr.astype(np.int32), "int32_t", float_fmt="{:.8e}")
                else:
                    _write_array(c, entry["data_name"], arr.astype(np.float32), "float")

        for entry in q_entries:
            _write_shape(c, entry["shape_name"], entry["shape"])
            c.write("\n")
            if not bin_output:
                if entry["dtype"] == "UHD_I16":
                    _write_array(c, entry["data_name"], entry["q_arr"].astype(np.int16), "int16_t")
                elif entry["dtype"] == "UHD_U8":
                    _write_array(c, entry["data_name"], entry["q_arr"].astype(np.uint8), "uint8_t")
                else:
                    _write_array(c, entry["data_name"], entry["q_arr"].astype(np.int8), "int8_t")
                _write_array(c, entry["scale_name"], entry["scale"].astype(np.float32), "float")

        c.write("const UhdConfig kUhdConfig = {\n")
        c.write(f"    {int(input_channels)},\n")
        c.write(f"    {int(overrides['num_classes'])},\n")
        c.write(f"    {int(overrides['stride'])},\n")
        c.write(f"    {int(len(anchors))},\n")
        c.write(f"    {int(overrides['c_stem'])},\n")
        c.write(f"    {int(overrides['stride'])},\n")
        c.write(f"    {int(bool(overrides['use_improved_head']))},\n")
        c.write(f"    {int(bool(overrides['use_head_ese']))},\n")
        c.write(f"    {int(bool(overrides['use_iou_aware_head']))},\n")
        c.write(f"    {int(bool(overrides['use_context_rfb']))},\n")
        c.write(f"    {int(bool(overrides['use_large_obj_branch']))},\n")
        c.write(f"    {int(overrides['large_obj_branch_depth'])},\n")
        c.write(f"    {float(overrides['large_obj_branch_expansion']):.8e}f,\n")
        c.write(f"    {float(overrides['quality_power']):.8e}f,\n")
        c.write(f"    {int(overrides['context_dilation'])},\n")
        c.write(f"    {int(getattr(cfg, 'w_bits', 0))},\n")
        c.write(f"    {int(getattr(cfg, 'a_bits', 0))},\n")
        c.write(f"    {1 if fold_bn else 0},\n")
        c.write(f"    {1 if q_entries else 0},\n")
        c.write(f"    {1 if act_scales else 0},\n")
        c.write(f"    {1 if bin_output else 0},\n")
        c.write(f'    "{bin_path_cfg}",\n')
        c.write(f'    "{quant_target}",\n')
        c.write(f'    "{lowbit_quant_target}",\n')
        c.write(f"    {int(lowbit_w_bits)},\n")
        c.write(f"    {int(lowbit_a_bits)},\n")
        c.write(f'    "{highbit_quant_target}",\n')
        c.write(f"    {int(highbit_w_bits)},\n")
        c.write(f"    {int(highbit_a_bits)},\n")
        c.write(f'    "{activation}",\n')
        c.write(f'    "{resize_mode}",\n')
        c.write("    kUhdAnchors,\n")
        c.write(f"    {int(anchors_np.size)},\n")
        c.write("};\n\n")

        c.write("const UhdTensor kUhdTensors[] = {\n")
        if not bin_output:
            for entry in entries:
                dtype = entry["dtype"]
                data_name = entry["data_name"]
                shape_name = entry["shape_name"]
                c.write("    {\n")
                c.write(f'        "{entry["name"]}",\n')
                c.write(f"        {data_name},\n")
                c.write(f"        {shape_name},\n")
                c.write(f"        {int(entry['ndim'])},\n")
                c.write(f"        {dtype},\n")
                c.write("    },\n")
        c.write("};\n\n")
        if bin_output:
            c.write("const size_t kUhdTensorCount = 0;\n")
        else:
            c.write("const size_t kUhdTensorCount = sizeof(kUhdTensors) / sizeof(kUhdTensors[0]);\n")
        c.write("\n")
        c.write("const UhdTensorBin kUhdTensorsBin[] = {\n")
        if bin_output:
            for entry in bin_entries:
                c.write("    {\n")
                c.write(f'        "{entry["name"]}",\n')
                c.write(f"        {int(entry['offset'])}ULL,\n")
                c.write(f"        {int(entry['nbytes'])}ULL,\n")
                c.write(f"        {entry['shape_name']},\n")
                c.write(f"        {int(entry['ndim'])},\n")
                c.write(f"        {entry['dtype']},\n")
                c.write(f"        {int(entry['packed_bits'])},\n")
                c.write("    },\n")
        c.write("};\n\n")
        if bin_output:
            c.write("const size_t kUhdTensorBinCount = sizeof(kUhdTensorsBin) / sizeof(kUhdTensorsBin[0]);\n")
        else:
            c.write("const size_t kUhdTensorBinCount = 0;\n")
        c.write("\n")
        c.write("const UhdQuantizedTensor kUhdQuantizedTensors[] = {\n")
        if not bin_output:
            for entry in q_entries:
                c.write("    {\n")
                c.write(f'        "{entry["name"]}",\n')
                c.write(f"        {entry['data_name']},\n")
                c.write(f"        {entry['scale_name']},\n")
                c.write(f"        {entry['shape_name']},\n")
                c.write(f"        {int(entry['ndim'])},\n")
                c.write(f"        {int(entry['scale'].size)},\n")
                c.write(f"        {int(entry['bits'])},\n")
                c.write(f"        {int(entry['ch_axis'])},\n")
                c.write(f"        {entry['dtype']},\n")
                c.write(f"        {int(entry['packed_bits'])},\n")
                c.write("    },\n")
        c.write("};\n\n")
        if bin_output:
            c.write("const size_t kUhdQuantizedTensorCount = 0;\n")
        else:
            c.write("const size_t kUhdQuantizedTensorCount = sizeof(kUhdQuantizedTensors) / sizeof(kUhdQuantizedTensors[0]);\n")
        c.write("\n")
        c.write("const UhdQuantizedTensorBin kUhdQuantizedTensorsBin[] = {\n")
        if bin_output:
            for entry in bin_q_entries:
                c.write("    {\n")
                c.write(f'        "{entry["name"]}",\n')
                c.write(f"        {int(entry['data_offset'])}ULL,\n")
                c.write(f"        {int(entry['data_bytes'])}ULL,\n")
                c.write(f"        {int(entry['scale_offset'])}ULL,\n")
                c.write(f"        {int(entry['scale_bytes'])}ULL,\n")
                c.write(f"        {entry['shape_name']},\n")
                c.write(f"        {int(entry['ndim'])},\n")
                c.write(f"        {int(entry['scale_len'])},\n")
                c.write(f"        {int(entry['bits'])},\n")
                c.write(f"        {int(entry['ch_axis'])},\n")
                c.write(f"        {entry['dtype']},\n")
                c.write(f"        {int(entry['packed_bits'])},\n")
                c.write("    },\n")
        c.write("};\n\n")
        if bin_output:
            c.write("const size_t kUhdQuantizedTensorBinCount = sizeof(kUhdQuantizedTensorsBin) / sizeof(kUhdQuantizedTensorsBin[0]);\n")
        else:
            c.write("const size_t kUhdQuantizedTensorBinCount = 0;\n")
        c.write("\n")
        c.write("const UhdActivationScale kUhdActivationScales[] = {\n")
        if not bin_output:
            for entry in act_scales:
                c.write("    {\n")
                c.write(f'        "{entry["name"]}",\n')
                c.write(f"        {float(entry['scale']):.8e}f,\n")
                c.write(f"        {float(entry['amax']):.8e}f,\n")
                c.write(f"        {int(entry['bits'])},\n")
                c.write("    },\n")
        c.write("};\n\n")
        if bin_output:
            c.write("const size_t kUhdActivationScaleCount = 0;\n")
        else:
            c.write("const size_t kUhdActivationScaleCount = sizeof(kUhdActivationScales) / sizeof(kUhdActivationScales[0]);\n")
        c.write("\n")
        c.write("const UhdActivationScaleBin kUhdActivationScalesBin[] = {\n")
        if bin_output:
            for entry in bin_act_entries:
                c.write("    {\n")
                c.write(f'        "{entry["name"]}",\n')
                c.write(f"        {int(entry['scale_offset'])}ULL,\n")
                c.write(f"        {int(entry['scale_bytes'])}ULL,\n")
                c.write(f"        {int(entry['amax_offset'])}ULL,\n")
                c.write(f"        {int(entry['amax_bytes'])}ULL,\n")
                c.write(f"        {int(entry['bits_offset'])}ULL,\n")
                c.write(f"        {int(entry['bits_bytes'])}ULL,\n")
                c.write("    },\n")
        c.write("};\n\n")
        if bin_output:
            c.write("const size_t kUhdActivationScaleBinCount = sizeof(kUhdActivationScalesBin) / sizeof(kUhdActivationScalesBin[0]);\n")
        else:
            c.write("const size_t kUhdActivationScaleBinCount = 0;\n")

    print(f"Wrote: {header_path}")
    print(f"Wrote: {source_path}")


if __name__ == "__main__":
    main()
