from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

SPOT_DEFAULT_K_MIN = -20
SPOT_DEFAULT_K_MAX = 2
SPOT_MODULE_NAMES = {"ConvBNAct", "DWConvBlock"}


def _is_pointwise_conv(conv: Optional[nn.Conv2d]) -> bool:
    if not isinstance(conv, nn.Conv2d):
        return False
    k = conv.kernel_size
    if isinstance(k, tuple):
        k_h, k_w = k
    else:
        k_h = k_w = int(k)
    if k_h != 1 or k_w != 1:
        return False
    return int(conv.groups) == 1


def _spot_approx_abs_torch(abs_w: torch.Tensor, k_min: int, k_max: int) -> torch.Tensor:
    pow2_min = float(2.0 ** int(k_min))
    pow2_min = max(pow2_min, 1e-12)
    log2_abs = torch.log2(torch.clamp(abs_w, min=pow2_min))
    k_floor = torch.floor(log2_abs)
    k_ceil = torch.ceil(log2_abs)
    candidates = (k_floor, k_ceil, k_floor - 1.0, k_ceil + 1.0)

    best_err = None
    best_approx = None
    pow2_min_tensor = torch.tensor(pow2_min, dtype=abs_w.dtype, device=abs_w.device)
    for k1 in candidates:
        k1 = torch.clamp(k1, k_min, k_max)
        p1 = torch.pow(2.0, k1)
        r = abs_w - p1
        use_second = r >= pow2_min_tensor
        r_clamped = torch.clamp(r, min=pow2_min)
        k2 = torch.round(torch.log2(r_clamped))
        k2 = torch.clamp(k2, k_min, k_max)
        p2 = torch.where(use_second, torch.pow(2.0, k2), torch.zeros_like(p1))
        approx = p1 + p2
        err = torch.abs(abs_w - approx)
        if best_err is None:
            best_err = err
            best_approx = approx
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_approx = torch.where(better, approx, best_approx)

    if best_approx is None:
        return torch.zeros_like(abs_w)
    return torch.where(abs_w < pow2_min_tensor, torch.zeros_like(best_approx), best_approx)


def spot_quantize_weight(weight: torch.Tensor, k_min: int, k_max: int) -> torch.Tensor:
    abs_w = torch.abs(weight)
    sign = torch.sign(weight)
    approx = _spot_approx_abs_torch(abs_w, k_min, k_max)
    w_hat = sign * approx
    return weight + (w_hat - weight).detach()


def _spot_approx_abs_numpy(abs_w: np.ndarray, k_min: int, k_max: int) -> np.ndarray:
    pow2_min = float(2.0 ** int(k_min))
    pow2_min = max(pow2_min, 1e-12)
    log2_abs = np.log2(np.maximum(abs_w, pow2_min))
    k_floor = np.floor(log2_abs)
    k_ceil = np.ceil(log2_abs)
    candidates = (k_floor, k_ceil, k_floor - 1.0, k_ceil + 1.0)

    best_err = None
    best_approx = None
    for k1 in candidates:
        k1 = np.clip(k1, k_min, k_max)
        p1 = np.power(2.0, k1)
        r = abs_w - p1
        use_second = r >= pow2_min
        r_clamped = np.maximum(r, pow2_min)
        k2 = np.round(np.log2(r_clamped))
        k2 = np.clip(k2, k_min, k_max)
        p2 = np.where(use_second, np.power(2.0, k2), 0.0)
        approx = p1 + p2
        err = np.abs(abs_w - approx)
        if best_err is None:
            best_err = err
            best_approx = approx
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_approx = np.where(better, approx, best_approx)

    if best_approx is None:
        return np.zeros_like(abs_w)
    return np.where(abs_w < pow2_min, 0.0, best_approx)


def spot_quantize_weight_numpy(weight: np.ndarray, k_min: int, k_max: int) -> np.ndarray:
    abs_w = np.abs(weight)
    sign = np.sign(weight)
    approx = _spot_approx_abs_numpy(abs_w, k_min, k_max)
    return (sign * approx).astype(weight.dtype, copy=False)


def enable_spot_on_model(model: nn.Module, k_min: int, k_max: int) -> int:
    count = 0
    for module in model.modules():
        if module.__class__.__name__ in SPOT_MODULE_NAMES:
            module.spot_enabled = True
            module.spot_k_min = int(k_min)
            module.spot_k_max = int(k_max)
            count += 1
    return count


def bake_spot_weights(model: nn.Module, k_min: int, k_max: int) -> int:
    count = 0
    with torch.no_grad():
        for module in model.modules():
            name = module.__class__.__name__
            if name == "ConvBNAct":
                conv = getattr(module, "conv", None)
                if _is_pointwise_conv(conv):
                    w_hat = spot_quantize_weight(conv.weight, k_min, k_max)
                    conv.weight.copy_(w_hat)
                    count += int(conv.weight.numel())
            elif name == "DWConvBlock":
                conv = getattr(module, "pw", None)
                if _is_pointwise_conv(conv):
                    w_hat = spot_quantize_weight(conv.weight, k_min, k_max)
                    conv.weight.copy_(w_hat)
                    count += int(conv.weight.numel())
    return count
