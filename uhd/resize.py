import warnings
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

VALID_RESIZE_MODES = {
    "torch_bilinear",
    "torch_nearest",
    "opencv_inter_linear",
    "opencv_inter_nearest",
    "opencv_inter_nearest_yuv422",
}
YUV422_RESIZE_MODE = "opencv_inter_nearest_yuv422"


def normalize_resize_mode(mode: str) -> str:
    """Normalize/validate resize mode strings."""
    if mode is None:
        mode = "opencv_inter_nearest"
    mode_norm = str(mode).lower().replace("-", "_")
    if mode_norm not in VALID_RESIZE_MODES:
        raise ValueError(f"Unknown resize mode: {mode}")
    return mode_norm


def is_yuv422_mode(mode: str) -> bool:
    return normalize_resize_mode(mode) == YUV422_RESIZE_MODE


def rgb_to_yuyv422(img: np.ndarray) -> np.ndarray:
    """Convert RGB float image [0,1] to YUYV422 packed 2-channel float [0,1] (Y, UV interleaved by x parity)."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {img.shape}")
    h, w, _ = img.shape
    if w % 2 != 0:
        raise ValueError(f"YUYV422 requires even width; got {w}")
    img_u8 = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    img_u8 = np.ascontiguousarray(img_u8)
    yuyv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YUV_YUYV)
    return yuyv.astype(np.float32) / 255.0


def resize_input(x: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    """
    x    : torch.Tensor (N, C, H, W), float32
    size : (out_h, out_w)
    mode :
        - torch_bilinear
        - torch_nearest
        - opencv_inter_linear
        - opencv_inter_nearest
        - opencv_inter_nearest_yuv422 (handled in resize_image_numpy)
    """

    out_h, out_w = size
    mode = normalize_resize_mode(mode)

    # --------------------------------------------------
    # Torch bilinear (ONNX linear + half_pixel 正解系)
    # --------------------------------------------------
    if mode == "torch_bilinear":
        return F.interpolate(
            x,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )

    # --------------------------------------------------
    # Torch nearest (ONNX nearest 正解系)
    # --------------------------------------------------
    elif mode == "torch_nearest":
        return F.interpolate(
            x,
            size=(out_h, out_w),
            mode="nearest",
        )

    # --------------------------------------------------
    # OpenCV INTER_LINEAR（近似系・half_pixel 非互換）
    # --------------------------------------------------
    elif mode == "opencv_inter_linear":
        warnings.warn(
            "opencv_inter_linear is NOT half_pixel compatible. "
            "Use only when inference is also OpenCV-based.",
            UserWarning,
        )

        assert x.ndim == 4, "Expected NCHW tensor"
        x_np = x.detach().cpu().numpy()  # (N, C, H, W)

        n, c, _, _ = x_np.shape
        out = np.empty((n, c, out_h, out_w), dtype=x_np.dtype)

        for i in range(n):
            for ch in range(c):
                out[i, ch] = cv2.resize(
                    x_np[i, ch],
                    (out_w, out_h),
                    interpolation=cv2.INTER_LINEAR,
                )

        return torch.from_numpy(out).to(x.device)

    # --------------------------------------------------
    # OpenCV INTER_NEAREST（ONNX nearest とほぼ整合）
    # --------------------------------------------------
    elif mode == "opencv_inter_nearest":
        # 最近傍は丸め差のみ → 実務上安全
        assert x.ndim == 4, "Expected NCHW tensor"
        x_np = x.detach().cpu().numpy()  # (N, C, H, W)

        n, c, _, _ = x_np.shape
        out = np.empty((n, c, out_h, out_w), dtype=x_np.dtype)

        for i in range(n):
            for ch in range(c):
                out[i, ch] = cv2.resize(
                    x_np[i, ch],
                    (out_w, out_h),
                    interpolation=cv2.INTER_NEAREST,
                )

        return torch.from_numpy(out).to(x.device)

    else:
        raise ValueError(f"Unknown resize mode: {mode}")


def resize_image_numpy(img: np.ndarray, size: Tuple[int, int], mode: str) -> np.ndarray:
    """Resize a HWC float image [0,1] using the shared resize_input implementation."""
    if img.ndim != 3 or img.shape[2] not in (1, 3):
        raise ValueError(f"Expected HWC image with 1 or 3 channels, got shape {img.shape}")
    mode = normalize_resize_mode(mode)
    if mode == YUV422_RESIZE_MODE:
        resized = resize_image_numpy(img, size=size, mode="opencv_inter_nearest")
        return rgb_to_yuyv422(resized)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    resized = resize_input(tensor, size=size, mode=mode)
    out = resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out, 0.0, 1.0)
