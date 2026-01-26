#!/usr/bin/env python3
"""
uv run python demo_uhd.py \
--camera 0 \
--img-size 64x64 \
--onnx ultratinyod_anc8_w32_64x64_opencv_inter_nearest_static_nopost.onnx

uv run python demo_uhd.py \
--camera 0 \
--img-size 64x64 \
--onnx ultratinyod_anc8_w40_64x64_opencv_inter_nearest_static_nopost.onnx
"""
import argparse
import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import onnx
import cv2
import numpy as np
import onnxruntime as ort
from onnx import numpy_helper
from PIL import Image


def _parse_meta_list(val: Optional[str]) -> Optional[List[int]]:
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return [int(v) for v in val]
    s = str(val).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    s = s.strip("[]()")
    parts = [p for p in s.replace(" ", "").split(",") if p]
    if not parts:
        return None
    out = []
    for p in parts:
        try:
            out.append(int(float(p)))
        except Exception:
            continue
    return out or None


def _parse_meta_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def preprocess(img_bgr: np.ndarray, img_size: Tuple[int, int], dynamic_resize: bool) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if dynamic_resize:
        arr = img_rgb.astype(np.float32) / 255.0
    else:
        resized = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_NEAREST)
        arr = resized.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return chw[np.newaxis, ...]


def _quantize_input(arr: np.ndarray, scale: float, zero_point: int, dtype: np.dtype) -> np.ndarray:
    if scale <= 0.0:
        return arr.astype(dtype)
    q = np.round(arr / float(scale) + float(zero_point))
    info = np.iinfo(np.dtype(dtype))
    q = np.clip(q, info.min, info.max)
    return q.astype(dtype)


def _dequantize_output(arr: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0.0:
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - float(zero_point)) * float(scale)


def preprocess_litert(
    img_bgr: np.ndarray,
    img_size: Tuple[int, int],
    input_details: dict,
    dynamic_resize: bool,
) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if dynamic_resize:
        arr = img_rgb.astype(np.float32) / 255.0
    else:
        resized = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_NEAREST)
        arr = resized.astype(np.float32) / 255.0

    input_shape = input_details.get("shape")
    input_channels = int(input_shape[-1]) if input_shape is not None else 3
    if input_channels == 1:
        gray = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        arr = gray.astype(np.float32) / 255.0
        arr = arr[..., None]
    elif input_channels != 3:
        raise ValueError(f"LiteRT input expects {input_channels} channels; only 1 or 3 are supported.")

    dtype = input_details.get("dtype", np.float32)
    scale, zero = input_details.get("quantization", (0.0, 0))
    if np.issubdtype(dtype, np.integer):
        arr = _quantize_input(arr, scale, int(zero), dtype)
    else:
        arr = arr.astype(dtype)
    return arr[np.newaxis, ...]


def postprocess(detections: np.ndarray, orig_shape: Tuple[int, int], conf_thresh: float) -> List[Tuple[float, int, float, float, float, float]]:
    h, w = orig_shape
    out: List[Tuple[float, int, float, float, float, float]] = []
    for det in detections:
        if len(det) < 6:
            continue
        score, cls_id, cx, cy, bw, bh = det[:6]
        if score < conf_thresh:
            continue
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        # clamp to valid range to avoid NaN/inf impacting drawing
        x1 = max(0.0, min(x1, w))
        x2 = max(0.0, min(x2, w))
        y1 = max(0.0, min(y1, h))
        y2 = max(0.0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((float(score), int(cls_id), float(x1), float(y1), float(x2), float(y2)))
    return out


def non_max_suppression(
    boxes: List[Tuple[float, int, float, float, float, float]], iou_thresh: float
) -> List[Tuple[float, int, float, float, float, float]]:
    if not boxes:
        return boxes
    arr = np.array(boxes, dtype=np.float32)
    scores = arr[:, 0]
    cls_ids = arr[:, 1].astype(np.int32)
    x1, y1, x2, y2 = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
    areas = (x2 - x1) * (y2 - y1)

    keep: List[int] = []
    for cls in np.unique(cls_ids):
        cls_mask = np.where(cls_ids == cls)[0]
        order = cls_mask[np.argsort(-scores[cls_mask])]
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            remain = np.where(iou <= iou_thresh)[0]
            order = order[remain + 1]
    return [boxes[i] for i in keep]


def draw_boxes(
    img_bgr: np.ndarray,
    boxes: List[Tuple[float, int, float, float, float, float]],
    color: Tuple[int, int, int],
    class0_color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    out = img_bgr.copy()
    for score, cls_id, x1, y1, x2, y2 in boxes:
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        box_color = class0_color if cls_id == 0 else color
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), box_color, 2)
    return out


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def softplus_np(x: np.ndarray, cap: Optional[float] = None) -> np.ndarray:
    # numerically stable softplus without truncating large positives (softplus(x) ~= x when x>>0)
    sp = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    if cap is not None:
        sp = np.minimum(sp, cap)
    return sp


def _is_decoded_shape(shape) -> bool:
    if shape is None:
        return False
    try:
        size = len(shape)
    except TypeError:
        return False
    return size >= 3 and int(shape[-1]) == 6


def _is_decoded_shape_litert(shape) -> bool:
    if shape is None:
        return False
    try:
        size = len(shape)
    except TypeError:
        return False
    return size >= 2 and int(shape[-1]) == 6


def _parse_anchor_hint_from_path(onnx_path: str) -> Optional[int]:
    """Infer anchor count from filename pattern like '*_anc8_*'."""
    name = os.path.basename(onnx_path).lower()
    m = re.search(r"anc(\d+)", name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _infer_anchor_count_from_channels(c: int) -> Optional[int]:
    """Heuristic to guess anchor count from channel size."""
    candidates = [3, 4, 5, 6, 8, 9, 12, 16]
    for na in candidates:
        if c % na == 0 and (c // na) >= 5:
            return na
    return None


def _build_fallback_anchors(na: int) -> np.ndarray:
    return np.stack(
        [
            np.linspace(0.08, 0.32, na, dtype=np.float32),
            np.linspace(0.10, 0.40, na, dtype=np.float32),
        ],
        axis=1,
    )


def _detect_raw_parts(outputs_info) -> Optional[dict]:
    name_map = {o.name.lower(): o.name for o in outputs_info}
    if "box" in name_map and "obj" in name_map:
        return {
            "box": name_map["box"],
            "obj": name_map["obj"],
            "quality": name_map.get("quality"),
            "cls": name_map.get("cls"),
            "attr": name_map.get("attr"),
        }
    return None


def _detect_primitive_outputs(outputs_info) -> Optional[dict]:
    name_map = {o.name.lower(): o.name for o in outputs_info}

    def _pick(*names: str) -> Optional[str]:
        for n in names:
            if n in name_map:
                return name_map[n]
        return None

    cx = _pick("cx")
    cy = _pick("cy")
    bw = _pick("bw")
    bh = _pick("bh")
    score = _pick("score", "score_base", "scoremap", "score_map")
    if not (cx and cy and bw and bh and score):
        return None

    return {
        "cx": cx,
        "cy": cy,
        "bw": bw,
        "bh": bh,
        "score": score,
        "cls_scores": _pick("cls_scores", "cls_score", "cls"),
        "attr_scores": _pick("attr_scores", "attr_score", "attr"),
    }


def _detect_raw_parts_litert(output_details: List[dict]) -> Optional[dict]:
    name_map = {d.get("name", "").lower(): d for d in output_details}
    if "box" in name_map and "obj" in name_map:
        raw_parts = {
            "box": name_map["box"]["index"],
            "obj": name_map["obj"]["index"],
            "quality": name_map.get("quality", {}).get("index") if "quality" in name_map else None,
            "cls": name_map.get("cls", {}).get("index") if "cls" in name_map else None,
            "attr": name_map.get("attr", {}).get("index") if "attr" in name_map else None,
        }
        detail_map = {d["index"]: d for d in output_details}
        box_detail = detail_map.get(raw_parts["box"])
        obj_detail = detail_map.get(raw_parts["obj"])
        cls_detail = detail_map.get(raw_parts["cls"]) if raw_parts.get("cls") is not None else None
        if not box_detail or not obj_detail:
            return None
        box_shape = box_detail.get("shape")
        obj_shape = obj_detail.get("shape")
        cls_shape = cls_detail.get("shape") if cls_detail else None
        if box_shape is None or obj_shape is None:
            return None
        a = int(obj_shape[-1])
        if a <= 0 or int(box_shape[-1]) != 4 * a:
            return None
        if cls_shape is not None and int(cls_shape[-1]) % a != 0:
            return None
        if raw_parts.get("quality") is not None:
            quality_detail = detail_map.get(raw_parts["quality"])
            quality_shape = quality_detail.get("shape") if quality_detail else None
            if quality_shape is None or int(quality_shape[-1]) != a:
                return None
        return raw_parts
    return None


def _infer_raw_parts_litert(
    interpreter,
    input_details: dict,
    output_details: List[dict],
) -> Optional[dict]:
    candidates = [d for d in output_details if d.get("shape") is not None and len(d["shape"]) == 4]
    if len(candidates) < 3:
        return None
    ordered = [(d["index"], int(d["shape"][-1])) for d in output_details if d.get("shape") is not None and len(d["shape"]) == 4]
    a = min(ch for _, ch in ordered)
    if a <= 0:
        return None

    box_idx = None
    for idx, ch in ordered:
        if ch == 4 * a:
            box_idx = idx
            break
    if box_idx is None:
        return None

    a_candidates = [idx for idx, ch in ordered if ch == a and idx != box_idx]
    cls_candidates = [idx for idx, ch in ordered if (ch % a == 0 and ch not in (a, 4 * a) and idx != box_idx)]

    cls_idx = cls_candidates[0] if cls_candidates else None
    obj_idx = a_candidates[0] if a_candidates else None
    quality_idx = None
    if len(a_candidates) >= 2:
        quality_idx = a_candidates[1]
    if cls_idx is None:
        if len(a_candidates) >= 3:
            cls_idx = a_candidates[2]
        elif len(a_candidates) == 2:
            cls_idx = a_candidates[1]
            quality_idx = None

    if obj_idx is None:
        return None

    return {"box": box_idx, "obj": obj_idx, "quality": quality_idx, "cls": cls_idx}


def _assemble_raw_from_parts(
    box: np.ndarray,
    obj: np.ndarray,
    quality: Optional[np.ndarray],
    cls: Optional[np.ndarray],
) -> Tuple[np.ndarray, int]:
    if box is None or obj is None:
        raise ValueError("Missing box/obj output for raw parts assembly.")
    if box.ndim != 4 or obj.ndim != 4:
        raise ValueError("Expected 4D tensors for box/obj outputs.")
    b, c_box, h, w = box.shape
    _, c_obj, _, _ = obj.shape
    na = int(c_obj)
    if na <= 0:
        raise ValueError("Invalid anchor count from obj output.")
    if c_box != na * 4:
        raise ValueError(f"Unexpected box channels: {c_box} (anchors={na})")
    if cls is not None:
        if cls.ndim != 4:
            raise ValueError("Expected 4D tensor for cls output.")
        if cls.shape[1] % na != 0:
            raise ValueError(f"Unexpected cls channels: {cls.shape[1]} (anchors={na})")
        nc = int(cls.shape[1] // na)
    else:
        nc = 0

    box = box.reshape(b, na, 4, h, w)
    obj = obj.reshape(b, na, 1, h, w)
    parts = [box, obj]
    if quality is not None:
        if quality.ndim != 4 or quality.shape[1] != na:
            raise ValueError(f"Unexpected quality shape: {quality.shape}")
        quality = quality.reshape(b, na, 1, h, w)
        parts.append(quality)
    if cls is not None and nc > 0:
        cls = cls.reshape(b, na, nc, h, w)
        parts.append(cls)

    merged = np.concatenate(parts, axis=2)
    raw = merged.reshape(b, na * merged.shape[2], h, w)
    return raw, nc


def _strip_quant_suffix(stem: str) -> str:
    for suffix in ("_integer_quant", "_full_integer_quant", "_float32"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _candidate_sidecar_stems(stem: str) -> List[str]:
    suffixes = (
        "_with_int16_act",
        "_int16_act",
        "_integer_quant",
        "_full_integer_quant",
        "_float32",
        "_nocat",
        "_noconcat",
    )
    stems = {stem, _strip_quant_suffix(stem)}
    changed = True
    while changed:
        changed = False
        for s in list(stems):
            for suf in suffixes:
                if s.endswith(suf):
                    trimmed = s[: -len(suf)]
                    if trimmed and trimmed not in stems:
                        stems.add(trimmed)
                        changed = True
    return list(stems)


def _load_litert_sidecar(path: Path, name: str) -> Optional[np.ndarray]:
    stem = path.stem
    candidates = [path.with_name(f"{s}_{name}.npy") for s in _candidate_sidecar_stems(stem)]
    for candidate in candidates:
        if candidate.is_file():
            return np.load(str(candidate)).astype(np.float32)
    return None


def load_anchors_from_onnx(onnx_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    """Extract anchors/wh_scale (if present) from the ONNX initializers."""
    anchors = None
    wh_scale = None
    has_quality = False
    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:
        print(f"[WARN] Failed to load ONNX for anchor lookup: {exc}")
        return anchors, wh_scale, has_quality

    for init in model.graph.initializer:
        name_l = init.name.lower()
        arr = numpy_helper.to_array(init)
        if arr.ndim == 2 and arr.shape[1] == 2:
            if "anchor" in name_l and anchors is None:
                anchors = arr.astype(np.float32)
                continue
            if "wh_scale" in name_l and wh_scale is None:
                wh_scale = arr.astype(np.float32)
                has_quality = True
                continue
            if anchors is None and arr.shape[0] <= 16:
                anchors = arr.astype(np.float32)
        if "quality" in name_l:
            has_quality = True
    return anchors, wh_scale, has_quality


def inspect_resize_info(onnx_path: str) -> dict:
    """Infer input Resize presence/mode and relevant metadata from ONNX."""
    info = {
        "dynamic_resize": False,
        "resize_mode": None,
        "score_mode": None,
        "quality_power": None,
        "disable_cls": False,
        "multi_label_mode": None,
        "multi_label_det_classes": None,
        "multi_label_attr_classes": None,
    }
    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception:
        return info

    meta = {m.key.lower(): m.value for m in model.metadata_props}
    if "dynamic_resize" in meta:
        info["dynamic_resize"] = str(meta["dynamic_resize"]).lower() == "true"
    if "resize_mode" in meta:
        info["resize_mode"] = meta["resize_mode"]
    if "score_mode" in meta:
        info["score_mode"] = meta.get("score_mode") or None
    if "quality_power" in meta:
        info["quality_power"] = _parse_meta_float(meta.get("quality_power"))
    if "disable_cls" in meta:
        info["disable_cls"] = str(meta.get("disable_cls")).lower() == "true"
    if "multi_label_mode" in meta:
        info["multi_label_mode"] = (meta.get("multi_label_mode") or "none").lower()
    if "multi_label_det_classes" in meta:
        info["multi_label_det_classes"] = _parse_meta_list(meta.get("multi_label_det_classes"))
    if "multi_label_attr_classes" in meta:
        info["multi_label_attr_classes"] = _parse_meta_list(meta.get("multi_label_attr_classes"))

    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        if node.name == "InputResize" or (node.input and node.input[0] == "images"):
            info["dynamic_resize"] = True
            if info["resize_mode"] is None:
                op_mode = None
                coord = None
                nearest_mode = None
                for attr in node.attribute:
                    if attr.name == "mode":
                        op_mode = attr.s.decode("utf-8") if attr.s else None
                    elif attr.name == "coordinate_transformation_mode":
                        coord = attr.s.decode("utf-8") if attr.s else None
                    elif attr.name == "nearest_mode":
                        nearest_mode = attr.s.decode("utf-8") if attr.s else None
                if op_mode == "linear":
                    info["resize_mode"] = "torch_bilinear" if coord == "half_pixel" else "opencv_inter_linear"
                elif op_mode == "nearest":
                    info["resize_mode"] = "torch_nearest" if coord == "half_pixel" else "opencv_inter_nearest"
            break
    return info


def inspect_tflite_resize_info(tflite_path: str) -> dict:
    info = {"dynamic_resize": False, "resize_mode": None}
    name = os.path.basename(tflite_path).lower()
    if "dynamic" in name:
        info["dynamic_resize"] = True
    for mode in ("torch_bilinear", "torch_nearest", "opencv_inter_linear", "opencv_inter_nearest"):
        if mode in name:
            info["resize_mode"] = mode
            break
    return info


def decode_ultratinyod_raw(
    raw_out: np.ndarray,
    anchors: np.ndarray,
    conf_thresh: float,
    has_quality: bool = False,
    wh_scale: Optional[np.ndarray] = None,
    score_mode: str = "obj_quality_cls",
    quality_power: float = 1.0,
    topk: int = 100,
    multi_label_mode: str = "none",
    det_class_indices: Optional[List[int]] = None,
    attr_logits: Optional[np.ndarray] = None,
    attr_class_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Decode raw UltraTinyOD output [B, C, H, W] -> [N, 6] (score, cls, cx, cy, bw, bh), normalized coords.
    """
    if raw_out.ndim == 3:
        raw_out = raw_out[None, ...]
    if raw_out.ndim != 4:
        raise ValueError(f"Unexpected raw output ndim={raw_out.ndim}; expected 4D map.")
    b, c, h, w = raw_out.shape
    anchors = np.asarray(anchors, dtype=np.float32)
    na = anchors.shape[0]
    if c % na != 0:
        raise ValueError(f"Channel/anchor mismatch: C={c}, anchors={na}")
    per_anchor = c // na
    quality_extra = 1 if has_quality and per_anchor >= 6 else 0
    num_classes = per_anchor - (5 + quality_extra)
    if num_classes < 0:
        raise ValueError(f"Unexpected channels per anchor: {per_anchor} (quality={quality_extra})")

    pred = raw_out.reshape(b, na, per_anchor, h, w).transpose(0, 1, 3, 4, 2)
    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    obj = pred[..., 4]
    quality = pred[..., 5] if quality_extra else None
    cls_logits = pred[..., (5 + quality_extra) :] if num_classes > 0 else None

    obj_sig = sigmoid_np(obj)
    cls_sig = sigmoid_np(cls_logits) if cls_logits is not None else None
    quality_sig = sigmoid_np(quality) if quality is not None else None
    qp = float(quality_power) if quality_power is not None else 1.0
    if quality_sig is not None and qp != 1.0:
        quality_sig = np.power(np.clip(quality_sig, 0.0, 1.0), qp)
    mode = str(score_mode or "obj_quality_cls").lower()
    if mode == "cls":
        score_base = np.ones_like(obj_sig, dtype=np.float32)
    elif mode == "quality_cls" and quality_sig is not None:
        score_base = quality_sig
    elif mode == "quality" and quality_sig is not None:
        score_base = quality_sig
    elif mode == "obj_cls":
        score_base = obj_sig
    elif mode == "obj":
        score_base = obj_sig
    else:
        score_base = obj_sig
        if quality_sig is not None:
            score_base = score_base * quality_sig

    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    gx = gx.reshape(1, 1, h, w)
    gy = gy.reshape(1, 1, h, w)

    anchor_use = anchors
    if wh_scale is not None and wh_scale.shape == anchors.shape:
        anchor_use = anchor_use * wh_scale
    pw = anchor_use[:, 0].reshape(1, na, 1, 1)
    ph = anchor_use[:, 1].reshape(1, na, 1, 1)

    cx = (sigmoid_np(tx) + gx) / float(w)
    cy = (sigmoid_np(ty) + gy) / float(h)
    bw = pw * softplus_np(tw)  # no cap; allow large scaling for tiny anchors
    bh = ph * softplus_np(th)

    def _topk_multi_label(
        scores_in: np.ndarray,
        cx_in: np.ndarray,
        cy_in: np.ndarray,
        bw_in: np.ndarray,
        bh_in: np.ndarray,
        class_map: Optional[List[int]],
    ) -> np.ndarray:
        bsz, _, _, _, c = scores_in.shape
        scores_flat = scores_in.reshape(bsz, -1)
        if conf_thresh > 0:
            scores_flat = np.where(scores_flat >= conf_thresh, scores_flat, np.zeros_like(scores_flat))
        k = min(int(topk), scores_flat.shape[1])
        top_idx = np.argsort(-scores_flat, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores_flat, top_idx, axis=1)
        top_cls = top_idx % c
        top_cell = top_idx // c

        cx_flat = cx_in.reshape(bsz, -1)
        cy_flat = cy_in.reshape(bsz, -1)
        bw_flat = bw_in.reshape(bsz, -1)
        bh_flat = bh_in.reshape(bsz, -1)
        top_cx = np.take_along_axis(cx_flat, top_cell, axis=1)
        top_cy = np.take_along_axis(cy_flat, top_cell, axis=1)
        top_bw = np.take_along_axis(bw_flat, top_cell, axis=1)
        top_bh = np.take_along_axis(bh_flat, top_cell, axis=1)
        if class_map is not None and len(class_map) == c:
            cm = np.asarray(class_map, dtype=np.float32)
            top_cls = cm[top_cls]
        dets = []
        for i in range(bsz):
            mask = (top_scores[i] > 0.0)
            if not np.any(mask):
                continue
            stacked = np.stack(
                [
                    top_scores[i][mask],
                    top_cls[i][mask].astype(np.float32),
                    top_cx[i][mask],
                    top_cy[i][mask],
                    top_bw[i][mask],
                    top_bh[i][mask],
                ],
                axis=-1,
            )
            finite_mask = np.all(np.isfinite(stacked), axis=-1)
            stacked = stacked[finite_mask]
            dets.append(stacked)
        if not dets:
            return np.zeros((0, 6), dtype=np.float32)
        return dets[0]

    if num_classes <= 0:
        scores_flat = score_base.reshape(b, -1)
        if conf_thresh > 0:
            scores_flat = np.where(scores_flat >= conf_thresh, scores_flat, np.zeros_like(scores_flat))
        k = min(int(topk), scores_flat.shape[1])
        top_idx = np.argsort(-scores_flat, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores_flat, top_idx, axis=1)
        top_cls = np.zeros_like(top_scores, dtype=np.float32)
        top_cell = top_idx
        cx_flat = cx.reshape(b, -1)
        cy_flat = cy.reshape(b, -1)
        bw_flat = bw.reshape(b, -1)
        bh_flat = bh.reshape(b, -1)
        top_cx = np.take_along_axis(cx_flat, top_cell, axis=1)
        top_cy = np.take_along_axis(cy_flat, top_cell, axis=1)
        top_bw = np.take_along_axis(bw_flat, top_cell, axis=1)
        top_bh = np.take_along_axis(bh_flat, top_cell, axis=1)
        dets = []
        for i in range(b):
            mask = (top_scores[i] > 0.0)
            if not np.any(mask):
                continue
            stacked = np.stack(
                [
                    top_scores[i][mask],
                    top_cls[i][mask].astype(np.float32),
                    top_cx[i][mask],
                    top_cy[i][mask],
                    top_bw[i][mask],
                    top_bh[i][mask],
                ],
                axis=-1,
            )
            finite_mask = np.all(np.isfinite(stacked), axis=-1)
            stacked = stacked[finite_mask]
            dets.append(stacked)
        if not dets:
            return np.zeros((0, 6), dtype=np.float32)
        return dets[0]

    scores = score_base[..., None] * cls_sig  # [B, A, H, W, C]

    mode_label = str(multi_label_mode or "none").lower()
    if mode_label in ("single", "separate"):
        det_scores = scores
        class_map = det_class_indices
        if mode_label == "separate" and attr_logits is not None:
            if attr_logits.ndim == 4:
                if attr_logits.shape[1] % na != 0:
                    raise ValueError(f"Unexpected attr channels {attr_logits.shape[1]} for anchors={na}.")
                attr_nc = int(attr_logits.shape[1] // na)
                attr_scores = sigmoid_np(attr_logits).reshape(b, na, attr_nc, h, w).transpose(0, 1, 3, 4, 2)
                attr_scores = score_base[..., None] * attr_scores
                det_scores = np.concatenate([det_scores, attr_scores], axis=-1)
                if det_class_indices is not None or attr_class_indices is not None:
                    det_map = det_class_indices or list(range(det_scores.shape[-1] - attr_nc))
                    attr_map = attr_class_indices or list(range(attr_nc))
                    class_map = det_map + attr_map
        return _topk_multi_label(det_scores, cx, cy, bw, bh, class_map)

    if conf_thresh > 0:
        scores = np.where(scores >= conf_thresh, scores, np.zeros_like(scores))
    best_cls = scores.argmax(axis=-1)  # [B, A, H, W]
    best_scores = scores.max(axis=-1)

    cx_flat = cx.reshape(b, -1)
    cy_flat = cy.reshape(b, -1)
    bw_flat = bw.reshape(b, -1)
    bh_flat = bh.reshape(b, -1)
    scores_flat = best_scores.reshape(b, -1)
    cls_flat = best_cls.reshape(b, -1)

    k = min(int(topk), scores_flat.shape[1])
    top_idx = np.argsort(-scores_flat, axis=1)[:, :k]

    def _gather(t: np.ndarray) -> np.ndarray:
        return np.take_along_axis(t, top_idx, axis=1)

    top_scores = _gather(scores_flat)
    top_cls = _gather(cls_flat)
    if det_class_indices is not None and len(det_class_indices) > 0:
        cm = np.asarray(det_class_indices, dtype=np.float32)
        top_cls = cm[top_cls]
    top_cx = _gather(cx_flat)
    top_cy = _gather(cy_flat)
    top_bw = _gather(bw_flat)
    top_bh = _gather(bh_flat)

    dets = []
    for i in range(b):
        mask = (top_scores[i] > 0.0)
        if not np.any(mask):
            continue
        stacked = np.stack(
            [
                top_scores[i][mask],
                top_cls[i][mask].astype(np.float32),
                top_cx[i][mask],
                top_cy[i][mask],
                top_bw[i][mask],
                top_bh[i][mask],
            ],
            axis=-1,
        )
        finite_mask = np.all(np.isfinite(stacked), axis=-1)
        stacked = stacked[finite_mask]
        dets.append(stacked)
    if not dets:
        return np.zeros((0, 6), dtype=np.float32)
        return dets[0]


def decode_ultratinyod_maps(
    cx: np.ndarray,
    cy: np.ndarray,
    bw: np.ndarray,
    bh: np.ndarray,
    score: np.ndarray,
    conf_thresh: float,
    topk: int = 100,
    multi_label_mode: str = "none",
    det_class_indices: Optional[List[int]] = None,
    cls_scores: Optional[np.ndarray] = None,
    attr_scores: Optional[np.ndarray] = None,
    attr_class_indices: Optional[List[int]] = None,
) -> np.ndarray:
    if cx.ndim == 3:
        cx = cx[None, ...]
    if cy.ndim == 3:
        cy = cy[None, ...]
    if bw.ndim == 3:
        bw = bw[None, ...]
    if bh.ndim == 3:
        bh = bh[None, ...]
    if score.ndim == 3:
        score = score[None, ...]

    if cx.ndim != 4 or cy.ndim != 4 or bw.ndim != 4 or bh.ndim != 4 or score.ndim != 4:
        raise ValueError("Primitive map outputs must be 4D (B, A, H, W).")
    b, na, h, w = score.shape

    def _prep_scores(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.ndim == 4:
            if arr.shape[1] % na == 0:
                c = int(arr.shape[1] // na)
                return arr.reshape(b, na, c, h, w).transpose(0, 1, 3, 4, 2)
            return arr[..., None]
        if arr.ndim == 5:
            return arr
        raise ValueError(f"Unexpected scores ndim={arr.ndim}")

    cls_scores = _prep_scores(cls_scores)
    attr_scores = _prep_scores(attr_scores)

    def _topk_multi_label(
        scores_in: np.ndarray,
        cx_in: np.ndarray,
        cy_in: np.ndarray,
        bw_in: np.ndarray,
        bh_in: np.ndarray,
        class_map: Optional[List[int]],
    ) -> np.ndarray:
        bsz, _, _, _, c = scores_in.shape
        scores_flat = scores_in.reshape(bsz, -1)
        if conf_thresh > 0:
            scores_flat = np.where(scores_flat >= conf_thresh, scores_flat, np.zeros_like(scores_flat))
        k = min(int(topk), scores_flat.shape[1])
        top_idx = np.argsort(-scores_flat, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores_flat, top_idx, axis=1)
        top_cls = top_idx % c
        top_cell = top_idx // c

        cx_flat = cx_in.reshape(bsz, -1)
        cy_flat = cy_in.reshape(bsz, -1)
        bw_flat = bw_in.reshape(bsz, -1)
        bh_flat = bh_in.reshape(bsz, -1)
        top_cx = np.take_along_axis(cx_flat, top_cell, axis=1)
        top_cy = np.take_along_axis(cy_flat, top_cell, axis=1)
        top_bw = np.take_along_axis(bw_flat, top_cell, axis=1)
        top_bh = np.take_along_axis(bh_flat, top_cell, axis=1)
        if class_map is not None and len(class_map) == c:
            cm = np.asarray(class_map, dtype=np.float32)
            top_cls = cm[top_cls]
        dets = []
        for i in range(bsz):
            mask = (top_scores[i] > 0.0)
            if not np.any(mask):
                continue
            stacked = np.stack(
                [
                    top_scores[i][mask],
                    top_cls[i][mask].astype(np.float32),
                    top_cx[i][mask],
                    top_cy[i][mask],
                    top_bw[i][mask],
                    top_bh[i][mask],
                ],
                axis=-1,
            )
            finite_mask = np.all(np.isfinite(stacked), axis=-1)
            stacked = stacked[finite_mask]
            dets.append(stacked)
        if not dets:
            return np.zeros((0, 6), dtype=np.float32)
        return dets[0]

    if cls_scores is None:
        scores_flat = score.reshape(b, -1)
        if conf_thresh > 0:
            scores_flat = np.where(scores_flat >= conf_thresh, scores_flat, np.zeros_like(scores_flat))
        k = min(int(topk), scores_flat.shape[1])
        top_idx = np.argsort(-scores_flat, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores_flat, top_idx, axis=1)
        top_cls = np.zeros_like(top_scores, dtype=np.float32)
        top_cell = top_idx

        cx_flat = cx.reshape(b, -1)
        cy_flat = cy.reshape(b, -1)
        bw_flat = bw.reshape(b, -1)
        bh_flat = bh.reshape(b, -1)
        top_cx = np.take_along_axis(cx_flat, top_cell, axis=1)
        top_cy = np.take_along_axis(cy_flat, top_cell, axis=1)
        top_bw = np.take_along_axis(bw_flat, top_cell, axis=1)
        top_bh = np.take_along_axis(bh_flat, top_cell, axis=1)

        dets = []
        for i in range(b):
            mask = (top_scores[i] > 0.0)
            if not np.any(mask):
                continue
            stacked = np.stack(
                [
                    top_scores[i][mask],
                    top_cls[i][mask].astype(np.float32),
                    top_cx[i][mask],
                    top_cy[i][mask],
                    top_bw[i][mask],
                    top_bh[i][mask],
                ],
                axis=-1,
            )
            finite_mask = np.all(np.isfinite(stacked), axis=-1)
            stacked = stacked[finite_mask]
            dets.append(stacked)
        if not dets:
            return np.zeros((0, 6), dtype=np.float32)
        return dets[0]

    scores = score[..., None] * cls_scores
    mode_label = str(multi_label_mode or "none").lower()
    if mode_label in ("single", "separate"):
        det_scores = scores
        class_map = det_class_indices
        if mode_label == "separate" and attr_scores is not None:
            det_scores = np.concatenate([det_scores, score[..., None] * attr_scores], axis=-1)
            if det_class_indices is not None or attr_class_indices is not None:
                det_map = det_class_indices or list(range(det_scores.shape[-1] - attr_scores.shape[-1]))
                attr_map = attr_class_indices or list(range(attr_scores.shape[-1]))
                class_map = det_map + attr_map
        return _topk_multi_label(det_scores, cx, cy, bw, bh, class_map)

    if conf_thresh > 0:
        scores = np.where(scores >= conf_thresh, scores, np.zeros_like(scores))
    best_cls = scores.argmax(axis=-1)
    best_scores = scores.max(axis=-1)

    cx_flat = cx.reshape(b, -1)
    cy_flat = cy.reshape(b, -1)
    bw_flat = bw.reshape(b, -1)
    bh_flat = bh.reshape(b, -1)
    scores_flat = best_scores.reshape(b, -1)
    cls_flat = best_cls.reshape(b, -1)

    k = min(int(topk), scores_flat.shape[1])
    top_idx = np.argsort(-scores_flat, axis=1)[:, :k]

    def _gather(t: np.ndarray) -> np.ndarray:
        return np.take_along_axis(t, top_idx, axis=1)

    top_scores = _gather(scores_flat)
    top_cls = _gather(cls_flat)
    if det_class_indices is not None and len(det_class_indices) > 0:
        cm = np.asarray(det_class_indices, dtype=np.float32)
        top_cls = cm[top_cls]
    top_cx = _gather(cx_flat)
    top_cy = _gather(cy_flat)
    top_bw = _gather(bw_flat)
    top_bh = _gather(bh_flat)

    dets = []
    for i in range(b):
        mask = (top_scores[i] > 0.0)
        if not np.any(mask):
            continue
        stacked = np.stack(
            [
                top_scores[i][mask],
                top_cls[i][mask].astype(np.float32),
                top_cx[i][mask],
                top_cy[i][mask],
                top_bw[i][mask],
                top_bh[i][mask],
            ],
            axis=-1,
        )
        finite_mask = np.all(np.isfinite(stacked), axis=-1)
        stacked = stacked[finite_mask]
        dets.append(stacked)
    if not dets:
        return np.zeros((0, 6), dtype=np.float32)
    return dets[0]


def load_session(onnx_path: str, img_size: Tuple[int, int]):
    """Load ONNX session and infer whether outputs already include post-process."""
    resize_info = inspect_resize_info(onnx_path)
    dynamic_resize = bool(resize_info.get("dynamic_resize", False))
    resize_mode = resize_info.get("resize_mode")
    score_mode = resize_info.get("score_mode")
    quality_power = resize_info.get("quality_power")
    disable_cls = bool(resize_info.get("disable_cls", False))
    multi_label_mode = resize_info.get("multi_label_mode") or "none"
    det_class_indices = resize_info.get("multi_label_det_classes")
    attr_class_indices = resize_info.get("multi_label_attr_classes")
    if disable_cls and score_mode:
        mode_l = str(score_mode).lower()
        if mode_l in ("obj_quality_cls", "obj_quality"):
            score_mode = "obj_quality"
        elif mode_l in ("quality_cls", "quality"):
            score_mode = "quality"
        elif mode_l in ("obj_cls", "obj", "cls"):
            score_mode = "obj"
    if dynamic_resize:
        print(f"[INFO] Detected input Resize in ONNX (mode={resize_mode or 'unknown'})")
    else:
        print("[INFO] No input Resize detected; preprocessing will resize with OpenCV.")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    outputs_info = session.get_outputs()
    anchor_hint = _parse_anchor_hint_from_path(onnx_path)
    raw_parts = _detect_raw_parts(outputs_info)
    primitive_outputs = _detect_primitive_outputs(outputs_info)

    decoded_output = None
    if raw_parts is None and primitive_outputs is None:
        for o in outputs_info:
            if _is_decoded_shape(o.shape):
                decoded_output = o.name
                break

    anchors = None
    wh_scale = None
    has_quality = False
    raw_channels = None
    raw_output = None

    if decoded_output is None and primitive_outputs is None:
        # Probe with a dummy forward to inspect actual shapes and capture anchors/wh_scale outputs if present.
        _, c_in, h_in, w_in = input_info.shape

        def _dim_or_fallback(dim, fallback: int) -> int:
            try:
                return int(fallback if dim in (None, "None") else dim)
            except Exception:
                return int(fallback)

        c_probe = _dim_or_fallback(c_in, 3)
        h_probe = _dim_or_fallback(h_in, img_size[0])
        w_probe = _dim_or_fallback(w_in, img_size[1])
        dummy = np.zeros((1, c_probe, h_probe, w_probe), dtype=np.float32)
        outs = session.run(None, {input_info.name: dummy})
        out_map = {meta.name: val for meta, val in zip(outputs_info, outs)}
        for meta, val in zip(outputs_info, outs):
            if raw_parts is None and val.ndim == 4 and raw_output is None:
                raw_output = meta.name
                raw_channels = val.shape[1] if val.ndim == 4 else val.shape[-1]
            elif val.ndim == 2 and val.shape[1] == 2:
                name_l = meta.name.lower()
                if anchors is None and ("anchor" in name_l or raw_output is None):
                    anchors = val.astype(np.float32)
                if wh_scale is None and ("wh_scale" in name_l or "scale" in name_l):
                    wh_scale = val.astype(np.float32)
                    has_quality = True
        if raw_parts is not None:
            box = out_map.get(raw_parts["box"])
            obj = out_map.get(raw_parts["obj"])
            quality = out_map.get(raw_parts.get("quality") or "")
            cls = out_map.get(raw_parts["cls"]) if raw_parts.get("cls") else None
            if obj is not None:
                na = int(obj.shape[1])
                if anchor_hint is None:
                    anchor_hint = na
                c_box = int(box.shape[1]) if box is not None else 0
                c_obj = int(obj.shape[1])
                c_quality = int(quality.shape[1]) if quality is not None else 0
                c_cls = int(cls.shape[1]) if cls is not None else 0
                raw_channels = c_box + c_obj + c_quality + c_cls
                has_quality = has_quality or quality is not None or wh_scale is not None
        if raw_output is None and raw_parts is None and outputs_info:
            raw_output = outputs_info[0].name
            raw_shape = outs[0].shape if outs else outputs_info[0].shape
            raw_channels = raw_shape[1] if isinstance(raw_shape, tuple) and len(raw_shape) >= 2 else None
        if anchors is None or wh_scale is None:
            anchors_f, wh_scale_f, has_quality_f = load_anchors_from_onnx(onnx_path)
            anchors = anchors if anchors is not None else anchors_f
            wh_scale = wh_scale if wh_scale is not None else wh_scale_f
            has_quality = has_quality or has_quality_f
        if anchors is None and anchor_hint:
            anchors = _build_fallback_anchors(anchor_hint)
            print(f"[INFO] Using anchors inferred from filename (anc{anchor_hint}).")
        if anchors is None:
            print("[WARN] Could not find anchors in ONNX; raw decode may fail.")
        decoded = False
        if raw_parts is not None:
            output_shape = {k: (out_map.get(v).shape if out_map.get(v) is not None else None) for k, v in raw_parts.items() if v}
        else:
            output_shape = outs[0].shape if outs else outputs_info[0].shape
    else:
        decoded = True
        if decoded_output is not None:
            output_shape = next(o.shape for o in outputs_info if o.name == decoded_output)
        else:
            decoded = False
            output_shape = {k: next(o.shape for o in outputs_info if o.name == v) for k, v in primitive_outputs.items() if v}

    if decoded:
        kind = "decoded output"
    elif primitive_outputs is not None:
        kind = "primitive decoded maps"
    elif raw_parts is not None:
        kind = "raw parts output + demo post-process"
    else:
        kind = "raw output + demo post-process"
    print(f"[INFO] Detected {kind} (output shape: {output_shape})")
    if raw_parts is not None and not raw_parts.get("cls"):
        disable_cls = True
    if primitive_outputs is not None and not primitive_outputs.get("cls_scores"):
        disable_cls = True
    if not disable_cls and raw_channels is not None:
        na_guess = None
        if anchors is not None and anchors.shape[0] > 0:
            na_guess = int(anchors.shape[0])
        elif anchor_hint:
            na_guess = int(anchor_hint)
        if na_guess and raw_channels % na_guess == 0:
            per_anchor = int(raw_channels) // int(na_guess)
            if per_anchor == 5:
                disable_cls = True
            elif per_anchor == 6 and (has_quality or wh_scale is not None):
                disable_cls = True
    if disable_cls and not score_mode:
        score_mode = "obj_quality" if (has_quality or wh_scale is not None) else "obj"
    return session, {
        "decoded": decoded,
        "anchors": anchors,
        "wh_scale": wh_scale,
        "has_quality": has_quality or wh_scale is not None,
        "raw_channels": raw_channels if not decoded else None,
        "anchor_hint": anchor_hint,
        "input_name": input_info.name,
        "decoded_output": decoded_output,
        "raw_output": raw_output,
        "raw_parts": raw_parts,
        "primitive_outputs": primitive_outputs,
        "dynamic_resize": dynamic_resize,
        "resize_mode": resize_mode,
        "score_mode": score_mode,
        "quality_power": quality_power,
        "disable_cls": disable_cls,
        "multi_label_mode": multi_label_mode,
        "det_class_indices": det_class_indices,
        "attr_class_indices": attr_class_indices,
        "backend": "onnx",
        "window_title": "UHD ONNX",
    }


def load_litert_session(tflite_path: str, img_size: Tuple[int, int]):
    """Load LiteRT (TFLite) interpreter and infer whether outputs already include post-process."""
    resize_info = inspect_tflite_resize_info(tflite_path)
    dynamic_resize = bool(resize_info.get("dynamic_resize", False))
    resize_mode = resize_info.get("resize_mode")
    if dynamic_resize:
        print(f"[INFO] Detected input Resize in LiteRT (mode={resize_mode or 'unknown'})")
    else:
        print("[INFO] No input Resize detected in LiteRT; preprocessing will resize with OpenCV.")

    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception as exc:
        raise RuntimeError("LiteRT interpreter not available. Install ai-edge-litert.") from exc

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    anchor_hint = _parse_anchor_hint_from_path(tflite_path)
    raw_parts = _detect_raw_parts_litert(output_details)
    if raw_parts is None:
        raw_parts = _infer_raw_parts_litert(interpreter, input_details, output_details)

    decoded_output = None
    if raw_parts is None:
        for o in output_details:
            if _is_decoded_shape_litert(o.get("shape")):
                decoded_output = o.get("index")
                break

    raw_channels = None
    disable_cls = False
    raw_output = None
    if decoded_output is None and raw_parts is None:
        for o in output_details:
            shape = o.get("shape")
            if shape is not None and len(shape) == 4:
                raw_output = o.get("index")
                raw_channels = int(shape[-1])
                break
        if raw_output is None and output_details:
            raw_output = output_details[0].get("index")
            shape = output_details[0].get("shape")
            if shape is not None and len(shape) >= 1:
                raw_channels = int(shape[-1])
    elif raw_parts is not None:
        detail_map = {d["index"]: d for d in output_details}
        obj_detail = detail_map.get(raw_parts["obj"])
        box_detail = detail_map.get(raw_parts["box"])
        quality_detail = detail_map.get(raw_parts.get("quality")) if raw_parts.get("quality") is not None else None
        cls_detail = detail_map.get(raw_parts["cls"]) if raw_parts.get("cls") is not None else None
        if obj_detail and box_detail:
            na = int(obj_detail.get("shape")[-1])
            c_box = int(box_detail.get("shape")[-1])
            c_quality = int(quality_detail.get("shape")[-1]) if quality_detail else 0
            c_cls = int(cls_detail.get("shape")[-1]) if cls_detail else 0
            if c_box != na * 4 and quality_detail and c_quality == na * 4:
                c_box, c_quality = c_quality, c_box
            raw_channels = c_box + na + c_quality + c_cls

    sidecar_path = Path(tflite_path)
    anchors = _load_litert_sidecar(sidecar_path, "anchors")
    wh_scale = _load_litert_sidecar(sidecar_path, "wh_scale")

    has_quality = wh_scale is not None or (raw_parts is not None and raw_parts.get("quality") is not None)
    if raw_channels is not None:
        na_hint = anchor_hint or _infer_anchor_count_from_channels(int(raw_channels))
        if na_hint:
            per_anchor = int(raw_channels) // int(na_hint)
            if per_anchor >= 7:
                has_quality = True
            if per_anchor == 5:
                disable_cls = True
            elif per_anchor == 6 and has_quality:
                disable_cls = True

    input_shape = input_details.get("shape")
    input_hw = None
    if input_shape is not None and len(input_shape) >= 3:
        input_hw = (int(input_shape[1]), int(input_shape[2]))
    swap_on_logit = ("_integer_quant" in Path(tflite_path).stem) and ("_full_integer_quant" not in Path(tflite_path).stem)

    if decoded_output is not None:
        kind = "decoded output"
    elif raw_parts is not None:
        kind = "raw parts output + demo post-process"
    else:
        kind = "raw output + demo post-process"
    output_shape = None
    if decoded_output is not None:
        for o in output_details:
            if o.get("index") == decoded_output:
                output_shape = o.get("shape")
                break
    if output_shape is None and raw_parts is not None:
        detail_map = {d["index"]: d for d in output_details}
        output_shape = detail_map.get(raw_parts["box"], {}).get("shape")
    if output_shape is None and output_details:
        output_shape = output_details[0].get("shape")
    print(f"[INFO] Detected {kind} (output shape: {output_shape})")
    score_mode = None
    if disable_cls:
        score_mode = "obj_quality" if has_quality else "obj"
    return interpreter, {
        "decoded": decoded_output is not None,
        "anchors": anchors,
        "wh_scale": wh_scale,
        "has_quality": has_quality,
        "raw_channels": raw_channels if decoded_output is None else None,
        "anchor_hint": anchor_hint,
        "input_details": input_details,
        "output_details": output_details,
        "input_index": input_details.get("index"),
        "decoded_output": decoded_output,
        "raw_output": raw_output,
        "raw_parts": raw_parts,
        "dynamic_resize": dynamic_resize,
        "resize_mode": resize_mode,
        "score_mode": score_mode,
        "quality_power": None,
        "disable_cls": disable_cls or (raw_parts is not None and not raw_parts.get("cls")),
        "multi_label_mode": "none",
        "det_class_indices": None,
        "attr_class_indices": None,
        "backend": "litert",
        "window_title": "UHD LiteRT",
        "input_hw": input_hw if input_hw else img_size,
        "swap_logit_range": swap_on_logit,
    }


def run_and_decode_onnx(
    session: ort.InferenceSession,
    session_info: dict,
    inp: np.ndarray,
    conf_thresh: float,
) -> np.ndarray:
    score_mode = session_info.get("score_mode", "obj_quality_cls")
    quality_power = session_info.get("quality_power", 1.0)
    disable_cls = bool(session_info.get("disable_cls", False))
    if disable_cls and score_mode:
        mode_l = str(score_mode).lower()
        if mode_l in ("obj_quality_cls", "obj_quality"):
            score_mode = "obj_quality"
        elif mode_l in ("quality_cls", "quality"):
            score_mode = "quality"
        elif mode_l in ("obj_cls", "obj", "cls"):
            score_mode = "obj"
    multi_label_mode = session_info.get("multi_label_mode", "none")
    det_class_indices = session_info.get("det_class_indices")
    attr_class_indices = session_info.get("attr_class_indices")
    if session_info.get("primitive_outputs"):
        parts = session_info["primitive_outputs"]
        outputs = [parts["cx"], parts["cy"], parts["bw"], parts["bh"], parts["score"]]
        if parts.get("cls_scores") is not None:
            outputs.append(parts["cls_scores"])
        if parts.get("attr_scores") is not None:
            outputs.append(parts["attr_scores"])
        run_outs = session.run(outputs, {session_info["input_name"]: inp})
        out_map = {name: val for name, val in zip(outputs, run_outs)}
        return decode_ultratinyod_maps(
            out_map[parts["cx"]],
            out_map[parts["cy"]],
            out_map[parts["bw"]],
            out_map[parts["bh"]],
            out_map[parts["score"]],
            conf_thresh=0.0,
            multi_label_mode=multi_label_mode,
            det_class_indices=det_class_indices,
            cls_scores=out_map.get(parts.get("cls_scores") or ""),
            attr_scores=out_map.get(parts.get("attr_scores") or ""),
            attr_class_indices=attr_class_indices,
        )
    if session_info.get("raw_parts"):
        parts = session_info["raw_parts"]
        anchors = session_info.get("anchors")
        wh_scale = session_info.get("wh_scale")

        outputs = [parts["box"], parts["obj"], parts.get("quality"), parts.get("cls")]
        if parts.get("attr") is not None:
            outputs.append(parts.get("attr"))
        outputs = [o for o in outputs if o]
        if anchors is None or wh_scale is None:
            outputs = [o.name for o in session.get_outputs()]
        run_outs = session.run(outputs, {session_info["input_name"]: inp})
        out_map = {name: val for name, val in zip(outputs, run_outs)}

        box = out_map.get(parts["box"])
        obj = out_map.get(parts["obj"])
        quality = out_map.get(parts.get("quality") or "")
        cls = out_map.get(parts["cls"]) if parts.get("cls") else None
        attr = out_map.get(parts.get("attr") or "")

        if box is None or obj is None:
            raise RuntimeError("Missing raw parts outputs from ONNX session.")

        for name, val in out_map.items():
            if val.ndim == 2 and val.shape[1] == 2:
                name_l = name.lower()
                if anchors is None and ("anchor" in name_l or True):
                    anchors = val.astype(np.float32)
                if wh_scale is None and ("wh_scale" in name_l or "scale" in name_l):
                    wh_scale = val.astype(np.float32)
                    session_info["has_quality"] = True

        raw, _ = _assemble_raw_from_parts(box, obj, quality, cls)

        if anchors is None:
            na = int(obj.shape[1])
            na_hint = session_info.get("anchor_hint")
            na = na_hint if na_hint is not None else na
            anchors = _build_fallback_anchors(int(na))
            print(f"[WARN] Anchors not found in ONNX; using fallback anchors (A={na}).")

        session_info["anchors"] = anchors
        if wh_scale is not None:
            session_info["wh_scale"] = wh_scale

        has_quality = quality is not None or session_info.get("has_quality", False) or wh_scale is not None
        return decode_ultratinyod_raw(
            raw,
            anchors=anchors,
            conf_thresh=0.0,
            has_quality=has_quality,
            wh_scale=wh_scale,
            score_mode=score_mode,
            quality_power=quality_power,
            multi_label_mode=multi_label_mode,
            det_class_indices=det_class_indices,
            attr_logits=attr,
            attr_class_indices=attr_class_indices,
        )

    if session_info.get("decoded", False):
        dets = session.run([session_info["decoded_output"]], {session_info["input_name"]: inp})[0]
        return dets[0] if dets.ndim >= 3 else dets

    # Raw path: prefer cached anchors/wh_scale from ONNX outputs
    anchors = session_info.get("anchors")
    wh_scale = session_info.get("wh_scale")
    outputs = [session_info.get("raw_output") or session.get_outputs()[0].name]
    if anchors is None or wh_scale is None:
        # If anchors/wh_scale were not cached, fetch them from ONNX outputs
        outputs = [o.name for o in session.get_outputs()]
    run_outs = session.run(outputs, {session_info["input_name"]: inp})

    # Identify raw / anchors / wh_scale in the returned list
    raw = None
    attr = None
    for name, val in zip(outputs, run_outs):
        name_l = name.lower()
        if raw is None and val.ndim == 4:
            raw = val
        elif val.ndim == 2 and val.shape[1] == 2:
            if anchors is None and ("anchor" in name_l or True):
                anchors = val.astype(np.float32)
            if wh_scale is None and ("wh_scale" in name_l or "scale" in name_l):
                wh_scale = val.astype(np.float32)
                session_info["has_quality"] = True
        elif val.ndim == 4 and "attr" in name_l:
            attr = val

    if raw is None:
        raw = run_outs[0]

    if anchors is None:
        c = raw.shape[1] if raw.ndim == 4 else raw.shape[-1]
        na_hint = session_info.get("anchor_hint")
        na_guess = _infer_anchor_count_from_channels(int(c))
        na = na_hint if na_hint is not None else (na_guess if na_guess is not None else 3)
        anchors = _build_fallback_anchors(na)
        print(f"[WARN] Anchors not found in ONNX; using fallback anchors (A={na}).")

    session_info["anchors"] = anchors
    if wh_scale is not None:
        session_info["wh_scale"] = wh_scale

    return decode_ultratinyod_raw(
        raw,
        anchors=anchors,
        conf_thresh=0.0,  # avoid double-thresholding; postprocess will apply user conf
        has_quality=session_info.get("has_quality", False),
        wh_scale=wh_scale,
        score_mode=score_mode,
        quality_power=quality_power,
        multi_label_mode=multi_label_mode,
        det_class_indices=det_class_indices,
        attr_logits=attr,
        attr_class_indices=attr_class_indices,
    )


def run_and_decode_litert(
    interpreter,
    session_info: dict,
    inp: np.ndarray,
    conf_thresh: float,
) -> np.ndarray:
    score_mode = session_info.get("score_mode", "obj_quality_cls")
    quality_power = session_info.get("quality_power", 1.0)
    disable_cls = bool(session_info.get("disable_cls", False))
    if disable_cls and score_mode:
        mode_l = str(score_mode).lower()
        if mode_l in ("obj_quality_cls", "obj_quality"):
            score_mode = "obj_quality"
        elif mode_l in ("quality_cls", "quality"):
            score_mode = "quality"
        elif mode_l in ("obj_cls", "obj", "cls"):
            score_mode = "obj"
    multi_label_mode = session_info.get("multi_label_mode", "none")
    det_class_indices = session_info.get("det_class_indices")
    interpreter.set_tensor(session_info["input_index"], inp)
    interpreter.invoke()

    output_details = session_info.get("output_details", [])
    detail_map = {d["index"]: d for d in output_details}
    raw_parts = session_info.get("raw_parts")
    if raw_parts:
        def _get_tensor(idx: int) -> np.ndarray:
            detail = detail_map.get(idx)
            if detail is None:
                raise RuntimeError(f"LiteRT output index {idx} not found.")
            val = interpreter.get_tensor(idx)
            scale, zero = detail.get("quantization", (0.0, 0))
            if np.issubdtype(detail.get("dtype", np.float32), np.integer):
                val = _dequantize_output(val, float(scale), int(zero))
            return val

        box = _get_tensor(raw_parts["box"])
        obj = _get_tensor(raw_parts["obj"])
        cls = _get_tensor(raw_parts["cls"]) if raw_parts.get("cls") is not None else None
        quality = _get_tensor(raw_parts["quality"]) if raw_parts.get("quality") is not None else None

        na = int(obj.shape[-1])
        if box.shape[-1] != na * 4:
            raise ValueError(f"Unexpected box channels {box.shape[-1]} for anchors={na}.")
        if quality is not None and quality.shape[-1] != na:
            raise ValueError(f"Unexpected quality channels {quality.shape[-1]} for anchors={na}.")
        if cls is not None and cls.shape[-1] % na != 0:
            raise ValueError(f"Unexpected cls channels {cls.shape[-1]} for anchors={na}.")
        if cls is not None and session_info.get("swap_logit_range") and quality is not None and quality.shape == cls.shape and quality.shape[-1] == na:
            qmax = float(np.max(quality))
            cmax = float(np.max(cls))
            qhist = session_info.setdefault("swap_logit_qmax", [])
            chist = session_info.setdefault("swap_logit_cmax", [])
            qhist.append(qmax)
            chist.append(cmax)
            max_samples = int(session_info.get("swap_logit_samples", 5))
            if len(qhist) > max_samples:
                qhist.pop(0)
                chist.pop(0)
            if not session_info.get("swap_logit_decided") and len(qhist) >= max_samples:
                avg_q = float(np.mean(qhist))
                avg_c = float(np.mean(chist))
                if avg_q > (avg_c + 4.0):
                    print("[WARN] Swapping quality/cls outputs based on logit range (locked).")
                    session_info["swap_logit_decided"] = "swap"
                else:
                    session_info["swap_logit_decided"] = "noswap"
            if session_info.get("swap_logit_decided") == "swap":
                quality, cls = cls, quality

        def _nhwc_to_nchw(arr: np.ndarray) -> np.ndarray:
            return np.transpose(arr, (0, 3, 1, 2))

        raw_box = _nhwc_to_nchw(box)
        raw_obj = _nhwc_to_nchw(obj)
        raw_cls = _nhwc_to_nchw(cls) if cls is not None else None
        raw_quality = _nhwc_to_nchw(quality) if quality is not None else None

        raw, _ = _assemble_raw_from_parts(raw_box, raw_obj, raw_quality, raw_cls)

        anchors = session_info.get("anchors")
        wh_scale = session_info.get("wh_scale")
        if anchors is None:
            na_hint = session_info.get("anchor_hint")
            na_use = na_hint if na_hint is not None else na
            anchors = _build_fallback_anchors(int(na_use))
            print(f"[WARN] Anchors not found in LiteRT; using fallback anchors (A={na_use}).")
        session_info["anchors"] = anchors
        if wh_scale is not None:
            session_info["wh_scale"] = wh_scale

        has_quality = quality is not None or session_info.get("has_quality", False) or wh_scale is not None
        return decode_ultratinyod_raw(
            raw,
            anchors=anchors,
            conf_thresh=0.0,
            has_quality=has_quality,
            wh_scale=wh_scale,
            score_mode=score_mode,
            quality_power=quality_power,
            multi_label_mode=multi_label_mode,
            det_class_indices=det_class_indices,
        )

    if session_info.get("decoded", False):
        out_idx = session_info.get("decoded_output")
        out_detail = next((o for o in output_details if o.get("index") == out_idx), output_details[0])
        dets = interpreter.get_tensor(out_detail["index"])
        scale, zero = out_detail.get("quantization", (0.0, 0))
        if np.issubdtype(out_detail.get("dtype", np.float32), np.integer):
            dets = _dequantize_output(dets, float(scale), int(zero))
        return dets[0] if dets.ndim >= 3 else dets

    anchors = session_info.get("anchors")
    wh_scale = session_info.get("wh_scale")
    raw = None
    for detail in output_details:
        val = interpreter.get_tensor(detail["index"])
        scale, zero = detail.get("quantization", (0.0, 0))
        if np.issubdtype(detail.get("dtype", np.float32), np.integer):
            val = _dequantize_output(val, float(scale), int(zero))
        if raw is None and val.ndim == 4:
            raw = val
        elif val.ndim == 2 and val.shape[1] == 2:
            name_l = detail.get("name", "").lower()
            if anchors is None and ("anchor" in name_l or True):
                anchors = val.astype(np.float32)
            if wh_scale is None and ("wh_scale" in name_l or "scale" in name_l):
                wh_scale = val.astype(np.float32)
                session_info["has_quality"] = True

    if raw is None and output_details:
        detail = output_details[0]
        raw = interpreter.get_tensor(detail["index"])
        scale, zero = detail.get("quantization", (0.0, 0))
        if np.issubdtype(detail.get("dtype", np.float32), np.integer):
            raw = _dequantize_output(raw, float(scale), int(zero))

    if raw is None:
        raise RuntimeError("LiteRT output not found.")

    if raw.ndim == 4:
        raw = np.transpose(raw, (0, 3, 1, 2))
    elif raw.ndim == 3:
        raw = np.transpose(raw, (2, 0, 1))[None, ...]

    if anchors is None:
        c = raw.shape[1] if raw.ndim == 4 else raw.shape[-1]
        na_hint = session_info.get("anchor_hint")
        na_guess = _infer_anchor_count_from_channels(int(c))
        na = na_hint if na_hint is not None else (na_guess if na_guess is not None else 3)
        anchors = _build_fallback_anchors(na)
        print(f"[WARN] Anchors not found in LiteRT; using fallback anchors (A={na}).")

    session_info["anchors"] = anchors
    if wh_scale is not None:
        session_info["wh_scale"] = wh_scale

    c = raw.shape[1] if raw.ndim == 4 else raw.shape[-1]
    if anchors is not None and (c % anchors.shape[0] == 0):
        per_anchor = int(c) // int(anchors.shape[0])
        if per_anchor >= 7:
            session_info["has_quality"] = True

    return decode_ultratinyod_raw(
        raw,
        anchors=anchors,
        conf_thresh=0.0,
        has_quality=session_info.get("has_quality", False),
        wh_scale=wh_scale,
        score_mode=score_mode,
        quality_power=quality_power,
        multi_label_mode=multi_label_mode,
        det_class_indices=det_class_indices,
    )


def run_and_decode(
    session,
    session_info: dict,
    inp: np.ndarray,
    conf_thresh: float,
) -> np.ndarray:
    if session_info.get("backend") == "litert":
        return run_and_decode_litert(session, session_info, inp, conf_thresh)
    return run_and_decode_onnx(session, session_info, inp, conf_thresh)


def run_images(
    session,
    session_info: dict,
    img_dir: Path,
    out_dir: Path,
    img_size: Tuple[int, int],
    conf_thresh: float,
    actual_size: bool,
    use_nms: bool,
    nms_iou: float,
    use_dynamic_threshold: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    dynamic_resize = bool(session_info.get("dynamic_resize", False))
    backend = session_info.get("backend", "onnx")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    if not images:
        print(f"No images found under {img_dir}")
        return

    for img_path in images:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Skip unreadable file: {img_path}")
            continue
        h, w = img_bgr.shape[:2]
        target_h, target_w = img_size if actual_size else (h, w)
        if backend == "litert":
            inp = preprocess_litert(img_bgr, img_size, session_info["input_details"], dynamic_resize=dynamic_resize)
        else:
            inp = preprocess(img_bgr, img_size, dynamic_resize=dynamic_resize)
        dets = run_and_decode(session, session_info, inp, conf_thresh)
        used_thresh = conf_thresh
        boxes = postprocess(dets, (target_h, target_w), conf_thresh)
        if use_dynamic_threshold and not boxes and dets.size > 0 and conf_thresh > 0.05:
            fallback_thresh = max(0.05, conf_thresh * 0.5)
            boxes = postprocess(dets, (target_h, target_w), fallback_thresh)
            used_thresh = fallback_thresh
        if use_nms:
            boxes = non_max_suppression(boxes, nms_iou)
        base = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR) if actual_size else img_bgr
        vis_out = draw_boxes(base, boxes, (0, 0, 255))
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), vis_out)
        print(f"Saved {save_path} (detections: {len(boxes)}, conf_thresh={used_thresh:.2f})")


def run_camera(
    session,
    session_info: dict,
    camera_id: int,
    img_size: Tuple[int, int],
    conf_thresh: float,
    record_path: Optional[Path] = None,
    actual_size: bool = False,
    use_nms: bool = False,
    nms_iou: float = 0.8,
    use_dynamic_threshold: bool = False,
) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera id {camera_id}")

    writer = None
    last_time = None
    dynamic_resize = bool(session_info.get("dynamic_resize", False))
    backend = session_info.get("backend", "onnx")
    window_title = session_info.get("window_title", "UHD Demo")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        target_h, target_w = img_size if actual_size else (h, w)
        if backend == "litert":
            inp = preprocess_litert(frame, img_size, session_info["input_details"], dynamic_resize=dynamic_resize)
        else:
            inp = preprocess(frame, img_size, dynamic_resize=dynamic_resize)
        dets = run_and_decode(session, session_info, inp, conf_thresh)
        used_thresh = conf_thresh
        boxes = postprocess(dets, (target_h, target_w), conf_thresh)
        if use_dynamic_threshold and not boxes and dets.size > 0 and conf_thresh > 0.05:
            fallback_thresh = max(0.05, conf_thresh * 0.5)
            boxes = postprocess(dets, (target_h, target_w), fallback_thresh)
            used_thresh = fallback_thresh
        if use_nms:
            boxes = non_max_suppression(boxes, nms_iou)
        base = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR) if actual_size else frame
        vis = draw_boxes(base, boxes, (255, 0, 0))

        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        last_time = t1
        if not actual_size:
            label = f"{ms:.2f} ms"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            pad = 6
            x, y = 10, 30
            cv2.rectangle(
                vis,
                (x - pad, y - th - pad),
                (x + tw + pad, y + baseline + pad),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                vis,
                label,
                (x, y),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
            )

        vis_out = cv2.resize(vis, img_size, interpolation=cv2.INTER_LINEAR) if actual_size else vis

        if record_path:
            if writer is None:
                h, w = vis_out.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                if fps <= 0:
                    fps = 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                record_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(record_path), fourcc, fps, (w, h))
            writer.write(vis_out)

        if boxes:
            print(f"[INFO] detections={len(boxes)} conf_thresh={used_thresh:.2f}")

        cv2.imshow(f"{window_title} (press q to quit)", vis_out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if writer is not None:
        writer.release()
        print(f"Saved recording to {record_path}")
    cap.release()
    cv2.destroyAllWindows()


def parse_size(arg: str) -> Tuple[int, int]:
    s = str(arg).lower().replace(" ", "")
    if "x" in s:
        h, w = s.split("x")
        return int(float(h)), int(float(w))
    v = int(float(s))
    return v, v


def build_args():
    parser = argparse.ArgumentParser(description="UltraTinyOD ONNX/LiteRT demo (CPU).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--images", type=str, help="Directory with images to run batch inference.")
    mode.add_argument("--camera", type=int, help="USB camera id for realtime inference.")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--onnx", help="Path to ONNX model (CPU).")
    model.add_argument("--tflite", help="Path to LiteRT (TFLite) model.")
    parser.add_argument("--output", type=str, default="demo_output", help="Output directory for image mode.")
    parser.add_argument("--img-size", type=str, default="64x64", help="Input size HxW, e.g., 64x64.")
    parser.add_argument("--conf-thresh", type=float, default=0.80, help="Confidence threshold.")
    parser.add_argument(
        "--score-mode",
        type=str,
        default=None,
        choices=["obj_quality_cls", "quality_cls", "obj_cls", "obj_quality", "quality", "obj", "cls"],
        help="Score mode for raw outputs (defaults to model metadata when available).",
    )
    parser.add_argument(
        "--quality-power",
        type=float,
        default=None,
        help="Exponent for quality score (defaults to model metadata when available).",
    )
    parser.add_argument(
        "--swap-logit-samples",
        type=int,
        default=5,
        help="Number of frames to decide quality/cls swap (integer_quant only).",
    )
    parser.add_argument(
        "--record",
        type=str,
        default="camera_record.mp4",
        help="MP4 path for automatic recording when --camera is used.",
    )
    parser.add_argument(
        "--actual-size",
        action="store_true",
        help="Display and recording use the model input resolution instead of the original frame size.",
    )
    parser.add_argument(
        "--use-nms",
        action="store_true",
        help="Apply Non-Maximum Suppression on decoded boxes (default IoU=0.8).",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (effective only when --use-nms is set).",
    )
    parser.add_argument(
        "--use-dynamic-threshold",
        action="store_true",
        help="Enable dynamic confidence threshold fallback when no boxes are found.",
    )
    return parser


def main():
    args = build_args().parse_args()
    img_size = parse_size(args.img_size)
    if args.onnx:
        session, session_info = load_session(args.onnx, img_size)
    else:
        session, session_info = load_litert_session(args.tflite, img_size)
        input_hw = session_info.get("input_hw")
        if input_hw and not session_info.get("dynamic_resize", False) and img_size != input_hw:
            print(f"[WARN] Overriding --img-size {img_size} to match LiteRT input {input_hw}.")
            img_size = input_hw
    score_mode = args.score_mode or session_info.get("score_mode") or "obj_quality_cls"
    session_info["score_mode"] = score_mode
    quality_power = args.quality_power if args.quality_power is not None else session_info.get("quality_power")
    if quality_power is None:
        quality_power = 1.0
    session_info["quality_power"] = float(quality_power)
    session_info["swap_logit_samples"] = int(args.swap_logit_samples)

    if args.images:
        run_images(
            session,
            session_info,
            Path(args.images),
            Path(args.output),
            img_size,
            args.conf_thresh,
            args.actual_size,
            args.use_nms,
            args.nms_iou,
            args.use_dynamic_threshold,
        )
    else:
        record_path = Path(args.record) if args.record else None
        run_camera(
            session,
            session_info,
            int(args.camera),
            img_size,
            args.conf_thresh,
            record_path,
            args.actual_size,
            args.use_nms,
            args.nms_iou,
            args.use_dynamic_threshold,
        )


if __name__ == "__main__":
    main()
