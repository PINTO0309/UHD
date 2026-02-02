#!/usr/bin/env python3
"""
Lite demo for UltraTinyOD ONNX models that output only box + quality (+ anchors/wh_scale).

Example:
  python demo_uhd_lite.py \
    --images partial_images \
    --onnx ultratinyod_anc8_w32_64x64_opencv_inter_nearest_static_nopost.onnx

  python demo_uhd_lite.py \
    --camera 0 \
    --onnx ultratinyod_anc8_w40_64x64_opencv_inter_nearest_static_nopost.onnx
"""
import argparse
import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


def parse_size(arg: str) -> Tuple[int, int]:
    s = str(arg).lower().replace(" ", "")
    if "x" in s:
        h, w = s.split("x")
        return int(float(h)), int(float(w))
    v = int(float(s))
    return v, v


def preprocess(img_bgr: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_NEAREST)
    arr = resized.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return chw[np.newaxis, ...]


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _parse_anchor_hint_from_path(onnx_path: str) -> Optional[int]:
    name = os.path.basename(onnx_path).lower()
    m = re.search(r"anc(\d+)", name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _build_fallback_anchors(na: int) -> np.ndarray:
    return np.stack(
        [
            np.linspace(0.08, 0.32, na, dtype=np.float32),
            np.linspace(0.10, 0.40, na, dtype=np.float32),
        ],
        axis=1,
    )


def decode_box_quality(
    box: np.ndarray,
    quality: np.ndarray,
    anchors: np.ndarray,
    wh_scale: Optional[np.ndarray],
    conf_thresh: float,
    target_hw: Tuple[int, int],
    topk: int,
    topk_before_conf: bool,
) -> List[Tuple[float, float, float, float, float]]:
    if box.ndim == 3:
        box = box[None, ...]
    if quality.ndim == 3:
        quality = quality[None, ...]
    if box.ndim != 4 or quality.ndim != 4:
        raise ValueError("Expected 4D tensors for box/quality.")

    b, c_box, h, w = box.shape
    na = int(quality.shape[1])
    if c_box != na * 4:
        raise ValueError(f"Unexpected box channels {c_box} for anchors={na}.")

    anchors = np.asarray(anchors, dtype=np.float32)
    if anchors.shape[0] != na or anchors.shape[1] != 2:
        raise ValueError(f"Unexpected anchors shape {anchors.shape} for anchors={na}.")
    if wh_scale is not None and wh_scale.shape == anchors.shape:
        anchors = anchors * wh_scale

    pred = box.reshape(b, na, 4, h, w)
    tx = pred[:, :, 0]
    ty = pred[:, :, 1]
    tw = pred[:, :, 2]
    th = pred[:, :, 3]

    score = sigmoid_np(quality)

    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    gx = gx.reshape(1, 1, h, w)
    gy = gy.reshape(1, 1, h, w)

    pw = anchors[:, 0].reshape(1, na, 1, 1)
    ph = anchors[:, 1].reshape(1, na, 1, 1)

    cx = (sigmoid_np(tx) + gx) / float(w)
    cy = (sigmoid_np(ty) + gy) / float(h)
    bw = pw * softplus_np(tw)
    bh = ph * softplus_np(th)

    scores = score[0]
    scores_flat = scores.reshape(-1)
    cx_flat = cx[0].reshape(-1)
    cy_flat = cy[0].reshape(-1)
    bw_flat = bw[0].reshape(-1)
    bh_flat = bh[0].reshape(-1)

    if topk_before_conf and topk is not None and topk > 0 and scores_flat.size > topk:
        top_idx = np.argsort(-scores_flat)[:topk]
        if conf_thresh > 0:
            top_idx = top_idx[scores_flat[top_idx] >= conf_thresh]
        idx = top_idx
    else:
        if conf_thresh > 0:
            idx = np.where(scores_flat >= conf_thresh)[0]
        else:
            idx = np.arange(scores_flat.size)
        if topk is not None and topk > 0 and idx.size > topk:
            sub_scores = scores_flat[idx]
            order = np.argsort(-sub_scores)[:topk]
            idx = idx[order]

    if idx.size == 0:
        return []

    cx_sel = cx_flat[idx]
    cy_sel = cy_flat[idx]
    bw_sel = bw_flat[idx]
    bh_sel = bh_flat[idx]
    sc_sel = scores_flat[idx]

    target_h, target_w = target_hw
    x1 = (cx_sel - bw_sel / 2.0) * target_w
    y1 = (cy_sel - bh_sel / 2.0) * target_h
    x2 = (cx_sel + bw_sel / 2.0) * target_w
    y2 = (cy_sel + bh_sel / 2.0) * target_h

    x1 = np.clip(x1, 0.0, float(target_w))
    x2 = np.clip(x2, 0.0, float(target_w))
    y1 = np.clip(y1, 0.0, float(target_h))
    y2 = np.clip(y2, 0.0, float(target_h))

    out = []
    for s, a, b_, c, d in zip(sc_sel, x1, y1, x2, y2):
        if c <= a or d <= b_:
            continue
        out.append((float(s), float(a), float(b_), float(c), float(d)))
    return out


def draw_boxes(img_bgr: np.ndarray, boxes: List[Tuple[float, float, float, float, float]]) -> np.ndarray:
    out = img_bgr.copy()
    for score, x1, y1, x2, y2 in boxes:
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)
        label = f"{score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        tx = x1i
        ty = max(y1i - 4, th + 4)
        cv2.rectangle(out, (tx, ty - th - baseline), (tx + tw, ty + baseline), (0, 0, 255), -1)
        cv2.putText(out, label, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def nms_iou_single_class(
    boxes: List[Tuple[float, float, float, float, float]],
    iou_thresh: float,
) -> List[Tuple[float, float, float, float, float]]:
    if not boxes:
        return boxes
    arr = np.array(boxes, dtype=np.float32)
    scores = arr[:, 0]
    x1, y1, x2, y2 = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []
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


def load_session(onnx_path: str):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.get_outputs()
    name_map = {o.name.lower(): o.name for o in outputs}
    if "box" not in name_map or "quality" not in name_map:
        raise RuntimeError("ONNX outputs must include 'box' and 'quality'.")
    box_name = name_map["box"]
    quality_name = name_map["quality"]
    anchors_name = name_map.get("anchors")
    wh_scale_name = name_map.get("wh_scale")
    return sess, {
        "input_name": input_name,
        "box": box_name,
        "quality": quality_name,
        "anchors": anchors_name,
        "wh_scale": wh_scale_name,
    }


def dump_anchors_wh_scale(
    onnx_path: str,
    sess: ort.InferenceSession,
    info: dict,
    img_size: Tuple[int, int],
) -> None:
    anchor_name = info.get("anchors")
    wh_scale_name = info.get("wh_scale")
    if not anchor_name and not wh_scale_name:
        print("[WARN] anchors/wh_scale outputs not found; skip .npy export.")
        return

    inp = sess.get_inputs()[0]
    shape = inp.shape
    c = 3
    h, w = img_size
    if isinstance(shape, (list, tuple)) and len(shape) >= 4:
        if shape[1] not in (None, "None"):
            try:
                c = int(shape[1])
            except Exception:
                c = 3
        if shape[2] not in (None, "None"):
            try:
                h = int(shape[2])
            except Exception:
                h = img_size[0]
        if shape[3] not in (None, "None"):
            try:
                w = int(shape[3])
            except Exception:
                w = img_size[1]

    dummy = np.zeros((1, c, h, w), dtype=np.float32)
    outputs = []
    if anchor_name:
        outputs.append(anchor_name)
    if wh_scale_name:
        outputs.append(wh_scale_name)
    out_vals = sess.run(outputs, {info["input_name"]: dummy})
    out_map = {name: val for name, val in zip(outputs, out_vals)}

    path = Path(onnx_path)
    if anchor_name and anchor_name in out_map:
        np.save(path.with_name(f"{path.stem}_anchors.npy"), out_map[anchor_name])
    if wh_scale_name and wh_scale_name in out_map:
        np.save(path.with_name(f"{path.stem}_wh_scale.npy"), out_map[wh_scale_name])


def load_sidecar_anchors_wh_scale(onnx_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    path = Path(onnx_path)
    anchors_path = path.with_name(f"{path.stem}_anchors.npy")
    wh_scale_path = path.with_name(f"{path.stem}_wh_scale.npy")
    anchors = np.load(str(anchors_path)).astype(np.float32) if anchors_path.is_file() else None
    wh_scale = np.load(str(wh_scale_path)).astype(np.float32) if wh_scale_path.is_file() else None
    return anchors, wh_scale


def run_inference(
    sess: ort.InferenceSession,
    info: dict,
    inp: np.ndarray,
    anchors_cache: Optional[np.ndarray],
    wh_scale_cache: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    outputs = [info["box"], info["quality"]]
    if info.get("anchors") and anchors_cache is None:
        outputs.append(info["anchors"])
    if info.get("wh_scale") and wh_scale_cache is None:
        outputs.append(info["wh_scale"])
    out_vals = sess.run(outputs, {info["input_name"]: inp})
    out_map = {name: val for name, val in zip(outputs, out_vals)}
    box = out_map[info["box"]]
    quality = out_map[info["quality"]]
    anchors = out_map.get(info.get("anchors") or "") if anchors_cache is None else anchors_cache
    wh_scale = out_map.get(info.get("wh_scale") or "") if wh_scale_cache is None else wh_scale_cache
    return box, quality, anchors, wh_scale


def run_images(
    sess: ort.InferenceSession,
    info: dict,
    img_dir: Path,
    out_dir: Path,
    img_size: Tuple[int, int],
    conf_thresh: float,
    use_nms: bool,
    nms_iou: float,
    topk: int,
    topk_before_conf: bool,
    anchors_init: Optional[np.ndarray],
    wh_scale_init: Optional[np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    if not images:
        print(f"No images found under {img_dir}")
        return

    anchors_cache = anchors_init
    wh_scale_cache = wh_scale_init
    anchor_hint = _parse_anchor_hint_from_path(info.get("onnx_path", ""))

    for img_path in images:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Skip unreadable file: {img_path}")
            continue
        h, w = img_bgr.shape[:2]
        inp = preprocess(img_bgr, img_size)
        box, quality, anchors, wh_scale = run_inference(sess, info, inp, anchors_cache, wh_scale_cache)
        if anchors is None:
            if anchor_hint is None:
                raise RuntimeError("Anchors not found; export ONNX with anchors output.")
            anchors = _build_fallback_anchors(anchor_hint)
        anchors_cache = anchors
        wh_scale_cache = wh_scale

        boxes = decode_box_quality(box, quality, anchors, wh_scale, conf_thresh, (h, w), topk, topk_before_conf)
        if use_nms:
            boxes = nms_iou_single_class(boxes, nms_iou)
        vis = draw_boxes(img_bgr, boxes)
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), vis)
        print(f"Saved {save_path} (detections: {len(boxes)})")


def run_camera(
    sess: ort.InferenceSession,
    info: dict,
    camera_id: int,
    img_size: Tuple[int, int],
    conf_thresh: float,
    record_path: Optional[Path] = None,
    use_nms: bool = False,
    nms_iou: float = 0.45,
    topk: int = 20,
    topk_before_conf: bool = False,
    anchors_init: Optional[np.ndarray] = None,
    wh_scale_init: Optional[np.ndarray] = None,
) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera id {camera_id}")

    writer = None
    anchors_cache = anchors_init
    wh_scale_cache = wh_scale_init
    anchor_hint = _parse_anchor_hint_from_path(info.get("onnx_path", ""))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        inp = preprocess(frame, img_size)
        box, quality, anchors, wh_scale = run_inference(sess, info, inp, anchors_cache, wh_scale_cache)
        if anchors is None:
            if anchor_hint is None:
                raise RuntimeError("Anchors not found; export ONNX with anchors output.")
            anchors = _build_fallback_anchors(anchor_hint)
        anchors_cache = anchors
        wh_scale_cache = wh_scale

        boxes = decode_box_quality(box, quality, anchors, wh_scale, conf_thresh, (h, w), topk, topk_before_conf)
        if use_nms:
            boxes = nms_iou_single_class(boxes, nms_iou)
        vis = draw_boxes(frame, boxes)

        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        label = f"{ms:.2f} ms"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, label, (10, 30), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        if record_path:
            if writer is None:
                hh, ww = vis.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                if fps <= 0:
                    fps = 30.0
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                record_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(record_path), fourcc, fps, (ww, hh))
            writer.write(vis)

        if boxes:
            print(f"[INFO] detections={len(boxes)} conf_thresh={conf_thresh:.2f}")

        cv2.imshow("UHD Lite (press q to quit)", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if writer is not None:
        writer.release()
        print(f"Saved recording to {record_path}")
    cap.release()
    cv2.destroyAllWindows()


def build_args():
    parser = argparse.ArgumentParser(description="UltraTinyOD ONNX lite demo (box+quality only).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--images", type=str, help="Directory with images to run batch inference.")
    mode.add_argument("--camera", type=int, help="USB camera id for realtime inference.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model (CPU).")
    parser.add_argument("--output", type=str, default="demo_output", help="Output directory for image mode.")
    parser.add_argument("--img-size", type=str, default="64x64", help="Input size HxW, e.g., 64x64.")
    parser.add_argument("--conf-thresh", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--record", type=str, default="camera_record.mp4", help="MP4 path for recording in camera mode.")
    parser.add_argument("--use-nms", action="store_true", help="Apply IoU NMS (single class).")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--topk", type=int, default=20, help="Keep top-K boxes before NMS.")
    parser.add_argument("--topk-before-conf", action="store_true", help="Apply top-K before confidence threshold.")
    return parser


def main():
    args = build_args().parse_args()
    img_size = parse_size(args.img_size)
    sess, info = load_session(args.onnx)
    info["onnx_path"] = args.onnx
    dump_anchors_wh_scale(args.onnx, sess, info, img_size)
    anchors_sidecar, wh_scale_sidecar = load_sidecar_anchors_wh_scale(args.onnx)

    if args.images:
        run_images(
            sess,
            info,
            Path(args.images),
            Path(args.output),
            img_size,
            args.conf_thresh,
            args.use_nms,
            args.nms_iou,
            args.topk,
            args.topk_before_conf,
            anchors_sidecar,
            wh_scale_sidecar,
        )
    else:
        record_path = Path(args.record) if args.record else None
        run_camera(
            sess,
            info,
            int(args.camera),
            img_size,
            args.conf_thresh,
            record_path,
            args.use_nms,
            args.nms_iou,
            args.topk,
            args.topk_before_conf,
            anchors_sidecar,
            wh_scale_sidecar,
        )


if __name__ == "__main__":
    main()
