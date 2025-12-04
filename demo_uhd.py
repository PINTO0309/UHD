#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(img_bgr: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return chw[np.newaxis, ...]


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
        out.append((float(score), int(cls_id), float(x1), float(y1), float(x2), float(y2)))
    return out


def draw_boxes(img_bgr: np.ndarray, boxes: List[Tuple[float, int, float, float, float, float]], color: Tuple[int, int, int]) -> np.ndarray:
    out = img_bgr.copy()
    for score, cls_id, x1, y1, x2, y2 in boxes:
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 2)
        label = f"{score:.2f}"
        cv2.putText(out, label, (x1i, max(0, y1i - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def load_session(onnx_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def run_images(session: ort.InferenceSession, img_dir: Path, out_dir: Path, img_size: Tuple[int, int], conf_thresh: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

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
        inp = preprocess(img_bgr, img_size)
        dets = session.run([output_name], {input_name: inp})[0][0]
        boxes = postprocess(dets, (h, w), conf_thresh)
        vis = draw_boxes(img_bgr, boxes, (0, 0, 255))
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), vis)
        print(f"Saved {save_path} (detections: {len(boxes)})")


def run_camera(
    session: ort.InferenceSession,
    camera_id: int,
    img_size: Tuple[int, int],
    conf_thresh: float,
    record_path: Optional[Path] = None,
) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera id {camera_id}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    writer = None
    last_time = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        inp = preprocess(frame, img_size)
        dets = session.run([output_name], {input_name: inp})[0][0]
        boxes = postprocess(dets, (h, w), conf_thresh)
        vis = draw_boxes(frame, boxes, (255, 0, 0))

        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        last_time = t1
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

        if record_path:
            if writer is None:
                h, w = vis.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                if fps <= 0:
                    fps = 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                record_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(record_path), fourcc, fps, (w, h))
            writer.write(vis)

        cv2.imshow("UHD ONNX (press q to quit)", vis)
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
    parser = argparse.ArgumentParser(description="UltraTinyOD ONNX demo (CPU).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--images", type=str, help="Directory with images to run batch inference.")
    mode.add_argument("--camera", type=int, help="USB camera id for realtime inference.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model (CPU).")
    parser.add_argument("--output", type=str, default="demo_output", help="Output directory for image mode.")
    parser.add_argument("--img-size", type=str, default="64x64", help="Input size HxW, e.g., 64x64.")
    parser.add_argument("--conf-thresh", type=float, default=0.30, help="Confidence threshold.")
    parser.add_argument(
        "--record",
        type=str,
        default="camera_record.mp4",
        help="MP4 path for automatic recording when --camera is used.",
    )
    return parser


def main():
    args = build_args().parse_args()
    img_size = parse_size(args.img_size)
    session = load_session(args.onnx)

    if args.images:
        run_images(session, Path(args.images), Path(args.output), img_size, args.conf_thresh)
    else:
        record_path = Path(args.record) if args.record else None
        run_camera(session, int(args.camera), img_size, args.conf_thresh, record_path)


if __name__ == "__main__":
    main()
