import argparse
import glob
import os
import time
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort


def parse_img_size(arg):
    if arg is None:
        return None
    s = str(arg).lower().replace(" ", "")
    if "x" in s:
        parts = s.split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    try:
        v = int(float(s))
        return v, v
    except ValueError:
        return None


def parse_anchors_str(arg: str) -> List[Tuple[float, float]]:
    anchors: List[Tuple[float, float]] = []
    if not arg:
        return anchors
    for part in str(arg).split():
        nums = part.split(",")
        if len(nums) != 2:
            continue
        try:
            anchors.append((float(nums[0]), float(nums[1])))
        except ValueError:
            continue
    return anchors


def get_providers(device: str | None = None) -> List[str]:
    available = ort.get_available_providers()
    if device == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cpu" or "CUDAExecutionProvider" not in available:
        return ["CPUExecutionProvider"]
    # default: try CUDA first when present
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def gather_images(image_dir: str, recursive: bool = True) -> List[str]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths: List[str] = []
    glob_opt = "**/" if recursive else ""
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(image_dir, glob_opt + pat), recursive=recursive))
    return sorted(paths)


def to_numpy_box(box) -> np.ndarray:
    return np.asarray(box, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nms_per_class(boxes: List[Tuple[float, int, np.ndarray]], iou_thresh: float) -> List[Tuple[float, int, np.ndarray]]:
    if not boxes:
        return []
    by_cls = {}
    for sc, cls, box in boxes:
        by_cls.setdefault(cls, []).append((sc, box))
    kept: List[Tuple[float, int, np.ndarray]] = []
    for cls, items in by_cls.items():
        items = sorted(items, key=lambda x: x[0], reverse=True)
        while items:
            sc, ref = items.pop(0)
            keep_rest = []
            kept.append((sc, cls, ref))
            if not items:
                break
            ref_xyxy = _cxcywh_to_xyxy(ref)
            for sc2, box2 in items:
                iou = _box_iou_np(ref_xyxy, _cxcywh_to_xyxy(box2))
                if iou < iou_thresh:
                    keep_rest.append((sc2, box2))
            items = keep_rest
    kept.sort(key=lambda x: x[0], reverse=True)
    return kept


def _apply_nms_list(
    decoded: List[List[Tuple[float, int, np.ndarray]]], iou_thresh: float
) -> List[List[Tuple[float, int, np.ndarray]]]:
    if iou_thresh <= 0:
        return decoded
    out: List[List[Tuple[float, int, np.ndarray]]] = []
    for dets in decoded:
        out.append(_nms_per_class(dets, iou_thresh))
    return out


def _cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in box]
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def _box_iou_np(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> float:
    x1 = max(a_xyxy[0], b_xyxy[0])
    y1 = max(a_xyxy[1], b_xyxy[1])
    x2 = min(a_xyxy[2], b_xyxy[2])
    y2 = min(a_xyxy[3], b_xyxy[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a_xyxy[2] - a_xyxy[0]) * max(0.0, a_xyxy[3] - a_xyxy[1])
    area_b = max(0.0, b_xyxy[2] - b_xyxy[0]) * max(0.0, b_xyxy[3] - b_xyxy[1])
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def _decode_centernet_np(
    hm: np.ndarray, off: np.ndarray, wh: np.ndarray, conf_thresh: float, topk: int
) -> List[List[Tuple[float, int, np.ndarray]]]:
    b, c, h, w = hm.shape
    hm_flat = hm.reshape(b, -1)
    decoded: List[List[Tuple[float, int, np.ndarray]]] = []
    for bi in range(b):
        scores_all = hm_flat[bi]
        k = min(topk, scores_all.shape[0])
        idxs = np.argpartition(-scores_all, k - 1)[:k]
        idxs = idxs[np.argsort(-scores_all[idxs])]
        boxes_img: List[Tuple[float, int, np.ndarray]] = []
        for idx in idxs:
            score = float(scores_all[idx])
            if score < conf_thresh:
                continue
            cls = int(idx // (h * w))
            rem = idx % (h * w)
            y = rem // w
            x = rem % w
            dx = off[bi, 0, y, x]
            dy = off[bi, 1, y, x]
            pw = wh[bi, 0, y, x]
            ph = wh[bi, 1, y, x]
            cx = (x + dx) / float(w)
            cy = (y + dy) / float(h)
            bw = pw / float(w)
            bh = ph / float(h)
            boxes_img.append((score, cls, np.array([cx, cy, bw, bh], dtype=np.float32)))
        decoded.append(boxes_img)
    return decoded


def _decode_detr_np(logits: np.ndarray, boxes: np.ndarray, conf_thresh: float) -> List[List[Tuple[float, int, np.ndarray]]]:
    # logits expected shape Q x B x (C+1); boxes Q x B x 4
    if logits.shape[:2] != boxes.shape[:2]:
        # try B x Q x C case
        logits = np.transpose(logits, (1, 0, 2))
        boxes = np.transpose(boxes, (1, 0, 2))
    q, b, num_classes_bg = logits.shape
    num_classes = num_classes_bg - 1
    decoded: List[List[Tuple[float, int, np.ndarray]]] = []
    for bi in range(b):
        boxes_img: List[Tuple[float, int, np.ndarray]] = []
        for qi in range(q):
            logits_q = logits[qi, bi]
            cls_logits = logits_q[:num_classes]
            # stable softmax on (C+1)
            max_l = np.max(logits_q)
            exp_all = np.exp(logits_q - max_l)
            probs = exp_all / np.sum(exp_all)
            cls_prob = probs[:num_classes]
            cls_idx = int(np.argmax(cls_prob))
            sc = float(cls_prob[cls_idx])
            if sc < conf_thresh or sc <= float(probs[num_classes]):  # skip if background higher
                continue
            box = boxes[qi, bi]
            boxes_img.append((sc, cls_idx, to_numpy_box(box)))
        decoded.append(boxes_img)
    return decoded


def _decode_anchor_np(
    pred: np.ndarray,
    anchors: List[Tuple[float, float]],
    num_classes: int,
    conf_thresh: float,
    nms_thresh: float,
) -> List[List[Tuple[float, int, np.ndarray]]]:
    b, _, h, w = pred.shape
    if b == 0:
        return []
    anchors_np = np.asarray(anchors, dtype=np.float32)
    na = anchors_np.shape[0]
    denom = (pred.shape[1] // na) if na > 0 else (5 + num_classes)
    extra = max(0, denom - (5 + num_classes))
    has_quality = extra >= 1
    pred = pred.reshape(b, na, 5 + extra + num_classes, h, w).transpose(0, 1, 3, 4, 2)
    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    obj = _sigmoid(pred[..., 4])
    quality = _sigmoid(pred[..., 5]) if has_quality else None
    cls = _sigmoid(pred[..., (5 + extra):])

    def _softplus_np(x):
        # stable softplus
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing="xy")
    gx = gx.reshape(1, 1, h, w)
    gy = gy.reshape(1, 1, h, w)
    pred_cx = (_sigmoid(tx) + gx) / float(w)
    pred_cy = (_sigmoid(ty) + gy) / float(h)
    pred_w = anchors_np[:, 0].reshape(1, na, 1, 1) * np.clip(_softplus_np(tw), None, 4.0)
    pred_h = anchors_np[:, 1].reshape(1, na, 1, 1) * np.clip(_softplus_np(th), None, 4.0)
    decoded: List[List[Tuple[float, int, np.ndarray]]] = []
    score_base = obj
    if quality is not None:
        score_base = score_base * quality
    scores = score_base[..., None] * cls  # B x A x H x W x C
    for bi in range(b):
        boxes_img: List[Tuple[float, int, np.ndarray]] = []
        score_map = scores[bi]  # A x H x W x C
        flat_scores = score_map.reshape(-1, num_classes)
        max_scores = flat_scores.max(axis=1)
        max_cls = flat_scores.argmax(axis=1)
        mask = max_scores >= conf_thresh
        if np.any(mask):
            idxs = np.nonzero(mask)[0]
            a_idx = idxs // (h * w)
            rem = idxs % (h * w)
            gy_idx = rem // w
            gx_idx = rem % w
            cx_sel = pred_cx[bi, a_idx, gy_idx, gx_idx]
            cy_sel = pred_cy[bi, a_idx, gy_idx, gx_idx]
            bw_sel = pred_w[bi, a_idx, gy_idx, gx_idx]
            bh_sel = pred_h[bi, a_idx, gy_idx, gx_idx]
            sel_scores = max_scores[mask]
            sel_cls = max_cls[mask]
            for sc, cls_id, cx, cy, bw, bh in zip(sel_scores, sel_cls, cx_sel, cy_sel, bw_sel, bh_sel):
                boxes_img.append((float(sc), int(cls_id), np.array([cx, cy, bw, bh], dtype=np.float32)))
            boxes_img = _nms_per_class(boxes_img, iou_thresh=nms_thresh)
        decoded.append(boxes_img)
    return decoded


class OnnxDetector:
    def __init__(self, onnx_path: str, img_size: Tuple[int, int] | None = None, device: str | None = None) -> None:
        providers = get_providers(device)
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.model_anchors = self._load_model_anchors(onnx_path)
        self.arch = self._detect_arch()
        self.input_hw = self._resolve_input_hw(img_size)

    def _detect_arch(self) -> str:
        outs = set(self.output_names)
        if "detections" in outs:
            return "merged"
        if {"hm", "off", "wh"} <= outs:
            return "centernet"
        if {"logits", "boxes"} <= outs:
            return "transformer"
        if len(outs) == 1:
            return "anchor"
        return "unknown"

    def _resolve_input_hw(self, override: Tuple[int, int] | None) -> Tuple[int, int]:
        if override:
            return override
        shape = self.session.get_inputs()[0].shape
        if len(shape) >= 4:
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return int(h), int(w)
        return 64, 64

    def _load_model_anchors(self, onnx_path: str) -> List[Tuple[float, float]]:
        """
        Best-effort extraction of anchor tensors embedded in the ONNX graph.
        Used to mirror the training/validation decode when anchors are not supplied.
        """
        try:
            import onnx
            from onnx import numpy_helper
        except Exception:
            return []
        try:
            model = onnx.load(onnx_path)
        except Exception:
            return []

        direct_pairs: List[np.ndarray] = []
        split_pairs: List[np.ndarray] = []
        for init in model.graph.initializer:
            try:
                arr = numpy_helper.to_array(init)
            except Exception:
                continue
            if not isinstance(arr, np.ndarray):
                continue
            if arr.dtype.kind not in ("f", "i"):
                continue
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                direct_pairs.append(arr.astype(np.float32))
            elif arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[2] == 1 and arr.shape[3] == 1:
                flat = arr.reshape(-1).astype(np.float32)
                if flat.size > 1:
                    split_pairs.append(flat)

        if direct_pairs:
            arr = direct_pairs[0]
            return [(float(w), float(h)) for w, h in arr.tolist()]
        for i, a in enumerate(split_pairs):
            for b in split_pairs[i + 1 :]:
                if a.shape == b.shape:
                    return [(float(w), float(h)) for w, h in zip(a.tolist(), b.tolist())]
        return []

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h_in, w_in = self.input_hw
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return arr, (img_bgr.shape[1], img_bgr.shape[0])

    def _decode(
        self,
        outputs: Sequence[np.ndarray],
        conf_thresh: float,
        topk: int,
        anchors: List[Tuple[float, float]],
        num_classes: int,
        nms_thresh: float,
    ) -> List[List[Tuple[float, int, np.ndarray]]]:
        if self.arch == "merged":
            det_array = outputs[0]
            if det_array.ndim == 2:
                det_array = det_array[None, ...]
            decoded: List[List[Tuple[float, int, np.ndarray]]] = []
            for dets in det_array:
                batch_list: List[Tuple[float, int, np.ndarray]] = []
                for row in dets:
                    score = float(row[0])
                    if score < conf_thresh:
                        continue
                    cls = int(round(row[1]))
                    box = np.asarray(row[2:6], dtype=np.float32)
                    batch_list.append((score, cls, box))
                decoded.append(batch_list)
            return _apply_nms_list(decoded, nms_thresh)

        out_map = {name: outputs[idx] for idx, name in enumerate(self.output_names)}
        if self.arch == "centernet":
            hm = out_map["hm"]
            off = out_map["off"]
            wh = out_map["wh"]
            decoded = _decode_centernet_np(hm, off, wh, conf_thresh=conf_thresh, topk=topk)
            return _apply_nms_list(decoded, nms_thresh)
        if self.arch == "transformer":
            logits = out_map.get("logits", outputs[0])
            boxes = out_map.get("boxes", outputs[1] if len(outputs) > 1 else outputs[0])
            decoded = _decode_detr_np(logits, boxes, conf_thresh=conf_thresh)
            return _apply_nms_list(decoded, nms_thresh)
        if self.arch == "anchor":
            if not anchors:
                anchors = self.model_anchors
            if not anchors:
                raise ValueError("Anchor outputs require --anchors (normalized w,h pairs).")
            pred = outputs[0]
            return _decode_anchor_np(pred, anchors=anchors, num_classes=num_classes, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
        raise ValueError(f"Unsupported ONNX outputs: {self.output_names}")

    def predict(
        self,
        img_bgr: np.ndarray,
        conf_thresh: float,
        topk: int,
        anchors: List[Tuple[float, float]],
        num_classes: int,
        nms_thresh: float,
    ) -> List[Tuple[float, int, np.ndarray]]:
        inp, orig_size = self.preprocess(img_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: inp})
        decoded = self._decode(outputs, conf_thresh, topk, anchors, num_classes, nms_thresh)
        if not decoded:
            return []
        # decoded is per-batch; we only send one image at a time
        norm_dets = []
        for score, cls, box in decoded[0]:
            norm_dets.append((float(score), int(cls), to_numpy_box(box)))
        norm_dets = self._maybe_normalize_boxes(norm_dets)
        return self.scale_boxes(norm_dets, orig_size)

    def _maybe_normalize_boxes(
        self, detections: List[Tuple[float, int, np.ndarray]]
    ) -> List[Tuple[float, int, np.ndarray]]:
        """Ensure boxes are normalized; if values look like pixels, scale by input size."""
        h_in, w_in = self.input_hw
        normed: List[Tuple[float, int, np.ndarray]] = []
        for score, cls, box in detections:
            box_np = to_numpy_box(box).astype(np.float32)
            max_abs = float(np.max(np.abs(box_np))) if box_np.size else 0.0
            if max_abs <= 1.5:
                normed.append((score, cls, box_np))
                continue
            cx, cy, bw, bh = box_np
            normed.append((score, cls, np.array([cx / w_in, cy / h_in, bw / w_in, bh / h_in], dtype=np.float32)))
        return normed

    @staticmethod
    def scale_boxes(
        detections: List[Tuple[float, int, np.ndarray]], orig_size: Tuple[int, int]
    ) -> List[Tuple[float, int, Tuple[int, int, int, int]]]:
        ow, oh = orig_size
        scaled: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
        for score, cls, box in detections:
            cx, cy, bw, bh = [float(x) for x in box.tolist()]
            x1 = (cx - bw / 2.0) * ow
            y1 = (cy - bh / 2.0) * oh
            x2 = (cx + bw / 2.0) * ow
            y2 = (cy + bh / 2.0) * oh
            x1 = max(0, min(int(round(x1)), ow - 1))
            y1 = max(0, min(int(round(y1)), oh - 1))
            x2 = max(0, min(int(round(x2)), ow - 1))
            y2 = max(0, min(int(round(y2)), oh - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            scaled.append((score, cls, (x1, y1, x2, y2)))
        return scaled


def draw_detections(
    img_bgr: np.ndarray,
    detections: List[Tuple[float, int, Tuple[int, int, int, int]]],
    class_names: Sequence[str],
    header_text: str | None = None,
) -> np.ndarray:
    colors = [
        (0, 165, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    out = img_bgr.copy()
    if header_text:
        cv2.putText(
            out,
            header_text,
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    for score, cls, (x1, y1, x2, y2) in detections:
        color = colors[cls % len(colors)]
        label = class_names[cls] if cls < len(class_names) else f"id{cls}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(
            out,
            f"{label}:{score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def run_batch(
    detector: OnnxDetector,
    image_dir: str,
    output_dir: str,
    anchors: List[Tuple[float, float]],
    class_names: Sequence[str],
    conf_thresh: float,
    topk: int,
    num_classes: int,
    nms_thresh: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    paths = gather_images(image_dir)
    if not paths:
        print(f"No images found under {image_dir}")
        return
    print(f"Found {len(paths)} images. Saving results to {output_dir}")
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read {path}")
            continue
        t0 = time.perf_counter()
        dets = detector.predict(img, conf_thresh, topk, anchors, num_classes, nms_thresh)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        vis = draw_detections(img, dets, class_names, header_text=f"{infer_ms:.1f} ms")
        save_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(save_path, vis)
    print("Batch inference done.")


def run_realtime(
    detector: OnnxDetector,
    anchors: List[Tuple[float, float]],
    class_names: Sequence[str],
    conf_thresh: float,
    topk: int,
    num_classes: int,
    nms_thresh: float,
    camera: int = 0,
    save_video: str | None = None,
) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Could not open camera index {camera}")
        return
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        writer = cv2.VideoWriter(save_video, fourcc, fps, (w, h))
        print(f"Recording to {save_video} ({w}x{h}@{fps:.1f}fps)")
    print("Press 'q' or ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        dets = detector.predict(frame, conf_thresh, topk, anchors, num_classes, nms_thresh)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        vis = draw_detections(frame, dets, class_names, header_text=f"{infer_ms:.1f} ms")
        if writer is not None:
            writer.write(vis)
        cv2.imshow("UHD demo", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="UHD ONNX demo: batch folder inference and realtime webcam.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model exported by export_onnx.py")
    parser.add_argument("--input-dir", help="Directory containing images for batch inference.")
    parser.add_argument("--output-dir", default="demo_output", help="Where to save rendered images.")
    parser.add_argument("--img-size", default=None, help="Override resize HxW (default: inferred from ONNX input).")
    parser.add_argument("--conf-thresh", type=float, default=0.3, help="Score threshold (mirrors training validation default).")
    parser.add_argument("--topk", type=int, default=50, help="Top-K used for Centernet/raw CNN decoding.")
    parser.add_argument("--anchors", default="", help='Anchor list like "0.08,0.10 0.15,0.20" for raw anchor ONNX outputs (auto-detected when omitted).')
    parser.add_argument("--num-classes", type=int, default=1, help="Class count for raw CNN/anchor ONNX outputs.")
    parser.add_argument("--nms-thresh", type=float, default=0.5, help="IoU threshold used in anchor NMS.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="ONNX Runtime device (default: auto).")
    parser.add_argument("--realtime", action="store_true", help="Run realtime webcam demo.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for realtime demo.")
    parser.add_argument("--save-video", default=None, help="Optional path to save webcam output (mp4).")
    parser.add_argument("--class-names", default="person", help="Comma-separated class names used for labels.")
    args = parser.parse_args()

    if not args.input_dir and not args.realtime:
        parser.error("Specify --input-dir for batch inference and/or --realtime for webcam demo.")

    img_hw = parse_img_size(args.img_size)
    anchors = parse_anchors_str(args.anchors)
    class_names = [c for c in (name.strip() for name in args.class_names.split(",")) if c]

    detector = OnnxDetector(args.onnx, img_size=img_hw, device=args.device)
    print(f"Loaded ONNX model {args.onnx}")
    print(f"Detected output type: {detector.arch}, input size: {detector.input_hw}")
    if not anchors and detector.model_anchors:
        anchors = detector.model_anchors
        print(f"Using {len(anchors)} anchors parsed from ONNX graph.")

    if args.input_dir:
        run_batch(
            detector,
            args.input_dir,
            args.output_dir,
            anchors,
            class_names,
            args.conf_thresh,
            args.topk,
            args.num_classes,
            args.nms_thresh,
        )
    if args.realtime:
        run_realtime(
            detector,
            anchors,
            class_names,
            args.conf_thresh,
            args.topk,
            args.num_classes,
            args.nms_thresh,
            camera=args.camera,
            save_video=args.save_video,
        )


if __name__ == "__main__":
    main()
