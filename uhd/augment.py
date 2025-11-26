import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes.T
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=1)


def _clip_boxes(boxes: np.ndarray) -> np.ndarray:
    boxes[:, 0::2] = boxes[:, 0::2].clip(0.0, 1.0)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0.0, 1.0)
    return boxes


def _filter_boxes(boxes: np.ndarray, labels: np.ndarray, min_area: float) -> Tuple[np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return boxes, labels
    xyxy = _cxcywh_to_xyxy(boxes)
    areas = (xyxy[:, 2] - xyxy[:, 0]).clip(min=0) * (xyxy[:, 3] - xyxy[:, 1]).clip(min=0)
    keep = areas >= min_area
    return boxes[keep], labels[keep]


def _apply_hsv(img: np.ndarray, hue_gain: float, saturation_gain: float, value_gain: float) -> np.ndarray:
    r = np.random.uniform(-1, 1, 3) * np.array([hue_gain, saturation_gain, value_gain]) + 1
    img_hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[..., 0] = (img_hsv[..., 0] * r[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * r[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * r[2], 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img.astype(np.float32) / 255.0


class AugmentationPipeline:
    def __init__(
        self,
        cfg: Dict,
        img_w: int,
        img_h: int,
        class_swap_map: Optional[Dict[int, int]] = None,
        dataset=None,
    ) -> None:
        self.cfg = cfg or {}
        self.img_w = img_w
        self.img_h = img_h
        self.class_swap_map = class_swap_map or {}
        self.pd_transform = None  # lazily built per-call when encountered
        self.dataset = dataset

    def __call__(self, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # img: HWC, float32 [0,1]; boxes: N x 4 normalized cxcywh; labels: N
        if img.size == 0 or boxes.size == 0:
            # still allow color-only augs
            pass
        # Sequentially apply ops in YAML order (dict preserves order in Python 3.7+)
        for name, aug in self.cfg.items():
            if aug is None:
                continue
            img, boxes, labels = self._apply_op(name, aug, img, boxes, labels)

        return img, boxes, labels

    def _apply_op(self, name: str, aug_cfg, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        prob = None
        if isinstance(aug_cfg, dict) and "prob" in aug_cfg:
            prob = float(aug_cfg.get("prob", 0.0))
        elif isinstance(aug_cfg, (int, float)):
            prob = float(aug_cfg)

        def should_apply(default=0.0):
            p = prob if prob is not None else default
            return random.random() < p

        if name == "HorizontalFlip":
            if should_apply(0.0):
                img = np.ascontiguousarray(img[:, ::-1, :])
                if boxes.size:
                    boxes[:, 0] = 1.0 - boxes[:, 0]
                if self.class_swap_map and labels.size:
                    labels = np.array([self.class_swap_map.get(int(l), int(l)) for l in labels], dtype=labels.dtype)

        elif name == "VerticalFlip":
            if should_apply(0.0):
                img = np.ascontiguousarray(img[::-1, :, :])
                if boxes.size:
                    boxes[:, 1] = 1.0 - boxes[:, 1]

        elif name == "RandomScale":
            if should_apply(0.0):
                s_min, s_max = aug_cfg.get("scale_range", [1.0, 1.0])
                s = random.uniform(float(s_min), float(s_max))
                keep_aspect = bool(aug_cfg.get("keep_aspect_ratio", True))
                if keep_aspect and boxes.size:
                    boxes[:, 0] = boxes[:, 0] * s + 0.5 * (1 - s)
                    boxes[:, 1] = boxes[:, 1] * s + 0.5 * (1 - s)
                    boxes[:, 2] *= s
                    boxes[:, 3] *= s
                    boxes = _xyxy_to_cxcywh(_clip_boxes(_cxcywh_to_xyxy(boxes)))
                new_w = max(1, int(self.img_w * s))
                new_h = max(1, int(self.img_h * s if keep_aspect else self.img_h * s))
                img_scaled = cv2.resize((img * 255).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                canvas = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 114
                y0 = max(0, (self.img_h - new_h) // 2)
                x0 = max(0, (self.img_w - new_w) // 2)
                y1 = min(self.img_h, y0 + new_h)
                x1 = min(self.img_w, x0 + new_w)
                canvas[y0:y1, x0:x1, :] = img_scaled[: y1 - y0, : x1 - x0, :]
                img = canvas.astype(np.float32) / 255.0

        elif name == "Translation":
            if should_apply(0.0):
                translate = float(aug_cfg.get("translate", 0.0))
                dx = random.uniform(-translate, translate)
                dy = random.uniform(-translate, translate)
                fill = aug_cfg.get("fill", [114, 114, 114])
                fill = [int(v) for v in fill]
                shift_x = int(dx * self.img_w)
                shift_y = int(dy * self.img_h)
                canvas = np.ones_like(img) * np.array(fill, dtype=np.float32)[None, None, :] / 255.0
                x_from = max(0, -shift_x)
                y_from = max(0, -shift_y)
                x_to = min(self.img_w, self.img_w - shift_x)
                y_to = min(self.img_h, self.img_h - shift_y)
                canvas[y_from + shift_y : y_to + shift_y, x_from + shift_x : x_to + shift_x] = img[y_from:y_to, x_from:x_to]
                img = canvas
                if boxes.size:
                    xyxy = _cxcywh_to_xyxy(boxes)
                    xyxy[:, [0, 2]] += dx
                    xyxy[:, [1, 3]] += dy
                    xyxy = _clip_boxes(xyxy)
                    boxes = _xyxy_to_cxcywh(xyxy)
                    boxes, labels = _filter_boxes(boxes, labels, min_area=1e-6)

        elif name == "RemoveOutliers":
            thr = float(aug_cfg)
            boxes, labels = _filter_boxes(boxes, labels, min_area=thr)

        elif name == "RandomHSV":
            if should_apply(0.0):
                img = _apply_hsv(
                    img,
                    hue_gain=float(aug_cfg.get("hue_gain", 0.0)),
                    saturation_gain=float(aug_cfg.get("saturation_gain", 0.0)),
                    value_gain=float(aug_cfg.get("value_gain", 0.0)),
                )

        elif name == "RandomPhotometricDistort":
            if should_apply(0.5):
                if self.pd_transform is None:
                    p = prob if prob is not None else 0.5
                    self.pd_transform = T.RandomPhotometricDistort(p=1.0)
                    # we already sample prob manually
                img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
                img_t = self.pd_transform(img_t)
                img = img_t.permute(1, 2, 0).clamp(0.0, 1.0).numpy()

        elif name == "Blur":
            if should_apply(0.0):
                img = self._apply_cv2_blur(img, aug_cfg.get("blur_limit", [3, 7]))

        elif name == "MedianBlur":
            if should_apply(0.0):
                img = self._apply_cv2_median_blur(img, aug_cfg.get("blur_limit", [3, 7]))

        elif name == "CLAHE":
            if should_apply(0.0):
                img = self._apply_clahe(
                    img,
                    clip_limit=aug_cfg.get("clip_limit", 2.0),
                    tile=aug_cfg.get("tile_grid_size", [8, 8]),
                )

        elif name == "ToGray":
            if should_apply(0.0):
                img = self._apply_gray(img)

        elif name == "MixUp":
            if self.dataset is not None and should_apply(1.0):
                img2, boxes2, labels2 = self.dataset.sample_random()[:3]
                lam = 0.5
                img = (lam * img + (1 - lam) * img2).clip(0.0, 1.0)
                boxes = np.concatenate([boxes, boxes2], axis=0) if boxes.size and boxes2.size else (boxes2 if boxes.size == 0 else boxes)
                labels = np.concatenate([labels, labels2], axis=0) if labels.size and labels2.size else (labels2 if labels.size == 0 else labels)

        elif name == "Mosaic":
            if self.dataset is not None and should_apply(1.0):
                imgs = [img]
                boxes_list = [boxes]
                labels_list = [labels]
                for _ in range(3):
                    img_i, b_i, l_i = self.dataset.sample_random()[:3]
                    imgs.append(img_i)
                    boxes_list.append(b_i)
                    labels_list.append(l_i)
                canvas = np.ones((self.img_h * 2, self.img_w * 2, 3), dtype=np.float32) * 114 / 255.0
                positions = [(0, 0), (self.img_w, 0), (0, self.img_h), (self.img_w, self.img_h)]
                new_boxes = []
                new_labels = []
                for im, bx, lb, (ox, oy) in zip(imgs, boxes_list, labels_list, positions):
                    h, w = im.shape[:2]
                    canvas[oy : oy + h, ox : ox + w] = im
                    if bx.size:
                        cx = bx[:, 0] * w + ox
                        cy = bx[:, 1] * h + oy
                        bw = bx[:, 2] * w
                        bh = bx[:, 3] * h
                        new_boxes.append(np.stack([cx, cy, bw, bh], axis=1))
                        new_labels.append(lb)
                if new_boxes:
                    boxes = np.concatenate(new_boxes, axis=0)
                    labels = np.concatenate(new_labels, axis=0)
                else:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    labels = np.zeros((0,), dtype=np.int64)
                cx = random.uniform(self.img_w * 0.5, self.img_w * 1.5)
                cy = random.uniform(self.img_h * 0.5, self.img_h * 1.5)
                x0 = int(cx - self.img_w / 2)
                y0 = int(cy - self.img_h / 2)
                x1 = x0 + self.img_w
                y1 = y0 + self.img_h
                x0 = max(0, min(x0, 2 * self.img_w - self.img_w))
                y0 = max(0, min(y0, 2 * self.img_h - self.img_h))
                x1 = x0 + self.img_w
                y1 = y0 + self.img_h
                canvas = canvas[y0:y1, x0:x1]
                if boxes.size:
                    boxes[:, 0] -= x0
                    boxes[:, 1] -= y0
                    boxes[:, 0] /= self.img_w
                    boxes[:, 1] /= self.img_h
                    boxes[:, 2] /= self.img_w
                    boxes[:, 3] /= self.img_h
                    boxes, labels = _filter_boxes(boxes, labels, min_area=1e-6)
                img = canvas

        elif name == "CopyPaste":
            if self.dataset is not None and should_apply(aug_cfg.get("prob", 0.3)):
                sample_num = int(aug_cfg.get("sample_num", 1))
                max_objs = int(aug_cfg.get("max_paste_objects", 5))
                scale_jitter = aug_cfg.get("scale_jitter", [0.8, 1.2])
                for _ in range(sample_num):
                    img2, boxes2, labels2 = self.dataset.sample_random()[:3]
                    if boxes2.size == 0:
                        continue
                    num_paste = min(max_objs, boxes2.shape[0])
                    sel = np.random.choice(boxes2.shape[0], num_paste, replace=False)
                    for idx in sel:
                        cx, cy, bw, bh = boxes2[idx]
                        x1 = int((cx - bw / 2) * self.img_w)
                        y1 = int((cy - bh / 2) * self.img_h)
                        x2 = int((cx + bw / 2) * self.img_w)
                        y2 = int((cy + bh / 2) * self.img_h)
                        patch = img2[max(0, y1) : min(self.img_h, y2), max(0, x1) : min(self.img_w, x2)]
                        if patch.size == 0:
                            continue
                        scale = random.uniform(float(scale_jitter[0]), float(scale_jitter[1]))
                        new_w = max(1, int(patch.shape[1] * scale))
                        new_h = max(1, int(patch.shape[0] * scale))
                        patch_resized = cv2.resize((patch * 255).astype(np.uint8), (new_w, new_h))
                        patch_resized = patch_resized.astype(np.float32) / 255.0
                        px = random.randint(0, max(0, self.img_w - new_w))
                        py = random.randint(0, max(0, self.img_h - new_h))
                        img[py : py + new_h, px : px + new_w] = patch_resized
                        ncx = (px + new_w / 2) / self.img_w
                        ncy = (py + new_h / 2) / self.img_h
                        nbw = new_w / self.img_w
                        nbh = new_h / self.img_h
                        boxes = np.concatenate([boxes, np.array([[ncx, ncy, nbw, nbh]], dtype=np.float32)], axis=0)
                        labels = np.concatenate([labels, np.array([labels2[idx]], dtype=labels.dtype)], axis=0)

        elif name == "RandomCrop":
            if should_apply(0.0):
                crop_ratio = float(aug_cfg) if not isinstance(aug_cfg, dict) else float(aug_cfg.get("ratio", 0.9))
                ch = int(self.img_h * crop_ratio)
                cw = int(self.img_w * crop_ratio)
                if ch < 1 or cw < 1:
                    return img, boxes, labels
                y0 = random.randint(0, self.img_h - ch)
                x0 = random.randint(0, self.img_w - cw)
                img = img[y0 : y0 + ch, x0 : x0 + cw]
                if boxes.size:
                    boxes[:, 0] = (boxes[:, 0] * self.img_w - x0) / cw
                    boxes[:, 1] = (boxes[:, 1] * self.img_h - y0) / ch
                    boxes[:, 2] *= self.img_w / cw
                    boxes[:, 3] *= self.img_h / ch
                    boxes = _xyxy_to_cxcywh(_clip_boxes(_cxcywh_to_xyxy(boxes)))
                    boxes, labels = _filter_boxes(boxes, labels, min_area=1e-6)
                img = cv2.resize((img * 255).astype(np.uint8), (self.img_w, self.img_h)).astype(np.float32) / 255.0

        elif name == "RandomResizedCrop":
            if should_apply(aug_cfg.get("prob", 0.0)):
                scale = aug_cfg.get("scale", [0.6, 1.0])
                ratio = aug_cfg.get("ratio", [0.75, 1.33])
                min_vis = float(aug_cfg.get("min_visibility", 0.25))
                for _ in range(10):
                    target_area = self.img_w * self.img_h * random.uniform(scale[0], scale[1])
                    aspect = random.uniform(ratio[0], ratio[1])
                    cw = int(round((target_area * aspect) ** 0.5))
                    ch = int(round((target_area / aspect) ** 0.5))
                    if cw <= self.img_w and ch <= self.img_h:
                        x0 = random.randint(0, self.img_w - cw)
                        y0 = random.randint(0, self.img_h - ch)
                        img_crop = img[y0 : y0 + ch, x0 : x0 + cw]
                        if boxes.size:
                            boxes_xyxy = _cxcywh_to_xyxy(boxes)
                            inter_x1 = np.maximum(boxes_xyxy[:, 0] * self.img_w, x0)
                            inter_y1 = np.maximum(boxes_xyxy[:, 1] * self.img_h, y0)
                            inter_x2 = np.minimum(boxes_xyxy[:, 2] * self.img_w, x0 + cw)
                            inter_y2 = np.minimum(boxes_xyxy[:, 3] * self.img_h, y0 + ch)
                            inter_w = np.maximum(0.0, inter_x2 - inter_x1)
                            inter_h = np.maximum(0.0, inter_y2 - inter_y1)
                            inter_area = inter_w * inter_h
                            box_area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) * self.img_w * self.img_h
                            vis = inter_area / (box_area + 1e-6)
                            keep = vis >= min_vis
                            if keep.sum() == 0:
                                continue
                            boxes_xyxy = np.stack(
                                [
                                    (inter_x1[keep] - x0) / cw,
                                    (inter_y1[keep] - y0) / ch,
                                    (inter_x2[keep] - x0) / cw,
                                    (inter_y2[keep] - y0) / ch,
                                ],
                                axis=1,
                            )
                            boxes = _xyxy_to_cxcywh(boxes_xyxy)
                            labels = labels[keep]
                        img = cv2.resize((img_crop * 255).astype(np.uint8), (self.img_w, self.img_h)).astype(np.float32) / 255.0
                        break

        elif name == "RandomBrightness":
            if should_apply(aug_cfg.get("prob", 0.0)):
                fr = aug_cfg.get("factor_range", [0.8, 1.2])
                factor = random.uniform(float(fr[0]), float(fr[1]))
                img = np.clip(img * factor, 0.0, 1.0)

        elif name == "RandomContrast":
            if should_apply(aug_cfg.get("prob", 0.0)):
                fr = aug_cfg.get("factor_range", [0.8, 1.2])
                factor = random.uniform(float(fr[0]), float(fr[1]))
                mean = img.mean(axis=(0, 1), keepdims=True)
                img = np.clip((img - mean) * factor + mean, 0.0, 1.0)

        elif name == "RandomSaturation":
            if should_apply(aug_cfg.get("prob", 0.0)):
                fr = aug_cfg.get("factor_range", [0.8, 1.2])
                factor = random.uniform(float(fr[0]), float(fr[1]))
                gray = img.mean(axis=2, keepdims=True)
                img = np.clip((img - gray) * factor + gray, 0.0, 1.0)

        elif name == "MotionBlur":
            if should_apply(aug_cfg.get("prob", 0.0)):
                bl = aug_cfg.get("blur_limit", [5, 15])
                k = random.randrange(int(bl[0]), int(bl[1]) + 1)
                k = max(3, k | 1)
                angle = random.uniform(0, 180)
                kernel = np.zeros((k, k), dtype=np.float32)
                xs = np.cos(np.deg2rad(angle))
                ys = np.sin(np.deg2rad(angle))
                center = k // 2
                for i in range(k):
                    x = center + int((i - center) * xs)
                    y = center + int((i - center) * ys)
                    if 0 <= x < k and 0 <= y < k:
                        kernel[y, x] = 1
                kernel /= kernel.sum() if kernel.sum() != 0 else 1
                img = cv2.filter2D((img * 255).astype(np.uint8), -1, kernel).astype(np.float32) / 255.0

        elif name == "GaussianBlur":
            if should_apply(aug_cfg.get("prob", 0.0)):
                bl = aug_cfg.get("blur_limit", [3, 7])
                sl = aug_cfg.get("sigma_limit", [0.1, 2.0])
                k = random.randrange(int(bl[0]), int(bl[1]) + 1)
                k = max(1, k | 1)
                sigma = random.uniform(float(sl[0]), float(sl[1]))
                img = cv2.GaussianBlur((img * 255).astype(np.uint8), (k, k), sigma).astype(np.float32) / 255.0

        elif name == "GaussNoise":
            if should_apply(aug_cfg.get("prob", 0.0)):
                mean = float(aug_cfg.get("mean", 0.0))
                var = aug_cfg.get("var_limit", [10.0, 50.0])
                sigma = random.uniform(float(var[0]), float(var[1]))
                noise = np.random.normal(mean, sigma, img.shape) / 255.0
                img = np.clip(img + noise, 0.0, 1.0)

        elif name == "ImageCompression":
            if should_apply(aug_cfg.get("prob", 0.0)):
                quality_range = aug_cfg.get("quality_range", [40, 90])
                q = int(random.uniform(float(quality_range[0]), float(quality_range[1])))
                _, enc = cv2.imencode(".jpg", (img * 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), q])
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        elif name == "ISONoise":
            if should_apply(aug_cfg.get("prob", 0.0)):
                intensity = aug_cfg.get("intensity", [0.05, 0.15])
                color_shift = aug_cfg.get("color_shift", [0.01, 0.05])
                std = random.uniform(float(intensity[0]), float(intensity[1]))
                shift = random.uniform(float(color_shift[0]), float(color_shift[1]))
                noise = np.random.normal(0.0, std, img.shape)
                color = np.random.normal(0.0, shift, (1, 1, 3))
                img = np.clip(img + noise + color, 0.0, 1.0)

        elif name == "RandomRain":
            if should_apply(aug_cfg.get("prob", 0.0)):
                density = aug_cfg.get("density", [0.002, 0.006])
                drops = int(self.img_w * self.img_h * random.uniform(float(density[0]), float(density[1])))
                rain = img.copy()
                for _ in range(drops):
                    x = random.randint(0, self.img_w - 1)
                    y = random.randint(0, self.img_h - 1)
                    length = random.randint(*aug_cfg.get("drop_length", [15, 30]))
                    thickness = random.randint(*aug_cfg.get("drop_width_range", [1, 2]))
                    rain = cv2.line(
                        rain,
                        (x, y),
                        (x + random.randint(*aug_cfg.get("slant_range", [-10, 10])), y + length),
                        color=(200 / 255, 200 / 255, 200 / 255),
                        thickness=thickness,
                    )
                img = np.clip(rain, 0.0, 1.0)

        elif name == "RandomFog":
            if should_apply(aug_cfg.get("prob", 0.0)):
                fog_coef = aug_cfg.get("fog_coef", [0.3, 0.6])
                alpha = random.uniform(float(fog_coef[0]), float(fog_coef[1]))
                haze = np.ones_like(img) * random.uniform(0.7, 1.0)
                img = np.clip(img * (1 - alpha) + haze * alpha, 0.0, 1.0)

        elif name == "RandomSunFlare":
            if should_apply(aug_cfg.get("prob", 0.0)):
                src_radius = aug_cfg.get("src_radius_range", [50, 150])
                radius = random.randint(int(src_radius[0]), int(src_radius[1]))
                intensity = aug_cfg.get("src_intensity", [0.6, 1.0])
                flare = np.zeros_like(img)
                center = (random.randint(0, self.img_w - 1), random.randint(0, self.img_h - 1))
                cv2.circle(flare, center, radius, (1, 1, 1), -1)
                img = np.clip(img + flare * random.uniform(float(intensity[0]), float(intensity[1])), 0.0, 1.0)

        # unknown ops are ignored
        return img, boxes, labels

    def _apply_cv2_blur(self, img: np.ndarray, blur_limit):
        k1, k2 = blur_limit if isinstance(blur_limit, (list, tuple)) else (blur_limit, blur_limit)
        k = random.randrange(int(k1), int(k2) + 1)
        k = max(1, k | 1)
        blurred = cv2.blur((img * 255).astype(np.uint8), (k, k))
        return blurred.astype(np.float32) / 255.0

    def _apply_cv2_median_blur(self, img: np.ndarray, blur_limit):
        k1, k2 = blur_limit if isinstance(blur_limit, (list, tuple)) else (blur_limit, blur_limit)
        k = random.randrange(int(k1), int(k2) + 1)
        k = max(1, k | 1)
        blurred = cv2.medianBlur((img * 255).astype(np.uint8), k)
        return blurred.astype(np.float32) / 255.0

    def _apply_clahe(self, img: np.ndarray, clip_limit: float, tile):
        tile = tile if isinstance(tile, (list, tuple)) else [8, 8]
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile[0]), int(tile[1])))
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    def _apply_gray(self, img: np.ndarray):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray3 = np.stack([gray, gray, gray], axis=-1)
        return gray3.astype(np.float32) / 255.0


def build_augmentation_pipeline(cfg: Dict, img_w: int, img_h: int, class_swap_map: Optional[Dict[int, int]] = None, dataset=None):
    if not cfg:
        return None
    # only include enabled entries (non-None)
    enabled = {}
    for k, v in cfg.items():
        if v is None:
            continue
        enabled[k] = v
    if not enabled:
        return None
    return AugmentationPipeline(enabled, img_w=img_w, img_h=img_h, class_swap_map=class_swap_map, dataset=dataset)
