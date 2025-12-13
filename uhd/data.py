import glob
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import build_augmentation_pipeline


def _resolve_path(entry: str, image_dir: str, list_path: Optional[str]) -> Optional[str]:
    """Try multiple locations to find an image path."""
    candidates = []
    if os.path.isabs(entry):
        candidates.append(entry)
    else:
        if list_path:
            candidates.append(os.path.join(os.path.dirname(list_path), entry))
        candidates.append(os.path.join(image_dir, entry))
        candidates.append(os.path.join(image_dir, os.path.basename(entry)))
        candidates.append(entry)

    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def _read_label(path: str, class_to_idx: Dict[int, int]) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid_raw = int(float(parts[0]))
            if cid_raw not in class_to_idx:
                continue
            cid = class_to_idx[cid_raw]
            x, y, w, h = map(float, parts[1:5])
            boxes.append((cid, x, y, w, h))
    return boxes


class YoloDataset(Dataset):
    """Dataset for YOLO-format labels, filtering by a single class."""

    def __init__(
        self,
        image_dir: str,
        list_path: Optional[str] = None,
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
        img_size: Tuple[int, int] = (64, 64),
        augment: bool = False,
        class_ids: Sequence[int] = (0,),
        augment_cfg: Optional[Dict] = None,
        items: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.image_dir = image_dir
        self.list_path = list_path
        self.split = split
        self.val_split = val_split
        self.seed = seed
        if isinstance(img_size, tuple):
            self.img_h, self.img_w = img_size
        else:
            self.img_h = self.img_w = int(img_size)
        self.augment = augment
        self.class_ids = list(class_ids)
        if not self.class_ids:
            raise ValueError("class_ids must contain at least one class id.")
        self.class_to_idx: Dict[int, int] = {cid: i for i, cid in enumerate(self.class_ids)}
        self.augment_cfg = augment_cfg
        self.pipeline = None
        if self.augment and augment_cfg:
            # convert class_swap_map to internal indices if present
            class_swap_map = None
            hf_cfg = augment_cfg.get("HorizontalFlip") if isinstance(augment_cfg, dict) else None
            if hf_cfg and isinstance(hf_cfg, dict) and "class_swap_map" in hf_cfg:
                class_swap_map = {}
                for k, v in hf_cfg["class_swap_map"].items():
                    if int(k) in self.class_to_idx and int(v) in self.class_to_idx:
                        class_swap_map[self.class_to_idx[int(k)]] = self.class_to_idx[int(v)]
            cfg_body = augment_cfg.get("data_augment", augment_cfg) if isinstance(augment_cfg, dict) else None
            self.pipeline = build_augmentation_pipeline(
                cfg_body, img_w=self.img_w, img_h=self.img_h, class_swap_map=class_swap_map, dataset=self
            )

        if items is not None:
            items_list = items
        else:
            items_list = self._gather_items()
        if not items_list:
            raise ValueError("No samples found. Check image_dir/list_path and labels.")

        # Deterministic split
        rng = random.Random(seed)
        rng.shuffle(items_list)
        if val_split <= 0 or split == "all":
            self.items = items_list
        else:
            split_idx = int(len(items_list) * (1.0 - val_split))
            if split == "train":
                self.items = items_list[:split_idx]
            elif split == "val":
                self.items = items_list[split_idx:]
            else:
                self.items = items_list

    def _gather_items(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if self.list_path:
            with open(self.list_path, "r") as f:
                entries = [ln.strip() for ln in f if ln.strip()]
            for ent in entries:
                img_path = _resolve_path(ent, self.image_dir, self.list_path)
                if not img_path:
                    continue
                label_path = os.path.splitext(img_path)[0] + ".txt"
                pairs.append((img_path, label_path))
        else:
            patterns = ["*.jpg", "*.jpeg", "*.png"]
            for pat in patterns:
                for img_path in glob.glob(os.path.join(self.image_dir, pat)):
                    label_path = os.path.splitext(img_path)[0] + ".txt"
                    pairs.append((img_path, label_path))

        filtered: List[Tuple[str, str]] = []
        for img_path, label_path in pairs:
            boxes = _read_label(label_path, self.class_to_idx)
            if boxes:
                filtered.append((img_path, label_path))
        return filtered

    def __len__(self) -> int:
        return len(self.items)

    def _load_raw(self, idx: int):
        img_path, label_path = self.items[idx]
        im_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        arr = im_rgb.astype(np.float32) / 255.0
        raw_boxes = _read_label(label_path, self.class_to_idx)
        boxes: List[Sequence[float]] = []
        labels: List[int] = []
        for cid, cx, cy, w, h in raw_boxes:
            boxes.append([cx, cy, w, h])
            labels.append(cid)
        boxes_np = np.array(boxes, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int64)
        return arr, boxes_np, labels_np, img_path

    def sample_random(self):
        ridx = random.randrange(len(self.items))
        return self._load_raw(ridx)

    def __getitem__(self, idx: int):
        max_retry = 10
        last_sample = None
        for attempt in range(max_retry):
            cur_idx = idx if attempt == 0 else random.randrange(len(self.items))
            arr, boxes_np, labels_np, img_path = self._load_raw(cur_idx)
            h0, w0 = arr.shape[:2]
            arr = cv2.resize(arr, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            if self.pipeline:
                img_np, boxes_np, labels_np = self.pipeline(arr, boxes_np, labels_np)
            else:
                img_np = arr

            if img_np.shape[0] != self.img_h or img_np.shape[1] != self.img_w:
                img_np = cv2.resize(img_np, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            last_sample = (img_np, boxes_np, labels_np, img_path, (h0, w0))
            if boxes_np.size > 0:
                target = {
                    "boxes": torch.tensor(boxes_np, dtype=torch.float32),
                    "labels": torch.tensor(labels_np, dtype=torch.long),
                    "image_id": img_path,
                    "orig_size": (h0, w0),
                }
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                return img_tensor, target

        # fallback: return last attempt even if empty to avoid infinite loop
        if last_sample is None:
            raise ValueError("Dataset is empty.")
        img_np, boxes_np, labels_np, img_path, orig_size = last_sample
        target = {
            "boxes": torch.tensor(boxes_np, dtype=torch.float32),
            "labels": torch.tensor(labels_np, dtype=torch.long),
            "image_id": img_path,
            "orig_size": orig_size,
        }
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor, target


def detection_collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets
