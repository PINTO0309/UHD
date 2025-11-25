import glob
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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

        items = self._gather_items()
        if not items:
            raise ValueError("No samples found. Check image_dir/list_path and labels.")

        # Deterministic split
        rng = random.Random(seed)
        rng.shuffle(items)
        if val_split <= 0 or split == "all":
            self.items = items
        else:
            split_idx = int(len(items) * (1.0 - val_split))
            if split == "train":
                self.items = items[:split_idx]
            elif split == "val":
                self.items = items[split_idx:]
            else:
                self.items = items

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

    def __getitem__(self, idx: int):
        img_path, label_path = self.items[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if self.augment:
                if random.random() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    flipped = True
                else:
                    flipped = False
            else:
                flipped = False
            im = im.resize((self.img_w, self.img_h), Image.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)

        raw_boxes = _read_label(label_path, self.class_to_idx)
        boxes: List[Sequence[float]] = []
        labels: List[int] = []
        for cid, cx, cy, w, h in raw_boxes:
            if flipped:
                cx = 1.0 - cx
            boxes.append([cx, cy, w, h])
            labels.append(cid)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": img_path,
        }
        return img, target


def detection_collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets
