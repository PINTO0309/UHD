import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def move_targets(targets, device):
    moved = []
    for t in targets:
        moved.append(
            {
                "boxes": t["boxes"].to(device),
                "labels": t.get("labels", torch.zeros(len(t["boxes"]), dtype=torch.long)).to(device),
                "image_id": t["image_id"],
                "orig_size": t.get("orig_size"),
            }
        )
    return moved
