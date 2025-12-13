import argparse
import glob
import json
import os
import random
import math
from copy import deepcopy
from typing import Dict, Sequence, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from uhd.data import YoloDataset, detection_collate
from uhd.losses import anchor_loss, centernet_loss, detr_loss
from uhd.metrics import decode_anchor, decode_centernet, decode_detr, evaluate_map
from uhd.backbones import load_dinov3_backbone
from uhd.models import build_model
from uhd.utils import default_device, ensure_dir, move_targets, set_seed
from uhd.resize import normalize_resize_mode


class ModelEma:
    """Exponential Moving Average of model parameters/buffers."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9998, device: torch.device = None) -> None:
        self.decay = decay
        self.device = device
        self.ema = deepcopy(model)
        self.ema.eval()
        self.updates = 0
        if self.device is not None:
            self.ema.to(self.device)

    def _get_decay(self) -> float:
        # Warmup EMA decay to let early steps catch up; similar to timm/yolo practice.
        self.updates += 1
        warmup = (1 + self.updates) / (10 + self.updates)
        return min(self.decay, warmup)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self._get_decay()
        ema_params = dict(self.ema.named_parameters())
        model_params = dict(model.named_parameters())
        for k, ema_v in ema_params.items():
            model_v = model_params[k].detach()
            if self.device is not None:
                model_v = model_v.to(self.device)
            ema_v.mul_(d).add_(model_v, alpha=1.0 - d)

        ema_buffers = dict(self.ema.named_buffers())
        model_buffers = dict(model.named_buffers())
        for k, ema_b in ema_buffers.items():
            model_b = model_buffers[k]
            if model_b.dtype.is_floating_point:
                if self.device is not None:
                    model_b = model_b.to(self.device)
                ema_b.mul_(d).add_(model_b, alpha=1.0 - d)
            else:
                ema_b.copy_(model_b)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.ema.state_dict())


def parse_args():
    parser = argparse.ArgumentParser(description="Ultra-lightweight detection trainer (CNN/Transformer).")
    parser.add_argument("--arch", choices=["cnn", "transformer", "ultratinyod"], default="cnn")
    parser.add_argument("--image-dir", default="data/wholebody34/obj_train_data", help="Directory with images and YOLO txt labels.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data for training.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation.")
    parser.add_argument(
        "--img-size",
        default="64x64",
        help="Input size as HxW, e.g., 64x64. If single int, applies to both sides.",
    )
    parser.set_defaults(resize_mode="torch_bilinear")
    resize_group = parser.add_mutually_exclusive_group()
    resize_group.add_argument(
        "--resize-mode",
        choices=["torch_bilinear", "torch_nearest", "opencv_inter_linear", "opencv_inter_nearest"],
        dest="resize_mode",
        help="Resize mode used during training preprocessing.",
    )
    resize_group.add_argument(
        "--torch_bilinear",
        dest="resize_mode",
        action="store_const",
        const="torch_bilinear",
        help="Shortcut for --resize-mode torch_bilinear.",
    )
    resize_group.add_argument(
        "--torch_nearest",
        dest="resize_mode",
        action="store_const",
        const="torch_nearest",
        help="Shortcut for --resize-mode torch_nearest.",
    )
    resize_group.add_argument(
        "--opencv_inter_linear",
        dest="resize_mode",
        action="store_const",
        const="opencv_inter_linear",
        help="Shortcut for --resize-mode opencv_inter_linear.",
    )
    resize_group.add_argument(
        "--opencv_inter_nearest",
        dest="resize_mode",
        action="store_const",
        const="opencv_inter_nearest",
        help="Shortcut for --resize-mode opencv_inter_nearest.",
    )
    parser.add_argument("--exp-name", default="default", help="Experiment name; logs will be saved under runs/<exp-name>.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint to initialize weights (no optimizer state).")
    parser.add_argument("--ckpt-non-strict", action="store_true", help="Load --ckpt weights with strict=False (ignore missing/unexpected keys).")
    parser.add_argument("--teacher-ckpt", default=None, help="Path to teacher checkpoint for distillation.")
    parser.add_argument("--teacher-arch", default=None, help="Teacher architecture override (default: checkpoint arch or student arch).")
    parser.add_argument("--teacher-num-queries", type=int, default=None, help="Teacher num-queries (defaults to student).")
    parser.add_argument("--teacher-d-model", type=int, default=None, help="Teacher d-model (defaults to student).")
    parser.add_argument("--teacher-heads", type=int, default=None, help="Teacher heads (defaults to student).")
    parser.add_argument("--teacher-layers", type=int, default=None, help="Teacher layers (defaults to student).")
    parser.add_argument("--teacher-dim-feedforward", type=int, default=None, help="Teacher FFN dim (defaults to student).")
    parser.add_argument("--teacher-use-skip", action="store_true", help="Force teacher use-skip on (otherwise checkpoint/student default).")
    parser.add_argument("--teacher-activation", choices=["relu", "swish"], default=None, help="Teacher activation (defaults to checkpoint or student).")
    parser.add_argument("--teacher-use-fpn", action="store_true", help="Force teacher use-fpn on (otherwise checkpoint/student default).")
    parser.add_argument("--teacher-backbone", default=None, help="Path to teacher backbone checkpoint for feature distillation (e.g., DINOv3).")
    parser.add_argument("--teacher-backbone-arch", default=None, help="Teacher backbone architecture hint (e.g., dinov3_vits16, dinov3_vitb16).")
    parser.add_argument("--teacher-backbone-norm", default="imagenet", choices=["imagenet", "none"], help="Normalization applied to teacher backbone input.")
    parser.add_argument("--distill-kl", type=float, default=0.0, help="Weight for KL distillation loss (transformer).")
    parser.add_argument("--distill-box-l1", type=float, default=0.0, help="Weight for box L1 distillation (transformer).")
    parser.add_argument("--distill-cosine", action="store_true", help="Use cosine ramp-up of distill weights over epochs.")
    parser.add_argument("--distill-temperature", type=float, default=1.0, help="Temperature for teacher logits in distillation.")
    parser.add_argument("--distill-feat", type=float, default=0.0, help="Weight for feature-map distillation from teacher backbone (CNN only).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="Optimizer to use (UltraTinyOD can benefit from SGD).")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="Global gradient norm clip value (0 to disable).")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda or cpu. Defaults to cuda if available.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=50, help="Top-K for CNN decoding.")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision training.")
    parser.add_argument("--aug-config", default="uhd/aug.yaml", help="Path to YAML file specifying data augmentations.")
    parser.add_argument("--use-ema", action="store_true", help="Enable EMA of model weights for evaluation/checkpointing.")
    parser.add_argument("--ema-decay", type=float, default=0.9998, help="EMA decay factor (ignored if EMA disabled).")
    parser.add_argument("--coco-eval", action="store_true", help="Run COCO-style evaluation (requires faster-coco-eval or pycocotools).")
    parser.add_argument("--coco-per-class", action="store_true", help="Log per-class COCO AP when COCO eval is enabled.")
    parser.add_argument("--val-only", action="store_true", help="Run validation only using --ckpt or --resume weights and exit.")
    parser.add_argument("--val-count", type=int, default=None, help="Limit number of validation images when using --val-only.")
    parser.add_argument(
        "--use-improved-head",
        action="store_true",
        help="Enable enhanced UltraTinyOD head (quality-aware obj, learnable WH scale, extra context, IoU/quality scoring).",
    )
    parser.add_argument(
        "--use-iou-aware-head",
        action="store_true",
        help="UltraTinyOD head: task-aligned IoU-aware scoring (quality*cls) with split towers; keeps legacy path when off.",
    )
    parser.add_argument(
        "--classes",
        default="0",
        help="Comma-separated list of target class ids to train on (e.g., '0,1,3').",
    )
    parser.add_argument("--activation", choices=["relu", "swish"], default="swish", help="Activation function to use.")
    # CNN params
    parser.add_argument("--cnn-width", type=int, default=32)
    parser.add_argument("--use-skip", action="store_true", help="Enable skip connections in the CNN model.")
    parser.add_argument("--utod-residual", action="store_true", help="Enable residual skips inside the UltraTinyOD backbone.")
    parser.add_argument("--utod-head-ese", action="store_true", help="UltraTinyOD head: apply lightweight eSE on shared features.")
    parser.add_argument(
        "--utod-context-rfb",
        action="store_true",
        help="UltraTinyOD head: add a receptive-field block (dilated + wide depthwise) before prediction layers.",
    )
    parser.add_argument(
        "--utod-context-dilation",
        type=int,
        default=2,
        help="Dilation used in UltraTinyOD receptive-field block (only when --utod-context-rfb).",
    )
    parser.add_argument(
        "--utod-large-obj-branch",
        action="store_true",
        help="UltraTinyOD head: add a downsampled large-object refinement branch (no FPN).",
    )
    parser.add_argument(
        "--utod-large-obj-depth",
        type=int,
        default=2,
        help="Number of depthwise blocks in the large-object branch (only when --utod-large-obj-branch).",
    )
    parser.add_argument(
        "--utod-large-obj-ch-scale",
        type=float,
        default=1.0,
        help="Channel scale for the large-object branch (relative to head channels).",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        choices=["microcspnet", "ultratinyresnet", "enhanced-shufflenet", "none", None],
        help="Optional lightweight CNN backbone. Default: None (built-in tiny CNN).",
    )
    parser.add_argument("--backbone-channels", default=None, help="Comma-separated channels for ultratinyresnet (e.g., '16,24,32,48').")
    parser.add_argument("--backbone-blocks", default=None, help="Comma-separated residual block counts per stage for ultratinyresnet (e.g., '1,1,2,1').")
    parser.add_argument("--backbone-se", choices=["none", "se", "ese"], default="none", help="Apply SE/eSE on backbone output (custom backbones only).")
    parser.add_argument("--backbone-skip", action="store_true", help="Add long skip fusion across custom backbone stages (ultratinyresnet).")
    parser.add_argument("--backbone-skip-cat", action="store_true", help="Use concat+1x1 fusion for long skips (ultratinyresnet); implies --backbone-skip.")
    parser.add_argument(
        "--backbone-skip-shuffle-cat",
        action="store_true",
        help="Use stride+shuffle concat fusion for long skips (ultratinyresnet); implies --backbone-skip.",
    )
    parser.add_argument(
        "--backbone-skip-s2d-cat",
        action="store_true",
        help="Use space-to-depth concat fusion for long skips (ultratinyresnet); implies --backbone-skip.",
    )
    parser.add_argument("--backbone-fpn", action="store_true", help="Enable a tiny FPN fusion inside custom backbones (ultratinyresnet).")
    parser.add_argument("--backbone-out-stride", type=int, default=None, help="Override custom backbone output stride (e.g., 8 or 16).")
    parser.add_argument("--use-anchor", action="store_true", help="Use anchor-based head for CNN (YOLO-style).")
    parser.add_argument(
        "--output-stride",
        type=int,
        default=None,
        help="Final feature stride for CNN (4, 8, or 16). Defaults: CNN/Transformer=16, UltraTinyOD=8.",
    )
    parser.add_argument(
        "--anchors",
        default="",
        help='Anchor sizes as normalized "w,h w,h ..." (e.g., "0.08,0.10 0.15,0.20 0.30,0.35").',
    )
    parser.add_argument("--auto-anchors", action="store_true", help="Compute anchors from training labels when using anchor head.")
    parser.add_argument("--num-anchors", type=int, default=3, help="Number of anchors to use when auto-computing.")
    parser.add_argument("--iou-loss", choices=["iou", "giou", "ciou"], default="giou", help="IoU loss type for anchor head.")
    parser.add_argument("--anchor-assigner", choices=["legacy", "simota"], default="legacy", help="Anchor assigner strategy.")
    parser.add_argument("--anchor-cls-loss", choices=["bce", "vfl"], default="bce", help="Classification loss for anchor head.")
    parser.add_argument("--simota-topk", type=int, default=10, help="Top-K IoUs for dynamic-k in SimOTA.")
    parser.add_argument("--last-se", choices=["none", "se", "ese"], default="none", help="Apply SE/eSE only on the last CNN block.")
    parser.add_argument("--use-batchnorm", action="store_true", help="Enable BatchNorm layers (default: off).")
    parser.add_argument("--last-width-scale", type=float, default=1.0, help="Channel scale for last CNN block (e.g., 1.25).")
    parser.add_argument("--quality-power", type=float, default=1.0, help="Exponent for quality score when using IoU-aware head scoring.")
    # Transformer params
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--use-fpn", action="store_true", help="Enable simple FPN for transformer backbone.")
    return parser.parse_args()


def parse_img_size(arg: str):
    if isinstance(arg, (tuple, list)):
        if len(arg) != 2:
            raise ValueError("img-size tuple/list must have length 2 (H, W).")
        return int(arg[0]), int(arg[1])
    s = str(arg).lower().replace(" ", "")
    if "x" in s:
        parts = s.split("x")
        if len(parts) != 2:
            raise ValueError("img-size must be HxW, e.g., 64x64.")
        return int(parts[0]), int(parts[1])
    val = int(float(s))
    return val, val


def parse_classes(arg: str):
    if arg is None:
        return [0]
    if isinstance(arg, (list, tuple)):
        return [int(x) for x in arg]
    parts = str(arg).replace(" ", "").split(",")
    return [int(p) for p in parts if p != ""]


def parse_int_list(arg):
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        return [int(x) for x in arg]
    s = str(arg).replace(" ", "")
    if s == "":
        return None
    return [int(p) for p in s.split(",") if p != ""]


def parse_anchors_str(arg: str):
    anchors = []
    if not arg:
        return anchors
    for part in str(arg).split():
        nums = part.split(",")
        if len(nums) != 2:
            continue
        try:
            w = float(nums[0])
            h = float(nums[1])
        except ValueError:
            continue
        anchors.append((w, h))
    return anchors


def _wh_iou(boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """IoU using only w/h (no center), boxes: N x 2, anchors: K x 2."""
    inter = np.minimum(boxes[:, None, :], anchors[None, :, :]).prod(axis=2)
    union = (boxes[:, 0] * boxes[:, 1])[:, None] + (anchors[:, 0] * anchors[:, 1])[None, :] - inter + 1e-9
    return inter / union


def auto_compute_anchors(boxes: np.ndarray, k: int = 3, iters: int = 20) -> np.ndarray:
    """Simple k-means on box widths/heights (normalized) using IoU distance."""
    if boxes.size == 0:
        return np.zeros((k, 2), dtype=np.float32)
    n = boxes.shape[0]
    if n < k:
        # pad with duplicates if very few boxes
        boxes = np.concatenate([boxes, boxes[np.random.choice(n, k - n)]], axis=0)
        n = boxes.shape[0]
    # init centers by random choice
    rng = np.random.default_rng(0)
    centers = boxes[rng.choice(n, k, replace=False)]
    for _ in range(iters):
        ious = _wh_iou(boxes, centers)
        assignments = ious.argmax(axis=1)
        new_centers = []
        for ki in range(k):
            mask = assignments == ki
            if mask.sum() == 0:
                new_centers.append(centers[ki])
            else:
                new_centers.append(boxes[mask].mean(axis=0))
        new_centers = np.stack(new_centers, axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    centers = centers[np.argsort(centers.prod(axis=1))]  # sort by area
    return centers


def load_aug_config(path: str):
    if not path:
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


def log_scalars(writer: SummaryWriter, prefix: str, values: Dict[str, float], ordered_keys, step: int):
    """Log scalars with forced ordering by prefixing numeric indices."""
    logged = set()
    idx = 0
    for k in ordered_keys:
        if k in values:
            writer.add_scalar(f"{prefix}/{idx:02d}_{k}", values[k], step)
            logged.add(k)
            idx += 1
    for k in sorted(values.keys()):
        if k not in logged:
            writer.add_scalar(f"{prefix}/{idx:02d}_{k}", values[k], step)
            idx += 1


def _parse_best_filename(path: str):
    """Parse best checkpoint filename to extract (arch, epoch, map)."""
    base = os.path.splitext(os.path.basename(path))[0]  # without .pt
    if not base.startswith("best_") or "_map_" not in base:
        return None
    try:
        prefix, map_part = base.split("_map_", 1)  # e.g., best_cnn_0001 , 0.12345
        parts = prefix.split("_")
        if len(parts) < 3:
            return None
        arch = parts[1]
        epoch = int(parts[2])
        map_val = float(map_part)
        return arch, epoch, map_val
    except ValueError:
        return None


def _parse_last_filename(path: str):
    base = os.path.splitext(os.path.basename(path))[0]
    if not base.startswith("last_"):
        return None
    try:
        epoch = int(base.split("_")[1])
        return epoch
    except (IndexError, ValueError):
        return None


def _prune_best(run_dir: str, arch_tag: str, keep: int = 10):
    pattern = os.path.join(run_dir, f"best_{arch_tag}_*_map_*.pt")
    entries = []
    for p in glob.glob(pattern):
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            continue
        entries.append((mtime, p))
    entries.sort(key=lambda x: x[0], reverse=True)  # keep most recent
    for _, path in entries[keep:]:
        try:
            os.remove(path)
        except OSError:
            pass


def _prune_last(run_dir: str, keep: int = 10):
    pattern = os.path.join(run_dir, "last_*.pt")
    entries = []
    for p in glob.glob(pattern):
        epoch = _parse_last_filename(p)
        if epoch is not None:
            entries.append((epoch, p))
    entries.sort(key=lambda x: x[0], reverse=True)
    for _, path in entries[keep:]:
        try:
            os.remove(path)
        except OSError:
            pass


def _remove_dir(path: str):
    try:
        for root, _, files in os.walk(path, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            os.rmdir(root)
    except OSError:
        pass


def _prune_epoch_dirs(run_dir: str, keep: int = 10):
    dirs = []
    for name in os.listdir(run_dir):
        full = os.path.join(run_dir, name)
        if os.path.isdir(full) and name.isdigit():
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            dirs.append((mtime, full))
    dirs.sort(key=lambda x: x[0], reverse=True)  # newest first
    for _, path in dirs[keep:]:
        try:
            for root, _, files in os.walk(path, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                os.rmdir(root)
        except OSError:
            pass


def make_datasets(args, class_ids, aug_cfg, resize_mode: str):
    img_h, img_w = parse_img_size(args.img_size)
    base = YoloDataset(
        image_dir=args.image_dir,
        list_path=None,
        split="all",
        val_split=0.0,
        seed=args.seed,
        img_size=(img_h, img_w),
        resize_mode=resize_mode,
        augment=False,
        class_ids=class_ids,
        augment_cfg=aug_cfg,
    )
    items = list(base.items)
    if args.train_split + args.val_split > 1.0 + 1e-6:
        raise ValueError("train-split + val-split must be <= 1.0")
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * args.train_split)
    n_val = int(n * args.val_split)
    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    if not val_items:
        raise ValueError("Validation split produced no samples; adjust train-split/val-split.")
    # When running val-only, optionally cap validation set size for quick checks.
    if getattr(args, "val_only", False) and getattr(args, "val_count", None):
        val_cap = max(0, int(args.val_count))
        if val_cap > 0:
            val_items = val_items[:val_cap]
            if not val_items:
                raise ValueError("val-count reduced validation set to zero samples.")
            print(f"val-only: restricting validation set to {len(val_items)} samples (val-count={val_cap}).")

    train_ds = YoloDataset(
        image_dir=args.image_dir,
        list_path=None,
        split="all",
        val_split=0.0,
        seed=args.seed,
        img_size=(img_h, img_w),
        resize_mode=resize_mode,
        augment=True,
        class_ids=class_ids,
        augment_cfg=aug_cfg,
        items=train_items,
    )
    val_ds = YoloDataset(
        image_dir=args.image_dir,
        list_path=None,
        split="all",
        val_split=0.0,
        seed=args.seed,
        img_size=(img_h, img_w),
        resize_mode=resize_mode,
        augment=False,
        class_ids=class_ids,
        augment_cfg=aug_cfg,
        items=val_items,
    )
    return train_ds, val_ds


def collect_box_wh(dataset) -> np.ndarray:
    """Collect normalized (w,h) from YOLO labels for anchor calculation."""
    wh_list = []
    for _, label_path in dataset.items:
        if not os.path.exists(label_path):
            continue
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid_raw = int(float(parts[0]))
                except ValueError:
                    continue
                if cid_raw not in dataset.class_to_idx:
                    continue
                try:
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue
                wh_list.append((w, h))
    return np.array(wh_list, dtype=np.float32)


def _infer_cnn_feat_channels(model: torch.nn.Module, use_anchor: bool, num_classes: int) -> int:
    """Best-effort inference of CNN feature channels (pre-head)."""
    # UltraTinyOD / anchor-style heads expose in_channels or backbone.out_channels; prefer those.
    if use_anchor:
        head = getattr(model, "head", None)
        if head is not None:
            if hasattr(head, "in_channels"):
                try:
                    return int(head.in_channels)
                except Exception:
                    pass
            if hasattr(head, "backbone") and hasattr(head.backbone, "out_channels"):
                try:
                    return int(head.backbone.out_channels)
                except Exception:
                    pass
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "out_channels"):
        try:
            return int(backbone.out_channels)
        except Exception:
            pass
    if use_anchor and hasattr(model, "head"):
        head = getattr(model, "head", None)
        if isinstance(head, torch.nn.Conv2d):
            w = head.weight
            if isinstance(w, torch.Tensor) and w.dim() == 4:
                return int(w.shape[1])
    head_hm = getattr(model, "head_hm", None)
    if isinstance(head_hm, torch.nn.Conv2d):
        w = head_hm.weight
        if isinstance(w, torch.Tensor) and w.dim() == 4:
            return int(w.shape[1])
    stage3 = getattr(model, "stage3", None)
    if isinstance(stage3, torch.nn.Module) and hasattr(stage3, "pw"):
        pw = getattr(stage3, "pw", None)
        if isinstance(pw, torch.nn.Conv2d):
            return int(pw.out_channels)
    return max(num_classes, 1)


def run_coco_eval(images, annos, dets, class_ids, per_class=False):
    try:
        try:
            from faster_coco_eval.api import COCO, COCOeval  # type: ignore
        except ImportError:
            from pycocotools.coco import COCO  # type: ignore
            from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception as e:
        print(f"COCO eval skipped: {e}")
        return {}

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annos,
        "categories": [{"id": int(cid), "name": str(cid)} for cid in class_ids],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(dets) if dets else coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # mAP, mAP50, etc.
    out = {
        "coco_mAP": float(stats[0]) if stats is not None else 0.0,
        "coco_mAP50": float(stats[1]) if stats is not None else 0.0,
    }
    if per_class:
        precisions = coco_eval.eval["precision"]  # [TxRxKxAxM]
        if precisions is not None:
            T, R, K, A, M = precisions.shape
            for k in range(K):
                p = precisions[:, :, k, 0, -1]
                p = p[p > -1]
                out[f"coco_AP_class{class_ids[k]}"] = float(p.mean()) if p.size else 0.0
    return out


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    arch: str,
    log_interval: int,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    total_epochs: int,
    num_classes: int,
    grad_clip_norm: float = 0.0,
    ema: ModelEma = None,
    teacher_model: torch.nn.Module = None,
    teacher_backbone: torch.nn.Module = None,
    teacher_arch: str = None,
    teacher_use_anchor: bool = False,
    teacher_anchors: torch.Tensor = None,
    teacher_num_classes: int = None,
    feature_adapter: torch.nn.Module = None,
    teacher_backbone_norm: str = "imagenet",
    distill_kl: float = 0.0,
    distill_box_l1: float = 0.0,
    distill_cosine: bool = False,
    distill_temperature: float = 1.0,
    distill_feat: float = 0.0,
    use_anchor: bool = False,
    anchors: torch.Tensor = None,
    iou_loss_type: str = "giou",
    anchor_assigner: str = "legacy",
    anchor_cls_loss: str = "bce",
    simota_topk: int = 10,
) -> Dict[str, float]:
    model.train()
    arch_cnn_like = arch in ("cnn", "ultratinyod")
    anchor_head = bool(use_anchor or arch == "ultratinyod")
    total_loss = 0.0
    total_hm = 0.0
    total_off = 0.0
    total_wh = 0.0
    total_anchor_obj = 0.0
    total_anchor_cls = 0.0
    total_anchor_box = 0.0
    total_anchor_quality = 0.0
    total_cls = 0.0
    total_l1 = 0.0
    total_iou = 0.0
    total_distill_kl = 0.0
    total_distill_l1 = 0.0
    total_distill_feat = 0.0
    steps = 0
    distill_scale = 1.0
    if distill_cosine and (distill_kl > 0 or distill_box_l1 > 0 or distill_feat > 0):
        distill_scale = 0.5 * (1 - math.cos(math.pi * (epoch + 1) / total_epochs))
    use_quality_head = bool(getattr(model, "has_quality_head", getattr(model, "use_improved_head", False)))
    score_mode = getattr(model, "score_mode", None)
    quality_power = getattr(model, "quality_power", 1.0)
    wh_scale_tensor = None
    if hasattr(model, "head"):
        if hasattr(model.head, "wh_scale"):
            wh_scale_tensor = model.head.wh_scale
        if score_mode is None and hasattr(model.head, "score_mode"):
            score_mode = model.head.score_mode
        quality_power = getattr(model.head, "quality_power", quality_power)
    clip_params = list(model.parameters())
    if feature_adapter is not None:
        clip_params += [p for p in feature_adapter.parameters() if p.requires_grad]

    def _decode_anchor_pred(
        pred_raw: torch.Tensor,
        anchor_tensor: torch.Tensor,
        num_cls: int,
        has_quality: bool = False,
        wh_scale: Optional[torch.Tensor] = None,
    ):
        b, _, h, w = pred_raw.shape
        na = anchor_tensor.shape[0]
        extra = 1 if has_quality else 0
        pred = pred_raw.view(b, na, 5 + extra + num_cls, h, w)
        tx = pred[:, :, 0]
        ty = pred[:, :, 1]
        tw = pred[:, :, 2]
        th = pred[:, :, 3]
        cls_logit = pred[:, :, (5 + extra):]
        # grid
        gy, gx = torch.meshgrid(torch.arange(h, device=pred_raw.device), torch.arange(w, device=pred_raw.device), indexing="ij")
        gx = gx.view(1, 1, h, w)
        gy = gy.view(1, 1, h, w)
        cx = (tx.sigmoid() + gx) / float(w)
        cy = (ty.sigmoid() + gy) / float(h)
        anchor_use = anchor_tensor
        if wh_scale is not None:
            anchor_use = anchor_use * wh_scale.to(pred_raw.device)
        pw = anchor_use[:, 0].view(1, na, 1, 1)
        ph = anchor_use[:, 1].view(1, na, 1, 1)
        bw = pw * torch.clamp(torch.nn.functional.softplus(tw), max=4.0)
        bh = ph * torch.clamp(torch.nn.functional.softplus(th), max=4.0)
        boxes = torch.stack([cx, cy, bw, bh], dim=-1)  # B x A x H x W x 4
        cls_logits = cls_logit.permute(0, 1, 3, 4, 2)  # B x A x H x W x C
        return boxes, cls_logits, h, w

    def _interp_anchor_map(tensor_map: torch.Tensor, target_hw):
        # tensor_map: B x A x H x W x F
        b, a, h, w, f = tensor_map.shape
        tensor_map = tensor_map.reshape(b, a * f, h, w)
        tensor_map = torch.nn.functional.interpolate(tensor_map, size=target_hw, mode="bilinear", align_corners=False)
        tensor_map = tensor_map.view(b, a, target_hw[0], target_hw[1], f)
        return tensor_map
    pbar = tqdm(loader, total=len(loader), desc=f"Train {epoch+1}/{total_epochs}", ncols=120, dynamic_ncols=True)
    warn_anchor_mismatch = False
    warn_class_mismatch = False
    warn_nonfinite_kl = False
    warn_nonfinite_loss = False
    for step, (imgs, targets) in enumerate(pbar):
        imgs = imgs.to(device)
        targets_dev = move_targets(targets, device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16, enabled=scaler.is_enabled()):
            teacher_logits = None
            teacher_boxes = None
            teacher_feats = None
            teacher_anchor_pred = None
            teacher_ct_pred = None
            student_feats = None
            if teacher_model is not None:
                with torch.no_grad():
                    if (teacher_arch or "transformer") == "transformer":
                        t_out = teacher_model(imgs)
                        if isinstance(t_out, dict):
                            teacher_logits = None
                            teacher_boxes = None
                        else:
                            teacher_logits, teacher_boxes = t_out
                    else:
                        t_out = teacher_model(imgs, return_feat=True)
                        if isinstance(t_out, tuple) and len(t_out) == 2:
                            pred_part, feat_part = t_out
                            teacher_feats = feat_part
                            if isinstance(pred_part, dict):
                                teacher_ct_pred = pred_part
                            else:
                                teacher_anchor_pred = pred_part
                        elif isinstance(t_out, dict):
                            teacher_ct_pred = t_out
            if teacher_backbone is not None and distill_feat > 0 and arch_cnn_like:
                with torch.no_grad():
                    imgs_t = imgs
                    if teacher_backbone_norm == "imagenet":
                        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device, dtype=torch.float32).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device, dtype=torch.float32).view(1, 3, 1, 1)
                        imgs_t = imgs_t.float()
                        imgs_t = (imgs_t - mean) / std
                    teacher_feats = teacher_backbone(imgs_t)
            if arch_cnn_like:
                need_feats = teacher_feats is not None and distill_feat > 0
                outputs = model(imgs, return_feat=need_feats)
                if need_feats:
                    outputs, student_feats = outputs
                if anchor_head:
                    loss_dict = anchor_loss(
                        outputs,
                        targets_dev,
                        anchors=anchors,
                        num_classes=num_classes,
                        iou_loss=iou_loss_type,
                        assigner=anchor_assigner,
                        cls_loss_type=anchor_cls_loss,
                        simota_topk=simota_topk,
                        use_quality=use_quality_head,
                        wh_scale=wh_scale_tensor,
                    )
                else:
                    loss_dict = centernet_loss(outputs, targets_dev, num_classes=num_classes)
                if need_feats and student_feats is not None:
                    t_feat = teacher_feats
                    if t_feat is not None and t_feat.shape[2:] != student_feats.shape[2:]:
                        t_feat = torch.nn.functional.interpolate(
                            t_feat, size=student_feats.shape[2:], mode="bilinear", align_corners=False
                        )
                    if t_feat is not None:
                        with torch.amp.autocast(device_type=device.type, enabled=False):
                            s_in = student_feats.float()
                            s_proj = feature_adapter(s_in) if feature_adapter is not None else s_in
                            t_feat_f = t_feat.float()
                            s_norm = F.normalize(torch.nan_to_num(s_proj), dim=1, eps=1e-6)
                            t_norm = F.normalize(torch.nan_to_num(t_feat_f), dim=1, eps=1e-6)
                            loss_feat = F.l1_loss(torch.nan_to_num(s_norm), torch.nan_to_num(t_norm), reduction="mean")
                        loss_dict["loss"] = loss_dict["loss"] + (distill_feat * distill_scale) * loss_feat
                        loss_dict["distill_feat"] = loss_feat
                # CNN logit/box distillation from CNN teacher
                if teacher_model is not None and (teacher_arch or "cnn") in ("cnn", "ultratinyod"):
                    if anchor_head and teacher_use_anchor and teacher_anchor_pred is not None and anchors is not None and teacher_anchors is not None:
                        t_nc = teacher_num_classes if teacher_num_classes is not None else num_classes
                        if t_nc != num_classes:
                            if not warn_class_mismatch:
                                print(f"Teacher classes ({t_nc}) != student ({num_classes}); skipping logit/box distill for anchor head.")
                                warn_class_mismatch = True
                        else:
                            t_na = teacher_anchors.shape[0]
                            s_na = anchors.shape[0]
                            if t_na != s_na:
                                if not warn_anchor_mismatch:
                                    print(f"Teacher anchors ({t_na}) != student ({s_na}); skipping anchor logit/box distill.")
                                    warn_anchor_mismatch = True
                            else:
                                t_has_quality = bool(
                                    getattr(teacher_model, "has_quality_head", getattr(teacher_model, "use_improved_head", False))
                                )
                                s_has_quality = use_quality_head
                                t_boxes, t_cls_logits, th, tw = _decode_anchor_pred(
                                    teacher_anchor_pred,
                                    teacher_anchors.to(device),
                                    t_nc,
                                    has_quality=t_has_quality,
                                    wh_scale=getattr(teacher_model.head, "wh_scale", None) if t_has_quality and hasattr(teacher_model, "head") else None,
                                )
                                s_boxes, s_cls_logits, sh, sw = _decode_anchor_pred(
                                    outputs,
                                    anchors,
                                    num_classes,
                                    has_quality=s_has_quality,
                                    wh_scale=wh_scale_tensor if s_has_quality else None,
                                )
                                if (th, tw) != (sh, sw):
                                    t_boxes = _interp_anchor_map(t_boxes, (sh, sw))
                                    t_cls_logits = _interp_anchor_map(t_cls_logits, (sh, sw))
                                if distill_kl > 0:
                                    temp = max(distill_temperature, 1e-6)
                                    t_prob = torch.nan_to_num(
                                        (t_cls_logits.detach() / temp).softmax(dim=-1), nan=0.0, posinf=1.0, neginf=0.0
                                    )
                                    s_logprob = torch.nan_to_num(
                                        (s_cls_logits / temp).log_softmax(dim=-1), nan=0.0, posinf=0.0, neginf=0.0
                                    )
                                    kl = torch.nn.functional.kl_div(
                                        s_logprob.reshape(-1, num_classes),
                                        t_prob.reshape(-1, num_classes),
                                        reduction="batchmean",
                                    ) * (temp * temp)
                                    if torch.isfinite(kl):
                                        loss_dict["loss"] = loss_dict["loss"] + (distill_kl * distill_scale) * kl
                                        loss_dict["distill_kl"] = kl
                                    elif not warn_nonfinite_kl:
                                        print("Non-finite anchor KL distill value; skipping this term for stability.")
                                        warn_nonfinite_kl = True
                                if distill_box_l1 > 0:
                                    l1d = torch.nn.functional.l1_loss(
                                        torch.nan_to_num(s_boxes),
                                        torch.nan_to_num(t_boxes.detach()),
                                        reduction="mean",
                                    )
                                    loss_dict["loss"] = loss_dict["loss"] + (distill_box_l1 * distill_scale) * l1d
                                    loss_dict["distill_box_l1"] = l1d
                    elif (not anchor_head) and teacher_ct_pred is not None and isinstance(teacher_ct_pred, dict):
                        t_hm = teacher_ct_pred.get("hm")
                        t_off = teacher_ct_pred.get("off")
                        t_wh = teacher_ct_pred.get("wh")
                        sh, sw = outputs["hm"].shape[2], outputs["hm"].shape[3]
                        if t_hm is not None:
                            if t_hm.shape[1] != outputs["hm"].shape[1] and not warn_class_mismatch:
                                print(f"Teacher heatmap classes ({t_hm.shape[1]}) != student ({outputs['hm'].shape[1]}); skipping heatmap distill.")
                                warn_class_mismatch = True
                            elif t_hm.shape[1] == outputs["hm"].shape[1] and distill_kl > 0:
                                t_hm_resized = torch.nn.functional.interpolate(t_hm, size=(sh, sw), mode="bilinear", align_corners=False)
                                s_prob = torch.clamp(outputs["hm"], 1e-6, 1 - 1e-6)
                                t_prob = torch.clamp(t_hm_resized.detach(), 1e-6, 1 - 1e-6)
                                kl = torch.nn.functional.kl_div(torch.log(s_prob), t_prob, reduction="batchmean")
                                loss_dict["loss"] = loss_dict["loss"] + (distill_kl * distill_scale) * kl
                                loss_dict["distill_kl"] = kl
                        if distill_box_l1 > 0:
                            comp = []
                            if t_off is not None:
                                t_off_r = torch.nn.functional.interpolate(t_off, size=(sh, sw), mode="bilinear", align_corners=False)
                                comp.append(torch.nn.functional.l1_loss(outputs["off"], t_off_r.detach(), reduction="mean"))
                            if t_wh is not None:
                                t_wh_r = torch.nn.functional.interpolate(t_wh, size=(sh, sw), mode="bilinear", align_corners=False)
                                comp.append(torch.nn.functional.l1_loss(outputs["wh"], t_wh_r.detach(), reduction="mean"))
                            if comp:
                                l1d = sum(comp) / len(comp)
                                loss_dict["loss"] = loss_dict["loss"] + (distill_box_l1 * distill_scale) * l1d
                                loss_dict["distill_box_l1"] = l1d
            else:
                logits, box_pred = model(imgs)
                loss_dict = detr_loss(logits, box_pred, targets_dev, num_classes=num_classes)
                if teacher_logits is not None and teacher_boxes is not None:
                    # KL distillation on logits (Q,B,C+1)
                    if distill_kl > 0:
                        temp = max(distill_temperature, 1e-6)
                        t_prob = torch.nan_to_num((teacher_logits.detach() / temp).softmax(-1), nan=0.0, posinf=1.0, neginf=0.0)
                        s_logprob = torch.nan_to_num((logits / temp).log_softmax(-1), nan=0.0, posinf=0.0, neginf=0.0)
                        kl = torch.nn.functional.kl_div(s_logprob, t_prob, reduction="batchmean") * (temp * temp)
                        if torch.isfinite(kl):
                            loss_dict["loss"] = loss_dict["loss"] + (distill_kl * distill_scale) * kl
                            loss_dict["distill_kl"] = kl
                        elif not warn_nonfinite_kl:
                            print("Non-finite transformer KL distill value; skipping this term for stability.")
                            warn_nonfinite_kl = True
                    if distill_box_l1 > 0:
                        l1d = torch.nn.functional.l1_loss(box_pred.detach(), teacher_boxes.detach(), reduction="mean")
                        loss_dict["loss"] = loss_dict["loss"] + (distill_box_l1 * distill_scale) * l1d
                        loss_dict["distill_box_l1"] = l1d
            loss = loss_dict["loss"]
        if not torch.isfinite(loss):
            if not warn_nonfinite_loss:
                print(f"Non-finite loss detected at step {step}; skipping backward/optimizer step.")
                warn_nonfinite_loss = True
            continue
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(clip_params, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                clip_grad_norm_(clip_params, grad_clip_norm)
            optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += float(loss.item())
        if arch_cnn_like:
            if anchor_head:
                total_anchor_obj += float(loss_dict["obj"].item())
                total_anchor_cls += float(loss_dict["cls"].item())
                total_anchor_box += float(loss_dict["box"].item())
                if "quality" in loss_dict:
                    total_anchor_quality += float(loss_dict["quality"].item())
            else:
                total_hm += float(loss_dict["hm"].item())
                total_off += float(loss_dict["off"].item())
                total_wh += float(loss_dict["wh"].item())
            if "distill_feat" in loss_dict:
                total_distill_feat += float(loss_dict["distill_feat"].item())
            if "distill_kl" in loss_dict:
                total_distill_kl += float(loss_dict["distill_kl"].item())
            if "distill_box_l1" in loss_dict:
                total_distill_l1 += float(loss_dict["distill_box_l1"].item())
        else:
            total_cls += float(loss_dict["cls"].item())
            total_l1 += float(loss_dict["l1"].item())
            total_iou += float(loss_dict["iou"].item())
            if "distill_kl" in loss_dict:
                total_distill_kl += float(loss_dict["distill_kl"].item())
            if "distill_box_l1" in loss_dict:
                total_distill_l1 += float(loss_dict["distill_box_l1"].item())
        steps += 1

        if (step + 1) % log_interval == 0:
            if arch_cnn_like:
                if anchor_head:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        obj=f"{loss_dict['obj'].item():.4f}",
                        cls=f"{loss_dict['cls'].item():.4f}",
                        box=f"{loss_dict['box'].item():.4f}",
                        quality=f"{loss_dict.get('quality', 0.0):.4f}",
                        step=f"{step+1}/{len(loader)}",
                    )
                else:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        hm=f"{loss_dict['hm'].item():.4f}",
                        off=f"{loss_dict['off'].item():.4f}",
                        wh=f"{loss_dict['wh'].item():.4f}",
                        step=f"{step+1}/{len(loader)}",
                    )
            else:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    cls=f"{loss_dict['cls'].item():.4f}",
                    l1=f"{loss_dict['l1'].item():.4f}",
                    iou=f"{loss_dict['iou'].item():.4f}",
                    step=f"{step+1}/{len(loader)}",
                )

    logs = {"loss": total_loss / steps if steps else 0.0}
    if arch_cnn_like:
        if anchor_head:
            logs.update(
                {
                    "obj": total_anchor_obj / steps if steps else 0.0,
                    "cls": total_anchor_cls / steps if steps else 0.0,
                    "box": total_anchor_box / steps if steps else 0.0,
                    "quality": total_anchor_quality / steps if steps else 0.0,
                }
            )
        else:
            logs.update(
                {
                    "hm": total_hm / steps if steps else 0.0,
                    "off": total_off / steps if steps else 0.0,
                    "wh": total_wh / steps if steps else 0.0,
                }
            )
        if distill_feat > 0:
            logs["distill_feat"] = total_distill_feat / steps if steps else 0.0
        if distill_kl > 0:
            logs["distill_kl"] = total_distill_kl / steps if steps else 0.0
        if distill_box_l1 > 0:
            logs["distill_box_l1"] = total_distill_l1 / steps if steps else 0.0
    else:
        logs.update(
            {
                "cls": total_cls / steps if steps else 0.0,
                "l1": total_l1 / steps if steps else 0.0,
                "iou": total_iou / steps if steps else 0.0,
            }
        )
        if distill_kl > 0:
            logs["distill_kl"] = total_distill_kl / steps if steps else 0.0
        if distill_box_l1 > 0:
            logs["distill_box_l1"] = total_distill_l1 / steps if steps else 0.0
    return logs


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    arch: str,
    conf_thresh: float,
    topk: int,
    iou_thresh: float = 0.5,
    use_amp: bool = False,
    num_classes: int = 1,
    sample_dir: str = None,
    class_ids = None,
    sample_limit: int = 10,
    coco_eval: bool = False,
    coco_per_class: bool = False,
    use_anchor: bool = False,
    anchors: torch.Tensor = None,
    iou_loss_type: str = "giou",
    anchor_assigner: str = "legacy",
    anchor_cls_loss: str = "bce",
    simota_topk: int = 10,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    sample_count = 0
    total_loss = 0.0
    total_hm = 0.0
    total_off = 0.0
    total_wh = 0.0
    total_anchor_obj = 0.0
    total_anchor_cls = 0.0
    total_anchor_box = 0.0
    total_anchor_quality = 0.0
    total_cls = 0.0
    total_l1 = 0.0
    total_iou = 0.0
    steps = 0
    coco_images = []
    coco_annos = []
    coco_dets = []
    anno_id = 1
    global_img_idx = 0
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "orange",
        "purple",
        "lime",
        "pink",
    ]
    arch_cnn_like = arch in ("cnn", "ultratinyod")
    anchor_head = bool(use_anchor or arch == "ultratinyod")
    use_quality_head = bool(getattr(model, "has_quality_head", getattr(model, "use_improved_head", False)))
    score_mode = getattr(model, "score_mode", None)
    quality_power = getattr(model, "quality_power", 1.0)
    wh_scale_tensor = None
    if hasattr(model, "head"):
        if hasattr(model.head, "wh_scale"):
            wh_scale_tensor = model.head.wh_scale
        if score_mode is None and hasattr(model.head, "score_mode"):
            score_mode = model.head.score_mode
        quality_power = getattr(model.head, "quality_power", quality_power)

    def render_sample(img_path, pred_list, save_path):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            w, h = im.size
            for score, cls, box in pred_list:
                cx, cy, bw, bh = box.tolist()
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                # Clamp to image bounds to avoid invalid rectangles
                x1 = max(0.0, min(x1, w))
                x2 = max(0.0, min(x2, w))
                y1 = max(0.0, min(y1, h))
                y2 = max(0.0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue
                color = colors[cls % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, y1), f"{score:.2f}", fill=color, font=font)
            im.save(save_path)

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Eval", ncols=120)
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets_cpu = move_targets(targets, torch.device("cpu"))
            targets_dev = move_targets(targets, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                if arch_cnn_like:
                    outputs = model(imgs)
                    if anchor_head:
                        loss_dict = anchor_loss(
                            outputs,
                            targets_dev,
                            anchors=anchors,
                            num_classes=num_classes,
                            iou_loss=iou_loss_type,
                            assigner=anchor_assigner,
                            cls_loss_type=anchor_cls_loss,
                            simota_topk=simota_topk,
                            use_quality=use_quality_head,
                            wh_scale=wh_scale_tensor,
                        )
                        preds = decode_anchor(
                            outputs,
                            anchors=anchors,
                            num_classes=num_classes,
                            conf_thresh=conf_thresh,
                            nms_thresh=0.5,
                            has_quality=use_quality_head,
                            wh_scale=wh_scale_tensor,
                            score_mode=score_mode or "obj_quality_cls",
                            quality_power=quality_power,
                        )
                    else:
                        loss_dict = centernet_loss(outputs, targets_dev, num_classes=num_classes)
                        preds = decode_centernet(outputs, conf_thresh=conf_thresh, topk=topk)
                else:
                    logits, box_pred = model(imgs)
                    loss_dict = detr_loss(logits, box_pred, targets_dev, num_classes=num_classes)
                    preds = decode_detr(logits, box_pred, conf_thresh=conf_thresh)
            # accumulate losses
            total_loss += float(loss_dict["loss"].item())
            if arch_cnn_like:
                if anchor_head:
                    total_anchor_obj += float(loss_dict["obj"].item())
                    total_anchor_cls += float(loss_dict["cls"].item())
                    total_anchor_box += float(loss_dict["box"].item())
                    if "quality" in loss_dict:
                        total_anchor_quality += float(loss_dict["quality"].item())
                else:
                    total_hm += float(loss_dict["hm"].item())
                    total_off += float(loss_dict["off"].item())
                    total_wh += float(loss_dict["wh"].item())
            else:
                total_cls += float(loss_dict["cls"].item())
                total_l1 += float(loss_dict["l1"].item())
                total_iou += float(loss_dict["iou"].item())
            steps += 1
            # move predictions to CPU for metric computation
            preds_cpu = []
            for p_img in preds:
                preds_cpu.append([(score, cls, box.detach().cpu()) for score, cls, box in p_img])
            all_preds.extend(preds_cpu)
            all_targets.extend(targets_cpu)
            if sample_dir and sample_count < sample_limit:
                for b_idx, pred_img in enumerate(preds):
                    if sample_count >= sample_limit:
                        break
                    img_path = targets[b_idx]["image_id"]
                    filename = os.path.basename(img_path)
                    stem = os.path.splitext(filename)[0]
                    save_name = f"{sample_count:02d}_" + stem + ".png"
                    save_path = os.path.join(sample_dir, save_name)
                    render_sample(img_path, pred_img, save_path)
                    sample_count += 1
            if coco_eval:
                bsz = imgs.shape[0]
                for b_idx in range(bsz):
                    orig = targets_cpu[b_idx].get("orig_size")
                    if orig:
                        h, w = orig
                    else:
                        h, w = imgs.shape[2], imgs.shape[3]
                    img_id = global_img_idx
                    coco_images.append({"id": img_id, "width": w, "height": h})
                    gt_boxes = targets_cpu[b_idx]["boxes"]
                    gt_labels = targets_cpu[b_idx]["labels"]
                    for j, (cx, cy, bw, bh) in enumerate(gt_boxes):
                        x = (cx - bw / 2) * w
                        y = (cy - bh / 2) * h
                        bw_abs = bw * w
                        bh_abs = bh * h
                        coco_annos.append(
                            {
                                "id": anno_id,
                                "image_id": img_id,
                                "category_id": int(class_ids[int(gt_labels[j].item())]) if class_ids else int(gt_labels[j].item()),
                                "bbox": [float(x), float(y), float(bw_abs), float(bh_abs)],
                                "area": float(max(bw_abs, 0) * max(bh_abs, 0)),
                                "iscrowd": 0,
                            }
                        )
                        anno_id += 1
                    for score, cls, box in preds[b_idx]:
                        cx, cy, bw, bh = box.tolist()
                        x = (cx - bw / 2) * w
                        y = (cy - bh / 2) * h
                        bw_abs = bw * w
                        bh_abs = bh * h
                        coco_dets.append(
                            {
                                "image_id": img_id,
                                "category_id": int(class_ids[cls]) if class_ids else int(cls),
                                "bbox": [float(x), float(y), float(bw_abs), float(bh_abs)],
                                "score": float(score),
                            }
                        )
                    global_img_idx += 1

    metrics = evaluate_map(all_preds, all_targets, num_classes=num_classes, iou_thresh=iou_thresh)
    if steps > 0:
        metrics["loss"] = total_loss / steps
        if arch_cnn_like:
            if anchor_head:
                metrics["obj"] = total_anchor_obj / steps
                metrics["cls"] = total_anchor_cls / steps
                metrics["box"] = total_anchor_box / steps
                metrics["quality"] = total_anchor_quality / steps
            else:
                metrics["hm"] = total_hm / steps
                metrics["off"] = total_off / steps
                metrics["wh"] = total_wh / steps
        else:
            metrics["cls"] = total_cls / steps
            metrics["l1"] = total_l1 / steps
            metrics["iou"] = total_iou / steps
            if coco_eval:
                coco_metrics = run_coco_eval(coco_images, coco_annos, coco_dets, class_ids or list(range(num_classes)), per_class=coco_per_class)
                metrics.update(coco_metrics)
    return metrics


def main():
    args = parse_args()
    class_ids = parse_classes(args.classes)
    num_classes = len(class_ids)
    aug_cfg = load_aug_config(args.aug_config)
    resize_mode = normalize_resize_mode(args.resize_mode)
    use_skip = bool(args.use_skip)
    utod_residual = bool(args.utod_residual)
    backbone = args.backbone
    backbone_channels = parse_int_list(args.backbone_channels)
    backbone_blocks = parse_int_list(args.backbone_blocks)
    backbone_se = args.backbone_se
    backbone_skip = bool(args.backbone_skip)
    backbone_skip_cat = bool(args.backbone_skip_cat)
    backbone_skip_shuffle_cat = bool(args.backbone_skip_shuffle_cat)
    backbone_skip_s2d_cat = bool(args.backbone_skip_s2d_cat)
    if backbone_skip_s2d_cat:
        backbone_skip = True
        backbone_skip_shuffle_cat = False
        backbone_skip_cat = False
    elif backbone_skip_shuffle_cat:
        backbone_skip = True
        backbone_skip_cat = False
    elif backbone_skip_cat:
        backbone_skip = True  # cat fusion implies enabling long skip
    backbone_fpn = bool(args.backbone_fpn)
    backbone_out_stride = int(args.backbone_out_stride) if args.backbone_out_stride is not None else None
    grad_clip_norm = float(args.grad_clip_norm)
    activation = args.activation
    use_ema = bool(args.use_ema)
    ema_decay = float(args.ema_decay)
    use_fpn = bool(args.use_fpn)
    distill_kl = float(args.distill_kl)
    distill_box_l1 = float(args.distill_box_l1)
    distill_temperature = float(args.distill_temperature)
    distill_cosine = bool(args.distill_cosine)
    distill_feat = float(args.distill_feat)
    cnn_width = int(args.cnn_width)
    teacher_backbone = args.teacher_backbone
    teacher_backbone_arch = args.teacher_backbone_arch
    teacher_backbone_norm = args.teacher_backbone_norm
    teacher_ckpt = args.teacher_ckpt
    teacher_arch = args.teacher_arch
    teacher_num_queries = args.teacher_num_queries
    teacher_d_model = args.teacher_d_model
    teacher_heads = args.teacher_heads
    teacher_layers = args.teacher_layers
    teacher_dim_feedforward = args.teacher_dim_feedforward
    teacher_use_skip = bool(args.teacher_use_skip)
    teacher_activation = args.teacher_activation
    teacher_use_fpn = bool(args.teacher_use_fpn)
    teacher_use_anchor = False
    teacher_anchor_tensor = None
    teacher_num_classes = None
    arch_cnn_like = args.arch in ("cnn", "ultratinyod")
    use_anchor = bool(args.use_anchor)
    anchor_list = parse_anchors_str(args.anchors)
    auto_anchors = bool(args.auto_anchors)
    num_anchors = int(args.num_anchors if args.num_anchors else 0) or 3
    if anchor_list:
        num_anchors = len(anchor_list)
    iou_loss_type = args.iou_loss
    anchor_assigner = args.anchor_assigner
    anchor_cls_loss = args.anchor_cls_loss
    simota_topk = int(args.simota_topk)
    last_se = args.last_se
    last_width_scale = float(args.last_width_scale)
    use_batchnorm = bool(args.use_batchnorm)
    output_stride = int(args.output_stride) if args.output_stride else 16
    use_improved_head = False
    use_iou_aware_head = bool(args.use_iou_aware_head)
    quality_power = float(args.quality_power)
    utod_head_ese = bool(args.utod_head_ese)
    utod_context_rfb = bool(args.utod_context_rfb)
    utod_context_dilation = int(args.utod_context_dilation)
    utod_large_obj_branch = bool(args.utod_large_obj_branch)
    utod_large_obj_depth = int(args.utod_large_obj_depth)
    utod_large_obj_ch_scale = float(args.utod_large_obj_ch_scale)
    if args.arch == "ultratinyod":
        use_anchor = True
        # Allow stride-4 variant; default to 8 when not explicitly set.
        output_stride = int(args.output_stride) if args.output_stride and int(args.output_stride) in (4, 8, 16) else 8
        use_improved_head = bool(args.use_improved_head)
    if backbone is not None and backbone_out_stride is not None:
        output_stride = backbone_out_stride
    if args.arch != "cnn" and backbone not in (None, "none"):
        print(f"--backbone is only used for arch=cnn; ignoring backbone={backbone}")
        backbone = None
    if backbone in ("none", ""):
        backbone = None
    if backbone is None:
        backbone_channels = None
        backbone_blocks = None
        backbone_se = "none"
        backbone_skip = False
        backbone_skip_cat = False
        backbone_skip_shuffle_cat = False
        backbone_skip_s2d_cat = False
        backbone_fpn = False
        backbone_out_stride = None
    if args.resume and args.ckpt:
        raise ValueError("--resume and --ckpt cannot be used together.")
    if args.val_only and not (args.resume or args.ckpt):
        raise ValueError("--val-only requires either --resume or --ckpt to load weights.")

    pretrain_meta = torch.load(args.ckpt, map_location="cpu") if args.ckpt else None
    ckpt_meta = torch.load(args.resume, map_location="cpu") if args.resume else None
    img_h, img_w = parse_img_size(args.img_size)

    def apply_meta(meta: Dict, label: str, allow_distill: bool = False):
        nonlocal class_ids, num_classes, aug_cfg, resize_mode, use_skip, utod_residual, grad_clip_norm, activation, use_ema, ema_decay, use_fpn, backbone, backbone_channels, backbone_blocks, backbone_se, backbone_skip, backbone_skip_cat, backbone_skip_shuffle_cat, backbone_skip_s2d_cat, backbone_fpn, backbone_out_stride, use_batchnorm, cnn_width, use_improved_head, utod_head_ese, use_iou_aware_head, quality_power, utod_context_rfb, utod_context_dilation, utod_large_obj_branch, utod_large_obj_depth, utod_large_obj_ch_scale
        nonlocal teacher_ckpt, teacher_arch, teacher_num_queries, teacher_d_model, teacher_heads, teacher_layers, teacher_dim_feedforward, teacher_use_skip, teacher_activation, teacher_use_fpn, teacher_backbone, teacher_backbone_arch, teacher_backbone_norm
        nonlocal distill_kl, distill_box_l1, distill_temperature, distill_cosine, distill_feat
        nonlocal use_anchor, anchor_list, auto_anchors, num_anchors, iou_loss_type, anchor_assigner, anchor_cls_loss, simota_topk
        nonlocal last_se, last_width_scale, output_stride
        if "cnn_width" in meta:
            ckpt_width = int(meta["cnn_width"])
            if ckpt_width != cnn_width:
                print(f"Overriding CLI cnn-width={cnn_width} with {label} cnn-width={ckpt_width}")
            cnn_width = ckpt_width
        if "classes" in meta:
            ckpt_classes = [int(c) for c in meta["classes"]]
            if set(ckpt_classes) != set(class_ids):
                print(f"Overriding CLI classes {class_ids} with {label} classes {ckpt_classes}")
            class_ids = ckpt_classes
            num_classes = len(class_ids)
        if "augment_cfg" in meta:
            aug_cfg = meta["augment_cfg"]
        if "resize_mode" in meta and meta["resize_mode"]:
            try:
                new_resize_mode = normalize_resize_mode(meta["resize_mode"])
                if new_resize_mode != resize_mode:
                    print(f"Overriding CLI resize-mode={resize_mode} with {label} resize-mode={new_resize_mode}")
                resize_mode = new_resize_mode
            except ValueError:
                print(f"[WARN] Unsupported {label} resize-mode={meta['resize_mode']}; keeping {resize_mode}")
        if "use_skip" in meta and bool(meta["use_skip"]) != use_skip:
            print(f"Overriding CLI use-skip={use_skip} with {label} use-skip={bool(meta['use_skip'])}")
            use_skip = bool(meta["use_skip"])
        if "use_batchnorm" in meta and bool(meta["use_batchnorm"]) != use_batchnorm:
            print(f"Overriding CLI use-batchnorm={use_batchnorm} with {label} use-batchnorm={bool(meta['use_batchnorm'])}")
            use_batchnorm = bool(meta["use_batchnorm"])
        if "utod_residual" in meta and bool(meta["utod_residual"]) != utod_residual:
            print(f"Overriding CLI utod-residual={utod_residual} with {label} utod-residual={bool(meta['utod_residual'])}")
            utod_residual = bool(meta["utod_residual"])
        if "utod_context_rfb" in meta:
            utod_context_rfb = bool(meta["utod_context_rfb"])
        if "utod_context_dilation" in meta and meta["utod_context_dilation"]:
            utod_context_dilation = int(meta["utod_context_dilation"])
        if "utod_large_obj_branch" in meta:
            utod_large_obj_branch = bool(meta["utod_large_obj_branch"])
        if "utod_large_obj_depth" in meta and meta["utod_large_obj_depth"]:
            utod_large_obj_depth = int(meta["utod_large_obj_depth"])
        if "utod_large_obj_ch_scale" in meta and meta["utod_large_obj_ch_scale"]:
            utod_large_obj_ch_scale = float(meta["utod_large_obj_ch_scale"])
        if "use_fpn" in meta and bool(meta["use_fpn"]) != use_fpn:
            print(f"Overriding CLI use-fpn={use_fpn} with {label} use-fpn={bool(meta['use_fpn'])}")
            use_fpn = bool(meta["use_fpn"])
        if "backbone" in meta and meta["backbone"] not in (None, "none", ""):
            ckpt_backbone = meta["backbone"]
            if ckpt_backbone != backbone:
                print(f"Overriding CLI backbone={backbone} with {label} backbone={ckpt_backbone}")
            backbone = ckpt_backbone
        if "backbone_channels" in meta and meta["backbone_channels"]:
            backbone_channels = [int(x) for x in meta["backbone_channels"]]
        if "backbone_blocks" in meta and meta["backbone_blocks"]:
            backbone_blocks = [int(x) for x in meta["backbone_blocks"]]
        if "backbone_se" in meta and meta["backbone_se"]:
            backbone_se = meta["backbone_se"]
        if "backbone_skip" in meta:
            backbone_skip = bool(meta["backbone_skip"])
        if "backbone_skip_cat" in meta:
            backbone_skip_cat = bool(meta["backbone_skip_cat"])
            if backbone_skip_cat:
                backbone_skip = True
        if "backbone_skip_shuffle_cat" in meta:
            backbone_skip_shuffle_cat = bool(meta["backbone_skip_shuffle_cat"])
            if backbone_skip_shuffle_cat:
                backbone_skip = True
                backbone_skip_cat = False
        if "backbone_skip_s2d_cat" in meta:
            backbone_skip_s2d_cat = bool(meta["backbone_skip_s2d_cat"])
            if backbone_skip_s2d_cat:
                backbone_skip = True
                backbone_skip_shuffle_cat = False
                backbone_skip_cat = False
        if "backbone_fpn" in meta:
            backbone_fpn = bool(meta["backbone_fpn"])
        if "backbone_out_stride" in meta and meta["backbone_out_stride"]:
            backbone_out_stride = int(meta["backbone_out_stride"])
        if "use_anchor" in meta:
            use_anchor = bool(meta["use_anchor"])
        if "anchors" in meta and meta["anchors"]:
            anchor_list = [tuple(a) for a in meta["anchors"]]
            num_anchors = len(anchor_list)
        if "auto_anchors" in meta:
            auto_anchors = bool(meta["auto_anchors"])
        if "num_anchors" in meta:
            num_anchors = int(meta["num_anchors"])
        if "iou_loss" in meta and meta["iou_loss"]:
            iou_loss_type = meta["iou_loss"]
        if "anchor_assigner" in meta and meta["anchor_assigner"]:
            anchor_assigner = meta["anchor_assigner"]
        if "anchor_cls_loss" in meta and meta["anchor_cls_loss"]:
            anchor_cls_loss = meta["anchor_cls_loss"]
        if "simota_topk" in meta and meta["simota_topk"]:
            simota_topk = int(meta["simota_topk"])
        if "use_improved_head" in meta:
            use_improved_head = bool(meta["use_improved_head"])
        if "use_iou_aware_head" in meta:
            use_iou_aware_head = bool(meta["use_iou_aware_head"])
        if "quality_power" in meta:
            quality_power = float(meta["quality_power"])
        if "utod_head_ese" in meta:
            utod_head_ese = bool(meta["utod_head_ese"])
        if "last_se" in meta and meta["last_se"]:
            last_se = meta["last_se"]
        if "last_width_scale" in meta and meta["last_width_scale"]:
            last_width_scale = float(meta["last_width_scale"])
        if "grad_clip_norm" in meta and abs(float(meta["grad_clip_norm"]) - grad_clip_norm) > 1e-8:
            print(f"Overriding CLI grad-clip-norm={grad_clip_norm} with {label} grad-clip-norm={float(meta['grad_clip_norm'])}")
            grad_clip_norm = float(meta["grad_clip_norm"])
        if "activation" in meta and meta["activation"] != activation:
            print(f"Overriding CLI activation={activation} with {label} activation={meta['activation']}")
            activation = meta["activation"]
        if "output_stride" in meta and meta["output_stride"]:
            if int(meta["output_stride"]) != output_stride:
                print(f"Overriding CLI output-stride={output_stride} with {label} output-stride={int(meta['output_stride'])}")
            output_stride = int(meta["output_stride"])
        if "use_ema" in meta and bool(meta["use_ema"]) != use_ema:
            print(f"Overriding CLI use-ema={use_ema} with {label} use-ema={bool(meta['use_ema'])}")
            use_ema = bool(meta["use_ema"])
        if "ema_decay" in meta and abs(float(meta["ema_decay"]) - ema_decay) > 1e-8:
            print(f"Overriding CLI ema-decay={ema_decay} with {label} ema-decay={float(meta['ema_decay'])}")
            ema_decay = float(meta["ema_decay"])
        if allow_distill:
            if "teacher_ckpt" in meta and meta["teacher_ckpt"]:
                teacher_ckpt = meta["teacher_ckpt"]
            if "teacher_arch" in meta and meta["teacher_arch"]:
                teacher_arch = meta["teacher_arch"]
            if "teacher_num_queries" in meta and meta["teacher_num_queries"] is not None:
                teacher_num_queries = int(meta["teacher_num_queries"])
            if "teacher_d_model" in meta and meta["teacher_d_model"] is not None:
                teacher_d_model = int(meta["teacher_d_model"])
            if "teacher_heads" in meta and meta["teacher_heads"] is not None:
                teacher_heads = int(meta["teacher_heads"])
            if "teacher_layers" in meta and meta["teacher_layers"] is not None:
                teacher_layers = int(meta["teacher_layers"])
            if "teacher_dim_feedforward" in meta and meta["teacher_dim_feedforward"] is not None:
                teacher_dim_feedforward = int(meta["teacher_dim_feedforward"])
            if "teacher_use_skip" in meta:
                teacher_use_skip = bool(meta["teacher_use_skip"])
            if "teacher_activation" in meta and meta["teacher_activation"]:
                teacher_activation = meta["teacher_activation"]
            if "teacher_use_fpn" in meta:
                teacher_use_fpn = bool(meta["teacher_use_fpn"])
            if "teacher_backbone" in meta and meta["teacher_backbone"]:
                teacher_backbone = meta["teacher_backbone"]
            if "teacher_backbone_arch" in meta and meta["teacher_backbone_arch"]:
                teacher_backbone_arch = meta["teacher_backbone_arch"]
            if "teacher_backbone_norm" in meta and meta["teacher_backbone_norm"]:
                teacher_backbone_norm = meta["teacher_backbone_norm"]
            if "distill_kl" in meta:
                distill_kl = float(meta["distill_kl"])
            if "distill_box_l1" in meta:
                distill_box_l1 = float(meta["distill_box_l1"])
            if "distill_temperature" in meta:
                distill_temperature = float(meta["distill_temperature"])
            if "distill_cosine" in meta:
                distill_cosine = bool(meta["distill_cosine"])
            if "distill_feat" in meta:
                distill_feat = float(meta["distill_feat"])

    if pretrain_meta is not None:
        apply_meta(pretrain_meta, f"ckpt {args.ckpt}", allow_distill=False)
    if ckpt_meta is not None:
        apply_meta(ckpt_meta, f"resume {args.resume}", allow_distill=True)
    if args.arch == "ultratinyod":
        use_anchor = True
    anchor_head = bool(use_anchor or args.arch == "ultratinyod")
    if arch_cnn_like and distill_feat > 0 and not (teacher_backbone or teacher_ckpt):
        print("distill-feat requested but no teacher backbone or teacher checkpoint provided; disabling feature distillation.")
        distill_feat = 0.0
    set_seed(args.seed)
    device = default_device(args.device)
    run_dir = os.path.join("runs", args.exp_name)
    ensure_dir(run_dir)
    log_path = os.path.join(run_dir, "train.log")
    existing_log = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                existing_log = f.read()
        except OSError:
            existing_log = ""
    cli_dump = json.dumps(vars(args), sort_keys=True, indent=2, default=str)
    header_lines = [
        "=== CLI args ===",
        cli_dump,
        f"resolved_resize_mode: {resize_mode}",
        "",
    ]
    with open(log_path, "w") as f:
        f.write("\n".join(header_lines))
        if existing_log:
            if not existing_log.startswith("\n"):
                f.write("\n")
            f.write(existing_log)
    writer = SummaryWriter(log_dir=run_dir)
    use_amp = bool(args.use_amp and device.type == "cuda")

    train_ds, val_ds = make_datasets(args, class_ids, aug_cfg, resize_mode=resize_mode)
    anchors_tensor = None
    if arch_cnn_like and anchor_head:
        if anchor_list:
            anchors_np = np.array(anchor_list, dtype=np.float32)
        elif auto_anchors:
            wh = collect_box_wh(train_ds)
            anchors_np = auto_compute_anchors(wh, k=num_anchors)
        else:
            anchors_np = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=np.float32)
        if anchors_np.size == 0:
            anchors_np = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=np.float32)
        if anchors_np.shape[0] > num_anchors:
            anchors_np = anchors_np[:num_anchors]
        num_anchors = anchors_np.shape[0]
        anchors_tensor = torch.tensor(anchors_np, dtype=torch.float32)
        anchors_tensor = anchors_tensor.to(device)
        anchor_list = [tuple(map(float, a)) for a in anchors_np.tolist()]

    def _load_with_log(target, state, strict: bool, label: str):
        missing, unexpected = target.load_state_dict(state, strict=strict)
        if not strict:
            if missing:
                print(f"[{label}] missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
            if unexpected:
                print(f"[{label}] unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    if args.arch != "cnn":
        backbone = None
    model = build_model(
        args.arch,
        width=cnn_width,
        num_queries=args.num_queries,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=num_classes,
        use_skip=use_skip,
        activation=activation,
        use_fpn=use_fpn,
        use_anchor=use_anchor,
        utod_use_residual=utod_residual,
        use_head_ese=utod_head_ese,
        num_anchors=num_anchors,
        anchors=anchor_list,
        use_iou_aware_head=use_iou_aware_head,
        quality_power=quality_power,
        last_se=last_se,
        last_width_scale=last_width_scale,
        output_stride=output_stride,
        utod_context_rfb=utod_context_rfb,
        utod_context_dilation=utod_context_dilation,
        utod_large_obj_branch=utod_large_obj_branch,
        utod_large_obj_depth=utod_large_obj_depth,
        utod_large_obj_ch_scale=utod_large_obj_ch_scale,
        backbone=backbone,
        backbone_channels=backbone_channels,
        backbone_blocks=backbone_blocks,
        backbone_se=backbone_se,
        backbone_skip=backbone_skip,
        backbone_skip_cat=backbone_skip_cat,
        backbone_skip_shuffle_cat=backbone_skip_shuffle_cat,
        backbone_skip_s2d_cat=backbone_skip_s2d_cat,
        backbone_fpn=backbone_fpn,
        backbone_out_stride=backbone_out_stride,
        anchor_assigner=anchor_assigner,
        anchor_cls_loss=anchor_cls_loss,
        simota_topk=simota_topk,
        use_batchnorm=use_batchnorm,
        use_improved_head=use_improved_head if args.arch == "ultratinyod" else False,
    ).to(device)
    output_stride = getattr(model, "out_stride", output_stride)
    if use_anchor and anchors_tensor is not None and hasattr(model, "set_anchors"):
        model.set_anchors(anchors_tensor)
    ema_helper = None
    teacher_backbone_model = None
    teacher_feature_dim = None
    feature_adapter = None
    teacher_model_arch = None

    if pretrain_meta is not None:
        _load_with_log(
            model,
            pretrain_meta["model"],
            strict=not args.ckpt_non_strict,
            label=f"ckpt model ({'strict' if not args.ckpt_non_strict else 'non-strict'})",
        )
        if use_ema:
            ema_helper = ModelEma(model, decay=ema_decay, device=device)
            if "ema" in pretrain_meta and pretrain_meta["ema"] is not None:
                _load_with_log(
                    ema_helper.ema,
                    pretrain_meta["ema"],
                    strict=not args.ckpt_non_strict,
                    label=f"ckpt ema ({'strict' if not args.ckpt_non_strict else 'non-strict'})",
                )
                if "ema_updates" in pretrain_meta:
                    ema_helper.updates = int(pretrain_meta["ema_updates"])
            else:
                ema_helper.ema.load_state_dict(model.state_dict())
        print(f"Initialized model weights from {args.ckpt}")
    backbone_cfg = None
    if arch_cnn_like and distill_feat > 0 and teacher_backbone:
        try:
            teacher_backbone_model, backbone_cfg = load_dinov3_backbone(
                teacher_backbone, img_size=(img_h, img_w), device=device, arch_hint=teacher_backbone_arch
            )
            teacher_dim = getattr(teacher_backbone_model, "embed_dim", None)
            student_channels = _infer_cnn_feat_channels(model, use_anchor=use_anchor, num_classes=num_classes)
            if teacher_dim is None:
                teacher_dim = student_channels
            teacher_feature_dim = teacher_dim
            if student_channels != teacher_dim:
                feature_adapter = torch.nn.Conv2d(student_channels, teacher_dim, kernel_size=1).to(device)
            else:
                feature_adapter = torch.nn.Identity()
            print(f"Loaded teacher backbone from {teacher_backbone} (dim={teacher_dim}, stride={backbone_cfg.get('out_stride', 'auto') if backbone_cfg else 'auto'})")
        except Exception as e:
            print(f"Failed to load teacher backbone {teacher_backbone}: {e}")
            teacher_backbone_model = None
            feature_adapter = None
            distill_feat = 0.0
    params = list(model.parameters())
    if feature_adapter is not None and len(list(feature_adapter.parameters())) > 0:
        params += list(feature_adapter.parameters())
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    try:
        scaler = torch.amp.GradScaler(enabled=bool(use_amp and device.type == "cuda"))
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and device.type == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)
    start_epoch = 0
    best_map = float("-inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if feature_adapter is not None and "feat_adapter" in ckpt and ckpt["feat_adapter"] is not None:
            try:
                feature_adapter.load_state_dict(ckpt["feat_adapter"])
            except Exception as e:
                print(f"Warning: could not load feature adapter from resume checkpoint: {e}")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if use_ema:
            if ema_helper is None:
                ema_helper = ModelEma(model, decay=ema_decay, device=device)
            if "ema" in ckpt and ckpt["ema"] is not None:
                ema_helper.ema.load_state_dict(ckpt["ema"])
                if "ema_updates" in ckpt:
                    ema_helper.updates = int(ckpt["ema_updates"])
            else:
                ema_helper.ema.load_state_dict(model.state_dict())
        start_epoch = int(ckpt.get("epoch", 0))
        best_map = float(ckpt.get("best_map", ckpt.get("metrics", {}).get("mAP@0.5", float("-inf"))))
        best_map_print = best_map if best_map != float("-inf") else float("nan")
        print(f"Resumed from {args.resume} at epoch {start_epoch} with best mAP@0.5={best_map_print:.4f}")

    if use_ema and ema_helper is None:
        ema_helper = ModelEma(model, decay=ema_decay, device=device)

    if use_anchor and anchors_tensor is not None and hasattr(model, "set_anchors"):
        model.set_anchors(anchors_tensor)
        if ema_helper is not None and hasattr(ema_helper.ema, "set_anchors"):
            ema_helper.ema.set_anchors(anchors_tensor)

    # Align recorded output stride to actual model if available
    output_stride = getattr(model, "out_stride", output_stride)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
        persistent_workers=True,
    )
    teacher_model = None
    if teacher_ckpt and not args.val_only:
        t_meta = torch.load(teacher_ckpt, map_location="cpu")
        t_arch = (teacher_arch or t_meta.get("arch", args.arch)).lower()
        teacher_model_arch = t_arch
        if t_arch == "transformer":
            t_use_skip = teacher_use_skip or bool(t_meta.get("use_skip", use_skip))
            t_activation = teacher_activation or t_meta.get("activation", activation)
            t_use_fpn = teacher_use_fpn or bool(t_meta.get("use_fpn", use_fpn))
            teacher_model = build_model(
                t_arch,
                width=int(t_meta.get("cnn_width", args.cnn_width)),
                num_queries=teacher_num_queries or args.num_queries,
                d_model=teacher_d_model or args.d_model,
                heads=teacher_heads or args.heads,
                layers=teacher_layers or args.layers,
                dim_feedforward=teacher_dim_feedforward or args.dim_feedforward,
                num_classes=num_classes,
                use_skip=t_use_skip,
                activation=t_activation,
                use_fpn=t_use_fpn,
                use_batchnorm=bool(t_meta.get("use_batchnorm", True)),
            ).to(device)
            teacher_model.load_state_dict(t_meta["model"])
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            teacher_num_classes = num_classes
            print(f"Loaded teacher from {teacher_ckpt} (arch={t_arch})")
        elif t_arch in ("cnn", "ultratinyod"):
            t_use_skip = teacher_use_skip or bool(t_meta.get("use_skip", use_skip))
            t_activation = teacher_activation or t_meta.get("activation", activation)
            t_use_fpn = teacher_use_fpn or bool(t_meta.get("use_fpn", use_fpn))
            t_use_anchor = True if t_arch == "ultratinyod" else bool(t_meta.get("use_anchor", use_anchor))
            t_anchor_list = t_meta.get("anchors", anchor_list) or []
            t_anchor_list = [tuple(map(float, a)) for a in t_anchor_list]
            t_num_anchors = int(t_meta.get("num_anchors", len(t_anchor_list) if t_anchor_list else num_anchors))
            t_last_se = t_meta.get("last_se", last_se)
            t_last_width_scale = t_meta.get("last_width_scale", last_width_scale)
            t_output_stride = int(t_meta.get("output_stride", output_stride))
            t_utod_residual = bool(t_meta.get("utod_residual", utod_residual))
            t_cnn_width = int(t_meta.get("cnn_width", args.cnn_width))
            t_use_improved_head = bool(t_meta.get("use_improved_head", use_improved_head))
            t_utod_head_ese = bool(t_meta.get("utod_head_ese", utod_head_ese))
            t_use_iou_aware_head = bool(t_meta.get("use_iou_aware_head", use_iou_aware_head))
            t_quality_power = float(t_meta.get("quality_power", quality_power))
            t_utod_context_rfb = bool(t_meta.get("utod_context_rfb", utod_context_rfb))
            t_utod_context_dilation = int(t_meta.get("utod_context_dilation", utod_context_dilation))
            t_utod_large_obj_branch = bool(t_meta.get("utod_large_obj_branch", utod_large_obj_branch))
            t_utod_large_obj_depth = int(t_meta.get("utod_large_obj_depth", utod_large_obj_depth))
            t_utod_large_obj_ch_scale = float(t_meta.get("utod_large_obj_ch_scale", utod_large_obj_ch_scale))
            t_backbone = t_meta.get("backbone", backbone)
            t_backbone_channels = t_meta.get("backbone_channels", backbone_channels)
            t_backbone_blocks = t_meta.get("backbone_blocks", backbone_blocks)
            t_backbone_se = t_meta.get("backbone_se", backbone_se)
            t_backbone_skip = t_meta.get("backbone_skip", backbone_skip)
            t_backbone_skip_cat = t_meta.get("backbone_skip_cat", backbone_skip_cat)
            if t_backbone_skip_cat:
                t_backbone_skip = True
            t_backbone_skip_shuffle_cat = t_meta.get("backbone_skip_shuffle_cat", backbone_skip_shuffle_cat)
            if t_backbone_skip_shuffle_cat:
                t_backbone_skip = True
                t_backbone_skip_cat = False
            t_backbone_skip_s2d_cat = t_meta.get("backbone_skip_s2d_cat", backbone_skip_s2d_cat)
            if t_backbone_skip_s2d_cat:
                t_backbone_skip = True
                t_backbone_skip_shuffle_cat = False
                t_backbone_skip_cat = False
            t_backbone_fpn = t_meta.get("backbone_fpn", backbone_fpn)
            t_backbone_out_stride = t_meta.get("backbone_out_stride", backbone_out_stride)
            t_anchor_assigner = t_meta.get("anchor_assigner", anchor_assigner)
            t_anchor_cls_loss = t_meta.get("anchor_cls_loss", anchor_cls_loss)
            t_simota_topk = int(t_meta.get("simota_topk", simota_topk))
            t_classes = [int(c) for c in t_meta.get("classes", class_ids)]
            t_num_classes = len(t_classes)
            t_use_batchnorm = bool(t_meta.get("use_batchnorm", use_batchnorm))
            teacher_model = build_model(
                t_arch,
                width=t_cnn_width,
                num_classes=t_num_classes,
                use_skip=t_use_skip,
                activation=t_activation,
                use_fpn=t_use_fpn,
                use_anchor=t_use_anchor,
                num_anchors=t_num_anchors,
                anchors=t_anchor_list,
                last_se=t_last_se,
                last_width_scale=t_last_width_scale,
                output_stride=t_output_stride,
                use_improved_head=t_use_improved_head,
                use_iou_aware_head=t_use_iou_aware_head,
                quality_power=t_quality_power,
                utod_head_ese=t_utod_head_ese,
                utod_context_rfb=t_utod_context_rfb,
                utod_context_dilation=t_utod_context_dilation,
                utod_large_obj_branch=t_utod_large_obj_branch,
                utod_large_obj_depth=t_utod_large_obj_depth,
                utod_large_obj_ch_scale=t_utod_large_obj_ch_scale,
                backbone=t_backbone,
                backbone_channels=t_backbone_channels,
                backbone_blocks=t_backbone_blocks,
                backbone_se=t_backbone_se,
                backbone_skip=t_backbone_skip,
                backbone_skip_cat=t_backbone_skip_cat,
                backbone_skip_shuffle_cat=t_backbone_skip_shuffle_cat,
                backbone_skip_s2d_cat=t_backbone_skip_s2d_cat,
                backbone_fpn=t_backbone_fpn,
                backbone_out_stride=t_backbone_out_stride,
                anchor_assigner=t_anchor_assigner,
                anchor_cls_loss=t_anchor_cls_loss,
                simota_topk=t_simota_topk,
                utod_use_residual=t_utod_residual,
                use_batchnorm=t_use_batchnorm,
            ).to(device)
            teacher_model.load_state_dict(t_meta["model"])
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            if t_use_anchor and t_anchor_list and hasattr(teacher_model, "set_anchors"):
                teacher_model.set_anchors(torch.tensor(t_anchor_list, dtype=torch.float32, device=device))
            teacher_use_anchor = t_use_anchor
            teacher_num_classes = t_num_classes
            if hasattr(teacher_model, "anchors"):
                teacher_anchor_tensor = getattr(teacher_model, "anchors")
                if isinstance(teacher_anchor_tensor, torch.Tensor):
                    teacher_anchor_tensor = teacher_anchor_tensor.to(device)
            if arch_cnn_like and distill_feat > 0:
                if teacher_feature_dim is None:
                    teacher_feature_dim = _infer_cnn_feat_channels(teacher_model, use_anchor=t_use_anchor, num_classes=t_num_classes)
                student_channels = _infer_cnn_feat_channels(model, use_anchor=use_anchor, num_classes=num_classes)
                if teacher_feature_dim is not None and feature_adapter is None:
                    if student_channels != teacher_feature_dim:
                        feature_adapter = torch.nn.Conv2d(student_channels, teacher_feature_dim, kernel_size=1).to(device)
                    else:
                        feature_adapter = torch.nn.Identity()
            print(f"Loaded teacher from {teacher_ckpt} (arch={t_arch})")
        else:
            print(f"Teacher arch {t_arch} not supported for distillation (only transformer/cnn/ultratinyod). Skipping teacher.")
    if teacher_model_arch:
        teacher_arch = teacher_model_arch

    if args.val_only:
        sample_dir = os.path.join(run_dir, "val_only")
        ensure_dir(sample_dir)
        val_sample_limit = max(0, int(args.val_count)) if args.val_count is not None else 10
        eval_model = ema_helper.ema if (use_ema and ema_helper is not None) else model
        metrics = validate(
            eval_model,
            val_loader,
            device=device,
            arch=args.arch,
            conf_thresh=args.conf_thresh,
            topk=args.topk,
            iou_thresh=0.5,
            use_amp=use_amp,
            num_classes=num_classes,
            sample_dir=sample_dir,
            class_ids=class_ids,
            sample_limit=val_sample_limit,
            coco_eval=args.coco_eval,
            coco_per_class=args.coco_per_class,
            use_anchor=use_anchor,
            anchors=anchors_tensor,
            iou_loss_type=iou_loss_type,
            anchor_assigner=anchor_assigner,
            anchor_cls_loss=anchor_cls_loss,
            simota_topk=simota_topk,
        )
        fmt_val = {k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in metrics.items()}
        val_msg = f"val-only: {fmt_val}"
        print(val_msg)
        with open(log_path, "a") as f:
            f.write(val_msg + "\n")
        log_scalars(
            writer,
            "val",
            metrics,
            ordered_keys=["mAP@0.5", "AP@0.5_class0", "AP@0.5", "ap@0.5", "map@0.5", "loss", "obj", "cls", "box", "quality", "hm", "off", "wh", "l1", "iou"],
            step=0,
        )
        writer.close()
        return

    for epoch in range(start_epoch, args.epochs):
        train_logs = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            arch=args.arch,
            log_interval=args.log_interval,
            scaler=scaler,
            epoch=epoch,
            total_epochs=args.epochs,
            num_classes=num_classes,
            grad_clip_norm=grad_clip_norm,
            ema=ema_helper,
            teacher_model=teacher_model,
            teacher_backbone=teacher_backbone_model,
            teacher_arch=teacher_arch,
            teacher_use_anchor=teacher_use_anchor,
            teacher_anchors=teacher_anchor_tensor,
            teacher_num_classes=teacher_num_classes,
            feature_adapter=feature_adapter,
            teacher_backbone_norm=teacher_backbone_norm,
            distill_kl=distill_kl,
            distill_box_l1=distill_box_l1,
            distill_cosine=distill_cosine,
            distill_temperature=distill_temperature,
            distill_feat=distill_feat if arch_cnn_like else 0.0,
            use_anchor=use_anchor,
            anchors=anchors_tensor,
            iou_loss_type=iou_loss_type,
            anchor_assigner=anchor_assigner,
            anchor_cls_loss=anchor_cls_loss,
            simota_topk=simota_topk,
        )
        fmt_train = {k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in train_logs.items()}
        train_msg = f"epoch {epoch+1}/{args.epochs} train: {fmt_train}"
        print(train_msg)
        with open(log_path, "a") as f:
            f.write(train_msg + "\n")
        log_scalars(
            writer,
            "train",
            train_logs,
            ordered_keys=["loss", "obj", "cls", "box", "quality", "hm", "off", "wh", "l1", "iou", "distill_feat", "distill_kl", "distill_box_l1"],
            step=epoch + 1,
        )

        metrics = None
        if (epoch + 1) % args.eval_interval == 0:
            epoch_dir = os.path.join(run_dir, f"{epoch+1:04d}")
            ensure_dir(epoch_dir)
            eval_model = ema_helper.ema if ema_helper is not None else model
            metrics = validate(
                eval_model,
                val_loader,
                device=device,
                arch=args.arch,
                conf_thresh=args.conf_thresh,
                topk=args.topk,
                iou_thresh=0.5,
                use_amp=use_amp,
                num_classes=num_classes,
                sample_dir=epoch_dir,
                class_ids=class_ids,
                sample_limit=10,
                coco_eval=args.coco_eval,
                coco_per_class=args.coco_per_class,
                use_anchor=use_anchor,
                anchors=anchors_tensor,
                iou_loss_type=iou_loss_type,
                anchor_assigner=anchor_assigner,
                anchor_cls_loss=anchor_cls_loss,
                simota_topk=simota_topk,
            )
            fmt_val = {k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in metrics.items()}
            val_msg = f"epoch {epoch+1}/{args.epochs} val: {fmt_val}"
            print(val_msg)
            with open(log_path, "a") as f:
                f.write(val_msg + "\n")
            log_scalars(
                writer,
                "val",
                metrics,
                ordered_keys=["mAP@0.5", "AP@0.5_class0", "AP@0.5", "ap@0.5", "map@0.5", "loss", "obj", "cls", "box", "quality", "hm", "off", "wh", "l1", "iou"],
                step=epoch + 1,
            )

            # Save checkpoints: best only when improved mAP; keep latest 10 by recency
            map_val = metrics.get("mAP@0.5", 0.0)
            arch_tag = "utod" if args.arch == "ultratinyod" else ("cnn" if args.arch == "cnn" else "tf")
            if map_val > best_map:
                best_map = map_val
                state_metrics = metrics
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "scheduler": scheduler.state_dict(),
                    "metrics": state_metrics,
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "classes": class_ids,
                    "augment_cfg": aug_cfg,
                    "resize_mode": resize_mode,
                    "use_skip": use_skip,
                    "utod_residual": utod_residual,
                    "use_fpn": use_fpn,
                    "backbone": backbone,
                    "backbone_channels": backbone_channels,
                    "backbone_blocks": backbone_blocks,
                    "backbone_se": backbone_se,
                    "backbone_skip": backbone_skip,
                    "backbone_skip_cat": backbone_skip_cat,
                    "backbone_skip_shuffle_cat": backbone_skip_shuffle_cat,
                    "backbone_skip_s2d_cat": backbone_skip_s2d_cat,
                    "backbone_fpn": backbone_fpn,
                    "backbone_out_stride": backbone_out_stride,
                    "use_anchor": use_anchor,
                    "anchors": anchor_list,
                    "auto_anchors": auto_anchors,
                    "num_anchors": num_anchors,
                    "iou_loss": iou_loss_type,
                    "anchor_assigner": anchor_assigner,
                    "anchor_cls_loss": anchor_cls_loss,
                    "simota_topk": simota_topk,
                    "last_se": last_se,
                    "last_width_scale": last_width_scale,
                    "output_stride": output_stride,
                    "grad_clip_norm": grad_clip_norm,
                    "cnn_width": cnn_width,
                    "use_improved_head": use_improved_head,
                    "use_iou_aware_head": use_iou_aware_head,
                    "quality_power": quality_power,
                    "utod_head_ese": utod_head_ese,
                    "utod_context_rfb": utod_context_rfb,
                    "utod_context_dilation": utod_context_dilation,
                    "utod_large_obj_branch": utod_large_obj_branch,
                    "utod_large_obj_depth": utod_large_obj_depth,
                    "utod_large_obj_ch_scale": utod_large_obj_ch_scale,
                    "activation": activation,
                    "use_batchnorm": use_batchnorm,
                    "best_map": best_map,
                    "use_ema": use_ema,
                    "ema_decay": ema_decay,
                    "ema": ema_helper.ema.state_dict() if ema_helper is not None else None,
                    "ema_updates": ema_helper.updates if ema_helper is not None else 0,
                    "teacher_ckpt": teacher_ckpt,
                    "teacher_arch": teacher_arch,
                    "teacher_num_queries": teacher_num_queries,
                    "teacher_d_model": teacher_d_model,
                    "teacher_heads": teacher_heads,
                    "teacher_layers": teacher_layers,
                    "teacher_dim_feedforward": teacher_dim_feedforward,
                    "teacher_use_skip": teacher_use_skip,
                    "teacher_activation": teacher_activation,
                    "teacher_use_fpn": teacher_use_fpn,
                    "teacher_backbone_norm": teacher_backbone_norm,
                    "distill_kl": distill_kl,
                    "distill_box_l1": distill_box_l1,
                    "distill_temperature": distill_temperature,
                    "distill_cosine": distill_cosine,
                    "teacher_backbone": teacher_backbone,
                    "teacher_backbone_arch": teacher_backbone_arch,
                    "distill_feat": distill_feat,
                    "feat_adapter": feature_adapter.state_dict() if feature_adapter is not None else None,
                }
                best_name = f"best_{arch_tag}_{epoch+1:04d}_map_{map_val:.5f}.pt"
                best_path = os.path.join(run_dir, best_name)
                torch.save(state, best_path)
                _prune_best(run_dir, arch_tag, keep=10)
            else:
                # Remove eval sample dir if not a new best
                _remove_dir(epoch_dir)

        # Save last checkpoint every epoch (keep latest 10)
        state_for_last = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "scheduler": scheduler.state_dict(),
            "metrics": metrics if metrics is not None else {},
            "epoch": epoch + 1,
            "arch": args.arch,
            "classes": class_ids,
            "augment_cfg": aug_cfg,
            "resize_mode": resize_mode,
            "use_skip": use_skip,
            "utod_residual": utod_residual,
            "use_fpn": use_fpn,
            "backbone": backbone,
            "backbone_channels": backbone_channels,
            "backbone_blocks": backbone_blocks,
            "backbone_se": backbone_se,
            "backbone_skip": backbone_skip,
            "backbone_skip_cat": backbone_skip_cat,
            "backbone_skip_shuffle_cat": backbone_skip_shuffle_cat,
            "backbone_skip_s2d_cat": backbone_skip_s2d_cat,
            "backbone_fpn": backbone_fpn,
            "backbone_out_stride": backbone_out_stride,
            "use_anchor": use_anchor,
            "anchors": anchor_list,
            "auto_anchors": auto_anchors,
            "num_anchors": num_anchors,
            "iou_loss": iou_loss_type,
            "anchor_assigner": anchor_assigner,
            "anchor_cls_loss": anchor_cls_loss,
            "simota_topk": simota_topk,
            "last_se": last_se,
            "last_width_scale": last_width_scale,
            "output_stride": output_stride,
            "grad_clip_norm": grad_clip_norm,
            "cnn_width": cnn_width,
            "use_improved_head": use_improved_head,
            "use_iou_aware_head": use_iou_aware_head,
            "quality_power": quality_power,
            "utod_head_ese": utod_head_ese,
            "utod_context_rfb": utod_context_rfb,
            "utod_context_dilation": utod_context_dilation,
            "utod_large_obj_branch": utod_large_obj_branch,
            "utod_large_obj_depth": utod_large_obj_depth,
            "utod_large_obj_ch_scale": utod_large_obj_ch_scale,
            "activation": activation,
            "use_batchnorm": use_batchnorm,
            "best_map": best_map,
            "use_ema": use_ema,
            "ema_decay": ema_decay,
            "ema": ema_helper.ema.state_dict() if ema_helper is not None else None,
            "ema_updates": ema_helper.updates if ema_helper is not None else 0,
            "teacher_ckpt": teacher_ckpt,
            "teacher_arch": teacher_arch,
            "teacher_num_queries": teacher_num_queries,
            "teacher_d_model": teacher_d_model,
            "teacher_heads": teacher_heads,
            "teacher_layers": teacher_layers,
            "teacher_dim_feedforward": teacher_dim_feedforward,
            "teacher_use_skip": teacher_use_skip,
            "teacher_activation": teacher_activation,
            "teacher_use_fpn": teacher_use_fpn,
            "teacher_backbone_norm": teacher_backbone_norm,
            "distill_kl": distill_kl,
            "distill_box_l1": distill_box_l1,
            "distill_temperature": distill_temperature,
            "distill_cosine": distill_cosine,
            "teacher_backbone": teacher_backbone,
            "teacher_backbone_arch": teacher_backbone_arch,
            "distill_feat": distill_feat,
            "feat_adapter": feature_adapter.state_dict() if feature_adapter is not None else None,
        }
        last_name = f"last_{epoch+1:04d}.pt"
        last_path = os.path.join(run_dir, last_name)
        torch.save(state_for_last, last_path)
        _prune_last(run_dir, keep=10)
        _prune_epoch_dirs(run_dir, keep=10)

        scheduler.step()

    writer.close()

if __name__ == "__main__":
    main()
