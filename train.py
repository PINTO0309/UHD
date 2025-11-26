import argparse
import glob
import os
from typing import Dict, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from uhd.data import YoloDataset, detection_collate
from uhd.losses import centernet_loss, detr_loss
from uhd.metrics import decode_centernet, decode_detr, evaluate_map
from uhd.models import build_model
from uhd.utils import default_device, ensure_dir, move_targets, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Ultra-lightweight detection trainer (CNN/Transformer).")
    parser.add_argument("--arch", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--image-dir", default="data/wholebody34/obj_train_data", help="Directory with images and YOLO txt labels.")
    parser.add_argument("--train-list", default=None, help="Optional list file of images (YOLO style).")
    parser.add_argument("--val-list", default=None, help="Optional list file for validation.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation when val-list is not provided.")
    parser.add_argument(
        "--img-size",
        default="64x64",
        help="Input size as HxW, e.g., 64x64. If single int, applies to both sides.",
    )
    parser.add_argument("--exp-name", default="default", help="Experiment name; logs will be saved under runs/<exp-name>.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="cuda or cpu. Defaults to cuda if available.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=50, help="Top-K for CNN decoding.")
    parser.add_argument("--save-dir", default="outputs", help="Directory to save checkpoints.")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision training.")
    parser.add_argument("--aug-config", default="uhd/aug.yaml", help="Path to YAML file specifying data augmentations.")
    parser.add_argument(
        "--classes",
        default="0",
        help="Comma-separated list of target class ids to train on (e.g., '0,1,3').",
    )
    # CNN params
    parser.add_argument("--cnn-width", type=int, default=32)
    # Transformer params
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=128)
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


def load_aug_config(path: str):
    if not path:
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
        parsed = _parse_best_filename(p)
        if parsed:
            _, _, map_val = parsed
            entries.append((map_val, p))
    entries.sort(key=lambda x: x[0], reverse=True)
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


def make_datasets(args, class_ids, aug_cfg):
    img_h, img_w = parse_img_size(args.img_size)
    train_ds = YoloDataset(
        image_dir=args.image_dir,
        list_path=args.train_list,
        split="train",
        val_split=args.val_split,
        seed=args.seed,
        img_size=(img_h, img_w),
        augment=True,
        class_ids=class_ids,
        augment_cfg=aug_cfg,
    )
    if args.val_list:
        val_ds = YoloDataset(
            image_dir=args.image_dir,
            list_path=args.val_list,
            split="all",
            val_split=0.0,
            seed=args.seed,
            img_size=(img_h, img_w),
            augment=False,
            class_ids=class_ids,
        )
    else:
        val_ds = YoloDataset(
            image_dir=args.image_dir,
            list_path=args.train_list,
            split="val",
            val_split=args.val_split,
            seed=args.seed,
            img_size=(img_h, img_w),
            augment=False,
            class_ids=class_ids,
        )
    return train_ds, val_ds


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
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_hm = 0.0
    total_off = 0.0
    total_wh = 0.0
    total_cls = 0.0
    total_l1 = 0.0
    total_iou = 0.0
    steps = 0
    pbar = tqdm(loader, total=len(loader), desc=f"Train {epoch+1}/{total_epochs}", ncols=100)
    for step, (imgs, targets) in enumerate(pbar):
        imgs = imgs.to(device)
        targets_dev = move_targets(targets, device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16, enabled=scaler.is_enabled()):
            if arch == "cnn":
                outputs = model(imgs)
                loss_dict = centernet_loss(outputs, targets_dev, num_classes=num_classes)
            else:
                logits, box_pred = model(imgs)
                loss_dict = detr_loss(logits, box_pred, targets_dev, num_classes=num_classes)
            loss = loss_dict["loss"]
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        if arch == "cnn":
            total_hm += float(loss_dict["hm"].item())
            total_off += float(loss_dict["off"].item())
            total_wh += float(loss_dict["wh"].item())
        else:
            total_cls += float(loss_dict["cls"].item())
            total_l1 += float(loss_dict["l1"].item())
            total_iou += float(loss_dict["iou"].item())
        steps += 1

        if (step + 1) % log_interval == 0:
            if arch == "cnn":
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
    if arch == "cnn":
        logs.update(
            {
                "hm": total_hm / steps if steps else 0.0,
                "off": total_off / steps if steps else 0.0,
                "wh": total_wh / steps if steps else 0.0,
            }
        )
    else:
        logs.update(
            {
                "cls": total_cls / steps if steps else 0.0,
                "l1": total_l1 / steps if steps else 0.0,
                "iou": total_iou / steps if steps else 0.0,
            }
        )
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
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    sample_count = 0
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

    def render_sample(img_tensor, pred_list, save_path):
        img_np = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        im = Image.fromarray(img_np)
        draw = ImageDraw.Draw(im)
        w, h = im.size
        for score, cls, box in pred_list:
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            color = colors[cls % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label_cls = class_ids[cls] if class_ids and cls < len(class_ids) else cls
            draw.text((x1, y1), f"{label_cls}:{score:.2f}", fill=color)
        im.save(save_path)

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets_cpu = move_targets(targets, torch.device("cpu"))
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                if arch == "cnn":
                    outputs = model(imgs)
                    preds = decode_centernet(outputs, conf_thresh=conf_thresh, topk=topk)
                else:
                    logits, box_pred = model(imgs)
                    preds = decode_detr(logits, box_pred, conf_thresh=conf_thresh)
            # move predictions to CPU for metric computation
            preds_cpu = []
            for p_img in preds:
                preds_cpu.append([(score, box.detach().cpu()) for score, box in p_img])
            all_preds.extend(preds_cpu)
            all_targets.extend(targets_cpu)
            if sample_dir and sample_count < sample_limit:
                for b_idx, pred_img in enumerate(preds):
                    if sample_count >= sample_limit:
                        break
                    filename = os.path.basename(targets[b_idx]["image_id"])
                    stem = os.path.splitext(filename)[0]
                    save_name = f"{sample_count:02d}_" + stem + ".png"
                    save_path = os.path.join(sample_dir, save_name)
                    render_sample(imgs[b_idx], pred_img, save_path)
                    sample_count += 1

    metrics = evaluate_map(all_preds, all_targets, num_classes=num_classes, iou_thresh=iou_thresh)
    return metrics


def main():
    args = parse_args()
    class_ids = parse_classes(args.classes)
    num_classes = len(class_ids)
    aug_cfg = load_aug_config(args.aug_config)
    ckpt_meta = None
    if args.resume:
        ckpt_meta = torch.load(args.resume, map_location="cpu")
        if "classes" in ckpt_meta:
            ckpt_classes = [int(c) for c in ckpt_meta["classes"]]
            if set(ckpt_classes) != set(class_ids):
                print(f"Overriding CLI classes {class_ids} with checkpoint classes {ckpt_classes}")
            class_ids = ckpt_classes
            num_classes = len(class_ids)
        if "augment_cfg" in ckpt_meta:
            aug_cfg = ckpt_meta["augment_cfg"]
    set_seed(args.seed)
    device = default_device(args.device)
    ensure_dir(args.save_dir)
    run_dir = os.path.join("runs", args.exp_name)
    ensure_dir(run_dir)
    log_path = os.path.join(run_dir, "train.log")
    writer = SummaryWriter(log_dir=run_dir)
    use_amp = bool(args.use_amp and device.type == "cuda")

    model = build_model(
        args.arch,
        width=args.cnn_width,
        num_queries=args.num_queries,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)
    start_epoch = 0
    best_map = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_map = float(ckpt.get("metrics", {}).get("mAP@0.5", 0.0))
        print(f"Resumed from {args.resume} at epoch {start_epoch} with best mAP@0.5={best_map:.4f}")

    train_ds, val_ds = make_datasets(args, class_ids, aug_cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
    )

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
        )
        train_msg = f"epoch {epoch+1}/{args.epochs} train: {train_logs}"
        print(train_msg)
        with open(log_path, "a") as f:
            f.write(train_msg + "\n")
        for k, v in train_logs.items():
            writer.add_scalar(f"train/{k}", v, epoch + 1)

        metrics = None
        if (epoch + 1) % args.eval_interval == 0:
            epoch_dir = os.path.join(run_dir, f"{epoch+1:04d}")
            ensure_dir(epoch_dir)
            metrics = validate(
                model,
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
            )
            val_msg = f"epoch {epoch+1}/{args.epochs} val: {metrics}"
            print(val_msg)
            with open(log_path, "a") as f:
                f.write(val_msg + "\n")
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch + 1)

            # Save checkpoints: best top-10 by mAP and last top-10 by recency
            map_val = metrics.get("mAP@0.5", 0.0)
            best_map = max(best_map, map_val)
            arch_tag = "cnn" if args.arch == "cnn" else "tf"
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
            }
            best_name = f"best_{arch_tag}_{epoch+1:04d}_map_{map_val:.5f}.pt"
            best_path = os.path.join(run_dir, best_name)
            torch.save(state, best_path)
            _prune_best(run_dir, arch_tag, keep=10)

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
        }
        last_name = f"last_{epoch+1:04d}.pt"
        last_path = os.path.join(run_dir, last_name)
        torch.save(state_for_last, last_path)
        _prune_last(run_dir, keep=10)

        scheduler.step()

    writer.close()

if __name__ == "__main__":
    main()
