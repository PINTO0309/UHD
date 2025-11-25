from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device if a.numel() else b.device)
    a_xyxy = cxcywh_to_xyxy(a)
    b_xyxy = cxcywh_to_xyxy(b)
    tl = torch.max(a_xyxy[:, None, :2], b_xyxy[None, :, :2])
    br = torch.min(a_xyxy[:, None, 2:], b_xyxy[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a_xyxy[:, 2] - a_xyxy[:, 0]) * (a_xyxy[:, 3] - a_xyxy[:, 1])
    area_b = (b_xyxy[:, 2] - b_xyxy[:, 0]) * (b_xyxy[:, 3] - b_xyxy[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    pos_mask = target.eq(1.0)
    neg_mask = target.lt(1.0)
    neg_weights = torch.pow(1.0 - target, beta)

    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask

    num_pos = pos_mask.sum()
    if num_pos == 0:
        return -(neg_loss.sum())  # no positives, only negatives
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def build_centernet_targets(
    targets: Sequence[Dict[str, torch.Tensor]], feat_h: int, feat_w: int, num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = len(targets)
    hm = torch.zeros((b, num_classes, feat_h, feat_w))
    off = torch.zeros((b, 2, feat_h, feat_w))
    wh = torch.zeros((b, 2, feat_h, feat_w))
    mask = torch.zeros((b, 1, feat_h, feat_w))

    for i, tgt in enumerate(targets):
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        if boxes.numel() == 0:
            continue
        for box, cls in zip(boxes, labels):
            cls_idx = int(cls.item())
            if cls_idx < 0 or cls_idx >= num_classes:
                continue
            cx, cy, w, h = box.tolist()
            cx *= feat_w
            cy *= feat_h
            cx_int = int(cx)
            cy_int = int(cy)
            if cx_int < 0 or cy_int < 0 or cx_int >= feat_w or cy_int >= feat_h:
                continue
            hm[i, cls_idx, cy_int, cx_int] = 1.0
            off[i, :, cy_int, cx_int] = torch.tensor([cx - cx_int, cy - cy_int])
            wh[i, :, cy_int, cx_int] = torch.tensor([w * feat_w, h * feat_h])
            mask[i, 0, cy_int, cx_int] = 1.0
    return hm, off, wh, mask


def centernet_loss(
    outputs: Dict[str, torch.Tensor], targets: Sequence[Dict[str, torch.Tensor]], num_classes: int
) -> Dict[str, torch.Tensor]:
    hm_pred = outputs["hm"]
    off_pred = outputs["off"]
    wh_pred = outputs["wh"]
    b, _, feat_h, feat_w = hm_pred.shape

    device = hm_pred.device
    hm_t, off_t, wh_t, mask = build_centernet_targets(targets, feat_h, feat_w, num_classes=num_classes)
    hm_t = hm_t.to(device)
    off_t = off_t.to(device)
    wh_t = wh_t.to(device)
    mask = mask.to(device)

    hm_loss = focal_loss(hm_pred, hm_t)
    pos = mask.sum()
    if pos > 0:
        off_loss = F.l1_loss(off_pred * mask, off_t * mask, reduction="sum") / pos
        wh_loss = F.l1_loss(wh_pred * mask, wh_t * mask, reduction="sum") / pos
    else:
        off_loss = torch.tensor(0.0, device=device)
        wh_loss = torch.tensor(0.0, device=device)
    total = hm_loss + off_loss + wh_loss
    return {"loss": total, "hm": hm_loss, "off": off_loss, "wh": wh_loss}


def greedy_match(cost: torch.Tensor) -> List[Tuple[int, int]]:
    """Greedy one-to-one matching for small sizes (gt x queries)."""
    matches: List[Tuple[int, int]] = []
    if cost.numel() == 0:
        return matches
    used_rows = set()
    used_cols = set()
    flat = cost.flatten()
    sorted_idx = torch.argsort(flat)
    num_rows, num_cols = cost.shape
    for idx in sorted_idx:
        r = int(idx // num_cols)
        c = int(idx % num_cols)
        if r in used_rows or c in used_cols:
            continue
        matches.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
        if len(used_rows) == num_rows or len(used_cols) == num_cols:
            break
    return matches


def detr_loss(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
    num_classes: int,
    lambda_cls: float = 1.0,
    lambda_l1: float = 5.0,
    lambda_iou: float = 2.0,
) -> Dict[str, torch.Tensor]:
    # logits: Q, B, C+1; boxes: Q, B, 4 normalized
    q, bsz, _ = logits.shape
    device = logits.device
    cls_losses = []
    l1_losses = []
    iou_losses = []
    for b in range(bsz):
        gt_boxes = targets[b]["boxes"].to(device)
        gt_labels = targets[b]["labels"].to(device)
        logit_b = logits[:, b, :]
        box_b = boxes[:, b, :]
        bg_class = num_classes

        if gt_boxes.numel() == 0:
            cls_target = torch.full((q,), bg_class, dtype=torch.long, device=device)
            cls_losses.append(F.cross_entropy(logit_b, cls_target))
            continue

        cost_l1 = torch.cdist(box_b, gt_boxes, p=1)  # Q x G
        ious = box_iou(box_b, gt_boxes)  # Q x G
        cost = lambda_l1 * cost_l1 + lambda_iou * (1 - ious)
        matches = greedy_match(cost.t())  # list of (gt, query)

        cls_target = torch.full((q,), bg_class, dtype=torch.long, device=device)
        if matches:
            for gt_idx, q_idx in matches:
                cls_target[q_idx] = int(gt_labels[gt_idx].item())
                l1_losses.append(F.l1_loss(box_b[q_idx], gt_boxes[gt_idx]))
                iou_losses.append(1.0 - ious[q_idx, gt_idx])
        cls_losses.append(F.cross_entropy(logit_b, cls_target))

    cls_loss = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device)
    l1_loss = torch.stack(l1_losses).mean() if l1_losses else torch.tensor(0.0, device=device)
    iou_loss = torch.stack(iou_losses).mean() if iou_losses else torch.tensor(0.0, device=device)
    total = lambda_cls * cls_loss + lambda_l1 * l1_loss + lambda_iou * iou_loss
    return {"loss": total, "cls": cls_loss, "l1": l1_loss, "iou": iou_loss}
