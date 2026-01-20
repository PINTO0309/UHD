from typing import Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


def _bbox_iou_single(
    boxes1: torch.Tensor, boxes2: torch.Tensor, iou_type: str = "iou", eps: float = 1e-7
) -> torch.Tensor:
    """Compute IoU/GIoU/CIoU for aligned pairs of boxes (cxcywh)."""
    x1, y1, x2, y2 = cxcywh_to_xyxy(boxes1).unbind(-1)
    x1g, y1g, x2g, y2g = cxcywh_to_xyxy(boxes2).unbind(-1)

    inter_x1 = torch.max(x1, x1g)
    inter_y1 = torch.max(y1, y1g)
    inter_x2 = torch.min(x2, x2g)
    inter_y2 = torch.min(y2, y2g)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area2 = (x2g - x1g).clamp(min=0) * (y2g - y1g).clamp(min=0)
    union = area1 + area2 - inter_area + eps
    iou = inter_area / union

    if iou_type == "iou":
        return iou

    # enclosing box
    cw = (torch.max(x2, x2g) - torch.min(x1, x1g)).clamp(min=0)
    ch = (torch.max(y2, y2g) - torch.min(y1, y1g)).clamp(min=0)
    c_area = cw * ch + eps

    giou = iou - (c_area - union) / c_area
    if iou_type == "giou":
        return giou

    # CIoU penalty terms
    rho2 = ((boxes1[..., 0] - boxes2[..., 0]) ** 2 + (boxes1[..., 1] - boxes2[..., 1]) ** 2)
    c2 = (cw ** 2 + ch ** 2).clamp(min=eps)
    v = (4 / (math.pi**2)) * torch.pow(torch.atan((boxes1[..., 2] / (boxes1[..., 3] + eps))) - torch.atan((boxes2[..., 2] / (boxes2[..., 3] + eps))), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - rho2 / c2 - alpha * v
    return ciou


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


def anchor_loss(
    pred: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
    anchors: torch.Tensor,
    num_classes: int,
    iou_loss: str = "giou",
    assigner: str = "legacy",
    cls_loss_type: str = "bce",
    simota_topk: int = 10,
    use_quality: bool = False,
    wh_scale: Optional[torch.Tensor] = None,
    multi_label: bool = False,
    loss_weight_box: float = 1.0,
    loss_weight_obj: float = 1.0,
    loss_weight_cls: float = 1.0,
    loss_weight_quality: float = 1.0,
    obj_loss_type: str = "bce",
    obj_target: str = "auto",
) -> Dict[str, torch.Tensor]:
    """
    YOLO-style anchor loss with optional IoU/GIoU/CIoU regression.
    pred: B x (A*(5+C)) x H x W
    anchors: A x 2 (normalized w,h relative to input image size)
    """
    if anchors is None:
        raise ValueError("anchors must be provided for anchor_loss")

    def _activate_wh(tw: torch.Tensor, th: torch.Tensor, max_scale: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        w = F.softplus(tw)
        h = F.softplus(th)
        if max_scale is not None:
            w = torch.clamp(w, max=max_scale)
            h = torch.clamp(h, max=max_scale)
        return w, h

    device = pred.device
    b, _, h, w = pred.shape
    na = anchors.shape[0]
    extra = 1 if use_quality else 0
    pred = pred.view(b, na, 5 + extra + num_classes, h, w).permute(0, 1, 3, 4, 2)
    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    obj_logit = pred[..., 4]
    qual_logit = pred[..., 5] if use_quality else None
    cls_logit = pred[..., (5 + extra):]

    anchors_dev = anchors.to(device)
    if wh_scale is not None:
        anchors_dev = anchors_dev * wh_scale.to(device)
    target_obj = torch.zeros_like(obj_logit)
    target_cls = torch.zeros((b, na, h, w, num_classes), device=device)
    target_box = torch.zeros((b, na, h, w, 4), device=device)
    target_quality = torch.zeros_like(obj_logit) if use_quality else None

    # grid for decoding centers
    gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    gx = gx.view(1, 1, h, w)
    gy = gy.view(1, 1, h, w)

    pred_cx = (tx.sigmoid() + gx) / w
    pred_cy = (ty.sigmoid() + gy) / h
    act_w, act_h = _activate_wh(tw, th)
    pred_w = anchors_dev[:, 0].view(1, na, 1, 1) * act_w
    pred_h = anchors_dev[:, 1].view(1, na, 1, 1) * act_h
    pred_box = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)

    if assigner == "legacy":
        for bi, tgt in enumerate(targets):
            boxes = tgt["boxes"].to(device)
            labels = tgt["labels"].to(device)
            if boxes.numel() == 0:
                continue
            gxs = (boxes[:, 0] * w).clamp(0, w - 1e-3)
            gys = (boxes[:, 1] * h).clamp(0, h - 1e-3)
            gis = gxs.long()
            gjs = gys.long()
            wh = boxes[:, 2:4]
            # anchor matching by IoU on w,h only
            awh = anchors_dev[:, None, :]  # A x 1 x 2
            inter = torch.min(awh, wh[None, :, :]).prod(dim=2)
            union = (awh[:, :, 0] * awh[:, :, 1]) + (wh[None, :, 0] * wh[None, :, 1]) - inter + 1e-7
            anchor_iou = inter / union  # A x G
            best_anchor = anchor_iou.argmax(dim=0)  # G
            for gi, gj, a, box, cls in zip(gis, gjs, best_anchor, boxes, labels):
                if gi < 0 or gj < 0 or gi >= w or gj >= h:
                    continue
                target_obj[bi, a, gj, gi] = 1.0
                target_cls[bi, a, gj, gi, int(cls.item())] = 1.0
                target_box[bi, a, gj, gi] = box
                if use_quality and target_quality is not None:
                    # placeholder, actual IoU will be filled after pred_box computed
                    target_quality[bi, a, gj, gi] = 1.0
    elif assigner == "simota":
        n_pos_total = 0
        for bi, tgt in enumerate(targets):
            boxes = tgt["boxes"].to(device)
            labels = tgt["labels"].to(device)
            if boxes.numel() == 0:
                continue
            # flatten predictions
            pb = pred_box[bi].reshape(-1, 4)  # N x 4
            pb = torch.nan_to_num(pb, nan=0.0, posinf=1e4, neginf=0.0)
            boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
            obj_b = obj_logit[bi].reshape(-1)
            cls_b = cls_logit[bi].reshape(-1, num_classes)
            ious = box_iou(pb, boxes)  # N x G
            assigned = torch.zeros(pb.shape[0], dtype=torch.bool, device=device)
            for gt_idx in range(boxes.shape[0]):
                cls_id = int(labels[gt_idx].item())
                if ious.numel() == 0:
                    continue
                iou_g = torch.nan_to_num(ious[:, gt_idx], nan=0.0, posinf=0.0, neginf=0.0)
                if iou_g.numel() == 0:
                    continue
                topk = min(simota_topk, iou_g.numel())
                if topk == 0:
                    continue
                topk_vals, topk_idx = torch.topk(iou_g, k=topk, dim=0)
                dynamic_k = max(int(topk_vals.sum().item()), 1)
                dynamic_k = min(dynamic_k, topk)
                if dynamic_k < topk:
                    topk_idx = topk_idx[:dynamic_k]
                # build cost: cls + 3*(1-iou)
                cls_target = torch.ones_like(topk_idx, dtype=torch.float32, device=device)
                pred_cls_logit = cls_b[topk_idx, cls_id]
                cls_cost = F.binary_cross_entropy_with_logits(pred_cls_logit, cls_target, reduction="none")
                iou_cost = 1.0 - iou_g[topk_idx]
                cost = cls_cost + 3.0 * iou_cost
                # select lowest cost anchors (dynamic_k)
                order = torch.argsort(cost)
                selected = topk_idx[order[:dynamic_k]]
                for idx in selected:
                    was_assigned = assigned[idx]
                    if was_assigned and not multi_label:
                        continue
                    if not was_assigned:
                        assigned[idx] = True
                        n_pos_total += 1
                        a = int(idx // (h * w))
                        rem = int(idx % (h * w))
                        gj = rem // w
                        gi = rem % w
                        target_obj[bi, a, gj, gi] = 1.0
                        target_box[bi, a, gj, gi] = boxes[gt_idx]
                        if use_quality and target_quality is not None:
                            target_quality[bi, a, gj, gi] = 1.0
                    else:
                        a = int(idx // (h * w))
                        rem = int(idx % (h * w))
                        gj = rem // w
                        gi = rem % w
                    # For IoU-aware heads, keep cls target high (1.0) and let quality carry IoU.
                    cls_target_val = 1.0 if use_quality else iou_g[idx]
                    current = target_cls[bi, a, gj, gi, cls_id]
                    if cls_target_val > current:
                        target_cls[bi, a, gj, gi, cls_id] = cls_target_val
    else:
        raise ValueError(f"Unknown assigner: {assigner}")

    bce_obj = nn.BCEWithLogitsLoss(reduction="mean")
    bce_cls = nn.BCEWithLogitsLoss(reduction="sum")
    quality_loss = torch.tensor(0.0, device=device)

    obj_loss_type = str(obj_loss_type or "bce").lower()
    if obj_loss_type in ("smooth_l1", "smooth-l1"):
        obj_loss_type = "smoothl1"
    if obj_loss_type not in ("bce", "smoothl1"):
        raise ValueError(f"Unknown obj_loss_type: {obj_loss_type}")
    obj_target_mode = str(obj_target or "auto").lower()
    if obj_target_mode not in ("auto", "binary", "iou"):
        raise ValueError(f"Unknown obj_target: {obj_target_mode}")
    loss_weight_box = float(loss_weight_box)
    loss_weight_obj = float(loss_weight_obj)
    loss_weight_cls = float(loss_weight_cls)
    loss_weight_quality = float(loss_weight_quality)

    pos_mask = target_obj > 0.5
    num_pos = int(pos_mask.sum().item())
    iou_val = None
    if num_pos > 0:
        t_box = target_box[pos_mask]
        p_box = pred_box[pos_mask]
        iou_val = _bbox_iou_single(p_box, t_box, iou_type=iou_loss)
        box_loss = (1.0 - iou_val).mean()
        if use_quality and qual_logit is not None:
            if target_quality is None:
                target_quality = torch.zeros_like(obj_logit)
            # clamp IoU to [0, 1] and match dtype to avoid negative BCE loss under CIoU
            tq = iou_val.detach().clamp(min=0.0, max=1.0).to(target_quality.dtype)
            target_quality[pos_mask] = tq
            quality_loss = bce_obj(qual_logit, target_quality)
        if cls_loss_type == "vfl":
            # varifocal: target carries IoU quality; negatives are zero
            t = torch.zeros_like(cls_logit)
            t[pos_mask] = target_cls[pos_mask].to(t.dtype)
            cls_loss = varifocal_loss(cls_logit, t, alpha=0.75, gamma=2.0)
        elif cls_loss_type == "ce":
            if num_classes <= 1:
                cls_loss = bce_cls(cls_logit[pos_mask], target_cls[pos_mask]) / max(1, num_pos)
            else:
                cls_targets = target_cls[pos_mask]
                cls_target_idx = cls_targets.argmax(dim=-1)
                cls_weight = cls_targets.max(dim=-1).values
                ce = F.cross_entropy(cls_logit[pos_mask], cls_target_idx, reduction="none")
                cls_loss = (ce * cls_weight.to(ce.dtype)).sum() / max(1, num_pos)
        else:
            cls_loss = bce_cls(cls_logit[pos_mask], target_cls[pos_mask]) / max(1, num_pos)
    else:
        box_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

    obj_target_tensor = target_obj
    use_iou_for_obj = obj_target_mode == "iou" or (obj_target_mode == "auto" and use_quality)
    if use_iou_for_obj:
        if target_quality is not None:
            obj_target_tensor = target_quality
        elif iou_val is not None and num_pos > 0:
            obj_target_tensor = target_obj.clone()
            obj_target_tensor[pos_mask] = iou_val.detach().clamp(min=0.0, max=1.0).to(obj_target_tensor.dtype)
    if obj_loss_type == "smoothl1":
        obj_pred = obj_logit.sigmoid()
        obj_loss = F.smooth_l1_loss(obj_pred, obj_target_tensor, reduction="mean")
    else:
        obj_loss = bce_obj(obj_logit, obj_target_tensor)

    total = (
        loss_weight_box * box_loss
        + loss_weight_obj * obj_loss
        + loss_weight_cls * cls_loss
        + loss_weight_quality * quality_loss
    )
    return {"loss": total, "box": box_loss, "obj": obj_loss, "cls": cls_loss, "quality": quality_loss}


def anchor_attr_loss(
    attr_logit: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
    anchors: torch.Tensor,
    num_classes: int,
    assigner: str = "legacy",
    simota_topk: int = 10,
    wh_scale: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Attribute loss for anchor head (multi-label BCE on positive anchors).
    attr_logit: B x (A*C) x H x W
    """
    if num_classes <= 0:
        device = attr_logit.device
        zero = torch.tensor(0.0, device=device)
        return {"attr": zero}

    if assigner != "legacy":
        # Attribute assignment uses center-based legacy strategy for now.
        assigner = "legacy"

    device = attr_logit.device
    b, _, h, w = attr_logit.shape
    na = anchors.shape[0]
    attr_logit = attr_logit.view(b, na, num_classes, h, w).permute(0, 1, 3, 4, 2)
    anchors_dev = anchors.to(device)
    if wh_scale is not None:
        anchors_dev = anchors_dev * wh_scale.to(device)
    target_attr = torch.zeros((b, na, h, w, num_classes), device=device)

    for bi, tgt in enumerate(targets):
        boxes = tgt["boxes"].to(device)
        labels = tgt["labels"].to(device)
        if boxes.numel() == 0:
            continue
        gxs = (boxes[:, 0] * w).clamp(0, w - 1e-3)
        gys = (boxes[:, 1] * h).clamp(0, h - 1e-3)
        gis = gxs.long()
        gjs = gys.long()
        wh = boxes[:, 2:4]
        # anchor matching by IoU on w,h only
        awh = anchors_dev[:, None, :]  # A x 1 x 2
        inter = torch.min(awh, wh[None, :, :]).prod(dim=2)
        union = (awh[:, :, 0] * awh[:, :, 1]) + (wh[None, :, 0] * wh[None, :, 1]) - inter + 1e-7
        anchor_iou = inter / union  # A x G
        best_anchor = anchor_iou.argmax(dim=0)  # G
        for gi, gj, a, cls in zip(gis, gjs, best_anchor, labels):
            if gi < 0 or gj < 0 or gi >= w or gj >= h:
                continue
            cls_id = int(cls.item())
            if cls_id < 0 or cls_id >= num_classes:
                continue
            target_attr[bi, a, gj, gi, cls_id] = 1.0

    pos_mask = target_attr.sum(dim=-1) > 0
    num_pos = int(pos_mask.sum().item())
    if num_pos > 0:
        bce = nn.BCEWithLogitsLoss(reduction="sum")
        loss = bce(attr_logit[pos_mask], target_attr[pos_mask]) / max(1, num_pos)
    else:
        loss = torch.tensor(0.0, device=device)
    return {"attr": loss}


def varifocal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """
    Varifocal Loss (adapted from VFNet). pred: logits, target: quality scores (0..1).
    """
    pred_sigmoid = pred.sigmoid()
    weight = target * target + alpha * (1 - target) * torch.pow(pred_sigmoid, gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * weight
    return loss.mean()
