from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.nn.functional as F

from .losses import box_iou, cxcywh_to_xyxy


def decode_centernet(
    outputs: Dict[str, torch.Tensor], conf_thresh: float = 0.3, topk: int = 50
) -> List[List[Tuple[float, int, torch.Tensor]]]:
    hm = outputs["hm"]
    off = outputs["off"]
    wh = outputs["wh"]
    b, c, h, w = hm.shape
    preds: List[List[Tuple[float, int, torch.Tensor]]] = []
    hm_flat = hm.view(b, -1)  # B, C*H*W
    k = min(topk, hm_flat.shape[1])
    scores, inds = torch.topk(hm_flat, k=k, dim=1)
    for i in range(b):
        boxes_i: List[Tuple[float, int, torch.Tensor]] = []
        for s, idx in zip(scores[i], inds[i]):
            if s < conf_thresh:
                continue
            cls = int(idx // (h * w))
            rem = idx % (h * w)
            y = rem // w
            x = rem % w
            dx = off[i, 0, y, x]
            dy = off[i, 1, y, x]
            pw = wh[i, 0, y, x]
            ph = wh[i, 1, y, x]
            cx = (x.float() + dx) / w
            cy = (y.float() + dy) / h
            bw = pw / w
            bh = ph / h
            boxes_i.append((float(s), cls, torch.stack([cx, cy, bw, bh])))
        preds.append(boxes_i)
    return preds


def decode_detr(
    logits: torch.Tensor, boxes: torch.Tensor, conf_thresh: float = 0.3
) -> List[List[Tuple[float, int, torch.Tensor]]]:
    # logits: Q, B, num_classes+1; boxes: Q, B, 4
    probs = logits.softmax(-1)
    q, bsz, num_classes_bg = probs.shape
    num_classes = num_classes_bg - 1
    preds: List[List[Tuple[float, int, torch.Tensor]]] = []
    for b in range(bsz):
        boxes_i: List[Tuple[float, int, torch.Tensor]] = []
        for qidx in range(q):
            # pick best non-background class
            cls_prob, cls_idx = probs[qidx, b, :-1].max(dim=0)
            sc = cls_prob.item()
            if sc < conf_thresh:
                continue
            boxes_i.append((sc, int(cls_idx.item()), boxes[qidx, b, :].detach().cpu()))
        preds.append(boxes_i)
    return preds


def decode_anchor(
    pred: torch.Tensor,
    anchors: torch.Tensor,
    num_classes: int,
    conf_thresh: float = 0.3,
    nms_thresh: float = 0.5,
    has_quality: bool = False,
    wh_scale: Optional[torch.Tensor] = None,
    score_mode: str = "obj_quality_cls",
    quality_power: float = 1.0,
    multi_label: bool = False,
    cls_logits_override: Optional[torch.Tensor] = None,
    class_offset: int = 0,
) -> List[List[Tuple[float, int, torch.Tensor]]]:
    """
    YOLO-style decoding: pred shape B x (A*(5+Q+C)) x H x W, anchors A x 2 (normalized w,h).
    Q=1 when has_quality=True (obj+quality), otherwise 0.
    """

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
    extra = 1 if has_quality else 0
    pred = pred.view(b, na, 5 + extra + num_classes, h, w).permute(0, 1, 3, 4, 2)
    anchors_dev = anchors.to(device)
    if wh_scale is not None:
        anchors_dev = anchors_dev * wh_scale.to(device)

    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    obj = pred[..., 4].sigmoid()
    quality = pred[..., 5].sigmoid() if has_quality else None
    if cls_logits_override is None:
        cls_logits = pred[..., (5 + extra):]
    else:
        cls_logits = cls_logits_override.view(b, na, num_classes, h, w).permute(0, 1, 3, 4, 2)
    cls = cls_logits.sigmoid()

    gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    gx = gx.view(1, 1, h, w)
    gy = gy.view(1, 1, h, w)

    pred_cx = (tx.sigmoid() + gx) / w
    pred_cy = (ty.sigmoid() + gy) / h
    pw = anchors_dev[:, 0].view(1, na, 1, 1)
    ph = anchors_dev[:, 1].view(1, na, 1, 1)
    act_w, act_h = _activate_wh(tw, th)
    pred_w = pw * act_w
    pred_h = ph * act_h

    preds: List[List[Tuple[float, int, torch.Tensor]]] = []
    score_mode = (score_mode or "obj_quality_cls").lower()
    qp = float(quality_power)
    if quality is not None and qp != 1.0:
        quality = torch.pow(quality.clamp(min=0.0), qp)

    if score_mode == "quality_cls" and quality is not None:
        score_base = quality
    elif score_mode == "obj_cls":
        score_base = obj
    else:
        score_base = obj
        if quality is not None:
            score_base = score_base * quality
    # Fallback to obj when no quality is present to avoid zero scores
    scores = score_base.unsqueeze(-1) * cls  # B x A x H x W x C
    for bi in range(b):
        boxes_i: List[Tuple[float, int, torch.Tensor]] = []
        score_map = scores[bi]
        pred_cx_i = pred_cx[bi]
        pred_cy_i = pred_cy[bi]
        pred_w_i = pred_w[bi]
        pred_h_i = pred_h[bi]

        # reshape handles non-contiguous tensors from broadcasting/permutation safely
        flat_scores = score_map.reshape(-1, num_classes)
        boxes_raw = []
        if multi_label:
            for cls_id in range(num_classes):
                cls_scores = flat_scores[:, cls_id]
                mask = cls_scores >= conf_thresh
                if not mask.any():
                    continue
                idxs = mask.nonzero(as_tuple=False).squeeze(1)
                a_idx = idxs // (h * w)
                rem = idxs % (h * w)
                gy_idx = rem // w
                gx_idx = rem % w
                cx_sel = pred_cx_i[a_idx, gy_idx, gx_idx]
                cy_sel = pred_cy_i[a_idx, gy_idx, gx_idx]
                bw_sel = pred_w_i[a_idx, gy_idx, gx_idx]
                bh_sel = pred_h_i[a_idx, gy_idx, gx_idx]
                sc_sel = cls_scores[mask]
                for sc, cx, cy, bw, bh in zip(sc_sel, cx_sel, cy_sel, bw_sel, bh_sel):
                    boxes_raw.append(
                        (float(sc), int(cls_id + class_offset), torch.stack([cx, cy, bw, bh]).detach().cpu())
                    )
        else:
            max_scores, max_cls = flat_scores.max(dim=1)
            mask = max_scores >= conf_thresh
            if mask.any():
                sel_scores = max_scores[mask]
                sel_cls = max_cls[mask]
                idxs = mask.nonzero(as_tuple=False).squeeze(1)
                a_idx = idxs // (h * w)
                rem = idxs % (h * w)
                gy_idx = rem // w
                gx_idx = rem % w
                cx_sel = pred_cx_i[a_idx, gy_idx, gx_idx]
                cy_sel = pred_cy_i[a_idx, gy_idx, gx_idx]
                bw_sel = pred_w_i[a_idx, gy_idx, gx_idx]
                bh_sel = pred_h_i[a_idx, gy_idx, gx_idx]
                for sc, cls_id, cx, cy, bw, bh in zip(sel_scores, sel_cls, cx_sel, cy_sel, bw_sel, bh_sel):
                    boxes_raw.append(
                        (float(sc), int(cls_id.item() + class_offset), torch.stack([cx, cy, bw, bh]).detach().cpu())
                    )
        if boxes_raw:
            boxes_i = nms_per_class(boxes_raw, iou_thresh=nms_thresh)
        preds.append(boxes_i)
    return preds


def nms_per_class(boxes: List[Tuple[float, int, torch.Tensor]], iou_thresh: float = 0.5) -> List[Tuple[float, int, torch.Tensor]]:
    """Apply per-class NMS on a list of (score, cls, box[cx,cy,w,h]) tuples."""
    if not boxes:
        return []
    out: List[Tuple[float, int, torch.Tensor]] = []
    boxes_by_cls = {}
    for sc, cls, box in boxes:
        boxes_by_cls.setdefault(cls, []).append((sc, box))
    for cls, items in boxes_by_cls.items():
        # sort by score
        items = sorted(items, key=lambda x: x[0], reverse=True)
        keep = []
        while items:
            sc, box = items.pop(0)
            keep.append((sc, cls, box))
            if not items:
                break
            ious = box_iou(box.unsqueeze(0), torch.stack([b for _, b in items], dim=0)).squeeze(0)
            remaining = []
            for (sc2, b2), iou in zip(items, ious):
                if float(iou) < iou_thresh:
                    remaining.append((sc2, b2))
            items = remaining
        out.extend(keep)
    # sort combined by score
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _average_precision(preds, iou_thresh: float, npos: int) -> float:
    if len(preds) == 0 or npos == 0:
        return 0.0
    preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
    tp = []
    fp = []
    for score, box, img_idx, gt_boxes, matched in preds_sorted:
        if gt_boxes.numel() == 0:
            tp.append(0)
            fp.append(1)
            continue
        ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_idx = (ious.max().item(), int(torch.argmax(ious).item()))
        if best_iou >= iou_thresh and not matched[best_idx]:
            matched[best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp_cum = 0
    fp_cum = 0
    precisions: List[float] = []
    recalls: List[float] = []
    for t, f in zip(tp, fp):
        tp_cum += t
        fp_cum += f
        precisions.append(tp_cum / (tp_cum + fp_cum + 1e-8))
        recalls.append(tp_cum / (npos + 1e-8))
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for i in range(len(mrec) - 1):
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def evaluate_map(
    preds: Sequence[Sequence[Tuple[float, int, torch.Tensor]]],
    targets: Sequence[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    # Prepare GTs per image per class
    gt_boxes_by_img = []
    matched_by_img = []
    for tgt in targets:
        boxes = tgt["boxes"].cpu()
        labels = tgt["labels"].cpu()
        per_class = {}
        per_class_matched = {}
        for c in range(num_classes):
            cls_mask = labels == c
            cls_boxes = boxes[cls_mask]
            per_class[c] = cls_boxes
            per_class_matched[c] = [False] * cls_boxes.shape[0]
        gt_boxes_by_img.append(per_class)
        matched_by_img.append(per_class_matched)

    ap_per_class = {}
    for c in range(num_classes):
        class_preds = []
        npos = 0
        for img_idx, tgt in enumerate(targets):
            gt_cls_boxes = gt_boxes_by_img[img_idx][c]
            npos += gt_cls_boxes.shape[0]
        if npos == 0:
            ap_per_class[c] = 0.0
            continue
        # collect predictions of this class
        for img_idx, pred_img in enumerate(preds):
            for score, cls, box in pred_img:
                if cls != c:
                    continue
                class_preds.append((score, box, img_idx, gt_boxes_by_img[img_idx][c], matched_by_img[img_idx][c]))
        if not class_preds:
            ap_per_class[c] = 0.0
            continue
        ap_per_class[c] = _average_precision(class_preds, iou_thresh, npos)

    mAP = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0
    out = {"mAP@0.5": mAP}
    for c, ap in ap_per_class.items():
        out[f"AP@0.5_class{c}"] = ap
    return out
