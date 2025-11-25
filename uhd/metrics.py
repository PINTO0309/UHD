from typing import Dict, Iterable, List, Sequence, Tuple

import torch

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
