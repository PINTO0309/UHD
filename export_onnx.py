#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from uhd.ultratinyod import UltraTinyOD, UltraTinyODConfig
from uhd.resize import Y_BIN_RESIZE_MODE, Y_ONLY_RESIZE_MODE, Y_TRI_RESIZE_MODE, YUV422_RESIZE_MODE, normalize_resize_mode


def parse_img_size(arg: str) -> Tuple[int, int]:
    arg = str(arg).lower().replace(" ", "")
    if "x" in arg:
        h, w = arg.split("x")
        return int(float(h)), int(float(w))
    v = int(float(arg))
    return v, v


def _is_state_dict(obj) -> bool:
    return isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values())


def load_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Returns (state_dict, meta)."""
    ckpt = torch.load(path, map_location="cpu")
    meta: Dict = ckpt if isinstance(ckpt, dict) else {}
    state: Optional[Dict[str, torch.Tensor]] = None
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            use_ema = False
            if "ema" in ckpt and ckpt["ema"] is not None:
                if "use_ema" in meta:
                    use_ema = bool(meta.get("use_ema"))
                else:
                    use_ema = True
            state = ckpt["ema"] if use_ema else ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif _is_state_dict(ckpt):
            state = ckpt
    if state is None:
        raise ValueError(f"Unsupported checkpoint format for {path}")
    return state, meta


def infer_utod_config(state: Dict[str, torch.Tensor], meta: Dict, args) -> Tuple[UltraTinyODConfig, Dict]:
    anchors_tensor = state.get("head.anchors", None)
    anchors_from_meta = meta.get("anchors") if isinstance(meta, dict) else None
    anchors: Optional[Sequence[Tuple[float, float]]] = None
    num_anchors = anchors_tensor.shape[0] if anchors_tensor is not None else None
    anchors_source = "default"
    anchors_np = anchors_tensor.cpu().numpy() if anchors_tensor is not None else None
    if anchors_np is not None:
        anchors = anchors_np.tolist()
        num_anchors = anchors_tensor.shape[0]
        anchors_source = "state"
    elif anchors_from_meta:
        anchors = anchors_from_meta
        num_anchors = len(anchors_from_meta)
        anchors_source = "meta"

    # num_classes: prefer head shapes; fall back to metadata
    cls_weight = state.get("head.cls_out.weight")
    disable_cls = bool(meta.get("disable_cls", False)) if isinstance(meta, dict) else False
    if cls_weight is None:
        disable_cls = True
    det_from_state = int(cls_weight.shape[0] // num_anchors) if cls_weight is not None and num_anchors else None
    attr_weight = state.get("head.attr_out.weight")
    attr_from_state = int(attr_weight.shape[0] // num_anchors) if attr_weight is not None and num_anchors else None

    meta_classes = meta.get("classes") if isinstance(meta, dict) else None
    class_ids = [int(c) for c in meta_classes] if meta_classes else []
    multi_label_mode = str(meta.get("multi_label_mode", "none") or "none").lower() if isinstance(meta, dict) else "none"
    det_class_ids = [int(c) for c in meta.get("multi_label_det_classes", [])] if isinstance(meta, dict) else []
    attr_class_ids = [int(c) for c in meta.get("multi_label_attr_classes", [])] if isinstance(meta, dict) else []
    if multi_label_mode == "none" and attr_from_state:
        multi_label_mode = "separate"

    class_to_idx = {cid: i for i, cid in enumerate(class_ids)} if class_ids else {}
    det_class_indices = [class_to_idx[c] for c in det_class_ids if c in class_to_idx] if det_class_ids and class_to_idx else []
    attr_class_indices = [class_to_idx[c] for c in attr_class_ids if c in class_to_idx] if attr_class_ids and class_to_idx else []

    if multi_label_mode == "separate":
        num_classes = det_from_state or (len(det_class_indices) if det_class_indices else len(det_class_ids) if det_class_ids else None)
        if num_classes is None:
            num_classes = len(class_ids) if class_ids else 1
        attr_num_classes = attr_from_state or (len(attr_class_indices) if attr_class_indices else len(attr_class_ids) if attr_class_ids else 0)
    else:
        num_classes = det_from_state or (len(class_ids) if class_ids else 1)
        attr_num_classes = 0
    if disable_cls:
        num_classes = 0
        attr_num_classes = 0
        multi_label_mode = "none"

    # backbone width from meta > weight shape
    if "cnn_width" in meta:
        c_stem = int(meta["cnn_width"])
    else:
        stem_w = state.get("backbone.stem.conv.weight")
        c_stem = int(stem_w.shape[0]) if stem_w is not None else 16

    stride = int(meta.get("output_stride") or 8)

    use_improved_head = bool(
        meta.get("use_improved_head")
        or any(k.startswith("head.quality") for k in state.keys())
    )
    use_iou_aware_head = bool(
        meta.get("use_iou_aware_head")
        or any(k.startswith("head.box_tower") or k.startswith("head.quality_tower") or k.startswith("head.cls_tower") for k in state.keys())
    )
    quality_power = max(0.0, float(meta.get("quality_power", 1.0)))
    score_mode = meta.get("score_mode") if isinstance(meta, dict) else None
    if disable_cls and score_mode:
        mode_l = str(score_mode).lower()
        if mode_l in ("obj_quality_cls", "obj_quality"):
            score_mode = "obj_quality"
        elif mode_l in ("quality_cls", "quality"):
            score_mode = "quality"
        elif mode_l in ("obj_cls", "obj", "cls"):
            score_mode = "obj"
    activation = str(meta.get("activation", "swish") or "swish")
    has_ese_weights = ("head.head_se.fc.weight" in state) and ("head.head_se.fc.bias" in state)
    if meta.get("utod_head_ese") and not has_ese_weights:
        print("[WARN] eSE requested in checkpoint meta but head.head_se weights missing; disabling eSE for export.")
    use_head_ese = bool(has_ese_weights)
    use_residual = bool(meta.get("utod_residual") or "backbone.block3_skip.conv.weight" in state)
    use_context_rfb = bool(meta.get("utod_context_rfb") or any(k.startswith("head.head_rfb") for k in state.keys()))
    context_dilation = int(meta.get("utod_context_dilation", 2))
    use_large_obj_branch = bool(meta.get("utod_large_obj_branch") or any(k.startswith("head.large_obj_") for k in state.keys()))
    large_obj_depth = int(meta.get("utod_large_obj_depth", 1))
    large_obj_ch_scale = float(meta.get("utod_large_obj_ch_scale", 1.0))
    sppf_scale_mode = str(meta.get("utod_sppf_scale", "none") or "none").lower()
    if sppf_scale_mode in ("conv1x1", "1x1", "conv"):
        sppf_scale_mode = "conv"
    elif sppf_scale_mode == "bn":
        sppf_scale_mode = "bn"
    else:
        sppf_scale_mode = "none"
    if sppf_scale_mode == "none":
        if any(k.startswith("backbone.sppf.scale_x.conv.") for k in state.keys()):
            sppf_scale_mode = "conv"
        elif any(k.startswith("backbone.sppf.scale_x.weight") for k in state.keys()):
            sppf_scale_mode = "bn"

    cfg = UltraTinyODConfig(
        num_classes=num_classes,
        attr_num_classes=attr_num_classes,
        anchors=anchors,
        stride=stride,
        use_improved_head=use_improved_head,
        use_head_ese=use_head_ese,
        use_iou_aware_head=use_iou_aware_head,
        quality_power=quality_power,
        score_mode=score_mode,
        disable_cls=disable_cls,
        activation=activation,
        use_context_rfb=use_context_rfb,
        context_dilation=context_dilation,
        use_large_obj_branch=use_large_obj_branch,
        large_obj_branch_depth=large_obj_depth,
        large_obj_branch_expansion=large_obj_ch_scale,
        sppf_scale_mode=sppf_scale_mode,
    )
    overrides = {
        "num_classes": num_classes,
        "anchors": anchors,
        "stride": stride,
        "c_stem": c_stem,
        "use_improved_head": use_improved_head,
        "use_head_ese": use_head_ese,
        "use_residual": use_residual,
        "use_iou_aware_head": use_iou_aware_head,
        "quality_power": quality_power,
        "score_mode": score_mode,
        "disable_cls": disable_cls,
        "anchors_source": anchors_source,
        "activation": activation,
        "multi_label_mode": multi_label_mode,
        "det_class_indices": det_class_indices,
        "attr_class_indices": attr_class_indices,
        "attr_num_classes": attr_num_classes,
        "use_context_rfb": use_context_rfb,
        "context_dilation": context_dilation,
        "use_large_obj_branch": use_large_obj_branch,
        "large_obj_branch_depth": large_obj_depth,
        "large_obj_branch_expansion": large_obj_ch_scale,
        "sppf_scale_mode": sppf_scale_mode,
    }
    return cfg, overrides


class UltraTinyODWithPost(nn.Module):
    def __init__(
        self,
        model: UltraTinyOD,
        topk: int = 100,
        conf_thresh: float = 0.0,
        multi_label_mode: str = "none",
        det_class_indices: Optional[Sequence[int]] = None,
        attr_class_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.topk = int(topk)
        self.conf_thresh = float(conf_thresh)
        self.has_quality = bool(getattr(model.head, "has_quality", False))
        self.score_mode = getattr(model, "score_mode", getattr(model.head, "score_mode", "obj_quality_cls"))
        self.quality_power = float(getattr(model, "quality_power", getattr(model.head, "quality_power", 1.0)))
        self.disable_cls = bool(getattr(model.head, "disable_cls", False))
        self.multi_label_mode = (multi_label_mode or "none").lower()
        self.det_class_indices = list(det_class_indices) if det_class_indices else None
        self.attr_class_indices = list(attr_class_indices) if attr_class_indices else None
        self.use_cls = (not self.disable_cls) and int(getattr(model.head, "nc", 0)) > 0
        self.need_obj, self.need_quality, self.need_cls, self.need_attr = self._infer_needed_branches()

    def _infer_needed_branches(self):
        smode = (self.score_mode or "obj_quality_cls").lower()
        need_quality = self.has_quality and smode in ("obj_quality_cls", "obj_quality", "quality_cls", "quality")
        need_obj = smode in ("obj_quality_cls", "obj_quality", "obj_cls", "obj")
        if smode in ("quality_cls", "quality") and not self.has_quality:
            need_obj = True
        need_cls = self.use_cls and (self.multi_label_mode in ("single", "separate") or smode in ("obj_quality_cls", "quality_cls", "obj_cls", "cls"))
        need_attr = self.multi_label_mode == "separate" and getattr(self.model.head, "attr_out", None) is not None
        return need_obj, need_quality, need_cls, need_attr

    def _score_base(self, obj: torch.Tensor, quality: Optional[torch.Tensor]) -> torch.Tensor:
        quality_use = quality
        if quality_use is not None and self.quality_power != 1.0:
            quality_use = torch.pow(quality_use, self.quality_power)
        smode = (self.score_mode or "obj_quality_cls").lower()
        if smode == "quality_cls" and quality_use is not None:
            score_base = quality_use
        elif smode == "quality" and quality_use is not None:
            score_base = quality_use
        elif smode == "obj_cls":
            score_base = obj
        elif smode == "obj":
            score_base = obj
        elif smode == "cls":
            score_base = torch.ones_like(obj)
        else:
            score_base = obj
            if quality_use is not None:
                score_base = score_base * quality_use
        return score_base

    def _topk_multi_label(
        self,
        scores: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        bw: torch.Tensor,
        bh: torch.Tensor,
        class_map: Optional[Sequence[int]],
    ) -> torch.Tensor:
        b, _, _, _, c = scores.shape
        scores_flat = scores.reshape(b, -1)
        if self.conf_thresh > 0:
            scores_flat = torch.where(scores_flat >= self.conf_thresh, scores_flat, torch.zeros_like(scores_flat))
        k = min(self.topk, scores_flat.shape[1])
        top_scores, top_idx = torch.topk(scores_flat, k=k, dim=1)
        top_cls = top_idx % c
        top_cell = top_idx // c
        cx_flat = cx.reshape(b, -1)
        cy_flat = cy.reshape(b, -1)
        bw_flat = bw.reshape(b, -1)
        bh_flat = bh.reshape(b, -1)
        top_cx = torch.gather(cx_flat, 1, top_cell)
        top_cy = torch.gather(cy_flat, 1, top_cell)
        top_bw = torch.gather(bw_flat, 1, top_cell)
        top_bh = torch.gather(bh_flat, 1, top_cell)
        if class_map is not None:
            cm = torch.tensor(class_map, device=scores.device, dtype=top_cls.dtype)
            top_cls = cm[top_cls]
        detections = torch.stack(
            [top_scores, top_cls.float(), top_cx, top_cy, top_bw, top_bh],
            dim=-1,
        )
        return detections

    # Softplus optimization
    def _softplus(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(x) + torch.relu(-x) # abs
        b = torch.exp(-a)
        c = torch.log(1.0 + b)
        d = torch.relu(x)
        y = d + c
        return y

    def forward(self, x: torch.Tensor):
        feat = self.model.backbone(x)
        box, obj, quality, cls, attr_logits = self.model.head.forward_raw_parts(
            feat,
            need_obj=self.need_obj,
            need_quality=self.need_quality,
            need_cls=self.need_cls,
            need_attr=self.need_attr,
        )

        b, _, h, w = box.shape
        na = self.model.num_anchors
        box_map = box.view(b, na, 4, h, w).permute(0, 1, 3, 4, 2)
        tx = box_map[..., 0]
        ty = box_map[..., 1]
        tw = box_map[..., 2]
        th = box_map[..., 3]
        if obj is None:
            obj_scores = tx.new_zeros((b, na, h, w))
        else:
            obj_scores = obj.view(b, na, h, w).sigmoid()
        if quality is None:
            quality_scores = None
        else:
            quality_scores = quality.view(b, na, h, w).sigmoid()
        if cls is None:
            use_cls = False
            cls_scores = None
        else:
            cls_logits = cls.view(b, na, -1, h, w).permute(0, 1, 3, 4, 2)
            use_cls = self.use_cls and (cls_logits.shape[-1] > 0)
            cls_scores = cls_logits.sigmoid() if use_cls else None

        # grid
        gy, gx = torch.meshgrid(
            torch.arange(h, device=tx.device),
            torch.arange(w, device=tx.device),
            indexing="ij",
        )
        gx = gx.view(1, 1, h, w)
        gy = gy.view(1, 1, h, w)

        anchors = self.model.head.anchors.to(tx.device)
        if self.model.use_improved_head:
            anchors = anchors * self.model.head.wh_scale.to(tx.device)
        pw = anchors[:, 0].view(1, na, 1, 1)
        ph = anchors[:, 1].view(1, na, 1, 1)

        cx = (tx.sigmoid() + gx) / float(w)
        cy = (ty.sigmoid() + gy) / float(h)

        bw: torch.Tensor = pw * self._softplus(tw)
        bh: torch.Tensor = ph * self._softplus(th)

        score_base = self._score_base(obj_scores, quality_scores)
        if use_cls and self.multi_label_mode in ("single", "separate"):
            if self.multi_label_mode == "separate" and attr_logits is not None:
                attr_scores = attr_logits.view(b, na, -1, h, w).permute(0, 1, 3, 4, 2).sigmoid()
                det_scores = score_base.unsqueeze(-1) * cls_scores
                attr_scores = score_base.unsqueeze(-1) * attr_scores
                combined_scores = torch.cat([det_scores, attr_scores], dim=-1)
                class_map = None
                if self.det_class_indices is not None or self.attr_class_indices is not None:
                    det_map = self.det_class_indices or list(range(det_scores.shape[-1]))
                    attr_map = self.attr_class_indices or list(range(attr_scores.shape[-1]))
                    class_map = det_map + attr_map
                return self._topk_multi_label(combined_scores, cx, cy, bw, bh, class_map)
            return self._topk_multi_label(
                score_base.unsqueeze(-1) * cls_scores, cx, cy, bw, bh, self.det_class_indices
            )

        if use_cls:
            scores = score_base.unsqueeze(-1) * cls_scores  # [B, A, H, W, C]
            if self.conf_thresh > 0:
                scores = torch.where(scores >= self.conf_thresh, scores, torch.zeros_like(scores))
            best_scores, best_cls = scores.max(dim=-1)  # [B, A, H, W]

            best_scores_flat = best_scores.view(b, -1)
            best_cls_flat = best_cls.view(b, -1)
            cx_flat = cx.view(b, -1)
            cy_flat = cy.view(b, -1)
            bw_flat = bw.view(b, -1)
            bh_flat = bh.view(b, -1)

            k = min(self.topk, best_scores_flat.shape[1])
            top_scores, top_idx = torch.topk(best_scores_flat, k=k, dim=1)
            top_cls = torch.gather(best_cls_flat, 1, top_idx)
            top_cx = torch.gather(cx_flat, 1, top_idx)
            top_cy = torch.gather(cy_flat, 1, top_idx)
            top_bw = torch.gather(bw_flat, 1, top_idx)
            top_bh = torch.gather(bh_flat, 1, top_idx)
            if self.det_class_indices is not None:
                cm = torch.tensor(self.det_class_indices, device=top_cls.device, dtype=top_cls.dtype)
                top_cls = cm[top_cls]
        else:
            scores_flat = score_base.view(b, -1)
            if self.conf_thresh > 0:
                scores_flat = torch.where(scores_flat >= self.conf_thresh, scores_flat, torch.zeros_like(scores_flat))
            k = min(self.topk, scores_flat.shape[1])
            top_scores, top_idx = torch.topk(scores_flat, k=k, dim=1)
            top_cls = torch.zeros_like(top_scores, dtype=torch.float32, device=top_scores.device)
            cx_flat = cx.view(b, -1)
            cy_flat = cy.view(b, -1)
            bw_flat = bw.view(b, -1)
            bh_flat = bh.view(b, -1)
            top_cx = torch.gather(cx_flat, 1, top_idx)
            top_cy = torch.gather(cy_flat, 1, top_idx)
            top_bw = torch.gather(bw_flat, 1, top_idx)
            top_bh = torch.gather(bh_flat, 1, top_idx)

        detections = torch.stack(
            [top_scores, top_cls.float(), top_cx, top_cy, top_bw, top_bh],
            dim=-1,
        )
        return detections


class UltraTinyODPrimitivePost(nn.Module):
    """
    Primitive postprocess that avoids topk/where/min/max/stack/cat by returning
    decoded boxes and score maps without filtering.
    """

    def __init__(
        self,
        model: UltraTinyOD,
        multi_label_mode: str = "none",
    ) -> None:
        super().__init__()
        self.model = model
        self.has_quality = bool(getattr(model.head, "has_quality", False))
        self.score_mode = getattr(model, "score_mode", getattr(model.head, "score_mode", "obj_quality_cls"))
        self.quality_power = float(getattr(model, "quality_power", getattr(model.head, "quality_power", 1.0)))
        self.disable_cls = bool(getattr(model.head, "disable_cls", False))
        self.multi_label_mode = (multi_label_mode or "none").lower()
        self.use_cls = (not self.disable_cls) and int(getattr(model.head, "nc", 0)) > 0
        self.has_attr = getattr(model.head, "attr_out", None) is not None
        self.need_obj, self.need_quality, self.need_cls, self.need_attr = self._infer_needed_branches()

    def _infer_needed_branches(self):
        smode = (self.score_mode or "obj_quality_cls").lower()
        need_quality = self.has_quality and smode in ("obj_quality_cls", "obj_quality", "quality_cls", "quality")
        need_obj = smode in ("obj_quality_cls", "obj_quality", "obj_cls", "obj")
        if smode in ("quality_cls", "quality") and not self.has_quality:
            need_obj = True
        need_cls = self.use_cls and (self.multi_label_mode in ("single", "separate") or smode in ("obj_quality_cls", "quality_cls", "obj_cls", "cls"))
        need_attr = self.multi_label_mode == "separate" and self.has_attr
        return need_obj, need_quality, need_cls, need_attr

    def _score_base(self, obj: torch.Tensor, quality: Optional[torch.Tensor]) -> torch.Tensor:
        quality_use = quality
        if quality_use is not None and self.quality_power != 1.0:
            quality_use = torch.pow(quality_use, self.quality_power)
        smode = (self.score_mode or "obj_quality_cls").lower()
        if smode == "quality_cls" and quality_use is not None:
            score_base = quality_use
        elif smode == "quality" and quality_use is not None:
            score_base = quality_use
        elif smode == "obj_cls":
            score_base = obj
        elif smode == "obj":
            score_base = obj
        elif smode == "cls":
            score_base = torch.ones_like(obj)
        else:
            score_base = obj
            if quality_use is not None:
                score_base = score_base * quality_use
        return score_base

    def _softplus(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(x) + torch.relu(-x) # abs
        b = torch.exp(-a)
        c = torch.log(1.0 + b)
        d = torch.relu(x)
        y = d + c
        return y

    def forward(self, x: torch.Tensor):
        feat = self.model.backbone(x)
        box, obj, quality, cls, attr = self.model.head.forward_raw_parts(
            feat,
            need_obj=self.need_obj,
            need_quality=self.need_quality,
            need_cls=self.need_cls,
            need_attr=self.need_attr,
        )
        b, _, h, w = box.shape
        na = self.model.num_anchors
        box_map = box.view(b, na, 4, h, w).permute(0, 1, 3, 4, 2)
        tx = box_map[..., 0]
        ty = box_map[..., 1]
        tw = box_map[..., 2]
        th = box_map[..., 3]
        obj_scores = obj.view(b, na, h, w).sigmoid() if obj is not None else tx.new_zeros((b, na, h, w))
        quality_scores = quality.view(b, na, h, w).sigmoid() if quality is not None else None
        if cls is not None:
            cls_logits = cls.view(b, na, -1, h, w).permute(0, 1, 3, 4, 2)
            cls_scores = cls_logits.sigmoid()
        else:
            cls_scores = None
        if attr is not None:
            attr_scores = attr.view(b, na, -1, h, w).permute(0, 1, 3, 4, 2).sigmoid()
        else:
            attr_scores = None

        gx = torch.arange(w, device=tx.device).view(1, 1, 1, w).expand(1, 1, h, w)
        gy = torch.arange(h, device=tx.device).view(1, 1, h, 1).expand(1, 1, h, w)

        anchors = self.model.head.anchors.to(tx.device)
        if self.model.use_improved_head:
            anchors = anchors * self.model.head.wh_scale.to(tx.device)
        pw = anchors[:, 0].view(1, na, 1, 1)
        ph = anchors[:, 1].view(1, na, 1, 1)

        cx = (tx.sigmoid() + gx) / float(w)
        cy = (ty.sigmoid() + gy) / float(h)
        bw = pw * self._softplus(tw)
        bh = ph * self._softplus(th)

        score_base = self._score_base(obj_scores, quality_scores)

        outputs = (cx, cy, bw, bh, score_base)
        if cls_scores is not None:
            outputs = outputs + (cls_scores,)
        if attr_scores is not None:
            outputs = outputs + (attr_scores,)
        return outputs


class UltraTinyODRawWithAnchors(nn.Module):
    """
    Wrap UltraTinyOD to export raw logits together with anchors/wh_scale so post-process can be done externally.
    """

    def __init__(self, model: UltraTinyOD) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        if getattr(self.model.head, "attr_out", None) is not None:
            out = self.model(x, decode=False, return_attr=True)
            raw, attr = out if isinstance(out, (tuple, list)) else (out, None)
        else:
            out = self.model(x, decode=False)
            raw = out[0] if isinstance(out, (tuple, list)) else out
            attr = None
        anchors = self.model.head.anchors
        wh_scale = self.model.head.wh_scale
        if attr is None:
            return raw, anchors, wh_scale
        return raw, attr, anchors, wh_scale


class UltraTinyODRawConcatSlimWithAnchors(nn.Module):
    """
    Export a slim raw-concat tensor that includes only the branches required by score_mode.
    """

    def __init__(self, model: UltraTinyOD, score_mode: Optional[str] = None, multi_label_mode: str = "none") -> None:
        super().__init__()
        self.model = model
        self.score_mode = score_mode or getattr(model, "score_mode", getattr(model.head, "score_mode", "obj_quality_cls"))
        self.multi_label_mode = (multi_label_mode or "none").lower()
        self.has_quality = bool(getattr(model.head, "has_quality", False))
        self.disable_cls = bool(getattr(model.head, "disable_cls", False))
        self.use_cls = (not self.disable_cls) and int(getattr(model.head, "nc", 0)) > 0
        self.has_attr = getattr(model.head, "attr_out", None) is not None
        self.need_obj, self.need_quality, self.need_cls, self.need_attr = self._infer_needed_branches()
        self.raw_concat_fields = ["box"]
        if self.need_obj:
            self.raw_concat_fields.append("obj")
        if self.need_quality:
            self.raw_concat_fields.append("quality")
        if self.need_cls:
            self.raw_concat_fields.append("cls")

    def _infer_needed_branches(self):
        smode = (self.score_mode or "obj_quality_cls").lower()
        need_quality = self.has_quality and smode in ("obj_quality_cls", "obj_quality", "quality_cls", "quality")
        need_obj = smode in ("obj_quality_cls", "obj_quality", "obj_cls", "obj")
        if smode in ("quality_cls", "quality") and not self.has_quality:
            need_obj = True
        need_cls = self.use_cls and (self.multi_label_mode in ("single", "separate") or smode in ("obj_quality_cls", "quality_cls", "obj_cls", "cls"))
        need_attr = self.multi_label_mode == "separate" and self.has_attr
        return need_obj, need_quality, need_cls, need_attr

    def forward(self, x: torch.Tensor):
        feat = self.model.backbone(x)
        box, obj, quality, cls, attr = self.model.head.forward_raw_parts(
            feat,
            need_obj=self.need_obj,
            need_quality=self.need_quality,
            need_cls=self.need_cls,
            need_attr=self.need_attr,
        )
        b, _, h, w = box.shape
        na = self.model.num_anchors
        parts = [box.view(b, na, 4, h, w)]
        if self.need_obj and obj is not None:
            parts.append(obj.view(b, na, 1, h, w))
        if self.need_quality and quality is not None:
            parts.append(quality.view(b, na, 1, h, w))
        if self.need_cls and cls is not None:
            parts.append(cls.view(b, na, -1, h, w))
        raw = torch.cat(parts, dim=2).permute(0, 1, 3, 4, 2).reshape(b, -1, h, w)
        anchors = self.model.head.anchors
        wh_scale = self.model.head.wh_scale
        if self.need_attr and attr is not None:
            return raw, attr, anchors, wh_scale
        return raw, anchors, wh_scale


class UltraTinyODRawPartsWithAnchors(nn.Module):
    """
    Export raw head branches without concatenating box/obj/quality/cls.
    """

    def __init__(self, model: UltraTinyOD, score_mode: Optional[str] = None, multi_label_mode: str = "none") -> None:
        super().__init__()
        self.model = model
        self.score_mode = score_mode or getattr(model, "score_mode", getattr(model.head, "score_mode", "obj_quality_cls"))
        self.multi_label_mode = (multi_label_mode or "none").lower()
        self.has_quality = bool(getattr(model.head, "has_quality", False))
        self.disable_cls = bool(getattr(model.head, "disable_cls", False))
        self.use_cls = (not self.disable_cls) and int(getattr(model.head, "nc", 0)) > 0
        self.has_attr = getattr(model.head, "attr_out", None) is not None
        self.need_obj, self.need_quality, self.need_cls, self.need_attr = self._infer_needed_branches()

    def _infer_needed_branches(self):
        smode = (self.score_mode or "obj_quality_cls").lower()
        need_quality = self.has_quality and smode in ("obj_quality_cls", "obj_quality", "quality_cls", "quality")
        need_obj = smode in ("obj_quality_cls", "obj_quality", "obj_cls", "obj")
        if smode in ("quality_cls", "quality") and not self.has_quality:
            need_obj = True
        need_cls = self.use_cls and (self.multi_label_mode in ("single", "separate") or smode in ("obj_quality_cls", "quality_cls", "obj_cls", "cls"))
        need_attr = self.multi_label_mode == "separate" and self.has_attr
        return need_obj, need_quality, need_cls, need_attr

    def forward(self, x: torch.Tensor):
        feat = self.model.backbone(x)
        box, obj, quality, cls, attr = self.model.head.forward_raw_parts(
            feat,
            need_obj=self.need_obj,
            need_quality=self.need_quality,
            need_cls=self.need_cls,
            need_attr=self.need_attr,
        )
        anchors = self.model.head.anchors
        wh_scale = self.model.head.wh_scale
        outputs = [box]
        if self.need_obj and obj is not None:
            outputs.append(obj)
        if self.need_quality and quality is not None:
            outputs.append(quality)
        if self.need_cls and cls is not None:
            outputs.append(cls)
        if self.need_attr and attr is not None:
            outputs.append(attr)
        outputs.extend([anchors, wh_scale])
        return tuple(outputs)


def _resize_attrs_from_mode(resize_mode: str):
    rm = normalize_resize_mode(resize_mode)
    if rm == "torch_bilinear":
        return "linear", "half_pixel", "floor"
    if rm == "torch_nearest":
        return "nearest", "asymmetric", "floor"
    if rm == "opencv_inter_linear":
        return "linear", "asymmetric", "floor"
    return "nearest", "asymmetric", "floor"


def add_input_resize(
    onnx_path: str,
    target_size: Tuple[int, int],
    input_name: str = "input_rgb",
    resize_mode: str = "torch_nearest",
    input_channels: int = 3,
) -> None:
    """
    Reload an ONNX model, make input dynamic, and insert a Resize to the fixed target_size
    at the head of the graph. Overwrites the ONNX file in-place.
    """
    import onnx
    from onnx import helper, numpy_helper

    mode_attr, ctm_attr, nearest_attr = _resize_attrs_from_mode(resize_mode)
    model = onnx.load(onnx_path, load_external_data=False)
    g = model.graph
    inp = None
    for i in g.input:
        if i.name == input_name:
            inp = i
            break
    if inp is None:
        raise ValueError(f"Input '{input_name}' not found in ONNX graph.")
    dims = inp.type.tensor_type.shape.dim
    if len(dims) < 4:
        raise ValueError(f"Input '{input_name}' has unexpected rank: {len(dims)}")
    # dynamic batch/H/W (avoid setting dim_value=0 which may cause ORT Resize failure)
    dims[2].dim_param = dims[2].dim_param or "H"
    if not dims[2].HasField("dim_value") or dims[2].dim_value == 0:
        dims[2].ClearField("dim_value")
    dims[3].dim_param = dims[3].dim_param or "W"
    if not dims[3].HasField("dim_value") or dims[3].dim_value == 0:
        dims[3].ClearField("dim_value")

    target_h, target_w = target_size
    sizes = numpy_helper.from_array(np.array([1, input_channels, target_h, target_w], dtype=np.int64), name=f"{input_name}_sizes")
    g.initializer.append(sizes)

    resize_out = f"{input_name}_resized"
    resize_node = helper.make_node(
        "Resize",
        inputs=[input_name, "", "", sizes.name],
        outputs=[resize_out],
        mode=mode_attr,
        coordinate_transformation_mode=ctm_attr,
        cubic_coeff_a=-0.75,
        nearest_mode=nearest_attr,
        name="InputResize",
    )
    g.node.insert(0, resize_node)
    for node in g.node[1:]:
        node.input[:] = [resize_out if inp_name == input_name else inp_name for inp_name in node.input]

    model.opset_import[0].version = max(model.opset_import[0].version, 11)
    onnx.save(model, onnx_path)


def export_onnx(
    model: nn.Module,
    output_path: str,
    img_size: Tuple[int, int],
    opset: int,
    simplify: bool = True,
    output_names: Optional[Sequence[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    input_channels: int = 3,
    input_name: str = "input_rgb",
) -> None:
    model.eval()
    h, w = img_size
    dummy = torch.zeros(1, input_channels, h, w, device=next(model.parameters()).device)
    input_names = [input_name]
    if output_names is None:
        output_names = ["score_classid_cxcywh"]
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    if simplify:
        simplify_onnx_path(output_path)
    rename_depthwise_conv_nodes(output_path)


def simplify_onnx_path(output_path: str) -> bool:
    """Run onnx-simplifier on an ONNX file, in-place. Returns True on success."""
    try:
        import onnx
        from onnxsim import simplify as onnx_simplify
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[WARN] Failed to import onnx/onnxsim for simplification: {exc}")
        return False
    model_onnx = onnx.load(output_path)
    model_simplified, check = onnx_simplify(model_onnx)
    if not check:
        print("[WARN] onnx-simplifier check failed; keeping original export.")
        return False
    onnx.save(model_simplified, output_path)
    return True


def rename_depthwise_conv_nodes(onnx_path: str, prefix: str = "/depthwiseconv") -> bool:
    try:
        import onnx
        from onnx import numpy_helper
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[WARN] Failed to import onnx for depthwise rename: {exc}")
        return False
    try:
        model = onnx.load(onnx_path)
    except Exception as exc:  # pragma: no cover - IO failure
        print(f"[WARN] Failed to load ONNX for depthwise rename: {exc}")
        return False
    init_by_name = {init.name: init for init in model.graph.initializer}
    existing_names = {node.name for node in model.graph.node if node.name}
    prefix = prefix.rstrip("/")
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    changed = 0
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        groups = int(attrs.get("group", 1) or 1)
        if groups <= 1:
            continue
        if len(node.input) < 2:
            continue
        weight_init = init_by_name.get(node.input[1])
        if weight_init is None:
            continue
        weight = numpy_helper.to_array(weight_init)
        if weight.ndim != 4 or weight.shape[1] != 1:
            continue
        if node.name and (node.name == prefix or node.name.startswith(prefix + "/")):
            continue
        base = node.name or (node.output[0] if node.output else f"conv_{idx}")
        base = base.lstrip("/")
        candidate = f"{prefix}/{base}"
        unique = candidate
        suffix = 1
        while unique in existing_names:
            unique = f"{candidate}_{suffix}"
            suffix += 1
        node.name = unique
        existing_names.add(unique)
        changed += 1
    if not changed:
        return False
    onnx.save(model, onnx_path)
    print(f"Updated ONNX: depthwise conv node prefixes: {changed}")
    return True


def update_metadata_props(output_path: str, props: Dict[str, str]) -> None:
    """Add/update metadata_props on an ONNX file; silently skip if onnx is unavailable."""
    try:
        import onnx
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[WARN] Failed to import onnx for metadata update: {exc}")
        return
    try:
        model = onnx.load(output_path, load_external_data=False)
    except Exception as exc:  # pragma: no cover - IO failure
        print(f"[WARN] Failed to load ONNX for metadata update: {exc}")
        return
    kv = {p.key: p.value for p in model.metadata_props}
    kv.update({str(k): str(v) for k, v in props.items() if v is not None})
    model.metadata_props.clear()
    for k, v in kv.items():
        prop = model.metadata_props.add()
        prop.key = k
        prop.value = v
    onnx.save(model, output_path)


def _normalize_score_mode(mode: str, disable_cls: bool) -> str:
    if not mode:
        return ""
    mode_l = str(mode).lower()
    if disable_cls:
        if mode_l in ("obj_quality_cls", "obj_quality"):
            return "obj_quality"
        if mode_l in ("quality_cls", "quality"):
            return "quality"
        if mode_l in ("obj_cls", "obj", "cls"):
            return "obj"
    if mode_l in ("obj_quality_cls", "quality_cls", "obj_cls", "obj_quality", "quality", "obj", "cls"):
        return mode_l
    return "obj_quality_cls"


def _describe_decode_score(mode: str, quality_power: float) -> str:
    if not mode:
        mode = "obj_quality_cls"
    mode_l = str(mode).lower()
    qp = float(quality_power)
    qp_str = f"^{qp:g}" if abs(qp - 1.0) > 1e-6 else ""
    q_term = f"sigmoid(quality){qp_str}"
    if mode_l == "quality_cls":
        return f"score = {q_term} * sigmoid(cls)"
    if mode_l == "quality":
        return f"score = {q_term}"
    if mode_l == "obj_cls":
        return "score = sigmoid(obj) * sigmoid(cls)"
    if mode_l == "obj":
        return "score = sigmoid(obj)"
    if mode_l == "cls":
        return "score = sigmoid(cls)"
    if mode_l == "obj_quality":
        return f"score = sigmoid(obj) * {q_term}"
    return f"score = sigmoid(obj) * {q_term} * sigmoid(cls)"


def verify_outputs(
    model: UltraTinyODWithPost,
    onnx_path: str,
    img_size: Tuple[int, int],
    input_channels: int = 3,
    input_name: str = "input_rgb",
) -> Dict[str, float]:
    import numpy as np
    import onnxruntime as ort

    h, w = img_size
    torch.manual_seed(0)
    sample = torch.randn(1, input_channels, h, w)
    with torch.no_grad():
        torch_out = model(sample)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {input_name: sample.numpy()})

    deltas = {}
    ref = torch_out.detach().cpu().numpy()
    o_out = ort_outs[0]
    scores_ref = ref[..., 0]
    scores_onnx = o_out[..., 0]
    active_mask = (scores_ref > 1e-3) & (scores_onnx > 1e-3)
    full_delta = float(np.max(np.abs(ref - o_out)))
    if active_mask.any():
        active_delta = float(np.max(np.abs(ref[active_mask] - o_out[active_mask])))
    else:
        active_delta = full_delta
    deltas["detections"] = full_delta
    deltas["detections_active"] = active_delta
    return deltas


def build_argparser():
    parser = argparse.ArgumentParser(description="Export UltraTinyOD to ONNX with merged postprocess.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Training checkpoint (.pt) containing optimizer/ema/etc.")
    src.add_argument("--weights", help="Weights/state_dict only (.pt).")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--topk", type=int, default=100, help="Number of detections to keep (Top-K).")
    parser.add_argument("--conf-thresh", type=float, default=0.0, help="Optional confidence threshold before Top-K.")
    parser.add_argument("--no-merge-postprocess", dest="merge_postprocess", action="store_false", help="Export raw model only.")
    parser.set_defaults(merge_postprocess=True)
    parser.add_argument(
        "--merge-primitive-postprocess",
        action="store_true",
        help="Merge a primitive postprocess that avoids min/max/where/topk/stack/cat and returns decoded maps.",
    )
    parser.add_argument(
        "--noconcat_box_obj_quality_cls",
        action="store_true",
        help="When exporting raw model, do not concatenate box/obj/quality/cls into a single tensor.",
    )
    parser.add_argument("--no-simplify", action="store_true", help="Skip onnx-simplifier.")
    parser.add_argument("--non-strict", action="store_true", help="Load weights with strict=False.")
    parser.add_argument("--verify", action="store_true", help="Run a quick ONNXRuntime vs PyTorch diff check.")
    parser.add_argument(
        "--dynamic-resize",
        action="store_true",
        help="Add a Resize op at the graph head and make input dynamic after export/simplify.",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.noconcat_box_obj_quality_cls and args.merge_postprocess:
        parser.error("--noconcat_box_obj_quality_cls requires --no-merge-postprocess.")
    if args.merge_primitive_postprocess and not args.merge_postprocess:
        parser.error("--merge-primitive-postprocess cannot be used with --no-merge-postprocess.")
    if args.merge_primitive_postprocess and args.noconcat_box_obj_quality_cls:
        parser.error("--merge-primitive-postprocess cannot be used with --noconcat_box_obj_quality_cls.")

    ckpt_path = args.checkpoint or args.weights
    state, meta = load_checkpoint(ckpt_path)
    try:
        resize_mode = normalize_resize_mode(meta.get("resize_mode", "opencv_inter_nearest"))
    except Exception:
        print("[WARN] resize_mode missing or invalid in checkpoint meta; defaulting to opencv_inter_nearest.")
        resize_mode = "opencv_inter_nearest"
    if resize_mode == YUV422_RESIZE_MODE:
        input_channels = 2
        input_name = "input_yuv422"
    elif resize_mode in (Y_ONLY_RESIZE_MODE, Y_BIN_RESIZE_MODE, Y_TRI_RESIZE_MODE):
        input_channels = 1
        input_name = "input_y"
    else:
        input_channels = 3
        input_name = "input_rgb"
    cfg, inferred = infer_utod_config(state, meta, args)
    multi_label_mode = inferred.get("multi_label_mode", "none")
    det_class_indices = inferred.get("det_class_indices") or None
    attr_class_indices = inferred.get("attr_class_indices") or None

    model = UltraTinyOD(
        num_classes=inferred["num_classes"],
        config=cfg,
        c_stem=inferred["c_stem"],
        in_channels=input_channels,
        use_residual=inferred["use_residual"],
        use_improved_head=inferred["use_improved_head"],
        use_head_ese=inferred["use_head_ese"],
        use_iou_aware_head=inferred.get("use_iou_aware_head", False),
        quality_power=inferred.get("quality_power", 1.0),
        activation=inferred.get("activation", "swish"),
    )
    try:
        model.load_state_dict(state, strict=not args.non_strict)
    except RuntimeError as e:
        if "head.head_se" in str(e):
            print("[WARN] head_se weights missing in checkpoint; loading non-strict for eSE and keeping initializer.")
            model.load_state_dict(state, strict=False)
        else:
            raise
    model.eval()

    # Ensure anchors in the model match the chosen source (state/override/meta)
    if inferred.get("anchors") is not None:
        anchors_tensor = torch.as_tensor(inferred["anchors"], dtype=torch.float32)
        if anchors_tensor.ndim == 1:
            anchors_tensor = anchors_tensor.view(-1, 2)
        if anchors_tensor.ndim == 2 and anchors_tensor.shape[1] == 2:
            model.head.set_anchors(anchors_tensor)

    if args.merge_primitive_postprocess:
        export_module = UltraTinyODPrimitivePost(
            model,
            multi_label_mode=multi_label_mode,
        )
        output_names = ["cx", "cy", "bw", "bh", "score"]
        if getattr(export_module, "need_cls", False):
            output_names.append("cls_scores")
        if getattr(export_module, "need_attr", False):
            output_names.append("attr_scores")
    elif not args.merge_postprocess:
        if args.noconcat_box_obj_quality_cls:
            export_module = UltraTinyODRawPartsWithAnchors(model, multi_label_mode=multi_label_mode)
            output_names = ["box"]
            if export_module.need_obj:
                output_names.append("obj")
            if export_module.need_quality:
                output_names.append("quality")
            if export_module.need_cls:
                output_names.append("cls")
            if export_module.need_attr:
                output_names.append("attr")
            output_names.extend(["anchors", "wh_scale"])
        else:
            export_module = UltraTinyODRawConcatSlimWithAnchors(model, multi_label_mode=multi_label_mode)
            disable_cls = bool(getattr(model.head, "disable_cls", False))
            raw_tag = "txtywh_" + "_".join(getattr(export_module, "raw_concat_fields", ["box"])) + "_x8"
            if getattr(model.head, "attr_out", None) is not None and getattr(export_module, "need_attr", False):
                output_names = [raw_tag, "attr_logits", "anchors", "wh_scale"]
            else:
                if disable_cls:
                    output_names = [
                        raw_tag,
                        "anchors",
                        "wh_scale",
                    ]
                else:
                    output_names = [raw_tag, "anchors", "wh_scale"]
    else:
        export_module = UltraTinyODWithPost(
            model,
            topk=args.topk,
            conf_thresh=args.conf_thresh,
            multi_label_mode=multi_label_mode,
            det_class_indices=det_class_indices,
            attr_class_indices=attr_class_indices,
        )
        output_names = ["score_classid_cxcywh"]

    device = torch.device("cpu")
    img_size_meta = meta.get("img_size") if isinstance(meta, dict) else None
    if img_size_meta is None:
        print("[WARN] img_size missing in checkpoint meta; defaulting to 64x64.")
        img_size = parse_img_size("64x64")
    elif isinstance(img_size_meta, (tuple, list)) and len(img_size_meta) == 2:
        img_size = (int(img_size_meta[0]), int(img_size_meta[1]))
    else:
        img_size = parse_img_size(str(img_size_meta))
    export_module.to(device)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    export_onnx(
        export_module,
        args.output,
        img_size=img_size,
        opset=int(args.opset),
        simplify=not args.no_simplify,
        output_names=output_names,
        dynamic_axes=None,
        input_channels=input_channels,
        input_name=input_name,
    )
    print(f"Exported ONNX to {args.output}")

    if args.verify and args.merge_postprocess:
        deltas = verify_outputs(
            export_module,
            args.output,
            img_size=img_size,
            input_channels=input_channels,
            input_name=input_name,
        )
        print("Verification max abs diff:", deltas)

    # Inject input resize and dynamic axes after export/simplification
    if args.dynamic_resize:
        add_input_resize(
            args.output,
            target_size=img_size,
            input_name=input_name,
            resize_mode=resize_mode,
            input_channels=input_channels,
        )
        print(f"Injected input Resize to fixed size {img_size} with dynamic input axes (mode={resize_mode}).")
        if not args.no_simplify:
            simplify_onnx_path(args.output)
            print("Re-simplified ONNX after injecting Resize.")

    meta_score_mode = (
        meta.get("score_mode")
        if isinstance(meta, dict) and meta.get("score_mode")
        else inferred.get("score_mode", "") if isinstance(inferred, dict) else ""
    )
    resolved_score_mode = _normalize_score_mode(meta_score_mode, bool(getattr(model.head, "disable_cls", False)))
    quality_power = float(getattr(export_module, "quality_power", getattr(model, "quality_power", 1.0)))
    decode_score_desc = _describe_decode_score(resolved_score_mode, quality_power)
    raw_concat_fields = None
    raw_concat_layout = None
    if not args.merge_postprocess and not args.noconcat_box_obj_quality_cls:
        raw_concat_fields = (
            ",".join(getattr(export_module, "raw_concat_fields", [])) if hasattr(export_module, "raw_concat_fields") else None
        )
        raw_concat_layout = "anchor_interleaved"
    postprocess_mode = "full" if args.merge_postprocess else "raw"
    if args.merge_primitive_postprocess:
        postprocess_mode = "primitive"

    update_metadata_props(
        args.output,
        {
            "resize_mode": resize_mode,
            "dynamic_resize": str(bool(args.dynamic_resize)).lower(),
            "raw_concat": str(not args.noconcat_box_obj_quality_cls).lower() if not args.merge_postprocess else "true",
            "decode_score": decode_score_desc,
            "decode_bbox": "cx = (sigmoid(tx)+gx)/w, cy = (sigmoid(ty)+gy)/h, bw = anchor_w*softplus(tw)*wh_scale, bh = anchor_h*softplus(th)*wh_scale; boxes = (cx±bw/2, cy±bh/2)",
            "quality_power": str(quality_power),
            "img_size": meta.get("img_size", "") if isinstance(meta, dict) else "",
            "score_mode": resolved_score_mode,
            "disable_cls": str(bool(getattr(model.head, "disable_cls", False))).lower(),
            "raw_concat_fields": raw_concat_fields,
            "raw_concat_layout": raw_concat_layout,
            "postprocess_mode": postprocess_mode,
            "multi_label_mode": (
                meta.get("multi_label_mode")
                if isinstance(meta, dict) and meta.get("multi_label_mode")
                else inferred.get("multi_label_mode", "none") if isinstance(inferred, dict) else "none"
            ),
            "multi_label_det_classes": (
                meta.get("multi_label_det_classes")
                if isinstance(meta, dict) and meta.get("multi_label_det_classes")
                else inferred.get("det_class_indices", "") if isinstance(inferred, dict) else ""
            ),
            "multi_label_attr_classes": (
                meta.get("multi_label_attr_classes")
                if isinstance(meta, dict) and meta.get("multi_label_attr_classes")
                else inferred.get("attr_class_indices", "") if isinstance(inferred, dict) else ""
            ),
            "multi_label_attr_weight": meta.get("multi_label_attr_weight", "") if isinstance(meta, dict) else "",
        },
    )


if __name__ == "__main__":
    main()
