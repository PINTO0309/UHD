#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UltraTinyOD: Ultra-tiny anchor-based object detector for 64x64 inputs.

- Anchor-based, YOLO-style single-scale detector
- 想定入力解像度: 64x64
- CPU 推論 2ms 以下を目標にした極小構成
- PyTorch 2.x / 1.10+ を想定

使い方（例）
-----------
from ultratinyod.model import UltraTinyOD

model = UltraTinyOD(num_classes=1)
x = torch.randn(1, 3, 64, 64)
raw_out, decoded = model(x, decode=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _make_activation(name: str) -> nn.Module:
    act = (name or "silu").lower()
    if act in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


# ============================================================
# 基本ブロック
# ============================================================

class ConvBNAct(nn.Module):
    """
    Conv -> BatchNorm -> SiLU の基本ブロック

    - 3x3 / 1x1 いずれにも利用可能
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        bias: bool = False,
        act: bool = True,
        act_name: str = "silu",
    ):
        super().__init__()
        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = _make_activation(act_name) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """
    Depthwise Separable Conv
    - 3x3 depthwise conv + 1x1 pointwise conv
    - UltraTinyOD の主力ブロック（FLOPs を大きく削減）
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        s: int = 1,
        act: bool = True,
        act_name: str = "silu",
    ):
        super().__init__()
        # depthwise
        self.dw = ConvBNAct(
            c_in,
            c_in,
            k=k,
            s=s,
            p=k // 2,
            g=c_in,
            bias=False,
            act=act,
            act_name=act_name,
        )
        # pointwise
        self.pw = ConvBNAct(
            c_in,
            c_out,
            k=1,
            s=1,
            p=0,
            g=1,
            bias=False,
            act=act,
            act_name=act_name,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return x


class EfficientSE(nn.Module):
    """軽量eSE (squeeze + 1x1 conv)。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3), keepdim=True)
        w = torch.sigmoid(self.fc(w))
        return x * w


class ReceptiveFieldEnhancer(nn.Module):
    """Lightweight receptive-field block mixing dilated and wide depthwise convs."""

    def __init__(self, channels: int, dilation: int = 2, act_name: str = "silu") -> None:
        super().__init__()
        d = max(1, int(dilation))
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=d, dilation=d, groups=channels, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03),
            _make_activation(act_name),
        )
        self.branch_wide = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03),
            _make_activation(act_name),
        )
        self.fuse = ConvBNAct(channels * 2, channels, k=1, s=1, p=0, act_name=act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_dilated(x)
        b2 = self.branch_wide(x)
        return x + self.fuse(torch.cat([b1, b2], dim=1))


class SPPFmin(nn.Module):
    """
    かなり軽量化した SPPF (Spatial Pyramid Pooling - Fast) 風ブロック

    標準の SPPF よりチャネル数と演算量を抑え、
    UltraTinyOD 用に最小限構成にしている。
    """

    def __init__(self, c_in: int, c_out: int, pool_k: int = 5, act_name: str = "silu"):
        super().__init__()
        # まずチャネルを半減
        c_hidden = c_in // 2
        self.cv1 = ConvBNAct(c_in, c_hidden, k=1, s=1, p=0, act_name=act_name)
        # 1 回だけの MaxPool（pool_k×pool_k）
        self.pool = nn.MaxPool2d(kernel_size=pool_k, stride=1, padding=pool_k // 2)
        # 出力チャネルを c_out に整える
        self.cv2 = ConvBNAct(c_hidden * 2, c_out, k=1, s=1, p=0, act_name=act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y = self.pool(x)
        x = torch.cat([x, y], dim=1)
        x = self.cv2(x)
        return x


# ============================================================
# UltraTinyOD Backbone + Head
# ============================================================

class UltraTinyODBackbone(nn.Module):
    """
    UltraTinyOD 用バックボーン

    入力:  [B, C, 64, 64]
    出力:  [B, C, H', W'] (H', W' は stride に依存: 4/8/16 のいずれか)

    構成:
        stem: Conv 3->16, stride 2 (64 -> 32)
        block1: DWConv 16->32, stride 2 (32 -> 16)
        block2: DWConv 32->64, stride 2 (16 -> 8)
        block3: DWConv 64->128, stride 1 (8 -> 8)
        block4: DWConv 128->128, stride 1 (8 -> 8)
        sppf: SPPFmin 128->64 (8 -> 8)
    """

    def __init__(self, c_stem: int = 16, in_channels: int = 3, use_residual: bool = False, out_stride: int = 8, activation: str = "silu"):
        super().__init__()
        if out_stride not in (4, 8, 16):
            raise ValueError(f"UltraTinyODBackbone only supports out_stride 4, 8, or 16; got {out_stride}")
        self.use_residual = bool(use_residual)
        self.out_stride = int(out_stride)
        act_name = activation
        # 64 -> 32
        self.stem = ConvBNAct(in_channels, c_stem, k=3, s=2, act_name=act_name)

        # 32 -> 16
        self.block1 = DWConv(c_stem, c_stem * 2, k=3, s=2, act_name=act_name)   # 16 -> 32
        # 16 -> 8 (stride 8) or keep 16 (stride 4)
        stride_block2 = 2 if self.out_stride >= 8 else 1
        self.block2 = DWConv(c_stem * 2, c_stem * 4, k=3, s=stride_block2, act_name=act_name)  # 32 -> 64
        # 8 -> 8 or 8 -> 4 (stride16 case)
        stride_block3 = 2 if self.out_stride == 16 else 1
        self.block3 = DWConv(c_stem * 4, c_stem * 8, k=3, s=stride_block3, act_name=act_name)  # 64 -> 128
        self.block4 = DWConv(c_stem * 8, c_stem * 8, k=3, s=1, act_name=act_name)  # 128 -> 128
        if self.use_residual:
            # project block2 output (64ch) to match block3 output (128ch)
            self.block3_skip = ConvBNAct(c_stem * 4, c_stem * 8, k=1, s=stride_block3, p=0, act=False, act_name=act_name)
            self.block4_skip = nn.Identity()

        # SPPF-min: 128 -> 64
        self.sppf = SPPFmin(c_stem * 8, c_stem * 4, act_name=act_name)

        self.out_channels = c_stem * 4  # 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x2)
        if self.use_residual:
            x3 = x3 + self.block3_skip(x2)
        x4_in = x3
        x4 = self.block4(x3)
        if self.use_residual:
            x4 = x4 + self.block4_skip(x4_in)
        x = self.sppf(x4)
        return x


@dataclass
class UltraTinyODConfig:
    """
    UltraTinyOD の設定

    - num_classes : クラス数
    - anchors     : [(w, h), ...] のリスト（入力に対する正規化値, e.g., w=0.125 は 8px/64px）
    - stride      : この Head が担当する stride (通常 8、主に情報用途)
    - cls_bottleneck_ratio : cls ブランチのチャネル圧縮率 (0<r<=1)
    - use_improved_head : 追加の品質スコア・WHスケーリング等を有効化
    - use_head_ese : Head入口にeSEを挿入して軽量に文脈強調
    - use_iou_aware_head : IoU/quality をクラス信頼度に直結させるタスクアラインドヘッド
    - quality_power : quality スコアの指数。IoU-aware スコアリングの鋭さを調整
    """

    num_classes: int = 1
    stride: int = 8
    anchors: Optional[Sequence[Tuple[float, float]]] = None
    cls_bottleneck_ratio: float = 0.5
    use_improved_head: bool = False
    use_head_ese: bool = False
    use_iou_aware_head: bool = False
    quality_power: float = 1.0
    use_fpn: bool = False
    fpn_strides: Optional[Sequence[int]] = None
    use_fpn_strict: bool = False
    activation: str = "silu"
    use_context_rfb: bool = False
    context_dilation: int = 2
    use_large_obj_branch: bool = False
    large_obj_branch_depth: int = 1
    large_obj_branch_expansion: float = 1.0

    def __post_init__(self):
        if self.anchors is None:
            # 64x64 前提のゆるい初期 Anchor（正規化スケール）
            self.anchors = [
                (8 / 64.0, 16 / 64.0),   # small person (遠距離・上半身・頭肩)
                (12 / 64.0, 28 / 64.0),  # medium person（腰から上）
                (20 / 64.0, 40 / 64.0),  # full person（全身・近距離）
            ]
        # ensure float tuples
        self.anchors = [(float(w), float(h)) for w, h in self.anchors]
        self.cls_bottleneck_ratio = float(max(0.05, min(1.0, self.cls_bottleneck_ratio)))
        self.use_iou_aware_head = bool(self.use_iou_aware_head)
        self.quality_power = float(max(0.0, self.quality_power))
        self.use_fpn = bool(self.use_fpn)
        self.use_fpn_strict = bool(self.use_fpn_strict)
        if self.fpn_strides is None:
            self.fpn_strides = [self.stride]
        self.fpn_strides = [int(s) for s in self.fpn_strides]
        act = (self.activation or "silu").lower()
        self.activation = "silu" if act == "swish" else act
        self.use_context_rfb = bool(self.use_context_rfb)
        self.context_dilation = max(1, int(self.context_dilation))
        self.use_large_obj_branch = bool(self.use_large_obj_branch)
        self.large_obj_branch_depth = max(1, int(self.large_obj_branch_depth))
        self.large_obj_branch_expansion = float(max(0.25, self.large_obj_branch_expansion))


class UltraTinyODHead(nn.Module):
    """
    UltraTinyOD の Detection Head（単一スケール）

    出力:
        raw: [B, na, ny, nx, (5 + nc)]
        decode=True 時には decode 結果も返す
    """

    def __init__(self, in_channels: int, cfg: UltraTinyODConfig, activation: str = "silu"):
        super().__init__()
        self.nc = cfg.num_classes
        self.stride = cfg.stride
        self.in_channels = in_channels
        self.cls_ratio = float(getattr(cfg, "cls_bottleneck_ratio", 0.5))
        self.cls_mid = max(8, min(in_channels, int(round(in_channels * self.cls_ratio))))
        act_name = activation
        self.use_improved_head = bool(getattr(cfg, "use_improved_head", False))
        self.use_head_ese = bool(getattr(cfg, "use_head_ese", False))
        self.use_iou_aware_head = bool(getattr(cfg, "use_iou_aware_head", False))
        self.use_context_rfb = bool(getattr(cfg, "use_context_rfb", False))
        self.context_dilation = max(1, int(getattr(cfg, "context_dilation", 2)))
        self.use_large_obj_branch = bool(getattr(cfg, "use_large_obj_branch", False))
        self.large_obj_branch_depth = max(1, int(getattr(cfg, "large_obj_branch_depth", 1)))
        self.large_obj_branch_expansion = float(max(0.25, getattr(cfg, "large_obj_branch_expansion", 1.0)))
        self.has_quality = self.use_improved_head or self.use_iou_aware_head
        self.quality_power = float(getattr(cfg, "quality_power", 1.0))
        anchor_tensor = torch.as_tensor(cfg.anchors, dtype=torch.float32)
        if anchor_tensor.numel() == 0:
            anchor_tensor = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float32)
        if anchor_tensor.ndim != 2 or anchor_tensor.shape[1] != 2:
            anchor_tensor = anchor_tensor.reshape(-1, 2)
        self.register_buffer("anchors", anchor_tensor)  # (A,2) normalized w,h
        self.num_anchors = int(anchor_tensor.shape[0])
        self.no = self.nc + 5 + (1 if self.has_quality else 0)  # (x, y, w, h, obj[, qual]) + cls
        self.in_channels = in_channels
        if self.use_improved_head:
            self.wh_scale = nn.Parameter(torch.ones(self.num_anchors, 2, dtype=torch.float32))
        else:
            self.register_buffer("wh_scale", torch.ones(self.num_anchors, 2, dtype=torch.float32))
        # task-aligned score uses quality*cls (no obj) when enabled
        self.score_mode = "quality_cls" if self.use_iou_aware_head else "obj_quality_cls"
        self.quality_power = float(max(0.0, self.quality_power))
        self.has_quality_head = self.has_quality

        # 軽い文脈強調
        self.context = DWConv(in_channels, in_channels, k=3, s=1, act=True, act_name=act_name)
        if self.use_improved_head:
            self.context_res = nn.Sequential(
                DWConv(in_channels, in_channels, k=3, s=1, act=True, act_name=act_name),
                ConvBNAct(in_channels, in_channels, k=1, s=1, p=0, act_name=act_name),
                DWConv(in_channels, in_channels, k=3, s=1, act=True, act_name=act_name),
            )
        if self.use_head_ese:
            self.head_se = EfficientSE(in_channels)
        self.head_rfb = (
            ReceptiveFieldEnhancer(in_channels, dilation=self.context_dilation, act_name=act_name)
            if self.use_context_rfb
            else None
        )
        if self.use_large_obj_branch:
            lob_ch = int(round(in_channels * self.large_obj_branch_expansion))
            self.large_obj_down = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.03),
                _make_activation(act_name),
                ConvBNAct(in_channels, lob_ch, k=1, s=1, p=0, act_name=act_name),
            )
            self.large_obj_blocks = nn.Sequential(
                *[DWConv(lob_ch, lob_ch, k=3, s=1, act=True, act_name=act_name) for _ in range(self.large_obj_branch_depth)]
            )
            self.large_obj_fuse = ConvBNAct(lob_ch, in_channels, k=1, s=1, p=0, act_name=act_name)
        else:
            self.large_obj_down = None

        # box ブランチ
        if self.use_iou_aware_head:
            self.box_tower = nn.Sequential(
                DWConv(in_channels, in_channels, k=3, s=1, act=True, act_name=act_name),
                DWConv(in_channels, in_channels, k=3, s=1, act=True, act_name=act_name),
            )
        else:
            self.box_conv = DWConv(in_channels, in_channels, k=3, s=1, act=True, act_name=act_name)
        self.box_out = nn.Conv2d(
            in_channels,
            self.num_anchors * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        if self.has_quality:
            if self.use_iou_aware_head:
                self.quality_tower = nn.Sequential(
                    DWConv(in_channels, in_channels, k=3, s=1, act=True, act_name=act_name),
                    DWConv(in_channels, in_channels, k=3, s=1, act=True, act_name=act_name),
                )
            else:
                self.quality_conv = DWConv(in_channels, in_channels, k=3, s=1, act=True, act_name=act_name)
            self.quality_out = nn.Conv2d(
                in_channels,
                self.num_anchors * 1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

        # obj ブランチ
        self.obj_conv = DWConv(in_channels, in_channels, k=3, s=1, act=True, act_name=act_name)
        self.obj_out = nn.Conv2d(
            in_channels,
            self.num_anchors * 1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # cls ブランチ
        if self.use_iou_aware_head:
            self.cls_tower = nn.Sequential(
                ConvBNAct(in_channels, self.cls_mid, k=1, s=1, p=0, act_name=act_name),
                DWConv(self.cls_mid, self.cls_mid, k=3, s=1, act=True, act_name=act_name),
                DWConv(self.cls_mid, self.cls_mid, k=3, s=1, act=True, act_name=act_name),
            )
        else:
            self.cls_reduce = ConvBNAct(in_channels, self.cls_mid, k=1, s=1, p=0, act_name=act_name)
            self.cls_conv = DWConv(self.cls_mid, self.cls_mid, k=3, s=1, act=True, act_name=act_name)
        self.cls_out = nn.Conv2d(
            self.cls_mid,
            self.num_anchors * self.nc,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        perm = self._build_raw_map_perm()
        self.register_buffer("raw_map_perm", perm, persistent=False)

        self.reset_output_bias()

    def _build_raw_map_perm(self) -> torch.Tensor:
        """Build channel permutation to interleave anchor outputs without 5D reshapes."""
        a = int(self.num_anchors)
        nc = int(self.nc)
        perm = []
        if self.has_quality:
            base_box = 0
            base_obj = a * 4
            base_quality = base_obj + a
            base_cls = base_quality + a
            for anchor in range(a):
                base_box_anchor = base_box + anchor * 4
                perm.extend([base_box_anchor + i for i in range(4)])
                perm.append(base_obj + anchor)
                perm.append(base_quality + anchor)
                base_cls_anchor = base_cls + anchor * nc
                perm.extend([base_cls_anchor + i for i in range(nc)])
        else:
            base_box = 0
            base_obj = a * 4
            base_cls = base_obj + a
            for anchor in range(a):
                base_box_anchor = base_box + anchor * 4
                perm.extend([base_box_anchor + i for i in range(4)])
                perm.append(base_obj + anchor)
                base_cls_anchor = base_cls + anchor * nc
                perm.extend([base_cls_anchor + i for i in range(nc)])
        return torch.tensor(perm, dtype=torch.long)

    def set_anchors(self, anchors: torch.Tensor) -> None:
        """
        anchors: Tensor [A, 2] 正規化 w,h。Anchor 個数が変わる場合は head を再初期化。
        """
        if anchors is None:
            return
        anchor_tensor = anchors.detach().to(self.box_out.weight.device)
        if anchor_tensor.ndim == 1:
            anchor_tensor = anchor_tensor.view(-1, 2)
        if anchor_tensor.ndim != 2 or anchor_tensor.shape[1] != 2:
            raise ValueError(f"anchors must have shape (A,2); got {anchor_tensor.shape}")
        self.anchors = anchor_tensor
        if anchor_tensor.shape[0] != self.num_anchors:
            self.num_anchors = int(anchor_tensor.shape[0])
            # 再初期化（Anchor 数依存の出力 Conv）
            self.box_out = nn.Conv2d(
                self.in_channels,
                self.num_anchors * 4,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ).to(anchor_tensor.device)
            self.obj_out = nn.Conv2d(
                self.in_channels,
                self.num_anchors * 1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ).to(anchor_tensor.device)
            if self.has_quality:
                self.quality_out = nn.Conv2d(
                    self.in_channels,
                    self.num_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ).to(anchor_tensor.device)
            self.cls_out = nn.Conv2d(
                self.cls_mid,
                self.num_anchors * self.nc,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ).to(anchor_tensor.device)
            nn.init.kaiming_normal_(self.box_out.weight, mode="fan_out", nonlinearity="relu")
            nn.init.kaiming_normal_(self.obj_out.weight, mode="fan_out", nonlinearity="relu")
            if self.has_quality:
                nn.init.kaiming_normal_(self.quality_out.weight, mode="fan_out", nonlinearity="relu")
                if self.use_improved_head and isinstance(self.wh_scale, nn.Parameter):
                    self.wh_scale = nn.Parameter(torch.ones(self.num_anchors, 2, device=anchor_tensor.device))
            nn.init.kaiming_normal_(self.cls_out.weight, mode="fan_out", nonlinearity="relu")
            self.reset_output_bias()
            if not self.use_improved_head:
                self.wh_scale = torch.ones(self.num_anchors, 2, device=anchor_tensor.device)
            self.raw_map_perm = self._build_raw_map_perm().to(anchor_tensor.device)
        self.grid = None

    def reset_output_bias(self, p_obj: float = 0.01, p_cls: float = 0.01) -> None:
        """Set conservative initial biases to reduce early false positives."""
        obj_bias = float(math.log(p_obj / (1.0 - p_obj)))
        cls_bias = float(math.log(p_cls / (1.0 - p_cls)))
        if self.obj_out.bias is not None:
            with torch.no_grad():
                bias = self.obj_out.bias.view(self.num_anchors, 1)
                bias.zero_()
                bias[:, 0] = obj_bias
                self.obj_out.bias.copy_(bias.view(-1))
        if self.has_quality and self.quality_out.bias is not None:
            nn.init.constant_(self.quality_out.bias, 0.0)
        if self.cls_out.bias is not None:
            nn.init.constant_(self.cls_out.bias, cls_bias)

    def forward(
        self,
        x: torch.Tensor,
        decode: bool = False,
        conf_thresh: float = 0.3,
        nms_thresh: float = 0.5,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x : torch.Tensor
            BackBone 出力 [B, C, H, W]
        decode : bool
            True の場合は YOLO 形式で decode したリストも返す（decode_anchor 相当）。

        Returns
        -------
        raw_preds : torch.Tensor
            [B, na, ny, nx, (5 + nc)] の raw 出力（活性化前）
        decoded : list or None
            decode=True の場合: decode_anchor 形式の [(score, cls, box), ...] をバッチごとに格納したリスト
            decode=False の場合: None
        """
        b, c, h, w = x.shape

        box, obj, quality, cls = self.forward_raw_parts(x)

        # merge to [B, na, (5+nc), H, W] (tx,ty,tw,th,obj,cls...)
        if self.has_quality and quality is not None:
            pred = torch.cat([box, obj, quality, cls], dim=1)
        else:
            pred = torch.cat([box, obj, cls], dim=1)
        raw_map = pred.index_select(1, self.raw_map_perm)

        if not decode:
            return raw_map, None

        # decode は pipeline の decode_anchor と同じパラメータ化 (tx/ty sigmoid, tw/th softplus)
        # anchor は正規化前提
        anchor_tensor = self.anchors.to(raw_map.device)
        try:
            from .metrics import decode_anchor  # 遅延 import で循環を避ける
        except ImportError:
            from uhd.metrics import decode_anchor  # type: ignore

        decoded = decode_anchor(
            raw_map,
            anchors=anchor_tensor,
            num_classes=self.nc,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
            has_quality=self.has_quality,
            wh_scale=self.wh_scale if self.use_improved_head else None,
            score_mode=self.score_mode,
            quality_power=self.quality_power,
        )
        return raw_map, decoded

    def forward_raw_parts(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Return raw head outputs without concatenation (box, obj, quality, cls)."""
        b, c, h, w = x.shape

        # 軽い文脈強調
        x = self.context(x).contiguous()
        if self.use_improved_head:
            x = x + self.context_res(x)
        if self.head_rfb is not None:
            x = self.head_rfb(x)
        if self.large_obj_down is not None:
            lob = self.large_obj_down(x)
            lob = self.large_obj_blocks(lob)
            lob = F.interpolate(lob, size=(h, w), mode="nearest")
            x = x + self.large_obj_fuse(lob)
        if self.use_head_ese:
            x = self.head_se(x)

        # box ブランチ
        if self.use_iou_aware_head:
            box_feat = self.box_tower(x)
        else:
            box_feat = self.box_conv(x)
        box = self.box_out(box_feat)
        # obj ブランチ
        obj = self.obj_conv(x)
        obj = self.obj_out(obj)
        # quality ブランチ
        quality = None
        if self.has_quality:
            if self.use_iou_aware_head:
                quality_feat = self.quality_tower(x)
            else:
                quality_feat = self.quality_conv(x)
            quality = self.quality_out(quality_feat)
        # cls ブランチ
        if self.use_iou_aware_head:
            cls_feat = self.cls_tower(x)
        else:
            cls_feat = self.cls_reduce(x)
            cls_feat = self.cls_conv(cls_feat)
        cls = self.cls_out(cls_feat)

        return box, obj, quality, cls


class UltraTinyOD(nn.Module):
    """
    UltraTinyOD 本体

    - アンカーベース YOLO スタイルの単一スケール detector
    - 入力は 64x64 前提だが、stride=8 の倍数であれば他の解像度も一応動作可能

    Parameters
    ----------
    num_classes : int
        クラス数
    config : UltraTinyODConfig or None
        UltraTinyODConfig を明示的に指定したい場合のみ使用
    c_stem : int
        ステムのチャネル数基準。小さくするとさらに軽量化できる。
        （デフォルト: 16）
    """

    def __init__(
        self,
        num_classes: int = 1,
        config: Optional[UltraTinyODConfig] = None,
        c_stem: int = 16,
        in_channels: int = 3,
        use_residual: bool = False,
        use_improved_head: bool = False,
        use_head_ese: bool = False,
        use_iou_aware_head: bool = False,
        quality_power: float = 1.0,
        activation: str = "silu",
    ):
        super().__init__()

        if config is None:
            config = UltraTinyODConfig(
                num_classes=num_classes,
                use_improved_head=use_improved_head,
                use_head_ese=use_head_ese,
                use_iou_aware_head=use_iou_aware_head,
                quality_power=quality_power,
                activation=activation,
            )
        else:
            # config の num_classes を上書き
            config.num_classes = num_classes
            if not hasattr(config, "activation"):
                config.activation = activation
            if not hasattr(config, "use_improved_head"):
                config.use_improved_head = bool(use_improved_head)
            if not hasattr(config, "use_head_ese"):
                config.use_head_ese = bool(use_head_ese)
            if not hasattr(config, "use_iou_aware_head"):
                config.use_iou_aware_head = bool(use_iou_aware_head)
            if not hasattr(config, "quality_power"):
                config.quality_power = float(quality_power)

        act_name = "silu" if str(config.activation).lower() == "swish" else str(config.activation).lower()

        self.backbone = UltraTinyODBackbone(
            c_stem=c_stem,
            in_channels=in_channels,
            use_residual=use_residual,
            out_stride=int(config.stride),
            activation=act_name,
        )
        self.head = UltraTinyODHead(self.backbone.out_channels, config, activation=act_name)
        self.anchors = self.head.anchors
        self.out_stride = int(config.stride)
        self.use_improved_head = bool(getattr(config, "use_improved_head", False))
        self.use_iou_aware_head = bool(getattr(config, "use_iou_aware_head", False))
        self.has_quality_head = bool(self.head.has_quality)
        self.score_mode = getattr(self.head, "score_mode", "obj_quality_cls")
        self.quality_power = getattr(self.head, "quality_power", 1.0)
        self.activation = act_name

        # モデル初期化（簡易版）
        self._init_weights()
        # 出力バイアスを抑制気味に初期化（_init_weights で 0 リセット後に実行）
        self.head.reset_output_bias()
        self.num_anchors = self.head.num_anchors

    # --------------------------------------------------------
    # 初期化
    # --------------------------------------------------------
    def _init_weights(self):
        """
        Conv / BN を簡易初期化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal
                # PyTorch calculate_gain does not support "silu"; use relu gain as a close proxy.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        decode: bool = False,
        return_feat: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x : torch.Tensor
            入力画像 [B, C, H, W] （H, W は 64 を想定）
        decode : bool
            True にすると、YOLO 形式の decode 結果も返す
            （推論・後処理で便利）
        return_feat : bool
            True にするとバックボーン特徴も返す（蒸留用途など）

        Returns
        -------
        raw_preds : torch.Tensor
            [B, na, ny, nx, (5 + nc)] の raw 出力（活性化前）
        decoded : list or None
            decode=True の場合: decode_anchor 形式の [(score, cls, box), ...] をバッチごとに格納したリスト
            decode=False の場合: None
        """
        feat = self.backbone(x)
        raw_preds, decoded = self.head(feat, decode=decode)
        if return_feat:
            return raw_preds, feat
        if decode:
            return raw_preds, decoded
        return raw_preds

    def set_anchors(self, anchors: torch.Tensor) -> None:
        self.head.set_anchors(anchors)
        self.anchors = self.head.anchors
        self.num_anchors = self.head.num_anchors


# ============================================================
# 簡易テスト
# ============================================================

if __name__ == "__main__":
    # 簡単な動作確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UltraTinyOD(num_classes=1).to(device)
    model.eval()

    x = torch.randn(1, 3, 64, 64, device=device)
    with torch.no_grad():
        raw, dec = model(x, decode=True)

    print("raw shape:", raw.shape)   # [B, na, ny, nx, 5+nc]
    if dec:
        print(f"decoded boxes (image 0): {len(dec[0])}")
    else:
        print("decoded boxes: 0")
