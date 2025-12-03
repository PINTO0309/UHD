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
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return x


class SPPFmin(nn.Module):
    """
    かなり軽量化した SPPF (Spatial Pyramid Pooling - Fast) 風ブロック

    標準の SPPF よりチャネル数と演算量を抑え、
    UltraTinyOD 用に最小限構成にしている。
    """

    def __init__(self, c_in: int, c_out: int, pool_k: int = 5):
        super().__init__()
        # まずチャネルを半減
        c_hidden = c_in // 2
        self.cv1 = ConvBNAct(c_in, c_hidden, k=1, s=1, p=0)
        # 1 回だけの MaxPool（pool_k×pool_k）
        self.pool = nn.MaxPool2d(kernel_size=pool_k, stride=1, padding=pool_k // 2)
        # 出力チャネルを c_out に整える
        self.cv2 = ConvBNAct(c_hidden * 2, c_out, k=1, s=1, p=0)

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

    入力:  [B, 3, 64, 64]
    出力:  [B, C, H', W'] (H', W' は 8x8 を想定, stride = 8)

    構成:
        stem: Conv 3->16, stride 2 (64 -> 32)
        block1: DWConv 16->32, stride 2 (32 -> 16)
        block2: DWConv 32->64, stride 2 (16 -> 8)
        block3: DWConv 64->128, stride 1 (8 -> 8)
        block4: DWConv 128->128, stride 1 (8 -> 8)
        sppf: SPPFmin 128->64 (8 -> 8)
    """

    def __init__(self, c_stem: int = 16, use_residual: bool = False, out_stride: int = 8):
        super().__init__()
        if out_stride not in (4, 8):
            raise ValueError(f"UltraTinyODBackbone only supports out_stride 4 or 8, got {out_stride}")
        self.use_residual = bool(use_residual)
        self.out_stride = int(out_stride)
        # 64 -> 32
        self.stem = ConvBNAct(3, c_stem, k=3, s=2)

        # 32 -> 16
        self.block1 = DWConv(c_stem, c_stem * 2, k=3, s=2)   # 16 -> 32
        # 16 -> 8 (stride 8) or keep 16 (stride 4)
        stride_block2 = 2 if self.out_stride == 8 else 1
        self.block2 = DWConv(c_stem * 2, c_stem * 4, k=3, s=stride_block2)  # 32 -> 64
        # 8 -> 8
        self.block3 = DWConv(c_stem * 4, c_stem * 8, k=3, s=1)  # 64 -> 128
        self.block4 = DWConv(c_stem * 8, c_stem * 8, k=3, s=1)  # 128 -> 128
        if self.use_residual:
            # project block2 output (64ch) to match block3 output (128ch)
            self.block3_skip = ConvBNAct(c_stem * 4, c_stem * 8, k=1, s=1, p=0, act=False)
            self.block4_skip = nn.Identity()

        # SPPF-min: 128 -> 64
        self.sppf = SPPFmin(c_stem * 8, c_stem * 4)

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
    """

    num_classes: int = 1
    stride: int = 8
    anchors: Optional[Sequence[Tuple[float, float]]] = None

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


class UltraTinyODHead(nn.Module):
    """
    UltraTinyOD の Detection Head（単一スケール）

    出力:
        raw: [B, na, ny, nx, (5 + nc)]
        decode=True 時には decode 結果も返す
    """

    def __init__(self, in_channels: int, cfg: UltraTinyODConfig):
        super().__init__()
        self.nc = cfg.num_classes
        self.stride = cfg.stride
        self.in_channels = in_channels
        anchor_tensor = torch.as_tensor(cfg.anchors, dtype=torch.float32)
        if anchor_tensor.numel() == 0:
            anchor_tensor = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float32)
        if anchor_tensor.ndim != 2 or anchor_tensor.shape[1] != 2:
            anchor_tensor = anchor_tensor.reshape(-1, 2)
        self.register_buffer("anchors", anchor_tensor)  # (A,2) normalized w,h
        self.num_anchors = int(anchor_tensor.shape[0])
        self.no = self.nc + 5  # (x, y, w, h, obj) + cls
        self.in_channels = in_channels

        # 軽い文脈強調
        self.context = DWConv(in_channels, in_channels, k=3, s=1, act=True)

        # box+obj ブランチ
        self.box_obj_conv = DWConv(in_channels, in_channels, k=3, s=1, act=True)
        self.box_obj_out = nn.Conv2d(
            in_channels,
            self.num_anchors * 5,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # cls ブランチ
        self.cls_conv = DWConv(in_channels, in_channels, k=3, s=1, act=True)
        self.cls_out = nn.Conv2d(
            in_channels,
            self.num_anchors * self.nc,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.reset_output_bias()

    def set_anchors(self, anchors: torch.Tensor) -> None:
        """
        anchors: Tensor [A, 2] 正規化 w,h。Anchor 個数が変わる場合は head を再初期化。
        """
        if anchors is None:
            return
        anchor_tensor = anchors.detach().to(self.box_obj_out.weight.device)
        if anchor_tensor.ndim == 1:
            anchor_tensor = anchor_tensor.view(-1, 2)
        if anchor_tensor.ndim != 2 or anchor_tensor.shape[1] != 2:
            raise ValueError(f"anchors must have shape (A,2); got {anchor_tensor.shape}")
        self.anchors = anchor_tensor
        if anchor_tensor.shape[0] != self.num_anchors:
            self.num_anchors = int(anchor_tensor.shape[0])
            # 再初期化（Anchor 数依存の出力 Conv）
            self.box_obj_out = nn.Conv2d(
                self.in_channels,
                self.num_anchors * 5,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ).to(anchor_tensor.device)
            self.cls_out = nn.Conv2d(
                self.in_channels,
                self.num_anchors * self.nc,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ).to(anchor_tensor.device)
            nn.init.kaiming_normal_(self.box_obj_out.weight, mode="fan_out", nonlinearity="relu")
            nn.init.kaiming_normal_(self.cls_out.weight, mode="fan_out", nonlinearity="relu")
            self.reset_output_bias()
        self.grid = None

    def reset_output_bias(self, p_obj: float = 0.01, p_cls: float = 0.01) -> None:
        """Set conservative initial biases to reduce early false positives."""
        obj_bias = float(math.log(p_obj / (1.0 - p_obj)))
        cls_bias = float(math.log(p_cls / (1.0 - p_cls)))
        if self.box_obj_out.bias is not None:
            with torch.no_grad():
                bias = self.box_obj_out.bias.view(self.num_anchors, 5)
                bias.zero_()
                bias[:, 4] = obj_bias
                self.box_obj_out.bias.copy_(bias.view(-1))
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

        # 軽い文脈強調
        x = self.context(x).contiguous()

        # box+obj ブランチ
        box_obj = self.box_obj_conv(x)
        box_obj = self.box_obj_out(box_obj).view(b, self.num_anchors, 5, h, w)
        # cls ブランチ
        cls = self.cls_conv(x)
        cls = self.cls_out(cls).view(b, self.num_anchors, self.nc, h, w)

        # merge to [B, na, (5+nc), H, W]
        pred = torch.cat([box_obj, cls], dim=2)
        raw_map = pred.view(b, self.num_anchors * self.no, h, w)

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
        )
        return raw_map, decoded


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
        use_residual: bool = False,
    ):
        super().__init__()

        if config is None:
            config = UltraTinyODConfig(num_classes=num_classes)
        else:
            # config の num_classes を上書き
            config.num_classes = num_classes

        self.backbone = UltraTinyODBackbone(c_stem=c_stem, use_residual=use_residual, out_stride=int(config.stride))
        self.head = UltraTinyODHead(self.backbone.out_channels, config)
        self.anchors = self.head.anchors
        self.out_stride = int(config.stride)

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
            入力画像 [B, 3, H, W] （H, W は 64 を想定）
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
