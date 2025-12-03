from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .ultratinyod import UltraTinyOD, UltraTinyODConfig


def _make_activation(name: str) -> nn.Module:
    act = name.lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "swish":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBNAct(nn.Module):
    """Standard conv-bn-activation block."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, stride: int = 1, activation: str = "swish") -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, activation: str = "swish") -> None:
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class SEModule(nn.Module):
    """Standard SE block with reduction."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3), keepdim=True)
        w = self.fc2(F.silu(self.fc1(w)))
        return x * torch.sigmoid(w)


class EfficientSEModule(nn.Module):
    """One-layer SE (eSE) without reduction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc(w))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, activation: str = "swish") -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3, stride=1, activation=activation)
        self.conv2 = ConvBNAct(channels, channels, kernel_size=3, stride=1, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class CSPTinyBlock(nn.Module):
    """Very small CSP-style block (keeps channels constant)."""

    def __init__(self, channels: int, activation: str = "swish") -> None:
        super().__init__()
        mid = max(1, channels // 2)
        self.conv1 = ConvBNAct(channels, mid, kernel_size=1, stride=1, activation=activation)
        self.conv2 = ConvBNAct(mid, mid, kernel_size=3, stride=1, activation=activation)
        self.conv3 = ConvBNAct(mid, mid, kernel_size=3, stride=1, activation=activation)
        self.conv4 = ConvBNAct(mid * 2, channels, kernel_size=1, stride=1, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        return self.conv4(torch.cat([y1, y3], dim=1))


class MicroCSPNet(nn.Module):
    """Tiny CSP-style backbone with stride 8 output."""

    def __init__(self, activation: str = "swish") -> None:
        super().__init__()
        self.conv1 = ConvBNAct(3, 16, kernel_size=3, stride=1, activation=activation)
        self.conv2 = ConvBNAct(16, 32, kernel_size=3, stride=2, activation=activation)
        self.csp2 = CSPTinyBlock(32, activation=activation)
        self.conv3 = ConvBNAct(32, 64, kernel_size=3, stride=2, activation=activation)
        self.csp3 = CSPTinyBlock(64, activation=activation)
        self.conv4 = ConvBNAct(64, 128, kernel_size=3, stride=2, activation=activation)
        self.compress = ConvBNAct(128, 64, kernel_size=1, stride=1, activation=activation)
        self.out_channels = 64
        self.out_stride = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.csp2(self.conv2(x))
        x = self.csp3(self.conv3(x))
        x = self.compress(self.conv4(x))
        return x


class UltraTinyResNet(nn.Module):
    """Minimal ResNet-like backbone ending at stride 2^(stages-1) (defaults to 8)."""

    def __init__(
        self,
        activation: str = "swish",
        channels=None,
        blocks=None,
        use_long_skip: bool = False,
        skip_mode: str = "add",
        use_fpn: bool = False,
        target_out_stride: int = None,
    ) -> None:
        super().__init__()
        ch_list = list(channels) if channels is not None else [16, 24, 32, 48]
        blk_list = list(blocks) if blocks is not None else [1, 1, 2, 1]
        if len(ch_list) != len(blk_list):
            raise ValueError(f"UltraTinyResNet requires matching channels/blocks lengths; got {len(ch_list)} vs {len(blk_list)}")
        if len(ch_list) < 1:
            raise ValueError("UltraTinyResNet requires at least one stage.")
        self.use_long_skip = bool(use_long_skip)
        self.skip_mode = (skip_mode or "add").lower()
        if self.skip_mode not in ("add", "cat", "shuffle_cat", "s2d_cat"):
            raise ValueError(f"UltraTinyResNet skip_mode must be 'add', 'cat', 'shuffle_cat', or 's2d_cat'; got {self.skip_mode}")
        self.use_fpn = bool(use_fpn)
        self.stem = ConvBNAct(3, ch_list[0], kernel_size=3, stride=1, activation=activation)

        self.downs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.skip_concat_reduce = None
        self.skip_down_factors = []
        # stage 0 (no downsample)
        self.downs.append(nn.Identity())
        self.blocks.append(nn.Sequential(*[ResidualBlock(ch_list[0], activation=activation) for _ in range(max(blk_list[0], 0))]))
        if self.use_long_skip:
            stride0 = max(1, 2 ** (len(ch_list) - 1)) if self.skip_mode == "shuffle_cat" else 1
            factor0 = 2 ** (len(ch_list) - 1)
            in_ch0 = ch_list[0] * (factor0 ** 2) if self.skip_mode == "s2d_cat" else ch_list[0]
            self.skip_proj.append(
                ConvBNAct(
                    in_ch0,
                    ch_list[-1],
                    kernel_size=3 if self.skip_mode == "shuffle_cat" else 1,
                    stride=stride0,
                    activation=activation,
                )
            )
            self.skip_down_factors.append(factor0)
        else:
            self.skip_proj.append(nn.Identity())
            self.skip_down_factors.append(1)
        # subsequent stages with stride-2 downsamples
        prev_ch = ch_list[0]
        for idx, (ch, num_blk) in enumerate(zip(ch_list[1:], blk_list[1:]), start=1):
            self.downs.append(ConvBNAct(prev_ch, ch, kernel_size=3, stride=2, activation=activation))
            self.blocks.append(nn.Sequential(*[ResidualBlock(ch, activation=activation) for _ in range(max(num_blk, 0))]))
            if self.use_long_skip:
                # stride to match final spatial size when using shuffle_cat; otherwise keep stride 1 and pool later
                factor = 2 ** (len(ch_list) - idx - 1)
                stride = max(1, factor) if self.skip_mode == "shuffle_cat" else 1
                in_ch = ch * (factor ** 2) if self.skip_mode == "s2d_cat" else ch
                self.skip_proj.append(
                    ConvBNAct(
                        in_ch,
                        ch_list[-1],
                        kernel_size=3 if self.skip_mode == "shuffle_cat" else 1,
                        stride=stride,
                        activation=activation,
                    )
                )
                self.skip_down_factors.append(factor)
            else:
                self.skip_proj.append(nn.Identity())
                self.skip_down_factors.append(1)
            prev_ch = ch
        if self.use_fpn:
            for ch in ch_list[:-1]:
                self.fpn_convs.append(ConvBNAct(ch, ch_list[-1], kernel_size=1, stride=1, activation=activation))
        if self.use_long_skip and self.skip_mode in ("cat", "shuffle_cat", "s2d_cat"):
            # fuse concatenated skips back to out_channels for the head
            num_concat = len(ch_list)  # one per stage including the final
            self.skip_concat_reduce = ConvBNAct(ch_list[-1] * num_concat, ch_list[-1], kernel_size=1, stride=1, activation=activation)
        self.out_channels = ch_list[-1]
        # stride doubles for every downsample stage
        self.out_stride = 2 ** (len(ch_list) - 1)
        self.extra_downs = nn.ModuleList()
        if target_out_stride is not None and target_out_stride > self.out_stride:
            cur = self.out_stride
            while cur < target_out_stride:
                self.extra_downs.append(ConvBNAct(self.out_channels, self.out_channels, kernel_size=3, stride=2, activation=activation))
                cur *= 2
            self.out_stride = cur

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        feats = []
        for down, block in zip(self.downs, self.blocks):
            x = block(down(x))
            feats.append(x)
        out = feats[-1] if feats else x
        if self.use_long_skip and feats:
            target_hw = out.shape[2:]
            skips = []
            for f, proj, factor in zip(feats[:-1], self.skip_proj[:-1], self.skip_down_factors[:-1]):
                if self.skip_mode == "shuffle_cat":
                    s = proj(f)
                    if s.shape[2:] != target_hw:
                        s = F.adaptive_avg_pool2d(s, target_hw)
                    s = channel_shuffle(s, groups=2)
                elif self.skip_mode == "s2d_cat":
                    if factor > 1 and f.shape[2] % factor == 0 and f.shape[3] % factor == 0:
                        s = F.pixel_unshuffle(f, downscale_factor=factor)
                    else:
                        s = F.adaptive_avg_pool2d(f, target_hw)
                    s = proj(s)
                else:
                    # Plain add/concat skip: pool to target spatial size first, then project.
                    pooled = f if f.shape[2:] == target_hw else F.adaptive_avg_pool2d(f, target_hw)
                    s = proj(pooled)
                skips.append(s)
            if self.skip_mode in ("cat", "shuffle_cat", "s2d_cat"):
                merged = torch.cat([out] + skips, dim=1)
                out = self.skip_concat_reduce(merged) if self.skip_concat_reduce is not None else merged
            else:
                for s in skips:
                    out = out + s
        if self.use_fpn and feats:
            target_hw = feats[-1].shape[2:]
            cur = feats[-1]
            for f, conv in zip(reversed(feats[:-1]), reversed(self.fpn_convs)):
                cur = F.interpolate(cur, size=f.shape[2:], mode="nearest")
                cur = cur + conv(f)
            if cur.shape[2:] != target_hw:
                cur = F.interpolate(cur, size=target_hw, mode="nearest")
            out = cur
        for down in self.extra_downs:
            out = down(out)
        return out


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.size()
    if c % groups != 0:
        return x
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).reshape(b, c, h, w)
    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, activation: str = "swish") -> None:
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        branch_out = c_out // 2
        act = activation
        if stride == 1:
            assert c_in == c_out and c_in % 2 == 0
            self.branch2 = nn.Sequential(
                ConvBNAct(branch_out, branch_out, kernel_size=1, stride=1, activation=act),
                nn.Conv2d(branch_out, branch_out, kernel_size=3, stride=1, padding=1, groups=branch_out, bias=False),
                nn.BatchNorm2d(branch_out),
                ConvBNAct(branch_out, branch_out, kernel_size=1, stride=1, activation=act),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(c_in, c_in, kernel_size=3, stride=2, padding=1, groups=c_in, bias=False),
                nn.BatchNorm2d(c_in),
                ConvBNAct(c_in, branch_out, kernel_size=1, stride=1, activation=act),
            )
            self.branch2 = nn.Sequential(
                ConvBNAct(c_in, branch_out, kernel_size=1, stride=1, activation=act),
                nn.Conv2d(branch_out, branch_out, kernel_size=3, stride=2, padding=1, groups=branch_out, bias=False),
                nn.BatchNorm2d(branch_out),
                ConvBNAct(branch_out, branch_out, kernel_size=1, stride=1, activation=act),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out)


class EnhancedShuffleNet(nn.Module):
    """
    ShuffleNetV2+ inspired (arXiv:2111.00902) scaled-down backbone (0.25x-ish) with stride 8 output.
    Deeper stages and channel squeeze vs the prior minimal variant.
    """

    def __init__(self, activation: str = "swish", use_global_skip: bool = False) -> None:
        super().__init__()
        act = activation
        self.use_global_skip = bool(use_global_skip)
        self.conv1 = ConvBNAct(3, 32, kernel_size=3, stride=2, activation=act)  # 32x32
        # Stage2: stride 2 entry, progressive widening via squeeze/expand (doubled depth)
        self.stage2 = nn.Sequential(
            ShuffleV2Block(32, 64, stride=2, activation=act),  # 16x16
            ConvBNAct(64, 80, kernel_size=1, stride=1, activation=act),
            ShuffleV2Block(80, 80, stride=1, activation=act),
            ShuffleV2Block(80, 80, stride=1, activation=act),
            ConvBNAct(80, 96, kernel_size=1, stride=1, activation=act),
            ShuffleV2Block(96, 96, stride=1, activation=act),
            ShuffleV2Block(96, 96, stride=1, activation=act),
            ConvBNAct(96, 112, kernel_size=1, stride=1, activation=act),
            ShuffleV2Block(112, 112, stride=1, activation=act),
            ShuffleV2Block(112, 112, stride=1, activation=act),
        )
        # Stage3: stride 2 to reach 8x8, further widening then squeeze at end (doubled depth)
        self.stage3 = nn.Sequential(
            ShuffleV2Block(112, 144, stride=2, activation=act),  # 8x8
            ConvBNAct(144, 160, kernel_size=1, stride=1, activation=act),
            ShuffleV2Block(160, 160, stride=1, activation=act),
            ShuffleV2Block(160, 160, stride=1, activation=act),
            ConvBNAct(160, 176, kernel_size=1, stride=1, activation=act),
            ShuffleV2Block(176, 176, stride=1, activation=act),
            ShuffleV2Block(176, 176, stride=1, activation=act),
            ConvBNAct(176, 192, kernel_size=1, stride=1, activation=act),
            ShuffleV2Block(192, 192, stride=1, activation=act),
            ShuffleV2Block(192, 192, stride=1, activation=act),
        )
        self.out_conv = ConvBNAct(192, 128, kernel_size=1, stride=1, activation=act)
        self.out_channels = 128
        self.out_stride = 8
        # Stage-level shortcuts
        # Downsample stage2 input to match stage2 output spatial size
        self.stage2_shortcut = ConvBNAct(32, 112, kernel_size=1, stride=2, activation=act)
        self.stage3_shortcut = ConvBNAct(112, 192, kernel_size=1, stride=2, activation=act)
        if self.use_global_skip:
            self.global_proj1 = ConvBNAct(32, 128, kernel_size=1, stride=4, activation=act)  # 32x32 -> 8x8
            self.global_proj2 = ConvBNAct(112, 128, kernel_size=1, stride=2, activation=act)  # 16x16 -> 8x8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.conv1(x)  # 32x32
        s2_in = s1
        s2 = self.stage2(s1)  # 16x16
        s2 = s2 + self.stage2_shortcut(s2_in)
        s3_in = s2
        s3 = self.stage3(s2)  # 8x8
        s3 = s3 + self.stage3_shortcut(s3_in)
        out = self.out_conv(s3)
        if self.use_global_skip:
            # fuse shallow signals into final output
            skip1 = self.global_proj1(s1)
            skip2 = self.global_proj2(s2)
            out = out + F.adaptive_avg_pool2d(skip1, out.shape[2:]) + F.adaptive_avg_pool2d(skip2, out.shape[2:])
        return out


class MiniCenterNet(nn.Module):
    """Minimal anchor-free detector with heatmap + offsets + size."""

    def __init__(
        self,
        width: int = 32,
        num_classes: int = 1,
        use_skip: bool = False,
        activation: str = "swish",
        last_se: str = "none",
        last_width_scale: float = 1.0,
        out_stride: int = 4,
        backbone: nn.Module = None,
        backbone_out_channels: int = None,
    ) -> None:
        super().__init__()
        if out_stride not in (4, 8, 16):
            raise ValueError(f"out_stride must be one of (4, 8, 16); got {out_stride}")
        last_scale = max(1.0, float(last_width_scale))
        self.custom_backbone = backbone is not None
        self.out_stride = getattr(backbone, "out_stride", out_stride) if self.custom_backbone else out_stride
        out_c = int(backbone_out_channels if backbone_out_channels is not None else round(width * last_scale))
        if self.custom_backbone:
            self.backbone = backbone
            self.use_skip = False
            self.skip_s2 = None
            self.skip_s1 = None
            self.stem = None
            self.stage1 = None
            self.stage2 = None
            self.stage3 = None
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width),
                _make_activation(activation),
            )
            self.use_skip = use_skip
            if self.use_skip:
                self.skip_s2 = nn.Conv2d(width, out_c, kernel_size=1, bias=False) if out_c != width else None
                self.skip_s1 = nn.Conv2d(width, out_c, kernel_size=1, bias=False) if out_c != width else None
            # Choose strides to reach the requested output stride (total downsample factor).
            if out_stride == 4:
                s1, s2, s3 = 2, 1, 1
            elif out_stride == 8:
                s1, s2, s3 = 2, 2, 1
            else:  # 16
                s1, s2, s3 = 2, 2, 2
            self.stage1 = DWConvBlock(width, width, stride=s1, activation=activation)
            self.stage2 = DWConvBlock(width, width, stride=s2, activation=activation)
            self.stage3 = DWConvBlock(width, out_c, stride=s3, activation=activation)
        if last_se == "se":
            self.se = SEModule(out_c)
        elif last_se == "ese":
            self.se = EfficientSEModule(out_c)
        else:
            self.se = None
        self.head_hm = nn.Conv2d(out_c, num_classes, kernel_size=1)
        self.head_off = nn.Conv2d(out_c, 2, kernel_size=1)
        self.head_wh = nn.Conv2d(out_c, 2, kernel_size=1)

    def forward(self, x: torch.Tensor, return_feat: bool = False) -> Dict[str, torch.Tensor]:
        if self.custom_backbone:
            s3 = self.backbone(x)
        else:
            x = self.stem(x)
            s1 = self.stage1(x)
            s2 = self.stage2(s1)
            s3 = self.stage3(s2)
            if self.use_skip:
                skip2 = F.adaptive_avg_pool2d(s2, s3.shape[2:])
                skip1 = F.adaptive_avg_pool2d(s1, s3.shape[2:])
                if self.skip_s2 is not None:
                    skip2 = self.skip_s2(skip2)
                if self.skip_s1 is not None:
                    skip1 = self.skip_s1(skip1)
                s3 = s3 + skip2 + skip1
        if self.se is not None:
            s3 = self.se(s3)
        feats = s3
        out = {
            "hm": torch.sigmoid(self.head_hm(s3)),
            "off": self.head_off(s3),
            "wh": self.head_wh(s3),
        }
        if return_feat:
            return out, feats
        return out


class AnchorCNN(nn.Module):
    """Lightweight anchor-based detector (YOLO-style head)."""

    def __init__(
        self,
        width: int = 32,
        num_classes: int = 1,
        num_anchors: int = 3,
        anchors: Tuple[Tuple[float, float], ...] = (),
        use_skip: bool = False,
        activation: str = "swish",
        last_se: str = "none",
        last_width_scale: float = 1.0,
        out_stride: int = 4,
        backbone: nn.Module = None,
        backbone_out_channels: int = None,
    ) -> None:
        super().__init__()
        if out_stride not in (4, 8, 16):
            raise ValueError(f"out_stride must be one of (4, 8, 16); got {out_stride}")
        last_scale = max(1.0, float(last_width_scale))
        self.custom_backbone = backbone is not None
        self.out_stride = getattr(backbone, "out_stride", out_stride) if self.custom_backbone else out_stride
        out_c = int(backbone_out_channels if backbone_out_channels is not None else round(width * last_scale))
        if self.custom_backbone:
            self.backbone = backbone
            self.use_skip = False
            self.skip_s2 = None
            self.skip_s1 = None
            self.stem = None
            self.stage1 = None
            self.stage2 = None
            self.stage3 = None
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(width),
                _make_activation(activation),
            )
            self.use_skip = use_skip
            if self.use_skip:
                self.skip_s2 = nn.Conv2d(width, out_c, kernel_size=1, bias=False) if out_c != width else None
                self.skip_s1 = nn.Conv2d(width, out_c, kernel_size=1, bias=False) if out_c != width else None
            if out_stride == 4:
                s1, s2, s3 = 2, 1, 1
            elif out_stride == 8:
                s1, s2, s3 = 2, 2, 1
            else:  # 16
                s1, s2, s3 = 2, 2, 2
            self.stage1 = DWConvBlock(width, width, stride=s1, activation=activation)
            self.stage2 = DWConvBlock(width, width, stride=s2, activation=activation)
            self.stage3 = DWConvBlock(width, out_c, stride=s3, activation=activation)
        if last_se == "se":
            self.se = SEModule(out_c)
        elif last_se == "ese":
            self.se = EfficientSEModule(out_c)
        else:
            self.se = None
        out_ch = num_anchors * (5 + num_classes)
        self.head = nn.Conv2d(out_c, out_ch, kernel_size=1)
        # anchors are normalized w,h pairs
        if anchors:
            anchor_tensor = torch.tensor(anchors, dtype=torch.float32)
        else:
            anchor_tensor = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float32)
        self.register_buffer("anchors", anchor_tensor)

    def set_anchors(self, anchors: torch.Tensor) -> None:
        if anchors is not None:
            self.anchors = anchors.to(self.anchors.device)

    def forward(self, x: torch.Tensor, return_feat: bool = False) -> torch.Tensor:
        if self.custom_backbone:
            s3 = self.backbone(x)
        else:
            x = self.stem(x)
            s1 = self.stage1(x)
            s2 = self.stage2(s1)
            s3 = self.stage3(s2)
            if self.use_skip:
                skip2 = F.adaptive_avg_pool2d(s2, s3.shape[2:])
                skip1 = F.adaptive_avg_pool2d(s1, s3.shape[2:])
                if self.skip_s2 is not None:
                    skip2 = self.skip_s2(skip2)
                if self.skip_s1 is not None:
                    skip1 = self.skip_s1(skip1)
                s3 = s3 + skip2 + skip1
        if self.se is not None:
            s3 = self.se(s3)
        feats = s3
        out = self.head(s3)
        if return_feat:
            return out, feats
        return out


def _get_2d_sincos_pos_embed(h: int, w: int, dim: int) -> torch.Tensor:
    y_embed = torch.arange(h, dtype=torch.float32).unsqueeze(1).repeat(1, w)
    x_embed = torch.arange(w, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
    omega = torch.arange(dim // 4, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 2)))
    out = []
    for embed in (x_embed, y_embed):
        out.append(torch.sin(embed.flatten()[:, None] * omega[None, :]))
        out.append(torch.cos(embed.flatten()[:, None] * omega[None, :]))
    pos = torch.cat(out, dim=1)
    return pos  # (h*w, dim)


class TinyDETR(nn.Module):
    """Very small DETR-like model."""

    def __init__(
        self,
        num_queries: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 128,
        num_classes: int = 1,
        activation: str = "swish",
        use_fpn: bool = False,
    ) -> None:
        super().__init__()
        self.use_fpn = use_fpn
        if self.use_fpn:
            self.stem = nn.Conv2d(3, d_model, kernel_size=3, stride=2, padding=1)
            self.fpn_high = DWConvBlock(d_model, d_model, stride=1, activation=activation)  # extra high-res stage
            self.fpn1 = DWConvBlock(d_model, d_model, stride=2, activation=activation)
            self.fpn2 = DWConvBlock(d_model, d_model, stride=2, activation=activation)
        else:
            # Higher-res single-scale patch embedding (stride 2 instead of 4)
            self.patch = nn.Conv2d(3, d_model, kernel_size=3, stride=2, padding=1)
        self.d_model = d_model
        self.num_classes = num_classes
        act = activation.lower()
        if act == "swish":
            transformer_activation = F.silu
        elif act == "relu":
            transformer_activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,
            activation=transformer_activation,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,
            activation=transformer_activation,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head = nn.Linear(d_model, num_classes + 1)  # classes + no-object
        self.box_head = nn.Linear(d_model, 4)  # cx, cy, w, h normalized

        self.pos_cache = {}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        if self.use_fpn:
            feats = []
            f0 = self.stem(x)  # stride 2
            f1 = self.fpn_high(f0)  # stride 2 refined
            f2 = self.fpn1(f1)  # stride 4
            f3 = self.fpn2(f2)  # stride 8
            feats.extend([f0, f1, f2, f3])
            src_list = []
            for feat in feats:
                h, w = feat.shape[2], feat.shape[3]
                pos = self._pos_encoding(h, w, feat.device).unsqueeze(1).repeat(1, b, 1)
                src_list.append(feat.flatten(2).permute(2, 0, 1) + pos)
            src = torch.cat(src_list, dim=0)
        else:
            feat = self.patch(x)  # B, C, 16, 16 for 64x64 input
            h, w = feat.shape[2], feat.shape[3]
            pos = self._pos_encoding(h, w, x.device).unsqueeze(1).repeat(1, b, 1)
            src = feat.flatten(2).permute(2, 0, 1) + pos

        memory = self.encoder(src)
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)
        hs = self.decoder(tgt, memory)
        logits = self.class_head(hs)  # Q, B, 2
        boxes = self.box_head(hs).sigmoid()  # normalized
        return logits, boxes

    def _pos_encoding(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, device.type)
        if key not in self.pos_cache or self.pos_cache[key].device != device:
            self.pos_cache[key] = _get_2d_sincos_pos_embed(h=h, w=w, dim=self.d_model).to(device)
        return self.pos_cache[key]


class _BackboneWithSE(nn.Module):
    """Wrapper to apply SE/eSE on backbone output."""

    def __init__(self, backbone: nn.Module, mode: str = "none"):
        super().__init__()
        self.backbone = backbone
        out_c = getattr(backbone, "out_channels", None)
        if mode == "se":
            self.se = SEModule(out_c)
        elif mode == "ese":
            self.se = EfficientSEModule(out_c)
        else:
            self.se = None
        self.out_channels = out_c
        self.out_stride = getattr(backbone, "out_stride", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if self.se is not None:
            x = self.se(x)
        return x


def _build_backbone(
    name: str,
    activation: str,
    backbone_channels=None,
    backbone_blocks=None,
    backbone_se: str = "none",
    backbone_skip: bool = False,
    backbone_skip_cat: bool = False,
    backbone_skip_shuffle_cat: bool = False,
    backbone_skip_s2d_cat: bool = False,
    backbone_fpn: bool = False,
    backbone_out_stride: int = None,
):
    if not name or str(name).lower() in ("none", "null"):
        return None, None
    name = name.lower()
    if name in ("microcspnet", "micro-cspnet", "micro_cspnet"):
        bb = MicroCSPNet(activation=activation)
    elif name in ("ultratinyresnet", "ultra-tiny-resnet", "ultra_tiny_resnet"):
        skip_mode = "add"
        if backbone_skip_s2d_cat:
            skip_mode = "s2d_cat"
        elif backbone_skip_shuffle_cat:
            skip_mode = "shuffle_cat"
        elif backbone_skip_cat:
            skip_mode = "cat"
        bb = UltraTinyResNet(
            activation=activation,
            channels=backbone_channels,
            blocks=backbone_blocks,
            use_long_skip=backbone_skip,
            skip_mode=skip_mode,
            use_fpn=backbone_fpn,
            target_out_stride=backbone_out_stride,
        )
    elif name in (
        "enhanced-shufflenet",
        "enhanced_shufflenet",
        "enhancedshufflenet",
    ):
        bb = EnhancedShuffleNet(activation=activation, use_global_skip=backbone_skip)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    se_mode = (backbone_se or "none").lower()
    if se_mode not in ("none", "se", "ese"):
        raise ValueError(f"Unknown backbone_se: {backbone_se}")
    if se_mode != "none":
        bb = _BackboneWithSE(bb, mode=se_mode)
    return bb, getattr(bb, "out_channels", None)


def _disable_bn(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.Identity())
        else:
            _disable_bn(child)


def build_model(arch: str, **kwargs) -> nn.Module:
    arch = arch.lower()
    use_batchnorm = kwargs.get("use_batchnorm", True)
    model: nn.Module
    if arch == "cnn":
        width = kwargs.get("width", 32)
        num_classes = kwargs.get("num_classes", 1)
        use_skip = kwargs.get("use_skip", False)
        activation = kwargs.get("activation", "swish")
        last_se = kwargs.get("last_se", "none")
        last_width_scale = kwargs.get("last_width_scale", 1.0)
        out_stride = kwargs.get("output_stride", 4)
        backbone_name = kwargs.get("backbone")
        backbone_module, backbone_out_channels = (
            _build_backbone(
                backbone_name,
                activation=activation,
                backbone_channels=kwargs.get("backbone_channels"),
                backbone_blocks=kwargs.get("backbone_blocks"),
                backbone_se=kwargs.get("backbone_se", "none"),
                backbone_skip=kwargs.get("backbone_skip", False),
                backbone_skip_cat=kwargs.get("backbone_skip_cat", False),
                backbone_skip_shuffle_cat=kwargs.get("backbone_skip_shuffle_cat", False),
                backbone_skip_s2d_cat=kwargs.get("backbone_skip_s2d_cat", False),
                backbone_fpn=kwargs.get("backbone_fpn", False),
                backbone_out_stride=kwargs.get("backbone_out_stride"),
            )
            if backbone_name
            else (None, None)
        )
        if backbone_module is not None:
            out_stride = getattr(backbone_module, "out_stride", out_stride)
        if kwargs.get("use_anchor", False):
            model = AnchorCNN(
                width=width,
                num_classes=num_classes,
                num_anchors=kwargs.get("num_anchors", 3),
                anchors=kwargs.get("anchors", ()),
                use_skip=use_skip,
                activation=activation,
                last_se=last_se,
                last_width_scale=last_width_scale,
                out_stride=out_stride,
                backbone=backbone_module,
                backbone_out_channels=backbone_out_channels,
            )
        else:
            model = MiniCenterNet(
                width=width,
                num_classes=num_classes,
                use_skip=use_skip,
                activation=activation,
                last_se=last_se,
                last_width_scale=last_width_scale,
                out_stride=out_stride,
                backbone=backbone_module,
                backbone_out_channels=backbone_out_channels,
            )
    elif arch == "ultratinyod":
        cfg = UltraTinyODConfig(
            num_classes=kwargs.get("num_classes", 1),
            stride=kwargs.get("output_stride", 8) or 8,
            anchors=kwargs.get("anchors") or None,
            use_improved_head=bool(kwargs.get("use_improved_head", False)),
        )
        stem_width = kwargs.get("c_stem", kwargs.get("width", 16))
        model = UltraTinyOD(
            num_classes=cfg.num_classes,
            config=cfg,
            c_stem=int(stem_width),
            use_residual=kwargs.get("utod_use_residual", False),
            use_improved_head=bool(kwargs.get("use_improved_head", False)),
        )
    elif arch == "transformer":
        model = TinyDETR(
            num_queries=kwargs.get("num_queries", 10),
            d_model=kwargs.get("d_model", 64),
            nhead=kwargs.get("heads", 4),
            num_encoder_layers=kwargs.get("encoder_layers", kwargs.get("layers", 3)),
            num_decoder_layers=kwargs.get("decoder_layers", kwargs.get("layers", 3)),
            dim_feedforward=kwargs.get("dim_feedforward", 128),
            num_classes=kwargs.get("num_classes", 1),
            activation=kwargs.get("activation", "swish"),
            use_fpn=kwargs.get("use_fpn", False),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    if not use_batchnorm:
        _disable_bn(model)
    return model
