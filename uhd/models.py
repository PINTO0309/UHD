from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(name: str) -> nn.Module:
    act = name.lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "swish":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class DWConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, activation: str = "swish") -> None:
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class MiniCenterNet(nn.Module):
    """Minimal anchor-free detector with heatmap + offsets + size."""

    def __init__(self, width: int = 32, num_classes: int = 1, use_skip: bool = False, activation: str = "swish") -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            _make_activation(activation),
        )
        self.use_skip = use_skip
        self.stage1 = DWConvBlock(width, width, stride=2, activation=activation)  # 64 -> 32
        self.stage2 = DWConvBlock(width, width, stride=2, activation=activation)  # 32 -> 16
        self.stage3 = DWConvBlock(width, width, stride=2, activation=activation)  # 16 -> 8
        self.head_hm = nn.Conv2d(width, num_classes, kernel_size=1)
        self.head_off = nn.Conv2d(width, 2, kernel_size=1)
        self.head_wh = nn.Conv2d(width, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        if self.use_skip:
            s3 = s3 + F.adaptive_avg_pool2d(s2, s3.shape[2:]) + F.adaptive_avg_pool2d(s1, s3.shape[2:])
        return {
            "hm": torch.sigmoid(self.head_hm(s3)),
            "off": self.head_off(s3),
            "wh": self.head_wh(s3),
        }


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


def build_model(arch: str, **kwargs) -> nn.Module:
    arch = arch.lower()
    if arch == "cnn":
        return MiniCenterNet(
            width=kwargs.get("width", 32),
            num_classes=kwargs.get("num_classes", 1),
            use_skip=kwargs.get("use_skip", False),
            activation=kwargs.get("activation", "swish"),
        )
    if arch == "transformer":
        return TinyDETR(
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
    raise ValueError(f"Unknown architecture: {arch}")
