from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_2d_sincos_pos_embed(h: int, w: int, dim: int) -> torch.Tensor:
    """2D sine-cosine positional embedding, not trainable."""
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


class RopeStub(nn.Module):
    """Placeholder to absorb rope_embed.periods weights from checkpoints."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.periods = nn.Parameter(torch.ones(dim))


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class DinoAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # present in checkpoints; we simply add it to the bias term
        self.qkv_bias_mask = nn.Parameter(torch.zeros(dim * 3))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,N,C
        qkv = self.qkv(x)
        if self.qkv_bias_mask is not None:
            qkv = qkv + self.qkv_bias_mask
        B, N, _ = qkv.shape
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.proj(out)


class DinoMLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class DinoGatedMLP(nn.Module):
    """3-layer gated MLP used by some DINO checkpoints (w1, w2, w3)."""

    def __init__(self, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, mlp_dim)
        self.w2 = nn.Linear(dim, mlp_dim)
        self.w3 = nn.Linear(mlp_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = self.w2(x)
        return self.w3(F.gelu(a) * b)


class DinoBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, use_glu: bool = False) -> None:
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DinoAttention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DinoGatedMLP(dim, mlp_dim) if use_glu else DinoMLP(dim, mlp_dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoV3Backbone(nn.Module):
    """Minimal ViT-style backbone to consume DINOv3 weights for feature distillation."""

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_glu: bool = False,
        add_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.add_pos_embed = add_pos_embed
        self.patch_embed = PatchEmbed(3, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, 4, embed_dim))
        self.rope_embed = RopeStub()
        self.blocks = nn.ModuleList([DinoBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_glu=use_glu) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.img_size = img_size
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.storage_tokens, std=0.02)
        nn.init.constant_(self.rope_embed.periods, 1.0)

    def _pos_embed(self, h: int, w: int, device) -> torch.Tensor:
        if not self.add_pos_embed:
            return None
        pos = _get_2d_sincos_pos_embed(h, w, self.embed_dim).to(device)
        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W -> features B,C,H',W'
        B = x.shape[0]
        x = self.patch_embed(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = self._pos_embed(h, w, x.device)
        if pos is not None:
            pos_cls = torch.zeros(1, 1, self.embed_dim, device=x.device, dtype=x.dtype)
            pos_full = torch.cat([pos_cls, pos.unsqueeze(0)], dim=1)
            x = x + pos_full
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        patch_tokens = x[:, 1:, :]
        feat = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)
        return feat


def _infer_dinov3_config(state: Dict[str, torch.Tensor], arch_hint: str = None) -> Dict[str, int]:
    embed_dim = int(state["cls_token"].shape[-1])
    patch = int(state["patch_embed.proj.weight"].shape[-1])
    blocks = [int(k.split(".")[1]) for k in state.keys() if k.startswith("blocks.") and k.split(".")[1].isdigit()]
    depth = max(blocks) + 1 if blocks else 12
    mlp_glu = any(".mlp.w3.weight" in k for k in state.keys())
    # use arch hint to override heads when known, else default to dim/64
    num_heads = embed_dim // 64 if embed_dim % 64 == 0 else 6
    if arch_hint:
        hint = arch_hint.lower()
        if "vitb" in hint:
            num_heads = 12
        elif "vits" in hint:
            num_heads = 6
    return {"embed_dim": embed_dim, "patch_size": patch, "depth": depth, "num_heads": num_heads, "mlp_ratio": 4.0, "use_glu": mlp_glu}


def load_dinov3_backbone(
    ckpt_path: str,
    img_size: Tuple[int, int],
    device: torch.device,
    arch_hint: str = None,
    add_pos_embed: bool = True,
) -> Tuple[DinoV3Backbone, Dict[str, int]]:
    state = torch.load(ckpt_path, map_location="cpu")
    # Normalize key naming for qkv bias mask if present
    if any("attn.qkv.bias_mask" in k for k in state.keys()):
        new_state = {}
        for k, v in state.items():
            if "attn.qkv.bias_mask" in k:
                new_state[k.replace("attn.qkv.bias_mask", "attn.qkv_bias_mask")] = v
            else:
                new_state[k] = v
        state = new_state
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} must be a state_dict-like mapping.")
    cfg = _infer_dinov3_config(state, arch_hint=arch_hint)
    model = DinoV3Backbone(
        img_size=img_size,
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        use_glu=cfg["use_glu"],
        add_pos_embed=add_pos_embed,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[DINOv3] Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[DINOv3] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    cfg["out_stride"] = cfg["patch_size"]
    return model, cfg
