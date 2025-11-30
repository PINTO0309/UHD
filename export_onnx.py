import argparse
import os

import onnx
import torch
from onnxsim import simplify

from uhd.models import build_model


def parse_img_size(arg: str):
    if isinstance(arg, (tuple, list)):
        if len(arg) != 2:
            raise ValueError("img-size tuple/list must have length 2 (H, W).")
        return int(arg[0]), int(arg[1])
    s = str(arg).lower().replace(" ", "")
    if "x" in s:
        parts = s.split("x")
        if len(parts) != 2:
            raise ValueError("img-size must be HxW, e.g., 64x64.")
        return int(parts[0]), int(parts[1])
    val = int(float(s))
    return val, val


def _parse_int_list(arg):
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        return [int(x) for x in arg]
    s = str(arg).replace(" ", "")
    if s == "":
        return None
    return [int(p) for p in s.split(",") if p != ""]


class CnnWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["hm"], out["off"], out["wh"]


class TransformerWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, boxes = self.model(x)
        return logits, boxes


class AnchorWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def _infer_transformer_config(state_dict, fallback_layers: int, fallback_num_queries: int, fallback_d_model: int):
    num_queries = fallback_num_queries
    d_model = fallback_d_model
    layers = fallback_layers
    # query_embed.weight: (num_queries, d_model)
    qe = state_dict.get("query_embed.weight")
    if qe is not None:
        num_queries = qe.shape[0]
        d_model = qe.shape[1]
    # encoder.layers.N.  pick max index + 1
    max_enc = -1
    max_dec = -1
    for k in state_dict.keys():
        if k.startswith("encoder.layers."):
            try:
                idx = int(k.split(".")[2])
                max_enc = max(max_enc, idx)
            except (IndexError, ValueError):
                pass
        if k.startswith("decoder.layers."):
            try:
                idx = int(k.split(".")[2])
                max_dec = max(max_dec, idx)
            except (IndexError, ValueError):
                pass
    if max_enc >= 0:
        layers = max_enc + 1
    if max_dec >= 0:
        layers = max(layers, max_dec + 1)
    return num_queries, d_model, layers


def _infer_cnn_width(state_dict, fallback_width: int) -> int:
    """Try to recover CNN width from checkpoint weights when metadata is missing."""
    if not isinstance(state_dict, dict):
        return int(fallback_width)
    # Prefer stem conv if present
    for key in ("stem.0.weight", "stem.weight", "stage1.dw.weight", "stage1.pw.weight"):
        w = state_dict.get(key)
        if isinstance(w, torch.Tensor):
            return int(w.shape[0])
    # Head conv stores width on channel dimension 1
    w = state_dict.get("head_hm.weight")
    if isinstance(w, torch.Tensor) and w.dim() > 1:
        return int(w.shape[1])
    w = state_dict.get("head.weight")
    if isinstance(w, torch.Tensor) and w.dim() > 1:
        return int(w.shape[1])
    return int(fallback_width)

def _infer_num_anchors(state_dict, num_classes: int, fallback: int) -> int:
    """Infer num_anchors from head shape when using anchor head."""
    if not isinstance(state_dict, dict):
        return int(fallback)
    w = state_dict.get("head.weight")
    if isinstance(w, torch.Tensor) and w.dim() == 4 and w.shape[2:] == (1, 1):
        out_ch = w.shape[0]
        denom = 5 + num_classes
        if out_ch % denom == 0:
            return int(out_ch // denom)
    return int(fallback)


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX (auto-detect arch).")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--output", default=None, help="Output ONNX path. Default: <checkpoint>.onnx")
    parser.add_argument("--arch", default=None, help="Override architecture (cnn/transformer).")
    parser.add_argument("--img-size", default="64x64", help="Input size HxW, e.g., 64x64.")
    parser.add_argument("--cnn-width", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--activation", choices=["relu", "swish"], default="swish", help="Activation function to use.")
    parser.add_argument("--use-skip", action="store_true", help="Enable skip connections for CNN export (defaults to checkpoint flag).")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights from checkpoint if available.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic height/width axes.")
    parser.add_argument("--topk", type=int, default=100, help="Top-K for postprocess (CNN).")
    parser.add_argument("--merge-postprocess", action="store_true", help="Export with postprocess merged.")
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size when dynamic is not set.")
    parser.add_argument("--use-anchor", action="store_true", help="Force anchor-based CNN head (overrides checkpoint flag).")
    parser.add_argument("--last-se", choices=["none", "se", "ese"], default=None, help="Override last SE mode for CNN (defaults to checkpoint).")
    parser.add_argument("--last-width-scale", type=float, default=None, help="Override last width scale for CNN (defaults to checkpoint).")
    parser.add_argument("--output-stride", type=int, default=None, help="Override CNN output stride (defaults to checkpoint).")
    parser.add_argument(
        "--backbone",
        default=None,
        choices=["microcspnet", "ultratinyresnet", "enhanced-shufflenet", "none", None],
        help="Optional lightweight CNN backbone (defaults to checkpoint).",
    )
    parser.add_argument("--backbone-channels", default=None, help="Comma-separated channels for ultratinyresnet (e.g., '16,24,32,48').")
    parser.add_argument("--backbone-blocks", default=None, help="Comma-separated residual block counts per stage for ultratinyresnet (e.g., '1,1,2,1').")
    parser.add_argument("--backbone-se", choices=["none", "se", "ese"], default=None, help="Apply SE/eSE on backbone output (custom backbones only).")
    parser.add_argument("--backbone-skip", action="store_true", help="Add long skip fusion across custom backbone stages (ultratinyresnet).")
    parser.add_argument("--backbone-skip-cat", action="store_true", help="Use concat+1x1 fusion for long skips (ultratinyresnet); implies --backbone-skip.")
    parser.add_argument(
        "--backbone-skip-shuffle-cat",
        action="store_true",
        help="Use stride+shuffle concat fusion for long skips (ultratinyresnet); implies --backbone-skip.",
    )
    parser.add_argument(
        "--backbone-skip-s2d-cat",
        action="store_true",
        help="Use space-to-depth concat fusion for long skips (ultratinyresnet); implies --backbone-skip.",
    )
    parser.add_argument("--backbone-fpn", action="store_true", help="Enable a tiny FPN fusion inside custom backbones (ultratinyresnet).")
    parser.add_argument("--backbone-out-stride", type=int, default=None, help="Override custom backbone output stride (e.g., 8 or 16).")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    arch = (args.arch or ckpt.get("arch", "cnn")).lower()
    ckpt_use_skip = bool(ckpt.get("use_skip", False))
    use_skip = ckpt_use_skip or bool(args.use_skip)
    use_fpn = bool(ckpt.get("use_fpn", False))
    backbone = args.backbone if args.backbone not in ("none", None) else None
    backbone_channels = _parse_int_list(args.backbone_channels)
    backbone_blocks = _parse_int_list(args.backbone_blocks)
    backbone_se = args.backbone_se
    backbone_skip = bool(args.backbone_skip)
    backbone_skip_cat = bool(args.backbone_skip_cat)
    backbone_skip_shuffle_cat = bool(args.backbone_skip_shuffle_cat)
    backbone_skip_s2d_cat = bool(args.backbone_skip_s2d_cat)
    if backbone_skip_s2d_cat:
        backbone_skip = True
        backbone_skip_shuffle_cat = False
        backbone_skip_cat = False
    elif backbone_skip_shuffle_cat:
        backbone_skip = True
        backbone_skip_cat = False
    elif backbone_skip_cat:
        backbone_skip = True
    backbone_fpn = bool(args.backbone_fpn)
    backbone_out_stride = int(args.backbone_out_stride) if args.backbone_out_stride is not None else None
    if backbone is None:
        backbone = ckpt.get("backbone")
    if backbone_channels is None:
        backbone_channels = _parse_int_list(ckpt.get("backbone_channels"))
    if backbone_blocks is None:
        backbone_blocks = _parse_int_list(ckpt.get("backbone_blocks"))
    if backbone_se is None:
        backbone_se = ckpt.get("backbone_se", "none")
    if "backbone_skip" in ckpt:
        backbone_skip = bool(ckpt.get("backbone_skip"))
    if "backbone_skip_cat" in ckpt:
        backbone_skip_cat = bool(ckpt.get("backbone_skip_cat"))
    if "backbone_skip_shuffle_cat" in ckpt:
        backbone_skip_shuffle_cat = bool(ckpt.get("backbone_skip_shuffle_cat"))
        if backbone_skip_shuffle_cat:
            backbone_skip = True
            backbone_skip_cat = False
    if "backbone_skip_s2d_cat" in ckpt:
        backbone_skip_s2d_cat = bool(ckpt.get("backbone_skip_s2d_cat"))
        if backbone_skip_s2d_cat:
            backbone_skip = True
            backbone_skip_shuffle_cat = False
            backbone_skip_cat = False
    if backbone_skip_cat and not (backbone_skip_shuffle_cat or backbone_skip_s2d_cat):
        backbone_skip = True
    if "backbone_fpn" in ckpt:
        backbone_fpn = bool(ckpt.get("backbone_fpn"))
    if backbone_out_stride is None and "backbone_out_stride" in ckpt and ckpt["backbone_out_stride"] is not None:
        backbone_out_stride = int(ckpt["backbone_out_stride"])
    if backbone in ("none", "") or arch != "cnn":
        backbone = None
        backbone_channels = None
        backbone_blocks = None
        backbone_se = "none"
        backbone_skip = False
        backbone_skip_cat = False
        backbone_skip_shuffle_cat = False
        backbone_skip_s2d_cat = False
        backbone_fpn = False
        backbone_out_stride = None
    activation = args.activation
    ckpt_activation = ckpt.get("activation")
    if ckpt_activation and ckpt_activation != activation:
        print(f"Overriding CLI activation={activation} with checkpoint activation={ckpt_activation}")
        activation = ckpt_activation
    classes = ckpt.get("classes", [0])
    num_classes = len(classes) if isinstance(classes, (list, tuple)) else int(classes)
    use_anchor = bool(ckpt.get("use_anchor", False) or args.use_anchor)
    anchors = ckpt.get("anchors", [])
    num_anchors = ckpt.get("num_anchors", None)
    last_se = args.last_se or ckpt.get("last_se", "none")
    last_width_scale = args.last_width_scale if args.last_width_scale is not None else ckpt.get("last_width_scale", 1.0)
    output_stride = args.output_stride if args.output_stride is not None else ckpt.get("output_stride", 4)
    if backbone is not None and backbone_out_stride is not None:
        output_stride = backbone_out_stride
    # Prefer checkpoint hyper-params when available; infer layers/queries/d_model from state_dict if missing
    num_queries = ckpt.get("num_queries")
    d_model = ckpt.get("d_model")
    layers = ckpt.get("layers")
    heads = ckpt.get("heads", args.heads)
    dim_feedforward = ckpt.get("dim_feedforward", args.dim_feedforward)
    if arch == "transformer":
        num_queries, d_model, layers = _infer_transformer_config(
            ckpt.get("model", {}), fallback_layers=layers or args.layers, fallback_num_queries=num_queries or args.num_queries, fallback_d_model=d_model or args.d_model
        )
    num_queries = int(num_queries if num_queries is not None else args.num_queries)
    d_model = int(d_model if d_model is not None else args.d_model)
    layers = int(layers if layers is not None else args.layers)
    heads = int(heads if heads is not None else args.heads)
    dim_feedforward = int(dim_feedforward if dim_feedforward is not None else args.dim_feedforward)
    h, w = parse_img_size(args.img_size)
    # Pick the state dict we will actually load (model vs EMA) for width inference as well
    state_dict = None
    if args.use_ema and "ema" in ckpt and ckpt["ema"] is not None:
        state_dict = ckpt["ema"]
    else:
        state_dict = ckpt["model"]
    width = ckpt.get("width")
    if arch == "cnn":
        width = _infer_cnn_width(state_dict, fallback_width=width or args.cnn_width)
        if num_anchors is None:
            if anchors:
                num_anchors = len(anchors)
            else:
                num_anchors = _infer_num_anchors(state_dict, num_classes=num_classes, fallback=3)
    width = int(width if width is not None else args.cnn_width)
    num_anchors = int(num_anchors if num_anchors is not None else 3)
    if anchors and len(anchors) != num_anchors:
        if len(anchors) > num_anchors:
            anchors = anchors[:num_anchors]
        else:
            anchors = list(anchors) + [anchors[-1]] * (num_anchors - len(anchors))

    model = build_model(
        arch,
        width=width,
        num_queries=num_queries,
        d_model=d_model,
        heads=heads,
        layers=layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        activation=activation,
        use_skip=use_skip,
        use_fpn=use_fpn,
        use_anchor=use_anchor,
        num_anchors=num_anchors,
        anchors=anchors,
        last_se=last_se,
        last_width_scale=last_width_scale,
        output_stride=output_stride,
        backbone=backbone,
        backbone_channels=backbone_channels,
        backbone_blocks=backbone_blocks,
        backbone_se=backbone_se,
        backbone_skip=backbone_skip,
        backbone_skip_cat=backbone_skip_cat,
        backbone_skip_shuffle_cat=backbone_skip_shuffle_cat,
        backbone_skip_s2d_cat=backbone_skip_s2d_cat,
        backbone_fpn=backbone_fpn,
        backbone_out_stride=backbone_out_stride,
    )
    output_stride = getattr(model, "out_stride", output_stride)
    output_stride = getattr(model, "out_stride", output_stride)
    model.load_state_dict(state_dict)
    model.eval()

    if arch == "cnn":
        if use_anchor:
            wrapper = AnchorWrapper(model)
            if args.merge_postprocess:
                class PostAnchor(torch.nn.Module):
                    def __init__(self, net, anchors, k):
                        super().__init__()
                        self.net = net
                        self.register_buffer("anchors", anchors if isinstance(anchors, torch.Tensor) else torch.tensor(anchors, dtype=torch.float32))
                        self.anchors: torch.Tensor
                        self.k = k

                    def forward(self, x):
                        pred: torch.Tensor = self.net(x)  # B x (A*(5+C)) x H x W
                        b, _, h, w = pred.shape
                        na = self.anchors.shape[0]
                        # reshape to B x A x H x W x (5+C)
                        num_classes = pred.shape[1] // na - 5
                        pred = pred.view(b, na, 5 + num_classes, h, w).permute(0, 1, 3, 4, 2)
                        tx, ty, tw, th = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
                        obj = pred[..., 4:5].sigmoid()
                        cls = pred[..., 5:6].sigmoid()
                        gy, gx = torch.meshgrid(torch.arange(h, device=pred.device), torch.arange(w, device=pred.device), indexing="ij")
                        gx = gx.view(1, 1, h, w)
                        gy = gy.view(1, 1, h, w)
                        pred_cx = (tx.sigmoid() + gx) / float(w)
                        pred_cy = (ty.sigmoid() + gy) / float(h)
                        pred_w = self.anchors[:, 0].view(1, na, 1, 1) * tw.exp()
                        pred_h = self.anchors[:, 1].view(1, na, 1, 1) * th.exp()

                        scores_all = obj * cls  # B x A x H x W x C
                        max_scores, max_cls = scores_all.max(dim=-1)  # B x A x H x W
                        n, c, h, w = max_scores.shape
                        flat_scores = max_scores.reshape(n, c*h*w)
                        k = min(self.k, flat_scores.shape[1])
                        scores, idxs = torch.topk(flat_scores, k=k, dim=1)
                        # flatten per-map while keeping batch dimension
                        n, c, h, w = max_cls.shape
                        flat_cls = max_cls.reshape(n, c*h*w)
                        n, c, h, w = pred_cx.shape
                        flat_cx = pred_cx.reshape(n, c*h*w)
                        n, c, h, w = pred_cy.shape
                        flat_cy = pred_cy.reshape(n, c*h*w)
                        n, c, h, w = pred_w.shape
                        flat_pw = pred_w.reshape(n, c*h*w)
                        n, c, h, w = pred_h.shape
                        flat_ph = pred_h.reshape(n, c*h*w)
                        cls_topk = torch.gather(flat_cls, 1, idxs)
                        cx = torch.gather(flat_cx, 1, idxs)
                        cy = torch.gather(flat_cy, 1, idxs)
                        pw = torch.gather(flat_pw, 1, idxs)
                        ph = torch.gather(flat_ph, 1, idxs)
                        dets = torch.stack([scores, cls_topk.float(), cx, cy, pw, ph], dim=-1)
                        return dets

                anchors_tensor = model.anchors if hasattr(model, "anchors") else torch.tensor(anchors if anchors else [[0.1, 0.1]], dtype=torch.float32)
                wrapper = PostAnchor(wrapper, anchors_tensor, args.topk)
                output_names = ["detections"]
            else:
                output_names = ["pred"]
        else:
            wrapper = CnnWrapper(model)
            if args.merge_postprocess:
                class PostCNN(torch.nn.Module):
                    def __init__(self, net, k):
                        super().__init__()
                        self.net = net
                        self.k = k

                    def forward(self, x):
                        hm, off, wh = self.net(x)
                        b, c, h, w = hm.shape
                        hm_flat = hm.view(b, -1)
                        k = min(self.k, hm_flat.shape[1])
                        scores, inds = torch.topk(hm_flat, k=k, dim=1)
                        cls = inds // (h * w)
                        rem = inds % (h * w)
                        ys = rem // w
                        xs = rem % w
                        off_flat = off.view(b, 2, -1)
                        wh_flat = wh.view(b, 2, -1)
                        off_g = torch.gather(off_flat, 2, inds.unsqueeze(1).expand(-1, 2, -1))
                        wh_g = torch.gather(wh_flat, 2, inds.unsqueeze(1).expand(-1, 2, -1))
                        cx = (xs.float() + off_g[:, 0, :]) / w
                        cy = (ys.float() + off_g[:, 1, :]) / h
                        bw = wh_g[:, 0, :] / w
                        bh = wh_g[:, 1, :] / h
                        dets = torch.stack([scores, cls.float(), cx, cy, bw, bh], dim=-1)
                        return dets

                wrapper = PostCNN(wrapper, args.topk)
                output_names = ["detections"]
            else:
                output_names = ["hm", "off", "wh"]
    elif arch == "transformer":
        wrapper = TransformerWrapper(model)
        if args.merge_postprocess:
            class PostTF(torch.nn.Module):
                def __init__(self, net, k):
                    super().__init__()
                    self.net = net
                    self.k = k

                def forward(self, x):
                    logits, boxes = self.net(x)  # Q,B,C+1 / Q,B,4
                    probs = torch.softmax(logits, dim=-1)
                    cls_prob, cls = probs[..., :-1].max(dim=-1)  # Q,B
                    boxes = boxes  # Q,B,4 normalized
                    cls_prob = cls_prob.permute(1, 0)  # B,Q
                    cls = cls.permute(1, 0)  # B,Q
                    boxes = boxes.permute(1, 0, 2)  # B,Q,4
                    k = min(self.k, cls_prob.shape[1])
                    scores, idxs = torch.topk(cls_prob, k=k, dim=1)
                    # gather boxes and classes
                    idxs_exp = idxs.unsqueeze(-1).expand(-1, -1, 4)  # B,k,4
                    boxes_topk = torch.gather(boxes, 1, idxs_exp)
                    cls_topk = torch.gather(cls, 1, idxs)
                    dets = torch.stack(
                        [scores, cls_topk.float(), boxes_topk[..., 0], boxes_topk[..., 1], boxes_topk[..., 2], boxes_topk[..., 3]],
                        dim=-1,
                    )  # B,k,6
                    return dets

            wrapper = PostTF(wrapper, args.topk)
            output_names = ["detections"]
        else:
            output_names = ["logits", "boxes"]
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    dummy = torch.zeros(args.batch_size if not args.dynamic else 1, 3, h, w, dtype=torch.float32)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    onnx_path = args.output or os.path.splitext(args.checkpoint)[0] + ".onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=output_names,
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
    )
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model, dynamic_input_shape=bool(args.dynamic))
    if not check:
        raise RuntimeError("onnx-simplifier check failed")
    # Rename detection dim-1 axis to "N"
    if args.merge_postprocess:
        def set_dim(param_name: str):
            dim = param_name
            for out in model_simp.graph.output:
                if out.name == "detections":
                    shape = out.type.tensor_type.shape.dim
                    if len(shape) > 1:
                        shape[1].dim_param = dim
            for vi in model_simp.graph.value_info:
                if vi.name == "detections":
                    shape = vi.type.tensor_type.shape.dim
                    if len(shape) > 1:
                        shape[1].dim_param = dim

        # set_dim("N")
    onnx.save(model_simp, onnx_path)
    print(f"Exported and simplified {arch} model to {onnx_path} (opset {args.opset}, dynamic={args.dynamic})")


if __name__ == "__main__":
    main()
