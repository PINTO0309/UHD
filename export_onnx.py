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
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    arch = (args.arch or ckpt.get("arch", "cnn")).lower()
    ckpt_use_skip = bool(ckpt.get("use_skip", False))
    use_skip = ckpt_use_skip or bool(args.use_skip)
    activation = args.activation
    ckpt_activation = ckpt.get("activation")
    if ckpt_activation and ckpt_activation != activation:
        print(f"Overriding CLI activation={activation} with checkpoint activation={ckpt_activation}")
        activation = ckpt_activation
    classes = ckpt.get("classes", [0])
    num_classes = len(classes) if isinstance(classes, (list, tuple)) else int(classes)
    h, w = parse_img_size(args.img_size)

    model = build_model(
        arch,
        width=args.cnn_width,
        num_queries=args.num_queries,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=num_classes,
        activation=activation,
        use_skip=use_skip,
    )
    if args.use_ema and "ema" in ckpt and ckpt["ema"] is not None:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    if arch == "cnn":
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

        set_dim("N")
    onnx.save(model_simp, onnx_path)
    print(f"Exported and simplified {arch} model to {onnx_path} (opset {args.opset}, dynamic={args.dynamic})")


if __name__ == "__main__":
    main()
