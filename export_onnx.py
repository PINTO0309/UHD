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
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic height/width axes.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    arch = (args.arch or ckpt.get("arch", "cnn")).lower()
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
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    if arch == "cnn":
        wrapper = CnnWrapper(model)
        output_names = ["hm", "off", "wh"]
    elif arch == "transformer":
        wrapper = TransformerWrapper(model)
        output_names = ["logits", "boxes"]
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    dummy = torch.zeros(1, 3, h, w, dtype=torch.float32)
    dynamic_axes = {"images": {0: "batch"}}
    if args.dynamic:
        dynamic_axes["images"].update({2: "height", 3: "width"})
        dynamic_axes[output_names[0]] = {0: "batch", 2: "height_out", 3: "width_out"} if arch == "cnn" else {0: "batch"}
        if arch == "cnn":
            dynamic_axes[output_names[1]] = {0: "batch", 2: "height_out", 3: "width_out"}
            dynamic_axes[output_names[2]] = {0: "batch", 2: "height_out", 3: "width_out"}
        else:
            dynamic_axes[output_names[1]] = {0: "batch"}
    else:
        dynamic_axes = {"images": {0: "batch"}, output_names[0]: {0: "batch"}, output_names[1]: {0: "batch"}}

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
    onnx.save(model_simp, onnx_path)
    print(f"Exported and simplified {arch} model to {onnx_path} (opset {args.opset}, dynamic={args.dynamic})")


if __name__ == "__main__":
    main()
