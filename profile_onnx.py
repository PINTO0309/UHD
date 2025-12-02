import argparse
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import onnx
import numpy as np


def _dim_to_int(dim) -> Optional[int]:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    return None


def _value_info_shape(value_info) -> Optional[List[int]]:
    if value_info is None or not value_info.type.HasField("tensor_type"):
        return None
    shape = []
    for d in value_info.type.tensor_type.shape.dim:
        shape.append(_dim_to_int(d))
    return shape


def _numel(shape: Sequence[Optional[int]]) -> Optional[int]:
    if not shape or any(d is None for d in shape):
        return None
    return int(np.prod(shape))


def _override_input_shape(model: onnx.ModelProto, shape: Sequence[int]) -> None:
    """Force the first graph input to a concrete shape."""
    if not model.graph.input:
        return
    tensor_shape = model.graph.input[0].type.tensor_type.shape
    if len(shape) != len(tensor_shape.dim):
        raise ValueError(f"Provided shape rank {len(shape)} does not match model input rank {len(tensor_shape.dim)}")
    for d, new_v in zip(tensor_shape.dim, shape):
        d.ClearField("dim_param")
        d.dim_value = int(new_v)


def _collect_shapes(model: onnx.ModelProto) -> Dict[str, List[Optional[int]]]:
    shape_map: Dict[str, List[Optional[int]]] = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        s = _value_info_shape(vi)
        if s is not None:
            shape_map[vi.name] = s
    for init in model.graph.initializer:
        shape_map[init.name] = list(init.dims)
    return shape_map


def _get_attr(node: onnx.NodeProto, name: str, default=None):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return int(attr.i)
            if attr.type == onnx.AttributeProto.FLOAT:
                return float(attr.f)
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)
    return default


def _conv_flops(node: onnx.NodeProto, shape_map: Dict[str, List[Optional[int]]]) -> Optional[int]:
    if len(node.input) < 2:
        return None
    x_shape = shape_map.get(node.input[0])
    w_shape = shape_map.get(node.input[1])
    y_shape = shape_map.get(node.output[0])
    if not x_shape or not w_shape or not y_shape:
        return None
    groups = _get_attr(node, "group", 1)
    # ONNX Conv weights: [out_c, in_c/group, kH, kW]
    out_c, in_c_per_group, k_h, k_w = w_shape[:4]
    batch, _, out_h, out_w = y_shape[:4]
    if None in (batch, out_c, in_c_per_group, k_h, k_w, out_h, out_w):
        return None
    ops_per_out = 2 * in_c_per_group * k_h * k_w
    total = batch * out_c * out_h * out_w * ops_per_out
    # optional bias add per output element
    if len(node.input) >= 3 and node.input[2] in shape_map:
        total += batch * out_c * out_h * out_w
    return int(total)


def _conv_transpose_flops(node: onnx.NodeProto, shape_map: Dict[str, List[Optional[int]]]) -> Optional[int]:
    if len(node.input) < 2:
        return None
    w_shape = shape_map.get(node.input[1])
    y_shape = shape_map.get(node.output[0])
    if not w_shape or not y_shape:
        return None
    groups = _get_attr(node, "group", 1)
    in_c, out_c_per_group, k_h, k_w = w_shape[:4]
    batch, out_c, out_h, out_w = y_shape[:4]
    if None in (batch, in_c, out_c_per_group, k_h, k_w, out_h, out_w):
        return None
    ops_per_out = 2 * in_c * k_h * k_w / groups
    total = batch * out_c * out_h * out_w * ops_per_out
    return int(total)


def _matmul_flops(a_shape: List[Optional[int]], b_shape: List[Optional[int]], out_shape: List[Optional[int]]) -> Optional[int]:
    if not a_shape or not b_shape or not out_shape:
        return None
    if len(a_shape) < 2 or len(b_shape) < 2 or len(out_shape) < 2:
        return None
    m = a_shape[-2]
    k = a_shape[-1]
    n = b_shape[-1]
    if None in (m, k, n):
        return None
    batch = int(np.prod([d for d in out_shape[:-2] if d is not None])) if out_shape[:-2] else 1
    return int(batch * m * n * k * 2)


def _gemm_flops(node: onnx.NodeProto, shape_map: Dict[str, List[Optional[int]]]) -> Optional[int]:
    a_shape = shape_map.get(node.input[0])
    b_shape = shape_map.get(node.input[1])
    if not a_shape or not b_shape:
        return None
    trans_a = bool(_get_attr(node, "transA", 0))
    trans_b = bool(_get_attr(node, "transB", 0))
    a_m, a_k = (a_shape[-1], a_shape[-2]) if trans_a else (a_shape[-2], a_shape[-1])
    b_k, b_n = (b_shape[-1], b_shape[-2]) if trans_b else (b_shape[-2], b_shape[-1])
    if None in (a_m, a_k, b_k, b_n) or a_k != b_k:
        return None
    return int(a_m * b_n * a_k * 2)


def _elementwise_flops(out_shape: List[Optional[int]]) -> Optional[int]:
    return _numel(out_shape)


def _pool_flops(node: onnx.NodeProto, shape_map: Dict[str, List[Optional[int]]]) -> Optional[int]:
    out_shape = shape_map.get(node.output[0])
    if not out_shape:
        return None
    k = _get_attr(node, "kernel_shape", None)
    if k:
        kernel_mul = int(np.prod(k))
    else:
        kernel_mul = 1
    out_elems = _numel(out_shape)
    if out_elems is None:
        return None
    return int(out_elems * kernel_mul)


def profile_flops(model: onnx.ModelProto, shape_map: Dict[str, List[Optional[int]]]) -> Tuple[int, Dict[str, int]]:
    total = 0
    per_op = defaultdict(int)
    elementwise_ops = {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Relu",
        "Sigmoid",
        "Tanh",
        "LeakyRelu",
        "Gelu",
        "Silu",
    }
    pool_ops = {"MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool"}

    for node in model.graph.node:
        flops = None
        if node.op_type == "Conv":
            flops = _conv_flops(node, shape_map)
        elif node.op_type == "ConvTranspose":
            flops = _conv_transpose_flops(node, shape_map)
        elif node.op_type == "MatMul":
            a_shape = shape_map.get(node.input[0])
            b_shape = shape_map.get(node.input[1])
            out_shape = shape_map.get(node.output[0])
            flops = _matmul_flops(a_shape, b_shape, out_shape)
        elif node.op_type == "Gemm":
            flops = _gemm_flops(node, shape_map)
        elif node.op_type in elementwise_ops:
            out_shape = shape_map.get(node.output[0])
            flops = _elementwise_flops(out_shape)
        elif node.op_type in pool_ops:
            flops = _pool_flops(node, shape_map)

        if flops is None:
            continue
        total += flops
        per_op[node.op_type] += flops

    return total, dict(per_op)


def count_parameters(model: onnx.ModelProto) -> int:
    return sum(int(np.prod(init.dims)) for init in model.graph.initializer)


def format_big(num: int, unit: str) -> str:
    if num >= 1e9:
        return f"{num/1e9:.3f} G{unit}"
    if num >= 1e6:
        return f"{num/1e6:.3f} M{unit}"
    if num >= 1e3:
        return f"{num/1e3:.3f} K{unit}"
    return f"{num} {unit}"


def parse_shape_arg(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    cleaned = arg.replace("x", ",").replace("X", ",")
    parts = [p for p in cleaned.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Compute FLOPs and parameter count for an ONNX model.")
    parser.add_argument("onnx_path", help="Path to ONNX file.")
    parser.add_argument(
        "--input-shape",
        default=None,
        help="Override first input shape, e.g., 1,3,64,64 or 1x3x64x64 (NCHW).",
    )
    parser.add_argument("--per-op", action="store_true", help="Print per-op FLOPs breakdown.")
    args = parser.parse_args()

    model = onnx.load(args.onnx_path)
    user_shape = parse_shape_arg(args.input_shape)
    if user_shape:
        _override_input_shape(model, user_shape)
    inferred = onnx.shape_inference.infer_shapes(model)
    shape_map = _collect_shapes(inferred)
    params = count_parameters(inferred)
    flops, per_op = profile_flops(inferred, shape_map)

    print(f"Params: {params:,} ({format_big(params, 'params')})")
    print(f"FLOPs:  {flops:,} ({flops/1e9:.3f} GFLOPs)")

    if args.per_op and per_op:
        print("\nPer-op breakdown:")
        for k, v in sorted(per_op.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {k:<16} {v:,} ({format_big(v, 'FLOPs')})")


if __name__ == "__main__":
    main()
