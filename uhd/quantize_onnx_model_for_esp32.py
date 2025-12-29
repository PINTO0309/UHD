import argparse
import glob
import os
import re
import tempfile

import cv2
import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader, Dataset
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.api.setting import QuantizationSettingFactory
from esp_ppq.core import TargetPlatform

try:
    from uhd.data import YoloDataset
    from uhd.resize import resize_image_numpy
except ImportError:
    from data import YoloDataset
    from resize import resize_image_numpy

DEFAULT_ONNX_MODEL_PATH = "ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx"
DEFAULT_ESPDL_MODEL_PATH = "ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.espdl"
DEFAULT_TARGET = "esp32s3"
DEFAULT_NUM_OF_BITS = 8
DEFAULT_DEVICE = "cpu"

DEVICE = DEFAULT_DEVICE


def sanitize_onnx_model(onnx_model_path, export_dir=None, batch_size=None, expand_group_conv=False):
    model = onnx.load(onnx_model_path)
    init_by_name = {init.name: init for init in model.graph.initializer}
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
        for name in ("anchors", "wh_scale"):
            init = init_by_name.get(name)
            if init is None:
                print(f"Initializer '{name}' not found; skipping export.")
                continue
            arr = onnx.numpy_helper.to_array(init)
            out_path = os.path.join(export_dir, f"{base_name}_{name}.npy")
            np.save(out_path, arr)
            print(f"Exported initializer '{name}' to: {out_path}")
    group_conv_updates = []
    if expand_group_conv:
        for node in model.graph.node:
            if node.op_type != "Conv":
                continue
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            groups = int(attrs.get("group", 1) or 1)
            if groups <= 1:
                continue
            if len(node.input) < 2:
                continue
            weight_name = node.input[1]
            weight_init = init_by_name.get(weight_name)
            if weight_init is None:
                continue
            weight = onnx.numpy_helper.to_array(weight_init)
            if weight.ndim != 4:
                continue
            in_per_group = weight.shape[1]
            in_channels = in_per_group * groups
            out_channels = weight.shape[0]
            if out_channels % groups != 0:
                continue
            out_per_group = out_channels // groups
            new_weight = np.zeros((out_channels, in_channels, weight.shape[2], weight.shape[3]), dtype=weight.dtype)
            for group_idx in range(groups):
                out_start = group_idx * out_per_group
                in_start = group_idx * in_per_group
                new_weight[
                    out_start : out_start + out_per_group,
                    in_start : in_start + in_per_group,
                    :,
                    :,
                ] = weight[out_start : out_start + out_per_group, :, :, :]
            weight_init.CopyFrom(onnx.numpy_helper.from_array(new_weight, weight_name))
            kept_attrs = [attr for attr in node.attribute if attr.name != "group"]
            del node.attribute[:]
            node.attribute.extend(kept_attrs)
            node.attribute.extend([onnx.helper.make_attribute("group", 1)])
            group_conv_updates.append(node.name or weight_name)
    resize_updates = []
    if batch_size is not None:
        for node in model.graph.node:
            if node.op_type != "Resize":
                continue
            if len(node.input) < 4:
                continue
            size_name = node.input[3]
            init = init_by_name.get(size_name)
            if init is None:
                continue
            arr = onnx.numpy_helper.to_array(init)
            if arr.ndim != 1 or arr.size != 4:
                continue
            old_n = int(arr[0])
            if old_n > 0 and old_n != batch_size:
                arr = arr.copy()
                arr[0] = batch_size
                init.CopyFrom(onnx.numpy_helper.from_array(arr, init.name))
                resize_updates.append(f"{size_name}: {old_n} -> {batch_size}")
    node_inputs = {name for node in model.graph.node for name in node.input}
    node_outputs = {name for node in model.graph.node for name in node.output}
    init_names = {init.name for init in model.graph.initializer}

    kept_outputs = [out for out in model.graph.output if out.name in node_outputs]
    kept_inputs = [
        inp for inp in model.graph.input if inp.name in node_inputs or inp.name in init_names
    ]

    removed_outputs = [out.name for out in model.graph.output if out.name not in node_outputs]
    removed_inputs = [
        inp.name
        for inp in model.graph.input
        if inp.name not in node_inputs and inp.name not in init_names
    ]

    if not removed_outputs and not removed_inputs and not resize_updates and not group_conv_updates:
        return onnx_model_path

    if removed_outputs:
        del model.graph.output[:]
        model.graph.output.extend(kept_outputs)
    if removed_inputs:
        del model.graph.input[:]
        model.graph.input.extend(kept_inputs)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx.save(model, tmp.name)
        sanitized_path = tmp.name

    if group_conv_updates:
        print(f"Sanitized ONNX: expanded group convs: {len(group_conv_updates)}")
    if resize_updates:
        print("Sanitized ONNX: updated Resize sizes:", ", ".join(resize_updates))
    if removed_outputs:
        print(f"Sanitized ONNX: removed unlinked outputs: {removed_outputs}")
    if removed_inputs:
        print(f"Sanitized ONNX: removed unlinked inputs: {removed_inputs}")
    print(f"Sanitized ONNX saved to: {sanitized_path}")
    return sanitized_path


def get_int16_platform(target):
    target = target.lower()
    if target == "esp32s3":
        return TargetPlatform.ESPDL_S3_INT16
    if target == "esp32p4":
        return TargetPlatform.ESPDL_INT16
    if target == "c":
        return TargetPlatform.ESPDL_C_INT16
    return TargetPlatform.ESPDL_INT16


def build_dispatching_override(onnx_model_path, patterns, target, auto_int16_depthwiseconv=False):
    if not patterns and not auto_int16_depthwiseconv:
        return None
    model = onnx.load(onnx_model_path)
    platform = get_int16_platform(target)
    override = {}
    matched = set()
    for node in model.graph.node:
        if not node.name:
            continue
        for pattern in patterns:
            if re.search(pattern, node.name):
                matched.add(node.name)
                break
    for name in matched:
        override[name] = platform
    if auto_int16_depthwiseconv:
        depthwise_prefix = "/depthwiseconv/"
        consumers = {}
        for node in model.graph.node:
            for inp in node.input:
                consumers.setdefault(inp, []).append(node)
        unary_ops = {"Relu", "Clip", "Sigmoid", "HardSigmoid", "LeakyRelu", "PRelu", "Elu"}
        bn_ops = {"BatchNormalization"}
        for node in model.graph.node:
            if not node.name or not node.name.startswith(depthwise_prefix):
                continue
            override[node.name] = platform
            for out_name in node.output:
                for consumer in consumers.get(out_name, []):
                    if consumer.name and consumer.op_type in unary_ops.union(bn_ops):
                        override[consumer.name] = platform
                    if consumer.op_type in bn_ops:
                        for out_bn in consumer.output:
                            for consumer2 in consumers.get(out_bn, []):
                                if consumer2.name and consumer2.op_type in unary_ops:
                                    override[consumer2.name] = platform
    if override:
        print(f"Forcing int16 for {len(override)} ops via pattern match.")
    else:
        print("No ops matched int16 patterns.")
    return override or None


def _resolve_path(entry, image_dir, list_path):
    if os.path.isabs(entry):
        return entry if os.path.exists(entry) else None
    candidates = []
    if list_path:
        candidates.append(os.path.join(os.path.dirname(list_path), entry))
    candidates.append(os.path.join(image_dir, entry))
    candidates.append(os.path.join(image_dir, os.path.basename(entry)))
    candidates.append(entry)
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


class ImageCalibrationDataset(Dataset):
    def __init__(self, image_dir, list_path=None, img_size=(64, 64), resize_mode="opencv_inter_nearest"):
        self.image_dir = image_dir
        self.list_path = list_path
        self.img_h, self.img_w = img_size
        self.resize_mode = resize_mode
        self.items = self._gather_items()
        if not self.items:
            raise ValueError("No images found. Check image_dir/list_path.")

    def _gather_items(self):
        items = []
        if self.list_path:
            with open(self.list_path, "r") as f:
                entries = [ln.strip() for ln in f if ln.strip()]
            for ent in entries:
                img_path = _resolve_path(ent, self.image_dir, self.list_path)
                if img_path:
                    items.append(img_path)
        else:
            patterns = ["*.jpg", "*.jpeg", "*.png"]
            for pat in patterns:
                items.extend(glob.glob(os.path.join(self.image_dir, pat)))
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx]
        im_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        arr = im_rgb.astype(np.float32) / 255.0
        arr_resized = resize_image_numpy(arr, size=(self.img_w, self.img_h), mode=self.resize_mode)
        arr_resized = np.ascontiguousarray(arr_resized)
        img_tensor = torch.from_numpy(arr_resized).permute(2, 0, 1)
        return img_tensor, {}


def parse_class_ids(value):
    return [int(part) for part in value.split(",") if part.strip()]


def collate_fn(batch):
    # Datasets return (image, target); only the image is needed.
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    return images.to(DEVICE)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Quantize an ONNX model for ESP-DL.")
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--dataset-type",
        default="image",
        choices=["yolo", "image"],
        help="Calibration dataset type.",
    )
    parser.add_argument(
        "--list-path",
        default=None,
        help="Optional path to a text file listing images to use.",
    )
    parser.add_argument(
        "--export-anchors-wh-scale-dir",
        default=None,
        help="Directory to save anchors/wh_scale as .npy (optional).",
    )
    parser.add_argument(
        "--expand-group-conv",
        action="store_true",
        help="Expand group conv (groups > 1) into group=1.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=64,
        help="Square input size used for calibration.",
    )
    parser.add_argument(
        "--resize-mode",
        default="opencv_inter_nearest_yuv422",
        help="Resize mode for calibration data.",
    )
    parser.add_argument(
        "--class-ids",
        default="0",
        help="Comma-separated class IDs to keep (for filtering, yolo only).",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "all"],
        help="Dataset split to use for calibration.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Validation split ratio (ignored when split=all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Calibration batch size.",
    )
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=32,
        help="Number of calibration steps.",
    )
    parser.add_argument(
        "--calib-algorithm",
        default="kl",
        help="Calibration algorithm (e.g., kl, minmax, mse, percentile).",
    )
    parser.add_argument(
        "--int16-op-pattern",
        action="append",
        default=[],
        help="Regex pattern to force matched ops to int16 (repeatable).",
    )
    parser.add_argument(
        "--auto-int16-depthwiseconv",
        action="store_true",
        help="Force depthwise conv blocks (/depthwiseconv/, /dw/bn/, /dw/act/) to int16.",
    )
    parser.add_argument(
        "--onnx-model",
        default=DEFAULT_ONNX_MODEL_PATH,
        help="Path to the input ONNX model.",
    )
    parser.add_argument(
        "--espdl-model",
        default=DEFAULT_ESPDL_MODEL_PATH,
        help="Path to the output .espdl file.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=["c", "esp32s3", "esp32p4"],
        help="Quantize target type.",
    )
    parser.add_argument(
        "--num-of-bits",
        type=int,
        default=DEFAULT_NUM_OF_BITS,
        help="Quantization bits.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Device for calibration.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    global DEVICE
    DEVICE = args.device

    onnx_model_path = args.onnx_model
    espdl_model_path = args.espdl_model
    target = args.target
    num_of_bits = args.num_of_bits
    export_dir = args.export_anchors_wh_scale_dir
    if export_dir is None:
        export_dir = os.path.dirname(espdl_model_path) or "."
    onnx_model_path = sanitize_onnx_model(
        onnx_model_path,
        export_dir=export_dir,
        batch_size=args.batch_size,
        expand_group_conv=args.expand_group_conv,
    )
    if args.dataset_type == "yolo":
        class_ids = parse_class_ids(args.class_ids)
        dataset = YoloDataset(
            image_dir=args.image_dir,
            list_path=args.list_path,
            split=args.split,
            val_split=args.val_split,
            img_size=(args.img_size, args.img_size),
            resize_mode=args.resize_mode,
            augment=False,
            class_ids=class_ids,
        )
    else:
        dataset = ImageCalibrationDataset(
            image_dir=args.image_dir,
            list_path=args.list_path,
            img_size=(args.img_size, args.img_size),
            resize_mode=args.resize_mode,
        )
    # The dataloader shuffle setting must be set to False.
    # Because the dataset is traversed multiple times when calculating the quantization error,
    # if shuffle is set to True, an incorrect quantization error will be obtained.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    sample_image, _ = dataset[0]
    input_shape = [1, *sample_image.shape]
    calib_steps = min(args.calib_steps, len(dataloader))

    setting = QuantizationSettingFactory.espdl_setting()
    if args.calib_algorithm:
        setting.quantize_activation_setting.calib_algorithm = args.calib_algorithm
    if args.auto_int16_depthwiseconv:
        depthwise_patterns = [
            r"^/depthwiseconv/",
            r"/dw/bn/",
            r"/dw/act/",
        ]
        for pattern in depthwise_patterns:
            if pattern not in args.int16_op_pattern:
                args.int16_op_pattern.append(pattern)
    # Mixed precision quantization
    # https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_mobilenetv2.html#mixed-precision-quantization
    dispatching_override = build_dispatching_override(
        onnx_model_path,
        args.int16_op_pattern,
        target,
        auto_int16_depthwiseconv=args.auto_int16_depthwiseconv,
    )
    # Layerwise equalization quantization
    # https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_mobilenetv2.html#layerwise-equalization-quantization
    setting.equalization = True
    setting.equalization_setting.iterations = 4
    setting.equalization_setting.value_threshold = .4
    setting.equalization_setting.opt_level = 2
    setting.equalization_setting.interested_layers = None

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_model_path,
        espdl_export_file=espdl_model_path,
        calib_dataloader=dataloader,
        calib_steps=calib_steps,  # Number of calibration steps
        input_shape=input_shape,  # Input shape, batch number 1
        inputs=None,
        target=target,  # Quantify target types
        num_of_bits=num_of_bits,  # Quantization bits
        setting=setting,
        collate_fn=collate_fn,
        dispatching_override=dispatching_override,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # Output detailed log information
    )


if __name__ == "__main__":
    main()
