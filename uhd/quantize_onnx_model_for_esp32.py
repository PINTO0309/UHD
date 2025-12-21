import argparse
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from esp_ppq.api import espdl_quantize_onnx

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
        default=64,
        help="Calibration batch size.",
    )
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=32,
        help="Number of calibration steps.",
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

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_model_path,
        espdl_export_file=espdl_model_path,
        calib_dataloader=dataloader,
        calib_steps=calib_steps,  # Number of calibration steps
        input_shape=input_shape,  # Input shape, batch number 1
        inputs=None,
        target=target,  # Quantify target types
        num_of_bits=num_of_bits,  # Quantization bits
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # Output detailed log information
    )


if __name__ == "__main__":
    main()
