import argparse
import os
import random
from typing import Any, List, Optional, Sequence, Sized, Tuple, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import esp_ppq.lib as PFL
from esp_ppq.api import get_target_platform
from esp_ppq.api.interface import load_onnx_graph
from esp_ppq.core import QuantizationVisibility, TargetPlatform, convert_any_to_torch_tensor
from esp_ppq.executor import TorchExecutor
from esp_ppq.quantization.optim import (
    ParameterQuantizePass,
    PassiveParameterQuantizePass,
    QuantAlignmentPass,
    QuantizeFusionPass,
    QuantizeSimplifyPass,
    RuntimeCalibrationPass,
)
from esp_ppq.IR import BaseGraph, QuantableOperation, TrainableGraph
from esp_ppq.parser import NativeExporter

try:
    from uhd.data import YoloDataset, detection_collate
    from uhd.losses import anchor_loss
    from uhd.metrics import decode_anchor, evaluate_map
    from uhd.utils import default_device, ensure_dir, move_targets, set_seed
except ImportError:
    from data import YoloDataset, detection_collate # ty: ignore
    from losses import anchor_loss # ty: ignore
    from metrics import decode_anchor, evaluate_map # ty: ignore
    from utils import default_device, ensure_dir, move_targets, set_seed # ty: ignore


def parse_img_size(arg: str) -> Tuple[int, int]:
    arg = str(arg).lower().replace(" ", "")
    if "x" in arg:
        h, w = arg.split("x")
        return int(float(h)), int(float(w))
    v = int(float(arg))
    return v, v


def parse_class_ids(value: str) -> List[int]:
    if value is None or str(value).strip() == "":
        return [0]
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    return out or [0]


class ImageOnlyDataset:
    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]
        return img


class OutputSpec:
    def __init__(
        self,
        raw_idx: Optional[int] = None,
        box_idx: Optional[int] = None,
        obj_idx: Optional[int] = None,
        quality_idx: Optional[int] = None,
        cls_idx: Optional[int] = None,
        anchors_idx: Optional[int] = None,
        wh_scale_idx: Optional[int] = None,
    ) -> None:
        self.raw_idx = raw_idx
        self.box_idx = box_idx
        self.obj_idx = obj_idx
        self.quality_idx = quality_idx
        self.cls_idx = cls_idx
        self.anchors_idx = anchors_idx
        self.wh_scale_idx = wh_scale_idx

    @classmethod
    def from_output_names(cls, output_names: Optional[Sequence[str]]):
        if not output_names:
            return cls()
        name_map = {str(name).lower(): idx for idx, name in enumerate(output_names)}
        return cls(
            raw_idx=name_map.get("txtywh_obj_quality_cls_x8"),
            box_idx=name_map.get("box"),
            obj_idx=name_map.get("obj"),
            quality_idx=name_map.get("quality"),
            cls_idx=name_map.get("cls"),
            anchors_idx=name_map.get("anchors"),
            wh_scale_idx=name_map.get("wh_scale"),
        )

    def has_raw_parts(self) -> bool:
        return self.box_idx is not None and self.obj_idx is not None and self.cls_idx is not None

    def extract_raw(
        self,
        outputs: Sequence[torch.Tensor],
        num_classes: int,
        use_quality: bool,
        num_anchors: int,
    ) -> torch.Tensor:
        if self.raw_idx is not None:
            return outputs[self.raw_idx]
        if not self.has_raw_parts():
            raise ValueError("Unable to locate raw output; export the ONNX with raw outputs.")
        box_idx = cast(int, self.box_idx)
        obj_idx = cast(int, self.obj_idx)
        cls_idx = cast(int, self.cls_idx)
        box = outputs[box_idx]
        obj = outputs[obj_idx]
        cls = outputs[cls_idx]
        parts = [box, obj]
        if use_quality:
            if self.quality_idx is None:
                raise ValueError("use_quality is True but quality output is missing.")
            parts.append(outputs[cast(int, self.quality_idx)])
        parts.append(cls)
        pred = torch.cat(parts, dim=1)
        perm = build_raw_map_perm(num_anchors=num_anchors, num_classes=num_classes, use_quality=use_quality)
        return pred.index_select(1, perm.to(pred.device))

    def extract_anchors(self, outputs: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.anchors_idx is None:
            return None
        return outputs[self.anchors_idx]

    def extract_wh_scale(self, outputs: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.wh_scale_idx is None:
            return None
        return outputs[self.wh_scale_idx]


def build_raw_map_perm(num_anchors: int, num_classes: int, use_quality: bool) -> torch.Tensor:
    perm = []
    a = int(num_anchors)
    nc = int(num_classes)
    if use_quality:
        base_box = 0
        base_obj = a * 4
        base_quality = base_obj + a
        base_cls = base_quality + a
        for anchor in range(a):
            base_box_anchor = base_box + anchor * 4
            perm.extend([base_box_anchor + i for i in range(4)])
            perm.append(base_obj + anchor)
            perm.append(base_quality + anchor)
            base_cls_anchor = base_cls + anchor * nc
            perm.extend([base_cls_anchor + i for i in range(nc)])
    else:
        base_box = 0
        base_obj = a * 4
        base_cls = base_obj + a
        for anchor in range(a):
            base_box_anchor = base_box + anchor * 4
            perm.extend([base_box_anchor + i for i in range(4)])
            perm.append(base_obj + anchor)
            base_cls_anchor = base_cls + anchor * nc
            perm.extend([base_cls_anchor + i for i in range(nc)])
    return torch.tensor(perm, dtype=torch.long)


def load_output_names(onnx_path: str) -> Optional[List[str]]:
    try:
        import onnx
    except Exception:
        return None
    model = onnx.load(onnx_path, load_external_data=False)
    return [out.name for out in model.graph.output]


def prepare_onnx_for_ppq(onnx_path: str) -> str:
    try:
        import onnx
        from onnx import numpy_helper
    except Exception:
        return onnx_path
    model = onnx.load(onnx_path, load_external_data=False)
    changed = False

    input_names = {inp.name for inp in model.graph.input}
    node_outputs = {out for node in model.graph.node for out in node.output}
    keep_outputs = []
    removed_outputs = []
    for out in model.graph.output:
        if out.name in input_names or out.name in node_outputs:
            keep_outputs.append(out)
        else:
            removed_outputs.append(out.name)
    if removed_outputs:
        del model.graph.output[:]
        model.graph.output.extend(keep_outputs)
        changed = True

    shape_by_name: dict[str, List[int]] = {}
    tensors = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    for tensor in tensors:
        if not tensor.type.HasField("tensor_type"):
            continue
        dims = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(int(dim.dim_value))
            else:
                dims.append(-1)
        shape_by_name[tensor.name] = dims

    init_by_name = {init.name: init for init in model.graph.initializer}
    resized = []
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        if len(node.input) < 4:
            continue
        sizes_name = node.input[3]
        if not sizes_name or sizes_name not in init_by_name:
            continue
        sizes_arr = numpy_helper.to_array(init_by_name[sizes_name]).astype(np.float32)
        if sizes_arr.size == 0:
            continue
        input_name = node.input[0]
        input_shape = shape_by_name.get(input_name)
        if not input_shape or len(input_shape) != int(sizes_arr.size):
            continue
        if any(dim <= 0 for dim in input_shape):
            continue
        scales_arr = sizes_arr / np.array(input_shape, dtype=np.float32)
        if not np.all(np.isfinite(scales_arr)):
            continue
        scales_name = node.input[2] if node.input[2] else f"{node.name or 'Resize'}_scales"
        if scales_name in init_by_name:
            existing = numpy_helper.to_array(init_by_name[scales_name]).astype(np.float32)
            if existing.shape != scales_arr.shape or not np.allclose(existing, scales_arr):
                base = scales_name
                suffix = 0
                while f"{base}_{suffix}" in init_by_name:
                    suffix += 1
                scales_name = f"{base}_{suffix}"
        scales_tensor = numpy_helper.from_array(scales_arr.astype(np.float32), name=scales_name)
        model.graph.initializer.append(scales_tensor)
        init_by_name[scales_name] = scales_tensor
        roi_name = node.input[1] if len(node.input) > 1 else ""
        if not roi_name or roi_name not in init_by_name:
            roi_base = f"{node.name or 'Resize'}_roi"
            roi_name = roi_base
            suffix = 0
            while roi_name in init_by_name:
                suffix += 1
                roi_name = f"{roi_base}_{suffix}"
            roi_tensor = numpy_helper.from_array(
                np.zeros((int(sizes_arr.size) * 2,), dtype=np.float32), name=roi_name
            )
            model.graph.initializer.append(roi_tensor)
            init_by_name[roi_name] = roi_tensor
        node.input[:] = [input_name, roi_name, scales_name]
        resized.append(node.name or sizes_name)
        changed = True

    if not changed:
        return onnx_path

    import tempfile

    base = os.path.splitext(os.path.basename(onnx_path))[0]
    out_dir = os.path.dirname(onnx_path) or "."
    fd, temp_path = tempfile.mkstemp(prefix=f"{base}_ppq_", suffix=".onnx", dir=out_dir)
    os.close(fd)
    onnx.save(model, temp_path)
    if removed_outputs:
        print(f"Stripped disconnected ONNX outputs for PPQ: {', '.join(removed_outputs)}")
    if resized:
        print(f"Converted Resize sizes to scales for PPQ: {', '.join(resized)}")
    return temp_path


def load_anchors_wh_scale_from_onnx(onnx_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        import onnx
        from onnx import numpy_helper
    except Exception:
        return None, None
    model = onnx.load(onnx_path, load_external_data=False)
    init_by_name = {init.name: init for init in model.graph.initializer}
    anchors = None
    wh_scale = None
    if "anchors" in init_by_name:
        anchors = numpy_helper.to_array(init_by_name["anchors"]).astype(np.float32)
    if "wh_scale" in init_by_name:
        wh_scale = numpy_helper.to_array(init_by_name["wh_scale"]).astype(np.float32)
    return anchors, wh_scale


def load_anchors_wh_scale_from_npy(onnx_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    base = os.path.splitext(os.path.basename(onnx_path))[0]
    anchors = None
    wh_scale = None
    anchors_path_found = None
    wh_scale_path_found = None
    candidates = [os.path.dirname(onnx_path) or ".", "."]
    seen = set()
    for cand_dir in candidates:
        if cand_dir in seen:
            continue
        seen.add(cand_dir)
        anchors_path = os.path.join(cand_dir, f"{base}_anchors.npy")
        wh_scale_path = os.path.join(cand_dir, f"{base}_wh_scale.npy")
        if anchors is None and os.path.exists(anchors_path):
            anchors = np.load(anchors_path).astype(np.float32)
            anchors_path_found = anchors_path
        if wh_scale is None and os.path.exists(wh_scale_path):
            wh_scale = np.load(wh_scale_path).astype(np.float32)
            wh_scale_path_found = wh_scale_path
        if anchors is not None and wh_scale is not None:
            break
    if anchors_path_found or wh_scale_path_found:
        found = [p for p in (anchors_path_found, wh_scale_path_found) if p]
        print(f"Loaded anchors/wh_scale from npy: {', '.join(found)}")
    return anchors, wh_scale


def normalize_anchor_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 1:
        tensor = tensor.view(-1, 2)
    return tensor


def normalize_anchor_array(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr


def snapshot_trainable_params(graph: BaseGraph) -> dict[str, torch.Tensor]:
    state = TrainableGraph(graph).state_dict()
    snapshot: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if tensor is None:
            continue
        snapshot[name] = tensor.detach().cpu().clone()
    return snapshot


def snapshot_quant_params(
    graph: BaseGraph,
) -> dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    snapshot: dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
    for op in graph.operations.values():
        if not isinstance(op, QuantableOperation):
            continue
        input_count = len(op.inputs)
        for idx, (cfg, var) in enumerate(op.config_with_variable):
            if cfg is None or var is None:
                continue
            scale = cfg.scale
            offset = cfg.offset
            if scale is None and offset is None:
                continue
            io = "in" if idx < input_count else "out"
            io_idx = idx if idx < input_count else idx - input_count
            key = f"{op.name}:{io}{io_idx}:{var.name}"
            scale_tensor = None
            offset_tensor = None
            if scale is not None:
                scale_tensor = convert_any_to_torch_tensor(scale, device="cpu").clone()
            if offset is not None:
                offset_tensor = convert_any_to_torch_tensor(offset, device="cpu").clone()
            snapshot[key] = (scale_tensor, offset_tensor)
    return snapshot


def report_param_deltas(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    top_k: int = 5,
) -> None:
    if not before:
        print("QAT weight change summary: no trainable parameters found.")
        return
    total_sq = 0.0
    base_sq = 0.0
    max_abs = 0.0
    per_param = []
    for name, before_tensor in before.items():
        after_tensor = after.get(name)
        if after_tensor is None:
            continue
        if before_tensor.shape != after_tensor.shape:
            per_param.append((name, float("inf"), float("inf"), float("inf")))
            continue
        diff = (after_tensor - before_tensor).float()
        diff_sq = float(diff.pow(2).sum().item())
        base_sq_i = float(before_tensor.float().pow(2).sum().item())
        total_sq += diff_sq
        base_sq += base_sq_i
        diff_max = float(diff.abs().max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, diff_max)
        if base_sq_i > 0:
            rel = (diff_sq ** 0.5) / (base_sq_i ** 0.5)
        else:
            rel = float("inf") if diff_sq > 0 else 0.0
        per_param.append((name, rel, diff_sq ** 0.5, diff_max))
    global_l2 = total_sq ** 0.5
    global_rel = global_l2 / ((base_sq ** 0.5) + 1e-12)
    print(
        "QAT weight change summary: "
        f"params={len(per_param)}, l2={global_l2:.6g}, rel={global_rel:.6g}, max_abs={max_abs:.6g}"
    )
    per_param = [item for item in per_param if item[1] != float("inf")]
    per_param.sort(key=lambda x: x[1], reverse=True)
    for name, rel, l2, diff_max in per_param[:top_k]:
        print(f"  {name}: rel={rel:.6g}, l2={l2:.6g}, max_abs={diff_max:.6g}")


def report_quant_param_deltas(
    before: dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    after: dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    top_k: int = 5,
) -> None:
    if not before:
        print("Quantization param change summary: no quantized configs found.")
        return
    entries = []
    changed = 0
    max_abs = 0.0
    for key, (scale_before, offset_before) in before.items():
        scale_after, offset_after = after.get(key, (None, None))
        entry_max = 0.0
        if scale_before is not None and scale_after is not None:
            if scale_before.shape == scale_after.shape:
                entry_max = max(entry_max, float((scale_after - scale_before).abs().max().item()))
            else:
                entry_max = float("inf")
        if offset_before is not None and offset_after is not None:
            if offset_before.shape == offset_after.shape:
                entry_max = max(entry_max, float((offset_after - offset_before).abs().max().item()))
            else:
                entry_max = float("inf")
        if entry_max > 0:
            changed += 1
        max_abs = max(max_abs, entry_max)
        entries.append((key, entry_max))
    print(
        "Quantization param change summary: "
        f"configs={len(entries)}, changed={changed}, max_abs={max_abs:.6g}"
    )
    entries = [item for item in entries if item[1] != float("inf")]
    entries.sort(key=lambda x: x[1], reverse=True)
    for name, diff_max in entries[:top_k]:
        print(f"  {name}: max_abs={diff_max:.6g}")


def infer_use_quality(
    raw_pred: torch.Tensor,
    num_classes: int,
    num_anchors: int,
    prefer_quality: Optional[bool] = None,
) -> bool:
    if prefer_quality is not None:
        return bool(prefer_quality)
    if raw_pred.ndim < 2:
        raise ValueError("Raw prediction tensor has unexpected rank.")
    channels = int(raw_pred.shape[1])
    if num_anchors <= 0 or channels % num_anchors != 0:
        raise ValueError("Unable to infer quality head from output shape.")
    per_anchor = channels // num_anchors
    if per_anchor == 5 + num_classes:
        return False
    if per_anchor == 6 + num_classes:
        return True
    raise ValueError(
        f"Unexpected channels per anchor: {per_anchor}, num_classes={num_classes}. "
        "Specify --use-quality or --no-quality."
    )


class QATTrainer:
    def __init__(
        self,
        graph: BaseGraph,
        output_spec: OutputSpec,
        anchors: torch.Tensor,
        wh_scale: Optional[torch.Tensor],
        num_classes: int,
        use_quality: bool,
        device: torch.device,
        iou_loss: str = "giou",
        assigner: str = "legacy",
        cls_loss_type: str = "bce",
        simota_topk: int = 10,
        lr: float = 1e-6,
        momentum: float = 0.937,
        weight_decay: float = 5e-4,
    ) -> None:
        self._epoch = 0
        self._step = 0
        self._device = device
        self.graph = graph
        self.output_spec = output_spec
        self.num_classes = int(num_classes)
        self.use_quality = bool(use_quality)
        self.anchors = anchors.to(self._device)
        self.wh_scale = wh_scale.to(self._device) if wh_scale is not None else None
        self.iou_loss = iou_loss
        self.assigner = assigner
        self.cls_loss_type = cls_loss_type
        self.simota_topk = int(simota_topk)

        self._executor = TorchExecutor(graph, device=str(self._device))
        self._training_graph = TrainableGraph(graph)
        for tensor in self._training_graph.parameters():
            tensor.requires_grad = True

        self._optimizer = torch.optim.SGD(
            params=[{"params": self._training_graph.parameters()}],
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        self._lr_scheduler = None

    def _extract_raw(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.output_spec.extract_raw(
            outputs,
            num_classes=self.num_classes,
            use_quality=self.use_quality,
            num_anchors=int(self.anchors.shape[0]),
        )

    def epoch(self, dataloader: DataLoader) -> float:
        epoch_loss = 0.0
        for batch in dataloader:
            _, loss = self.step(batch, training=True)
            epoch_loss += float(loss)
        self._epoch += 1
        avg = epoch_loss / max(len(dataloader), 1)
        print(f"Epoch {self._epoch}: loss={avg:.6f}")
        return avg

    def step(self, batch, training: bool = True):
        if not training:
            raise ValueError("Use eval() for evaluation.")
        imgs, targets = batch
        imgs = imgs.to(self._device, non_blocking=True)
        targets_dev = move_targets(targets, self._device)
        self._optimizer.zero_grad()
        outputs = self._executor.forward_with_gradient(imgs)
        raw = self._extract_raw(outputs)
        loss_dict = anchor_loss(
            raw,
            targets_dev,
            anchors=self.anchors,
            num_classes=self.num_classes,
            iou_loss=self.iou_loss,
            assigner=self.assigner,
            cls_loss_type=self.cls_loss_type,
            simota_topk=self.simota_topk,
            use_quality=self.use_quality,
            wh_scale=self.wh_scale,
        )
        loss = loss_dict["loss"]
        loss.backward()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step(epoch=self._epoch)
        self._optimizer.step()
        self._training_graph.zero_grad()
        self._step += 1
        return raw, loss.item()

    def eval(
        self,
        dataloader: DataLoader,
        conf_thresh: float = 0.3,
        nms_thresh: float = 0.5,
        score_mode: str = "obj_quality_cls",
        quality_power: float = 1.0,
        iou_thresh: float = 0.5,
    ) -> float:
        preds = []
        targets = []
        for imgs, tgt in dataloader:
            imgs = imgs.to(self._device, non_blocking=True)
            with torch.no_grad():
                if hasattr(self._executor, "forward"):
                    outputs = self._executor.forward(imgs)
                else:
                    outputs = self._executor.forward_with_gradient(imgs)
            raw = self._extract_raw(outputs)
            decoded = decode_anchor(
                raw,
                anchors=self.anchors,
                num_classes=self.num_classes,
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh,
                has_quality=self.use_quality,
                wh_scale=self.wh_scale,
                score_mode=score_mode,
                quality_power=quality_power,
            )
            preds.extend(decoded)
            targets.extend(tgt)
        metrics = evaluate_map(preds, targets, num_classes=self.num_classes, iou_thresh=iou_thresh)
        map50 = float(metrics.get("mAP@0.5", 0.0))
        print(f"Eval mAP@0.5: {map50:.4f}")
        return map50

    def save(self, espdl_path: str, native_path: str, platform: TargetPlatform) -> None:
        PFL.Exporter(platform=platform).export(file_path=espdl_path, graph=self.graph)
        exporter = NativeExporter()
        exporter.export(file_path=native_path, graph=self.graph)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QAT for UHD (UltraTinyOD) using ESP-PPQ.")
    parser.add_argument("--onnx-model", required=True, help="Path to raw ONNX model (no postprocess).")
    parser.add_argument("--image-dir", required=True, help="Directory with images and YOLO txt labels.")
    parser.add_argument("--list-path", default=None, help="Optional list file for dataset.")
    parser.add_argument(
        "--train-list",
        default=None,
        help="Optional train list file (one image path per line). Requires --val-list and ignores --val-split.",
    )
    parser.add_argument(
        "--val-list",
        default=None,
        help="Optional validation list file (one image path per line). Requires --train-list and ignores --val-split.",
    )
    parser.add_argument("--img-size", default="64x64", help="Input size as HxW (e.g., 64x64).")
    parser.add_argument(
        "--resize-mode",
        default="opencv_inter_nearest",
        choices=[
            "torch_bilinear",
            "torch_nearest",
            "opencv_inter_linear",
            "opencv_inter_nearest",
            "opencv_inter_nearest_y_bin",
            "opencv_inter_nearest_y",
            "opencv_inter_nearest_y_tri",
            "opencv_inter_nearest_yuv422",
        ],
        help="Resize mode used during training/calibration.",
    )
    parser.add_argument("--class-ids", default="0", help="Comma-separated class ids to keep (e.g., 0,1).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to auto.")
    parser.add_argument("--target", default="esp32p4", choices=["c", "esp32s3", "esp32p4"])
    parser.add_argument("--num-of-bits", type=int, default=8, help="Quantization bits.")
    parser.add_argument("--calib-steps", type=int, default=32)
    parser.add_argument("--use-quality", action="store_true", help="Force quality head on.")
    parser.add_argument("--no-quality", action="store_true", help="Force quality head off.")
    parser.add_argument("--iou-loss", choices=["iou", "giou", "ciou"], default="giou")
    parser.add_argument("--anchor-assigner", choices=["legacy", "simota"], default="legacy")
    parser.add_argument("--anchor-cls-loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--simota-topk", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--no-equalization",
        dest="equalization",
        action="store_false",
        help="Disable layerwise equalization (enabled by default).",
    )
    parser.set_defaults(equalization=True)
    parser.add_argument("--equalization-iterations", type=int, default=4)
    parser.add_argument("--equalization-value-threshold", type=float, default=0.4)
    parser.add_argument("--equalization-opt-level", type=int, default=2)
    parser.add_argument("--save-dir", default="runs/uhd_qat", help="Directory to save QAT outputs.")
    parser.add_argument("--output-prefix", default=None, help="Prefix for saved model files.")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation.")
    parser.add_argument("--conf-thresh", type=float, default=0.15)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--score-mode", default="obj_quality_cls")
    parser.add_argument("--quality-power", type=float, default=1.0)
    return parser


def _build_dispatching_table(graph: BaseGraph, quantizer: Any):
    dispatching_table = PFL.Dispatcher(graph=graph, method="conservative").dispatch(
        quantizer.quant_operation_types
    )
    for opname, platform in dispatching_table.items():
        if platform == TargetPlatform.UNSPECIFIED:
            dispatching_table[opname] = TargetPlatform(quantizer.target_platform)
    return dispatching_table


def _build_equalization_pass(args):
    try:
        from esp_ppq.quantization.optim import LayerwiseEqualizationPass
    except Exception as exc:
        raise ImportError("LayerwiseEqualizationPass is unavailable in this esp_ppq build.") from exc
    return LayerwiseEqualizationPass(
        iterations=int(args.equalization_iterations),
        value_threshold=float(args.equalization_value_threshold),
        optimize_level=int(args.equalization_opt_level),
        interested_layers=[],
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(int(args.seed))
    device = default_device(args.device)

    class_ids = parse_class_ids(args.class_ids)
    img_size = parse_img_size(args.img_size)

    if args.train_list or args.val_list:
        if not (args.train_list and args.val_list):
            raise ValueError("--train-list and --val-list must be provided together.")
        train_dataset = YoloDataset(
            image_dir=args.image_dir,
            list_path=args.train_list,
            split="all",
            val_split=0.0,
            img_size=img_size,
            resize_mode=args.resize_mode,
            augment=False,
            class_ids=class_ids,
        )
        val_dataset = YoloDataset(
            image_dir=args.image_dir,
            list_path=args.val_list,
            split="all",
            val_split=0.0,
            img_size=img_size,
            resize_mode=args.resize_mode,
            augment=False,
            class_ids=class_ids,
        )
    else:
        base_dataset = YoloDataset(
            image_dir=args.image_dir,
            list_path=args.list_path,
            split="all",
            val_split=0.0,
            img_size=img_size,
            resize_mode=args.resize_mode,
            augment=False,
            class_ids=class_ids,
        )
        items = list(base_dataset.items)
        rng = random.Random(int(args.seed))
        rng.shuffle(items)
        split_idx = int(len(items) * (1.0 - float(args.val_split))) if args.val_split else len(items)
        train_items = items[:split_idx]
        val_items = items[split_idx:] if args.val_split and args.val_split > 0 else []
        if args.val_split and args.val_split > 0 and not val_items:
            raise ValueError("Validation split produced no samples; adjust --val-split.")

        train_dataset = YoloDataset(
            image_dir=args.image_dir,
            list_path=args.list_path,
            split="all",
            val_split=0.0,
            img_size=img_size,
            resize_mode=args.resize_mode,
            augment=False,
            class_ids=class_ids,
            items=train_items,
        )
        val_dataset = None
        if val_items:
            val_dataset = YoloDataset(
                image_dir=args.image_dir,
                list_path=args.list_path,
                split="all",
                val_split=0.0,
                img_size=img_size,
                resize_mode=args.resize_mode,
                augment=False,
                class_ids=class_ids,
                items=val_items,
            )

    g = torch.Generator()
    g.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=min(int(args.num_workers), os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate,
        generator=g,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=min(int(args.num_workers), os.cpu_count() or 1),
            pin_memory=(device.type == "cuda"),
            collate_fn=detection_collate,
        )

    cali_dataset = ImageOnlyDataset(train_dataset)
    cali_loader = DataLoader(
        cast(Dataset[torch.Tensor], cali_dataset),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=min(int(args.num_workers), os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )

    sample_img, _ = train_dataset[0]
    input_shape = [1, *sample_img.shape]

    ppq_onnx_path = prepare_onnx_for_ppq(args.onnx_model)
    graph = load_onnx_graph(onnx_import_file=ppq_onnx_path)
    cfg_platform = get_target_platform(str(args.target), int(args.num_of_bits))
    quantizer = PFL.Quantizer(platform=cfg_platform, graph=graph)
    dispatching_table = _build_dispatching_table(graph, quantizer)
    for op in graph.operations.values():
        quantizer.quantize_operation(op_name=op.name, platform=dispatching_table[op.name])

    executor = TorchExecutor(graph=graph, device=str(device))
    executor.tracing_operation_meta(inputs=torch.zeros(input_shape, device=device))

    passes = []
    if args.equalization:
        passes.append(_build_equalization_pass(args))
    passes.extend(
        [
            QuantizeSimplifyPass(),
            QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
            ParameterQuantizePass(),
            RuntimeCalibrationPass(method="kl"),
            PassiveParameterQuantizePass(
                clip_visiblity=QuantizationVisibility.EXPORT_WHEN_ACTIVE
            ),
            QuantAlignmentPass(elementwise_alignment="Align to Output"),
        ]
    )
    pipeline = PFL.Pipeline(passes)

    calib_steps = min(int(args.calib_steps), len(cali_loader))
    pipeline.optimize(
        calib_steps=calib_steps,
        collate_fn=(lambda x: x.type(torch.float).to(device)),
        graph=graph,
        dataloader=cali_loader,
        executor=executor,
    )
    print(f"Calibrate images: {len(cast(Sized, cali_loader.dataset))}, steps: {calib_steps}")

    output_names = load_output_names(ppq_onnx_path)
    output_spec = OutputSpec.from_output_names(output_names)

    anchors_np, wh_scale_np = load_anchors_wh_scale_from_npy(args.onnx_model)
    if anchors_np is None or wh_scale_np is None:
        anchors_onnx, wh_scale_onnx = load_anchors_wh_scale_from_onnx(args.onnx_model)
        if anchors_np is None:
            anchors_np = anchors_onnx
        if wh_scale_np is None:
            wh_scale_np = wh_scale_onnx
    anchors_np = normalize_anchor_array(anchors_np)
    wh_scale_np = normalize_anchor_array(wh_scale_np)
    anchors = torch.from_numpy(anchors_np).float() if anchors_np is not None else None
    wh_scale = torch.from_numpy(wh_scale_np).float() if wh_scale_np is not None else None

    if anchors is None or wh_scale is None:
        with torch.no_grad():
            if hasattr(executor, "forward"):
                outputs = executor.forward(torch.zeros(input_shape, device=device))
            else:
                outputs = executor.forward_with_gradient(torch.zeros(input_shape, device=device))
        if anchors is None:
            anchors_out = output_spec.extract_anchors(outputs)
            if anchors_out is not None:
                anchors = normalize_anchor_tensor(anchors_out.detach().cpu())
        if wh_scale is None:
            wh_scale_out = output_spec.extract_wh_scale(outputs)
            if wh_scale_out is not None:
                wh_scale = normalize_anchor_tensor(wh_scale_out.detach().cpu())

    if anchors is None:
        raise ValueError("Anchors not found in ONNX outputs or initializers.")

    num_classes = len(class_ids)
    prefer_quality = None
    if args.use_quality and args.no_quality:
        raise ValueError("Specify only one of --use-quality or --no-quality.")
    if args.use_quality:
        prefer_quality = True
    if args.no_quality:
        prefer_quality = False

    with torch.no_grad():
        if hasattr(executor, "forward"):
            raw_outputs = executor.forward(torch.zeros(input_shape, device=device))
        else:
            raw_outputs = executor.forward_with_gradient(torch.zeros(input_shape, device=device))
    raw_pred = output_spec.extract_raw(
        raw_outputs,
        num_classes=num_classes,
        use_quality=prefer_quality if prefer_quality is not None else (output_spec.quality_idx is not None),
        num_anchors=int(anchors.shape[0]),
    )
    use_quality = infer_use_quality(
        raw_pred,
        num_classes=num_classes,
        num_anchors=int(anchors.shape[0]),
        prefer_quality=prefer_quality,
    )

    trainer = QATTrainer(
        graph=graph,
        output_spec=output_spec,
        anchors=anchors,
        wh_scale=wh_scale,
        num_classes=num_classes,
        use_quality=use_quality,
        device=device,
        iou_loss=args.iou_loss,
        assigner=args.anchor_assigner,
        cls_loss_type=args.anchor_cls_loss,
        simota_topk=args.simota_topk,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    param_snapshot_before = snapshot_trainable_params(trainer.graph)
    quant_snapshot_before = snapshot_quant_params(trainer.graph)

    base_name = os.path.splitext(os.path.basename(args.onnx_model))[0]
    output_prefix = args.output_prefix or base_name
    ensure_dir(args.save_dir)

    best_metric = -1.0
    for epoch in range(int(args.epochs)):
        trainer.epoch(train_loader)
        current_metric = None
        if val_loader is not None and not args.no_eval:
            current_metric = trainer.eval(
                val_loader,
                conf_thresh=float(args.conf_thresh),
                nms_thresh=float(args.nms_thresh),
                score_mode=str(args.score_mode),
                quality_power=float(args.quality_power),
                iou_thresh=float(args.iou_thresh),
            )

        epoch_dir = os.path.join(args.save_dir, f"epoch_{epoch:03d}")
        ensure_dir(epoch_dir)
        espdl_path = os.path.join(epoch_dir, f"{output_prefix}.espdl")
        native_path = os.path.join(epoch_dir, f"{output_prefix}.native")
        trainer.save(espdl_path, native_path, platform=TargetPlatform(cfg_platform))

        if current_metric is not None and current_metric > best_metric:
            best_metric = current_metric
            best_espdl = os.path.join(args.save_dir, f"best_{output_prefix}.espdl")
            best_native = os.path.join(args.save_dir, f"best_{output_prefix}.native")
            trainer.save(best_espdl, best_native, platform=TargetPlatform(cfg_platform))

    param_snapshot_after = snapshot_trainable_params(trainer.graph)
    quant_snapshot_after = snapshot_quant_params(trainer.graph)
    report_param_deltas(param_snapshot_before, param_snapshot_after)
    report_quant_param_deltas(quant_snapshot_before, quant_snapshot_after)

    try:
        from esp_ppq.quantization.analyse import graphwise_error_analyse
        from esp_ppq.quantization.analyse.layerwise import layerwise_error_analyse
    except Exception as exc:
        print(f"Quantization error analysis skipped: {exc}")
    else:
        analysis_steps = min(8, len(cali_loader))
        if analysis_steps > 0:
            collate_fn = lambda x: x.type(torch.float).to(device)
            graphwise_error_analyse(
                graph=trainer.graph,
                running_device=str(device),
                dataloader=cali_loader,
                collate_fn=collate_fn,
                steps=analysis_steps,
            )
            layerwise_error_analyse(
                graph=trainer.graph,
                running_device=str(device),
                dataloader=cali_loader,
                collate_fn=collate_fn,
                steps=analysis_steps,
            )


if __name__ == "__main__":
    main()
