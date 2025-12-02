# UHD
Ultra-lightweight human detection. The number of parameters does not correlate to inference speed.

|Variant|Size|mAP@0.5|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|S|13.5 MB|||[Download]()|
|C|21.0 MB|||[Download]()|
|M|30.1 MB|||[Download]()|
|L|30.1 MB|||[Download]()|

## Training Examples (full CLI)

UltraTinyOD (anchor-only, stride 8; `--cnn-width` controls stem width):

- If you want to include the residual in the backbone with UltraTinyOD, add `--utod-residual` (insert skip with projection in block3/4).

```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=128
LR=0.0005
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_lr${LR} \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--use-amp \
--classes 0 \
--cnn-width ${CNNWIDTH} \
--auto-anchors \
--num-anchors ${ANCHOR} \
--iou-loss ciou \
--conf-thresh 0.15 \
--utod-residual \
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=160
LR=0.0005
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_lr${LR} \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--use-amp \
--classes 0 \
--cnn-width ${CNNWIDTH} \
--auto-anchors \
--num-anchors ${ANCHOR} \
--iou-loss ciou \
--conf-thresh 0.15 \
--utod-residual \
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=192
LR=0.0004
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_lr${LR} \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--use-amp \
--classes 0 \
--cnn-width ${CNNWIDTH} \
--auto-anchors \
--num-anchors ${ANCHOR} \
--iou-loss ciou \
--conf-thresh 0.15 \
--utod-residual \
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=256
LR=0.0003
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_lr${LR} \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--use-amp \
--classes 0 \
--cnn-width ${CNNWIDTH} \
--auto-anchors \
--num-anchors ${ANCHOR} \
--iou-loss ciou \
--conf-thresh 0.15 \
--utod-residual \
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0
```

## CLI parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--arch` | Model architecture: `cnn`, `transformer`, or anchor-only `ultratinyod`. | `cnn` |
| `--image-dir` | Directory containing images and YOLO txt labels. | `data/wholebody34/obj_train_data` |
| `--train-split` | Fraction of data used for training. | `0.8` |
| `--val-split` | Fraction of data used for validation. | `0.2` |
| `--img-size` | Input size `HxW` (e.g., `64x64`). | `64x64` |
| `--exp-name` | Experiment name; logs saved under `runs/<exp-name>`. | `default` |
| `--batch-size` | Batch size. | `64` |
| `--epochs` | Number of epochs. | `100` |
| `--resume` | Checkpoint to resume training (loads optimizer/scheduler). | `None` |
| `--ckpt` | Initialize weights from checkpoint (no optimizer state). | `None` |
| `--ckpt-non-strict` | Load `--ckpt` with `strict=False` (ignore missing/unexpected keys). | `False` |
| `--teacher-ckpt` | Teacher checkpoint path for distillation. | `None` |
| `--teacher-arch` | Teacher architecture override. | `None` |
| `--teacher-num-queries` | Teacher DETR queries. | `None` |
| `--teacher-d-model` | Teacher model dimension. | `None` |
| `--teacher-heads` | Teacher attention heads. | `None` |
| `--teacher-layers` | Teacher encoder/decoder layers. | `None` |
| `--teacher-dim-feedforward` | Teacher FFN dimension. | `None` |
| `--teacher-use-skip` | Force teacher skip connections on. | `False` |
| `--teacher-activation` | Teacher activation (`relu`/`swish`). | `None` |
| `--teacher-use-fpn` | Force teacher FPN on. | `False` |
| `--teacher-backbone` | Teacher backbone checkpoint for feature distillation. | `None` |
| `--teacher-backbone-arch` | Teacher backbone architecture hint. | `None` |
| `--teacher-backbone-norm` | Teacher backbone input normalization. | `imagenet` |
| `--distill-kl` | KL distillation weight (transformer). | `0.0` |
| `--distill-box-l1` | Box L1 distillation weight (transformer). | `0.0` |
| `--distill-cosine` | Cosine ramp-up of distillation weights. | `False` |
| `--distill-temperature` | Teacher logits temperature. | `1.0` |
| `--distill-feat` | Feature-map distillation weight (CNN only). | `0.0` |
| `--lr` | Learning rate. | `0.001` |
| `--weight-decay` | Weight decay. | `0.0001` |
| `--grad-clip-norm` | Global gradient norm clip; set `0` to disable. | `5.0` |
| `--num-workers` | DataLoader workers. | `8` |
| `--device` | Device: `cuda` or `cpu`. | `cuda` if available |
| `--seed` | Random seed. | `42` |
| `--log-interval` | Steps between logging to progress bar. | `10` |
| `--eval-interval` | Epoch interval for evaluation. | `1` |
| `--conf-thresh` | Confidence threshold for decoding. | `0.3` |
| `--topk` | Top-K for CNN decoding. | `50` |
| `--use-amp` | Enable automatic mixed precision. | `False` |
| `--aug-config` | YAML for augmentations (applied in listed order). | `uhd/aug.yaml` |
| `--use-ema` | Enable EMA of model weights for evaluation/checkpointing. | `False` |
| `--ema-decay` | EMA decay factor (ignored if EMA disabled). | `0.9998` |
| `--coco-eval` | Run COCO-style evaluation. | `False` |
| `--coco-per-class` | Log per-class COCO AP when COCO eval is enabled. | `False` |
| `--classes` | Comma-separated target class IDs. | `0` |
| `--activation` | Activation function (`relu` or `swish`). | `swish` |
| `--cnn-width` | Width multiplier for CNN backbone. | `32` |
| `--backbone` | Optional lightweight CNN backbone (`microcspnet`, `ultratinyresnet`, `enhanced-shufflenet`, or `none`). | `None` |
| `--backbone-channels` | Comma-separated channels for `ultratinyresnet` (e.g., `16,32,48,64`). | `None` |
| `--backbone-blocks` | Comma-separated residual block counts per stage for `ultratinyresnet` (e.g., `1,2,2,1`). | `None` |
| `--backbone-se` | Apply SE/eSE on backbone output (custom backbones only). | `none` |
| `--backbone-skip` | Add long skip fusion across custom backbone stages (ultratinyresnet). | `False` |
| `--backbone-skip-cat` | Use concat+1x1 fusion for long skips (ultratinyresnet); implies `--backbone-skip`. | `False` |
| `--backbone-skip-shuffle-cat` | Use stride+shuffle concat fusion for long skips (ultratinyresnet); implies `--backbone-skip`. | `False` |
| `--backbone-skip-s2d-cat` | Use space-to-depth concat fusion for long skips (ultratinyresnet); implies `--backbone-skip`. | `False` |
| `--backbone-fpn` | Enable a tiny FPN fusion inside custom backbones (ultratinyresnet). | `False` |
| `--backbone-out-stride` | Override custom backbone output stride (e.g., `8` or `16`). | `None` |
| `--use-skip` | Enable skip-style fusion in the CNN head (sums pooled shallow features into the final stage). Stored in checkpoints and restored on resume. | `False` |
| `--use-anchor` | Use anchor-based head for CNN (YOLO-style). | `False` |
| `--output-stride` | Final CNN feature stride (downsample factor). Supported: `4`, `8`, `16`. | `16` |
| `--anchors` | Anchor sizes as normalized `w,h` pairs (space separated). | `""` |
| `--auto-anchors` | Compute anchors from training labels when using anchor head. | `False` |
| `--num-anchors` | Number of anchors to use when auto-computing. | `3` |
| `--iou-loss` | IoU loss type for anchor head (`iou`, `giou`, or `ciou`). | `giou` |
| `--anchor-assigner` | Anchor assigner strategy (`legacy`, `simota`). | `legacy` |
| `--anchor-cls-loss` | Anchor classification loss (`bce`, `vfl`). | `bce` |
| `--simota-topk` | Top-K IoUs for dynamic-k in SimOTA. | `10` |
| `--last-se` | Apply SE/eSE only on the last CNN block. | `none` |
| `--last-width-scale` | Channel scale for last CNN block (e.g., `1.25`). | `1.0` |
| `--num-queries` | Transformer query count. | `10` |
| `--d-model` | Transformer model dimension. | `64` |
| `--heads` | Transformer attention heads. | `4` |
| `--layers` | Transformer encoder/decoder layers. | `3` |
| `--dim-feedforward` | Transformer feedforward dimension. | `128` |
| `--use-fpn` | Enable simple FPN for transformer backbone. | `False` |

Tiny CNN backbones (`--backbone`, optional; default keeps the original built-in CNN):
- `microcspnet`: CSP-tiny style stem (16/32/64/128) compressed to 64ch, stride 8 output.
- `ultratinyresnet`: 16→24→32→48 channel ResNet-like stack with three downsample steps (stride 8). Channel widths and blocks per stage can be overridden via `--backbone-channels` / `--backbone-blocks`; optional long skips across stages via `--backbone-skip`; optional lightweight FPN fusion via `--backbone-fpn`.
- `enhanced-shufflenet`: Enhanced ShuffleNetV2+ inspired (arXiv:2111.00902) with progressive widening and doubled refinements, ending at ~128ch, stride 8.
All custom backbones can optionally apply SE/eSE on the backbone output via `--backbone-se {none,se,ese}`.

## Augmentation via YAML
- Specify a YAML file with `--aug-config` to run the `data_augment:` entries in the listed order (e.g., `--aug-config uhd/aug.yaml`).
- Supported ops (examples): Mosaic / MixUp / CopyPaste / HorizontalFlip (class_swap_map supported) / VerticalFlip / RandomScale / Translation / RandomCrop / RandomResizedCrop / RandomBrightness / RandomContrast / RandomSaturation / RandomHSV / RandomPhotometricDistort / Blur / MedianBlur / MotionBlur / GaussianBlur / GaussNoise / ImageCompression / ISONoise / RandomRain / RandomFog / RandomSunFlare / CLAHE / ToGray / RemoveOutliers.
- If `prob` is provided, it is used as the apply probability; otherwise defaults are used (most are 0, RandomPhotometricDistort defaults to 0.5). Unknown keys are ignored.

## Loss terms (CNN / CenterNet)
- `loss`: total loss (`hm + off + wh`)
- `hm`: focal loss on center heatmap
- `off`: L1 loss on center offsets (within-cell quantization correction)
- `wh`: L1 loss on width/height (feature-map scale)

## Loss terms (CNN / Anchor head, `--use-anchor`)
- `loss`: total anchor loss (`box + obj + cls`)
- `obj`: BCE on objectness for each anchor location (positive vs. background)
- `cls`: BCE on per-class logits for positive anchors (one-hot over target classes)
- `box`: (1 - IoU/GIoU/CIoU) on decoded boxes for positive anchors; IoU flavor set by `--iou-loss`

## Loss terms (Transformer)
- `loss`: total loss (`cls + l1 + iou`)
- `cls`: cross-entropy for class vs. background
- `l1`: L1 loss on box coordinates
- `iou`: 1 - IoU for matched predictions

## ONNX export
- Export a checkpoint to ONNX (auto-detects arch from checkpoint unless overridden):
  ```bash
  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=128
  uv run python export_onnx.py \
  --checkpoint runs/ultratinyod_res_anc8_w128_64x64/best_utod_0001_map_0.00000.pt \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}.onnx \
  --img-size ${SIZE} \
  --opset 17 \
  --merge-postprocess
  ```
  ```bash
  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=160
  uv run python export_onnx.py \
  --checkpoint runs/ultratinyod_res_anc8_w128_64x64/best_utod_0001_map_0.00000.pt \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}.onnx \
  --img-size ${SIZE} \
  --opset 17 \
  --merge-postprocess
  ```
  ```bash
  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=192
  uv run python export_onnx.py \
  --checkpoint runs/ultratinyod_res_anc8_w128_64x64/best_utod_0001_map_0.00000.pt \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}.onnx \
  --img-size ${SIZE} \
  --opset 17 \
  --merge-postprocess
  ```
  ```bash
  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=256
  uv run python export_onnx.py \
  --checkpoint runs/ultratinyod_res_anc8_w128_64x64/best_utod_0001_map_0.00000.pt \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}.onnx \
  --img-size ${SIZE} \
  --opset 17 \
  --merge-postprocess
  ```
- `--arch` can force `cnn`/`transformer`/`ultratinyod`; other model hyperparameters (`cnn-width`, `num-queries`, etc.) are available if needed. Opset defaults to 17.
- `--dynamic` exports with dynamic H/W axes (inputs and CNN outputs). Unknown axes remain fixed.
