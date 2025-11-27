# UHD
Ultra-lightweight human detection.

## Training Examples (full CLI)

CNN example (all parameters explicitly set):

```bash
SIZE=64x64
uv run python train.py \
--arch cnn \
--image-dir data/wholebody34/obj_train_data \
--train-split 0.8 \
--val-split 0.2 \
--img-size ${SIZE} \
--exp-name cnn_${SIZE} \
--batch-size 64 \
--epochs 500 \
--resume "" \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--seed 42 \
--log-interval 10 \
--eval-interval 1 \
--conf-thresh 0.3 \
--activation swish \
--topk 50 \
--use-amp \
--aug-config uhd/aug.yaml \
--classes 0 \
--cnn-width 32 \
--use-skip \
--use-ema
```

Transformer example (all parameters explicitly set):

```bash
SIZE=64x64
LAYERS=3
uv run python train.py \
--arch transformer \
--image-dir data/wholebody34/obj_train_data \
--train-split 0.8 \
--val-split 0.2 \
--img-size ${SIZE} \
--exp-name transformer_${SIZE} \
--batch-size 64 \
--epochs 500 \
--resume "" \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--seed 42 \
--log-interval 10 \
--eval-interval 1 \
--conf-thresh 0.3 \
--topk 50 \
--use-amp \
--aug-config uhd/aug.yaml \
--classes 0 \
--num-queries 10 \
--d-model 64 \
--heads 4 \
--layers ${LAYERS} \
--use-fpn \
--dim-feedforward 128 \
--use-ema \
--ema-decay 0.9999
```
```bash
SIZE=64x64
LAYERS=4
uv run python train.py \
--arch transformer \
--image-dir data/wholebody34/obj_train_data \
--train-split 0.8 \
--val-split 0.2 \
--img-size ${SIZE} \
--exp-name transformer_${SIZE} \
--batch-size 64 \
--epochs 500 \
--resume "" \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--seed 42 \
--log-interval 10 \
--eval-interval 1 \
--conf-thresh 0.3 \
--topk 50 \
--use-amp \
--aug-config uhd/aug.yaml \
--classes 0 \
--num-queries 10 \
--d-model 64 \
--heads 4 \
--layers ${LAYERS} \
--use-fpn \
--dim-feedforward 128 \
--use-ema \
--ema-decay 0.9999
```

CNN anchor head + G/CIoU

```bash
SIZE=64x64
uv run python train.py \
--arch cnn \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name cnn_anchor_${SIZE} \
--batch-size 64 \
--epochs 300 \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--use-amp \
--classes 0 \
--cnn-width 64 \
--use-anchor \
--auto-anchors \
--num-anchors 5 \
--iou-loss ciou \
--last-se se \
--last-width-scale 1.25 \
--use-skip \
--output-stride 16 \
--use-ema \
--ema-decay 0.9999
```

CNN + DINOv3 backbone feature distillation (teacher used only during training; ONNX export stays student-only):

```bash
SIZE=64x64
uv run python train.py \
--arch cnn \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name cnn_anchor_dino_${SIZE} \
--batch-size 64 \
--epochs 300 \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--use-amp \
--classes 0 \
--cnn-width 64 \
--use-anchor \
--auto-anchors \
--num-anchors 5 \
--iou-loss ciou \
--last-se se \
--last-width-scale 1.25 \
--use-skip \
--output-stride 8 \
--use-ema \
--ema-decay 0.9999 \
--teacher-backbone ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
--teacher-backbone-arch dinov3_vits16 \
--distill-feat 1.0
```

CNN + DINOv3 backbone feature distillation (stride 16):

```bash
SIZE=64x64
uv run python train.py \
--arch cnn \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name cnn_anchor_dino_s16_${SIZE} \
--batch-size 64 \
--epochs 300 \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--use-amp \
--classes 0 \
--cnn-width 64 \
--use-anchor \
--auto-anchors \
--num-anchors 5 \
--iou-loss ciou \
--last-se se \
--last-width-scale 1.25 \
--use-skip \
--output-stride 16 \
--use-ema \
--ema-decay 0.9999 \
--teacher-backbone ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
--teacher-backbone-arch dinov3_vits16 \
--distill-feat 1.0
```

CNN + DINOv3-B/16 backbone feature distillation:

```bash
SIZE=64x64
uv run python train.py \
--arch cnn \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name cnn_anchor_dino_b16_${SIZE} \
--batch-size 64 \
--epochs 300 \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--use-amp \
--classes 0 \
--cnn-width 64 \
--use-anchor \
--auto-anchors \
--num-anchors 5 \
--iou-loss ciou \
--last-se se \
--last-width-scale 1.25 \
--use-skip \
--output-stride 8 \
--use-ema \
--ema-decay 0.9999 \
--teacher-backbone ckpts/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
--teacher-backbone-arch dinov3_vitb16 \
--distill-feat 1.0
```

CNN + ViT-Tiny distillation backbone (vitt_distill.pt):

```bash
SIZE=64x64
uv run python train.py \
--arch cnn \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name cnn_anchor_vitt_${SIZE} \
--batch-size 64 \
--epochs 300 \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--use-amp \
--classes 0 \
--cnn-width 64 \
--use-anchor \
--auto-anchors \
--num-anchors 5 \
--iou-loss ciou \
--last-se se \
--last-width-scale 1.25 \
--use-skip \
--output-stride 8 \
--use-ema \
--ema-decay 0.9999 \
--teacher-backbone ckpts/vitt_distill.pt \
--teacher-backbone-arch vit_tiny \
--distill-feat 1.0
```

Options:
- `--last-se {none,se,ese}`: apply SE/eSE only on the last CNN block.
- `--last-width-scale 1.25`: scale only the last block channels (e.g., 1.25 = +25%).


Distillation example (transformer student 64x64 distilled from a higher-res transformer teacher):

```bash
SIZE=64x64
uv run python train.py \
--arch transformer \
--image-dir data/wholebody34/obj_train_data \
--train-split 0.8 \
--val-split 0.2 \
--img-size ${SIZE} \
--exp-name  transformer_distill_${SIZE} \
--batch-size 64 \
--epochs 200 \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--seed 42 \
--log-interval 10 \
--eval-interval 1 \
--conf-thresh 0.3 \
--topk 50 \
--use-amp \
--aug-config uhd/aug.yaml \
--classes 0 \
--num-queries 10 \
--d-model 64 \
--heads 4 \
--layers 4 \
--dim-feedforward 128 \
--use-ema \
--ema-decay 0.9999 \
--teacher-ckpt runs/teacher_640x640/best_tf_0500_map_0.12345.pt \
--teacher-arch transformer \
--teacher-layers 3 \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-temperature 2.0 \
--distill-cosine
```

Transformer example (From layers=3 To layers=4):

```
SIZE=64x64
uv run python train.py \
--arch transformer \
--image-dir data/wholebody34/obj_train_data \
--train-split 0.8 \
--val-split 0.2 \
--img-size ${SIZE} \
--exp-name transformer_${SIZE} \
--batch-size 64 \
--epochs 500 \
--ckpt runs/transformer_64x64/best_tf_0001_map_0.02340.pt \
--ckpt-non-strict \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--seed 42 \
--log-interval 10 \
--eval-interval 1 \
--conf-thresh 0.3 \
--topk 50 \
--use-amp \
--aug-config uhd/aug.yaml \
--classes 0 \
--num-queries 10 \
--d-model 64 \
--heads 4 \
--layers 4 \
--dim-feedforward 128 \
--use-ema \
--ema-decay 0.9999
```

## CLI parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--arch` | Model architecture: `cnn` or `transformer`. | `cnn` |
| `--image-dir` | Directory containing images and YOLO txt labels. | `data/wholebody34/obj_train_data` |
| `--train-split` | Fraction of data used for training. | `0.8` |
| `--val-split` | Fraction of data used for validation. | `0.2` |
| `--img-size` | Input size `HxW` (e.g., `64x64`). | `64x64` |
| `--exp-name` | Experiment name (logs/checkpoints under `runs/<exp-name>`). | `default` |
| `--batch-size` | Batch size. | `64` |
| `--epochs` | Number of epochs. | `50` |
| `--resume` | Path to checkpoint to resume training. | `""` |
| `--lr` | Learning rate. | `0.001` |
| `--weight-decay` | Weight decay. | `0.0001` |
| `--grad-clip-norm` | Global gradient norm clip; set `0` to disable. | `5.0` |
| `--activation` | Activation function (`relu` or `swish`). | `swish` |
| `--num-workers` | DataLoader workers. | `2` |
| `--device` | Device: `cuda` or `cpu`. | `cuda` if available |
| `--seed` | Random seed. | `42` |
| `--log-interval` | Steps between logging to progress bar. | `10` |
| `--eval-interval` | Epoch interval for evaluation. | `1` |
| `--conf-thresh` | Confidence threshold for decoding. | `0.3` |
| `--topk` | Top-K for CNN decoding. | `50` |
| `--use-amp` | Enable automatic mixed precision. | `False` |
| `--use-ema` | Enable EMA for evaluation/checkpointing. | `False` |
| `--ema-decay` | EMA decay (ignored if EMA disabled). | `0.9998` |
| `--aug-config` | YAML for augmentations (applied in listed order). | `uhd/aug.yaml` |
| `--classes` | Comma-separated target class IDs. | `0` |
| `--cnn-width` | Width multiplier for CNN backbone. | `32` |
| `--use-skip` | Enable skip-style fusion in the CNN head (sums pooled shallow features into the final stage). Stored in checkpoints and restored on resume. | `False` |
| `--output-stride` | Final CNN feature stride (downsample factor). Supported: `4`, `8`, `16`. | `16` |
| `--num-queries` | Transformer query count. | `10` |
| `--d-model` | Transformer model dimension. | `64` |
| `--heads` | Transformer attention heads. | `4` |
| `--layers` | Transformer encoder/decoder layers. | `3` |
| `--dim-feedforward` | Transformer feedforward dimension. | `128` |

## Augmentation via YAML
- Specify a YAML file with `--aug-config` to run the `data_augment:` entries in the listed order (e.g., `--aug-config uhd/aug.yaml`).
- Supported ops (examples): Mosaic / MixUp / CopyPaste / HorizontalFlip (class_swap_map supported) / VerticalFlip / RandomScale / Translation / RandomCrop / RandomResizedCrop / RandomBrightness / RandomContrast / RandomSaturation / RandomHSV / RandomPhotometricDistort / Blur / MedianBlur / MotionBlur / GaussianBlur / GaussNoise / ImageCompression / ISONoise / RandomRain / RandomFog / RandomSunFlare / CLAHE / ToGray / RemoveOutliers.
- If `prob` is provided, it is used as the apply probability; otherwise defaults are used (most are 0, RandomPhotometricDistort defaults to 0.5). Unknown keys are ignored.

## Loss terms (CNN / CenterNet)
- `loss`: total loss (`hm + off + wh`)
- `hm`: focal loss on center heatmap
- `off`: L1 loss on center offsets (within-cell quantization correction)
- `wh`: L1 loss on width/height (feature-map scale)

Transformer loss terms
- `loss`: total loss (`cls + l1 + iou`)
- `cls`: cross-entropy for class vs. background
- `l1`: L1 loss on box coordinates
- `iou`: 1 - IoU for matched predictions

## ONNX export
- Export a checkpoint to ONNX (auto-detects arch from checkpoint unless overridden):
  ```bash
  SIZE=64x64
  uv run python export_onnx.py \
  --checkpoint runs/default/best_cnn_0001_map_0.12345.pt \
  --output model_cnn_${SIZE}.onnx \
  --img-size ${SIZE} \
  --opset 17 \
  --merge-postprocess

  SIZE=64x64
  uv run python export_onnx.py \
  --checkpoint runs/transformer_64x64/best_tf_0002_map_0.01072.pt \
  --output model_tf_${SIZE}.onnx \
  --img-size ${SIZE} \
  --opset 17 \
  --merge-postprocess
  ```
- `--arch` can force `cnn`/`transformer`; other model hyperparameters (`cnn-width`, `num-queries`, etc.) are available if needed. Opset defaults to 17.
- `--dynamic` exports with dynamic H/W axes (inputs and CNN outputs). Unknown axes remain fixed.
