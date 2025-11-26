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
--layers 3 \
--dim-feedforward 128 \
--use-ema \
--ema-decay 0.9999
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
