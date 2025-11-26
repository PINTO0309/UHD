# UHD
Ultra-lightweight human detection.

## Training Examples (full CLI)

CNN example (all parameters explicitly set):

```bash
SIZE=64x64
uv run python train.py \
--arch cnn \
--image-dir data/wholebody34/obj_train_data \
--train-list data/wholebody34/train.txt \
--val-list data/wholebody34/train.txt \
--val-split 0.2 \
--img-size ${SIZE} \
--exp-name cnn_${SIZE} \
--batch-size 64 \
--epochs 50 \
--resume "" \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 2 \
--device cuda \
--seed 42 \
--log-interval 10 \
--eval-interval 1 \
--conf-thresh 0.3 \
--topk 50 \
--save-dir outputs \
--use-amp \
--aug-config uhd/aug.yaml \
--classes 0 \
--cnn-width 32 \
--num-queries 10 \
--d-model 64 \
--heads 4 \
--layers 3 \
--dim-feedforward 128
```

Transformer example (all parameters explicitly set):

```bash
SIZE=64x64
uv run python train.py \
--arch transformer \
--image-dir data/wholebody34/obj_train_data \
--train-list data/wholebody34/train.txt \
--val-list data/wholebody34/train.txt \
--val-split 0.2 \
--img-size ${SIZE} \
--exp-name transformer_${SIZE} \
--batch-size 64 \
--epochs 50 \
--resume "" \
--lr 0.001 \
--weight-decay 0.0001 \
--num-workers 2 \
--device cuda \
--seed 42 \
--log-interval 10 \
--eval-interval 1 \
--conf-thresh 0.3 \
--topk 50 \
--save-dir outputs \
--use-amp \
--aug-config uhd/aug.yaml \
--classes 0 \
--cnn-width 32 \
--num-queries 10 \
--d-model 64 \
--heads 4 \
--layers 3 \
--dim-feedforward 128
```

## Loss terms (CNN / CenterNet)
- `loss`: total loss (`hm + off + wh`)
- `hm`: focal loss on center heatmap
- `off`: L1 loss on center offsets (within-cell quantization correction)
- `wh`: L1 loss on width/height (feature-map scale)

## Augmentation via YAML
- Specify a YAML file with `--aug-config` to run the `data_augment:` entries in the listed order (e.g., `--aug-config uhd/aug.yaml`).
- Supported ops (examples): Mosaic / MixUp / CopyPaste / HorizontalFlip (class_swap_map supported) / VerticalFlip / RandomScale / Translation / RandomCrop / RandomResizedCrop / RandomBrightness / RandomContrast / RandomSaturation / RandomHSV / RandomPhotometricDistort / Blur / MedianBlur / MotionBlur / GaussianBlur / GaussNoise / ImageCompression / ISONoise / RandomRain / RandomFog / RandomSunFlare / CLAHE / ToGray / RemoveOutliers.
- If `prob` is provided, it is used as the apply probability; otherwise defaults are used (most are 0, RandomPhotometricDistort defaults to 0.5). Unknown keys are ignored.
