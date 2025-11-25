# UHD
Ultra-lightweight human detection.

## Training Examples (full CLI)

CNN example (all parameters explicitly set):

```bash
uv run python train.py \
--arch cnn \
--image-dir data/wholebody34/obj_train_data \
--train-list data/wholebody34/train.txt \
--val-list data/wholebody34/train.txt \
--val-split 0.1 \
--img-size 64x64 \
--exp-name cnn_example \
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
uv run python train.py \
--arch transformer \
--image-dir data/wholebody34/obj_train_data \
--train-list data/wholebody34/train.txt \
--val-list data/wholebody34/train.txt \
--val-split 0.1 \
--img-size 64x64 \
--exp-name transformer_example \
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
--classes 0 \
--cnn-width 32 \
--num-queries 10 \
--d-model 64 \
--heads 4 \
--layers 3 \
--dim-feedforward 128
```
