# UHD
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17790207.svg)](https://doi.org/10.5281/zenodo.17790207) ![GitHub License](https://img.shields.io/github/license/pinto0309/uhd) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/uhd)

Ultra-lightweight human detection. The number of parameters does not correlate to inference speed. For limited use cases, an input image resolution of 64x64 is sufficient. High-level object detection architectures such as YOLO are overkill.

This model is an experimental implementation and is not suitable for real-time inference using a USB camera, etc.

https://github.com/user-attachments/assets/6115de34-ec8a-4649-9e1a-7da46e6f370d

- Variant-S / `w ESE + IoU-aware + ReLU`

|Input<br>64x64|Output|Input<br>64x64|Output|
|:-:|:-:|:-:|:-:|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/59b941a4-86eb-41b9-9c0a-c367e6718b4c" />|<img width="640" height="427" alt="00_dist_000000075375" src="https://github.com/user-attachments/assets/5cb2e332-86eb-4933-bf80-3f500173306f" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/21791bff-2cf4-4a63-87b5-f6da52bb4964" />|<img width="480" height="360" alt="01_000000073639" src="https://github.com/user-attachments/assets/ed0f9b67-7a05-4e31-8f43-8e82a468ad38" />|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/6e5f1203-8aad-4d2a-8439-27a5c8d0a2a7" />|<img width="480" height="360" alt="02_000000051704" src="https://github.com/user-attachments/assets/20fba8fc-5bf6-4485-9165-ce06a1504644" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/5e7cd0e1-8840-4a4e-b932-b68dc71f3721" />|<img width="480" height="360" alt="07_000000016314" src="https://github.com/user-attachments/assets/b9aced38-c4eb-4ee7-a6c5-294090748eca" />|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/d4b5fc1c-324b-4ed3-9ab7-35c1d022b2d0" />|<img width="480" height="360" alt="28_000000044437" src="https://github.com/user-attachments/assets/499469d3-41a3-406b-a075-8b055297bf6e" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/9be24b81-0c6e-48c7-bda1-1372b6ae391f" />|<img width="480" height="360" alt="42_000000006864" src="https://github.com/user-attachments/assets/fed81463-3cd6-4d69-8fc3-5bf114d92ab8" />|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/05280dce-4faf-46e1-817c-98e4df4a9d4c" />|<img width="383" height="640" alt="09_dist_000000074177" src="https://github.com/user-attachments/assets/628540af-6407-4e93-b41e-9057671f49ca" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/23d932d4-ce48-421e-9d1f-0dd8386852a3" />|<img width="500" height="500" alt="08_dist_000000039322" src="https://github.com/user-attachments/assets/64c744c4-4425-46da-a53b-ffab2fd52060" />|

- w/o ESE

  |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  |N|1.38 M|0.18 G|0.40343|0.93 ms|5.6 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_quality_nopost.onnx)|
  |T|3.10 M|0.41 G|0.44529|1.50 ms|12.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_quality_nopost.onnx)|
  |S|5.43 M|0.71 G|0.44945|2.23 ms|21.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_quality_nopost.onnx)|
  |C|8.46 M|1.11 G|0.45005|2.66 ms|33.9 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_quality_nopost.onnx)|
  |M|12.15 M|1.60 G|0.44875|4.07 ms|48.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_quality_nopost.onnx)|
  |L|21.54 M|2.83 G|0.44686|6.23 ms|86.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_quality_nopost.onnx)|

- w ESE

  |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  |N|1.45 M|0.18 G|0.41018|1.05 ms|5.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_se_64x64_quality_nopost.onnx)|
  |T|3.22 M|0.41 G|0.44130|1.27 ms|12.9 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_se_64x64_quality_nopost.onnx)|
  |S|5.69 M|0.71 G|0.46612|2.10 ms|22.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_se_64x64_quality_nopost.onnx)|
  |C|8.87 M|1.11 G|0.45095|2.86 ms|35.5 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_se_64x64_quality_nopost.onnx)|
  |M|12.74 M|1.60 G|0.46502|3.95 ms|51.0 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_se_64x64_quality_nopost.onnx)|
  |L|22.59 M|2.83 G|0.45787|6.52 ms|90.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_se_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_se_64x64_quality_nopost.onnx)|

- w ESE + IoU-aware + Swish

  |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  |N|1.60 M|0.20 G|0.42806|1.25 ms|6.5 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_se_iou_64x64_quality_nopost.onnx)|
  |T|3.56 M|0.45 G|0.46502|1.82 ms|14.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_se_iou_64x64_quality_nopost.onnx)|
  |S|6.30 M|0.79 G|0.47473|2.78 ms|25.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_se_iou_64x64_quality_nopost.onnx)|
  |C|9.81 M|1.23 G|0.46235|3.58 ms|39.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_se_iou_64x64_quality_nopost.onnx)|
  |M|14.09 M|1.77 G|0.46562|5.05 ms|56.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_se_iou_64x64_quality_nopost.onnx)|
  |L|24.98 M|3.13 G|0.47774|7.46 ms|100 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_se_iou_64x64_quality_nopost.onnx)|

- w ESE + IoU-aware + ReLU

  |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
  |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
  |N|1.60 M|0.20 G|0.40910|0.63 ms|6.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_quality_relu_nopost.onnx)|
  |T|3.56 M|0.45 G|0.44618|1.08 ms|14.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_quality_relu_nopost.onnx)|
  |S|6.30 M|0.79 G|0.45776|1.71 ms|25.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_quality_relu_nopost.onnx)|
  |C|9.81 M|1.23 G|0.45385|2.51 ms|39.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_quality_relu_nopost.onnx)|
  |M|14.09 M|1.77 G|0.47468|3.54 ms|56.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_quality_relu_nopost.onnx)|
  |L|24.98 M|3.13 G|0.47774|6.14 ms|100 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_quality_relu_nopost.onnx)|


## Inference
- ONNX with post-processing
  ```bash
  uv run demo_uhd.py \
  --onnx ultratinyod_res_anc8_w256_64x64_quality.onnx \
  --camera 0
  ```
- ONNX without post-processing
  ```bash
  uv run demo_uhd.py \
  --onnx ultratinyod_res_anc8_w256_64x64_quality_nopost.onnx \
  --camera 0
  ```

## Training Examples (full CLI)

UltraTinyOD (anchor-only, stride 8; `--cnn-width` controls stem width):

<details><summary>use-improved-head</summary>

```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=64
LR=0.005
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=96
LR=0.004
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=128
LR=0.003
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=160
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=192
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=256
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head
```

</details>

<details><summary>use-improved-head + utod-head-ese</summary>

```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=64
LR=0.005
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head \
--utod-head-ese
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=96
LR=0.004
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head \
--utod-head-ese
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=128
LR=0.003
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head \
--utod-head-ese
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=160
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head \
--utod-head-ese
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=192
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head \
--utod-head-ese
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=256
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_${SIZE}_${IMPHEAD}_lr${LR} \
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
--grad-clip-norm 10.0 \
--use-batchnorm \
--use-improved-head \
--utod-head-ese
```

</details>

<details><summary>use-improved-head + use-iou-aware-head + utod-head-ese</summary>

```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=64
LR=0.005
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_iou_${SIZE}_${IMPHEAD}_lr${LR} \
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
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=96
LR=0.004
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_iou_${SIZE}_${IMPHEAD}_lr${LR} \
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
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=128
LR=0.003
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_iou_${SIZE}_${IMPHEAD}_lr${LR} \
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
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=160
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_iou_${SIZE}_${IMPHEAD}_lr${LR} \
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
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=192
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_iou_${SIZE}_${IMPHEAD}_lr${LR} \
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
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese
```
```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=256
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_se_iou_${SIZE}_${IMPHEAD}_lr${LR} \
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
--use-ema \
--ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese
```

</details>

## Validation-only Example

Example of running only validation on a trained checkpoint:

```bash
uv run python train.py \
--arch ultratinyod \
--img-size 64x64 \
--cnn-width 256 \
--classes 0 \
--conf-thresh 0.15 \
--ckpt runs/ultratinyod_res_anc8_w256_64x64_lr0.0003/best_utod_0297_map_0.44299.pt \
--val-only \
--use-ema
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
| `--val-only` | Run validation only with `--ckpt` or `--resume` weights and exit. | `False` |
| `--use-improved-head` | UltraTinyOD only: enable quality-aware head (IoU-aware obj, IoU score branch, learnable WH scale, extra context). | `False` |
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
| `--use-batchnorm` | Enable BatchNorm layers during training/export. | `False` |
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
- `loss`: total anchor loss (`box + obj + cls` [+ `quality`] when `--use-improved-head`)
- `obj`: BCE on objectness for each anchor location (positive vs. background)
- `cls`: BCE on per-class logits for positive anchors (one-hot over target classes)
- `box`: (1 - IoU/GIoU/CIoU) on decoded boxes for positive anchors; IoU flavor set by `--iou-loss`
- `quality` (improved head only): BCE on IoU-linked quality logit; obj targetもIoUでスケールされる

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
  CNNWIDTH=64
  uv run python export_onnx.py \
  --checkpoint runs/ultratinyod_res_anc8_w64_64x64_quality_lr0.007/best_utod_0299_map_0.40343.pt \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_quality.onnx \
  --opset 17

  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=96
  uv run python export_onnx.py \
  --checkpoint runs/ultratinyod_res_anc8_w96_64x64_quality_lr0.004/best_utod_0296_map_0.44529.pt \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_quality.onnx \
  --opset 17

  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=128
  uv run python export_onnx.py \
  --checkpoint runs/ultratinyod_res_anc8_w128_64x64_quality_lr0.003/best_utod_0300_map_0.44945.pt \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_quality.onnx \
  --opset 17

  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=160
  uv run python export_onnx.py \
  --checkpoint runs/ultratinyod_res_anc8_w160_64x64_quality_lr0.001/best_utod_0300_map_0.45005.pt \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_quality.onnx \
  --opset 17

  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=192
  uv run python export_onnx.py \
  --checkpoint runs/ultratinyod_res_anc8_w192_64x64_quality_lr0.001/best_utod_0300_map_0.44875.pt \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_quality.onnx \
  --opset 17

  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=256
  uv run python export_onnx.py \
  --checkpoint runs/ultratinyod_res_anc8_w256_64x64_quality_lr0.001/best_utod_0300_map_0.44686.pt \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_quality.onnx \
  --opset 17
  ```

## INT8 quantization

```bash
uv run onnx2tf \
-i ultratinyod_res_anc8_w64_64x64_quality_relu_nopost.onnx \
-cotof \
-oiqt
```

## Arch

<img width="350" alt="ultratinyod_res_anc8_w64_64x64_quality" src="https://github.com/user-attachments/assets/e0302a17-1311-4fb0-be8d-4525a8228042" />

## Ultra-lightweight classification model series
1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License
6. [HSC: Happy smile classifier](https://github.com/PINTO0309/HSC) - MIT License
7. [WHC: Waving Hand Classification](https://github.com/PINTO0309/WHC) - MIT License
8. [UHD: Ultra-lightweight human detection](https://github.com/PINTO0309/UHD) - MIT License

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025uhd,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/UHD},
  month     = {12},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17790207},
  url       = {https://github.com/PINTO0309/uhd},
  abstract  = {Ultra-lightweight human detection. The number of parameters does not correlate to inference speed.},
}
```
