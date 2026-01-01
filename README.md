# UHD
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17790207.svg)](https://doi.org/10.5281/zenodo.17790207) ![GitHub License](https://img.shields.io/github/license/pinto0309/uhd) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/uhd)

Ultra-lightweight human detection. The number of parameters does not correlate to inference speed. For limited use cases, an input image resolution of 64x64 is sufficient. High-level object detection architectures such as YOLO are overkill.

**Please note that the dataset used to train this model is a custom-created, ultra-high-quality dataset derived from MS-COCO. Therefore, a simple comparison with the Val mAP values ​​of other object detection models is completely meaningless. In particular, please note that the mAP values ​​of other MS-COCO-based models are unnecessarily high and do not accurately assess actual performance.**

This model is an experimental implementation and is not suitable for real-time inference using a USB camera, etc.

https://github.com/user-attachments/assets/afca301a-9fe6-4ecd-af01-6bacbfa88e52

- Variant-S / `w ESE + IoU-aware + ReLU`

|Input<br>64x64|Output|Input<br>64x64|Output|
|:-:|:-:|:-:|:-:|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/59b941a4-86eb-41b9-9c0a-c367e6718b4c" />|<img width="640" height="427" alt="00_dist_000000075375" src="https://github.com/user-attachments/assets/5cb2e332-86eb-4933-bf80-3f500173306f" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/21791bff-2cf4-4a63-87b5-f6da52bb4964" />|<img width="480" height="360" alt="01_000000073639" src="https://github.com/user-attachments/assets/ed0f9b67-7a05-4e31-8f43-8e82a468ad38" />|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/6e5f1203-8aad-4d2a-8439-27a5c8d0a2a7" />|<img width="480" height="360" alt="02_000000051704" src="https://github.com/user-attachments/assets/20fba8fc-5bf6-4485-9165-ce06a1504644" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/5e7cd0e1-8840-4a4e-b932-b68dc71f3721" />|<img width="480" height="360" alt="07_000000016314" src="https://github.com/user-attachments/assets/b9aced38-c4eb-4ee7-a6c5-294090748eca" />|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/d4b5fc1c-324b-4ed3-9ab7-35c1d022b2d0" />|<img width="480" height="360" alt="28_000000044437" src="https://github.com/user-attachments/assets/499469d3-41a3-406b-a075-8b055297bf6e" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/9be24b81-0c6e-48c7-bda1-1372b6ae391f" />|<img width="480" height="360" alt="42_000000006864" src="https://github.com/user-attachments/assets/fed81463-3cd6-4d69-8fc3-5bf114d92ab8" />|
|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/05280dce-4faf-46e1-817c-98e4df4a9d4c" />|<img width="383" height="640" alt="09_dist_000000074177" src="https://github.com/user-attachments/assets/628540af-6407-4e93-b41e-9057671f49ca" />|<img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/23d932d4-ce48-421e-9d1f-0dd8386852a3" />|<img width="500" height="500" alt="08_dist_000000039322" src="https://github.com/user-attachments/assets/64c744c4-4425-46da-a53b-ffab2fd52060" />|

## Download all ONNX files at once
**I don't recommend it as it downloads a crazy amount of files.**

```bash
sudo apt update && sudo apt install -y gh
gh release download onnx -R PINTO0309/UHD
```

## Models

- Legacy models

  <details><summary>Click to expand</summary>

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

  - ESE + IoU-aware + Swish

    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.60 M|0.20 G|0.42806|1.25 ms|6.5 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_se_iou_64x64_quality_nopost.onnx)|
    |T|3.56 M|0.45 G|0.46502|1.82 ms|14.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_se_iou_64x64_quality_nopost.onnx)|
    |S|6.30 M|0.79 G|0.47473|2.78 ms|25.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_se_iou_64x64_quality_nopost.onnx)|
    |C|9.81 M|1.23 G|0.46235|3.58 ms|39.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_se_iou_64x64_quality_nopost.onnx)|
    |M|14.09 M|1.77 G|0.46562|5.05 ms|56.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_se_iou_64x64_quality_nopost.onnx)|
    |L|24.98 M|3.13 G|0.47774|7.46 ms|100 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_se_iou_64x64_quality.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_se_iou_64x64_quality_nopost.onnx)|

  - ESE + IoU-aware + ReLU

    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.60 M|0.20 G|0.40910|0.63 ms|6.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_quality_relu_nopost.onnx)|
    |T|3.56 M|0.45 G|0.44618|1.08 ms|14.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_quality_relu_nopost.onnx)|
    |S|6.30 M|0.79 G|0.45776|1.71 ms|25.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_quality_relu_nopost.onnx)|
    |C|9.81 M|1.23 G|0.45385|2.51 ms|39.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_quality_relu_nopost.onnx)|
    |M|14.09 M|1.77 G|0.47468|3.54 ms|56.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_quality_relu_nopost.onnx)|
    |L|24.98 M|3.13 G|0.46965|6.14 ms|100 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_quality_relu.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_quality_relu_nopost.onnx)|

  - ESE + IoU-aware + large-object-branch + ReLU

    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.98 M|0.22 G|0.40903|0.77 ms|8.0 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_loese_nopost.onnx)|
    |T|4.40 M|0.49 G|0.46170|1.40 ms|17.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_loese_nopost.onnx)|
    |S|7.79 M|0.87 G|0.45860|2.30 ms|31.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_loese_nopost.onnx)|
    |C|12.13 M|1.35 G|0.47518|2.83 ms|48.6 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_loese_nopost.onnx)|
    |M|17.44 M|1.94 G|0.45816|4.37 ms|69.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_loese_nopost.onnx)|
    |L|30.92 M|3.44 G|0.48243|7.40 ms|123.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_loese.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_loese_nopost.onnx)|

  - **[For long distances and extremely small objects]** ESE + IoU-aware + ReLU + Distillation

    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.60 M|0.20 G|0.55224|0.63 ms|6.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_distill_nopost.onnx)|
    |T|3.56 M|0.45 G|0.56040|1.08 ms|14.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_distill_nopost.onnx)|
    |S|6.30 M|0.79 G|0.57361|1.71 ms|25.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_distill_nopost.onnx)|
    |C|9.81 M|1.23 G|0.56183|2.51 ms|39.3 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_distill_nopost.onnx)|
    |M|14.09 M|1.77 G|0.57666|3.54 ms|56.4 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_distill_nopost.onnx)|

  - **[For short/medium distance]** ESE + IoU-aware + large-object-branch + ReLU + Distillation

    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.98 M|0.22 G|0.54883|0.70 ms|8.0 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_loese_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_loese_distill_nopost.onnx)|
    |T|4.40 M|0.49 G|0.55663|1.18 ms|17.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_loese_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_loese_distill_nopost.onnx)|
    |S|7.79 M|0.87 G|0.57397|1.97 ms|31.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_loese_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_loese_distill_nopost.onnx)|
    |C|12.13 M|1.35 G|0.56768|2.74 ms|48.6 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_loese_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_loese_distill_nopost.onnx)|
    |M|17.44 M|1.94 G|0.57815|3.57 ms|69.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_loese_distill.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_loese_distill_nopost.onnx)|

  - `torch_bilinear_dynamic` + No resizing required + Not suitable for quantization
  
    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.98 M|0.22 G|0.55489|0.70 ms|8.0 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_torch_bilinear_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_torch_bilinear_dynamic_nopost.onnx)|
    |T|4.40 M|0.49 G|0.57824|1.18 ms|17.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_torch_bilinear_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_torch_bilinear_dynamic_nopost.onnx)|
    |S|7.79 M|0.87 G|0.58478|1.97 ms|31.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_torch_bilinear_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_torch_bilinear_dynamic_nopost.onnx)|
    |C|12.13 M|1.35 G|0.58459|2.74 ms|48.6 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_torch_bilinear_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_torch_bilinear_dynamic_nopost.onnx)|
    |M|17.44 M|1.94 G|0.59034|3.57 ms|69.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_torch_bilinear_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_torch_bilinear_dynamic_nopost.onnx)|
    |L|30.92 M|3.44 G|0.58929|7.16 ms|123.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_torch_bilinear_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_torch_bilinear_dynamic_nopost.onnx)|
  
  - `torch_nearest_dynamic` + No resizing required + Suitable for quantization
  
    |Variant|Params|FLOPs|mAP@0.5|Corei9 CPU<br>inference<br>latency|ONNX<br>File size|ONNX|w/o post<br>ONNX|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |N|1.98 M|0.22 G|0.53376|0.70 ms|8.0 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_torch_nearest_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_torch_nearest_dynamic_nopost.onnx)|
    |T|4.40 M|0.49 G|0.55561|1.18 ms|17.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_torch_nearest_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w96_64x64_torch_nearest_dynamic_nopost.onnx)|
    |S|7.79 M|0.87 G|0.56396|1.97 ms|31.2 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_torch_nearest_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w128_64x64_torch_nearest_dynamic_nopost.onnx)|
    |C|12.13 M|1.35 G|0.56328|2.74 ms|48.6 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_torch_nearest_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w160_64x64_torch_nearest_dynamic_nopost.onnx)|
    |M|17.44 M|1.94 G|0.57075|3.57 ms|69.8 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_torch_nearest_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w192_64x64_torch_nearest_dynamic_nopost.onnx)|
    |L|30.92 M|3.44 G|0.56787|7.16 ms|123.7 MB|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_torch_nearest_dynamic.onnx)|[Download](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_torch_nearest_dynamic_nopost.onnx)|

  </details>

- Variants
  ```
  R: ront, Y: yocto, Z: zepto, A: atto
  F: femto, P: pico, N: nano, T: tiny
  S: small, C: compact, M: medium, L: large
  ```
- `opencv_inter_nearest` + Optimized for OpenCV RGB downsampling + Suitable for quantization

  - ONNX

    |Var|Param|FLOPs|@0.5|Corei9<br>CPU<br>latency|ONNX<br>size|static|w/o post|dynamic|w/o post|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |R|0.13 M|0.01 G|0.21230|0.24 ms|863 KB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_dynamic_nopost.onnx)|
    |Y|0.29 M|0.03 G|0.28664|0.28 ms|1.2 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_dynamic_nopost.onnx)|
    |Z|0.51 M|0.05 G|0.32722|0.32 ms|2.1 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_dynamic_nopost.onnx)|
    |A|0.78 M|0.08 G|0.43661|0.37 ms|3.2 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_dynamic_nopost.onnx)|
    |F|1.12 M|0.12 G|0.47942|0.44 ms|4.5 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_dynamic_nopost.onnx)|
    |P|1.52 M|0.17 G|0.51094|0.50 ms|6.1 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_dynamic_nopost.onnx)|
    |N|1.98 M|0.22 G|0.55003|0.60 ms|8.0 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_dynamic_nopost.onnx)|
    |T|2.49 M|0.28 G|0.56550|0.70 ms|10.0 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_dynamic_nopost.onnx)|
    |S|3.07 M|0.34 G|0.57015|0.81 ms|12.3 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_dynamic_nopost.onnx)|
    |L|30.92 M|3.44 G|0.58399|7.16 ms|123.7 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_dynamic_nopost.onnx)|

  - ESPDL INT8 (.espdl, .info, .json, anchors.npy, wh_scale.npy)
  
    |Var|ESPDL size|s3<br>Emphasis<br>on speed|s3<br>latency|s3<br>Emphasis<br>on precision|s3<br>latency|p4<br>static<br>w/o post|p4<br>latency|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |R|218.3 KB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_static_nopost_nocat_espdl.tar.gz)|11.28 ms|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_static_nopost_nocat_espdl_highacc.tar.gz)|20.05 ms|||
    |Y|380.2 KB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_static_nopost_nocat_espdl.tar.gz)|26.15 ms|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_static_nopost_nocat_espdl_highacc.tar.gz)|28.51 ms|||
    |Z|601.1 KB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_static_nopost_nocat_espdl.tar.gz)|33.66 ms|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_static_nopost_nocat_espdl_highacc.tar.gz)|48.57 ms|||
    |A|882.8 KB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_static_nopost_nocat_espdl.tar.gz)|127.51 ms|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_static_nopost_nocat_espdl_highacc.tar.gz)|205.96 ms|||
    |F|1.2 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_static_nopost_nocat_espdl.tar_nocat.gz)|292.63 ms|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_static_nopost_nocat_espdl_highacc.tar_nocat_espdl_highacc.gz)|-|||
    |P|1.6 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_static_nopost_nocat_espdl.tar_nocat.gz)|545.65 ms|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_static_nopost_nocat_espdl_highacc.tar_nocat_espdl_highacc.gz)|-|||
    |N|2.1 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_static_nopost_nocat_espdl.tar_nocat.gz)|808.49 ms|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_static_nopost_nocat_espdl_highacc.tar_nocat_espdl_highacc.gz)|-|||

    - ESP32-S3-EYE - Emphasis on precision `R`

      https://github.com/user-attachments/assets/ad417641-0c06-4307-9224-a1bdb9402dbd

- `opencv_inter_nearest_yuv422` + Optimized for YUV422 + Suitable for quantization
  - `YUV422`
    ```python
    img_u8 = np.ones([64,64,3], dtype=np.uint8)
    yuyv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YUV_YUYV)
    print(yuyv.shape)
    (64, 64, 2)
    ```
  - With post-process model
    ```
    input_name.1: input_yuv422 shape: [1, 2, 64, 64] dtype: float32

    output_name.1: score_classid_cxcywh shape: [1, 100, 6] dtype: float32
    ```
  - Without post-process model
    ```
    input_name.1: input_yuv422 shape: [1, 2, 64, 64] dtype: float32

    output_name.1: txtywh_obj_quality_cls_x8 shape: [1, 56, 8, 8] dtype: float32
    output_name.2: anchors shape: [8, 2] dtype: float32
    output_name.3: wh_scale shape: [8, 2] dtype: float32

    https://github.com/PINTO0309/UHD/blob/e0bbfe69afa0da4f83cf1f09b530a500bcd2d685/demo_uhd.py#L203-L301

    score = sigmoid(obj) * (sigmoid(quality)) * sigmoid(cls)
    cx = (sigmoid(tx)+gx)/w
    cy = (sigmoid(ty)+gy)/h
    bw = anchor_w*softplus(tw)*wh_scale
    bh = anchor_h*softplus(th)*wh_scale
    boxes = (cx±bw/2, cy±bh/2)
    ```
  - ONNX

    |Var|Param|FLOPs|@0.5|Corei9<br>CPU<br>latency|ONNX<br>size|static|w/o post|dynamic|w/o post|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |R|0.13 M|0.01 G|0.22382|0.34 ms|863 KB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_yuv422_distill_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_yuv422_distill_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_yuv422_distill_dynamic_nopost.onnx)|
    |Y|0.29 M|0.03 G|0.29606|0.38 ms|1.2 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_yuv422_distill_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_yuv422_distill_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_yuv422_distill_dynamic_nopost.onnx)|
    |Z|0.51 M|0.05 G|0.36843|0.43 ms|2.1 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_yuv422_distill_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_yuv422_distill_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_yuv422_distill_dynamic_nopost.onnx)|
    |A|0.78 M|0.08 G|0.42872|0.48 ms|3.2 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_yuv422_distill_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_yuv422_distill_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_yuv422_distill_dynamic_nopost.onnx)|
    |F|1.12 M|0.12 G|0.49098|0.54 ms|4.5 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_yuv422_distill_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_yuv422_distill_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_yuv422_distill_dynamic_nopost.onnx)|
    |P|1.52 M|0.17 G|0.52665|0.63 ms|6.1 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_yuv422_distill_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_yuv422_distill_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_yuv422_distill_dynamic_nopost.onnx)|
    |N|1.98 M|0.22 G|0.54942|0.70 ms|8.0 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_yuv422_distill_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_yuv422_distill_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_yuv422_distill_dynamic_nopost.onnx)|
    |T|2.49 M|0.28 G|0.56300|0.83 ms|10.0 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_yuv422_distill_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_yuv422_distill_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_yuv422_distill_dynamic_nopost.onnx)|
    |S|3.07 M|0.34 G|0.57338|0.91 ms|12.3 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_yuv422_distill_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_yuv422_distill_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_yuv422_distill_dynamic_nopost.onnx)|
    |L|30.92 M|3.44 G|0.58642|7.16 ms|123.7 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_yuv422_distill_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_yuv422_distill_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_yuv422_distill_dynamic_nopost.onnx)|

  - Input image 480x360 -> OpenCV INTER_NEAREST -> 64x64 -> YUV422 (packed: YUY2/YUYV)

    |100%|800% zoom|
    |:-:|:-:|
    |<img width="64" height="64" alt="resized_64x64_nearest" src="https://github.com/user-attachments/assets/fd4d0ee3-1d6a-4819-a324-05f406de999c" />|<img width="515" height="515" alt="image" src="https://github.com/user-attachments/assets/ec62913a-236b-4c94-8ce7-e20ff67b69f8" />|

  - `Y` detection sample

    <img width="480" height="360" alt="00_000000019456" src="https://github.com/user-attachments/assets/fa2ee106-4208-492e-a2be-4ecc91281129" />

  - `F` detection sample

    <img width="480" height="360" alt="00_000000019456" src="https://github.com/user-attachments/assets/f759800b-f70b-43a9-b7aa-4ac0650b37ec" />

  - `N` detection sample

    <img width="480" height="360" alt="00_000000019456" src="https://github.com/user-attachments/assets/c8a75dbd-2956-4de1-b42a-8d381f20dc07" />

  - `S` detection sample

    <img width="480" height="360" alt="00_000000019456" src="https://github.com/user-attachments/assets/9c01234d-6aab-4815-bd33-688effdc6c3a" />

  - ESPDL INT8 (.espdl, .info, .json, anchors.npy, wh_scale.npy)
  
    |Var|ESPDL size|static w/o post<br>s3|static w/o post<br>p4|
    |:-:|:-:|:-:|:-:|
    |R|218.3 KB|||||
    |Y|380.2 KB|||||
    |Z|601.1 KB|||||
    |A|882.8 KB|||||
    |F|1.2 MB|||||
    |P|1.6 MB|||||
    |N|2.1 MB|||||

- `opencv_inter_nearest_y` + Optimized for Y (Luminance) only + Suitable for quantization
  - `Y`
    ```python
    img_u8 = np.ones([64,64,3], dtype=np.uint8)
    yuv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YUV)
    y = yuv[..., 0:1]
    print(y.shape)
    (64, 64, 1)
    ```
  - With post-process model
    ```
    input_name.1: input_y shape: [1, 1, 64, 64] dtype: float32

    output_name.1: score_classid_cxcywh shape: [1, 100, 6] dtype: float32
    ```
  - Without post-process model
    ```
    input_name.1: input_yuv422 shape: [1, 1, 64, 64] dtype: float32

    output_name.1: txtywh_obj_quality_cls_x8 shape: [1, 56, 8, 8] dtype: float32
    output_name.2: anchors shape: [8, 2] dtype: float32
    output_name.3: wh_scale shape: [8, 2] dtype: float32

    https://github.com/PINTO0309/UHD/blob/e0bbfe69afa0da4f83cf1f09b530a500bcd2d685/demo_uhd.py#L203-L301

    score = sigmoid(obj) * (sigmoid(quality)) * sigmoid(cls)
    cx = (sigmoid(tx)+gx)/w
    cy = (sigmoid(ty)+gy)/h
    bw = anchor_w*softplus(tw)*wh_scale
    bh = anchor_h*softplus(th)*wh_scale
    boxes = (cx±bw/2, cy±bh/2)
    ```
  - ONNX

    |Var|Param|FLOPs|@0.5|CPU<br>latency|ONNX<br>size|static|w/o post|dynamic|w/o post|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |R|0.13 M|0.01 G|0.21863|0.34 ms|863 KB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_y_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_y_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_y_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_y_dynamic_nopost.onnx)|
    |Y|0.29 M|0.03 G|0.29523|0.38 ms|1.2 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_y_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_y_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_y_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_y_dynamic_nopost.onnx)|
    |Z|0.51 M|0.05 G|0.33219|0.43 ms|2.1 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_y_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_y_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_y_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_y_dynamic_nopost.onnx)|
    |A|0.78 M|0.08 G|0.41927|0.48 ms|3.2 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_y_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_y_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_y_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_y_dynamic_nopost.onnx)|
    |F|1.12 M|0.12 G|0.46622|0.54 ms|4.5 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_y_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_y_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_y_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_y_dynamic_nopost.onnx)|
    |P|1.52 M|0.17 G|0.51670|0.63 ms|6.1 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_y_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_y_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_y_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_y_dynamic_nopost.onnx)|
    |N|1.98 M|0.22 G|0.54105|0.70 ms|8.0 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_y_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_y_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_y_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_y_dynamic_nopost.onnx)|
    |T|2.49 M|0.28 G|0.55636|0.83 ms|10.0 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_y_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_y_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_y_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_y_dynamic_nopost.onnx)|
    |S|3.07 M|0.34 G|0.56824|0.91 ms|12.3 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_y_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_y_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_y_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w80_64x64_opencv_inter_nearest_y_dynamic_nopost.onnx)|
    |L|30.92 M|3.44 G|0.58164|7.16 ms|123.7 MB|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_y_static.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_y_static_nopost.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_y_dynamic.onnx)|[DL](https://github.com/PINTO0309/UHD/releases/download/onnx/ultratinyod_res_anc8_w256_64x64_opencv_inter_nearest_y_dynamic_nopost.onnx)|

  - ESPDL INT8 (.espdl, .info, .json, anchors.npy, wh_scale.npy)

    |Var|ESPDL size|static w/o post<br>s3|static w/o post<br>p4|
    |:-:|:-:|:-:|:-:|
    |R|218.3 KB|||||
    |Y|380.2 KB|||||
    |Z|601.1 KB|||||
    |A|882.8 KB|||||
    |F|1.2 MB|||||
    |P|1.6 MB|||||
    |N|2.1 MB|||||

- `opencv_inter_nearest_y_tri` + Optimized for Y (Luminance) only + Y ternarization + Suitable for quantization
  - `opencv_inter_nearest_y_tri` uses fixed Y thresholds (1/3, 2/3) to quantize to 3 levels: 0.0, 0.5, 1.0.
  - `Y_TRI`
    ```python
    img_u8 = np.ones([64,64,3], dtype=np.uint8)
    yuv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YUV)
    y = yuv[..., 0:1]
    y = y / 255.0
    y = np.clip(y, 0.0, 1.0)
    out = np.zeros_like(y, dtype=np.float32)
    mid_mask = (y >= t1) & (y < t2)
    out[mid_mask] = 0.5
    out[y >= t2] = 1.0
    print(out.shape)
    (64, 64, 1)
    ```
  - With post-process model
    ```
    input_name.1: input_y_tri shape: [1, 1, 64, 64] dtype: float32

    output_name.1: score_classid_cxcywh shape: [1, 100, 6] dtype: float32
    ```
  - Without post-process model
    ```
    input_name.1: input_y_tri shape: [1, 1, 64, 64] dtype: float32

    output_name.1: txtywh_obj_quality_cls_x8 shape: [1, 56, 8, 8] dtype: float32
    output_name.2: anchors shape: [8, 2] dtype: float32
    output_name.3: wh_scale shape: [8, 2] dtype: float32

    https://github.com/PINTO0309/UHD/blob/e0bbfe69afa0da4f83cf1f09b530a500bcd2d685/demo_uhd.py#L203-L301

    score = sigmoid(obj) * (sigmoid(quality)) * sigmoid(cls)
    cx = (sigmoid(tx)+gx)/w
    cy = (sigmoid(ty)+gy)/h
    bw = anchor_w*softplus(tw)*wh_scale
    bh = anchor_h*softplus(th)*wh_scale
    boxes = (cx±bw/2, cy±bh/2)
    ```

  **WIP**

  <img width="480" height="360" alt="00_000000019456" src="https://github.com/user-attachments/assets/adebb29e-d310-4453-acde-ae88887ed4ef" />

- `opencv_inter_nearest_y_bin` + Optimized for Y (Luminance) only + Y binarization + Suitable for quantization
  - `opencv_inter_nearest_y_bin` uses a fixed Y threshold (0.5) to binarize to 0.0/1.0.
  - `Y_BIN`
    ```python
    img_u8 = np.ones([64,64,3], dtype=np.uint8)
    yuv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YUV)
    y = yuv[..., 0:1]
    y = y / 255.0
    y = np.clip(y, 0.0, 1.0)
    out = (y >= threshold).astype(np.float32)
    print(out.shape)
    (64, 64, 1)
    ```
  - With post-process model
    ```
    input_name.1: input_y_bin shape: [1, 1, 64, 64] dtype: float32

    output_name.1: score_classid_cxcywh shape: [1, 100, 6] dtype: float32
    ```
  - Without post-process model
    ```
    input_name.1: input_y_bin shape: [1, 1, 64, 64] dtype: float32

    output_name.1: txtywh_obj_quality_cls_x8 shape: [1, 56, 8, 8] dtype: float32
    output_name.2: anchors shape: [8, 2] dtype: float32
    output_name.3: wh_scale shape: [8, 2] dtype: float32

    https://github.com/PINTO0309/UHD/blob/e0bbfe69afa0da4f83cf1f09b530a500bcd2d685/demo_uhd.py#L203-L301

    score = sigmoid(obj) * (sigmoid(quality)) * sigmoid(cls)
    cx = (sigmoid(tx)+gx)/w
    cy = (sigmoid(ty)+gy)/h
    bw = anchor_w*softplus(tw)*wh_scale
    bh = anchor_h*softplus(th)*wh_scale
    boxes = (cx±bw/2, cy±bh/2)
    ```

  **WIP**

  <img width="480" height="360" alt="00_000000019456" src="https://github.com/user-attachments/assets/973ca376-51c4-40bf-b9dc-aacec4bfdcf7" />

## Inference

> [!CAUTION]
> If you preprocess your images and resize them to 64x64 with OpenCV or similar, use `Nearest` mode.

<details><summary>Click to expand</summary>

```bash
usage: demo_uhd.py
[-h]
(--images IMAGES | --camera CAMERA)
--onnx ONNX
[--output OUTPUT]
[--img-size IMG_SIZE]
[--conf-thresh CONF_THRESH]
[--record RECORD]
[--actual-size]
[--use-nms]
[--nms-iou NMS_IOU]

UltraTinyOD ONNX demo (CPU).

options:
  -h, --help
   show this help message and exit
  --images IMAGES
   Directory with images to run batch inference.
  --camera CAMERA
   USB camera id for realtime inference.
  --onnx ONNX
   Path to ONNX model (CPU).
  --output OUTPUT
   Output directory for image mode.
  --img-size IMG_SIZE
   Input size HxW, e.g., 64x64.
  --conf-thresh CONF_THRESH
   Confidence threshold. Default: 0.90
  --record RECORD
   MP4 path for automatic recording when --camera is used.
  --actual-size
   Display and recording use the model input resolution instead of
   the original frame size.
  --use-nms
   Apply Non-Maximum Suppression on decoded boxes (default IoU=0.8).
  --nms-iou NMS_IOU
   IoU threshold for NMS (effective only when --use-nms is set).
```

- ONNX with post-processing
  ```bash
  uv run demo_uhd.py \
  --onnx ultratinyod_res_anc8_w192_64x64_loese_distill.onnx \
  --camera 0 \
  --conf-thresh 0.90 \
  --use-nms \
  --actual-size
  ```
- ONNX without post-processing
  ```bash
  uv run demo_uhd.py \
  --onnx ultratinyod_res_anc8_w192_64x64_loese_distill_nopost.onnx \
  --camera 0 \
  --conf-thresh 0.90 \
  --use-nms \
  --actual-size
  ```
- ONNX with pre-processing (PIL equivalent of Resize) + post-processing
  ```bash
  uv run demo_uhd.py \
  --onnx ultratinyod_res_anc8_w256_64x64_torch_bilinear_dynamic.onnx \
  --camera 0 \
  --conf-thresh 0.90 \
  --use-nms \
  --actual-size
  ```
- ONNX with pre-processing (PIL equivalent of Resize) + without post-processing
  ```bash
  uv run demo_uhd.py \
  --onnx ultratinyod_res_anc8_w256_64x64_torch_bilinear_dynamic_nopost.onnx \
  --camera 0 \
  --conf-thresh 0.90 \
  --use-nms \
  --actual-size
  ```

</details>

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

<details><summary>use-improved-head + use-iou-aware-head + utod-head-ese + distillation</summary>

```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=64
LR=0.0003
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/4/ultratinyod_res_anc8_w64_se_iou_64x64_quality_lr0.005_relu/best_utod_0299_map_0.40910.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/4/ultratinyod_res_anc8_w256_se_iou_64x64_quality_lr0.001_relu/best_utod_0300_map_0.46965.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine


SIZE=64x64
ANCHOR=8
CNNWIDTH=96
LR=0.0003
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/3/ultratinyod_res_anc8_w96_se_iou_64x64_quality_lr0.004/best_utod_0293_map_0.46502.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/4/ultratinyod_res_anc8_w256_se_iou_64x64_quality_lr0.001_relu/best_utod_0300_map_0.46965.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine

SIZE=64x64
ANCHOR=8
CNNWIDTH=128
LR=0.0003
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/4/ultratinyod_res_anc8_w128_se_iou_64x64_quality_lr0.003_relu/best_utod_0293_map_0.45776.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/4/ultratinyod_res_anc8_w256_se_iou_64x64_quality_lr0.001_relu/best_utod_0300_map_0.46965.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine

SIZE=64x64
ANCHOR=8
CNNWIDTH=160
LR=0.0001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/4/ultratinyod_res_anc8_w160_se_iou_64x64_quality_lr0.001_relu/best_utod_0300_map_0.45385.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/4/ultratinyod_res_anc8_w256_se_iou_64x64_quality_lr0.001_relu/best_utod_0300_map_0.46965.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine

SIZE=64x64
ANCHOR=8
CNNWIDTH=192
LR=0.0001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/4/ultratinyod_res_anc8_w192_se_iou_64x64_quality_lr0.001_relu/best_utod_0300_map_0.47468.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/4/ultratinyod_res_anc8_w256_se_iou_64x64_quality_lr0.001_relu/best_utod_0300_map_0.46965.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine
```

</details>

<details><summary>use-improved-head + use-iou-aware-head + utod-head-ese + utod-large-obj-branch</summary>

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
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_loese_${SIZE}_${IMPHEAD}_lr${LR} \
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
--utod-head-ese \
--activation relu \
--utod-large-obj-branch \
--utod-large-obj-depth 2 \
--utod-large-obj-ch-scale 1.25

SIZE=64x64
ANCHOR=8
CNNWIDTH=96
LR=0.004
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_loese_${SIZE}_${IMPHEAD}_lr${LR} \
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
--utod-head-ese \
--activation relu \
--utod-large-obj-branch \
--utod-large-obj-depth 2 \
--utod-large-obj-ch-scale 1.25

SIZE=64x64
ANCHOR=8
CNNWIDTH=128
LR=0.003
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_loese_${SIZE}_${IMPHEAD}_lr${LR} \
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
--utod-head-ese \
--activation relu \
--utod-large-obj-branch \
--utod-large-obj-depth 2 \
--utod-large-obj-ch-scale 1.25

SIZE=64x64
ANCHOR=8
CNNWIDTH=160
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_loese_${SIZE}_${IMPHEAD}_lr${LR} \
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
--utod-head-ese \
--activation relu \
--utod-large-obj-branch \
--utod-large-obj-depth 2 \
--utod-large-obj-ch-scale 1.25

ANCHOR=8
CNNWIDTH=192
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_loese_${SIZE}_${IMPHEAD}_lr${LR} \
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
--utod-head-ese \
--activation relu \
--utod-large-obj-branch \
--utod-large-obj-depth 2 \
--utod-large-obj-ch-scale 1.25

SIZE=64x64
ANCHOR=8
CNNWIDTH=256
LR=0.001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_loese_${SIZE}_${IMPHEAD}_lr${LR} \
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
--utod-head-ese \
--activation relu \
--utod-large-obj-branch \
--utod-large-obj-depth 2 \
--utod-large-obj-ch-scale 1.25
```

</details>

<details><summary>use-improved-head + use-iou-aware-head + utod-head-ese + utod-large-obj-branch + distillation</summary>

```bash
SIZE=64x64
ANCHOR=8
CNNWIDTH=64
LR=0.0003
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/6/ultratinyod_res_anc8_w64_loese_64x64_quality_lr0.005/best_utod_0296_map_0.40903.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/6/ultratinyod_res_anc8_w256_loese_64x64_quality_lr0.001/best_utod_0179_map_0.48243.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine

SIZE=64x64
ANCHOR=8
CNNWIDTH=96
LR=0.0003
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/6/ultratinyod_res_anc8_w96_loese_64x64_quality_lr0.004/best_utod_0279_map_0.46170.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/6/ultratinyod_res_anc8_w256_loese_64x64_quality_lr0.001/best_utod_0179_map_0.48243.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine

SIZE=64x64
ANCHOR=8
CNNWIDTH=128
LR=0.0003
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/6/ultratinyod_res_anc8_w128_loese_64x64_quality_lr0.003/best_utod_0259_map_0.45860.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/6/ultratinyod_res_anc8_w256_loese_64x64_quality_lr0.001/best_utod_0179_map_0.48243.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine

SIZE=64x64
ANCHOR=8
CNNWIDTH=160
LR=0.0001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/6/ultratinyod_res_anc8_w160_loese_64x64_quality_lr0.001/best_utod_0210_map_0.47518.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/6/ultratinyod_res_anc8_w256_loese_64x64_quality_lr0.001/best_utod_0179_map_0.48243.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine

SIZE=64x64
ANCHOR=8
CNNWIDTH=192
LR=0.0001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/4/ultratinyod_res_anc8_w192_se_iou_64x64_quality_lr0.001_relu/best_utod_0300_map_0.47468.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/6/ultratinyod_res_anc8_w256_loese_64x64_quality_lr0.001/best_utod_0179_map_0.48243.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine

SIZE=64x64
ANCHOR=8
CNNWIDTH=256
LR=0.0001
IMPHEAD=quality
uv run python train.py \
--arch ultratinyod \
--image-dir data/wholebody34/obj_train_data \
--img-size ${SIZE} \
--exp-name ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${IMPHEAD}_lr${LR}_relu_distill \
--ckpt runs/4/ultratinyod_res_anc8_w256_se_iou_64x64_quality_lr0.001_relu/best_utod_0300_map_0.46965.pt \
--batch-size 64 \
--epochs 300 \
--lr ${LR} \
--weight-decay 0.0001 \
--num-workers 12 \
--device cuda \
--classes 0 \
--cnn-width 64 \
--auto-anchors \
--num-anchors 8 \
--iou-loss ciou \
--conf-thresh 0.15 \
--use-ema --ema-decay 0.9999 \
--grad-clip-norm 10.0 \
--use-batchnorm \
--utod-residual \
--use-improved-head \
--use-iou-aware-head \
--utod-head-ese \
--activation relu \
--teacher-ckpt runs/6/ultratinyod_res_anc8_w256_loese_64x64_quality_lr0.001/best_utod_0179_map_0.48243.pt \
--teacher-arch ultratinyod \
--distill-kl 1.0 \
--distill-box-l1 1.0 \
--distill-feat 0.5 \
--distill-temperature 2.0 \
--distill-cosine
```

</details>

## Validation-only Example

<details><summary>Click to expand</summary>

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
--use-ema \
--with-heatmap-mean
```

When `--with-heatmap-mean` or `--with-heatmap-sum` is set, validation sample outputs are written under
`runs/<exp-name>/val_only/<sample_name>/`, containing the usual detection PNG plus per-layer heatmap PNGs
overlayed on the detection image.

</details>

## CLI parameters

<details><summary>Click to expand</summary>

| Parameter | Description | Default |
| --- | --- | --- |
| `--arch` | Model architecture: `cnn`, `transformer`, or anchor-only `ultratinyod`. | `cnn` |
| `--image-dir` | Directory containing images and YOLO txt labels. | `data/wholebody34/obj_train_data` |
| `--train-split` | Fraction of data used for training. | `0.8` |
| `--val-split` | Fraction of data used for validation. | `0.2` |
| `--img-size` | Input size `HxW` (e.g., `64x64`). | `64x64` |
| `--resize-mode` | Resize mode for training preprocessing: `torch_bilinear`, `torch_nearest`, `opencv_inter_linear`, `opencv_inter_nearest`, `opencv_inter_nearest_y`, `opencv_inter_nearest_y_bin`, `opencv_inter_nearest_y_tri`, `opencv_inter_nearest_yuv422`. | `torch_bilinear` |
| `--torch_bilinear` | Shortcut for `--resize-mode torch_bilinear`. | `False` |
| `--torch_nearest` | Shortcut for `--resize-mode torch_nearest`. | `False` |
| `--opencv_inter_linear` | Shortcut for `--resize-mode opencv_inter_linear`. | `False` |
| `--opencv_inter_nearest` | Shortcut for `--resize-mode opencv_inter_nearest`. | `False` |
| `--opencv_inter_nearest_y` | Shortcut for `--resize-mode opencv_inter_nearest_y`. | `False` |
| `--opencv_inter_nearest_y_bin` | Shortcut for `--resize-mode opencv_inter_nearest_y_bin`. | `False` |
| `--opencv_inter_nearest_y_tri` | Shortcut for `--resize-mode opencv_inter_nearest_y_tri`. | `False` |
| `--opencv_inter_nearest_yuv422` | Shortcut for `--resize-mode opencv_inter_nearest_yuv422`. | `False` |
| `--exp-name` | Experiment name; logs saved under `runs/<exp-name>`. | `default` |
| `--batch-size` | Batch size. | `64` |
| `--epochs` | Number of epochs. | `100` |
| `--resume` | Checkpoint to resume training (loads optimizer/scheduler). | `None` |
| `--ckpt` | Initialize weights from checkpoint (no optimizer state). | `None` |
| `--ckpt-non-strict` | Load `--ckpt` with `strict=False` (ignore missing/unexpected keys). | `False` |
| `--val-only` | Run validation only with `--ckpt` or `--resume` weights and exit. | `False` |
| `--val-count` | Limit number of validation images when using `--val-only`. | `None` |
| `--with-heatmap-mean` | Save per-layer heatmaps using channel mean and overlay them on detection images. | `False` |
| `--with-heatmap-sum` | Save per-layer heatmaps using channel sum and overlay them on detection images. | `False` |
| `--use-improved-head` | UltraTinyOD only: enable quality-aware head (IoU-aware obj, IoU score branch, learnable WH scale, extra context). | `False` |
| `--use-iou-aware-head` | UltraTinyOD head: task-aligned IoU-aware scoring (quality*cls) with split towers. | `False` |
| `--quality-power` | Exponent for quality score when using IoU-aware head scoring. | `1.0` |
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
| `--distill-obj` | Objectness distillation weight (anchor head). | `0.0` |
| `--distill-quality` | Quality-score distillation weight (anchor head). | `0.0` |
| `--distill-cosine` | Cosine ramp-up of distillation weights. | `False` |
| `--distill-temperature` | Teacher logits temperature. | `1.0` |
| `--distill-feat` | Feature-map distillation weight (CNN only). | `0.0` |
| `--lr` | Learning rate. | `0.001` |
| `--weight-decay` | Weight decay. | `0.0001` |
| `--optimizer` | Optimizer (`adamw` or `sgd`). | `adamw` |
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
| `--utod-residual` | Enable residual skips inside the UltraTinyOD backbone. | `False` |
| `--utod-head-ese` | UltraTinyOD head: apply lightweight eSE on shared features. | `False` |
| `--utod-context-rfb` | UltraTinyOD head: add a receptive-field block (dilated + wide depthwise) before prediction layers. | `False` |
| `--utod-context-dilation` | Dilation used in UltraTinyOD receptive-field block (only when `--utod-context-rfb`). | `2` |
| `--utod-large-obj-branch` | UltraTinyOD head: add a downsampled large-object refinement branch (no FPN). | `False` |
| `--utod-large-obj-depth` | Number of depthwise blocks in the large-object branch (only when `--utod-large-obj-branch`). | `2` |
| `--utod-large-obj-ch-scale` | Channel scale for the large-object branch (relative to head channels). | `1.0` |
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

</details>

## Augmentation via YAML

<details><summary>Click to expand</summary>

- Specify a YAML file with `--aug-config` to run the `data_augment:` entries in the listed order (e.g., `--aug-config uhd/aug.yaml`).
- Supported ops (examples): Mosaic / MixUp / CopyPaste / HorizontalFlip (class_swap_map supported) / VerticalFlip / RandomScale / Translation / RandomCrop / RandomResizedCrop / RandomBrightness / RandomContrast / RandomSaturation / RandomHSV / RandomPhotometricDistort / Blur / MedianBlur / MotionBlur / GaussianBlur / GaussNoise / ImageCompression / ISONoise / RandomRain / RandomFog / RandomSunFlare / CLAHE / ToGray / RemoveOutliers.
- If `prob` is provided, it is used as the apply probability; otherwise defaults are used (most are 0, RandomPhotometricDistort defaults to 0.5). Unknown keys are ignored.

</details>

## Loss terms (CNN / CenterNet)

<details><summary>Click to expand</summary>

- `loss`: total loss (`hm + off + wh`)
- `hm`: focal loss on center heatmap
- `off`: L1 loss on center offsets (within-cell quantization correction)
- `wh`: L1 loss on width/height (feature-map scale)

</details>

## Loss terms (CNN / Anchor head, `--use-anchor`)

<details><summary>Click to expand</summary>

- `loss`: total anchor loss (`box + obj + cls` [+ `quality`] when `--use-improved-head`)
- `obj`: BCE on objectness for each anchor location (positive vs. background)
- `cls`: BCE on per-class logits for positive anchors (one-hot over target classes)
- `box`: (1 - IoU/GIoU/CIoU) on decoded boxes for positive anchors; IoU flavor set by `--iou-loss`
- `quality` (improved head only): BCE on IoU-linked quality logit; obj targetもIoUでスケールされる

</details>

## Loss terms (Transformer)

<details><summary>Click to expand</summary>

- `loss`: total loss (`cls + l1 + iou`)
- `cls`: cross-entropy for class vs. background
- `l1`: L1 loss on box coordinates
- `iou`: 1 - IoU for matched predictions

</details>

## The impact of image downsampling methods

<details><summary>Click to expand</summary>

PyTorch's Resize method is implemented using a downsampling method similar to PIL on the backend, but it is significantly different from OpenCV's downsampling implementation. Therefore, when downsampling images in preprocessing during training, it is important to note that the numerical characteristics of the images used by the model for training will be completely different depending on whether you use PyTorch's Resize method or OpenCV's Resize method. Below is the pixel-level error calculation when downsampling an image to 64x64 pixels. If the diff value is greater than 1.0, the images are completely different.

Therefore, it is easy to imagine that if the downsampling method used for preprocessing during learning is different from the downsampling method used during inference, the output inference results will be disastrous.

The internal workings of PyTorch's downsampling and PIL's downsampling are very similar but slightly different. When deploying and inferencing in Python and other environments, accuracy will be significantly degraded unless the model is deployed according to the following criteria. If you train using OpenCV's `cv2.INTER_LINEAR`, the model will never produce the correct output after preprocessing in PyTorch, TensorFlow, or ONNX other than OpenCV.

|Training|Deploy|
|:-|:-|
|When training while downsampling using PyTorch's Resize (`InterpolationMode.BILINEAR`)|Merge `Resize Linear + half-pixel` at the input of the ONNX model. This will result in the highest model accuracy. However, it will be limited to deployment on hardware, NPUs, and frameworks that support the resize operation of bilinear interpolation. It is not suitable for quantization.|
|When training while downsampling using PyTorch's Resize (`InterpolationMode.NEAREST`)|Merge `Resize Nearest` at the input of the ONNX model. It is the most versatile in terms of HW, NPU, and quantization deployment, but the accuracy of the model will be lower.|
|When training while downsampling using OpenCV's Resize (`cv2.INTER_NEAREST`)|Merge `Resize Nearest` at the input of the ONNX model or OpenCV `INTER_NEAREST`. Although the accuracy is low, it is highly versatile because the downsampling of images can be freely written on the program side. However, downsampling must be implemented manually.|

1. Error after PIL conversion when downsampling with PyTorch's Resize InterpolationMode.BILINEAR
    ```
    PyTorch(InterpolationMode.BILINEAR) -> Convert to PIL vs PyTorch Tensor(InterpolationMode.BILINEAR)
      max  diff : 1
      mean diff : 0.4949
      std  diff : 0.5000
    ```
2. Error when downsampling with PyTorch's Resize InterpolationMode.BILINEAR compared to downsampling with OpenCV's INTER_LINER
    ```
    PyTorch(InterpolationMode.BILINEAR) -> Convert to PIL vs OpenCV INTER_LINEAR
      max  diff : 104
      mean diff : 10.2930
      std  diff : 13.2792
    ```
3. Error when downsampling with PyTorch's Resize InterpolationMode.BILINEAR compared to downsampling with OpenCV's INTER_LINER
    ```
    PyTorch Tensor(InterpolationMode.BILINEAR) vs OpenCV INTER_LINEAR
      max  diff : 104
      mean diff : 10.3336
      std  diff : 13.2463
    ```
4. Accuracy and speed of each interpolation method when downsampling in OpenCV
    - Accuracy: `INTER_NEAREST < INTER_LINEAR < INTER_AREA`
    - Speed: `INTER_NEAREST > INTER_LINEAR > INTER_AREA`
    ```
    === Resize benchmark ===
    INTER_NEAREST : 0.0061 ms
    INTER_LINEAR  : 0.0143 ms
    INTER_AREA    : 0.3621 ms
    AREA / LINEAR ratio : 25.40x
    ```

</details>

## ONNX export

<details><summary>Click to expand</summary>

- Export a checkpoint to ONNX (auto-detects arch from checkpoint unless overridden):
  ```bash
  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=64
  RESIZEMODE=opencv_inter_nearest_y
  CKPT=runs/ultratinyod_res_anc8_w64_loese_64x64_lr0.005_impaug/best_utod_0001_map_0.00000.pt
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static.onnx \
  --opset 17
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic.onnx \
  --opset 17 \
  --dynamic-resize
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --noconcat_box_obj_quality_cls
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --dynamic-resize \
  --noconcat_box_obj_quality_cls

  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=96
  RESIZEMODE=opencv_inter_nearest_y
  CKPT=runs/ultratinyod_res_anc8_w96_loese_64x64_lr0.005_impaug/best_utod_0001_map_0.00000.pt
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static.onnx \
  --opset 17
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic.onnx \
  --opset 17 \
  --dynamic-resize
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --noconcat_box_obj_quality_cls
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --dynamic-resize \
  --noconcat_box_obj_quality_cls

  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=128
  RESIZEMODE=opencv_inter_nearest_y
  CKPT=runs/ultratinyod_res_anc8_w128_loese_64x64_lr0.005_impaug/best_utod_0001_map_0.00000.pt
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static.onnx \
  --opset 17
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic.onnx \
  --opset 17 \
  --dynamic-resize
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --noconcat_box_obj_quality_cls
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --dynamic-resize \
  --noconcat_box_obj_quality_cls

  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=160
  RESIZEMODE=opencv_inter_nearest_y
  CKPT=runs/ultratinyod_res_anc8_w160_loese_64x64_lr0.005_impaug/best_utod_0001_map_0.00000.pt
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static.onnx \
  --opset 17
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic.onnx \
  --opset 17 \
  --dynamic-resize
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --noconcat_box_obj_quality_cls
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --dynamic-resize \
  --noconcat_box_obj_quality_cls

  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=192
  RESIZEMODE=opencv_inter_nearest_y
  CKPT=runs/ultratinyod_res_anc8_w192_loese_64x64_lr0.005_impaug/best_utod_0001_map_0.00000.pt
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static.onnx \
  --opset 17
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic.onnx \
  --opset 17 \
  --dynamic-resize
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --noconcat_box_obj_quality_cls
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --dynamic-resize \
  --noconcat_box_obj_quality_cls

  SIZE=64x64
  ANCHOR=8
  CNNWIDTH=256
  RESIZEMODE=opencv_inter_nearest_y
  CKPT=runs/ultratinyod_res_anc8_w256_loese_64x64_lr0.005_impaug/best_utod_0001_map_0.00000.pt
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static.onnx \
  --opset 17
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic.onnx \
  --opset 17 \
  --dynamic-resize
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_static_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --noconcat_box_obj_quality_cls
  uv run python export_onnx.py \
  --checkpoint ${CKPT} \
  --output ultratinyod_res_anc${ANCHOR}_w${CNNWIDTH}_${SIZE}_${RESIZEMODE}_dynamic_nopost.onnx \
  --opset 17 \
  --no-merge-postprocess \
  --dynamic-resize \
  --noconcat_box_obj_quality_cls
  ```

</details>

## LiteRT (TFLite) quantization

<details><summary>Click to expand</summary>

```bash
uv run onnx2tf \
-i ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_static_nopost_nocat.onnx \
-cotof \
-oiqt \
-qnm "[[[[0.0, 0.0, 0.0]]]]" \
-qns "[[[[1.0, 1.0, 1.0]]]]"
```

</details>

## ESP-DL Quantization

This repository includes a calibration/quantization script for ESP-DL:
`uhd/quantize_onnx_model_for_esp32.py`.

### Image-only calibration (default)

```bash
### Example command that prioritizes maximum speed at the expense of accuracy
uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest \
--onnx-model ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_static_nopost_nocat.onnx \
--espdl-model ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_static_nopost_nocat.espdl \
--target esp32s3 \
--calib-algorithm kl

uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest \
--onnx-model ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_static_nopost_nocat.onnx \
--espdl-model ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_static_nopost_nocat.espdl \
--target esp32p4 \
--calib-algorithm kl

### Example command to minimize quantization error
uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest \
--onnx-model ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_static_nopost_nocat.onnx \
--espdl-model ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_static_nopost_nocat_acc.espdl \
--target esp32s3 \
--calib-algorithm kl \
--int16-op-pattern /box_out/Conv \
--int16-op-pattern /large_obj_down/large_obj_down.3/conv/Conv \
--int16-op-pattern /large_obj_down/large_obj_down.3/act/Relu \
--int16-op-pattern /backbone/block1/pw/conv/Conv \
--int16-op-pattern /backbone/block1/pw/act/Relu \
--int16-op-pattern /depthwiseconv/backbone/block2/dw/conv/Conv \
--int16-op-pattern /backbone/block2/dw/act/Relu \
--int16-op-pattern /backbone/block3_skip/conv/Conv \
--int16-op-pattern /backbone/block2/pw/conv/Conv \
--int16-op-pattern /backbone/block2/pw/act/Relu \
--int16-op-pattern /depthwiseconv/backbone/block1/dw/conv/Conv \
--int16-op-pattern /backbone/block1/dw/act/Relu \
--int16-op-pattern /backbone/sppf/cv1/conv/Conv \
--int16-op-pattern /backbone/sppf/cv1/act/Relu \
--int16-op-pattern /backbone/stem/conv/Conv \
--int16-op-pattern /backbone/stem/act/Relu \
--int16-op-pattern /large_obj_fuse/conv/Conv \
--int16-op-pattern /large_obj_fuse/act/Relu \
--int16-op-pattern /large_obj_blocks/large_obj_blocks.1/pw/conv/Conv \
--int16-op-pattern /large_obj_blocks/large_obj_blocks.0/pw/act/Relu \
--int16-op-pattern /large_obj_blocks/large_obj_blocks.0/pw/conv/Conv \
--int16-op-pattern /large_obj_blocks/large_obj_blocks.1/pw/act/Relu
```

<details><summary>Click to expand</summary>

```bash
uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest_yuv422 \
--onnx-model ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx \
--espdl-model ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.espdl \
--target "esp32s3" \
--calib-algorithm kl \
--int16-op-pattern "/model/backbone/block1/dw/conv/Conv" \
--int16-op-pattern "/model/head/context_res/context_res.2/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.1/dw/conv/Conv"

uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest_yuv422 \
--onnx-model ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx \
--espdl-model ultratinyod_res_anc8_w24_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.espdl \
--target esp32s3 \
--calib-algorithm kl \
--int16-op-pattern "/model/head/context_res/context_res.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/box_tower/box_tower.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/quality_tower/quality_tower.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.1/dw/conv/Conv"

uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest_yuv422 \
--onnx-model ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx \
--espdl-model ultratinyod_res_anc8_w32_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.espdl \
--target esp32s3 \
--calib-algorithm kl \
--int16-op-pattern "/model/head/context_res/context_res.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/context_res/context_res.2/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/box_tower/box_tower.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/quality_tower/quality_tower.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/obj_conv/dw/conv/Conv"

uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest_yuv422 \
--onnx-model ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx \
--espdl-model ultratinyod_res_anc8_w40_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.espdl \
--target esp32s3 \
--calib-algorithm kl \
--int16-op-pattern "/model/backbone/block1/dw/conv/Conv" \
--int16-op-pattern "/model/head/context_res/context_res.2/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_down/large_obj_down.0/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/box_tower/box_tower.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/obj_conv/dw/conv/Conv" \
--int16-op-pattern "/model/head/quality_tower/quality_tower.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/box_tower/box_tower.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/quality_tower/quality_tower.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/context_res/context_res.0/dw/conv/Conv"

uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest_yuv422 \
--onnx-model ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx \
--espdl-model ultratinyod_res_anc8_w48_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.espdl \
--target esp32s3 \
--calib-algorithm kl \
--int16-op-pattern "/model/backbone/block1/dw/conv/Conv" \
--int16-op-pattern "/model/head/context/dw/conv/Conv" \
--int16-op-pattern "/model/head/context_res/context_res.2/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/box_tower/box_tower.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/quality_tower/quality_tower.1/dw/conv/Conv"

uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest_yuv422 \
--onnx-model ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx \
--espdl-model ultratinyod_res_anc8_w56_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.espdl \
--target esp32s3 \
--calib-algorithm kl \
--int16-op-pattern "/model/head/context_res/context_res.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/context_res/context_res.2/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/box_tower/box_tower.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/quality_tower/quality_tower.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.0/dw/conv/Conv"

uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest_yuv422 \
--onnx-model ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx \
--espdl-model ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.espdl \
--target esp32s3 \
--calib-algorithm kl \
--int16-op-pattern "/model/head/context_res/context_res.2/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.0/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/box_tower/box_tower.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/quality_tower/quality_tower.1/dw/conv/Conv"

uv run python uhd/quantize_onnx_model_for_esp32.py \
--dataset-type image \
--image-dir data/wholebody34/obj_train_data \
--resize-mode opencv_inter_nearest_yuv422 \
--onnx-model ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx \
--espdl-model ultratinyod_res_anc8_w72_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.espdl \
--target esp32s3 \
--calib-algorithm kl \
--int16-op-pattern "/model/head/context_res/context_res.2/dw/conv/Conv" \
--int16-op-pattern "/model/head/large_obj_blocks/large_obj_blocks.1/dw/conv/Conv" \
--int16-op-pattern "/model/head/box_tower/box_tower.1/dw/conv/Conv"
```

Notes:
- The YUV422 models expect input shape `[1, 2, 64, 64]` with `opencv_inter_nearest_yuv422` preprocessing.
- `--dataset-type image` is the default and ignores labels.
- Adjust `--calib-steps`, `--batch-size`, `--target`, `--num-of-bits`, and `--device` as needed.

</details>

### CLI options

<details><summary>Click to expand</summary>

- `--image-dir`: Directory containing calibration images.
- `--dataset-type`: Calibration dataset type (`image` or `yolo`, default `image`).
- `--list-path`: Optional text file listing images to use.
- `--export-anchors-wh-scale-dir`: Directory to save `{onnx-model}_anchors.npy` and `{onnx-model}_wh_scale.npy` (default: same directory as `--espdl-model`).
- `--expand-group-conv`: Expand `groups > 1` conv into group=1 (default: disabled).
- `--img-size`: Square input size used for calibration (default `64`).
- `--resize-mode`: Resize mode (default `opencv_inter_nearest_yuv422`).
- `--class-ids`: Comma-separated class IDs to keep (yolo only, default `0`).
- `--split`: Dataset split for calibration (`train`, `val`, `all`, default `all`).
- `--val-split`: Validation split ratio (ignored when `--split all`, default `0.0`).
- `--batch-size`: Calibration batch size (default `1`).
- `--calib-steps`: Number of calibration steps (default `32`).
- `--calib-algorithm`: Calibration algorithm (default `kl`; examples: `minmax`, `mse`, `percentile`).
- `--int16-op-pattern`: Regex pattern to force matched ops to int16 (repeatable).
- `--onnx-model`: Path to the input ONNX model (default `ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.onnx`).
- `--espdl-model`: Path to the output `.espdl` file (default `ultratinyod_res_anc8_w16_64x64_opencv_inter_nearest_yuv422_distill_static_nopost.espdl`).
- `--target`: Quantize target type (`c`, `esp32s3`, `esp32p4`, default `esp32s3`).
- `--num-of-bits`: Quantization bits (default `8`).
- `--device`: Device for calibration (`cpu` or `cuda`, default `cpu`).

</details>

## Arch

<details><summary>Click to expand</summary>

|ONNX|LiteRT(TFLite)|
|:-:|:-:|
|<img width="350" alt="ultratinyod_res_anc8_w64_64x64_loese_distill" src="https://github.com/user-attachments/assets/ae5d3c70-8c5e-41f0-ad79-98f4024519a0" />|<img width="350" alt="ultratinyod_res_anc8_w64_64x64_loese_distill_float32" src="https://github.com/user-attachments/assets/c1e909b8-b029-4f0a-acf9-0f8259763ec3" />|

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

</details>

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

## Ref

- https://github.com/espressif/esp-dl
- https://github.com/espressif/esp-who
- https://github.com/espressif/esp-who/blob/master/docs/en/get-started/ESP32-S3-EYE_Getting_Started_Guide.md
- [ESP32 - ESP-IDF](https://zenn.dev/pinto0309/scraps/6b985872ea9a89)
  - ESP-WHO + human_face_recognition

    https://github.com/user-attachments/assets/f67c0470-f74a-4a70-9e14-213bfc141064

- **[WIP]** ESP32 UHD: https://github.com/PINTO0309/esp-who/tree/custom/examples/ultra_lightweight_human_detection
  - `opencv_inter_nearest` + `Z`, 33ms/pred

    https://github.com/user-attachments/assets/191894fe-fc70-4158-a8f2-474ba18942ab
