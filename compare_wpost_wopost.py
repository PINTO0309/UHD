from pathlib import Path
import cv2, numpy as np
from demo_uhd import load_session, preprocess, run_and_decode, postprocess, parse_size

img_dir = Path("partial_images")
img_size = parse_size("64x64")
models = {
    "post": "ultratinyod_res_anc8_w256_64x64_quality.onnx",
    "nopost": "ultratinyod_res_anc8_w256_64x64_quality_nopost.onnx",
}

sessions, infos = {}, {}
for k, p in models.items():
    sess, info = load_session(p, img_size)
    sessions[k], infos[k] = sess, info

imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
stats = {k: {"boxes":0,"images":0,"max_w":0,"max_h":0,"avg_w":0,"avg_h":0} for k in models}

for img_path in imgs:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None: continue
    h, w = img.shape[:2]
    for name in models:
        dets = run_and_decode(sessions[name], infos[name], preprocess(img, img_size), 0.3)
        boxes = postprocess(dets, (h, w), 0.3)
        if boxes:
            s = stats[name]
            s["images"] += 1; s["boxes"] += len(boxes)
            for _,_,x1,y1,x2,y2 in boxes:
                bw, bh = x2 - x1, y2 - y1
                s["max_w"] = max(s["max_w"], bw); s["max_h"] = max(s["max_h"], bh)
                s["avg_w"] += bw; s["avg_h"] += bh

for name, s in stats.items():
    count = s["boxes"] or 1
    s["avg_w"] /= count; s["avg_h"] /= count
    print(name, s)
