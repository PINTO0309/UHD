import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import time

# --------------------------------------------------
# 設定
# --------------------------------------------------
IMAGE_PATH = "data/wholebody34/obj_train_data/000000000036.jpg"   # 任意の画像
OUT_SIZE = (64, 64)       # (H, W) ← 小さいほど差が出る

# --------------------------------------------------
# 入力画像
# --------------------------------------------------
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)

# --------------------------------------------------
# torchvision Resize
# --------------------------------------------------
resize_tv = T.Resize(OUT_SIZE, interpolation=T.InterpolationMode.BILINEAR)

# (A) PIL → torchvision
tv_pil = np.array(resize_tv(pil_img))

# (B) Tensor → torchvision
tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
tv_tensor = resize_tv(tensor_img)
tv_tensor = (tv_tensor * 255.0).clamp(0, 255).byte()
tv_tensor = tv_tensor.permute(1, 2, 0).numpy()

# --------------------------------------------------
# OpenCV resize（デフォルト）
# --------------------------------------------------
cv_resized = cv2.resize(
    img_rgb,
    (OUT_SIZE[1], OUT_SIZE[0]),
    interpolation=cv2.INTER_LINEAR
)

# --------------------------------------------------
# 差分計算関数
# --------------------------------------------------
def diff_stat(a, b, name):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    print(f"\n{name}")
    print(f"  max  diff : {d.max()}")
    print(f"  mean diff : {d.mean():.4f}")
    print(f"  std  diff : {d.std():.4f}")

# --------------------------------------------------
# 実測
# --------------------------------------------------
diff_stat(tv_pil, tv_tensor, "torchvision PIL vs torchvision Tensor")
diff_stat(tv_pil, cv_resized, "torchvision PIL vs OpenCV INTER_LINEAR")
diff_stat(tv_tensor, cv_resized, "torchvision Tensor vs OpenCV INTER_LINEAR")


print('')
# --------------------------------------------------
# 設定
# --------------------------------------------------
H_OUT, W_OUT = 64, 64   # 出力解像度（UltraTiny系）
N_ITER = 2000           # 繰り返し回数（ウォームアップ後）

# --------------------------------------------------
# ダミー入力
# --------------------------------------------------
img = img_bgr

# --------------------------------------------------
# ウォームアップ
# --------------------------------------------------
for _ in range(50):
    cv2.resize(img, (W_OUT, H_OUT), interpolation=cv2.INTER_LINEAR)
    cv2.resize(img, (W_OUT, H_OUT), interpolation=cv2.INTER_AREA)

# --------------------------------------------------
# 計測関数
# --------------------------------------------------
def benchmark(interp):
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        cv2.resize(img, (W_OUT, H_OUT), interpolation=interp)
    t1 = time.perf_counter()
    return (t1 - t0) / N_ITER * 1000  # ms

# --------------------------------------------------
# 実測
# --------------------------------------------------
t_nearest = benchmark(cv2.INTER_NEAREST)
t_linear = benchmark(cv2.INTER_LINEAR)
t_area   = benchmark(cv2.INTER_AREA)

print("=== Resize benchmark ===")
print(f"INTER_NEAREST : {t_nearest:.4f} ms")
print(f"INTER_LINEAR : {t_linear:.4f} ms")
print(f"INTER_AREA   : {t_area:.4f} ms")
print(f"AREA / LINEAR ratio : {t_area / t_linear:.2f}x")