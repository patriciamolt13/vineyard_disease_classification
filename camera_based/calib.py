import os
import random
import numpy as np
from PIL import Image

folder = "calib_images"
out_file = "calib.npy"

classes = [
    "black_rot",
    "downy_mildew",
    "esca",
    "healthy",
    "powdery_mildew"
]

random.seed(42)

exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

imgs_per_class = 205   # 205 x 5 = 1025, depois cortamos para 1024

paths = []

for c in classes:
    class_folder = os.path.join(folder, c)

    class_paths = []

    for name in os.listdir(class_folder):
        if name.lower().endswith(exts):
            class_paths.append(os.path.join(class_folder, name))

    random.shuffle(class_paths)
    paths.extend(class_paths[:imgs_per_class])

random.shuffle(paths)
paths = paths[:1024]

imgs = []

for path in paths:
    img = Image.open(path).convert("RGB")
    img = img.resize((240, 240), Image.BICUBIC)
    img = np.array(img).astype(np.uint8)
    imgs.append(img)

calib = np.array(imgs)

print("Number of images:", len(imgs))
print("Shape:", calib.shape)
print("Dtype:", calib.dtype)
print("Min:", calib.min())
print("Max:", calib.max())
print("mean RGB:", calib.mean(axis=(0, 1, 2)))
print("std RGB:", calib.std(axis=(0, 1, 2)))

np.save(out_file, calib)

print("Saved:", out_file)
