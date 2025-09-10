import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
src1 = BASE_DIR/"training_images"/"detail_front"
src2 = BASE_DIR/"training_images"/"detail_top"
dst  = BASE_DIR/"training_images"/"detail_top_front"


os.makedirs(dst, exist_ok=True)

for src in [src1, src2]:
    for fname in os.listdir(src):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            src_path = os.path.join(src, fname)
            dst_path = os.path.join(dst, fname)
            shutil.copy(src_path, dst_path)

for src in [src1, src2]: 
     shutil.rmtree(src)
print("Images combined into:", dst)
