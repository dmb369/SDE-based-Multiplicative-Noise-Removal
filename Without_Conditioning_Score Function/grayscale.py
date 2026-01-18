import os
from PIL import Image
from tqdm import tqdm

input_dir = "./datasets/denoised_t100_ddim_False_det_False"     # folder with RGB PNGs
output_dir = "./datasets/denoised"  # new folder

os.makedirs(output_dir, exist_ok=True)

for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path)

    # Convert to grayscale (L mode = single channel)
    gray = img.convert("L")

    gray.save(os.path.join(output_dir, fname))
