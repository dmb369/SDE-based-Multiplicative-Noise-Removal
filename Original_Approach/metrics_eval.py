import os
import argparse
import csv
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms as T

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -----------------------------
# Optional LPIPS
# -----------------------------
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


# -----------------------------
# Utility functions
# -----------------------------
def list_images(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])


def load_grayscale_as_numpy(path, size=None):
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.array(img)


def load_grayscale_as_rgb_tensor(path, size=None):
    img = Image.open(path).convert("L").convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    tensor = T.ToTensor()(img)          # [0,1]
    tensor = tensor * 2.0 - 1.0          # [-1,1]
    return tensor


# -----------------------------
# LPIPS wrapper
# -----------------------------
class LPIPSEval:
    def __init__(self, device):
        self.loss_fn = lpips.LPIPS(net="alex").to(device)
        self.device = device

    def __call__(self, img1, img2):
        with torch.no_grad():
            return self.loss_fn(img1, img2).item()


# -----------------------------
# FID utilities
# -----------------------------
class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
        )
        self.model.fc = nn.Identity()
        self.model.eval()

    def forward(self, x):
        out = self.model(x)
        # torchvision may return InceptionOutputs
        if isinstance(out, tuple):
            out = out[0]
        return out


def inception_activations(image_paths, device):
    model = InceptionV3().to(device)

    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])

    acts = []

    with torch.no_grad():
        for p in tqdm(image_paths):
            img = Image.open(p).convert("L").convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            feat = model(img)
            acts.append(feat.cpu().numpy())

    return np.concatenate(acts, axis=0)


def compute_fid(clean_dir, denoised_dir, device):
    clean_paths = list_images(clean_dir)
    den_paths = list_images(denoised_dir)

    assert len(clean_paths) == len(den_paths), "Image count mismatch"

    a1 = inception_activations(clean_paths, device)
    a2 = inception_activations(den_paths, device)

    mu1, mu2 = a1.mean(axis=0), a2.mean(axis=0)
    sigma1 = np.cov(a1, rowvar=False)
    sigma2 = np.cov(a2, rowvar=False)

    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu1 - mu2) ** 2) + np.trace(
        sigma1 + sigma2 - 2 * covmean
    )

    return float(fid)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--denoised_dir", required=True)
    parser.add_argument("--out_csv", default="metrics.csv")
    parser.add_argument("--fid", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_paths = list_images(args.clean_dir)
    den_paths = list_images(args.denoised_dir)

    assert len(clean_paths) == len(den_paths), "Mismatch in image count"

    lpips_eval = LPIPSEval(device) if HAS_LPIPS else None

    rows = []
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for cp, dp in zip(clean_paths, den_paths):
        clean_img = Image.open(cp).convert("L")
        clean_np = np.array(clean_img)
        
        den_np = load_grayscale_as_numpy(dp, size=clean_img.size)

        psnr = peak_signal_noise_ratio(clean_np, den_np, data_range=255)
        ssim = structural_similarity(clean_np, den_np, data_range=255)

        lp = None
        if lpips_eval is not None:
            # Load clean image to get reference size
            clean_img = Image.open(cp).convert("L")
            ref_size = clean_img.size  # (W, H)
            t1 = load_grayscale_as_rgb_tensor(cp, size=ref_size).unsqueeze(0).to(device)
            t2 = load_grayscale_as_rgb_tensor(dp, size=ref_size).unsqueeze(0).to(device)

            lp = lpips_eval(t1, t2)
            lpips_vals.append(lp)

        psnr_vals.append(psnr)
        ssim_vals.append(ssim)

        rows.append([
            os.path.basename(cp),
            psnr,
            ssim,
            lp,
            ""
        ])

    fid_val = compute_fid(args.clean_dir, args.denoised_dir, device) if args.fid else None

    rows.append([
        "AVERAGE",
        np.mean(psnr_vals),
        np.mean(ssim_vals),
        np.mean(lpips_vals) if lpips_vals else "",
        fid_val if fid_val is not None else ""
    ])

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "psnr", "ssim", "lpips", "fid"])
        writer.writerows(rows)

    print(f"Done. Results saved to {args.out_csv}")


if __name__ == "__main__":
    main()