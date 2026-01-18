# Multiplicative Noise Removal using Score-Based SDE Models

This repository contains the implementation and experiments for **multiplicative noise removal** using **score-based diffusion / stochastic differential equation (SDE) models** with architectural and inference-time enhancements. The project focuses on **SAR and optical imagery** from the **Sentinel‑1 and Sentinel‑2** satellites and demonstrates robust denoising performance on highly complex, real-world geospatial data.

---

## 📌 Project Overview

Multiplicative noise (e.g., speckle in SAR images) is signal-dependent and significantly harder to remove than additive Gaussian noise. Traditional denoising methods often oversmooth textures or fail to preserve structural details.

This project extends a **score-based SDE framework** for multiplicative noise removal with the following key ideas:

* Adaptive conditioning using **global image statistics**
* **Spatial self-attention** in the UNet bottleneck
* **Image-guided score estimation** using the original corrupted observation
* **Langevin refinement** during inference

The framework is evaluated on **Sentinel‑1 (SAR)** and **Sentinel‑2 (RGB)** image pairs and shows strong improvements across both **reconstruction metrics (PSNR, SSIM)** and **perceptual metrics (LPIPS, FID)**.

---

## 🛰 Dataset Description

### Dataset Source

* **Platform:** Kaggle
* **Original Provider:** Technical University of Munich (TUM)
* **Dataset Name:** *Sentinel‑1 & Sentinel‑2 Image Pairs*
* **Authors:** Michael Schmitt et al., TUM

### Modalities

* **Sentinel‑1:** SAR (Synthetic Aperture Radar)
* **Sentinel‑2:** Optical RGB imagery

### Terrain Classes

Images were curated from the **fall season** and grouped into four land-cover classes:

* 🌾 Agricultural land
* 🌱 Grassland
* 🏜 Barren land
* 🏙 Urban areas

The dataset intentionally contains **high intra-class variability**, irregular geometry, and no fixed orientation, making it suitable for testing **model robustness**.

### Dataset Structure

```
v_2/
├── agri/
│   ├── sar/
│   └── optical/
├── barrenland/
│   ├── sar/
│   └── optical/
├── grassland/
│   ├── sar/
│   └── optical/
└── urban/
    ├── sar/
    └── optical/
```

* **Total files:** ~32,000
* **Version size:** ~2.73 GB

### Intended Usage

* Conditional GANs
* Diffusion / Score-based models
* SAR despeckling
* Robustness testing for vision models

### License

* **Creative Commons Attribution 4.0 (CC BY 4.0)**

---

## 🧠 Methodology Summary

### Base Model

* Score-based diffusion using **Stochastic Differential Equations (SDEs)**
* Multiplicative noise modeled in the **log domain**
* UNet backbone

### Enhancements

1. **Adaptive Conditioning**
   Global image statistics (mean, variance, spatial correlation, edge energy) embedded into the time-conditioning pipeline.

2. **Spatial Self-Attention**
   Enables long-range dependency modeling and improved texture preservation.

3. **Image-Guided Score Function**
   The score network is conditioned on both the current diffusion state and the original corrupted image.

4. **Langevin Refinement**
   Post-sampling MCMC refinement to further suppress residual noise and sharpen details.

---

## ⚙️ Training

### Training Command

```bash
python3 train_denoise_sde.py \
  --img-size 224 \
  --batch-size 32 \
  --img-dir ./datasets/sample \
  --img-clean-dir ./datasets/clean \
  --device mps \
  --cpu-workers 5 \
  --exp-name noise-removal-celeba-large-RGB-224-500 \
  --epochs 15 \
  --save-every 1 \
  --unet-channels 128 \
  --unet-in-ch 3 \
  --unet-out-ch 3 \
  --unet-num-res 2 \
  --ddpm-num-steps 500 \
  --lr 1e-6
```

### Key Training Details

* Image resolution: **224 × 224**
* Diffusion steps: **500**
* Optimizer: **Adam**
* Learning rate: **1e‑6**
* Training setup supports **CPU / Apple MPS**

---

## 🧪 Test Data Generation

Synthetic noisy test data is generated using partial diffusion trajectories.

```bash
python3 generate_test_data.py \
  --img-size 224 \
  --batch-size 16 \
  --dataset celeba \
  --img-dir ./datasets/img_celeba \
  --ddpm-num-steps 500 \
  --ddpm-target-steps 150 \
  --img-data-amount test \
  --img-out ./datasets/celeba_prepared/img_celeba_test_t150
```

---

## 🔍 Inference / Denoising

```bash
python3 denoise_sde.py \
  --img-size 224 \
  --batch-size 8 \
  --dataset folder \
  --img-dir ./datasets/test \
  --ddpm-num-steps 500 \
  --ddpm-target-steps 100 \
  --img-out ./datasets/denoised \
  --model-path ./models/model_14.pkl
```

### Output

* Final denoised images saved to `./datasets/denoised`
* Optional Langevin refinement applied automatically

---

## 📊 Evaluation Metrics

Quantitative evaluation is performed using:

* **PSNR** – Peak Signal-to-Noise Ratio
* **SSIM** – Structural Similarity Index
* **LPIPS** – Perceptual similarity
* **FID** – Distribution-level perceptual quality

### Metrics Evaluation Command

```bash
python3 metrics_eval.py \
  --clean_dir ./datasets/clean \
  --denoised_dir ./datasets/denoised \
  --fid
```

---

## 📚 References

* Schmitt et al., *The SEN1‑2 Dataset for Deep Learning in SAR‑Optical Data Fusion*, ISPRS, 2018
* Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021
* Vuong & Nguyen, *Perception-Based Multiplicative Noise Removal*, 2024
* Perera et al., *SAR Despeckling using DDPM*, IEEE GRSL 2023

---

## 👤 Authors

* **Dev M. Bandhiya**
  Department of Mathematics, Mahindra University

* **Dr. Mahipal Jetta**
  Department of Mathematics, Mahindra University

---

## ⭐ Acknowledgements

We acknowledge the **Technical University of Munich (TUM)** for providing the Sentinel‑1 & Sentinel‑2 dataset and the open-source community for diffusion modeling frameworks.
