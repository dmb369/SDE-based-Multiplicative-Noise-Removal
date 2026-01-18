# this script loads denoising models
# and perform noise removal on a folder
# RGB VERSION (GRAYSCALE ENFORCING REMOVED)

import os
import torch
import tqdm
import argparse
from torch.utils.data import DataLoader
from datasets_proc import data_transform, FolderDataset
from train_denoise_sde import (
    get_noise_schedule,
    save_image_2,
    one_step_denoising_ddim_deterministic,
)
from unet_ref import UNet


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def tied_noise_like(x):
    """Generate noise tied across RGB channels (kept unchanged)"""
    z = torch.randn(
        x.shape[0],
        1,
        x.shape[2],
        x.shape[3],
        device=x.device
    )
    return z.repeat(1, 3, 1, 1)


# --------------------------------------------------
# One-step denoising (stochastic)
# --------------------------------------------------

@torch.no_grad()
def one_step_denoising(
    model,
    hmodel,
    log_x,
    t,
    sigmas,
    device="cpu",
):
    scale_t = torch.gather(sigmas, 1, t + 1)[:, :, None, None]
    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))[:, :, None, None]
    scale_t_prev = torch.gather(sigmas, 1, t)[:, :, None, None]

    # tied noise (unchanged)
    z = tied_noise_like(log_x)

    noise_pred = model(log_x, t[0] + 1)

    var = scale_t - scale_0
    var_t = scale_t - scale_t_prev
    mean = log_x + 0.5 * var_t

    if hmodel is None:
        log_x_out = (
            mean
            - var_t / torch.sqrt(var) * noise_pred
            + torch.sqrt(var_t) * z
        )
    else:
        h_transform_pred = hmodel(log_x, t[0] + 1)
        log_x_out = (
            mean
            - var_t / torch.sqrt(var) * (noise_pred + h_transform_pred)
            + torch.sqrt(var_t) * z
        )

    return log_x_out.to(device)


# --------------------------------------------------
# One-step denoising (deterministic)
# --------------------------------------------------

@torch.no_grad()
def one_step_denoising_deterministic(
    model,
    hmodel,
    log_x,
    t,
    sigmas,
    device="cpu",
):
    scale_t = torch.gather(sigmas, 1, t + 1)[:, :, None, None]
    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))[:, :, None, None]
    scale_t_prev = torch.gather(sigmas, 1, t)[:, :, None, None]

    noise_pred = model(log_x, t[0] + 1)

    var = scale_t - scale_0
    var_t = scale_t - scale_t_prev
    mean = log_x + 0.5 * var_t

    if hmodel is None:
        log_x_out = mean - 0.5 * var_t / torch.sqrt(var) * noise_pred
    else:
        h_transform_pred = hmodel(log_x, t[0] + 1)
        log_x_out = mean - 0.5 * var_t / torch.sqrt(var) * (
            noise_pred + h_transform_pred
        )

    return log_x_out.to(device)


# --------------------------------------------------
# Sampling loop
# --------------------------------------------------

@torch.no_grad()
def sample_noise_removal(
    x_0,
    model,
    hmodel,
    timesteps,
    sigmas,
    device="cpu",
    return_history=False,
    num_samples=1,
    smooth_indices=None,
    deterministic=False,
    ddim=False,
):
    x = x_0
    hist = [x]

    for idx, t_ in tqdm.tqdm(
        enumerate(reversed(range(timesteps - 1))),
        total=timesteps - 1,
    ):
        t = torch.ones((x.shape[0], 1), dtype=int, device=device) * t_

        if smooth_indices is not None and idx in smooth_indices:
            xs = []
            for _ in range(num_samples):
                if not deterministic:
                    xs.append(one_step_denoising(model, hmodel, x, t, sigmas, device))
                else:
                    xs.append(one_step_denoising_deterministic(model, hmodel, x, t, sigmas, device))
            x = torch.mean(torch.stack(xs), dim=0)
        else:
            if not deterministic:
                x = one_step_denoising(model, hmodel, x, t, sigmas, device)
            else:
                if ddim:
                    x = one_step_denoising_ddim_deterministic(
                        model, x, t, sigmas, device
                    )
                else:
                    x = one_step_denoising_deterministic(
                        model, hmodel, x, t, sigmas, device
                    )

        hist.append(x)

    if return_history:
        return x, hist
    return x


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Noise removal by diffusion")

    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--img-out", type=str, required=True)
    parser.add_argument("--cpu-workers", type=int, default=5)
    parser.add_argument("--ddpm-num-steps", type=int, default=500)
    parser.add_argument("--ddpm-target-steps", type=int, required=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--ddim", action="store_true")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    # Langevin
    parser.add_argument("--langevin-steps", type=int, default=0)
    parser.add_argument("--langevin-step-size", type=float, default=1e-3)
    parser.add_argument("--langevin-noise-scale", type=float, default=0.01)

    args = parser.parse_args()
    device = args.device

    # Load model (RGB UNet)
    unet_conf = dict(
        init_channels=128,
        in_channels=3,
        out_channels=3,
        num_res_blocks=2,
        attn_resolutions=(16,),
        input_img_resolution=224,
        channels_multipliers=(1, 1, 2, 2, 4, 4),
    )
    model = UNet(**unet_conf).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()

    outdir = (
        args.img_out
        + f"_t{args.ddpm_target_steps}"
        + f"_ddim_{args.ddim}"
        + f"_det_{args.deterministic}"
    )
    os.makedirs(outdir, exist_ok=True)

    transform = data_transform(args.img_size, not args.no_rgb)
    dataset = FolderDataset(args.img_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.cpu_workers,
    )

    sigmas_base = get_noise_schedule(args.ddpm_num_steps)

    for idx, log_x in tqdm.tqdm(enumerate(loader), total=len(loader)):
        log_x = log_x[0].to(device)

        sigmas = sigmas_base.unsqueeze(0).repeat(log_x.shape[0], 1).to(device)

        log_x_denoised, _ = sample_noise_removal(
            log_x,
            model,
            None,
            args.ddpm_target_steps,
            sigmas,
            device=device,
            return_history=True,
            deterministic=args.deterministic,
            ddim=args.ddim,
        )

        # Langevin refinement (unchanged, no grayscale enforcement)
        if args.langevin_steps > 0:
            x_ref = log_x_denoised
            x_obs = log_x
            for _ in range(args.langevin_steps):
                delta = x_obs - x_ref
                noise = tied_noise_like(x_ref) * args.langevin_noise_scale
                x_ref = x_ref + args.langevin_step_size * delta + noise
            log_x_denoised = x_ref

        # Save
        for j, im in enumerate(log_x_denoised):
            save_image_2(
                im,
                os.path.join(outdir, f"{idx * args.batch_size + j}.png")
            )
