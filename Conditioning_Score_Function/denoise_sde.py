# this script loads denoising models
# and perform noise removal on a folder
# containing images

import os
import torch
import tqdm
import argparse
from torch.utils.data import DataLoader
from datasets_proc import data_transform, FolderDataset
from train_denoise_sde import get_noise_schedule, save_image_2, one_step_denoising_ddim_deterministic
from model_helpers import compute_image_statistics
from unet_ref import UNet

@torch.no_grad()
def one_step_denoising(
    model,
    hmodel,
    log_x,
    t,
    sigmas,
    device="cpu",
    cond=None,
):
    scale_t = torch.gather(sigmas, 1, t + 1)
    scale_t = scale_t[:, :, None, None]

    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))
    scale_0 = scale_0[:, :, None, None]

    scale_t_prev = torch.gather(sigmas, 1, t)
    scale_t_prev = scale_t_prev[:, :, None, None]

    z = torch.randn_like(log_x)

    noise_pred = model(log_x, t[0] + 1, cond)
    # noise_pred = torch.clamp(noise_pred, -1, 1)

    var = scale_t - scale_0
    var_t = scale_t - scale_t_prev
    mean = log_x + 0.5 * var_t
    # mean = log_x
    
    if hmodel is None:
        log_x_out = mean - var_t / torch.sqrt(var) * noise_pred + torch.sqrt(var_t) * z
    else:
        h_transform_pred = hmodel(log_x, t[0] + 1, cond)
        log_x_out = (
                mean
                - var_t / torch.sqrt(var) * (noise_pred + h_transform_pred)
                + torch.sqrt(var_t) * z
            )
    
    return log_x_out.to(device)


@torch.no_grad()
def one_step_denoising_deterministic(
    model,
    hmodel,
    log_x,
    t,
    sigmas,
    device="cpu",
    cond=None,
):
    scale_t = torch.gather(sigmas, 1, t + 1)
    scale_t = scale_t[:, :, None, None]

    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))
    scale_0 = scale_0[:, :, None, None]

    scale_t_prev = torch.gather(sigmas, 1, t)
    scale_t_prev = scale_t_prev[:, :, None, None]


    noise_pred = model(log_x, t[0] + 1, cond)
    # noise_pred = torch.clamp(noise_pred, -1, 1)

    var = scale_t - scale_0
    var_t = scale_t - scale_t_prev
    mean = log_x + 0.5 * var_t
    # mean = log_x
    
    if hmodel is None:
        log_x_out = mean - 0.5 * var_t / torch.sqrt(var) * noise_pred 
    else:
        h_transform_pred = hmodel(log_x, t[0] + 1, cond)
        log_x_out = mean - 0.5 * var_t / torch.sqrt(var) * (noise_pred + h_transform_pred)

    return log_x_out.to(device)

@torch.no_grad()
def sample_noise_removal(
    x_0: torch.Tensor,
    model: UNet,
    hmodel: UNet,
    timesteps: int,
    sigmas: torch.Tensor,
    device: str = "cpu",
    return_history = False,
    num_samples = 1,
    smooth_indices = None,
    deterministic = False,
    ddim = False,
    cond: torch.Tensor = None,
):
    x = x_0
    hist = [x]
    for idx, t_ in tqdm.tqdm(enumerate(reversed(range(timesteps - 1))), total=timesteps - 1):
        t = torch.ones(size=(x.shape[0], 1), dtype=int) * t_
        t = t.to(device)
        x_list = []
        if smooth_indices is not None and idx in smooth_indices:
            for i in range(num_samples):
                if not deterministic:
                    x_ = one_step_denoising(model, hmodel, x, t, sigmas, device=device, cond=cond)
                else:
                    x_ = one_step_denoising_deterministic(model, hmodel, x, t, sigmas, device=device)
                x_list.append(x_)
                # if idx > timesteps-20 or (idx + 1) % 10 == 0:
                #     hist.append(x)
            x = torch.mean(torch.stack(x_list), 0)
        else:
            if not deterministic:
                x = one_step_denoising(model, hmodel, x, t, sigmas, device=device, cond=cond)
            else:
                if ddim:
                    # no hmodel for ddim
                    x = one_step_denoising_ddim_deterministic(model, x, t, sigmas, device=device)
                else:
                    x = one_step_denoising_deterministic(model, hmodel, x, t, sigmas, device=device, cond=cond)
        hist.append(x)
    # x = torch.clamp(x, 0, np.log(2))
    if return_history:
        return x, hist
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Noise removal by diffusion", description="Noise removal by diffusion"
    )
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--dataset", type=str, default="celeba", required=True)
    parser.add_argument(
        "--img-dir", type=str, default="./datasets/img_celeba", required=True
    )
    parser.add_argument(
        "--img-out", type=str, required=True
    )
    parser.add_argument("--cpu-workers", type=int, default=5)
    parser.add_argument("--ddpm-num-steps", type=int, default=500)
    parser.add_argument("--ddpm-target-steps", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu", help="device to run on (cpu, mps, cuda:0 etc)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--ddim", action="store_true")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--langevin-steps", type=int, default=0,
                        help="number of Langevin refinement steps to run after denoising (default 0)")
    parser.add_argument("--langevin-step-size", type=float, default=1e-3,
                        help="step size used to nudge the output toward the observed speckled image")
    parser.add_argument("--langevin-noise-scale", type=float, default=0.01,
                        help="scale for the small random noise added during Langevin refinement")


    args = parser.parse_args()

    device = args.device

    # Init diffusion params
    batch_size = args.batch_size
    T = args.ddpm_num_steps

    # Load denoising dmodel
    unet_conf = {
        "init_channels": 128,
        "in_channels": 1,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
        "input_img_resolution": 224,
        "channels_multipliers": (1, 1, 2, 2, 4, 4),
    }
    dmodel = UNet(**unet_conf).to(device)

    total_params = sum([p.numel() for p in dmodel.parameters()])
    print("Diffusion model initialized, total params = ", total_params)

    model_path = args.model_path
    print(f"loading model from epoch {model_path}")
    dmodel.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )
    )

    # init output folder
    outdir = args.img_out + f"_t{args.ddpm_target_steps}" + f"_ddim_{args.ddim}" + f"_deterministic_{args.deterministic}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # init dataset
    img_size = args.img_size
    batch_size = args.batch_size
    rgb = True
    if args.no_rgb:
        rgb = False
    data_transform = data_transform(img_size=img_size, rgb=rgb)
    img_dir = args.img_dir

    data = FolderDataset(
        img_dir=img_dir, transform=data_transform
    )

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.cpu_workers,
        drop_last=False,
    )

    total = len(loader)

    print(f"Generating denoised images from {args.img_dir}")
    print(f"Dataset={args.dataset}, DDPM T={args.ddpm_num_steps}, t={args.ddpm_target_steps}")
    print(f"Images will be saved to {outdir}")

    pbar = tqdm.tqdm(enumerate(loader), total=total)
    for idx, log_x in pbar:
        log_x = log_x[0]
        log_x = log_x.to(device)

        sigmas = get_noise_schedule(T)
        # repeat into batch_size for easier indexing by t
        sigmas = sigmas.unsqueeze(0).repeat(log_x.shape[0], 1).to(device)

        # denoising
        h_transform_model = None # don't use hmodel

        # compute conditioning vector from the observed (speckled) image;
        # log-domain -> linear domain before computing stats
        cond = compute_image_statistics(torch.exp(log_x) - 1.0)

        log_x_denoised, hist_log_x  = sample_noise_removal(
                    log_x,
                    dmodel,
                    h_transform_model,
                    args.ddpm_target_steps,
                    sigmas,
                    device=device,
                    return_history=True,
                    num_samples=2,
                    smooth_indices=None,
                    deterministic=args.deterministic,
                    ddim = args.ddim,
                    cond=cond,
                )

        # Langevin refinement (small number of steps pulling the denoised image
        # toward the observed speckled image while adding tiny random noise).
        # Operates in log-domain (same space as model outputs / inputs).
        if args.langevin_steps and args.langevin_steps > 0:
            steps = args.langevin_steps
            step_size = float(args.langevin_step_size)
            noise_scale = float(args.langevin_noise_scale)
            # start from last denoised prediction
            x_ref = log_x_denoised
            # observed input (log domain)
            x_obs = log_x
            for _ in range(steps):
                # push toward observed and add small noise
                delta = x_obs - x_ref
                noise = torch.randn_like(x_ref) * (noise_scale)
                x_ref = x_ref + step_size * delta + noise
            log_x_denoised = x_ref

        for idx_, im in enumerate(log_x_denoised):
            num = idx * batch_size + idx_
            path = os.path.join(outdir, f"{num}.png")
            save_image_2(im, path)