import torch
from typing import List
import os

def save_model(
    filename: str,
    model: torch.nn.Module,
    data_parallel: bool=False
):
    s = filename.split("/")[:-1]
    f = os.path.join(*s)
    if not os.path.exists(f):
        os.makedirs(f)
    # Be defensive: if model is DataParallel, it has attribute `module`.
    # Allow callers to pass data_parallel flag but also detect DataParallel automatically.
    try:
        if data_parallel:
            if hasattr(model, "module"):
                torch.save(model.module.state_dict(), filename)
            else:
                # flag requested but model not wrapped; save model.state_dict()
                torch.save(model.state_dict(), filename)
        else:
            # not requested; if model is DataParallel save underlying module for convenience
            if hasattr(model, "module"):
                torch.save(model.module.state_dict(), filename)
            else:
                torch.save(model.state_dict(), filename)
    except Exception:
        # fallback: try to save whatever state_dict is available
        try:
            torch.save(model.state_dict(), filename)
        except Exception:
            # last resort: save whole model object
            torch.save(model, filename)


def load_model(filename: str, model: torch.nn.Module) -> torch.nn.Module:
    model = model.load_state_dict(torch.load(filename))
    model = model.eval()
    return model


def compute_image_statistics(img: torch.Tensor) -> torch.Tensor:
    """
    Compute a small conditioning vector from an image batch.

    img: (B, C, H, W) - float tensor. Expected to be in a linear intensity scale
         (e.g., values in [0,1]) or any consistent scale from training.

    Returns: (B, 4) tensor with [mean_gray, var_gray, downsample_diff, laplacian_energy]

    Notes: if images are in log-domain in your preprocessing, call this with
    torch.exp(img) - 1.0 to convert to linear domain first.
    """
    import torch.nn.functional as F

    B, C, H, W = img.shape
    device = img.device

    # grayscale approximation
    gray = img.mean(dim=1)  # (B, H, W)

    # 1) mean gray
    mean_gray = gray.mean(dim=[1, 2])  # (B,)

    # 2) global variance (population)
    var_gray = gray.var(dim=[1, 2], unbiased=False)  # (B,)

    # 3) downsample difference -> proxy for fine detail
    down = F.interpolate(gray.unsqueeze(1), scale_factor=0.5, mode="area").squeeze(1)
    up = F.interpolate(down.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
    downsample_diff = (gray - up).abs().mean(dim=[1, 2])  # (B,)

    # 4) laplacian energy
    laplace_kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=device,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray.unsqueeze(1), laplace_kernel, padding=1).squeeze(1)
    lap_energy = (lap ** 2).mean(dim=[1, 2])

    cond = torch.stack([mean_gray, var_gray, downsample_diff, lap_energy], dim=1)
    return cond
