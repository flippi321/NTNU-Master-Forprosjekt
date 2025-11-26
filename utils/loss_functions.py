import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import ssim
import torch.nn.functional as F

# ---------------------------------------------
# 2D Loss Functions
# ---------------------------------------------
# These take in single slices of truth and
# model reconstructions and compare them
# ---------------------------------------------

def binary_2d_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Reconstruction: BCE because inputs are in [0,1]
    bce = nn.functional.binary_cross_entropy(recon, target, reduction='mean')
    # KL divergence
    return bce

def ssim_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # SSIM loss
    return 1 - ssim(recon, target, data_range=1.0, size_average=True)

def l1_loss(recon: torch.Tensor, target: torch.Tensor, l1_weight=0.5, ssim_weight=0.5) -> torch.Tensor:
    # L1 loss
    return nn.functional.l1_loss(recon, target, reduction='mean')

def ssim_L1_2d_loss(recon: torch.Tensor, target: torch.Tensor, l1_weight=0.5, ssim_weight=0.5) -> torch.Tensor:
    # Return losses with weights
    return ssim_weight * ssim_loss(recon, target) + l1_weight * l1_loss(recon, target)

def ssim_L1_kl_loss(recon: torch.Tensor, target: torch.Tensor, mu, logvar, l1_weight=0.5, ssim_weight=0.5, kl_weight=1e-3) -> torch.Tensor:
    ssim_loss = 1 - ssim(recon, target, data_range=1.0, size_average=True)
    l1_loss   = F.l1_loss(recon, target, reduction='mean')
    
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return ssim_weight * ssim_loss + l1_weight * l1_loss + kl_weight * kl


# ---------------------------------------------
# 3D Loss Functions
# ---------------------------------------------
# These take in full volumes of truth and
# model reconstructions and compare them
# ---------------------------------------------

def tv_loss_3d(x):
    """
    Total Variation for 5D tensor (B, C, D, H, W).
    Returns scalar tensor.
    """
    dz = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()
    dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
    dx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
    return dz + dy + dx

# SSIM loss adapted for 3D volumes
def _create_3d_window(window_size: int, channel: int, device=None, dtype=None):
    """
    Create a 3D Gaussian-like window for SSIM.
    window_size: odd int, e.g. 11
    """
    # 1D Gaussian
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / float(2.0 * (window_size/6.0)**2))
    g = g / g.sum()

    # 3D separable kernel = g(z)*g(y)*g(x)
    g3 = g[:, None, None] * g[None, :, None] * g[None, None, :]
    g3 = g3 / g3.sum()  # normalize just to be sure

    # shape: (1,1,D,H,W) so we can use conv3d with groups=channel
    window = g3.view(1, 1, window_size, window_size, window_size)
    window = window.repeat(channel, 1, 1, 1, 1)  # (C,1,ks,ks,ks)
    return window

def ssim_3d(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Compute mean SSIM over batch+channel for 3D volumes.
    pred, target: (B, C, D, H, W) in [0,1]
    Returns: scalar tensor, higher is better (1 = perfect)
    """
    # Ensure same type/device/shape
    assert pred.shape == target.shape, "pred and target must have same shape"
    B, C, D, H, W = pred.shape

    # Create / cache window
    window = _create_3d_window(window_size, C, device=pred.device, dtype=pred.dtype)

    # conv3d with groups=C applies channel-wise local stats
    padding = window_size // 2

    mu_x = F.conv3d(pred, window, groups=C, padding=padding)
    mu_y = F.conv3d(target, window, groups=C, padding=padding)

    mu_x2  = mu_x * mu_x
    mu_y2  = mu_y * mu_y
    mu_xy  = mu_x * mu_y

    sigma_x2 = F.conv3d(pred * pred, window, groups=C, padding=padding) - mu_x2
    sigma_y2 = F.conv3d(target * target, window, groups=C, padding=padding) - mu_y2
    sigma_xy = F.conv3d(pred * target, window, groups=C, padding=padding) - mu_xy

    # SSIM map
    num   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denom = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (denom + 1e-8)

    # Average over spatial dims, then batch+channel
    return ssim_map.mean()

def ssim_loss_3d(pred, target, window_size=11):
    ssim_val = ssim_3d(pred, target, window_size=window_size)
    return 1.0 - ssim_val

def recon_loss(pred, target,
               w_l1=0.5,
               w_ssim=0.5,
               window_size=11):
    """
    pred, target: (B, C, D, H, W) in [0,1]
    Returns scalar loss
    """
    l1 = torch.mean(torch.abs(pred - target))
    ssim_l = ssim_loss_3d(pred, target, window_size=window_size)
    return w_l1 * l1 + w_ssim * ssim_l


