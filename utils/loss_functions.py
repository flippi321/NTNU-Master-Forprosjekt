import torch
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# ---------------------------------------------
# 2D Loss Functions
# ---------------------------------------------
# These take in single slices of truth and
# model reconstructions and compare them
# ---------------------------------------------

def binary_2d_loss(recon, target, mu, logvar):
    # Reconstruction: BCE because inputs are in [0,1]
    bce = nn.functional.binary_cross_entropy(recon, target, reduction='mean')
    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Small KL weight for stability
    return bce + 1e-3 * kld


def ssim_L1_2d_loss(recon, target, mu, logvar):
    # SSIM loss
    ssim_loss = 1 - ssim(recon, target, data_range=1.0, size_average=True)
    # L1 loss
    l1_loss = nn.functional.l1_loss(recon, target, reduction='mean')
    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine losses with weights
    return ssim_loss + l1_loss + 1e-3 * kld

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

