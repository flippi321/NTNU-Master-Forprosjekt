import torch
import torch.nn as nn

def tv_loss_3d(x):
    """
    Total Variation for 5D tensor (B, C, D, H, W).
    Returns scalar tensor.
    """
    dz = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()
    dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
    dx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
    return dz + dy + dx

def vae_loss(recon, target, mu, logvar):
    # Reconstruction: BCE because inputs are in [0,1]
    bce = nn.functional.binary_cross_entropy(recon, target, reduction='mean')
    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Small KL weight for stability
    return bce + 1e-3 * kld, bce, kld