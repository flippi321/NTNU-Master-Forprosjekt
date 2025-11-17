import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class MiniEncoder3D(nn.Module):
    """
    Minimal 3D encoder that returns multi-scale features (like an encoder of a 3D U-Net).
    Replace this with your pretrained 3D encoder to be fully faithful to MPGAN.
    """
    def __init__(self, in_ch=1, chs=(16, 32, 64, 128)):
        super().__init__()
        self.enc0 = ConvBlock3D(in_ch, chs[0])
        self.enc1 = ConvBlock3D(chs[0], chs[1])
        self.enc2 = ConvBlock3D(chs[1], chs[2])
        self.enc3 = ConvBlock3D(chs[2], chs[3])
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        feats = []
        x0 = self.enc0(x)           # (B, c0, D,H,W)
        feats.append(x0)
        x1 = self.pool(x0); x1 = self.enc1(x1)   # (B, c1, D/2,H/2,W/2)
        feats.append(x1)
        x2 = self.pool(x1); x2 = self.enc2(x2)   # (B, c2, D/4,H/4,W/4)
        feats.append(x2)
        x3 = self.pool(x2); x3 = self.enc3(x3)   # (B, c3, D/8,H/8,W/8)
        feats.append(x3)
        return feats  # list of feature maps at multiple scales

class PhiFeatureExtractor(nn.Module):
    """
    Frozen φ: any 3D encoder that returns a list of feature maps.
    Set requires_grad=False for φ's parameters so grads flow *through* to y_fake, not into φ.
    """
    def __init__(self, base_encoder: nn.Module):
        super().__init__()
        self.base = base_encoder
        self.eval()
        for p in self.base.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def features_real(self, y_real):
        # Safe to avoid grads on the target branch
        return self.base(y_real)

    def features_fake(self, y_fake):
        # No torch.no_grad(): keep graph so gradients flow back into G
        return self.base(y_fake)

def perceptual_loss_3d(y_fake, y_real, phi: PhiFeatureExtractor, layer_weights=None, reduction="l1"):
    """
    Lp = sum_l w_l * || φ_l(y_fake) - φ_l(y_real) ||_1 (default)
    """
    if layer_weights is None:
        layer_weights = None  # equal weight
    feats_fake = phi.features_fake(y_fake)
    feats_real = phi.features_real(y_real)

    if reduction == "l1":
        red = lambda t: t.abs().mean()
    elif reduction == "l2":
        red = lambda t: (t**2).mean()
    else:
        raise ValueError("reduction must be 'l1' or 'l2'")

    loss_p = 0.0
    for i, (ff, fr) in enumerate(zip(feats_fake, feats_real)):
        w = 1.0 if layer_weights is None else layer_weights[i]
        loss_p = loss_p + w * red(ff - fr)
    return loss_p
