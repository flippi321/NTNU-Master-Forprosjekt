import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# ----------------------------
# Residual 3D U-Net components
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # depthwise downsample (cheap) + conv block
        self.pool = nn.Conv3d(in_ch, in_ch, kernel_size=2, stride=2, groups=in_ch)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.reduce = nn.Conv3d(in_ch, out_ch, 1)
        self.conv = ConvBlock(out_ch * 2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        # pad to match (handles odd shapes)
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        x = F.pad(x, [0, dx, 0, dy, 0, dz])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResidualUNet3D(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = Down(base, base*2)
        self.e3 = Down(base*2, base*4)
        self.e4 = Down(base*4, base*8)
        self.bott = ConvBlock(base*8, base*16)
        self.bott_sec = ConvBlock(base*16, base*16)

        self.u4 = Up(base*16, base*8)
        self.u3 = Up(base*8, base*4)
        self.u2 = Up(base*4, base*2)
        self.u1 = Up(base*2, base)
        self.out = nn.Conv3d(base, 1, 1)  # residual Δ

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b  = self.bott_sec(self.bott(s4))
        d4 = self.u4(b, s4)
        d3 = self.u3(d4, s3)
        d2 = self.u2(d3, s2)
        d1 = self.u1(d2, s1)
        delta = self.out(d1)
        return x + delta, delta  # predicted follow-up and residual