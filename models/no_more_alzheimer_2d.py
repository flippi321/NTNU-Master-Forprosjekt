import torch
import torch.nn as nn

# Make CuDNN fast if available
torch.backends.cudnn.benchmark = True

# --- Helper blocks ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # upsample by 2x, halve channels
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch=out_ch * 2, out_ch=out_ch)  # after concat

    def forward(self, x, skip):
        x = self.up(x)
        # In case of odd shapes, center-crop skip to match x
        # TODO CHECK IF CAN REMOVE
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            skip = skip[..., dh//2:skip.size(-2)-(dh - dh//2), dw//2:skip.size(-1)-(dw - dw//2)]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# --- 2D U-Net ---
class UNet2D(nn.Module):
    """
    Input:  (B,1,192,224)
    Output: (B,1,192,224) in [0,1]

    """
    def __init__(self, latent_dim: int = 64, in_channels: int = 1, out_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.latent_dim = latent_dim 

        # Encoder path
        self.enc1 = ConvBlock(in_channels, base_ch)        # -> (B,32,192,224)
        self.pool1 = nn.MaxPool2d(2)                       # -> (B,32,96,112)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)        # -> (B,64,96,112)
        self.pool2 = nn.MaxPool2d(2)                       # -> (B,64,48,56)

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)    # -> (B,128,48,56)
        self.pool3 = nn.MaxPool2d(2)                       # -> (B,128,24,28)

        # Bottleneck (kept at 24x28 to mirror VAE encoder depth)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)  # -> (B,256,24,28)

        # Decoder path
        self.up3 = UpBlock(base_ch * 8, base_ch * 4)       # 24x28 -> 48x56, then concat with enc3 (128)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)       # 48x56 -> 96x112, concat with enc2 (64)
        self.up1 = UpBlock(base_ch * 2, base_ch)           # 96x112 -> 192x224, concat with enc1 (32)

        # Final projection
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()  # Black White output in [0,1]

        # For shape references (2D-Pipelines expected attributes)
        self.enc_out_h, self.enc_out_w, self.enc_out_c = 24, 28, base_ch * 8

        # Internal storage for skips between encode() and decode()
        self._skips = None

    # --- API-compatible methods ---
    def encode(self, x):
        # Down path with skip connections
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        h = self.bottleneck(p3)  # (B,256,24,28) with default base_ch=32

        # Save skips for decode()
        self._skips = (s1, s2, s3)
        
        return h

    def decode(self, z):
        assert self._skips is not None, "decode() called before encode(); call forward(x) or encode(x) first."
        s1, s2, s3 = self._skips

        x = self.up3(z, s3)   # -> (B,128,48,56)
        x = self.up2(x, s2)   # -> (B,64,96,112)
        x = self.up1(x, s1)   # -> (B,32,192,224)

        x = self.out_conv(x)  # -> (B,1,192,224)
        x = self.out_act(x)
        return x

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, None, None # Matches VAE output. Stupid solution but works