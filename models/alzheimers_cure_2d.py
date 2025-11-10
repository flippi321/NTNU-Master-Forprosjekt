import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

# --------- small blocks ----------
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
        )
    def forward(self, x):
        return F.silu(self.block(x) + x, inplace=True)

class DownUnit(nn.Module):
    # Conv (stride=1) -> residual refine -> AvgPool2d(2)
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(8, out_ch)
        self.rb    = ResidualBlock(out_ch)
        self.pool  = nn.AvgPool2d(2)  # anti-aliased downsample
    def forward(self, x):
        x = F.silu(self.gn1(self.conv1(x)), inplace=True)
        x = self.rb(x)
        return self.pool(x)

class UpUnit(nn.Module):
    # ConvTranspose2d upsample -> residual refine
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(8, out_ch)
        self.rb    = ResidualBlock(out_ch)
    def forward(self, x):
        x = F.silu(self.gn1(self.up(x)), inplace=True)
        return self.rb(x)

# --------- CNN VAE ----------
class VAE(nn.Module):
    def __init__(self, latent_dim: int, residual_output: bool = True, residual_scale: float = 0.2):
        super().__init__()
        self.residual_output = residual_output
        self.residual_scale  = residual_scale

        # Encoder: (B,1,192,224) -> (B,128,24,28)
        self.enc_in  = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ResidualBlock(32),
        )
        self.down1   = DownUnit(32, 64)    # 192x224 -> 96x112
        self.down2   = DownUnit(64, 128)   # 96x112  -> 48x56
        self.down3   = DownUnit(128, 128)  # 48x56    -> 24x28
        self.enc_refine = ResidualBlock(128)

        self.enc_out_h, self.enc_out_w, self.enc_out_c = 24, 28, 128
        enc_feat_dim = self.enc_out_c * self.enc_out_h * self.enc_out_w  # 128*24*28 = 86016

        self.fc_mu     = nn.Linear(enc_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_feat_dim, latent_dim)

        # Map latent back to encoder feature space
        self.fc_dec = nn.Linear(latent_dim, enc_feat_dim)

        # Decoder: mirror with residual blocks for detail
        self.dec_refine = ResidualBlock(128)
        self.up1   = UpUnit(128, 64)   # 24x28 -> 48x56
        self.up2   = UpUnit(64, 32)    # 48x56 -> 96x112
        self.up3   = UpUnit(32, 16)    # 96x112 -> 192x224

        # Two heads:
        #   - img_head: direct image
        #   - res_head: residual; if residual_output=True we add it to input for sharper details
        self.img_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # keep [0,1] if inputs are scaled
        )
        self.res_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh(),     # residual in [-1, 1]
        )

    # ---- VAE helpers ----
    def encode(self, x):
        h = self.enc_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.enc_refine(h)                        # (B,128,24,28)
        flat = h.view(h.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return h, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), self.enc_out_c, self.enc_out_h, self.enc_out_w)  # (B,128,24,28)
        h = self.dec_refine(h)
        h = self.up1(h)         # -> (B,64,48,56)
        h = self.up2(h)         # -> (B,32,96,112)
        h = self.up3(h)         # -> (B,16,192,224)
        img = self.img_head(h)  # [0,1]
        res = self.res_head(h) * self.residual_scale
        return img, res

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        img, res = self.decode(z)
        if self.residual_output:
            x_hat = torch.clamp(x + res, 0.0, 1.0)
        else:
            x_hat = img                                
        return x_hat