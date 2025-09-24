import numpy as np
import torch
import torch.nn as nn

# Make CuDNN fast if available
torch.backends.cudnn.benchmark = True

# --- Simple CNN VAE ---
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (B,1,192,224) -> (B,128,24,28)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # 192x224 -> 96x112
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 96x112 -> 48x56
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 48x56 -> 24x28
            nn.ReLU(inplace=True),
            # --- Added layer (no downsample): keeps 24x28 ---
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_out_h, self.enc_out_w, self.enc_out_c = 24, 28, 128
        enc_feat_dim = self.enc_out_c * self.enc_out_h * self.enc_out_w

        self.fc_mu = nn.Linear(enc_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_feat_dim, latent_dim)

        # Decoder: latent -> (B,128,24,28) -> (B,1,192,224)
        self.fc_dec = nn.Linear(latent_dim, enc_feat_dim)
        self.decoder = nn.Sequential(
            # --- Added mirror layer (no upsample): keeps 24x28 ---
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 24x28 -> 48x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 48x56 -> 96x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),   # 96x112 -> 192x224
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)                      # (B,128,24,28)
        h = h.view(h.size(0), -1)                # (B, enc_feat_dim)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.enc_out_c, self.enc_out_h, self.enc_out_w)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
