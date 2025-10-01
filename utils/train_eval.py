import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.hunt_data_loader import HuntDataLoader
from utils.loss_functions import tv_loss_3d
import random

def build_optimizer(model, lr=1e-4, wd=1e-4):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

def fit_2D(
        model,    
        device: torch.device,
        dataLoader: HuntDataLoader,
        training_pairs: list[tuple[str, str]],
        criterion, 
        epochs, 
        optimizer=None, 
        saved_snapshots=None, 
        print_every=-1, 
        save_every=-1,
        ):
    optimizer = optimizer or build_optimizer(model)
    saved_snapshots = saved_snapshots or []

    for epoch in range(epochs):
        model.train()

        # We load a random client data-pair
        client = random.randint(0, len(training_pairs)-1)
        xs = dataLoader.get_all_slices_as_tensor(training_pairs[client][0], crop_size=(192,224))[10:-10]  # (N, 192, 224)
        ys = dataLoader.get_all_slices_as_tensor(training_pairs[client][1], crop_size=(192,224))[10:-10]  # (N, 192, 224)

        num = min(len(xs), len(ys)) # They should be equal, but just in case
        running_loss = running_bce = running_kld = 0.0

        # Iterate over each slice
        for idx in range(num):
            x_slice = xs[idx]
            y_slice = ys[idx]

            x = dataLoader.to_torch_img(x_slice, device)   # (1,1,192,224)
            y = dataLoader.to_torch_img(y_slice, device)   # (1,1,192,224)

            optimizer.zero_grad()
            recon, mu, logvar = model(x)

            loss, bce, kld = criterion(recon, y, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_bce  += float(bce.item())
            running_kld  += float(kld.item())

        # --- Every Xth pair, save a snapshot of reconstruction vs target ---
        if (epoch % save_every == 0) and num > 0:
            # pick a safe index to visualize
            idx_to_show = 90 if num > 90 else (num // 2)
            with torch.no_grad():
                x_show = dataLoader.to_torch_img(xs[idx_to_show], device)
                y_show = dataLoader.to_torch_img(ys[idx_to_show], device)
                recon_show, _, _ = model(x_show)

                # convert to numpy for visualization
                x_np     = dataLoader.to_numpy_img(x_show)
                y_np     = dataLoader.to_numpy_img(y_show)      
                recon_np = dataLoader.to_numpy_img(recon_show) 

            saved_snapshots.append({"iter": epoch, "x": x_np, "y": y_np, "recon": recon_np})
            print(f"Saved snapshot for pair {epoch} at slice idx {idx_to_show}")
    return model, saved_snapshots

def fit_3D(
    model,
    device: torch.device,
    dataLoader: HuntDataLoader,
    training_pairs: list[tuple[str, str]],
    criterion=None,
    epochs=1,
    optimizer=None,
    saved_snapshots=None,
    print_every=-1,
    save_every=-1,
    trim_slices=0,
    crop_size=(192, 224),
    lambda_tv=1e-5,
):
    """
    Train a 3D model on full volumes using HuntDataLoader.
    - Expects model(x) -> either:
        * y_hat
        * (y_hat, delta)   where delta is a residual volume used for TV regularization
    - criterion is optional:
        * If provided and accepts (y_hat, y), we use it.
        * Otherwise we fall back to L1.
    - Snapshots show the mid-axial slice (H,W) for input, target, and recon.
    """
    
    optimizer = optimizer or build_optimizer(model)
    saved_snapshots = saved_snapshots or []

    for i in range(epochs):
        model.train()

        # pick a random pair and load the FULL volume as a stack of slices
        client = random.randint(0, len(training_pairs) - 1)
        x_path, y_path = training_pairs[client][0], training_pairs[client][1]

        xs_list = dataLoader.get_all_slices_as_tensor(x_path, crop_size=crop_size)  # list of (H,W) tensors
        ys_list = dataLoader.get_all_slices_as_tensor(y_path, crop_size=crop_size)

        # optional trimming at both ends along D
        if trim_slices and trim_slices > 0:
            xs_list = xs_list[trim_slices:-trim_slices]
            ys_list = ys_list[trim_slices:-trim_slices]

        # ensure equal depth
        D = min(len(xs_list), len(ys_list))
        xs_list = xs_list[:D]
        ys_list = ys_list[:D]

        # (D,H,W) -> (1,1,D,H,W)
        x = to_torch_vol(xs_list, device)
        y = to_torch_vol(ys_list, device)

        optimizer.zero_grad()

        # forward (support both y_hat or (y_hat, delta))
        out = model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            y_hat, delta = out[0], out[1]
        else:
            y_hat, delta = out, None

        # --- loss ---
        used_custom_criterion = False
        loss = None

        if criterion is not None:
            # Try criterion(y_hat, y) returning either a scalar or (loss, *extras)
            try:
                crit_out = criterion(y_hat, y)
                if isinstance(crit_out, (tuple, list)):
                    loss = crit_out[0]
                else:
                    loss = crit_out
                used_custom_criterion = True
            except TypeError:
                # The provided criterion didn't match (y_hat,y); we'll fall back to L1
                pass

        if loss is None:
            # fallback: L1
            loss = F.l1_loss(y_hat, y)

        # add small TV on residual if available
        if delta is not None and lambda_tv is not None and lambda_tv > 0:
            loss = loss + lambda_tv * tv_loss_3d(delta)

        loss.backward()
        optimizer.step()

        if (print_every > 0) and (i % print_every == 0):
            if used_custom_criterion:
                print(f"[Iter {i}] total: {loss.item():.6f} (custom criterion)"
                      + (f" + TV" if delta is not None and lambda_tv > 0 else ""))
            else:
                base = F.l1_loss(y_hat.detach(), y).item()
                tvv = tv_loss_3d(delta).item() if (delta is not None and lambda_tv > 0) else 0.0
                print(f"[Iter {i}] total: {loss.item():.6f} | L1: {base:.6f} | TVΔ: {tvv:.6f}")

        # --- snapshot ---
        if (i % save_every == 0):
            with torch.no_grad():
                x_np = mid_axial_slice_5d(x)
                y_np = mid_axial_slice_5d(y)
                recon_np = mid_axial_slice_5d(y_hat)

            saved_snapshots.append({"iter": i, "x": x_np, "xy": y_np, "recon": recon_np})
            print(f"Saved snapshot at iter {i} (mid-axial slice)")

    return model, saved_snapshots

def to_torch_vol(vol_DHW, device):
    """
    vol_DHW: torch tensor or numpy array with shape (D, H, W) or list of (H, W) slices
    -> returns (1, 1, D, H, W) float32 on device
    """
    if isinstance(vol_DHW, list):
        vol = torch.stack([v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
                           for v in vol_DHW], dim=0)
    elif isinstance(vol_DHW, np.ndarray):
        vol = torch.from_numpy(vol_DHW)
    else:
        vol = vol_DHW
    vol = vol.float().clamp(0, 1)  # keep in [0,1] like your 2D path
    vol = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    return vol.to(device)

def mid_axial_slice_5d(t5):
    """
    t5: (B, C, D, H, W) torch tensor
    -> numpy slice (H, W) in [0,1] from the middle of D
    """
    if isinstance(t5, torch.Tensor):
        t = t5.detach().cpu()
    else:
        t = torch.tensor(t5)
    _, _, D, H, W = t.shape
    mid = D // 2
    sl = t[0, 0, mid]  # (H, W)
    sl = sl.clamp(0, 1).numpy()
    return sl