import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.hunt_data_loader import HuntDataLoader
from utils.loss_functions import tv_loss_3d
from tqdm import tqdm

def build_optimizer(model, lr=1e-4, wd=1e-4):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def load_model_weights(model, model_path, device, verbose=False):
    """
    Loads model weights from the specified path into the given model.
    """
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        if verbose: print(f"Loaded model weights from {model_path}")
    else:
        if verbose: print(f"No model weights found at {model_path}. Starting with random weights.")

def save_model_weights(model, model_path, verbose=False):
    """
    Saves model weights from the given model to the specified path.
    """
    torch.save(model.state_dict(), model_path)
    if verbose: print(f"Saved model weights to {model_path}")

def fit_2d_models_per_slice(
        model_constructor,
        optimizer_constructor,
        device: torch.device,
        dataLoader,
        training_pairs: list[tuple[str, str]],
        criterion,
        epochs: int = 1000,
        save_every: int = -1,
        total_slices:int = 193,
        crop_size: tuple = (192, 224),
        idx_to_show: int = 93,  # Approx. middle slice
        model_dir: str = "",    # Assume load/save if defined
        model_name_prefix: str = "",
        batch_size: int = 49,
    ):
    """
    Train per-slice models in batches of size `batch_size`.
    Each model in the batch trains on the SAME client image per step,
    but on its own fixed slice index.
    """
    loss_histories = []
    snapshots = []

    # iterate over slice indices in chunks
    for start in tqdm(range(0, total_slices, batch_size), desc="Training Slices"):
        end = min(start + batch_size, total_slices)
        slice_indices = list(range(start, end))

        # build models/optimizers for this batch
        model_batch = [model_constructor().to(device) for _ in slice_indices]
        optimizer_batch = [optimizer_constructor(m) for m in model_batch]
        model_names = [f"{model_name_prefix}_slice_{s}.pt" for s in slice_indices]

        loss_history_i, saved_snapshots_i = fit_batch_on_slices(
            model_batch=model_batch,
            optimizer_batch=optimizer_batch,
            device=device,
            slice_indices=slice_indices,
            dataLoader=dataLoader,
            training_pairs=training_pairs,
            criterion=criterion,
            epochs=epochs,
            save_every=save_every,
            crop_size=crop_size,
            idx_to_show=idx_to_show,
            model_dir=model_dir,
            model_names=model_names,
        )

        loss_histories.append(np.array(loss_history_i, dtype=float))
        snapshots.extend(saved_snapshots_i)

        # cleanup
        del model_batch, optimizer_batch
        torch.cuda.empty_cache()

    # average the per-batch loss curves (pad with NaNs then nanmean)
    max_len = max(len(h) for h in loss_histories)
    padded = [np.pad(h, (0, max_len - len(h)), constant_values=np.nan) for h in loss_histories]
    loss_curve = np.nanmean(np.vstack(padded), axis=0)

    snapshots = sorted(snapshots, key=lambda x: (x.get("iter", 0), x.get("slice_idx", -1)))
    return loss_curve, snapshots

def fit_batch_on_slices(
        model_batch: list[nn.Module],
        optimizer_batch: list[optim.Optimizer],
        device: torch.device,
        slice_indices: list[int],
        dataLoader,                       
        training_pairs: list[tuple[str, str]],
        criterion,
        epochs: int,  
        save_every: int,
        crop_size: tuple,
        idx_to_show: int,  # Approx. middle slice
        model_dir: str,    # Assume load/save if defined
        model_names: list[str],
    ):
    """
    Trains a list of per-slice models. Model i is trained on slice i of each client volume.
    """
    assert len(model_batch) == len(optimizer_batch) == len(slice_indices), "Batch lists must align"
    if model_names is None:
        model_names = ["" for _ in model_batch]

    # load weights for all models if available
    if model_dir:
        for m, name in zip(model_batch, model_names):
            if name:
                load_model_weights(m, os.path.join(model_dir, name), device)

    loss_history = []
    snapshots = []

    num_clients = len(training_pairs)

    # compact per-batch epoch progress bar (keeps outer bar intact)
    for epoch in tqdm(range(epochs), desc=f"Epochs {slice_indices[0]}–{slice_indices[-1]}", leave=False):
        # pick one client; all models see the same image stacks this step
        client = random.randint(0, num_clients - 1)
        xs = dataLoader.get_all_slices_as_tensor(training_pairs[client][0], crop_size=crop_size)
        ys = dataLoader.get_all_slices_as_tensor(training_pairs[client][1], crop_size=crop_size)

        epoch_losses = []

        for s, m, opt in zip(slice_indices, model_batch, optimizer_batch):
            if s >= min(len(xs), len(ys)):
                continue

            m.train()
            x = dataLoader.to_torch_img(xs[s], device)  # (1,1,H,W)
            y = dataLoader.to_torch_img(ys[s], device)  # (1,1,H,W)

            opt.zero_grad()
            recon, mu, logvar = m(x)
            loss = criterion(recon, y, mu, logvar)
            loss.backward()
            opt.step()

            epoch_losses.append(float(loss.item()))

            # snapshot only if this model's slice is the one of interest
            if (save_every > 0) and (epoch % save_every == 0) and (s == idx_to_show):
                with torch.no_grad():
                    x_show = dataLoader.to_torch_img(xs[idx_to_show], device)
                    y_show = dataLoader.to_torch_img(ys[idx_to_show], device)
                    recon_show, _, _ = m(x_show)

                    x_np     = dataLoader.to_numpy_img(x_show)
                    y_np     = dataLoader.to_numpy_img(y_show)
                    recon_np = dataLoader.to_numpy_img(recon_show)

                snapshots.append({
                    "iter": epoch,
                    "slice_idx": s,
                    "x": x_np,
                    "y": y_np,
                    "recon": recon_np
                })

        if epoch_losses:
            loss_history.append(sum(epoch_losses) / len(epoch_losses))

    # save weights individually for all models in the batch
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        for m, name in zip(model_batch, model_names):
            if name:
                save_model_weights(m, os.path.join(model_dir, name))

    return loss_history, snapshots


def fit_2D(
        model: nn.Module,
        optimizer: optim.Optimizer,  
        device: torch.device,
        dataLoader: HuntDataLoader,
        training_pairs: list[tuple[str, str]],
        criterion,
        epochs: int, 
        save_every: int = -1,
        crop_size: tuple = (192, 224),
        idx_to_show: int = 93,
        model_dir: str = "",
        model_name: str = ""
        ):
    """
    Trains a model to reconstruct 2D slices from client volumes.
    """

    optimizer = optimizer or build_optimizer(model)
    saved_snapshots = []
    loss_history = []
    
    # Load existing weights if available
    if model_dir and model_name:
        model_path = os.path.join(model_dir, model_name + ".pt")
        load_model_weights(model, model_path, device, verbose=True)

    for epoch in tqdm(range(epochs), desc="Training 2D VAE Model on Epochs"):
        model.train()

        # We load a random client data-pair
        client = random.randint(0, len(training_pairs)-1)
        xs = dataLoader.get_all_slices_as_tensor(training_pairs[client][0], crop_size=crop_size)  # (193, 193, 224)
        ys = dataLoader.get_all_slices_as_tensor(training_pairs[client][1], crop_size=crop_size)  # (193, 193, 224)

        num_slices = min(len(xs), len(ys)) # They should be equal, but just in case
        loss_sum = 0.0

        # Iterate over each slice
        for i in range(num_slices):
            x_slice = xs[i]
            y_slice = ys[i]

            x = dataLoader.to_torch_img(x_slice, device)   # (1,1,193,224)
            y = dataLoader.to_torch_img(y_slice, device)   # (1,1,193,224)

            optimizer.zero_grad()
            recon, mu, logvar = model(x)

            loss = criterion(recon, y, mu, logvar)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
        
        # Log mean loss for all slices in this epoch
        loss_history.append(loss_sum / num_slices)

        # --- Every Xth pair, save a snapshot of reconstruction vs target ---
        if save_every > 0 and (epoch % save_every == 0) and num_slices > 0: 
            # pick a safe index to visualize
            with torch.no_grad():
                x_show = dataLoader.to_torch_img(xs[idx_to_show], device)
                y_show = dataLoader.to_torch_img(ys[idx_to_show], device)
                recon_show, _, _ = model(x_show)

                # convert to numpy for visualization
                x_np     = dataLoader.to_numpy_img(x_show)
                y_np     = dataLoader.to_numpy_img(y_show)      
                recon_np = dataLoader.to_numpy_img(recon_show) 

            saved_snapshots.append({"iter": epoch, "x": x_np, "y": y_np, "recon": recon_np})

    # Save final model weights
    if model_dir and model_name:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name + ".pt")
        save_model_weights(model, model_path, verbose=True)

    return model, loss_history, saved_snapshots

def fit_3D(
    model,
    device: torch.device,
    dataLoader: HuntDataLoader,
    training_pairs: list[tuple[str, str]],
    criterion=None,
    epochs=1,
    optimizer=None,
    print_every=-1,
    save_every=-1,
    trim_slices=0,
    crop_size=(193, 224),
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

            saved_snapshots.append({"iter": i, "x": x_np, "y": y_np, "recon": recon_np, "loss": loss.item()})
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