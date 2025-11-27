import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.hunt_data_loader import HuntDataLoader
from utils.loss_functions import ssim_loss_3d
from models.MiniEncoder3D import perceptual_loss_3d
from tqdm import tqdm

# ---------------------------------------------
# General Helper Functions
# ---------------------------------------------

def build_optimizer(model, lr=1e-4, wd=1e-4):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def cap_logged_loss(loss: torch.Tensor, loss_cap: float = 1e6) -> float:
    """
    Cap a tensor loss.
    Returns a float value suitable for logging.
    """
    capped = torch.clamp(loss, max=loss_cap)
    return float(capped.item())


# ---------------------------------------------
# 2D Training Loop
# ---------------------------------------------

def fit_2d_models_per_slice(
        model_constructor,
        optimizer_constructor,
        device: torch.device,
        dataLoader,
        training_pairs: list[tuple[str, str]],
        validation_pairs: list[tuple[str, str]],
        training_loss_function, 
        logged_loss_function,
        epochs: int = 1000,
        save_every: int = -1,
        total_slices: int = 193,
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
    
    Evaluates each batch of slice models on `validation_pairs` using 
    a batched slice-wise evaluator.
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
            start_slice=start,
            dataLoader=dataLoader,
            training_pairs=training_pairs,
            validation_pairs=validation_pairs,
            training_loss_function=training_loss_function,
            logged_loss_function=logged_loss_function,
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
        start_slice: int,
        dataLoader,                       
        training_pairs: list[tuple[str, str]],
        validation_pairs: list[tuple[str, str]],
        training_loss_function,
        logged_loss_function,
        epochs: int             = 1000,  
        save_every: int         = 100,
        eval_every: int         = 100,
        crop_size: tuple        = (193, 229),
        idx_to_show: int        = 93,  # Approx. middle slice
        model_dir: str          = "div",    # Assume load/save if defined
        model_names: list[str]  = None,
    ):
    """
    Trains a list of per-slice models. Model for global slice index i
    is trained on slice i of each client volume.

    Also evaluates this batch of slice models on `validation_pairs`
    using `evaluate_slice_model_batch`, and saves best-per-batch weights.
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
    snapshots    = []

    num_clients = len(training_pairs)
    best_val_loss = np.inf

    for epoch in range(epochs):
        # Pick a random client (reused for all slice-models in this epoch)
        client = random.randint(0, num_clients - 1)
        xs = dataLoader.get_all_slices_as_tensor(training_pairs[client][0], crop_size=crop_size)
        ys = dataLoader.get_all_slices_as_tensor(training_pairs[client][1], crop_size=crop_size)

        num_slices = min(len(xs), len(ys))
        epoch_losses = []

        for i, model, optimizer in zip(slice_indices, model_batch, optimizer_batch):
            if i >= num_slices:
                continue

            model.train()
            x = dataLoader.to_torch_img(xs[i], device)  # (1,1,H,W)
            y = dataLoader.to_torch_img(ys[i], device)  # (1,1,H,W)

            optimizer.zero_grad()
            recon, mu_opt, logvar_opt = model(x) 
            
            # UNET returns None for mu and logvar
            if mu_opt is not None and logvar_opt is not None:
                loss = training_loss_function(recon, y, mu_opt, logvar_opt)
            else:
                loss = training_loss_function(recon, y)

            loss.backward()
            optimizer.step()

            epoch_losses.append(cap_logged_loss(logged_loss_function(recon, y)))

            # snapshot only if this model's slice is the one of interest
            if (save_every > 0) and (epoch % save_every == 0 or epoch == epochs - 1):
                with torch.no_grad():
                    x_show = dataLoader.to_torch_img(xs[idx_to_show], device)
                    y_show = dataLoader.to_torch_img(ys[idx_to_show], device)
                    recon_show, _, _ = model(x_show)

                    x_np     = dataLoader.to_numpy_img(x_show)
                    y_np     = dataLoader.to_numpy_img(y_show)
                    recon_np = dataLoader.to_numpy_img(recon_show)

                if i == idx_to_show:
                    snapshots.append({
                        "iter": epoch,
                        "slice_idx": i,
                        "x": x_np,
                        "y": y_np,
                        "recon": recon_np
                    })

                # --- Evaluate this batch on validation set ---
            if eval_every > 0 and (epoch % eval_every == 0 or epoch == epochs - 1):
                # slice_indices[-1] because we only want to do this once per batch
                if model_dir and model_names and i == slice_indices[-1]:
                    val_loss = evaluate_slice_model_batch(
                        model_batch=model_batch,
                        subset=validation_pairs,
                        start_slice=start_slice,
                        data_loader=dataLoader,
                        device=device,
                        loss_function=logged_loss_function,
                        crop_size=crop_size,
                        hide_bar=True,
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"[Slices {slice_indices[0]}–{slice_indices[-1]}] "
                              f"New best val loss at epoch {epoch}: {best_val_loss:.6f}")

                        # save *_best.pt for all the models in the batch
                        if model_dir:
                            os.makedirs(model_dir, exist_ok=True)
                            for m, name in zip(model_batch, model_names):
                                if name:
                                    best_path = os.path.join(model_dir, name.replace(".pt", "_best.pt"))
                                    save_model_weights(m, best_path)

        if epoch_losses:
            loss_history.append(sum(epoch_losses) / len(epoch_losses))

    # save final weights individually for all models in the batch
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
        validation_pairs: list[tuple[str, str]],
        training_loss_function, 
        logged_loss_function,
        epochs: int, 
        save_every: int = 100,
        eval_every: int = 100,
        crop_size: tuple = (193, 229),
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
    best_val_loss = np.inf
    
    # Load existing weights if available
    if model_dir and model_name:
        model_path = os.path.join(model_dir, model_name + ".pt")
        load_model_weights(model, model_path, device, verbose=True)

    for epoch in tqdm(range(epochs), desc="Training 2D Model"):
        model.train()

        # We load a random client data-pair
        client = random.randint(0, len(training_pairs)-1)
        xs = dataLoader.get_all_slices_as_tensor(training_pairs[client][0], crop_size=crop_size)  # (193, 193, 229)
        ys = dataLoader.get_all_slices_as_tensor(training_pairs[client][1], crop_size=crop_size)  # (193, 193, 229)

        num_slices = min(len(xs), len(ys)) # They should be equal, but just in case
        loss_sum = 0.0

        # Iterate over each slice
        for i in range(num_slices):
            x_slice = xs[i]
            y_slice = ys[i]

            x = dataLoader.to_torch_img(x_slice, device)   # (1,1,193,229)
            y = dataLoader.to_torch_img(y_slice, device)   # (1,1,193,229)

            optimizer.zero_grad()
            recon, mu_opt, logvar_opt = model(x)

            # UNET returns None for mu and logvar
            if mu_opt is not None and logvar_opt is not None:
                loss = training_loss_function(recon, y, mu_opt, logvar_opt)
            else:
                loss = training_loss_function(recon, y)

            loss.backward()
            optimizer.step()

            loss_sum += cap_logged_loss(logged_loss_function(recon, y))
        
        # Log mean loss for all slices in this epoch
        loss_history.append(loss_sum / num_slices)

        # --- Every Xth pair, save a snapshot of reconstruction vs target ---
        if save_every > 0 and (epoch % save_every == 0 or epoch == epochs - 1) and num_slices > 0: 
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

            # --- Evaluate on validation set ---

            val_loss = evaluate_global_model(model=model,
                                             subset=validation_pairs,
                                             data_loader=dataLoader,
                                             device=device,
                                             loss_function=logged_loss_function,
                                             crop_size=crop_size)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best model at epoch {epoch} with val loss {best_val_loss:.6f}")
                
                if model_dir and model_name:
                    os.makedirs(model_dir, exist_ok=True)
                    save_model_weights(model, os.path.join(model_dir, model_name + "_best.pt"))

    # Save final model weights
    if model_dir and model_name:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name + ".pt")
        save_model_weights(model, model_path, verbose=True)

    return model, loss_history, saved_snapshots

# ---------------------------------------------
# 2D Helper Functions
# ---------------------------------------------

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

def evaluate_global_model(
        model, 
        subset, 
        data_loader, 
        device, 
        loss_function, 
        crop_size=(192,224), 
        hide_bar=False
    ):
    model.eval()
    
    avg_error  = 0.0

    with torch.no_grad():
        for x_path, y_path in tqdm(subset, desc="Evaluating Universal Model", leave=True, disable=hide_bar):
            x_vol = data_loader.load_from_path(x_path, crop_size)
            y_vol = data_loader.load_from_path(y_path, crop_size)

            depth = x_vol.shape[2]
            assert depth >= 192

            for i in range(192):
                x_slice = x_vol[:, :, i]
                y_slice = y_vol[:, :, i]

                x = data_loader.to_torch_img(x_slice, device)
                y = data_loader.to_torch_img(y_slice, device)

                recon, _, _  = model(x)
                
                avg_error  += loss_function(recon, y).item()
                
    total = len(subset) * 192
    return avg_error / total

def evaluate_slice_model_batch(
        model_batch,
        subset,
        start_slice: int,
        data_loader,
        device,
        loss_function,
        crop_size=(192, 224),
        hide_bar=False,
    ):
    """
    Evaluate a batch of slice-wise models.

    Args:
        model_batch: list of per-slice models. model_batch[i] is responsible for slice (start_slice + i).
        subset:      list of (x_path, y_path) pairs.
        start_slice: global slice index of model_batch[0]. The rest follow sequentially.
    """
    for m in model_batch:
        m.eval()

    total_error = 0.0
    n_evals = 0

    with torch.no_grad():
        desc = f"Evaluating slices {start_slice}–{start_slice + len(model_batch) - 1}"
        for x_path, y_path in tqdm(subset, desc=desc, leave=True, disable=hide_bar):
            x_vol = data_loader.load_from_path(x_path, crop_size)
            y_vol = data_loader.load_from_path(y_path, crop_size)

            depth = x_vol.shape[2]

            # iterate over models in this batch
            for local_idx, model in enumerate(model_batch):
                idx = start_slice + local_idx

                # safety: don't go beyond volume depth
                if idx >= depth:
                    continue

                x_slice = x_vol[:, :, idx]
                y_slice = y_vol[:, :, idx]

                x = data_loader.to_torch_img(x_slice, device)
                y = data_loader.to_torch_img(y_slice, device)

                recon, _, _ = model(x)
                total_error += loss_function(recon, y).item()
                n_evals += 1

    # Avoid division by zero in pathological cases
    return total_error / max(n_evals, 1)



# ---------------------------------------------
# 3D Training Loop
# ---------------------------------------------


def fit_3D(
    model,
    device: torch.device,
    dataLoader: HuntDataLoader,
    training_pairs: list[tuple[str, str]],
    criterion=None,
    logged_loss_function=None,
    epochs=1,
    optimizer=None,
    print_every=-1,
    save_every=-1,
    trim_slices=0,
    crop_size=(193, 224)
):
    """
    Train a 3D model on full volumes using HuntDataLoader.
    - Expects model(x) -> either:
        * y_hat
        * (y_hat, delta)   where delta is a residual volume previously used for TV regularization
    - criterion is optional:
        * If provided and accepts (y_hat, y), we use it.
        * Otherwise we fall back to L1.
    - Snapshots show the mid-axial slice (H,W) for input, target, and recon.
    """
    
    optimizer = optimizer or build_optimizer(model)
    saved_snapshots, loss_history = [], []

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
            y_hat, _ = out[0], out[1]
        else:
            y_hat, _ = out, None

        # --- loss ---
        loss = None

        if criterion is not None:
            # Try criterion(y_hat, y) returning either a scalar or (loss, *extras)
            try:
                crit_out = criterion(y_hat, y)
                if isinstance(crit_out, (tuple, list)):
                    loss = crit_out[0]
                else:
                    loss = crit_out
            except TypeError:
                # The provided criterion didn't match (y_hat,y); we'll fall back to L1
                pass

        if loss is None:
            # fallback: L1
            loss = F.l1_loss(y_hat, y)

        loss.backward()
        optimizer.step()

        # --- log ---
        if logged_loss_function is not None:
            logged_loss = cap_logged_loss(logged_loss_function(y_hat, y))
            loss_history.append(logged_loss)

        if (print_every > 0) and (i % print_every == 0):
            print(f"[Iter {i}] total: {loss.item():.6f}")

        # --- snapshot ---
        if (i % save_every == 0 or i == epochs - 1):
            with torch.no_grad():
                x_np = mid_axial_slice_5d(x)
                y_np = mid_axial_slice_5d(y)
                recon_np = mid_axial_slice_5d(y_hat)

            saved_snapshots.append({"iter": i, "x": x_np, "y": y_np, "recon": recon_np, "loss": loss.item()})
            print(f"Saved snapshot at iter {i} (mid-axial slice)")

    return model, loss_history, saved_snapshots

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

# ---------------------------------------------
# 3D GAN Training Loop
# ---------------------------------------------
# G: generator model
# D: discriminator model
# ---------------------------------------------

def fit_3D_gan(
    G,
    D,
    device: torch.device,
    dataLoader: HuntDataLoader,
    training_pairs: list[tuple[str, str]],
    epochs=1,
    lr_G=1e-4,
    opt_G=None,
    lr_D=1e-4,
    opt_D=None,
    alpha=25.0,         # weight on voxel-wise L1
    beta=25.0,          # weight on perceptual loss
    trim_slices=0,
    crop_size=(193,229),
    save_every=10,
    print_every=1,
    phi=None,            # <- pass PhiFeatureExtractor here
    layer_weights=None,  # optional list of per-layer weights
    logged_loss_function=None,
):
    assert phi is not None, "Pass a PhiFeatureExtractor instance as 'phi' for MPGAN."

    opt_G = opt_G or optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    opt_D = opt_D or optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

    saved_snaps = []
    loss_history = []

    bce = nn.BCEWithLogitsLoss()  # standard GAN BCE loss

    for it in range(epochs):
        G.train()
        D.train()

        # === 1) Load one random subject pair and build (1,1,D,H,W) ===
        client = random.randint(0, len(training_pairs)-1)
        x_path, y_path = training_pairs[client]

        xs_list = dataLoader.get_all_slices_as_tensor(x_path, crop_size=crop_size)
        ys_list = dataLoader.get_all_slices_as_tensor(y_path, crop_size=crop_size)

        if trim_slices and trim_slices > 0:
            xs_list = xs_list[trim_slices:-trim_slices]
            ys_list = ys_list[trim_slices:-trim_slices]

        Ddepth = min(len(xs_list), len(ys_list))
        xs_list = xs_list[:Ddepth]
        ys_list = ys_list[:Ddepth]

        x_vol = to_torch_vol(xs_list, device)  # (1,1,D,H,W) in [0,1]
        y_vol = to_torch_vol(ys_list, device)  # (1,1,D,H,W) in [0,1]

        # === 2) Generator forward ===
        # G should output predicted follow-up with same shape
        y_fake = G(x_vol)  # (1,1,D,H,W)

        # ------------------------------------------------------------------
        # Train D
        # ------------------------------------------------------------------
        opt_D.zero_grad()

        pred_real = D(y_vol)
        loss_D_real = bce(pred_real, torch.ones_like(pred_real, device=device))

        pred_fake = D(y_fake.detach())
        loss_D_fake = bce(pred_fake, torch.zeros_like(pred_fake, device=device))

        loss_D_total = 0.5 * (loss_D_real + loss_D_fake)
        loss_D_total.backward()
        opt_D.step()

        # ------------------------------------------------------------------
        # Train G
        # ------------------------------------------------------------------
        opt_G.zero_grad()

        # (a) adversarial: want D(y_fake) -> 1
        pred_fake_for_G = D(y_fake)
        loss_adv_G = bce(pred_fake_for_G, torch.ones_like(pred_fake_for_G, device=device))

        # (b) voxel-wise L1
        loss_l1 = torch.mean(torch.abs(y_fake - y_vol))

        # (c) perceptual via φ (frozen)
        loss_perc = perceptual_loss_3d(y_fake, y_vol, phi, layer_weights=layer_weights, reduction="l1")

        # total generator loss (MPGAN)
        loss_G_total = loss_adv_G + alpha * loss_l1 + beta * loss_perc
        loss_G_total.backward()
        opt_G.step()

        if logged_loss_function is not None:
            loss_history.append(cap_logged_loss(logged_loss_function(y_fake, y_vol)))

        # ------------------------------------------------------------------
        # Logging / snapshot
        # ------------------------------------------------------------------
        if (print_every > 0) and (it % print_every == 0):
            print(
                f"[Iter {it}] D_total={loss_D_total.item():.4f} "
                f"G_total={loss_G_total.item():.4f} "
                f"L1={loss_l1.item():.4f} "
                f"Perc={loss_perc.item():.4f} "
                f"AdvG={loss_adv_G.item():.4f}"
            )

        if it % save_every == 0 or it == epochs - 1:
            with torch.no_grad():
                x_np   = mid_axial_slice_5d(x_vol)
                y_np   = mid_axial_slice_5d(y_vol)
                yhat_np= mid_axial_slice_5d(y_fake)
            saved_snaps.append({
                "iter": it,
                "x": x_np,
                "y": y_np,
                "recon": yhat_np,
                "loss_D": float(loss_D_total.item()),
                "loss_G": float(loss_G_total.item()),
            })
            print(f"Saved snapshot at iter {it}")

    return G, D, loss_history, saved_snaps
