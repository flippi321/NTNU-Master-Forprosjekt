import numpy as np
import nibabel as nib
from sklearn.mixture import GaussianMixture

def _to_array(x):
    if isinstance(x, str):
        img = nib.load(x)
        arr = np.asanyarray(img.get_fdata(), dtype=np.float64)
    elif hasattr(x, "get_fdata"):
        arr = np.asanyarray(x.get_fdata(), dtype=np.float64)
    else:
        arr = np.asanyarray(x, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def _robust_normalize(a, p_lo=0.5, p_hi=99.5):
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if hi <= lo:
        return (a - a.min()) / (a.ptp() + 1e-12)
    a = np.clip(a, lo, hi)
    return (a - lo) / (hi - lo + 1e-12)

def calculate_bpf_from_skullstripped_t1_gmm(scan, sample_max=500_000, random_state=0) -> float:
    """
    BPF from a skull-stripped T1 using a 3-component GMM (soft counts).
    - scan: path | nib image | numpy array
    - returns a single float (BPF)
    Assumes non-zero voxels are brain (ICV). Robust to intensity shifts.
    """
    x = _to_array(scan)
    # ICV = non-zero (use tiny percentile to ignore tiny noise)
    bg_eps = float(np.percentile(x, 0.1))
    icv_mask = x > max(0.0, bg_eps)
    if icv_mask.sum() == 0:
        return float("nan")

    vals = x[icv_mask]
    vals = _robust_normalize(vals)  # map roughly to [0,1] and reduce bias/scale issues

    # Subsample for fitting if huge
    n = vals.size
    if n > sample_max:
        idx = np.random.default_rng(random_state).choice(n, size=sample_max, replace=False)
        sample = vals[idx][:, None]
    else:
        sample = vals[:, None]

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=random_state)
    gmm.fit(sample)

    # Predict responsibilities on all voxels (soft assignment)
    resp = gmm.predict_proba(vals[:, None])  # shape (N, 3)
    means = gmm.means_.ravel()

    # CSF should be the darkest on T1 -> smallest mean
    csf_k = int(np.argmin(means))

    # Soft voxel counts per class
    csf_vox = float(resp[:, csf_k].sum())
    total_vox = float(resp.sum())  # equals number of voxels
    bpv_vox = total_vox - csf_vox  # GM+WM

    # Ratio of soft counts (voxel volume cancels in the ratio)
    return float(bpv_vox / total_vox)

def calculate_bpf_from_skullstripped_t1_gmm_with_qc(scan, sample_max=500_000, random_state=0):
    def _to_array(x):
        if isinstance(x, str):
            img = nib.load(x); arr = np.asanyarray(img.get_fdata(), dtype=np.float64)
        elif hasattr(x, "get_fdata"):
            arr = np.asanyarray(x.get_fdata(), dtype=np.float64)
        else:
            arr = np.asanyarray(x, dtype=np.float64)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _robust_normalize(a, p_lo=0.5, p_hi=99.5):
        lo, hi = np.percentile(a, [p_lo, p_hi])
        if hi <= lo: return (a - a.min()) / (a.ptp() + 1e-12)
        a = np.clip(a, lo, hi); return (a - lo) / (hi - lo + 1e-12)

    x = _to_array(scan)
    bg_eps = float(np.percentile(x, 0.1))
    icv_mask = x > max(0.0, bg_eps)
    if icv_mask.sum() == 0:
        return {"bpf": float("nan"), "error": "empty ICV mask"}

    vals = _robust_normalize(x[icv_mask])

    n = vals.size
    rng = np.random.default_rng(random_state)
    sample = vals[rng.choice(n, size=min(n, sample_max), replace=False)][:, None]

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=random_state)
    gmm.fit(sample)
    resp = gmm.predict_proba(vals[:, None])  # (N,3)
    means = gmm.means_.ravel()
    order = np.argsort(means)                 # dark→bright
    csf_k, gm_k, wm_k = order[0], order[1], order[2]

    counts = resp.sum(axis=0)                 # soft voxel counts per component
    total = counts.sum()
    csf_vox = float(counts[csf_k])
    bpf = float((total - csf_vox) / total)

    return {
        "bpf": bpf,
        "means": means.tolist(),
        "ordered_means_dark_to_bright": means[order].tolist(),
        "fractions": {
            "csf": csf_vox / total,
            "gm": float(counts[gm_k] / total),
            "wm": float(counts[wm_k] / total),
        },
        "icv_voxels": int(icv_mask.sum()),
    }
