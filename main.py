import os
import sys
import numpy as np
os.environ['USE_PYGEOS'] = '0'

path = os.getcwd()
parent = os.sep.join(path.split(os.sep)[:-1])
sys.path.append(parent)

from scripts.data_loading import calculate_indices, load_stack, extract_bands
from scripts.features import prepare_data
from scripts.model import train_gmm, predict, print_cluster_distribution, save_model, save_labels
from scripts.visualization import plot_pca_clusters, plot_spatial_map
from scripts.lyzenga import apply_lyzenga
from scripts.config import (PERCENTILE_LOWER, PERCENTILE_UPPER, GMM_COMPONENTS, 
                            GMM_N_INIT, RESAMPLE_FREQ, RESAMPLE_AGG)
import scripts.config as config
import scripts.cache as cache

# ── Load ──────────────────────────────────────────────
stack = load_stack()
raw_bands = extract_bands(stack)

assert all(v is not None for v in config.LYZENGA_DW_VALUES.values()), \
    "Fill in config.LYZENGA_DW_VALUES with deep-water reflectance values before running."

# Load calibration zone pixels (from a homogeneous-bottom area)
# Replace the block below with your actual calibration pixel extraction.
regression_pixels = {
    name: np.load(config.LYZENGA_REGRESSION_ZONE)[name]
    for name in config.LYZENGA_BANDS
}

print("\nApplying Lyzenga water column correction...")
di_bands = apply_lyzenga(
    raw_bands=raw_bands,
    dw_values=config.LYZENGA_DW_VALUES,
    ki_kj_ratios=config.LYZENGA_KI_KJ,         # None → estimate from data
    regression_pixels=regression_pixels,
    use_bathy=False,                             # True if you have bathymetry
    bathy=None,
)

indices = calculate_indices(raw_bands)
bands_all = (
    list(raw_bands.values()) +
    list(di_bands.values()) +       # ← NEW: Lyzenga DI bands
    list(indices.values())
)

x_size = bands_all[0].shape[1]
y_size = bands_all[0].shape[2]

# ── 2022 baseline (train scaler + GMM) ───────────────
data_2022, mask_2022, scaler = cache.cache_numpy(
    f'features_2022_{len(bands_all)}F_{config.RESAMPLE_FREQ}_{config.RESAMPLE_AGG}',
    compute_fn=lambda: prepare_data(
        bands_all, year=2022,
        resample_freq=config.RESAMPLE_FREQ,
        resample_agg=config.RESAMPLE_AGG,
        lower=config.PERCENTILE_LOWER,
        upper=config.PERCENTILE_UPPER
    ),
    force_recompute=False
)

# ── Train ─────────────────────────────────────────────
RUN_NAME = f"k{GMM_COMPONENTS}init{GMM_N_INIT}_{RESAMPLE_FREQ}_{RESAMPLE_AGG}_{'-'.join([i[:2] for i in indices.keys()])}"
config.RUN_NAME = RUN_NAME

gmm = train_gmm(data_2022)
labels_2022, probs_2022 = predict(gmm, data_2022)
print_cluster_distribution(labels_2022)

# ── Monthly data (reuse scaler from 2022) ─────────────
data_monthly, mask_monthly, _ = cache.cache_numpy(
    f'features_monthly_{len(bands_all)}F_{config.RESAMPLE_FREQ}_{config.RESAMPLE_AGG}',
    compute_fn=lambda: prepare_data(
        bands_all, 
        year=None, 
        resample_freq='MS',
        resample_agg=config.RESAMPLE_AGG,
        scaler=scaler,
        lower=config.PERCENTILE_LOWER, 
        upper=config.PERCENTILE_UPPER
    ),
    force_recompute=False
)

# ── Cache monthly predictions ────────────────────
labels_monthly, probs_monthly = cache.cache_predictions(
    f'predictions_monthly_{RUN_NAME}',
    compute_fn=lambda: predict(gmm, data_monthly),
    force_recompute=False
)


# ── Visualize & save ──────────────────────────────────
pca_result = plot_pca_clusters(data_2022, labels_2022, gmm)
plot_spatial_map(labels_2022, "labels_2022", mask_2022, x_size, y_size)
save_model(gmm, "gmm_model")
save_model(pca_result, "pca")
save_labels(labels_2022, mask_2022, bands_all[0], "labels_2022")
save_labels(labels_monthly, mask_monthly, bands_all[0], "labels_monthly")