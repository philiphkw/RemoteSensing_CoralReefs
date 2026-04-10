import os
os.environ['USE_PYGEOS'] = '0'
path = os.getcwd()
import sys
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import rioxarray
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import xarray as xr


warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')

# Get project root to allow for relative imports
script_dir = Path(os.getcwd())
project_root = script_dir.parent

print(f"Project root: {project_root}")

# Add PROJECT ROOT to sys.path
sys.path.insert(0, str(project_root))

import scripts.config as config
import scripts.cache as cache
from scripts.features import prepare_data
from scripts.model import train_gmm, predict, save_labels_timeseries, print_cluster_distribution
from scripts.data_loading import compute_bands_all
from scripts.visualization import load_monthly_rgb, get_all_monthly_timestamps, percentile_stretch

# Get project root to allow for relative imports
script_dir = Path(os.getcwd())
project_root = script_dir.parent

print(f"Project root: {project_root}")

# Add PROJECT ROOT to sys.path
sys.path.insert(0, str(project_root))


import os
from os.path import join as pjoin

# KMeans-specific settings
KMEANS_K = 10
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 20

# Reuse same baseline / resampling logic as the GMM pipeline
BASELINE_YEAR = config.BASELINE_YEAR
RESAMPLE_FREQ = config.RESAMPLE_FREQ
RESAMPLE_AGG = config.RESAMPLE_AGG

# Local output root with more space
LOCAL_OUTPUT_ROOT = r"C:\Users\denni\OneDrive\Documenten\ADS\Spatial Simulation Modelling\Project\RemoteSensing_CoralReefs\interim"
LOCAL_INTERIM_DIR = pjoin(LOCAL_OUTPUT_ROOT, "interim")
LOCAL_MODELS_DIR = pjoin(LOCAL_OUTPUT_ROOT, "models")
LOCAL_FIGURES_DIR = pjoin(r"C:\Users\denni\OneDrive\Documenten\ADS\Spatial Simulation Modelling\Project\RemoteSensing_CoralReefs\reports", "figures")

os.makedirs(LOCAL_OUTPUT_ROOT, exist_ok=True)
os.makedirs(LOCAL_INTERIM_DIR, exist_ok=True)
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
os.makedirs(LOCAL_FIGURES_DIR, exist_ok=True)

# Optional: override config dirs for the rest of the notebook
config.INTERIM_DIR = LOCAL_INTERIM_DIR
config.MODELS_DIR = LOCAL_MODELS_DIR
config.FIGURES_DIR = LOCAL_FIGURES_DIR

# Names for saved outputs
YEAR_STRING = ",".join(str(y) for y in BASELINE_YEAR)
RUN_NAME_KMEANS = f"B({YEAR_STRING})_kmeans{KMEANS_K}_{RESAMPLE_FREQ}_{RESAMPLE_AGG}{'_bathy' if config.LYZENGA_ALG else ''}"

MODEL_NAME = f"kmeans_{RUN_NAME_KMEANS}"
LABELS_DIR_NAME = f"labels_monthly_{RUN_NAME_KMEANS}"
PNG_DIR = pjoin(config.FIGURES_DIR, f"monthly_kmeans_pngs_{RUN_NAME_KMEANS}")
os.makedirs(PNG_DIR, exist_ok=True)

print("RUN_NAME_KMEANS:", RUN_NAME_KMEANS)
print("LOCAL_OUTPUT_ROOT:", LOCAL_OUTPUT_ROOT)
print("INTERIM_DIR:", config.INTERIM_DIR)
print("MODELS_DIR:", config.MODELS_DIR)
print("FIGURES_DIR:", config.FIGURES_DIR)
print("PNG output folder:", PNG_DIR)

gmm = joblib.load(".//models//gmm_B(2022,2023)_k10_init3_MS_median_bathy_11F.pkl")
k_means = joblib.load(".//models//kmeans_B(2022,2023)_kmeans10_MS_median_bathy.pkl")
tiff_path = ".//data//raw//PSScene//20240607_000749_84_24e5_3B_AnalyticMS_SR_8b_clip_philip.tif"
band = rioxarray.open_rasterio(tiff_path)
data_monthly = np.load(".//data//interim//monthly_data_B(2022,2023)_k10_init3_MS_median_bathy_11F.npy")
# labels_monthly = np.load(".//data//interim//predictions_monthly_B(2022,2023)_k10_init3_MS_median_bathy_11F_labels.npy")
mask_monthly = np.load(".//data//interim//monthly_data_B(2022,2023)_k10_init3_MS_median_bathy_11F_mask.npy")
RUN_NAME_GMM = f"B({YEAR_STRING})_GMM{KMEANS_K}_{RESAMPLE_FREQ}_{RESAMPLE_AGG}{'_bathy' if config.LYZENGA_ALG else ''}"
RUN_NAME_KMEANS = f"B({YEAR_STRING})_kmeans{KMEANS_K}_{RESAMPLE_FREQ}_{RESAMPLE_AGG}{'_bathy' if config.LYZENGA_ALG else ''}"


MODEL_NAME = f"kmeans_{RUN_NAME_GMM}"
LABELS_DIR_NAME = f"labels_monthly_{RUN_NAME_GMM}"
PNG_DIR_KMEANS = pjoin(config.FIGURES_DIR, f"monthly_kmeans_pngs_{RUN_NAME_KMEANS}")
PNG_DIR_GMM = pjoin(config.FIGURES_DIR, f"monthly_GMM_pngs_{RUN_NAME_GMM}")
RESAMPLE_FREQ = config.RESAMPLE_FREQ
KMEANS_K = 10


bands_all = cache.cache_bands(
    "bands_all",
    compute_fn=lambda: compute_bands_all(
        lyzenga=config.LYZENGA_ALG,
    ),
    force_recompute=config.FORCE_BANDS
)

SUFFIX = "_11F"

x_size = 1237
y_size = 2030

num_timesteps_monthly = len(mask_monthly) // (x_size * y_size)
# perc_sample = f"{config.N_SAMPLES/len(data_baseline):.3f}" if config.N_SAMPLES else ""
# sample_prefix = f"sample{perc_sample}pct_" if config.N_SAMPLES else ""
sample_prefix = ""
# ================== Predict ==================
labels_monthly_kmeans, probs_monthly = cache.cache_predictions(
    f'{sample_prefix}{config.PREDICTIONS_NAME}{SUFFIX}',
    compute_fn=lambda: predict(k_means, data_monthly),
    force_recompute=config.FORCE_PREDICTIONS
)

labels_monthly_gmm, probs_monthly = cache.cache_predictions(
    f'{sample_prefix}{config.PREDICTIONS_NAME}{SUFFIX}',
    compute_fn=lambda: predict(gmm, data_monthly),
    force_recompute=config.FORCE_PREDICTIONS
)

def save_monthly_comparisons(
        monthly_rgb, 
        spatial_labels_monthly, 
        timestamps, all_timestamps, 
        output_dir, 
        from_year=None, 
        n_clusters=10
    ):
    """Save one PNG per month: original RGB on the left, KMeans clusters on the right."""
    os.makedirs(output_dir, exist_ok=True)
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    cmap.set_bad(color='black')
    model_name = "Kmeans" if "kmeans" in output_dir else "GMM"
    if from_year is not None:
        timestamps_to_plot = [ts for ts in timestamps if pd.Timestamp(ts).year >= from_year]
    else:
        timestamps_to_plot = list(timestamps)

    for ts in tqdm(timestamps_to_plot, desc=f'Saving monthly {model_name} comparisons'):
        spatial_idx = all_timestamps.index(ts)
        year = pd.Timestamp(ts).year
        month = pd.Timestamp(ts).month
        ts_str = pd.Timestamp(ts).strftime('%Y_%m')

        rgb = monthly_rgb.sel(
            time=(monthly_rgb.time.dt.year == year) & (monthly_rgb.time.dt.month == month)
        ).squeeze().values.transpose(1, 2, 0)

        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        axes[0].imshow(np.flipud(percentile_stretch(rgb)), origin='upper')
        axes[0].set_title(f'{ts_str} - Original RGB', fontsize=14)
        axes[0].axis('off')

        im = axes[1].imshow(
            np.flipud(spatial_labels_monthly[spatial_idx]),
            cmap=cmap,
            vmin=-0.5,
            vmax=n_clusters - 0.5,
            interpolation='nearest',
            origin='upper'
        )
        axes[1].set_title(f'{ts_str} - {model_name} ({n_clusters} clusters)', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Cluster ID', ticks=range(n_clusters))

        plt.tight_layout()
        plt.savefig(pjoin(output_dir, f'{model_name}_compare_{ts_str}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'All monthly PNGs saved to: {output_dir}')




print_cluster_distribution(labels_monthly_kmeans)

# Get the actual timestamps from one of your resampled bands
ref_band = bands_all[list(bands_all.keys())[0]]
timestamps = ref_band.sel(time=ref_band.time.dt.year.isin([2022, 2023, 2024]))\
                     .resample(time=config.RESAMPLE_FREQ)\
                     .median()\
                     .time.values

# ================== Save monthly predictions ==================
spatial_labels_monthly_kmeans = cache.cache_spatial_timeseries(
    f'{sample_prefix}{config.SPATIAL_NAME}{SUFFIX}',
    compute_fn=lambda: save_labels_timeseries(
        labels=labels_monthly_kmeans,
        mask=mask_monthly,
        reference_band=ref_band,
        name=f'{sample_prefix}{config.SPATIAL_NAME}{SUFFIX}',
        num_timesteps=num_timesteps_monthly,
        timestamps=timestamps
    ),
    timestamps=timestamps,
    reference_band=ref_band,
    force_recompute=config.FORCE_SPATIAL
)
spatial_labels_monthly_gmm = cache.cache_spatial_timeseries(
    f'{sample_prefix}{config.SPATIAL_NAME}{SUFFIX}',
    compute_fn=lambda: save_labels_timeseries(
        labels=labels_monthly_gmm,
        mask=mask_monthly,
        reference_band=ref_band,
        name=f'{sample_prefix}{config.SPATIAL_NAME}{SUFFIX}',
        num_timesteps=num_timesteps_monthly,
        timestamps=timestamps
    ),
    timestamps=timestamps,
    reference_band=ref_band,
    force_recompute=config.FORCE_SPATIAL
)


monthly_rgb = xr.concat(
    [
        bands_all["red"].resample(time=RESAMPLE_FREQ).mean(),
        bands_all["green"].resample(time=RESAMPLE_FREQ).mean(),
        bands_all["blue"].resample(time=RESAMPLE_FREQ).mean(),
    ],
    dim="band"
).transpose("time", "band", "y", "x")

all_timestamps = get_all_monthly_timestamps(bands_all)


save_monthly_comparisons(
    monthly_rgb=monthly_rgb,
    spatial_labels_monthly=spatial_labels_monthly_kmeans,
    timestamps=all_timestamps,
    all_timestamps=all_timestamps,
    output_dir=PNG_DIR_KMEANS,
    from_year=None,
    n_clusters=KMEANS_K,
)

save_monthly_comparisons(
    monthly_rgb=monthly_rgb,
    spatial_labels_monthly=spatial_labels_monthly_gmm,
    timestamps=all_timestamps,
    all_timestamps=all_timestamps,
    output_dir=PNG_DIR_GMM,
    from_year=None,
    n_clusters=KMEANS_K,
)