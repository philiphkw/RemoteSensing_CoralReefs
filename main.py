import os
os.environ['USE_PYGEOS'] = '0'
path = os.getcwd()
print(path)

from scripts.features import prepare_data
from scripts.model import train_gmm, predict, save_labels_timeseries, print_cluster_distribution
from scripts.data_loading import compute_bands_all
import scripts.cache as cache
import scripts.config as config

# ================== Preprocessing ==================
bands_all = cache.cache_bands(
    "bands_all",
    compute_fn=lambda: compute_bands_all(
        lyzenga=config.LYZENGA_ALG,
    ),
    force_recompute=config.FORCE_BANDS
)

# ================== 2022/2023 baseline (train scaler + GMM) ==================
data_baseline, mask_baseline, scaler = cache.cache_data(
    f'{config.BASELINE_NAME}_{len(bands_all)}F',
    compute_fn=lambda: prepare_data(
        bands_all, 
        year=config.BASELINE_YEAR,
        resample_freq=config.RESAMPLE_FREQ,
        resample_agg=config.RESAMPLE_AGG,
        lower=config.PERCENTILE_LOWER,
        upper=config.PERCENTILE_UPPER
    ),
    force_recompute=config.FORCE_BASELINE
)

# ================== Monthly data (reuse scaler from 2022/2023) ==================
data_monthly, mask_monthly, _ = cache.cache_data(
    f'{config.MONTHLY_NAME}_{len(bands_all)}F',
    compute_fn=lambda: prepare_data(
        bands_all, 
        year=None, 
        resample_freq=config.RESAMPLE_FREQ,
        resample_agg=config.RESAMPLE_AGG,
        scaler=scaler,
        lower=config.PERCENTILE_LOWER, 
        upper=config.PERCENTILE_UPPER
    ),
    force_recompute=config.FORCE_MONTHLY
)

# ================== Train GMM Model ==================
SUFFIX = f"_{len(bands_all)}F"

print(f"[RUN_NAME] {config.RUN_NAME}{SUFFIX}")

# config.N_SAMPLES = len(data_baseline)*0.1
perc_sample = f"{config.N_SAMPLES/len(data_baseline):.3f}" if config.N_SAMPLES else ""
sample_prefix = f"sample{perc_sample}pct_" if config.N_SAMPLES else ""

gmm = cache.cache_model(
    f'{sample_prefix}{config.GMM_NAME}{SUFFIX}',
    compute_fn=lambda: train_gmm(
        data_baseline, 
        n_samples=config.N_SAMPLES, 
        n_components=config.GMM_COMPONENTS, 
        n_init=config.GMM_N_INIT),
    force_recompute=config.FORCE_GMM
)

# ================== Get number of time steps ==================
x_size = bands_all[list(bands_all.keys())[0]].shape[1]
y_size = bands_all[list(bands_all.keys())[0]].shape[2]

num_timesteps_monthly = len(mask_monthly) // (x_size * y_size)

# ================== Predict ==================
labels_monthly, probs_monthly = cache.cache_predictions(
    f'{sample_prefix}{config.PREDICTIONS_NAME}{SUFFIX}',
    compute_fn=lambda: predict(gmm, data_monthly),
    force_recompute=config.FORCE_PREDICTIONS
)

print_cluster_distribution(labels_monthly)

# Get the actual timestamps from one of your resampled bands
ref_band = bands_all[list(bands_all.keys())[0]]
timestamps = ref_band.sel(time=ref_band.time.dt.year.isin([2022, 2023, 2024]))\
                     .resample(time=config.RESAMPLE_FREQ)\
                     .median()\
                     .time.values

# ================== Save monthly predictions ==================
spatial_labels_monthly = cache.cache_spatial_timeseries(
    f'{sample_prefix}{config.SPATIAL_NAME}{SUFFIX}',
    compute_fn=lambda: save_labels_timeseries(
        labels=labels_monthly,
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