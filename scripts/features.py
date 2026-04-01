import numpy as np
import gc
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def percentile_clip(data, lower=2, upper=98):
    data_clipped = data.copy()
    for i in range(data.shape[1]):
        col = data[:, i]
        data_clipped[:, i] = np.clip(col, np.percentile(col, lower), np.percentile(col, upper))
    return data_clipped


def build_feature_matrix(bands_all, year, resample_freq, resample_agg):
    """Resample bands to resample_freq, filter by year, flatten and stack."""
    flattened_bands = []
    for band in tqdm(bands_all, desc=f"Resampling {year or 'non-2022'}"):
        if year:
            band = band.sel(time=band.time.dt.year == year)
        else:
            band = band.sel(time=band.time.dt.year != 2022)
        
        resampled_obj = band.resample(time=resample_freq)
        resampled = getattr(resampled_obj, resample_agg)()  # ← calls .median(), .mean() etc. dynamically
        
        flattened_bands.append(resampled.values.astype(np.float32).reshape(-1))
        del resampled
        gc.collect()
    return np.stack(flattened_bands, axis=1)


def prepare_data(bands_all, year, resample_freq, 
                 resample_agg, scaler=None, 
                 lower=2, upper=98, no_data=0):
    """Full pipeline: resample → mask invalids → clip → normalize."""
    matrix = build_feature_matrix(bands_all, year, resample_freq, resample_agg)

    valid_mask = np.isfinite(matrix).all(axis=1) & (matrix != no_data).any(axis=1)
    matrix_valid = matrix[valid_mask]

    matrix_valid = percentile_clip(matrix_valid, lower, upper)

    if scaler is None:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(matrix_valid)
    else:
        normalized = scaler.transform(matrix_valid)

    return normalized, valid_mask, scaler