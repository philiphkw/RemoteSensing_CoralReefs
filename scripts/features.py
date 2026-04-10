import numpy as np
import gc
import rioxarray
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import scripts.config as config



def percentile_clip(data, lower=config.PERCENTILE_LOWER, upper=config.PERCENTILE_UPPER):
    print(f"   - Clipping to percentile {lower}-{upper}")
    data_clipped = data.copy()
    for i in range(data.shape[1]):
        col = data[:, i]
        data_clipped[:, i] = np.clip(col, np.percentile(col, lower), np.percentile(col, upper))

    return data_clipped



def build_feature_matrix(bands_all, year, resample_freq, resample_agg):
    """Resample bands to resample_freq, filter by year, flatten and stack."""
    print("   - Building feature matrix")
    flattened_bands = []
    for name, band in tqdm(bands_all.items(), desc=f"      - Resampling {year or 'non-2022'}"):
        if year is not None:
            if isinstance(year, (list, tuple)):
                band = band.sel(time=band.time.dt.year.isin(year))
            else:
                band = band.sel(time=band.time.dt.year == year)
        else:
            band = band  # no filtering
        
        resampled_obj = band.resample(time=resample_freq)
        resampled = getattr(resampled_obj, resample_agg)()  # ← calls .median(), .mean() etc. dynamically
        
        flattened_bands.append(resampled.values.astype(np.float32).reshape(-1))

        del resampled
        gc.collect()

    return np.stack(flattened_bands, axis=1)



def prepare_data(bands_all, 
                 year, 
                 resample_freq, 
                 resample_agg, 
                 scaler=None, 
                 lower=config.PERCENTILE_LOWER, 
                 upper=config.PERCENTILE_UPPER
                 ):
    
    """Full pipeline: resample → mask invalids → clip → normalize."""
    print(" - Preparing data for model training...")
    matrix = build_feature_matrix(bands_all, year, resample_freq, resample_agg)

    valid_mask = np.isfinite(matrix).all(axis=1)
    matrix_valid = matrix[valid_mask]

    matrix_valid = percentile_clip(matrix_valid, lower, upper)

    if scaler is None:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(matrix_valid)
    else:
        normalized = scaler.transform(matrix_valid)

    del matrix, matrix_valid
    gc.collect()

    return normalized, valid_mask, scaler