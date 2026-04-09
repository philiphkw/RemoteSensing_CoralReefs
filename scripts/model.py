import time
import os
import numpy as np
import joblib
import rioxarray
import xarray as xr
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from os.path import join as pjoin
import scripts.config as config


INTERIM_DIR = pjoin(config.DATA_DIR, 'interim')
os.makedirs(INTERIM_DIR, exist_ok=True)

def sample_data(data, perc_sample=None, n_samples=100_000, random_state=config.RANDOM_STATE):
    """Randomly sample n_samples rows from data."""
    
    if perc_sample:
        n_samples = round(len(data) * perc_sample)

    if len(data) <= n_samples:
        print(f"   - Data has only {len(data):,} rows — using all of it...")
        return data
    
    print(f"   - Sampling {n_samples} pixels...")
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(data), size=n_samples, replace=False)
    print(f"      - Sampled {n_samples:,} / {len(data):,} rows (i.e. pixels) ({100*n_samples/len(data):.1f}%)")
    return data[idx]



def train_model(data, model, perc_sample=None, n_samples=None):
    print(" - Training Model...")
    t0 = time.time()
    if n_samples:
        data_sample = sample_data(data, perc_sample=perc_sample, n_samples=n_samples)
        model.fit(data_sample)
    else:
        model.fit(data)

    print(f"    - Training complete in {time.time()-t0:.1f}s")
    return model



def predict_gmm(gmm, data):
    t0 = time.time()
    print(" - Predicting labels...")
    labels = gmm.predict(data)
    probs = gmm.predict_proba(data)
    print(f" - Predicting complete in {time.time()-t0:.1f}s")
    return labels, probs



def predict_kmeans(kmeans, data):
    t0 = time.time()
    print(" - Predicting labels...")
    labels = kmeans.predict(data)
    print(f" - Predicting complete in {time.time()-t0:.1f}s")
    return labels



def print_cluster_distribution(labels, model_type):
    labels_flat = labels.flatten()
    total = len(labels_flat)
    print(f"{model_type.upper()} Cluster distribution:")
    for i in range(int(labels_flat.max()) + 1):
        count = int(np.sum(labels_flat == i))
        print(f" - Cluster {i}: {count:,} pixels ({100*count/total:.1f}%)")



def save_model(model, name):
    joblib.dump(model, pjoin(config.MODELS_DIR, f'{name}_{config.RUN_NAME}.pkl'))
    print("Models saved")



def save_labels(labels, mask, reference_band, name, num_timesteps=None):
    """
    Save labels as GeoTIFF, inheriting CRS and transform from reference_band.
    
    If num_timesteps is provided, assumes labels are from resampled temporal data
    and aggregates across time using majority voting.
    """
    t0 = time.time()

    x_size = reference_band.shape[1]
    y_size = reference_band.shape[2]
    
    # If temporal data, aggregate across time steps
    if num_timesteps is not None:
        print(f"Aggregating {num_timesteps} time steps across space...")
        
        # Expand labels back to full size (filling invalid rows with -1)
        print(" - Expanding labels to full temporal-spatial shape...")
        labels_full = np.full(len(mask), -1, dtype=labels.dtype)
        labels_full[mask] = labels
        
        # Reshape to (num_timesteps, x_size, y_size)
        print(" - Reshaping to (time, x, y)...")
        labels_temporal_spatial = labels_full.reshape(num_timesteps, x_size, y_size)
        
        # Majority vote per pixel across time (ignoring -1 padding)
        print(" - Performing majority voting across time steps...")  
        from scipy.stats import mode
        labels_spatial = np.full((x_size, y_size), np.nan, dtype=np.float32)
        
        # Reshape to (num_pixels, num_timesteps) for easier vectorization
        labels_pixel_time = labels_temporal_spatial.reshape(num_timesteps, -1).T
        
        print(f" - Vectorized majority voting on {labels_pixel_time.shape[0]:,} pixels...")
        
        for pixel_idx in tqdm(range(len(labels_pixel_time)), desc="   - Majority voting"):
            timeseries = labels_pixel_time[pixel_idx]
            valid_labels = timeseries[timeseries != -1].astype(int)
            
            if len(valid_labels) > 0:
                counts = np.bincount(valid_labels)
                labels_spatial.flat[pixel_idx] = np.argmax(counts)
        
        print(" - Majority voting complete")
        
        # Create final spatial array
        spatial = labels_spatial.astype(np.float32)
        spatial = np.flipud(spatial)
    else:
        # Original non-temporal case
        spatial = np.full((x_size, y_size), np.nan, dtype=np.float32)
        spatial[mask.reshape(x_size, y_size)] = labels
        spatial = np.flipud(spatial)
 
    da = xr.DataArray(spatial, dims=['y', 'x'])
    da.rio.write_crs(reference_band.rio.crs, inplace=True)
    da.rio.write_transform(inplace=True)
    da.rio.to_raster(pjoin(config.MODELS_DIR, f'{name}.tif'))
    print(f" - Saving labels complete in {time.time()-t0:.1f}s")
    print(f" - Labels saved: {name}.tif")

    return spatial



def save_labels_timeseries(labels, mask, ref_band, name, num_timesteps, timestamps):
    """Save one GeoTIFF per timestep instead of majority-voting across time."""
    
    t0 = time.time()
    print(" - Saving monthly labels...")
    os.makedirs(pjoin(config.INTERIM_DIR, name), exist_ok=True)

    x_size = ref_band.shape[1]
    y_size = ref_band.shape[2]

    labels_full = np.full(len(mask), np.nan, dtype=np.float32)
    labels_full[mask] = labels

    # Reshape to (num_timesteps, x, y)
    labels_4d = labels_full.reshape(num_timesteps, x_size, y_size)

    for t_idx, ts in enumerate(timestamps):
        spatial = labels_4d[t_idx].astype(np.float32)
        spatial[spatial == -1] = np.nan
        spatial = np.flipud(spatial)

        da = xr.DataArray(spatial, dims=['y', 'x'])
        da.rio.write_crs(ref_band.rio.crs, inplace=True)
        da.rio.write_transform(inplace=True)

        ts_str = pd.Timestamp(ts).strftime('%Y%m')
        da.rio.to_raster(pjoin(config.INTERIM_DIR, rf"{name}\{name}_{ts_str}.tif"))
        print(f"   - Saved {name}_{ts_str}.tif")
    print(f"Saving labels complete in {time.time()-t0:.1f}s")
    
    return labels_4d