import os
import joblib
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from os.path import join as pjoin
import scripts.config as config

INTERIM_DIR = pjoin(config.DATA_DIR, 'interim')
os.makedirs(INTERIM_DIR, exist_ok=True)

def cache_bands(name, compute_fn, force_recompute=False):
    """Cache a dict of xarray DataArrays as numpy arrays + coordinates."""
    bands_dir = pjoin(config.INTERIM_DIR, name)

    band_names = ['bg', 'blue', 'brightness', 'cb', 'green', 'ndavi', 'nir', 'red', 'yellow']
    lyzenga_bands = [b for b in band_names if b not in ['bg', 'brightness']] + ['brightness-di', 'di-blue-green', 'di-cb-blue', 'di-cb-green']
    selected_bands = lyzenga_bands if config.LYZENGA_ALG else band_names

    missing = [b for b in selected_bands if not os.path.exists(pjoin(bands_dir, f'{b}.npy'))]

    if os.path.exists(bands_dir) and not force_recompute and not missing:
        print(f"[EXISTS] Loading cached bands {name} from interim/")
        coords = np.load(pjoin(bands_dir, '_coords.npz'), allow_pickle=True)
        time = coords['time']
        y    = coords['y']
        x    = coords['x']
        crs  = str(coords['crs']) if 'crs' in coords else None
        bands = {}
        for b in tqdm(selected_bands, desc=" - Loading bands"):
            arr = np.load(pjoin(bands_dir, f'{b}.npy'))
            da  = xr.DataArray(arr, dims=['time', 'y', 'x'],
                               coords={'time': time, 'y': y, 'x': x})
            if crs:
                da.rio.write_crs(crs, inplace=True)
            bands[b] = da
        return bands

    if missing:
        print(f"[MISSING] Missing bands: {missing}, recomputing...")
    else:
        print(f"[MISSING] No cache found for {name}, computing...")

    bands = compute_fn()
    os.makedirs(bands_dir, exist_ok=True)

    first = next(iter(bands.values()))
    np.savez(pjoin(bands_dir, '_coords.npz'),
             time=first.time.values,
             y=first.y.values,
             x=first.x.values,
             crs=np.array(str(first.rio.crs)))

    for key, da in tqdm(bands.items(), desc=" - Saving bands"):
        np.save(pjoin(bands_dir, f'{key}.npy'), da.values)
    print(f"   - Saved all bands to interim/{name}/")
    return bands


def cache_data(name, compute_fn, force_recompute=False):
    """Load from cache if exists, otherwise compute and save."""
    path = pjoin(INTERIM_DIR, f'{name}.npy')
    scaler_path = pjoin(INTERIM_DIR, f'{name}_scaler.pkl')
    mask_path = pjoin(INTERIM_DIR, f'{name}_mask.npy')
 
    if os.path.exists(path) and not force_recompute:
        print(f"[EXISTS] Loading cached {name} from interim/")
        data = np.load(path)
        mask = np.load(mask_path)
        scaler = joblib.load(scaler_path)
        return data, mask, scaler
    
    print(f"[MISSING] No cache found for {name}, computing...")
    data, mask, scaler = compute_fn()
    
    np.save(path, data)
    np.save(mask_path, mask)
    joblib.dump(scaler, scaler_path)
    
    print(f" - Saved {name} to interim/")
    return data, mask, scaler


def cache_model(name, compute_fn, force_recompute=False):
    """Load model from cache if exists, otherwise train and save."""
    path = pjoin(config.MODELS_DIR, f'{name}.pkl')
    
    if os.path.exists(path) and not force_recompute:
        print(f"[EXISTS] Loading cached {name} from models/")
        return joblib.load(path)
    
    print(f"[MISSING] No cache found for {name}, training...")
    result = compute_fn()
    joblib.dump(result, path)
    print(f" - Saved {name} to models/")
    return result


def cache_spatial(name, compute_fn, force_recompute=False):
    """Load spatial labels from cache if exist, otherwise compute and save."""
    path = pjoin(INTERIM_DIR, f'{name}_spatial.npy')
    
    if os.path.exists(path) and not force_recompute:
        print(f"[EXISTS] Loading cached spatial labels {name} from interim/")
        return np.load(path)
    
    if os.path.exists(path) and force_recompute:
        os.remove(path)
        print(f"[DELETED] {name}")
    
    print(f"[MISSING] No cache found for {name}, computing...")
    result = compute_fn()
    np.save(path, result)
    print(f" - Saved spatial labels {name} to interim/")
    return result


def cache_predictions(name, compute_fn, model_type, force_recompute=False):
    """Load predictions (labels and/or probs) from cache if exist, otherwise compute and save."""
    if model_type == "gmm":
        labels_path = pjoin(INTERIM_DIR, f'{name}_labels.npy')
        probs_path = pjoin(INTERIM_DIR, f'{name}_probs.npy')

        if os.path.exists(labels_path) and os.path.exists(probs_path) and not force_recompute:
            print(f"[EXISTS] Loading cached predictions {name} from interim/")
            labels = np.load(labels_path)
            probs = np.load(probs_path)
            return labels, probs
        
        print(f"[MISSING] No cache found for {name}")
        labels, probs = compute_fn()
        np.save(labels_path, labels)
        np.save(probs_path, probs)
        print(f" - Saved predictions {name} to interim/")
        return labels, probs
    
    elif model_type == "kmeans":
        labels_path = pjoin(INTERIM_DIR, f'{name}_labels.npy')

        if os.path.exists(labels_path) and not force_recompute:
            print(f"[EXISTS] Loading cached predictions {name} from interim/")
            labels = np.load(labels_path)
            return labels
        
        print(f"[MISSING] No cache found for {name}")
        labels = compute_fn()
        np.save(labels_path, labels)
        print(f" - Saved predictions {name} to interim/")
        return labels
    
    else:
        raise ValueError("[ERROR] Please provide a valid model_type ('gmm' or 'kmeans')")

def cache_spatial_timeseries(name, compute_fn, timestamps, ref_band, force_recompute=False):
    """
    Load a spatial timeseries from cache if it exists, otherwise compute and save.
    
    Stores:
      - A single .npy file (num_timesteps, x, y) for fast reloading
      - Individual per-timestep GeoTIFFs in interim/<name>/
    """

    npy_path = pjoin(INTERIM_DIR, f'{name}_spatial.npy')
    tifs_dir = pjoin(config.INTERIM_DIR, name)
    os.makedirs(tifs_dir, exist_ok=True)

    if os.path.exists(npy_path) and not force_recompute:
        print(f"[EXISTS] Loading cached spatial timeseries {name} from interim/")
        arr = np.load(npy_path)
    else:
        if os.path.exists(tifs_dir) and force_recompute:
            for f in os.listdir(tifs_dir):
                try:
                    os.remove(pjoin(tifs_dir, f))
                except PermissionError:
                    print(f"[WARNING] Could not delete {f} — file may be open in another program")
            print(f"[DELETED] contents of {name}")
        else:
            print(f"[MISSING] No cache found for {name}")

        arr = compute_fn()
        np.save(npy_path, arr)
        print(f" - Saved {name} to interim/")

    # Write any missing per-month GeoTIFFs
    missing = 0
    for t_idx, ts in enumerate(timestamps):
        ts_str = pd.Timestamp(ts).strftime('%Y%m')
        tif_path = pjoin(tifs_dir, f'{name}_{ts_str}.tif')
        if not os.path.exists(tif_path) or force_recompute:
            spatial = np.flipud(arr[t_idx].astype(np.float32))
            da = xr.DataArray(spatial, dims=['y', 'x'])
            da.rio.write_crs(ref_band.rio.crs, inplace=True)
            da.rio.write_transform(inplace=True)
            da.rio.to_raster(tif_path)
            missing += 1

    if missing:
        print(f" - Saved {missing} GeoTIFFs to interim/{name}/")
    else:
        print(f"[EXISTS] All GeoTIFFs already exist in models/{name}/")

    return arr