import os
import joblib
import numpy as np
import xarray as xr
from os.path import join as pjoin
from scripts.config import DATA_DIR, MODELS_DIR

INTERIM_DIR = pjoin(DATA_DIR, 'interim')
os.makedirs(INTERIM_DIR, exist_ok=True)


# def cache_netcdf(name, compute_fn, force_recompute=False):
#     path = pjoin(INTERIM_DIR, f'{name}.nc')
    
#     if os.path.exists(path) and not force_recompute:
#         print(f"✓ Loading cached {name} from interim/")
#         ds = xr.open_dataset(path)
#         da = ds[list(ds.data_vars)[0]]
#         print(f"  Loaded as {type(da)} with dims {da.dims}")  # ← debug line
#         return da
    
#     print(f"  No cache found for {name}, computing...")
#     result = compute_fn()
#     print(f"  Saving as {type(result)} with dims {result.dims}")  # ← debug line
#     result.to_dataset(name=name).to_netcdf(path)
#     print(f"✓ Saved {name} to interim/")
#     return result


def cache_numpy(name, compute_fn, force_recompute=False):
    """Load from cache if exists, otherwise compute and save."""
    path = pjoin(INTERIM_DIR, f'{name}.npy')
    scaler_path = pjoin(INTERIM_DIR, f'{name}_scaler.pkl')
    mask_path = pjoin(INTERIM_DIR, f'{name}_mask.npy')
 
    if os.path.exists(path) and not force_recompute:
        print(f"✓ Loading cached {name} from interim/")
        data = np.load(path)
        mask = np.load(mask_path)
        scaler = joblib.load(scaler_path)
        return data, mask, scaler
    
    print(f"No cache found for {name}, computing...")
    data, mask, scaler = compute_fn()
    
    np.save(path, data)
    np.save(mask_path, mask)
    joblib.dump(scaler, scaler_path)
    
    print(f"✓ Saved {name} to interim/")
    return data, mask, scaler


def cache_model(name, compute_fn, force_recompute=False):
    """Load model from cache if exists, otherwise train and save."""
    path = pjoin(MODELS_DIR, f'{name}.pkl')
    
    if os.path.exists(path) and not force_recompute:
        print(f"✓ Loading cached {name} from models/")
        return joblib.load(path)
    
    print(f"No cache found for {name}, training...")
    result = compute_fn()
    joblib.dump(result, path)
    print(f"✓ Saved {name} to models/")
    return result

def cache_spatial(name, compute_fn, force_recompute=False):
    """Load spatial labels from cache if exist, otherwise compute and save."""
    path = pjoin(INTERIM_DIR, f'{name}_spatial.npy')
    
    if os.path.exists(path) and not force_recompute:
        print(f"✓ Loading cached spatial labels {name} from interim/")
        return np.load(path)
    
    print(f"No cache found for {name}, computing...")
    result = compute_fn()
    np.save(path, result)
    print(f"✓ Saved spatial labels {name} to interim/")
    return result


def cache_predictions(name, compute_fn, force_recompute=False):
    """Load predictions (labels + probs) from cache if exist, otherwise compute and save."""
    labels_path = pjoin(INTERIM_DIR, f'{name}_labels.npy')
    probs_path = pjoin(INTERIM_DIR, f'{name}_probs.npy')
    
    if os.path.exists(labels_path) and os.path.exists(probs_path) and not force_recompute:
        print(f"✓ Loading cached predictions {name} from interim/")
        labels = np.load(labels_path)
        probs = np.load(probs_path)
        return labels, probs
    
    print(f"No cache found for {name}, computing...")
    labels, probs = compute_fn()
    np.save(labels_path, labels)
    np.save(probs_path, probs)
    print(f"✓ Saved predictions {name} to interim/")
    return labels, probs