import pandas as pd
import rioxarray
import xarray as xr
from glob import glob
from tqdm import tqdm
from os.path import join as pjoin
import scripts.config as config


def load_stack():
    files = sorted(glob(pjoin(config.DATA_DIR, config.FILE_PATTERN)))
    dates = [pd.Timestamp(f.split('\\')[-1][:8]) for f in files]

    print("Loading and stacking data...")
    arrays = []
    for f, date in tqdm(zip(files, dates), total=len(files)):
        da = rioxarray.open_rasterio(f)
        da = da.assign_coords(time=date)
        arrays.append(da)

    return xr.concat(arrays, dim='time')


def extract_bands(stack):
    print("Extracting raw bands...")
    band_dict = {1: "cb", 2: "blue", 4: "green", 5: "yellow", 6: "red", 8: "nir"}
    raw_bands = {}
    for i, band_name in tqdm(zip(band_dict.keys(), band_dict.values()), total=len(band_dict)):
        raw_bands[band_name] = stack.sel(band=i).astype(float)

    return raw_bands

def calculate_indices(raw_bands, di_bands=None):
    print("Calculating spectral indices...")
    cb    = raw_bands["cb"]
    nir   = raw_bands["nir"]

    # NDAVI stays on raw bands — NIR doesn't penetrate water
    ndavi = (nir - cb) / (nir + cb + 1e-8)

    indices = {"ndavi": ndavi}

    if di_bands is not None:
        # Brightness from DI bands = depth-invariant bottom reflectance intensity
        di_vals = list(di_bands.values())
        brightness_di = sum(di_vals) / len(di_vals)
        indices["brightness_di"] = brightness_di
        # blue_green is dropped — di_cb_green already captures this more cleanly
    else:
        # Fallback if no Lyzenga correction applied
        green      = raw_bands["green"]
        blue_green = cb / (green + 1e-8)
        brightness = (cb + raw_bands["blue"] + green) / 3
        indices["bg"] = blue_green
        indices["brightness"] = brightness

    return indices