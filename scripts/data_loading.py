import pandas as pd
import rioxarray
import xarray as xr
from glob import glob
from os.path import join as pjoin
import scripts.config as config


def load_stack():
    files = sorted(glob(pjoin(config.DATA_DIR, config.FILE_PATTERN)))
    dates = [pd.Timestamp(f.split('\\')[-1][:8]) for f in files]

    print("Loading and stacking data...")
    arrays = []
    for f, date in zip(files, dates):
        da = rioxarray.open_rasterio(f)
        da = da.assign_coords(time=date)
        arrays.append(da)

    return xr.concat(arrays, dim='time')


def extract_bands(stack):
    print("Extracting bands and calculating indices...")
    cb  = stack.sel(band=1).astype(float)  # coastal blue
    blue = stack.sel(band=2).astype(float)
    green = stack.sel(band=4).astype(float)
    yellow = stack.sel(band=5).astype(float)
    red = stack.sel(band=6).astype(float)
    nir = stack.sel(band=8).astype(float)

    raw_bands = {"cb": cb, "blue": blue, "green": green, "yellow": yellow, "red": red, "nir": nir}
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