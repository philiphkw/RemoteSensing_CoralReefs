import gc

import pandas as pd
import rioxarray
import xarray as xr
from glob import glob
from tqdm import tqdm
from os.path import join as pjoin
import scripts.config as config
from scripts.lyzenga import dem_resampling, depth_regression_data, apply_lyzenga

def load_stack():
    files = sorted(glob(pjoin(config.DATA_DIR, config.FILE_PATTERN)))
    dates = [pd.Timestamp(f.split('\\')[-1][:8]) for f in files]

    print(" - Loading and stacking data")
    arrays = []
    for f, date in tqdm(zip(files, dates), total=len(files), desc="   - "):
        da = rioxarray.open_rasterio(f)
        da = da.assign_coords(time=date)
        arrays.append(da)

    return xr.concat(arrays, dim='time')


def extract_bands(stack):
    print(" - Extracting raw bands")
    band_dict = {1: "cb", 2: "blue", 4: "green", 5: "yellow", 6: "red", 8: "nir"}
    raw_bands = {}
    for i, band_name in tqdm(zip(band_dict.keys(), band_dict.values()), total=len(band_dict), desc="   - "):
        raw_bands[band_name] = stack.sel(band=i).astype(float)

    return raw_bands


def load_dem():
    print(" - Loading DEM...")
    dem_path = pjoin(config.DATA_DIR, config.DEM_FILE)
    dem_file = rioxarray.open_rasterio(dem_path).squeeze().astype(float)
    return dem_file


def calculate_indices(raw_bands, di_bands=None):
    print(" - Calculating spectral indices")
    cb    = raw_bands["cb"]
    nir   = raw_bands["nir"]

    # NDAVI stays on raw bands — NIR doesn't penetrate water
    ndavi = (nir - cb) / (nir + cb + 1e-8)
    indices = {"ndavi": ndavi}
    print("   - Calculated: NDAVI")

    if di_bands is not None:
        # Brightness from DI bands = depth-invariant bottom reflectance intensity
        di_vals = list(di_bands.values())
        brightness_di = sum(di_vals) / len(di_vals)
        indices["brightness-di"] = brightness_di
        print("   - Calculated: Brightness-DI")
        # blue_green is dropped — di_cb_green already captures this more cleanly

        del di_vals, brightness_di 
        gc.collect()

    else:
        # Fallback if no Lyzenga correction applied
        green = raw_bands["green"]
        blue_green = cb / (green + 1e-8)
        indices["bg"] = blue_green
        print("   - Calculated: Blue-Green Ratio")
        brightness = (cb + raw_bands["blue"] + green) / 3
        indices["brightness"] = brightness
        print("   - Calculated: Brightness (Cb, Blue, Green)")

        del green, blue_green, brightness
        gc.collect()

    del cb, nir, ndavi
    gc.collect()
    
    return indices


def compute_bands_all(lyzenga=True):
    stack = load_stack()
    raw_bands = extract_bands(stack)
    del stack
    gc.collect()

    if lyzenga == True:
        dem = rioxarray.open_rasterio(pjoin(config.DATA_DIR, rf'external\{config.DEM_FILE}'))
        dem_resampled = dem_resampling(raw_bands, dem)

        del dem
        gc.collect()

        depth_regression, regression_pixels = depth_regression_data(raw_bands, dem_resampled)

        del dem_resampled
        gc.collect()

        print(" - Applying Lyzenga with full bathymetry calibration...")
        di_bands = apply_lyzenga(
            raw_bands=raw_bands,
            dw_values=config.LYZENGA_DW_VALUES,
            ki_kj_ratios=None,
            regression_pixels=regression_pixels,
            use_bathy=True,
            bathy=depth_regression
        )

        del depth_regression, regression_pixels
        gc.collect()

        indices = calculate_indices(raw_bands, di_bands=di_bands)

        bands_all = {**raw_bands, **di_bands, **indices}

        del raw_bands, di_bands, indices
        gc.collect()

        return bands_all
    
    else:
        indices = calculate_indices(raw_bands, di_bands=None)

        bands_all = {**raw_bands, **indices}

        del raw_bands, indices
        gc.collect()

        return bands_all