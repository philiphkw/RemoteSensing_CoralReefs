"""
Lyzenga (1978) water column correction for multispectral satellite imagery.
Produces depth-invariant bottom indices by removing the effect of variable
water depth on reflectance.

Reference:
https://github.com/teongu/lyzenga1978/
https://doi.org/10.30536/ijreses.v2i.14076
"""

import gc

import numpy as np
import xarray as xr
import rioxarray
import scripts.config as config
from itertools import combinations
from sklearn.linear_model import LinearRegression


def dem_resampling(raw_bands, dem):
    """Resample DEM data to match AOI and resolution of satellite data"""

    print(" - Resampling DEM to match raw bands...")
    aoi = raw_bands['cb'].isel(time=0)

    # Extract bouding box of AOI
    minx = float(aoi.x.min())
    maxx = float(aoi.x.max())
    miny = float(aoi.y.min())
    maxy = float(aoi.y.max())

    # Clip DEM to match
    print("   - Clipping DEM to area of interest")
    dem_aoi = dem.rio.clip_box(minx, miny, maxx, maxy)

    # Mask outliers after resampling, not before
    print("   - Removing extreme outliers from DEM")
    dem_resampled = dem_aoi.copy().astype(float)
    dem_resampled.values[dem_resampled.values < config.DEM_OUTLIER_THRESHOLD] = np.nan

    # Resample DEM to match resolution of raw bands
    print("   - Matching DEM resolution to raw bands")
    dem_resampled = dem_resampled.rio.reproject_match(raw_bands['cb'])

    del dem_aoi
    gc.collect()

    return dem_resampled

def depth_regression_data(raw_bands, dem_resampled):
    """
    Takes the median reflectance per pixel across all time steps to reduce noise,
    then filters to water-only pixels using the DEM (elevation < 0). The resulting
    two arrays of band reflectance (regression_pixels) and known depth (depth_regression) 
    are passed to estimate_ki_kj_bathy() to estimate how strongly each band is
    attenuated by water depth, which is needed to compute the depth-invariant indices.
    """

    print(" - Preparing regression data for ki/kj estimation...")

    # Select pixels used in estimating ki/kj
    regression_pixels = {}
    for name in config.LYZENGA_BANDS:
        print(f"   - Collapsing band to one time slice: {name}")
        band_median = raw_bands[name].median(dim='time')
        regression_pixels[name] = band_median.values.reshape(-1).astype(float)

    dem_flat = dem_resampled.values[0].reshape(-1).astype(float)

    # Filter to valid pixels
    print("   - Creating valid pixel mask")
    valid_mask = np.isfinite(regression_pixels['cb']) & \
                np.isfinite(regression_pixels['blue']) & \
                np.isfinite(regression_pixels['green']) & \
                np.isfinite(dem_flat) & \
                (dem_flat < config.DEM_WATER_THRESHOLD)  # only water (negative elevation)
    
    # Filter all pixel arrays
    for name in config.LYZENGA_BANDS:
        print(f"   - Filtering for valid pixels in band: {name}")
        regression_pixels[name] = regression_pixels[name][valid_mask]

    # Filter DEM
    print("   - Filtering for valid pixels in bathymetry")
    depth_regression = dem_flat[valid_mask]

    del valid_mask, dem_flat
    gc.collect()

    return depth_regression, regression_pixels



# Step 1: Xi transformation
def compute_Xi(band: xr.DataArray, dw_value: float):
    """
    Apply the Lyzenga log transformation to a single band.

    Xi = ln(Bi - Bi_inf)

    where Bi_inf is the mean reflectance of optically deep water in band i.
    Pixels where (Bi - Bi_inf) <= 0 are set to NaN (physically invalid).
    """

    diff = band - dw_value
    diff = diff.where(diff > 0)          # mask non-positive values → NaN
    return xr.apply_ufunc(np.log, diff)



# Step 2a: Estimate ki/kj from bathymetry (if available)
def estimate_ki_kj_bathy(Xi: np.ndarray, Xj: np.ndarray, bathy: np.ndarray):
    """
    Estimate the attenuation ratio ki/kj by regressing Xi and Xj
    independently against known water depth values.

    Returns ki/kj = slope_i / slope_j.
    """
    valid = np.isfinite(Xi) & np.isfinite(Xj) & np.isfinite(bathy)

    def slope(X):
        reg = LinearRegression().fit(X[valid].reshape(-1, 1), bathy[valid])
        return reg.coef_[0]

    ki = slope(Xi)
    kj = slope(Xj)

    del valid
    gc.collect()

    return ki / kj


# Step 2b: Estimate ki/kj from homogeneous bottom (no bathymetry needed)
def estimate_ki_kj_covariance(Xi: np.ndarray, Xj: np.ndarray):
    """
    Estimate ki/kj using Lyzenga's covariance method — no bathymetry required.

    Over a homogeneous bottom type, depth variation causes Xi and Xj to vary
    together. The slope of Xi regressed on Xj estimates ki/kj directly.

    Pick a zone that is:
      - Shallow enough to contain bottom signal (not deep water)
      - Visually homogeneous in bottom type (e.g. all sand)
    """
    valid = np.isfinite(Xi) & np.isfinite(Xj)
    reg = LinearRegression().fit(Xj[valid].reshape(-1, 1), Xi[valid])
    coeff = reg.coef_[0]
    return coeff


# Step 3: Compute depth-invariant index for a band pair
def depth_invariant_index(Xi: xr.DataArray, Xj: xr.DataArray, ki_kj: float):
    """
    Compute the depth-invariant bottom index (DII) for a pair of bands.

    DII_ij = Xi - (ki/kj) * Xj

    This combination is perpendicular to the depth axis in (Xi, Xj) space,
    so depth variation cancels out while bottom-type variation is preserved.
    """

    return Xi - ki_kj * Xj


# Step 4: Full pipeline
def apply_lyzenga(raw_bands: dict,
                  dw_values: dict,
                  ki_kj_ratios: dict | None = None,
                  regression_pixels: dict | None = None,
                  use_bathy: bool = False,
                  bathy: np.ndarray | None = None):
    """
    Full Lyzenga correction pipeline. Produces one depth-invariant index 
    per pair of correctable bands (those listed in dw_values).
    """
    band_names = list(dw_values.keys())

    # Xi transform
    Xi_bands = {}
    for name in band_names:
        Xi_bands[name] = compute_Xi(raw_bands[name], dw_values[name])
        print(f"   - Xi computed for {name}")

    # ki/kj ratios
    if ki_kj_ratios is None:
        if regression_pixels is None:
            raise ValueError(
                "Provide either ki_kj_ratios or regression_pixels to estimate attenuation coefficients."
            )
        ki_kj_ratios = {}
        for name_i, name_j in combinations(band_names, 2):
            Xi_reg = np.log(
                np.maximum(regression_pixels[name_i] - dw_values[name_i], 1e-6)
            )
            Xj_reg = np.log(
                np.maximum(regression_pixels[name_j] - dw_values[name_j], 1e-6)
            )
            if use_bathy and bathy is not None:
                ratio = estimate_ki_kj_bathy(Xi_reg, Xj_reg, bathy)
            else:
                ratio = estimate_ki_kj_covariance(Xi_reg, Xj_reg)
            ki_kj_ratios[(name_i, name_j)] = ratio
            print(f"   - k{name_i}/k{name_j} = {ratio:.4f}")

    # Depth-invariant indices
    di_bands = {}
    for (name_i, name_j), ratio in ki_kj_ratios.items():
        di_name = f"di-{name_i}-{name_j}"
        di_bands[di_name] = depth_invariant_index(
            Xi_bands[name_i], Xi_bands[name_j], ratio
        )
        print(f"   - Depth-invariant index computed: {di_name}")
    
    del Xi_bands, regression_pixels 
    gc.collect()
    
    return di_bands