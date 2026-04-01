import os
from os.path import join as pjoin

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = pjoin(ROOT_DIR, 'data')
FIGURES_DIR = pjoin(ROOT_DIR, r'reports\figures')
MODELS_DIR = pjoin(ROOT_DIR, 'models')

FILE_PATTERN = r'raw\PSScene\*_3B_AnalyticMS_SR_8b_clip*.tif'
RESAMPLE_FREQ = 'MS'
RESAMPLE_AGG = 'median'
GMM_COMPONENTS = 10
GMM_N_INIT = 3
PERCENTILE_LOWER = 2
PERCENTILE_UPPER = 98
RUN_NAME = ""

# ── Lyzenga water column correction ─────────────────────────────────────────
# Mean reflectance values over an optically deep water zone (no bottom signal).
# Must be measured from your specific image — sample a deep, clear-water area.
# Only bands that penetrate water meaningfully are corrected:
#   cb (coastal blue) penetrates deepest, blue next, green shallowest.
#   Red and NIR are absorbed too quickly to carry bottom signal.
LYZENGA_BANDS = ['cb', 'blue', 'green']   # bands to apply correction to
 
LYZENGA_DW_VALUES = {
    'cb':    400,   # ← replace with your deep-water mean for coastal blue band
    'blue':  300,   # ← replace with your deep-water mean for blue band
    'green': 100,   # ← replace with your deep-water mean for green band
}
 
# Optional: pre-computed ki/kj ratios (set to None to estimate from data)
# Format: {('band_i', 'band_j'): ratio}
# Example: {('cb', 'blue'): 1.23, ('cb', 'green'): 1.87, ('blue', 'green'): 1.52}
LYZENGA_KI_KJ = None
 
# Path to a .npy file containing pixel values from a homogeneous-bottom
# calibration zone (used to estimate ki/kj if LYZENGA_KI_KJ is None).
LYZENGA_REGRESSION_ZONE = pjoin(ROOT_DIR, r'scripts\lyzenga.py')