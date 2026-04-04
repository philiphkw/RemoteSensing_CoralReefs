import os
from os.path import join as pjoin

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = pjoin(ROOT_DIR, 'data')
FIGURES_DIR = pjoin(ROOT_DIR, r'reports\figures')
MODELS_DIR = pjoin(ROOT_DIR, 'models')
INTERIM_DIR = pjoin(DATA_DIR, 'interim')
EXTERNAL_DIR = pjoin(DATA_DIR, 'external')
DEM_FILE = "OTI_25cm_Q820_all_bathy_topo_DTM_refract_-1m_GM(bin,min,0.25,1).tif"


RESAMPLE_FREQ = 'MS'
RESAMPLE_AGG = 'median'
N_SAMPLES = None
GMM_COMPONENTS = 10
GMM_N_INIT = 3
PERCENTILE_LOWER = 2
PERCENTILE_UPPER = 98
NO_DATA_VALUE = 0


BASELINE_YEAR = [2022, 2023]
LYZENGA_ALG = True
bathy_suffix = "_bathy" if LYZENGA_ALG else ""
FORCE_RECOMPUTE = False


if FORCE_RECOMPUTE == False:
    # For manual toggling
    FORCE_BANDS         = False
    FORCE_BASELINE      = False
    FORCE_MONTHLY       = False
    FORCE_GMM           = False
    FORCE_PREDICTIONS   = False
    FORCE_SPATIAL       = False
else:
    FORCE_BANDS         = True
    FORCE_BASELINE      = True
    FORCE_MONTHLY       = True 
    FORCE_GMM           = True
    FORCE_PREDICTIONS   = True 
    FORCE_SPATIAL       = True

YEAR_STRING = ",".join(str(y) for y in BASELINE_YEAR)
RUN_NAME = f"B({YEAR_STRING})_k{GMM_COMPONENTS}_init{GMM_N_INIT}_{RESAMPLE_FREQ}_{RESAMPLE_AGG}{bathy_suffix}"
BASELINE_NAME = f'baseline_data_{RUN_NAME}'
MONTHLY_NAME = f'monthly_data_{RUN_NAME}'
GMM_NAME = f'gmm_{RUN_NAME}'
PREDICTIONS_NAME = f'predictions_monthly_{RUN_NAME}'
SPATIAL_NAME = f'labels_monthly_{RUN_NAME}'


# Lyzenga water column correction
DEM_WATER_THRESHOLD = 0
DEM_OUTLIER_THRESHOLD = -3000
LYZENGA_BANDS = ['cb', 'blue', 'green']   # bands to apply correction to
LYZENGA_DW_VALUES = {'cb': 400, 'blue': 300, 'green': 100}
LYZENGA_KI_KJ = None
LYZENGA_REGRESSION_ZONE = pjoin(ROOT_DIR, r'scripts\lyzenga.py')