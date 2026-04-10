import os
from os.path import join as pjoin

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = pjoin(ROOT_DIR, 'data')
FIGURES_DIR = pjoin(ROOT_DIR, r'reports\figures')
MODELS_DIR = pjoin(ROOT_DIR, 'models')
INTERIM_DIR = pjoin(DATA_DIR, 'interim')
EXTERNAL_DIR = pjoin(DATA_DIR, 'external')
DEM_FILE = "OTI_25cm_Q820_all_bathy_topo_DTM_refract_-1m_GM(bin,min,0.25,1).tif"



# General Parameters
RESAMPLE_FREQ = 'MS'
RESAMPLE_AGG = 'median'
N_SAMPLES = None        # Samples a specific number of data points from training data.
PERC_SAMPLE = None      # Samples a percentage of the training data. Overwrites N_SAMPLES.
N_CLUSTERS = 10
GMM_N_INIT = 3
KMEANS_K = 10
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 20
RANDOM_STATE = 42
PERCENTILE_LOWER = 2
PERCENTILE_UPPER = 98
TRAINING_YEAR = [2022, 2023]
LYZENGA_ALG = True      # Toggles whether Lyzenga Algorithm is used in preprocessing



# Toggle which pipelines are forced to ignore cached file and rerun
FORCE_RECOMPUTE = False
if FORCE_RECOMPUTE == False:
    # For manual toggling
    FORCE_BANDS         = False
    FORCE_TRAINING      = False
    FORCE_MONTHLY       = False
    FORCE_GMM           = False
    FORCE_KMEANS        = False
    FORCE_PREDICTIONS   = False
    FORCE_SPATIAL       = False
else:
    FORCE_BANDS         = True
    FORCE_TRAINING      = True
    FORCE_MONTHLY       = True 
    FORCE_GMM           = True
    FORCE_KMEANS        = True
    FORCE_PREDICTIONS   = True 
    FORCE_SPATIAL       = True



# Parameters for Validating Clusters
SAMPLE_SIZE_BOOT = 100000
N_BOOT = 500
SAMPLE_SIZE_BOOT_ARI = 10000
N_BOOT_ARI = 50



# Standardized naming scheme
bathy_suffix = "_bathy" if LYZENGA_ALG else ""
YEAR_STRING = ",".join(str(y) for y in TRAINING_YEAR)
RUN_NAME = f"B({YEAR_STRING})_k{N_CLUSTERS}_{RESAMPLE_FREQ}_{RESAMPLE_AGG}{bathy_suffix}"
TRAINING_NAME = f'baseline_data_{RUN_NAME}'
MONTHLY_NAME = f'monthly_data_{RUN_NAME}'
GMM_NAME = f'gmm_{RUN_NAME}'
KMEANS_NAME = f'kmeans_{RUN_NAME}'
PREDICTIONS_NAME = f'predictions_monthly_{RUN_NAME}'
SPATIAL_NAME = f'labels_monthly_{RUN_NAME}'



# Visualization parameters
ROLLING_WINDOW = 3
X_MINMAX = ('2024-01', '2024-05')


# Lyzenga Algorithm Parameters
DEM_WATER_THRESHOLD = 0
DEM_OUTLIER_THRESHOLD = -3000
LYZENGA_BANDS = ['cb', 'blue', 'green']   # bands to apply correction to
LYZENGA_DW_VALUES = {'cb': 400, 'blue': 300, 'green': 100}
LYZENGA_KI_KJ = None
LYZENGA_REGRESSION_ZONE = pjoin(ROOT_DIR, r'scripts\lyzenga.py')