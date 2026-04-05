import os
os.environ['USE_PYGEOS'] = '0'
path = os.getcwd()
import sys
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import rioxarray
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')


# Get project root to allow for relative imports
script_dir = Path(os.getcwd())
project_root = script_dir.parent

print(f"Project root: {project_root}")

# Add PROJECT ROOT to sys.path
sys.path.insert(0, str(project_root))

import config


gmm = joblib.load(".//models//gmm_B(2022,2023)_k10_init3_MS_median_bathy_11F.pkl")
tiff_path = ".//data//raw//PSScene//20240607_000749_84_24e5_3B_AnalyticMS_SR_8b_clip_philip.tif"
band = rioxarray.open_rasterio(tiff_path)
data_monthly = np.load(".//data//interim//monthly_data_B(2022,2023)_k10_init3_MS_median_bathy_11F.npy")
labels_monthly = np.load(".//data//interim//predictions_monthly_B(2022,2023)_k10_init3_MS_median_bathy_11F_labels.npy")
mask_monthly = np.load(".//data//interim//monthly_data_B(2022,2023)_k10_init3_MS_median_bathy_11F_mask.npy")

# Checking the amount of pixels per picture
size_x, size_y = band[1].shape[0], band[1].shape[1]
pixel_per_timestamp = size_x * size_y 

# Sanity check to ensure that the amount of data can be divided over 48 months
len(mask_monthly)/pixel_per_timestamp == 48.0

"""
Silhouette score measures how well each point fits its cluster vs neighbouring clusters.
Score ranges from -1 to 1:
    1.0  = perfect separation
    0.0  = overlapping clusters  
    -1.0  = points assigned to wrong cluster
"""

def bootstrap(sample_size, data, gmm):
    """Idea is that a mean proportion per cluster per month
    is computed. Then the Bootstrap CI95 will be computed
    as well as the stdev. 

    Returns:
        _type_: _description_
    """
    random_index = np.random.randint(len(data), size=sample_size)
    bootstrap_data = data[random_index]
    pred_boot = gmm.predict(bootstrap_data)
    pred_counts = np.bincount(pred_boot)
    props = pred_counts / pred_counts.sum()
    return props


start_index = 0
dict_info = defaultdict(list)
for t in range(48):

    mask_true = mask_monthly[t*pixel_per_timestamp:(t+1)*pixel_per_timestamp]
    valid_pixels = len(mask_true[mask_true])
    data = data_monthly[start_index:start_index + valid_pixels]
    labels = labels_monthly[start_index:start_index + valid_pixels]

    # Using all data for calculations was too slow, therefore a subsample is used
    sil_score = silhouette_score(data, labels, sample_size=config.SAMPLE_SIZE_BOOT, random_state=35)

    with ThreadPoolExecutor() as executor:
        proportions = list(
            executor.map(
                bootstrap,
                [config.SAMPLE_SIZE_BOOT] * config.N_BOOT,
                [data] * config.N_BOOT,
                [gmm] * config.N_BOOT
            )
        )

    proportions = np.array(proportions)
    mean  = proportions.mean(axis=0)
    stdev = proportions.std(axis=0)

    # Computes the bootstrap confidence interval
    lower = np.percentile(proportions, 2.5, axis=0)
    upper = np.percentile(proportions, 97.5, axis=0)

    dict_info["time"].append(t)
    dict_info["mean"].append(mean)
    dict_info["stdev"].append(stdev)
    dict_info["lower"].append(lower)
    dict_info["upper"].append(upper)
    dict_info["silhouette_score"].append(sil_score)

    start_index += valid_pixels
    # silhouette_scores.append(sil_score)

# Save to parquet as this helps maintain column information (which is a list)
pd.DataFrame(dict_info).to_parquet(".//data//processed//validation_results//bootstrap_mean_results.parquet")



# dict_ari = defaultdict(list)

# def bootstrap_ari(data, labels):
#     random_index = np.random.randint(len(data), size=10000)

#     bootstrap_data = data[random_index]
#     gmm_boot = GaussianMixture(n_components=config.GMM_COMPONENTS, random_state=35, n_init=config.GMM_N_INIT)
#     gmm_boot.fit(bootstrap_data)

#     # Predict labels for the original month's data
#     y_boot = gmm_boot.predict(data)

#     return adjusted_rand_score(labels, y_boot)

# os.environ["LOKY_MAX_CPU_COUNT"] = "4"
# n_boot_ari = 50
# start_index = 0

# for t in range(2):
#     mask_true = mask_monthly[t*pixel_per_timestamp:(t+1)*pixel_per_timestamp]
#     valid_pixels = len(mask_true[mask_true])
#     data = data_monthly[start_index:start_index + valid_pixels]
#     labels = labels_monthly[start_index:start_index + valid_pixels]

#     with ThreadPoolExecutor() as executor:
#         ari_scores = list(
#             executor.map(
#                 bootstrap_ari,
#                 [data] * n_boot_ari,
#                 [labels] * n_boot_ari,
#             )
#         )
#     print(ari_scores)
#     ari_scores = np.array(ari_scores)
#     mean  = ari_scores.mean(axis=0)
#     stdev = ari_scores.std(axis=0)
#     dict_ari["time"].append(t)
#     dict_ari["mean"].append(mean)
#     dict_ari["stdev"].append(stdev)
#     start_index += valid_pixels

# pd.DataFrame(dict_ari).to_parquet(".//data//processed//validation_results//ari_results.csv")


# print("Scripts done")


