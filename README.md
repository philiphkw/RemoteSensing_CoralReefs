# RemoteSensing_CoralReefs

## Introduction

Measuring change in One Tree Island Reef from 2023 to 2025 following the extreme 2024 bleaching event using atmospheric remote sensing data. Three spectral indices are compared based on their ability to distinguish healthier reef areas from bleached reef areas using unsupervised machine learning techniques.

> NOTE: To run the full pipeline in models_notebook.ipynb, a device with a minimum of 16GB of RAM and a minimum of 45GB of local storage is required. 


## Data & Methods

### Data
For the full details on the data used in this research, please refer to [DATA_SOURCES.md](DATA_SOURCES.md). 

The raw satellite data is found in [data/raw/PSScene](data/raw/PSScene). Each file contains a suffix with the name of the researcher responsible for downloading that particular instance of the Area of Interest (AOI). Because multiple researchers have downloaded multiple instances of this AOI, we  also have multiple metadata files. These metadata files were compiled into one coherent file which you can find in [data/processed/metadata_merged](data/processed/metadata_merged).

The temporal boundaries are between January 2022 and December 2025 with each month containing on average two observations. 

Imagery was acquired by ‘SuperDove’ tool which were preprocessed according to PlanetScope-Ortho-Analytic-8B-SR, resulting in images that are orthorectified, and display scaled Surface Reflectance comparable to Landsat images (Moon et al., 2021; Planet.com, 2025). 

Extreme values were reduced using percentile clipping at the 2nd and 98th percentiles. As varying water depths influence the absorption of different wavelengths unevenly, the spectral bands of Coastal blue, blue and green were corrected for water depth using the Lyzenga algorithm (Lyzenga, 1978). For more details on the Lyzenga Algorithm, please refer to [lyzenga_alg_details.md](reports/Lyzenga_alg_details.md)

### Method

<img src="reports/figures/methodology-flowchart.jpg" />

## Results & Report
- The general results of this research can be found in [reports/figures](reports/figures). 
- The model validation results can be found in [data/processed/validation_results](data/processed/validation_results). 
- The final research report can be found at [reports/Coral-Reefs-Unusp-ML.pdf](reports/Coral-Reefs-Unusp-ML.pdf). 

## How to Use

### Notebooks
- [exploratory_notebook.ipynb](notebooks/exploratory_notebook.ipynb) - Experimental notebook with initial exploratory code.
- [models_notebook.ipynb](notebooks/models_notebook.ipynb) - Finalized notebook with the entire pipeline as visualized in the flowchart, excluding validation results. 
- [validation_results.ipynb](notebooks/validation_results.ipynb) - Notebook containing cluster validation results.

### Scripts 
- [cache.py](scripts/cache.py) - For saving and reloading intermediary files and models to avoid having to rerun time-consuming pipelines
- [config.py](scripts/config.py) - For grouping different (hyper)parameters in one location for improved oversight and efficient changing.
- [data_loading.py](scripts/data_loading.py) - Functions for the preprocessing pipeline including loading raw data, extracting raw bands, applying lyzenga algorithm, calculating indices, and returning a stacked array.
- [features.py](scripts/features.py) - For engineering and final preprocess features before modeling.
- [lyzenga.py](scripts/lyzenga.py) - For functions for the lyzenga algorithm pipeline.
- [model.py](scripts/model.py) - For training, predicting, and evaluating GMM and Kmeans models.
- [validation_GMM.py](scripts/validation_GMM.py) - For validating GMM clustering results.
- [validation_KMeans.py](scripts/validation_KMeans.py) - For validating GMM clustering results.
- [visualization.py](scripts/visualization.py) - For plotting, visualizing, and saving results.
- [extra/](scripts/extra)
  - [compiling_metadata.py](scripts/extra/compiling_metadata.py) - Script used to compile general metadata into one file.
  - [scenes_collection.py](scripts/extra/scenes_collection.py) - Script used to compile metadata about the PSScenes into one file.

### File Naming Scheme
The results of this research has been stored in [reports/figures](reports/figures). The file names of these results contain the metadata about what type of run it was obtained from. The naming scheme follows the following structure:

1. `General Info`
2. `Model` - Either `gmm` or `kmeans`
3. `Training Year(s)` - 'B' stands for 'Baseline
4. `Number of clusters`
5. `Temporal Aggregation` - Whether the data is aggregated by month (`MS`) or year (`YS`)
6. `Data Aggregation` - How the temporal data is aggregated (using `medain`, `mean`, `max`, `min`, etc.)
7. `Bathy` - Whether it has been corrected for depth using bathymetric data

Final result:

`file_name = <1>_<2>_<3>_<4>_<5>_<6>_<7>.ext`

