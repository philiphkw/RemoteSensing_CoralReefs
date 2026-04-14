This document is dedicated for citing the sources of external data.

# Satellite Imagery:
- **Source:** Planet Labs PlanetScope via [Planet Explorer](https://www.planet.com/explorer/)

- **Item Type:** `PSScene`
- **Asset Type:** `ortho_analytic_8b_sr` (Surface Reflectance, 8-band)
- **File naming convention:** `*_3B_AnalyticMS_SR_8b_clip_[researcher].tif`
- **File Location:** `data\raw\PSScene`
- **Metadata Location:** `data\raw\metadata`

- **Sensor / Instrument:** Super Dove (PS2.SD)
    - 3rd generation PlanetScope CubeSat constellation
    - Sun-synchronous orbit, ~180+ satellites

- **Spectral Bands (8-band):**
    | Band | Name          | Wavelength      |
    |------|---------------|-----------------|
    | 1    | Coastal Blue  | 431 – 452 nm    |
    | 2    | Blue          | 465 – 515 nm    |
    | 3    | Green I       | 513 – 549 nm    |
    | 4    | Green II      | 547 – 583 nm    |
    | 5    | Yellow        | 600 – 620 nm    |
    | 6    | Red           | 650 – 680 nm    |
    | 7    | Red Edge      | 697 – 713 nm    |
    | 8    | NIR           | 845 – 885 nm    |

- **Download Filters Applied:**
    - Imagery Type: PlanetScope Scene (PSScene)
    - Instrument: Super Dove (PS2.SD)
    - Publishing stage: Standard & Finalized
    - Area coverage: 100%
    - Cloud cover: ≤ 10%
    - Ground sample distance: 0.1 m – 5 m
    - Surface Reflectance: `TRUE`

- **Radiometric Processing:**
    - Surface reflectance product — atmospherically corrected from TOA reflectance
    - DN to reflectance conversion: `reflectance = DN / 10000`

- **Spatial Resolution:** ~3 m GSD
- **Temporal Resolution:** Near-daily revisit

&nbsp;

# Digital Elevation Model (DEM):
- **File Name:** `OTI_25cm_Q820_all_bathy_topo_DTM_refract_-1m_GM(bin,min,0.25,1).tif`

- **File Location:** `data\external\`

- **Source:** 
    - Harris, Daniel L; Webster, Jody M; Vila-Concejo, Ana; Duce, Stephanie; Leon, Javier X; Hacker, Jorg (2023): One Tree Reef topographic and bathymetric LiDAR digital elevation model (2018) and roughness equivalent habitat data (2023) [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.963918

- **Description:**
    - A high-resolution LiDAR digital elevation model (DEM) of the One Tree Island Reef in Australia.

- **Study Site**
    - Location: One Tree Reef, Southern Great Barrier Reef
    - Full reef coverage achieved

- **Data Collection**
    - Date: 8 October 2018
    - Platform: Diamond Aircraft ECO-Dimona (small research aircraft)
    - Sensors:
        - Riegl VQ-820-G topo-bathymetric LiDAR (primary, two pulse rate settings)
        - Riegl Q680i-S topographic LiDAR scanner
        - Canon EOS 5D Mk4 DSLR (RGB imagery)
    - Two pulse rate settings on VQ-820-G: 284 kHz (max depth penetration) and 522 kHz (max spatial resolution)

- **DEM Specifications**
    - Cell size: 0.25 m
    - Relative vertical error: ±0.1 m
    - Format: GeoTIFF

- **Imagery**
    - Camera: Canon EOS 5D Mk4
    - Mosaic resolution: 0.14 m cell size
    - Processing: Agisoft PhotoScan (now Metashape), overlaid on LiDAR point cloud

- **Processing Software**
    - Riegl proprietary software
    - ARA-developed software
    - RAPIDLASSO LAStools
    - Bayesmap StripAlign™
    - Global Mapper v20
    - Agisoft PhotoScan / Metashape

- **Surface Roughness Methods**
    - Vector Ruggedness Measure (VRM) — filter radii of 8, 20, 100, and 400 cells
    - Multiscale Roughness (MR) from WhiteBox Tools — filter radii of 1–1500 cells (~0.5–750 m), step interval of 1 cell

- **Output Data Products**
    - LiDAR DEM (GeoTIFF, 0.25 m)
    - Roughness magnitude and scale rasters (GeoTIFF, from MR tool)
    - Roughness equivalent habitat zones (ESRI Shapefiles)

- **Reference Classification**
    - Geomorphic zones based on Roelfsema et al. (2018), doi:10.1016/j.rse.2018.02.005

![reports/figures/dem_aoi_B(2022,2023)_k10_init3_MS_median_bathy_11F.png](reports\figures\dem_aoi_B(2022,2023)_k10_init3_MS_median_bathy_11F.png)