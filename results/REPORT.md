# Seasonal wetland habitat change at Ichkeul National Park, 2017–2026

*Sentinel-2 composites, geospatial foundation models and rainfall attribution for a UNESCO World Heritage wetland in northern Tunisia*

Emna Saada · October 2026

## Abstract

Ichkeul's lake and marshes depend on a freshwater supply that upstream dams have cut since the 1980s. I built 20 seasonal Sentinel-2 composites (wet Jan–Mar and dry Jul–Sep, 2017–2026) over a 167 km² basin envelope and labelled eight habitat classes with weak labels from agreeing open datasets. On spatially held-out test chips the best model so far, U-Net (ResNet-34, random init, CPU), reaches mIoU 0.573 against weak labels, versus 0.514 for a Random Forest on spectral indices, and its maps drive the change analysis. **Open-water area follows rainfall**: season plus antecedent and previous-year rain explain R² = 0.91, and a time trend adds only 0.3 % (p = 0.462). **Summer marsh area fell by 18 %** between 2017–2019 and 2024–2026, and every classifier with full-area maps shows a decline (Random Forest · spectral indices: -0.96 km²/yr, p = 0.002; U-Net (ResNet-34, random init, CPU): -0.43 km²/yr, p = 0.074). Rainfall explains almost none of the marsh variation (unique R² 0.000 and 0.009), which points to non-climatic pressure (conversion at the marsh edge, grazing, reduced freshwater flushing). The trend is significant at p < 0.05 for 1 of 2 classifiers. A 40-year Landsat series (agreement with Sentinel-2: r = 0.99 over 10 seasons) puts 2017–2026 mean winter lake water at 91.9 km², below each earlier decade.

## 1. Questions

1. Do geospatial foundation models (Prithvi-EO-2.0, TerraMind) beat Random Forest, AlphaEarth-embedding Random Forest and U-Net baselines when labels are scarce and weak?
2. How have open water, freshwater marsh, halophytic marsh and saline flats changed in 2017–2026, by season?
3. How much of the change follows rainfall, and how much is left for non-climatic drivers?
4. How does the decade compare with the 40-year Landsat record that starts at the time the dams were built?

## 2. Study area and data

**Area.** A 22.4 × 17.9 km window (401 km²) on a 10 m UTM 32N grid (EPSG:32632) centred on 37.15° N, 9.665° E. It covers Lake Ichkeul, the Garaet Ichkeul marshes, Djebel Ichkeul, the surrounding Mateur–Tinja plain and, in the north-east corner, a strip of the Bizerte lagoon. The grid is snapped to the Sentinel-2 20 m lattice so that every band is read without resampling offsets. Statistics are reported inside a *basin envelope*: the JRC-flooded component that contains the lake (opened by 4 pixels to cut the Oued Tinja link to the Bizerte lagoon, hole-filled), buffered by 1 km, with the lagoon removed. The official WDPA boundary could not be downloaded from the processing environment.

**Imagery.** Sentinel-2 L2A Cloud-Optimised GeoTIFFs from the AWS open-data bucket (`sentinel-cogs`, MGRS tile 32SNG), read only inside the study window. Jan–Feb 2018 exists only as previews in that archive, so the ESA SAFE products on Google Cloud (`gcp-public-data-sentinel-2`, JPEG2000) fill seasons with fewer than 12 complete scenes. Surface reflectance uses each asset's scale and offset, except where the provider flags the processing-baseline offset as already removed (`earthsearch:boa_offset_applied`). Applying it twice gave negative water reflectance from mid-2018 onwards, and the deep-lake SWIR check now gives 0.003–0.026 in every season.

**Compositing.** For each year 2017–2026, a *wet* (Jan–Mar) and a *dry* (Jul–Sep) season. Scenes are ranked by their clear fraction inside the window, from the Scene Classification Layer: vegetation, bare, water, dark and unclassified are kept, while clouds, cirrus and shadows are removed with a 60 m dilation. Up to 10 scenes with ≥ 35 % clear pixels are used per season, and the per-pixel median of valid observations forms the composite. The three clearest scenes are also kept as single dates (gaps filled from the median) for the multi-temporal model.

**Auxiliary layers.** ESA WorldCover 2020 v100 and 2021 v200; JRC Global Surface Water v1.4 occurrence/seasonality/recurrence; Copernicus DEM GLO-30; AlphaEarth Foundations 64-D annual embeddings (public GeoTIFFs, de-quantised as sign(q)·(q/127.5)², orientation checked with a linear probe against JRC water: 98.8 % vs 91.1 % flipped); PERSIANN-CDR daily precipitation (0.25°) averaged over a box covering the Ichkeul catchment (9.0–9.95° E, 36.75–37.30° N); Landsat 5/7/8 Collection 1 Level-1 for 1984–2021.

## 3. Methods

### 3.1 Habitat classes and weak labels
Eight classes: permanent open water, seasonal shallow water, freshwater emergent marsh, halophytic marsh, bare saline flat / mudflat, cropland & grassland, shrub & woodland, built. A pixel is labelled only when independent sources agree (rules in Table 1, margins around every threshold). The habitat type of marsh pixels comes from the year's wet + dry pair. Freshwater emergent marsh is herbaceous wetland with vigorous winter–spring growth (Ichkeul's *Scirpus*/reed marshes senesce in summer, so summer greenness is the wrong test). Halophytic marsh is perennial, sparse vegetation with a small seasonal NDVI swing. The water state comes from the season itself. Everything else is unlabeled (−1). About 85 % of pixels receive a label in a typical season.

### 3.2 Chips and split
The window is tiled exactly by 10 × 8 chips of 224 × 224 px (2.24 km). Each chip location is one spatial block, assigned to train/val/test (70/15/15) by a seeded, stratified draw on its wetland share, and keeps that assignment for all 20 seasons. This gives 56/12/12 locations and 1,120/240/240 chip-seasons. Test locations are never seen in training in any season.

### 3.3 Models
- **B1 Random Forest (indices):** 10 bands + NDVI, NDWI, MNDWI, NDMI, AWEI, NDRE, NDCI, BSI; 200 trees, balanced class weights; up to 300 pixels per class per training chip.
- **B2 Random Forest (AlphaEarth):** the same pixels with the 64-D annual embedding as features. Embeddings exist for 2017–2025, so this baseline is trained and tested on those years only.
- **B3 U-Net:** ResNet-34 encoder (ImageNet initialisation), 6 Prithvi bands.
- **M1 Prithvi-EO-2.0-300M-TL** (TerraTorch 1.2.13): 6 bands, single date, UNet decoder on blocks 5/11/17/23, fine-tuned end to end. Ablations: frozen encoder, and 3 dates per season as the temporal dimension.
- **M2 TerraMind-1.0-base:** Sentinel-2 L2A (6 bands) + Copernicus DEM modalities, UNet decoder on blocks 2/5/8/11.

The deep models share one training script: AdamW, cosine schedule with warm-up, class-weighted cross-entropy ignoring unlabeled pixels, flips/rotations, mixed precision, 1,500 steps of batch 8, best checkpoint by validation mIoU. Label-efficiency runs keep 5/10/25/50/100 % of the training chip-seasons, with the identical seeded subset rule in the CPU and GPU code (unit-tested), three seeds each. Metrics are computed on every second pixel of the test chips (both axes) for all models.

### 3.4 Change analysis
The model with the highest mean test mIoU at 100 % labels (among models that produced full-area maps) is applied to all 20 composites. For each class and season type: area inside the basin envelope, Mann–Kendall test (tie-corrected) and Sen's slope with its 95 % confidence interval (Gilbert 1987). Transitions use the per-pixel majority class of 2017–2019 versus 2024–2026. Inundation frequency is the share of seasons in which a pixel is water. Per-pixel Mann–Kendall runs at 20 m on wet-season MNDWI and dry-season NDVI, and |z| > 1.96 is flagged.

### 3.5 Drivers
Antecedent rainfall is the Sep–Feb total before a wet composite and the Sep–Jun total before a dry composite, plus the previous hydrological year's total. Open-water, marsh and saline-flat areas are regressed (OLS) on season, antecedent rain, previous-year rain and year. The *unique R²* of a factor is the loss in R² when it is removed. A depth-limited LightGBM model with TreeSHAP ranks the same drivers, and its leave-one-year-out R² is reported.

### 3.6 Side analyses
**Forty years of water.** Landsat TOA reflectance (MTL coefficients, sun-elevation corrected), BQA cloud/shadow masking, seasonal median MNDWI at 30 m, one Otsu threshold pooled over every composite, and seasons with < 90 % clear basin pixels dropped. Seasons 2017–2021 overlap the Sentinel-2 maps for a sensor-agreement check.
**Submerged vegetation.** Inside the mapped permanent lake in wet seasons, NDAVI = (NIR − Blue)/(NIR + Blue) separates *Potamogeton* beds from turbid open water, with a single Otsu threshold for all years.

### 3.7 Gold sample (prepared)
600 points (300 each in the first and last wet season), stratified by mapped class inside the basin (≥ 25 per class, at least 100 m apart), labelled blind in the included web tool, then scored with error-adjusted areas and 95 % CIs (Olofsson et al. 2014; the implementation reproduces the paper's worked example in the unit tests).


## 4. Results

### 4.1 Models (test chips, weak labels)

| Model | Label share | mIoU (mean ± sd) | Overall accuracy | Macro F1 | κ | Seeds |
|---|---|---|---|---|---|---|
| Random Forest · spectral indices | 5 % | 0.432 ± 0.003 | 86.9 % | 0.545 | 0.750 | 3 |
| Random Forest · spectral indices | 10 % | 0.464 ± 0.009 | 89.0 % | 0.578 | 0.788 | 3 |
| Random Forest · spectral indices | 25 % | 0.483 ± 0.006 | 89.7 % | 0.596 | 0.801 | 3 |
| Random Forest · spectral indices | 50 % | 0.502 ± 0.003 | 90.7 % | 0.615 | 0.818 | 3 |
| Random Forest · spectral indices | 100 % | 0.514 ± 0.000 | 91.3 % | 0.626 | 0.829 | 3 |
| U-Net (ResNet-34, random init, CPU) | 100 % | 0.573 ± 0.000 | 95.5 % | 0.686 | 0.906 | 1 |

Per-class F1 at 100 % of training chips:

| Class | U-Net (ResNet-34, random init, CPU) | Random Forest · spectral indices |
|---|---|---|
| Permanent open water | 0.979 | 0.970 |
| Seasonal shallow water | 0.649 | 0.610 |
| Freshwater emergent marsh | 0.451 | 0.269 |
| Halophytic marsh | 0.393 | 0.301 |
| Bare saline flat / mudflat | 0.934 | 0.899 |
| Cropland & grassland | 0.978 | 0.946 |
| Shrub & woodland | 0.653 | 0.571 |
| Built & roads | 0.448 | 0.440 |

> **Foundation-model runs pending.** The Prithvi-EO-2.0-300M and TerraMind-1.0-base experiments (and the ImageNet-initialised U-Net) are fully implemented in `kaggle/train_gpu.py` and smoke-tested end to end, but the processing environment had no GPU and could not reach Hugging Face. They run with `make gpu-upload gpu-push gpu-wait gpu-fetch`, then `make analysis web`. This report and the website update automatically. Until then, question 1 is answered for the CPU models only.

Halophytic marsh and freshwater marsh are the hardest classes (RF F1 0.30 and 0.27). Water, cropland, woodland and built-up areas score high, as expected from their distinct spectra.

Maps for the change analysis come from **U-Net (ResNet-34, random init, CPU)** (highest mean test mIoU at 100 % labels among models with full-area maps).

### 4.2 Seasonal change, 2017–2026 (basin envelope)

| Class | Wet: Sen's slope km²/yr [95 % CI], p | Dry: Sen's slope km²/yr [95 % CI], p |
|---|---|---|
| Permanent open water | -0.58 [-2.30, +0.49], 0.152 | -0.12 [-1.45, +1.50], 0.721 |
| Seasonal shallow water | +0.62 [-0.38, +1.80], 0.107 | -0.08 [-0.39, +0.29], 0.721 |
| Open water (0+1) | -0.50 [-1.26, +0.99], 0.371 | -0.22 [-1.39, +1.11], 0.721 |
| Freshwater emergent marsh | -0.25 [-0.55, +0.32], 0.592 | -0.35 [-0.66, +0.17], 0.152 |
| Halophytic marsh | +0.09 [-0.14, +0.28], 0.592 | -0.09 [-0.25, +0.16], 0.592 |
| Marsh (2+3) | -0.20 [-0.49, +0.36], 0.283 | -0.43 [-0.74, +0.01], 0.074 |
| Bare saline flat / mudflat | +0.04 [-0.01, +0.18], 0.371 | +0.23 [-0.84, +1.26], 0.858 |
| Cropland & grassland | +0.05 [-1.00, +1.12], 0.858 | +0.17 [-0.19, +0.83], 0.283 |
| Shrub & woodland | -0.05 [-0.34, +0.26], 0.721 | +0.01 [-0.48, +0.35], 1.000 |
| Built & roads | -0.04 [-0.07, +0.01], 0.107 | -0.02 [-0.07, +0.02], 0.592 |

No class series reaches p < 0.05 with ten seasons per series. 
Dry-season marsh (freshwater + halophytic) has the most consistent decline (-0.43 km²/yr, p = 0.074). Between 2017–2019 and 2024–2026 (majority class per pixel, dry seasons), 1.2 km² of freshwater marsh and 0.6 km² of halophytic marsh became cropland/grassland, 0.7 km² of freshwater marsh became halophytic marsh, and 0.5 km² of halophytic marsh became saline flat.

*Stability check.* Woodland on Djebel Ichkeul and built-up land should not change much over a decade. Their trends in the map model are -0.05 / +0.01 km²/yr (woodland, wet/dry) and -0.04 / -0.02 km²/yr (built). None is significant, so the maps are temporally stable where they should be. The robustness table shows the same check for every classifier. The Random Forest maps show a spurious summer decline of built-up land (bright bare fields confused with buildings).

**Robustness across classifiers** (Sen's slope km²/yr, Mann–Kendall p, dry seasons):

| Map model | Marsh | Open water | Cropland & grassland | Woodland | Built |
|---|---|---|---|---|---|
| Random Forest · spectral indices | -0.96 (0.002) | -0.20 (0.858) | +0.81 (0.107) | +0.04 (1.000) | -0.18 (0.012) |
| U-Net (ResNet-34, random init, CPU) | -0.43 (0.074) | -0.22 (0.721) | +0.17 (0.283) | +0.01 (1.000) | -0.02 (0.592) |

**Inundation.** 68.4 km² were water in all 20 seasons and 101.6 km² at least once. 16.1 km² are flooded in most winters and dry in most summers (the seasonal pulse). Per-pixel Mann–Kendall on wet-season MNDWI flags 4.9 km² getting drier and 1.2 km² wetter. On dry-season NDVI it flags 9.9 km² browning and 5.5 km² greening.

### 4.3 Drivers

Open water (OLS, n = 20): R² = 0.91 (adjusted 0.89). Unique R²: antecedent rain 0.073, previous-year rain 0.034, year 0.003. Within seasons, +100 mm of antecedent rain goes with +4.3 km² of winter water (R² 0.50, p = 0.022) and +3.1 km² of summer water (R² 0.32, p = 0.091).

Marsh (OLS): R² = 0.39. The year term is -0.21 km²/yr (p = 0.128, unique R² 0.11), while rainfall terms add little (unique R² 0.000 and 0.009).

Gradient boosting with TreeSHAP on open water ranks: is_wet 8.17 km², rain_prev_hydro_year 2.59 km², year 1.30 km², rain_antecedent 0.90 km² (mean |SHAP|). Its leave-one-year-out R² is 0.74.

Interpretation: open-water extent behaves like a rain-fed system over this decade. The marsh decline does not, and it coincides with conversion to cropland and grassland at the marsh edge. This is consistent with agricultural encroachment and grazing pressure (Ramsar) and with reduced freshwater flushing, but without dam-release, salinity or land-use permit data it remains an association, not a proof.

### 4.4 Forty years of water (Landsat 1984–2021 + Sentinel-2)

- **Wet season** mean lake water: 96.4 km² (1984–95), 97.8 km² (1996–2006, Danger List), 99.0 km² (2007–16), 91.9 km² (2017–26). Whole series: Sen's slope -0.09 km²/yr, p = 0.483.
- **Dry season** mean lake water: 73.6 km² (1984–95), 78.6 km² (1996–2006, Danger List), 78.7 km² (2007–16), 75.9 km² (2017–26). Whole series: Sen's slope +0.06 km²/yr, p = 0.217.

Sensor agreement over 10 overlapping seasons: r = 0.991, mean absolute difference 1.0 km², bias (Landsat − Sentinel-2) +0.1 km². The Landsat threshold is a single pooled Otsu value (MNDWI 0.259).

### 4.5 Submerged vegetation (exploratory)

Candidate *Potamogeton* patches (NDAVI more than 0.15 above that winter's lake median) vary from 1.2 to 20.7 km² between winters, with no significant trend (p = 0.283). Only 1.9 km² recur in at least half of the winters. Lake-wide NDAVI shifts with turbidity, so this indicator needs field calibration before any ecological reading.

### 4.6 Independent accuracy (gold sample)

A stratified sample of 594 points (2017 and 2026 wet seasons, ≥ 25 per mapped class) is drawn and served in the labelling tool (`web/label.html`). Labels must come from a person looking at the imagery, ideally checked on a field visit, so no gold accuracy is reported yet. `make gold-score` adds overall accuracy, per-class F1 and Olofsson error-adjusted areas with 95 % CIs.

## 5. Limitations

- **Weak labels.** Training and test labels come from rules, so model scores measure agreement with confident weak labels on unseen locations. They are not map accuracy.
- **Marsh types** are separated by seasonal NDVI behaviour without field data.
- **No official park polygon.** Statistics use a reproducible basin envelope (JRC-flooded lake component + 1 km).
- **Short series.** Ten seasons per series give trend tests limited power, and twenty points limit the driver models.
- **No dam-storage, release, salinity or annual bird-count series** could be obtained. The non-rainfall component is a residual trend.
- **Mixed sensors** in the 40-year series (Landsat TOA MNDWI to 2021, Sentinel-2 maps after). The overlap quantifies the offset.
- **AlphaEarth embeddings are annual** (2017–2025 available), so that baseline cannot see the seasonal state.

## 6. Reproducibility

`make data labels baselines analysis web` reproduces everything on a CPU from open data (about 4 hours, < 10 GB). GPU experiments: `make gpu-upload gpu-push gpu-wait gpu-fetch` (Kaggle) or `notebooks/train_gpu_colab.ipynb`. Unit tests cover the indices, metrics, Mann–Kendall/Sen, Olofsson estimator (reproduces the worked example of Olofsson et al. 2014) and the shared label-fraction rule.

## References

- Olofsson, P. et al. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment* 148, 42–57.
- Szwarcman, D. et al. (2024). Prithvi-EO-2.0: A versatile multi-temporal foundation model for Earth observation applications. arXiv:2412.02732.
- Jakubik, J. et al. (2025). TerraMind: Large-scale generative multimodality for Earth observation. arXiv:2504.11171.
- Brown, C. F. et al. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291.
- Pekel, J.-F. et al. (2016). High-resolution mapping of global surface water and its long-term changes. *Nature* 540, 418–422.
- Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200. Zenodo.
- Ashouri, H. et al. (2015). PERSIANN-CDR: Daily precipitation climate data record from multisatellite observations. *BAMS* 96, 69–83.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *JASA* 63, 1379–1389.
- UNESCO World Heritage Centre: Ichkeul National Park (https://whc.unesco.org/en/list/8); IUCN World Heritage Outlook (https://worldheritageoutlook.iucn.org/node/983).
