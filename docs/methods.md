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
