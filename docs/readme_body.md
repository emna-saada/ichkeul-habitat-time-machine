## What is in this repository

```
configs/project.yaml       study grid, seasons, classes, split
src/ichkeul/
  s2.py                    Sentinel-2 access (AWS COGs + Google Cloud SAFE fallback), cloud masking, seasonal medians
  aux.py                   WorldCover, JRC water, DEM, AlphaEarth embeddings, PERSIANN-CDR rainfall
  labels.py                weak-label rules (agreement of independent sources)
  chips.py                 224 px chips, spatial-block split, GPU dataset folder
  baselines.py             Random Forest on indices / on AlphaEarth embeddings
  landsat.py               Landsat 5/7/8 seasonal MNDWI, 1984-2021
  analysis.py              areas, Mann-Kendall/Sen, transitions, inundation, drivers, side analyses
  gold.py                  stratified gold sample + Olofsson error-adjusted areas
  select_model.py          picks the map model from test scores
  export.py / report.py    website assets and results/REPORT.md, all numbers read from results/
  kaggle_run.py            Kaggle CLI driver for the GPU experiments
kaggle/train_gpu.py        U-Net, Prithvi-EO-2.0 (fine-tuned / frozen / 3 dates), TerraMind; fractions x seeds
notebooks/                 Colab version of the GPU run
results/                   metrics, analysis.json, REPORT.md
web/                       Vite + MapLibre + ECharts site, gold-labelling tool (label.html)
tests/                     pytest for indices, metrics, statistics, shared split rule
```

## Reproduce

```bash
uv sync                      # Python 3.12+, pinned dependencies
make data                    # composites 2017-2026, auxiliary layers, Landsat (cached, ~6 GB)
make labels                  # weak labels + chips
make baselines               # Random Forest baselines (CPU)
# GPU experiments: set KAGGLE_USERNAME / KAGGLE_KEY, then
make gpu-upload gpu-push gpu-wait gpu-fetch
make analysis web            # change analysis, website build
make gold-sample             # then label in web/label.html and run: make gold-score
make test
```

No account is needed for any data source: Sentinel-2 and Landsat come from public cloud buckets, and AlphaEarth embeddings from Google's public GeoTIFF bucket. Only the GPU step needs a Kaggle (or Colab) account.

## Data and licences

Contains modified Copernicus Sentinel data 2017–2026 (ESA). ESA WorldCover 2020/2021 (CC BY 4.0); JRC Global Surface Water v1.4 (EC JRC); Copernicus DEM GLO-30; AlphaEarth Foundations satellite embeddings (Google, CC BY 4.0); PERSIANN-CDR (NOAA CDR program); Landsat 5/7/8 (USGS, public domain); Prithvi-EO-2.0 and TerraMind weights (Apache 2.0). Basemap imagery on the website © Esri and partners; labels © OpenStreetMap contributors, © CARTO. Context figures (timeline, waterbird numbers) are cited on the site.

## Author

**Emna Saada**, geoinformatics engineering student, Bizerte, Tunisia · [GitHub](https://github.com/emna-saada)

Code released under the MIT licence.
