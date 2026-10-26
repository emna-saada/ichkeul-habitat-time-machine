# Ichkeul Habitat Time Machine: end-to-end pipeline
PY = uv run python -m

.PHONY: all data labels baselines gpu-upload gpu-push gpu-wait gpu-fetch analysis gold-sample gold-score web test

all: data labels baselines analysis web

data:            ## Sentinel-2 seasonal composites + auxiliary layers + Landsat (cached)
	$(PY) ichkeul.build_composites
	$(PY) ichkeul.aux
	$(PY) ichkeul.landsat

labels:          ## weak labels, 224 px chips, spatial split, GPU dataset folder
	$(PY) ichkeul.chips

baselines:       ## CPU baselines: RF on indices, RF on AlphaEarth embeddings
	$(PY) ichkeul.baselines rf_indices rf_alphaearth

gpu-upload:      ## upload data/chips as a private Kaggle dataset (needs KAGGLE_USERNAME/KAGGLE_KEY)
	$(PY) ichkeul.kaggle_run upload
gpu-push:        ## start the GPU kernel (U-Net, Prithvi-EO-2.0, TerraMind; fractions x seeds)
	$(PY) ichkeul.kaggle_run push
gpu-wait:
	$(PY) ichkeul.kaggle_run wait
gpu-fetch:       ## download results and predictions into results/gpu/
	$(PY) ichkeul.kaggle_run fetch
	$(PY) ichkeul.select_model

analysis:        ## change analysis, drivers, side analyses on the best model's maps
	$(PY) ichkeul.analysis

gold-sample:     ## stratified sample for the hand-labelling tool
	$(PY) ichkeul.gold sample
gold-score:      ## score maps against results/gold_labels.json (Olofsson areas)
	$(PY) ichkeul.gold score

web:             ## export assets and build the static site
	$(PY) ichkeul.export
	cd web && npm ci && npm run build

test:
	uv run python -m pytest -q tests
