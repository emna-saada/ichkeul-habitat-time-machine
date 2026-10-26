"""Builds every website asset in web/public/data from data/processed and results/.

Nothing on the website is typed by hand except the cited context (timeline, published
bird figures, data sources); every computed number comes from results/*.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import rasterio
from PIL import Image
from rasterio import features
from shapely.geometry import mapping, shape
import shapely
from pyproj import Transformer

from .analysis import AN, PX_KM2, load_map
from .chips import CHIPS, LABEL_DIR
from .config import PROC, RESULTS, ROOT, cfg, class_table, grid, season_keys
from .s2 import COMP_DIR, load_composite

WEB = ROOT / "web" / "public" / "data"
IMG = WEB / "img"
MODEL_LABEL = {
    "rf_indices": "Random Forest · spectral indices", "rf_alphaearth": "Random Forest · AlphaEarth embeddings",
    "unet": "U-Net (ResNet-34)", "unet_cpu": "U-Net (ResNet-34, random init, CPU)", "prithvi": "Prithvi-EO-2.0-300M", "prithvi_frozen": "Prithvi-EO-2.0 · frozen encoder",
    "prithvi_mt": "Prithvi-EO-2.0 · 3 dates", "terramind": "TerraMind-1.0-base",
}


def _hex(c: str) -> tuple[int, int, int]:
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def rgb_jpeg(x: np.ndarray, path, bands=(2, 1, 0), scale=(0.22, 0.22, 0.22), gamma=0.8, quality=84):
    a = np.stack([np.clip(x[b] / s, 0, 1) for b, s in zip(bands, scale)], -1) ** gamma
    Image.fromarray((a * 255).astype(np.uint8)).save(path, quality=quality, optimize=True, progressive=True)


def class_png(m: np.ndarray, path):
    pal = []
    for c in class_table():
        pal += list(_hex(c["color"]))
    pal += [0, 0, 0] * (256 - len(class_table()))
    im = Image.fromarray(np.where(m < 8, m, 255).astype(np.uint8), mode="P")
    im.putpalette(pal)
    im.save(path, optimize=True, transparency=255)


def ramp_png(v: np.ndarray, path, stops: list[tuple[float, str]], alpha_zero=True, vmin=0.0, vmax=1.0):
    t = np.clip((v - vmin) / (vmax - vmin), 0, 1)
    xs = [s[0] for s in stops]
    cols = np.array([_hex(s[1]) for s in stops], float)
    rgb = np.stack([np.interp(t, xs, cols[:, k]) for k in range(3)], -1)
    a = np.where(np.isfinite(v) & ((v != 0) if alpha_zero else True), 235, 0)
    Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA").save(path, optimize=True)


def diverging_png(v: np.ndarray, path, lim: float):
    t = np.clip(v / lim, -1, 1)
    neg, mid, pos = np.array(_hex("#d6604d"), float), np.array(_hex("#f7f7f7"), float), np.array(_hex("#2a9df4"), float)
    rgb = np.where(t[..., None] < 0, mid + (neg - mid) * (-t[..., None]), mid + (pos - mid) * t[..., None])
    a = np.where(v != 0, 230, 0)
    Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA").save(path, optimize=True)


def read(p):
    with rasterio.open(p) as s:
        return s.read(1)


def basin_geojson() -> dict:
    m = read(AN / "basin_mask.tif").astype(np.uint8)
    g = grid()
    to_ll = Transformer.from_crs(g.crs, "EPSG:4326", always_xy=True).transform
    feats = []
    for geom, val in features.shapes(m, mask=m == 1, transform=g.transform):
        poly = shape(geom).simplify(30)
        ll = shapely.transform(poly, lambda c: np.column_stack(to_ll(c[:, 0], c[:, 1])))
        feats.append({"type": "Feature", "properties": {}, "geometry": mapping(ll)})
    return {"type": "FeatureCollection", "features": feats}


# ------------------------------------------------------------------ models
MATCHED_LAST_YEAR = 2025  # last year with every input (AlphaEarth embeddings) available


def _matched_miou(r: dict) -> float | None:
    """mIoU on test seasons up to MATCHED_LAST_YEAR, so all models share one test set."""
    from .metrics import scores
    if r.get("confusion_by_season"):
        cm = sum(np.array(v) for k, v in r["confusion_by_season"].items() if int(k.split("_")[0]) <= MATCHED_LAST_YEAR)
        return scores(np.asarray(cm))["miou"]
    if r["model"] == "rf_alphaearth":  # its test set is already restricted to these years
        return scores(np.array(r["confusion"]))["miou"]
    return None


def model_results() -> dict:
    runs = []
    for f in RESULTS.glob("runs_rf_*.json"):
        for r in json.loads(f.read_text()):
            runs.append({"model": r["model"], "fraction": r["fraction"], "seed": r["seed"], "miou": r["miou"],
                         "oa": r["overall_accuracy"], "f1": r["macro_f1"], "kappa": r["kappa"],
                         "per_class_f1": r["per_class_f1"], "confusion": r["confusion"],
                         "n_train_chips": r["n_train_chips"], "matched": _matched_miou(r)})
    from .metrics import scores
    for f in (RESULTS / "gpu").glob("**/runs_gpu.jsonl"):
        for ln in f.read_text().splitlines():
            r = json.loads(ln)
            sc = scores(np.array(r["confusion"]))
            runs.append({"model": r["model"], "fraction": r["fraction"], "seed": r["seed"], "miou": sc["miou"],
                         "oa": sc["overall_accuracy"], "f1": sc["macro_f1"], "kappa": sc["kappa"],
                         "per_class_f1": sc["per_class_f1"], "confusion": r["confusion"],
                         "n_train_chips": r["n_train_chips"], "matched": _matched_miou(r)})
    groups = defaultdict(list)
    for r in {(r["model"], r["fraction"], r["seed"]): r for r in runs}.values():
        groups[(r["model"], r["fraction"])].append(r)
    rows = []
    for (m, fr), rs in sorted(groups.items()):
        mi = np.array([r["miou"] for r in rs])
        rows.append({"model": m, "label": MODEL_LABEL.get(m, m), "fraction": fr, "n": len(rs),
                     "miou_mean": float(mi.mean()), "miou_std": float(mi.std(ddof=1)) if len(rs) > 1 else 0.0,
                     "oa_mean": float(np.mean([r["oa"] for r in rs])), "oa_std": float(np.std([r["oa"] for r in rs])),
                     "f1_mean": float(np.mean([r["f1"] for r in rs])), "kappa_mean": float(np.mean([r["kappa"] for r in rs])),
                     "per_class_f1": np.mean([r["per_class_f1"] for r in rs], 0).round(4).tolist(),
                     "n_train_chips": int(np.mean([r["n_train_chips"] for r in rs])),
                     "miou_matched": (float(np.mean([r["matched"] for r in rs]))
                                      if all(r["matched"] is not None for r in rs) else None)})
    full = [r for r in rows if r["fraction"] == 1.0]
    rank = lambda r: r["miou_matched"] if r.get("miou_matched") is not None else r["miou_mean"]  # noqa: E731
    best = max(full, key=rank)["model"] if full else None
    cm = np.sum([np.array(r["confusion"]) for r in runs if r["model"] == best and r["fraction"] == 1.0], 0) if best else []
    # Wilcoxon-style paired comparison is not meaningful with 3 seeds; report seed spread instead
    return {"rows": full, "curves": rows, "best": best, "confusion": np.asarray(cm).tolist(), "confusion_model": best,
            "gpu_pending": not any((RESULTS / "gpu").glob("**/runs_gpu.jsonl")), "all_runs": len(runs)}


def compare_chips(models: list[str], year: int, season: str) -> list[dict]:
    split = json.loads((CHIPS / "split_locations.json").read_text())
    n = cfg()["aoi"]["chip_size"]
    lab = read(LABEL_DIR / f"labels_{year}_{season}.tif")
    basin = read(AN / "basin_mask.tif").astype(bool)
    x = load_composite(year, season)
    cands = []
    for loc in split["test"]:
        r, c = (int(t[1:]) for t in loc.split("_"))
        sl = (slice(r * n, (r + 1) * n), slice(c * n, (c + 1) * n))
        cands.append((basin[sl].mean(), loc, sl))
    cands.sort(reverse=True)
    out = []
    (IMG / "chips").mkdir(parents=True, exist_ok=True)
    for _, loc, sl in cands[:5]:
        cid = f"{year}_{season}_{loc}"
        rgb_jpeg(x[:, sl[0], sl[1]], IMG / "chips" / f"{cid}_s2.jpg", quality=90)
        class_png(np.where(lab[sl] >= 0, lab[sl], 255), IMG / "chips" / f"{cid}_label.png")
        preds = {}
        for m in models:
            p = PROC / "predictions" / f"{m}_{year}_{season}.tif"
            if p.exists():
                class_png(read(p)[sl], IMG / "chips" / f"{cid}_{m}.png")
                preds[m] = f"img/chips/{cid}_{m}.png"
        out.append({"id": loc, "year": year, "season": season, "s2": f"img/chips/{cid}_s2.jpg",
                    "label": f"img/chips/{cid}_label.png", "preds": preds})
    return out


# ------------------------------------------------------------------ static, cited context
TIMELINE = [
    {"year": "1980", "text": "<b>National park, Ramsar site and World Heritage</b> inscription."},
    {"year": "1983–94", "text": "<b>Joumine, Ghezala and Sejnane dams</b> built: ~40 % of the catchment and ~70 % of the lake's water supply now dam-controlled.", "bad": True},
    {"year": "1996", "text": "<b>Tinja sluice</b> starts regulating exchange with the Bizerte lagoon; site put on the <b>List of World Heritage in Danger</b> over salinity.", "bad": True},
    {"year": "2006", "text": "<b>Removed from the Danger List</b> after Tunisia stopped using lake water for agriculture and committed to freshwater releases."},
    {"year": "2000–17", "text": "<b>Catchment dam storage rises</b> from 279 to 383 million m³ (+37 %).", "bad": True},
    {"year": "2016", "text": "<b>Removed from Ramsar's Montreux Record</b>: officially recovered, just before this time series begins."},
    {"year": "2020", "text": "<b>Fewer than 6,000 waterbirds</b> in the January count, against a 2008 peak near 300,000.", "bad": True},
    {"year": "2025", "text": "Park director reports migrants down from ~250,000 to ~130,000 a year; <b>smaller dams proposed</b> as a new water source."},
]
BIRDS = [
    {"label": "2008 peak", "value": 300000, "kind": "count", "source": "worldheritagesite.org, citing Les Amis des Oiseaux"},
    {"label": "2008–2017 mean", "value": 141000, "kind": "mean", "source": "worldheritagesite.org, citing Les Amis des Oiseaux"},
    {"label": "Jan 2020 count", "value": 6000, "kind": "upper bound", "source": "worldheritagesite.org, citing Les Amis des Oiseaux"},
    {"label": "Earlier years (2025 interview)", "value": 250000, "kind": "approx.", "source": "TAP via AllAfrica, Sept 2025"},
    {"label": "Now (2025 interview)", "value": 130000, "kind": "approx.", "source": "TAP via AllAfrica, Sept 2025"},
]
RULES = [
    ("Permanent open water", "MNDWI > 0.20, NDVI < 0.05 and JRC occurrence ≥ 80 %"),
    ("Seasonal shallow water", "Water in this season, JRC occurrence < 60 %, inside the low (< 5 m) flood zone"),
    ("Freshwater emergent marsh", "WorldCover herbaceous wetland, winter NDVI > 0.42 (vigorous spring growth), summer NDVI > 0.15"),
    ("Halophytic marsh", "Flood zone, NDVI 0.12–0.35 in both seasons with a seasonal swing < 0.15 (perennial succulents)"),
    ("Bare saline flat / mudflat", "Flood zone, NDVI < 0.10 in both seasons, ever flooded (JRC > 0)"),
    ("Cropland & grassland", "WorldCover 2020 and 2021 agree (cropland, or grassland outside the flood zone), JRC ≤ 5 %"),
    ("Shrub & woodland", "WorldCover tree/shrub in both years and summer NDVI > 0.30"),
    ("Built & roads", "WorldCover built-up in both years"),
]
DATA_SOURCES = [
    ("Sentinel-2 L2A (AWS open data COGs; ESA SAFE on Google Cloud for Jan–Feb 2018)", "Seasonal composites 2017–2026", "Copernicus open licence"),
    ("ESA WorldCover 2020 v100 / 2021 v200", "Weak labels", "CC BY 4.0"),
    ("JRC Global Surface Water v1.4", "Water permanence, weak labels", "Copernicus / EC JRC, free use"),
    ("Copernicus DEM GLO-30", "Flood-zone elevation", "Copernicus DEM licence"),
    ("AlphaEarth Foundations satellite embeddings", "Baseline B2 (64-D annual)", "CC BY 4.0, Google"),
    ("PERSIANN-CDR daily precipitation (NOAA)", "Catchment rainfall 2015–2026", "NOAA CDR, open"),
    ("Landsat 5/7/8 Collection 1 (Google Cloud)", "Water extent 1984–2021", "USGS, public domain"),
    ("Prithvi-EO-2.0 / TerraMind-1.0 weights", "Foundation models", "Apache 2.0 (IBM, NASA, ESA)"),
]
SOURCES = [
    ("UNESCO World Heritage: Ichkeul National Park", "https://whc.unesco.org/en/list/8"),
    ("IUCN World Heritage Outlook: Ichkeul", "https://worldheritageoutlook.iucn.org/node/983"),
    ("Ramsar Sites Information Service: site 213", "https://rsis.ramsar.org/ris/213"),
    ("TAP via AllAfrica, Sept 2025 (park director)", "https://allafrica.com/stories/202509120602.html"),
    ("worldheritagesite.org: waterbird counts", "https://worldheritagesite.org/list/Ichkeul+National+Park"),
    ("Prithvi-EO-2.0 (Szwarcman et al.)", "https://arxiv.org/abs/2412.02732"),
    ("TerraMind (Jakubik et al.)", "https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base"),
    ("AlphaEarth Foundations", "https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/"),
    ("Olofsson et al. 2014, good practices for area estimation", "https://doi.org/10.1016/j.rse.2014.02.015"),
]


def limitations(an: dict, models: dict) -> list[str]:
    out = [
        "<b>Weak labels.</b> Training and test labels come from rules over WorldCover, JRC water and spectral indices, so the scores show how well each model reproduces confident weak labels on unseen locations, not true accuracy. A 600-point hand-labelled gold sample (tool included) is the next step.",
        "<b>Freshwater versus halophytic marsh</b> is inferred from seasonal NDVI behaviour, not from field data. It is the least certain class pair.",
        "<b>No official park boundary.</b> The WDPA polygon was not reachable from the processing environment, so statistics use a basin envelope: the JRC-flooded lake component buffered by 1 km, with the Bizerte lagoon excluded.",
        f"<b>Few seasons.</b> Ten points per seasonal series limit the trend tests and the driver models. The gradient-boosting leave-one-year-out R² is {an['drivers']['gbm_shap']['loyo_r2']:.2f}.",
        "<b>No dam-storage or annual bird-census series</b> could be obtained, so the non-rainfall share is a residual time trend, not a measured management effect.",
        "<b>Mixed sensors in the 40-year series:</b> Landsat top-of-atmosphere MNDWI to 2021, then Sentinel-2 maps. The overlap years quantify the offset.",
        "<b>AlphaEarth embeddings are annual</b> (2017–2025 released at the time of writing), so this baseline cannot see the wet/dry difference.",
    ]
    if models.get("gpu_pending"):
        out.insert(0, "<b>Foundation-model runs pending.</b> Prithvi-EO-2.0, TerraMind and the U-Net train on a GPU. Until those results are in, the maps on this page come from the best CPU model.")
    return out


def headline(an: dict) -> list[dict]:
    rob = an["robustness"][an["model"]]
    dm = an["trends"]["dry"]["marsh"]
    ow = an["drivers"]["ols"]["open_water"]
    lw = an["landsat"]["trend"]["wet"]
    rows = [r for r in an["landsat"]["rows"] if r["season"] == "wet" and r["source"] == "Landsat" and r["water_km2"] and r["year"] < 2017]
    before = float(np.mean([r["water_km2"] for r in rows]))
    pct = 100 * (rob["dry_marsh_last3"] / rob["dry_marsh_first3"] - 1)
    others = {MODEL_LABEL.get(m, m).split(" (")[0].split(" ·")[0]: v["marsh"]["dry"] for m, v in an["robustness"].items() if m != an["model"]}
    alt = "; ".join(f"{k}: {v['sen_slope']:+.2f} (p {v['p']:.3f})" for k, v in others.items())
    ys = cfg()["years"]
    return [
        {"label": "Summer marsh area", "value": f"{pct:+.0f} %",
         "sub": f"{ys[-3]}–{str(ys[-1])[2:]} vs {ys[0]}–{str(ys[2])[2:]}; Sen's slope {dm['sen_slope']:+.2f} km²/yr (p {dm['p']:.2f})"
                + (f"; {alt}" if alt else "")},
        {"label": "Open water explained by rain", "value": f"R² {ow['r2']:.2f}",
         "sub": f"season + rainfall; a time trend adds {ow['unique_r2']['year'] * 100:.1f} %"},
        {"label": "Winter lake vs 1984–2016", "value": f"{100 * (lw['mean_2017_2026'] / before - 1):+.0f} %",
         "sub": f"{lw['mean_2017_2026']:.1f} km² (2017–26) vs {before:.1f} km² (Landsat mean)"},
        {"label": "Winter flood pulse", "value": f"{an['inundation']['wet_only_km2']:.0f} km²",
         "sub": "flooded most winters, dry most summers"},
    ]


def build():
    IMG.mkdir(parents=True, exist_ok=True)
    an = json.loads((RESULTS / "analysis.json").read_text())
    model = an["model"]
    g = grid()
    seasons = []
    for y, s in season_keys():
        x = load_composite(y, s)
        s2p, mp = IMG / f"s2_{y}_{s}.jpg", IMG / f"map_{y}_{s}.png"
        rgb_jpeg(x, s2p)
        class_png(load_map(model, y, s), mp)
        meta = json.loads((COMP_DIR / f"s2_{y}_{s}.json").read_text())
        seasons.append({"year": y, "season": s, "s2": f"img/{s2p.name}", "map": f"img/{mp.name}",
                        "scenes": len(meta["scenes"]), "obs": round(meta["mean_valid_obs"], 1),
                        "dates": [i.split("_")[2] for i in meta["scenes"]]})
        if y in (cfg()["years"][0], cfg()["years"][-1]):
            rgb_jpeg(x, IMG / f"fc_{y}_{s}.jpg", bands=(8, 7, 2), scale=(0.40, 0.45, 0.22), gamma=0.9)
    blues = [(0, "#c6e9f5"), (0.5, "#4aa3df"), (1, "#0b3d91")]
    layers = {}
    for k, title in (("inundation_all", "Water frequency, all 20 seasons"), ("inundation_wet", "Water frequency, wet seasons"),
                     ("inundation_dry", "Water frequency, dry seasons")):
        v = read(AN / f"{k}.tif").astype(float) / 100
        ramp_png(v, IMG / f"{k}.png", blues)
        layers[k] = {"url": f"img/{k}.png", "legend": "ramp", "title": title, "ramp": [c for _, c in blues],
                     "labels": ["rarely", "half the seasons", "always"]}
    greens = [(0, "#d9f0a3"), (0.5, "#41ab5d"), (1, "#00441b")]
    v = read(AN / "sav_frequency.tif").astype(float) / 100
    ramp_png(v, IMG / "sav.png", greens)
    layers["sav"] = {"url": "img/sav.png", "legend": "ramp", "title": "Submerged vegetation frequency, wet seasons",
                     "ramp": [c for _, c in greens], "labels": ["1 year", "5 years", "10 years"]}
    for k, title, lim in (("trend_mndwi_wet", "Significant wet-season MNDWI trend (Sen's slope / yr)", 0.06),
                          ("trend_ndvi_dry", "Significant dry-season NDVI trend (Sen's slope / yr)", 0.03)):
        with rasterio.open(AN / f"{k.replace('trend_', 'trend_')}.tif") as src:
            v = src.read(1)
        diverging_png(v, IMG / f"{k}.png", lim)
        layers[k] = {"url": f"img/{k}.png", "legend": "div", "title": title, "ramp": ["#d6604d", "#f7f7f7", "#2a9df4"],
                     "labels": [f"−{lim}", "0", f"+{lim}"]}
    models = model_results()
    ls = an["landsat"]
    split = json.loads((CHIPS / "split_locations.json").read_text())
    site = {
        "meta": {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "map_model": model,
                 "map_model_label": MODEL_LABEL.get(model, model), "corners": g.corners_lonlat(),
                 "bounds": list(g.lonlat_bounds()), "center": cfg()["aoi"]["centre_lonlat"],
                 "basin_km2": an["basin_area_km2"], "years": cfg()["years"],
                 "train_locations": len(split["train"]), "val_locations": len(split["val"]),
                 "test_locations": len(split["test"])},
        "classes": class_table(), "seasons": seasons, "layers": layers, "basin": basin_geojson(),
        "areas": an["areas_basin"], "trends": an["trends"], "transitions": an["transitions"],
        "inundation": an["inundation"], "pixel_trends": an["pixel_trends"], "sav": an["submerged_vegetation"],
        "drivers": {k: v for k, v in an["drivers"].items() if k != "table"} | {"table": an["drivers"]["table"]},
        "rain": an["rain_features"], "landsat": ls["rows"], "landsat_overlap": ls["overlap"],
        "landsat_trend": ls["trend"], "robustness": an["robustness"],
        "models": {**models, "compute": compute_log()},
        "chips": compare_chips([m for m in ("rf_indices", "rf_alphaearth", "unet_cpu", "unet", "terramind", "prithvi", "prithvi_mt")
                                if any((PROC / "predictions").glob(f"{m}_*.tif"))], 2025, "wet"),
        "birds": BIRDS, "timeline": TIMELINE, "headline": headline(an),
        "rules": [{"cls": c, "rule": r} for c, r in RULES],
        "data_sources": [{"name": n, "use": u, "licence": lc} for n, u, lc in DATA_SOURCES],
        "limitations": limitations(an, models),
        "sources": [{"name": n, "url": u} for n, u in SOURCES],
    }
    (WEB / "site.json").write_text(json.dumps(site, separators=(",", ":"), default=float))
    size = sum(f.stat().st_size for f in WEB.rglob("*") if f.is_file()) / 1e6
    print(f"site.json written; web data {size:.1f} MB; map model {model}")


def compute_log() -> list[str]:
    p = RESULTS / "compute_log.md"
    return [ln.lstrip("- ").strip() for ln in p.read_text().splitlines() if ln.strip().startswith("-")] if p.exists() else []


if __name__ == "__main__":
    build()
