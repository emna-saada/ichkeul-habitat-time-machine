"""Change analysis on the seasonal habitat maps and driver attribution.

Outputs results/analysis.json plus rasters in data/processed/analysis/ that the
website export turns into map layers.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import ndimage

from . import aux
from .config import PROC, RESULTS, cfg, grid, season_keys
from .indices import indices
from .s2 import load_composite, write_tif
from .stats import mann_kendall, mk_pixels

AN = PROC / "analysis"
PX_KM2 = 1e-4  # one 10 m pixel
LAKE_CENTRE = (9.67, 37.16)
WATER = (0, 1)
CLASS_KEYS = [c["key"] for c in cfg()["classes"]]


def best_model() -> str:
    p = RESULTS / "model_comparison.json"
    if p.exists():
        return json.loads(p.read_text())["map_model"]
    return "rf_indices"


def load_map(model: str, y: int, s: str) -> np.ndarray:
    with rasterio.open(PROC / "predictions" / f"{model}_{y}_{s}.tif") as src:
        return src.read(1)


def basin_mask() -> np.ndarray:
    """Ichkeul lake + marsh envelope: the JRC-flooded component containing the lake,
    separated from the Bizerte lagoon by an opening, then buffered by 1 km."""
    p = AN / "basin_mask.tif"
    if p.exists():
        with rasterio.open(p) as s:
            return s.read(1).astype(bool)
    occ = aux.jrc()["occurrence"]
    wet = (occ > 0) & (occ != 255)
    opened = ndimage.binary_opening(wet, iterations=4)
    lab, _ = ndimage.label(opened)
    g = grid()
    x, y = Transformer.from_crs("EPSG:4326", g.crs, always_xy=True).transform(*LAKE_CENTRE)
    col, row = ~g.transform * (x, y)
    lake = lab == lab[int(row), int(col)]
    lake = ndimage.binary_fill_holes(lake)
    env = ndimage.binary_dilation(lake, iterations=100)
    other_water = wet & ~ndimage.binary_dilation(lake, iterations=10)
    env &= ~ndimage.binary_dilation(other_water & (occ > 50), iterations=5)
    write_tif(p, env.astype(np.uint8), dtype="uint8", nodata=None)
    return env


def class_areas(model: str, mask: np.ndarray) -> pd.DataFrame:
    rows = []
    for y, s in season_keys():
        m = load_map(model, y, s)
        cnt = np.bincount(m[mask], minlength=8)[:8] * PX_KM2
        rows.append({"year": y, "season": s, **{k: round(float(v), 4) for k, v in zip(CLASS_KEYS, cnt)}})
    return pd.DataFrame(rows)


def trends(df: pd.DataFrame) -> dict:
    out = {}
    for s in ("wet", "dry"):
        d = df[df.season == s].sort_values("year")
        series = {k: d[k].values for k in CLASS_KEYS}
        series["open_water"] = d["perm_water"].values + d["seas_water"].values
        series["marsh"] = d["fresh_marsh"].values + d["halo_marsh"].values
        out[s] = {k: mann_kendall(v, d["year"].values) for k, v in series.items()}
    return out


def majority(stack: np.ndarray) -> np.ndarray:
    counts = np.stack([(stack == k).sum(0) for k in range(8)])
    return counts.argmax(0).astype(np.uint8)


def transitions(model: str, mask: np.ndarray) -> dict:
    """Majority class of the first three vs the last three years, per season type (km2)."""
    ys = cfg()["years"]
    out = {}
    for s in ("wet", "dry"):
        a = majority(np.stack([load_map(model, y, s) for y in ys[:3]]))
        b = majority(np.stack([load_map(model, y, s) for y in ys[-3:]]))
        mat = np.bincount(8 * a[mask].astype(np.int64) + b[mask], minlength=64).reshape(8, 8) * PX_KM2
        out[s] = {"from_years": ys[:3], "to_years": ys[-3:], "matrix_km2": mat.round(4).tolist()}
        if s == "wet":
            write_tif(AN / "early_wet.tif", a, dtype="uint8", nodata=255)
            write_tif(AN / "late_wet.tif", b, dtype="uint8", nodata=255)
    return out


def inundation(model: str) -> dict:
    keys = season_keys()
    water = np.stack([np.isin(load_map(model, y, s), WATER) for y, s in keys])
    freq = water.mean(0)
    wet = water[[i for i, (_, s) in enumerate(keys) if s == "wet"]].mean(0)
    dry = water[[i for i, (_, s) in enumerate(keys) if s == "dry"]].mean(0)
    write_tif(AN / "inundation_all.tif", (freq * 100).round(), dtype="uint8", nodata=255)
    write_tif(AN / "inundation_wet.tif", (wet * 100).round(), dtype="uint8", nodata=255)
    write_tif(AN / "inundation_dry.tif", (dry * 100).round(), dtype="uint8", nodata=255)
    m = basin_mask()
    return {"always_water_km2": float((freq[m] == 1).sum() * PX_KM2),
            "ever_water_km2": float((freq[m] > 0).sum() * PX_KM2),
            "wet_only_km2": float(((wet[m] >= 0.5) & (dry[m] < 0.5)).sum() * PX_KM2),
            "breathing_ratio": float(((wet[m] >= 0.5) & (dry[m] < 0.5)).sum() / max((dry[m] >= 0.5).sum(), 1))}


def pixel_trends() -> dict:
    """Per-pixel Mann-Kendall on wet-season MNDWI and dry-season NDVI at 20 m."""
    ys = cfg()["years"]

    def agg(a):
        h, w = a.shape
        return a.reshape(h // 2, 2, w // 2, 2).mean((1, 3))

    res = {}
    for name, season, idx in (("mndwi_wet", "wet", "MNDWI"), ("ndvi_dry", "dry", "NDVI")):
        st = np.stack([agg(indices(load_composite(y, season))[idx]) for y in ys]).astype(np.float32)
        z, slope = mk_pixels(st)
        sig = np.abs(z) > 1.96
        write_tif_20m(AN / f"trend_{name}.tif", np.where(sig, slope, 0).astype(np.float32))
        m = basin_mask()[::2, ::2]
        res[name] = {"sig_increase_km2": float(((z > 1.96) & m).sum() * 4 * PX_KM2),
                     "sig_decrease_km2": float(((z < -1.96) & m).sum() * 4 * PX_KM2)}
    return res


def write_tif_20m(path, arr):
    g = grid()
    prof = dict(driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32", crs=g.crs,
                transform=g.transform * g.transform.scale(2), compress="deflate", nodata=None)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr[None])


SAV_ANOMALY = 0.15


def submerged_vegetation(model: str) -> dict:
    """Candidate submerged aquatic vegetation (Potamogeton beds) in the permanent lake, wet season.

    Lake-wide NDAVI = (NIR - Blue)/(NIR + Blue) is unimodal and shifts with each
    winter's turbidity, so a single absolute threshold would separate years rather
    than vegetation. Instead a pixel is a candidate when its NDAVI exceeds that
    winter's lake median by SAV_ANOMALY: a patch clearly greener than the water
    around it. The recurrence map shows where such patches persist across years.
    Uncalibrated (no field data): read the series as relative, not as bed area.
    """
    from .indices import _nd

    ys = cfg()["years"]
    m = basin_mask()
    series, freq = [], np.zeros(grid().shape, np.float32)
    for y in ys:
        x = load_composite(y, "wet")
        ndavi = _nd(x[6], x[0])
        lake = (load_map(model, y, "wet") == 0) & m
        med = float(np.median(ndavi[lake]))
        sav = lake & (ndavi - med > SAV_ANOMALY)
        sav = ndimage.binary_opening(sav, iterations=1)  # drop isolated pixels
        freq += sav
        series.append({"year": y, "sav_km2": float(sav.sum() * PX_KM2), "lake_km2": float(lake.sum() * PX_KM2),
                       "sav_share": float(sav.sum() / max(lake.sum(), 1)), "lake_median_ndavi": med})
    write_tif(AN / "sav_frequency.tif", (freq / len(ys) * 100).round(), dtype="uint8", nodata=255)
    persistent = float((freq >= len(ys) / 2).sum() * PX_KM2)
    return {"ndavi_threshold": SAV_ANOMALY, "method": "anomaly", "persistent_km2": persistent, "series": series,
            "trend": mann_kendall([r["sav_km2"] for r in series], ys)}


def landsat_series(model: str, df: pd.DataFrame) -> dict:
    """1984-2021 lake water area from Landsat seasonal MNDWI medians (30 m).

    One Otsu threshold is fitted on all basin pixels pooled over every composite, so
    no year is tuned individually. Seasons with < 90 % valid basin pixels are dropped.
    """
    from skimage.filters import threshold_otsu

    from .landsat import OUT as LS

    m10 = basin_mask()
    h, w = m10.shape[0] // 3, m10.shape[1] // 3
    basin = m10[: h * 3, : w * 3].reshape(h, 3, w, 3).mean((1, 3)) > 0.5
    comps = {}
    for f in sorted(LS.glob("mndwi_*.npy")):
        _, y, s = f.stem.split("_")
        comps[(int(y), s)] = np.load(f)
    pooled = np.concatenate([v[basin & np.isfinite(v)][::7] for v in comps.values()])
    thr = float(threshold_otsu(pooled))
    rows = []
    for (y, s), v in sorted(comps.items()):
        valid = np.isfinite(v) & basin
        frac = valid.sum() / basin.sum()
        ids = json.loads((LS / f"mndwi_{y}_{s}.json").read_text())
        area = float(((v > thr) & valid).sum() * 9e-4) if frac >= 0.9 else None
        rows.append({"year": y, "season": s, "water_km2": area, "source": "Landsat", "scenes": len(ids),
                     "valid_fraction": round(float(frac), 3)})
    for r in df.to_dict(orient="records"):
        rows.append({"year": int(r["year"]), "season": r["season"], "water_km2": float(r["perm_water"] + r["seas_water"]),
                     "source": "Sentinel-2", "scenes": None, "valid_fraction": 1.0})
    ls = {(r["year"], r["season"]): r["water_km2"] for r in rows if r["source"] == "Landsat"}
    s2 = {(r["year"], r["season"]): r["water_km2"] for r in rows if r["source"] == "Sentinel-2"}
    pairs = [(ls[k], s2[k]) for k in ls if k in s2 and ls[k] is not None]
    a, b = np.array(pairs).T if pairs else (np.array([]), np.array([]))
    overlap = {"n": len(pairs), "threshold": thr,
               "r": float(np.corrcoef(a, b)[0, 1]) if len(pairs) > 2 else None,
               "mean_abs_diff_km2": float(np.abs(a - b).mean()) if pairs else None,
               "bias_km2": float((a - b).mean()) if pairs else None,
               "pairs": [{"year": k[0], "season": k[1], "landsat": ls[k], "s2": s2[k]} for k in ls
                         if k in s2 and ls[k] is not None]}
    trend = {}
    for s in ("wet", "dry"):
        ser = [(r["year"], r["water_km2"]) for r in rows if r["season"] == s and r["water_km2"] is not None
               and (r["source"] == "Landsat" or r["year"] > 2021)]
        yy, vv = np.array(ser).T
        trend[s] = mann_kendall(vv, yy)
        for lo, hi in ((1984, 1995), (1996, 2006), (2007, 2016), (2017, 2026)):
            sel = (yy >= lo) & (yy <= hi)
            trend[s][f"mean_{lo}_{hi}"] = float(vv[sel].mean()) if sel.any() else None
    return {"rows": rows, "overlap": overlap, "trend": trend}


# ------------------------------------------------------------------ drivers
def rain_features() -> pd.DataFrame:
    r = aux.rainfall()
    rows = []
    for y in cfg()["years"]:
        hy = r[f"{y-1}-09-01":f"{y}-08-31"]
        rows.append({
            "year": y,
            "rain_wet_antecedent": float(r[f"{y-1}-09-01":f"{y}-02-28"].sum()),   # Sep-Feb before wet composite
            "rain_dry_antecedent": float(r[f"{y-1}-09-01":f"{y}-06-30"].sum()),   # whole wet season before summer
            "rain_prev_hydro_year": float(r[f"{y-2}-09-01":f"{y-1}-08-31"].sum()),
            "rain_hydro_year": float(hy.sum()) if hy.index.max() >= pd.Timestamp(f"{y}-07-31") else None,
        })
    return pd.DataFrame(rows)


def drivers(df: pd.DataFrame) -> dict:
    import lightgbm as lgb
    import shap
    import statsmodels.api as sm

    rf = rain_features()
    d = df.copy()
    d["open_water"] = d.perm_water + d.seas_water
    d["marsh"] = d.fresh_marsh + d.halo_marsh
    d = d.merge(rf, on="year")
    d["rain_antecedent"] = np.where(d.season == "wet", d.rain_wet_antecedent, d.rain_dry_antecedent)
    d["is_wet"] = (d.season == "wet").astype(int)
    out = {"table": d.round(4).to_dict(orient="records"), "ols": {}, "per_season": {}}
    for target in ("open_water", "marsh", "saline_flat"):
        X = sm.add_constant(d[["is_wet", "rain_antecedent", "rain_prev_hydro_year", "year"]].astype(float))
        fit = sm.OLS(d[target].astype(float), X).fit()
        full_r2 = fit.rsquared
        partial = {}
        for drop in ("rain_antecedent", "rain_prev_hydro_year", "year"):
            r2 = sm.OLS(d[target].astype(float), X.drop(columns=[drop])).fit().rsquared
            partial[drop] = float(max(full_r2 - r2, 0))
        out["ols"][target] = {"r2": float(full_r2), "adj_r2": float(fit.rsquared_adj),
                              "coef": fit.params.round(5).to_dict(), "p": fit.pvalues.round(4).to_dict(),
                              "ci95": fit.conf_int().round(5).values.tolist(), "unique_r2": partial}
        for s in ("wet", "dry"):
            ds = d[d.season == s]
            f2 = sm.OLS(ds[target].astype(float), sm.add_constant(ds[["rain_antecedent"]].astype(float))).fit()
            out["per_season"].setdefault(target, {})[s] = {
                "slope_km2_per_100mm": float(f2.params["rain_antecedent"] * 100), "r2": float(f2.rsquared),
                "p": float(f2.pvalues["rain_antecedent"]), "intercept": float(f2.params["const"])}
    # gradient boosting + SHAP (leave-one-year-out to report honest skill)
    feats = ["is_wet", "rain_antecedent", "rain_prev_hydro_year", "year"]
    X, yv = d[feats].astype(float).values, d["open_water"].values
    params = dict(n_estimators=200, learning_rate=0.05, num_leaves=4, min_child_samples=3, verbose=-1, random_state=0)
    pred = np.zeros(len(yv))
    for yr in d.year.unique():
        tr, te = d.year.values != yr, d.year.values == yr
        pred[te] = lgb.LGBMRegressor(**params).fit(X[tr], yv[tr]).predict(X[te])
    loyo_r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    model = lgb.LGBMRegressor(**params).fit(X, yv)
    sv = shap.TreeExplainer(model).shap_values(X)
    out["gbm_shap"] = {"target": "open_water", "features": feats, "loyo_r2": float(loyo_r2),
                       "mean_abs_shap": np.abs(sv).mean(0).round(4).tolist(),
                       "shap_values": sv.round(4).tolist(), "feature_values": X.round(2).tolist()}
    return out


def robustness(mask: np.ndarray) -> dict:
    """Key trends recomputed on the maps of every model that covers all 20 seasons."""
    out = {}
    for f in sorted((PROC / "predictions").glob("*_2026_dry.tif")):
        m = f.name[: -len("_2026_dry.tif")]
        if not all((PROC / "predictions" / f"{m}_{y}_{s}.tif").exists() for y, s in season_keys()):
            continue
        df = class_areas(m, mask)
        tr = trends(df)
        dry = df[df.season == "dry"].sort_values("year")
        marsh = (dry.fresh_marsh + dry.halo_marsh).values
        out[m] = {k: {s: {"sen_slope": tr[s][k]["sen_slope"], "p": tr[s][k]["p"]} for s in ("wet", "dry")}
                  for k in ("open_water", "marsh", "fresh_marsh", "halo_marsh", "saline_flat", "crop_grass", "shrub_wood", "built")}
        out[m]["dry_marsh_first3"] = float(marsh[:3].mean())
        out[m]["dry_marsh_last3"] = float(marsh[-3:].mean())
    return out


def run(model: str | None = None) -> dict:
    model = model or best_model()
    AN.mkdir(parents=True, exist_ok=True)
    mask = basin_mask()
    df = class_areas(model, mask)
    df_aoi = class_areas(model, np.ones(grid().shape, bool))
    res = {"model": model, "basin_area_km2": float(mask.sum() * PX_KM2),
           "areas_basin": df.to_dict(orient="records"), "areas_aoi": df_aoi.to_dict(orient="records"),
           "trends": trends(df), "transitions": transitions(model, mask), "inundation": inundation(model),
           "pixel_trends": pixel_trends(), "submerged_vegetation": submerged_vegetation(model),
           "drivers": drivers(df), "rain_features": rain_features().to_dict(orient="records"),
           "landsat": landsat_series(model, df), "robustness": robustness(mask)}
    (RESULTS / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("model", "basin_area_km2", "inundation")}, indent=1))
    for s in ("wet", "dry"):
        for k in ("open_water", "marsh", "saline_flat", "halo_marsh", "fresh_marsh"):
            t = res["trends"][s][k]
            print(f"{s:3s} {k:12s} sen={t['sen_slope']:+.3f} km2/yr p={t['p']:.3f} {t['trend']}")
    return res


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else None)
