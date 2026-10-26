"""Side analysis: four decades of lake water extent from Landsat 5/7/8 (1984-2021).

Collection 1 Level-1 scenes (WRS-2 192/034, plus 191/034 when a season is thin)
are converted to top-of-atmosphere reflectance from their MTL coefficients, masked
with the BQA cloud/shadow bits, and reduced to a seasonal median MNDWI on a 30 m
version of the study grid. Water area is measured inside the Ichkeul basin
envelope; Sentinel-2 maps (2017-2026) overlap for 2017-2021 to check consistency.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import requests
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from .config import PROC, grid

BUCKET = "gcp-public-data-landsat"
GCS = "https://storage.googleapis.com"
OUT = PROC / "landsat"
# green / SWIR1 band numbers per sensor
BANDS = {"LT05": (2, 5), "LE07": (2, 5), "LC08": (3, 6)}
SEASONS = {"wet": (1, 2, 3), "dry": (7, 8, 9)}
MAX_SCENES = 5


def grid30():
    g = grid()
    return Affine(30, 0, g.left, 0, -30, g.top), (g.height // 3, g.width // 3)


def _list(prefix: str) -> list[str]:
    out, token = [], None
    while True:
        q = {"prefix": prefix, "delimiter": "/", "maxResults": 1000}
        if token:
            q["pageToken"] = token
        d = requests.get(f"{GCS}/storage/v1/b/{BUCKET}/o", params=q, timeout=60).json()
        out += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            return out


def scene_index() -> list[dict]:
    p = OUT / "scenes.json"
    if p.exists():
        return json.loads(p.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for pr in ("192/034", "191/034"):
        for sat in BANDS:
            for sp in _list(f"{sat}/01/{pr}/"):
                sid = sp.rstrip("/").split("/")[-1]
                date = sid.split("_")[3]
                m = int(date[4:6])
                if any(m in v for v in SEASONS.values()) and sid.endswith(("T1", "T2")):
                    rows.append({"id": sid, "prefix": sp, "sat": sat, "pr": pr, "date": date})

    def mtl(r):
        try:
            t = requests.get(f"{GCS}/{BUCKET}/{r['prefix']}{r['id']}_MTL.txt", timeout=60).text
            r["cloud"] = float(re.search(r"CLOUD_COVER = ([\d.\-]+)", t).group(1))
            g, s = BANDS[r["sat"]]
            for b in (g, s):
                r[f"mult{b}"] = float(re.search(rf"REFLECTANCE_MULT_BAND_{b} = ([\d.E\-+]+)", t).group(1))
                r[f"add{b}"] = float(re.search(rf"REFLECTANCE_ADD_BAND_{b} = ([\d.E\-+]+)", t).group(1))
            r["sun_elev"] = float(re.search(r"SUN_ELEVATION = ([\d.\-]+)", t).group(1))
            return r
        except Exception:  # noqa: BLE001 - incomplete product
            return None

    with ThreadPoolExecutor(24) as ex:
        rows = [r for r in ex.map(mtl, rows) if r]
    p.write_text(json.dumps(rows))
    return rows


def _read(url: str, resampling=Resampling.bilinear) -> np.ndarray:
    tr, shape = grid30()
    with rasterio.open("/vsicurl/" + url) as src:
        with WarpedVRT(src, crs=grid().crs, transform=tr, width=shape[1], height=shape[0],
                       resampling=resampling, src_nodata=0, nodata=0) as v:
            return v.read(1)


def scene_mndwi(r: dict) -> np.ndarray:
    g, s = BANDS[r["sat"]]
    base = f"{GCS}/{BUCKET}/{r['prefix']}{r['id']}"
    sin = np.sin(np.radians(r["sun_elev"]))
    G = _read(f"{base}_B{g}.TIF").astype(np.float32)
    S = _read(f"{base}_B{s}.TIF").astype(np.float32)
    q = _read(f"{base}_BQA.TIF", Resampling.nearest).astype(np.int32)
    valid = (G > 0) & (S > 0)
    rg = (G * r[f"mult{g}"] + r[f"add{g}"]) / sin
    rs = (S * r[f"mult{s}"] + r[f"add{s}"]) / sin
    if r["sat"] == "LC08":
        cloud = ((q >> 4) & 1).astype(bool) | (((q >> 7) & 3) == 3) | (((q >> 11) & 3) == 3)
    else:
        cloud = ((q >> 4) & 1).astype(bool) | (((q >> 7) & 3) == 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (rg - rs) / (rg + rs)
    m[~valid | cloud | (q & 1).astype(bool)] = np.nan
    return m


def seasonal_mndwi(year: int, season: str, scenes: list[dict]) -> tuple[np.ndarray | None, list[str]]:
    p = OUT / f"mndwi_{year}_{season}.npy"
    meta_p = OUT / f"mndwi_{year}_{season}.json"
    if p.exists():
        return np.load(p), json.loads(meta_p.read_text())
    cand = [r for r in scenes if int(r["date"][:4]) == year and int(r["date"][4:6]) in SEASONS[season]
            and r["cloud"] >= 0]
    # prefer the centred path and fewer clouds; avoid SLC-off Landsat 7 when others exist
    cand.sort(key=lambda r: (r["cloud"] > 40, r["sat"] == "LE07" and year > 2003, r["pr"] != "192/034", r["cloud"]))
    cand = cand[:MAX_SCENES]
    if not cand:
        return None, []
    with ThreadPoolExecutor(5) as ex:
        stack = np.stack(list(ex.map(scene_mndwi, cand)))
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, 0).astype(np.float32)
    np.save(p, med)
    ids = [r["id"] for r in cand]
    meta_p.write_text(json.dumps(ids))
    return med, ids


def run():
    scenes = scene_index()
    print("landsat scenes", len(scenes), flush=True)
    for y in range(1984, 2022):
        for s in SEASONS:
            m, ids = seasonal_mndwi(y, s, scenes)
            print(y, s, len(ids), None if m is None else round(float(np.isfinite(m).mean()), 3), flush=True)


if __name__ == "__main__":
    run()
