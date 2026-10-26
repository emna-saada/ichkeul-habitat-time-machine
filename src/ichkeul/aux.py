"""Auxiliary layers on the analysis grid: ESA WorldCover, JRC Global Surface Water,
Copernicus DEM, AlphaEarth satellite embeddings and PERSIANN-CDR rainfall."""
from __future__ import annotations

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from .config import PROC, RAW, grid
from .s2 import write_tif

AUX = PROC / "aux"

WORLDCOVER = {
    2020: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_N36E009_Map.tif",
    2021: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N36E009_Map.tif",
}
JRC = "https://storage.googleapis.com/global-surface-water/downloads2021/{layer}/{layer}_0E_40Nv1_4_2021.tif"
DEM = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N37_00_E009_00_DEM/"
       "Copernicus_DSM_COG_10_N37_00_E009_00_DEM.tif")
AEF_BUCKET = "alphaearth_foundations"
PERSIANN = "https://noaa-cdr-precip-persiann-pds.s3.amazonaws.com"
# box over the Ichkeul catchment (Sejnane, Joumine, Ghezala, Douimis wadis; ~2,080 km2)
CATCHMENT_BOX = (9.0, 36.75, 9.95, 37.30)


def warp(src_path: str, resampling=Resampling.nearest, bands=None, dtype=None) -> np.ndarray:
    g = grid()
    with rasterio.open(src_path) as src:
        with WarpedVRT(src, crs=g.crs, transform=g.transform, width=g.width, height=g.height,
                       resampling=resampling) as vrt:
            a = vrt.read(bands) if bands is not None else vrt.read()
    return a.astype(dtype) if dtype else a


def worldcover() -> dict[int, np.ndarray]:
    out = {}
    for y, url in WORLDCOVER.items():
        p = AUX / f"worldcover_{y}.tif"
        if not p.exists():
            write_tif(p, warp("/vsicurl/" + url)[0], dtype="uint8", nodata=0)
        with rasterio.open(p) as s:
            out[y] = s.read(1)
    return out


def jrc() -> dict[str, np.ndarray]:
    """occurrence (0-100 %), seasonality (months/yr), recurrence (%)."""
    out = {}
    for layer in ("occurrence", "seasonality", "recurrence"):
        p = AUX / f"jrc_{layer}.tif"
        if not p.exists():
            write_tif(p, warp("/vsicurl/" + JRC.format(layer=layer))[0], dtype="uint8", nodata=255)
        with rasterio.open(p) as s:
            out[layer] = s.read(1)
    return out


def dem() -> np.ndarray:
    p = AUX / "dem.tif"
    if not p.exists():
        write_tif(p, warp("/vsicurl/" + DEM, Resampling.bilinear, dtype="float32")[0], dtype="float32", nodata=-9999)
    with rasterio.open(p) as s:
        return s.read(1)


# ---------------------------------------------------------------- AlphaEarth
def _aef_tiles(year: int) -> list[str]:
    names, token = [], None
    while True:
        params = {"prefix": f"satellite_embedding/v1/annual/{year}/32N/", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        d = requests.get(f"https://storage.googleapis.com/storage/v1/b/{AEF_BUCKET}/o", params=params, timeout=60).json()
        names += [i["name"] for i in d.get("items", []) if i["name"].endswith(".tiff")]
        token = d.get("nextPageToken")
        if not token:
            return names


def _aef_find(year: int) -> list[str]:
    """Tiles whose footprint intersects the AOI (headers read remotely, cached)."""
    idx_p = RAW / f"aef_index_{year}.json"
    if idx_p.exists():
        idx = json.loads(idx_p.read_text())
    else:
        def hdr(n):
            try:
                with rasterio.open(f"/vsicurl/https://storage.googleapis.com/{AEF_BUCKET}/{n}") as s:
                    b = rasterio.warp.transform_bounds(s.crs, grid().crs, *s.bounds)
                    return n, list(b)
            except Exception:  # noqa: BLE001 - skip unreadable tiles
                return n, None
        import rasterio.warp  # noqa: F401
        with ThreadPoolExecutor(24) as ex:
            idx = dict(ex.map(hdr, _aef_tiles(year)))
        idx_p.parent.mkdir(parents=True, exist_ok=True)
        idx_p.write_text(json.dumps(idx))
    l, b, r, t = grid().bounds
    hits = []
    for n, bb in idx.items():
        if not bb:
            continue
        x0, x1 = sorted((bb[0], bb[2]))
        y0, y1 = sorted((bb[1], bb[3]))  # these tiles are stored bottom-up
        if x0 < r and x1 > l and y0 < t and y1 > b:
            hits.append(n)
    return hits


def alphaearth(year: int) -> np.ndarray:
    """(64, H, W) float32 de-quantised AlphaEarth embedding for one year."""
    p = AUX / f"alphaearth_{year}.tif"
    if not p.exists():
        g = grid()
        acc = np.full((64, *g.shape), -128, np.int16)
        tiles = _aef_find(year)
        if not tiles:
            raise FileNotFoundError(f"no AlphaEarth embedding tiles published for {year}")
        for n in tiles:
            with rasterio.open(f"/vsicurl/https://storage.googleapis.com/{AEF_BUCKET}/{n}") as src:
                with WarpedVRT(src, crs=g.crs, transform=g.transform, width=g.width, height=g.height,
                               resampling=Resampling.nearest, src_nodata=-128, nodata=-128) as vrt:
                    a = vrt.read().astype(np.int16)
            fill = (acc[0] == -128) & (a[0] != -128)
            acc[:, fill] = a[:, fill]
        write_tif(p, acc, dtype="int8", nodata=-128)
    with rasterio.open(p) as s:
        q = s.read().astype(np.float32)
    nod = q == -128
    v = np.sign(q) * (q / 127.5) ** 2
    v[nod] = np.nan
    return v


# ---------------------------------------------------------------- rainfall
def _persiann_day(key: str) -> tuple[str, float]:
    import time

    import h5py
    for attempt in range(6):
        try:
            r = requests.get(f"{PERSIANN}/{key}", timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2 + 2 * attempt)
    with h5py.File(io.BytesIO(r.content), "r") as f:
        lat, lon = f["lat"][:], f["lon"][:]
        pr = f["precipitation"]
        lo0, la0, lo1, la1 = CATCHMENT_BOX
        li = np.where((lat >= la0) & (lat <= la1))[0]
        lj = np.where((lon >= lo0) & (lon <= lo1))[0]
        sub = pr[0][lj.min(): lj.max() + 1, li.min(): li.max() + 1] if pr.shape[1] == lon.size else \
            pr[0][li.min(): li.max() + 1, lj.min(): lj.max() + 1]
        sub = np.where(sub < 0, np.nan, sub)
    date = re.search(r"_(\d{8})_", key).group(1)
    return date, float(np.nanmean(sub))


def rainfall(start_year: int = 2015) -> pd.Series:
    """Daily catchment-mean precipitation (mm/day), PERSIANN-CDR 0.25 deg."""
    p = PROC / "rain_persiann_daily.csv"
    have = pd.read_csv(p, dtype={"date": str}) if p.exists() else pd.DataFrame(columns=["date", "mm"])
    done = set(have["date"])
    keys = []
    for y in range(start_year, 2027):
        token = None
        while True:
            q = {"list-type": 2, "prefix": f"data/{y}/"}
            if token:
                q["continuation-token"] = token
            xml = requests.get(PERSIANN + "/", params=q, timeout=60).text
            keys += [k for k in re.findall(r"<Key>([^<]+\.nc)</Key>", xml) if re.search(r"_(\d{8})_", k).group(1) not in done]
            m = re.search(r"<NextContinuationToken>([^<]+)<", xml)
            if not m:
                break
            token = m.group(1)
    for i in range(0, len(keys), 300):  # save progress in batches
        with ThreadPoolExecutor(12) as ex:
            rows = list(ex.map(_persiann_day, keys[i: i + 300]))
        have = pd.concat([have, pd.DataFrame(rows, columns=["date", "mm"])])
        have = have.drop_duplicates("date").sort_values("date")
        have.to_csv(p, index=False)
        print(f"  rainfall {len(have)} days", flush=True)
    s = pd.Series(have["mm"].astype(float).values, index=pd.to_datetime(have["date"]))
    return s.sort_index()


def build_all():
    AUX.mkdir(parents=True, exist_ok=True)
    print("worldcover", {k: np.bincount(v.ravel(), minlength=101)[[10, 20, 30, 40, 50, 60, 80, 90, 95]].tolist()
                         for k, v in worldcover().items()}, flush=True)
    j = jrc()
    print("jrc occurrence>50%:", float((j["occurrence"][j["occurrence"] != 255] > 50).mean()), flush=True)
    d = dem()
    print("dem range", float(np.nanmin(d)), float(np.nanmax(d)), flush=True)
    r = rainfall()
    print("rain days", len(r), r.index.min(), r.index.max(), flush=True)
    for y in range(2017, 2026):
        try:
            e = alphaearth(y)
            print("alphaearth", y, float(np.isfinite(e[0]).mean()), flush=True)
        except Exception as ex:  # noqa: BLE001
            print("alphaearth", y, "unavailable:", ex, flush=True)


if __name__ == "__main__":
    build_all()
