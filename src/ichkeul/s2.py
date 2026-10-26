"""Sentinel-2 L2A access (AWS open-data COGs) and seasonal compositing.

Every scene is read only inside the study-area window, cloud-masked with the
Scene Classification Layer (SCL), converted to surface reflectance with the
per-asset scale/offset (this handles the processing-baseline 04.00 offset), and
reduced to a per-season median composite.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation

from .config import PROC, RAW, cfg, grid

BAND_ASSET = {
    "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1", "B06": "rededge2",
    "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B11": "swir16", "B12": "swir22",
    "SCL": "scl",
}
VALID_SCL = (2, 4, 5, 6, 7)          # dark area, vegetation, bare, water, unclassified
CLOUD_SCL = (3, 8, 9, 10)            # shadow, cloud medium/high, cirrus
ITEM_DIR = RAW / "s2_items"
COMP_DIR = PROC / "composites"
SINGLE_DIR = PROC / "single_dates"

_session = requests.Session()


def _get(url: str, **kw) -> requests.Response:
    for attempt in range(5):
        try:
            r = _session.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404):
                r.raise_for_status()
        except requests.HTTPError:
            raise
        except requests.RequestException:
            if attempt == 4:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {url}")


def list_item_ids(year: int, month: int) -> list[str]:
    c = cfg()["sentinel2"]
    z, b, sq = c["mgrs"]
    prefix = f"{c['prefix']}/{z}/{b}/{sq}/{year}/{month}/"
    xml = _get(f"{c['bucket']}/?list-type=2&prefix={prefix}&delimiter=/").text
    ids = re.findall(r"(S2[ABC]_\d+[A-Z]{3}_\d{8}_\d+_L2A)/</Prefix>", xml)
    # several processing versions may exist for one acquisition: keep the latest
    latest: dict[tuple[str, str], str] = {}
    for i in sorted(ids, key=lambda s: int(s.split("_")[3])):
        sat, _, date, _, _ = i.split("_")
        latest[(sat, date)] = i
    return sorted(latest.values(), key=lambda s: s.split("_")[2])


def item(item_id: str) -> dict:
    ITEM_DIR.mkdir(parents=True, exist_ok=True)
    p = ITEM_DIR / f"{item_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    c = cfg()["sentinel2"]
    z, b, sq = c["mgrs"]
    _, _, date, _, _ = item_id.split("_")
    url = f"{c['bucket']}/{c['prefix']}/{z}/{b}/{sq}/{int(date[:4])}/{int(date[4:6])}/{item_id}/{item_id}.json"
    d = _get(url).json()
    p.write_text(json.dumps(d))
    return d


def _read(href: str, out_shape: tuple[int, int], resampling=Resampling.bilinear) -> np.ndarray:
    g = grid()
    for attempt in range(4):
        try:
            with rasterio.open(href) as src:
                win = from_bounds(*g.bounds, transform=src.transform)
                return src.read(1, window=win, out_shape=out_shape, resampling=resampling, boundless=False)
        except rasterio.errors.RasterioIOError:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(href)


def _scale_offset(it: dict, asset: dict) -> tuple[float, float]:
    """Scale/offset to surface reflectance.

    Earth Search marks scenes whose pixel values were already harmonised (the
    processing-baseline >= 04.00 offset of -1000 DN removed) with
    earthsearch:boa_offset_applied; their assets still list offset -0.1, which must
    then not be applied a second time (checked on deep-lake SWIR reflectance).
    """
    rb = (asset.get("raster:bands") or [{}])[0]
    off = float(rb.get("offset", 0.0))
    if it["properties"].get("earthsearch:boa_offset_applied"):
        off = 0.0
    return float(rb.get("scale", 1e-4)), off


GCS = "https://storage.googleapis.com"
GCS_BUCKET = "gcp-public-data-sentinel-2"
_GCS_RES = {"B02": "R10m", "B03": "R10m", "B04": "R10m", "B08": "R10m", "B05": "R20m", "B06": "R20m",
            "B07": "R20m", "B8A": "R20m", "B11": "R20m", "B12": "R20m", "SCL": "R20m"}


def _gcs_list(prefix: str, delimiter: str | None = "/") -> tuple[list[str], list[str]]:
    names, prefixes, token = [], [], None
    while True:
        q = {"prefix": prefix, "maxResults": 1000}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["pageToken"] = token
        d = _get(f"{GCS}/storage/v1/b/{GCS_BUCKET}/o", params=q).json()
        names += [i["name"] for i in d.get("items", [])]
        prefixes += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            return names, prefixes


def gcs_items(year: int, month: int) -> list[dict]:
    """Pseudo-STAC items for ESA L2A SAFE products on Google Cloud (JPEG2000).

    Used only to fill gaps in the AWS COG archive (e.g. Jan-Feb 2018). When both a
    2018-era (N02xx) and a reprocessed (N0500) product exist, the original is kept so
    that radiometry matches the neighbouring AWS scenes; offsets follow the baseline.
    """
    z, b, sq = cfg()["sentinel2"]["mgrs"]
    root = f"L2/tiles/{z}/{b}/{sq}/"
    _, safes = _gcs_list(f"{root}S2A_MSIL2A_{year}{month:02d}")
    _, safes_b = _gcs_list(f"{root}S2B_MSIL2A_{year}{month:02d}")
    by_date: dict[str, str] = {}
    for sp in sorted(safes + safes_b):
        name = sp.rstrip("/").split("/")[-1]
        date, baseline = name.split("_")[2][:8], name.split("_")[3]
        key = name[:3] + date
        if key not in by_date or baseline < by_date[key].split("_")[3]:
            by_date[key] = name
    out = []
    for key, name in sorted(by_date.items()):
        files, _ = _gcs_list(f"{root}{name}/GRANULE/", delimiter=None)
        assets = {}
        for bnd, res in _GCS_RES.items():
            hit = [f for f in files if f"/IMG_DATA/{res}/" in f and f.endswith(f"_{bnd}_{res[1:]}.jp2")]
            if hit:
                off = -0.1 if name.split("_")[3] >= "N0400" else 0.0
                assets[BAND_ASSET[bnd]] = {"href": f"/vsicurl/{GCS}/{GCS_BUCKET}/{hit[0]}",
                                           "raster:bands": [{"scale": 1e-4, "offset": off}]}
        if len(assets) == len(_GCS_RES):
            iid = f"{name[:3]}_32SNG_{key[3:]}_G_L2A"
            out.append({"id": iid, "properties": {"eo:cloud_cover": 0, "source": "gcs", "safe": name},
                        "assets": assets})
    return out


def clear_mask(it: dict) -> np.ndarray:
    """Boolean clear-sky mask on the 10 m grid (clouds and shadows dilated by 60 m)."""
    g = grid()
    scl = _read(it["assets"]["scl"]["href"], (g.height // 2, g.width // 2), Resampling.nearest)
    cloud = binary_dilation(np.isin(scl, CLOUD_SCL), iterations=3)
    ok = np.isin(scl, VALID_SCL) & ~cloud
    return np.repeat(np.repeat(ok, 2, axis=0), 2, axis=1)


def season_items(year: int, season: str) -> list[dict]:
    """Candidate scenes for one season with their AOI clear fraction (cached)."""
    c = cfg()
    idx = PROC / "scene_index.json"
    cache = json.loads(idx.read_text()) if idx.exists() else {}
    out = []
    for m in c["seasons"][season]:
        for iid in list_item_ids(year, m):
            try:
                it = item(iid)
            except requests.HTTPError:  # incomplete scene folder (preview only)
                continue
            if not all(BAND_ASSET[b] in it["assets"] for b in [*c["sentinel2"]["bands"], "SCL"]):
                continue
            if it["properties"].get("eo:cloud_cover", 100) > c["sentinel2"]["max_tile_cloud"]:
                continue
            if iid not in cache:
                try:
                    cache[iid] = float(clear_mask(it).mean())
                except rasterio.errors.RasterioIOError:  # asset not publicly readable
                    cache[iid] = -1.0
                idx.write_text(json.dumps(cache, indent=0))
            if cache[iid] >= 0:
                out.append({"id": iid, "clear": cache[iid], "item": it})
    if len(out) < 12:  # thin AWS coverage: add ESA SAFE products from Google Cloud
        have = {x["id"].split("_")[2] for x in out}
        for m in c["seasons"][season]:
            for it in gcs_items(year, m):
                iid = it["id"]
                if iid.split("_")[2] in have:
                    continue
                if iid not in cache:
                    cache[iid] = float(clear_mask(it).mean())
                    idx.write_text(json.dumps(cache, indent=0))
                out.append({"id": iid, "clear": cache[iid], "item": it})
    return out


def read_scene(it: dict, bands: list[str], clear: np.ndarray) -> np.ndarray:
    """(bands, H, W) float32 reflectance with NaN where not clear."""
    g = grid()

    def one(b):
        a = it["assets"][BAND_ASSET[b]]
        dn = _read(a["href"], g.shape).astype(np.float32)
        s, o = _scale_offset(it, a)
        r = dn * s + o
        r[(dn == 0) | ~clear] = np.nan
        return r

    with ThreadPoolExecutor(6) as ex:
        return np.stack(list(ex.map(one, bands)))


def write_tif(path: Path, arr: np.ndarray, dtype="int16", nodata=-32768, descriptions=None):
    g = grid()
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = arr if arr.ndim == 3 else arr[None]
    prof = dict(driver="GTiff", width=g.width, height=g.height, count=arr.shape[0], dtype=dtype,
                crs=g.crs, transform=g.transform, nodata=nodata, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256, predictor=2 if "int" in dtype else 3)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype))
        if descriptions:
            for i, d in enumerate(descriptions, 1):
                dst.set_band_description(i, d)


def build_season(year: int, season: str, log=print) -> dict:
    """Median composite + top single dates for one (year, season). Returns metadata."""
    c = cfg()["sentinel2"]
    out = COMP_DIR / f"s2_{year}_{season}.tif"
    meta_p = COMP_DIR / f"s2_{year}_{season}.json"
    if out.exists() and meta_p.exists():
        return json.loads(meta_p.read_text())
    cands = season_items(year, season)
    good = sorted([x for x in cands if x["clear"] >= c["min_aoi_clear"]], key=lambda x: -x["clear"])
    chosen = good[: c["max_scenes_per_season"]]
    if len(chosen) < 3:  # relax for very cloudy winters
        chosen = sorted(cands, key=lambda x: -x["clear"])[: max(3, len(chosen))]
    log(f"{year} {season}: {len(cands)} candidates, using {len(chosen)}")
    bands = c["bands"]
    # int16 reflectance x 10000, NODATA where cloudy: ~0.8 GB for 10 scenes instead of ~1.6 GB of float32
    NODATA = np.int16(-32768)
    stack = np.full((len(chosen), len(bands), *grid().shape), NODATA, np.int16)
    for k, x in enumerate(chosen):
        r = read_scene(x["item"], bands, clear_mask(x["item"]))
        stack[k] = np.where(np.isfinite(r), np.round(np.clip(r, -0.1, 1.5) * 10000), NODATA)
        del r
    n_valid = (stack[:, 0] != NODATA).sum(0).astype(np.int16)
    holes = n_valid == 0
    comp = np.empty((len(bands), *grid().shape), np.float32)
    for b in range(len(bands)):  # one band at a time keeps the median's temporaries small
        sb = stack[:, b].astype(np.float32)
        sb[stack[:, b] == NODATA] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                comp[b] = np.nanmedian(sb, axis=0) / 10000
        del sb
    # pixels never seen clear: fall back to the least-cloudy scene's raw values
    if holes.any():
        fb = read_scene(chosen[0]["item"], bands, np.ones(grid().shape, bool))
        comp[:, holes] = fb[:, holes]
        del fb
    write_tif(out, np.round(np.clip(comp, -0.1, 1.5) * 10000), descriptions=bands)
    write_tif(COMP_DIR / f"nvalid_{year}_{season}.tif", n_valid, nodata=None)
    # top single dates (Prithvi multi-temporal input), gaps filled with the median
    pb = [bands.index(b) for b in c["prithvi_bands"]]
    comp_i = np.round(np.clip(comp[pb], -0.1, 1.5) * 10000).astype(np.int16)
    dates = []
    for k, x in enumerate(chosen[: c["keep_single_dates"]]):
        sd = stack[k][pb]
        sd = np.where(sd != NODATA, sd, comp_i)
        d = x["id"].split("_")[2]
        dates.append(d)
        write_tif(SINGLE_DIR / f"s2_{year}_{season}_t{k}.tif", sd, descriptions=c["prithvi_bands"])
    del stack
    meta = {"year": year, "season": season, "n_candidates": len(cands), "scenes": [x["id"] for x in chosen],
            "clear_fraction": [round(x["clear"], 3) for x in chosen], "single_dates": dates,
            "mean_valid_obs": float(n_valid.mean()), "filled_fraction": float(holes.mean())}
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta


def load_composite(year: int, season: str) -> np.ndarray:
    """(10, H, W) float32 reflectance."""
    with rasterio.open(COMP_DIR / f"s2_{year}_{season}.tif") as s:
        return s.read().astype(np.float32) / 10000.0
