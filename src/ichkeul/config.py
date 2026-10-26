"""Project configuration and the analysis grid shared by every step."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from affine import Affine
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"

# GDAL reads remote COGs through the system proxy; make sure it trusts the CA bundle.
for _ca in ("/root/.ccr/ca-bundle.crt", "/etc/ssl/certs/ca-certificates.crt"):
    if os.path.exists(_ca):
        os.environ.setdefault("CURL_CA_BUNDLE", _ca)
        break
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")


@lru_cache
def cfg() -> dict:
    with open(ROOT / "configs" / "project.yaml") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Grid:
    """Fixed 10 m UTM 32N grid covering the study area, aligned to the S2 tile grid."""

    left: float
    top: float
    res: float
    width: int
    height: int
    crs: str

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.left, 0, -self.res, self.top)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.left, self.top - self.height * self.res, self.left + self.width * self.res, self.top)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def lonlat_bounds(self) -> tuple[float, float, float, float]:
        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        l, b, r, tp = self.bounds
        xs, ys = t.transform([l, r, l, r], [b, b, tp, tp])
        return (min(xs), min(ys), max(xs), max(ys))

    def corners_lonlat(self) -> list[list[float]]:
        """Corner coordinates (TL, TR, BR, BL) for MapLibre image sources."""
        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        l, b, r, tp = self.bounds
        pts = [(l, tp), (r, tp), (r, b), (l, b)]
        return [list(t.transform(x, y)) for x, y in pts]


@lru_cache
def grid() -> Grid:
    a = cfg()["aoi"]
    res, chip = a["resolution"], a["chip_size"]
    w, h = a["chips_x"] * chip, a["chips_y"] * chip
    t = Transformer.from_crs("EPSG:4326", a["crs"], always_xy=True)
    cx, cy = t.transform(*a["centre_lonlat"])
    # snap to the 20 m lattice of the Sentinel-2 tile (origin 499980, 4200000)
    left = round((cx - w * res / 2 - 499980) / 20) * 20 + 499980
    top = round((cy + h * res / 2 - 4200000) / 20) * 20 + 4200000
    return Grid(float(left), float(top), float(res), w, h, a["crs"])


def season_keys() -> list[tuple[int, str]]:
    c = cfg()
    return [(y, s) for y in c["years"] for s in ("wet", "dry")]


def class_table() -> list[dict]:
    return cfg()["classes"]
