"""224x224 chips for every (year, season), spatial-block split, and the
TerraTorch-ready dataset folder (data/chips) that is uploaded to the GPU.

The 10 x 8 chip lattice tiles the study area exactly; each 2.24 km chip is one
spatial block, so a chip location is in train, val or test for all 20 seasons.
"""
from __future__ import annotations

import json

import numpy as np
import rasterio
from affine import Affine

from .config import DATA, PROC, cfg, grid, season_keys
from .labels import StaticLayers, weak_labels

CHIPS = DATA / "chips"
LABEL_DIR = PROC / "labels"


def chip_windows() -> list[tuple[int, int]]:
    a = cfg()["aoi"]
    return [(r, c) for r in range(a["chips_y"]) for c in range(a["chips_x"])]


def make_split(wet_zone: np.ndarray) -> dict[str, list[str]]:
    """Stratified (by wetland share) random assignment of chip locations."""
    s = cfg()["split"]
    n = cfg()["aoi"]["chip_size"]
    rng = np.random.default_rng(s["seed"])
    locs = chip_windows()
    share = np.array([wet_zone[r * n:(r + 1) * n, c * n:(c + 1) * n].mean() for r, c in locs])
    strata = np.digitize(share, [0.02, 0.25, 0.6])
    out = {"train": [], "val": [], "test": []}
    for k in np.unique(strata):
        idx = rng.permutation(np.where(strata == k)[0])
        n_te = int(round(len(idx) * s["test"]))
        n_va = int(round(len(idx) * s["val"]))
        for j, i in enumerate(idx):
            part = "test" if j < n_te else "val" if j < n_te + n_va else "train"
            out[part].append(f"r{locs[i][0]}_c{locs[i][1]}")
    return {k: sorted(v) for k, v in out.items()}


def _chip_profile(r: int, c: int, count: int, dtype: str, nodata) -> dict:
    g, n = grid(), cfg()["aoi"]["chip_size"]
    tr = g.transform * Affine.translation(c * n, r * n)
    return dict(driver="GTiff", width=n, height=n, count=count, dtype=dtype, crs=g.crs, transform=tr,
                nodata=nodata, compress="deflate")


def build():
    st = StaticLayers()
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    split = make_split(st.wet_zone)
    CHIPS.mkdir(parents=True, exist_ok=True)
    (CHIPS / "split_locations.json").write_text(json.dumps(split, indent=1))
    n = cfg()["aoi"]["chip_size"]
    c2 = cfg()["sentinel2"]
    pb = [c2["bands"].index(b) for b in c2["prithvi_bands"]]
    for sub in ("images", "images_mt", "masks", "full"):
        (CHIPS / sub).mkdir(exist_ok=True)
    from .s2 import write_tif
    write_tif(CHIPS / "full" / "dem.tif", st.dem, dtype="float32", nodata=-9999)
    loc_part = {loc: part for part, locs in split.items() for loc in locs}
    stems = {"train": [], "val": [], "test": []}
    counts = np.zeros(8, np.int64)
    sums, sq, npx = np.zeros(6), np.zeros(6), 0
    for y, s in season_keys():
        lab_p = LABEL_DIR / f"labels_{y}_{s}.tif"
        lab = weak_labels(y, s, st)
        from .s2 import load_composite
        write_tif(lab_p, lab, dtype="int16", nodata=-1)
        x = (load_composite(y, s)[pb] * 10000).astype(np.int16)
        singles = []
        for t in range(c2["keep_single_dates"]):
            p = PROC / "single_dates" / f"s2_{y}_{s}_t{t}.tif"
            with rasterio.open(p if p.exists() else PROC / "single_dates" / f"s2_{y}_{s}_t0.tif") as src:
                singles.append(src.read())
        mt = np.concatenate(singles).astype(np.int16)  # (T*6, H, W), time-major
        write_tif(CHIPS / "full" / f"s2_{y}_{s}.tif", x, descriptions=c2["prithvi_bands"])
        write_tif(CHIPS / "full" / f"mt_{y}_{s}.tif", mt)
        for r, c in chip_windows():
            sl = (slice(r * n, (r + 1) * n), slice(c * n, (c + 1) * n))
            loc = f"r{r}_c{c}"
            stem = f"{y}_{s}_{loc}"
            m = lab[sl]
            with rasterio.open(CHIPS / "images" / f"{stem}_img.tif", "w", **_chip_profile(r, c, 6, "int16", None)) as d:
                d.write(x[:, sl[0], sl[1]])
            with rasterio.open(CHIPS / "images_mt" / f"{stem}_img.tif", "w",
                               **_chip_profile(r, c, mt.shape[0], "int16", None)) as d:
                d.write(mt[:, sl[0], sl[1]])
            with rasterio.open(CHIPS / "masks" / f"{stem}_mask.tif", "w", **_chip_profile(r, c, 1, "int16", -1)) as d:
                d.write(m.astype(np.int16)[None])
            part = loc_part[loc]
            stems[part].append(stem)
            if part == "train":
                v = m[m >= 0]
                counts += np.bincount(v, minlength=8)
                xi = x[:, sl[0], sl[1]].reshape(6, -1).astype(np.float64)
                sums += xi.sum(1)
                sq += (xi ** 2).sum(1)
                npx += xi.shape[1]
        print(f"chips {y} {s}: labeled {float((lab >= 0).mean()):.3f}", flush=True)
    for part, v in stems.items():
        (CHIPS / f"{part}.txt").write_text("\n".join(sorted(v)) + "\n")
    mean = sums / npx
    std = np.sqrt(sq / npx - mean ** 2)
    freq = counts / counts.sum()
    w = 1 / np.sqrt(np.maximum(freq, 1e-4))
    stats = {"bands": c2["prithvi_bands"], "means": mean.round(2).tolist(), "stds": std.round(2).tolist(),
             "train_class_pixels": counts.tolist(), "class_weights": (w / w.mean()).round(4).tolist(),
             "n_chips": {k: len(v) for k, v in stems.items()}}
    (CHIPS / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    build()
