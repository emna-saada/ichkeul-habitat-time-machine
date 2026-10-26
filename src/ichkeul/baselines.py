"""Pixel baselines on CPU: Random Forest on spectral indices (B1) and Random
Forest on AlphaEarth 64-D annual embeddings (B2).

Both use exactly the same chip split, label fractions and seeds as the deep
models: a label fraction keeps that share of the training chips (location x
season); evaluation uses every labeled pixel of the spatially held-out test chips.
"""
from __future__ import annotations

import json
import time

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier

from . import aux
from .chips import CHIPS, LABEL_DIR, chip_windows
from .config import PROC, RESULTS, cfg, season_keys
from .indices import feature_stack
from .metrics import confusion, scores
from .s2 import load_composite, write_tif

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]
PER_CLASS_PER_CHIP = 300
TEST_STRIDE = 2
PRED_DIR = PROC / "predictions"


def _labels(y, s):
    with rasterio.open(LABEL_DIR / f"labels_{y}_{s}.tif") as src:
        return src.read(1)


def _features(kind: str, y: int, s: str) -> np.ndarray:
    if kind == "rf_indices":
        return feature_stack(load_composite(y, s))
    e = aux.alphaearth(min(y, _aef_last_year()))
    return np.nan_to_num(e, nan=0.0)


def _aef_last_year() -> int:
    ys = [int(p.stem.split("_")[1]) for p in (PROC / "aux").glob("alphaearth_*.tif")]
    return max(ys)


def build_tables(kind: str):
    """Per-chip training pools and the stride-2 test table for one feature set."""
    split = json.loads((CHIPS / "split_locations.json").read_text())
    loc_part = {loc: p for p, locs in split.items() for loc in locs}
    n = cfg()["aoi"]["chip_size"]
    rng = np.random.default_rng(123)
    pool, Xte, yte, gte = {}, [], [], []
    for y, s in season_keys():
        if kind == "rf_alphaearth" and y > _aef_last_year():
            continue  # no embedding released for this year yet
        F = _features(kind, y, s)
        L = _labels(y, s)
        for r, c in chip_windows():
            loc = f"r{r}_c{c}"
            sl = (slice(r * n, (r + 1) * n), slice(c * n, (c + 1) * n))
            lab = L[sl]
            f = F[:, sl[0], sl[1]]
            if loc_part[loc] == "train":
                idx = []
                for k in range(8):
                    w = np.flatnonzero(lab.ravel() == k)
                    if w.size:
                        idx.append(rng.choice(w, min(PER_CLASS_PER_CHIP, w.size), replace=False))
                if idx:
                    idx = np.concatenate(idx)
                    pool[f"{y}_{s}_{loc}"] = (f.reshape(f.shape[0], -1)[:, idx].T.copy(), lab.ravel()[idx].copy())
            elif loc_part[loc] == "test":
                lt = lab[::TEST_STRIDE, ::TEST_STRIDE]
                ft = f[:, ::TEST_STRIDE, ::TEST_STRIDE]
                m = lt >= 0
                Xte.append(ft[:, m].T)
                yte.append(lt[m])
                gte.append(np.full(m.sum(), y * 10 + (0 if s == "wet" else 1)))
    return pool, np.concatenate(Xte), np.concatenate(yte), np.concatenate(gte)


def fraction_chips(all_chips: list[str], frac: float, seed: int) -> list[str]:
    """Same chip subset rule as the GPU script (seeded permutation)."""
    rng = np.random.default_rng(1000 + seed)
    k = max(1, int(round(len(all_chips) * frac)))
    return sorted(rng.permutation(sorted(all_chips))[:k].tolist())


def train_chips() -> list[str]:
    return [ln.strip() for ln in (CHIPS / "train.txt").read_text().splitlines() if ln.strip()]


def run(kind: str, fractions: list[float] = FRACTIONS) -> list[dict]:
    t0 = time.time()
    pool, Xte, yte, gte = build_tables(kind)
    print(f"{kind}: pools {len(pool)} chips, test px {len(yte)} ({time.time()-t0:.0f}s)", flush=True)
    chips = train_chips()
    rows = []
    for frac in fractions:
        for seed in SEEDS:
            sel = [c for c in fraction_chips(chips, frac, seed) if c in pool]
            X = np.concatenate([pool[c][0] for c in sel])
            y = np.concatenate([pool[c][1] for c in sel])
            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                                        n_jobs=-1, random_state=seed)
            rf.fit(X, y)
            pred = rf.predict(Xte)
            sc = scores(confusion(yte, pred))
            by_season = {f"{g // 10}_{'wet' if g % 10 == 0 else 'dry'}": confusion(yte[gte == g], pred[gte == g]).tolist()
                         for g in np.unique(gte)}
            rows.append({"model": kind, "fraction": frac, "seed": seed, "n_train_chips": len(sel),
                         "n_train_px": int(len(y)), **{k: sc[k] for k in ("miou", "overall_accuracy", "macro_f1",
                                                                         "kappa", "per_class_f1", "confusion")},
                         "confusion_by_season": by_season})
            print(f"  {kind} frac={frac:.2f} seed={seed} chips={len(sel)} mIoU={sc['miou']:.3f} "
                  f"OA={sc['overall_accuracy']:.3f}", flush=True)
            if frac == 1.0 and seed == 0:
                _predict_full(kind, rf)
    out = RESULTS / f"runs_{kind}.json"
    if out.exists() and fractions != FRACTIONS:  # partial re-run: replace only those fractions
        old = [r for r in json.loads(out.read_text()) if r["fraction"] not in fractions]
        rows = sorted(old + rows, key=lambda r: (r["fraction"], r["seed"]))
    out.write_text(json.dumps(rows, indent=1))
    return rows


def _predict_full(kind: str, rf):
    """Full-area maps for all 20 seasons with the 100 % / seed 0 model."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    # annual embeddings give identical wet/dry maps; only the season shown on the website is needed
    keys = [(2025, "wet")] if kind == "rf_alphaearth" else season_keys()
    for y, s in keys:
        p = PRED_DIR / f"{kind}_{y}_{s}.tif"
        if p.exists():
            continue
        F = _features(kind, y, s)
        X = F.reshape(F.shape[0], -1).T
        pred = np.concatenate([rf.predict(X[i:i + 500_000]) for i in range(0, len(X), 500_000)])
        write_tif(p, pred.reshape(F.shape[1:]).astype(np.uint8), dtype="uint8", nodata=255)
        print(f"    predicted {kind} {y} {s}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kinds", nargs="*", default=["rf_indices", "rf_alphaearth"])
    ap.add_argument("--fractions", type=float, nargs="*", default=FRACTIONS)
    a = ap.parse_args()
    for k in a.kinds:
        run(k, a.fractions)
