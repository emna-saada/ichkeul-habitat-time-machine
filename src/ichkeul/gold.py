"""Independent ("gold") accuracy assessment.

    python -m ichkeul.gold sample   # draw the stratified sample -> web/public/data/gold_points.json
    python -m ichkeul.gold score    # after labelling in the web tool, put the exported file at
                                    # results/gold_labels.json, then score the map and compute
                                    # error-adjusted areas (Olofsson et al. 2014)

Design: two map dates (first and last wet season), 300 points each, stratified by
mapped class inside the basin envelope (>= 25 points per class, the rest allocated
proportionally to mapped area), points at least 100 m apart.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from pyproj import Transformer

from .analysis import PX_KM2, basin_mask, best_model, load_map
from .config import RESULTS, ROOT, cfg, grid
from .metrics import confusion, scores
from .stats import olofsson

POINTS = ROOT / "web" / "public" / "data" / "gold_points.json"
N_PER_DATE, MIN_PER_CLASS = 300, 25


def sample(seed: int = 7):
    model = best_model()
    ys = cfg()["years"]
    mask = basin_mask()
    g = grid()
    to_ll = Transformer.from_crs(g.crs, "EPSG:4326", always_xy=True)
    rng = np.random.default_rng(seed)
    pts = []
    for y in (ys[0], ys[-1]):
        m = load_map(model, y, "wet")
        area = np.array([((m == k) & mask).sum() for k in range(8)], float)
        alloc = np.where(area > 0, MIN_PER_CLASS, 0)
        rest = N_PER_DATE - alloc.sum()
        alloc = alloc + np.floor(rest * area / area.sum()).astype(int)
        taken = np.zeros(g.shape, bool)
        for k in range(8):
            rr, cc = np.nonzero((m == k) & mask)
            order = rng.permutation(len(rr))
            got = 0
            for i in order:
                r, c = rr[i], cc[i]
                if taken[max(r - 10, 0):r + 10, max(c - 10, 0):c + 10].any():
                    continue
                taken[r, c] = True
                x, yy = g.transform * (c + 0.5, r + 0.5)
                lon, lat = to_ll.transform(x, yy)
                pts.append({"id": len(pts), "year": int(y), "season": "wet", "row": int(r), "col": int(c),
                            "lon": round(lon, 6), "lat": round(lat, 6), "map_class": k})
                got += 1
                if got >= alloc[k]:
                    break
    rng.shuffle(pts)  # blind the labeller to the map class order
    POINTS.write_text(json.dumps({"model": model, "design": __doc__.split("Design:")[1].strip(), "points": pts}))
    print(f"{len(pts)} points -> {POINTS}")


def score():
    lab = json.loads((RESULTS / "gold_labels.json").read_text())
    pts = {p["id"]: p for p in json.loads(POINTS.read_text())["points"]}
    model = json.loads(POINTS.read_text())["model"]
    mask = basin_mask()
    out = {"model": model, "n_labelled": 0, "dates": {}}
    for y in sorted({p["year"] for p in pts.values()}):
        rows = [(pts[int(k)]["map_class"], v) for k, v in lab["labels"].items()
                if int(k) in pts and pts[int(k)]["year"] == y and v is not None and v >= 0]
        if not rows:
            continue
        mp, ref = np.array(rows).T
        cm = confusion(ref, mp)          # rows = reference
        m = load_map(model, y, "wet")
        mapped = np.array([((m == k) & mask).sum() for k in range(8)], float) * PX_KM2
        adj = olofsson(cm.T, mapped)     # rows = map
        out["dates"][str(y)] = {"n": int(len(rows)), "metrics": scores(cm), "mapped_km2": mapped.tolist(),
                                "error_adjusted": adj}
        out["n_labelled"] += int(len(rows))
    (RESULTS / "gold_assessment.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {"n": v["n"], "oa": v["metrics"]["overall_accuracy"]} for k, v in out["dates"].items()}))


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
