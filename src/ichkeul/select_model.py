"""Collect GPU predictions and choose the model whose maps drive the change analysis.

The choice is the model with the highest mean test mIoU at 100 % of training chips,
among models that produced full-area maps for all 20 seasons.
"""
from __future__ import annotations

import json
import shutil

from .config import PROC, RESULTS, season_keys
from .export import model_results

PRED = PROC / "predictions"


def main():
    PRED.mkdir(parents=True, exist_ok=True)
    for f in (RESULTS / "gpu").glob("**/predictions/*.tif"):
        if not (PRED / f.name).exists():
            shutil.copy(f, PRED / f.name)
    res = model_results()
    complete = {r["model"] for r in res["rows"]
                if all((PRED / f"{r['model']}_{y}_{s}.tif").exists() for y, s in season_keys())}
    ranked = sorted([r for r in res["rows"] if r["model"] in complete], key=lambda r: -r["miou_mean"])
    choice = ranked[0]["model"]
    out = {"map_model": choice, "rule": __doc__.strip().splitlines()[-2:],
           "ranking": [{"model": r["model"], "miou_mean": r["miou_mean"], "miou_std": r["miou_std"]} for r in ranked]}
    (RESULTS / "model_comparison.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
