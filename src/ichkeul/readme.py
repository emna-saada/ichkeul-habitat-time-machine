"""Writes README.md; every number comes from web/public/data/site.json (built from results/)."""
from __future__ import annotations

import json

from .config import ROOT


def build():
    s = json.loads((ROOT / "web" / "public" / "data" / "site.json").read_text())
    head = (ROOT / "docs" / "readme_head.md").read_text()
    body = (ROOT / "docs" / "readme_body.md").read_text()
    cards = "\n".join(f"| **{h['value']}** | {h['label']} — {h['sub']} |" for h in s["headline"])
    hl = "| | Key results |\n|---|---|\n" + cards + "\n\n![Hero](docs/img/hero.jpg)\n"
    m = s["models"]
    rows = "\n".join(f"| {r['label']} | {r['miou_mean']:.3f} ± {r['miou_std']:.3f} | {r['oa_mean'] * 100:.1f} % | {r['f1_mean']:.3f} | {r['n']} |"
                     for r in sorted(m["rows"], key=lambda r: -r["miou_mean"]))
    rob = s["robustness"]
    from .export import MODEL_LABEL
    rrows = "\n".join(f"| {MODEL_LABEL.get(r, r)} | {v['marsh']['dry']['sen_slope']:+.2f} (p {v['marsh']['dry']['p']:.3f}) | "
                      f"{v['open_water']['dry']['sen_slope']:+.2f} (p {v['open_water']['dry']['p']:.3f}) | "
                      f"{v['open_water']['wet']['sen_slope']:+.2f} (p {v['open_water']['wet']['p']:.3f}) |" for r, v in rob.items())
    ow = s["drivers"]["ols"]["open_water"]
    mid = f"""
## Method in one picture

```
Sentinel-2 L2A (2017-2026)  ──► cloud-masked wet (Jan-Mar) / dry (Jul-Sep) medians ──► 20 composites, 10 m
WorldCover + JRC water + DEM + seasonal NDVI/MNDWI rules ──► weak labels (8 classes, ~85 % of pixels)
224 px chips, 80 locations, spatial-block split 56/12/12 ──► RF (indices) · RF (AlphaEarth) · U-Net · Prithvi-EO-2.0 · TerraMind
best model ──► 20 habitat maps ──► areas, Mann-Kendall / Sen, transitions, inundation frequency
PERSIANN-CDR rainfall ──► OLS unique R², gradient boosting + SHAP      Landsat 1984-2021 ──► 40-year water series
```

## Results

**Models** (test chips at 12 held-out locations × 20 seasons, scored against weak labels, 100 % of training chips):

| Model | mIoU | Overall accuracy | Macro F1 | Seeds |
|---|---|---|---|---|
{rows}

{"> Prithvi-EO-2.0 and TerraMind runs are implemented and smoke-tested (`kaggle/train_gpu.py`) and run with `make gpu-upload gpu-push gpu-wait gpu-fetch`. Their rows, the label-efficiency curves and the maps update automatically." if m["gpu_pending"] else ""}

**Change, 2017-2026** (basin envelope, {s['meta']['basin_km2']:.0f} km²; Sen's slope km²/yr and Mann-Kendall p, on each classifier's maps):

| Map model | Summer marsh | Summer open water | Winter open water |
|---|---|---|---|
{rrows}

**Drivers.** Season plus rainfall explain R² = {ow['r2']:.2f} of open-water area, and a time trend adds {ow['unique_r2']['year'] * 100:.1f} %. Rainfall explains almost none of the marsh variation. The 40-year Landsat series agrees with Sentinel-2 at r = {s['landsat_overlap']['r']:.2f}.

Full write-up with methods, tables and limitations: [`results/REPORT.md`](results/REPORT.md).

![Time machine](docs/img/timemachine.jpg)
"""
    (ROOT / "README.md").write_text(head.replace("{{HEADLINE}}", hl) + mid + "\n" + body)


if __name__ == "__main__":
    build()
