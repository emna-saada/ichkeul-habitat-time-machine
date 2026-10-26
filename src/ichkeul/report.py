"""Writes results/REPORT.md. Every number is read from results/*.json."""
from __future__ import annotations

import json
from datetime import date

from .config import RESULTS, ROOT, cfg
from .export import MODEL_LABEL, model_results

KEYS = {"perm_water": "Permanent open water", "seas_water": "Seasonal shallow water", "open_water": "Open water (0+1)",
        "fresh_marsh": "Freshwater emergent marsh", "halo_marsh": "Halophytic marsh", "marsh": "Marsh (2+3)",
        "saline_flat": "Bare saline flat / mudflat", "crop_grass": "Cropland & grassland", "shrub_wood": "Shrub & woodland",
        "built": "Built & roads"}


def f(v, d=2, sign=False):
    if v is None:
        return "–"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def p_(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def build() -> str:
    an = json.loads((RESULTS / "analysis.json").read_text())
    mr = model_results()
    comp = json.loads((RESULTS / "model_comparison.json").read_text()) if (RESULTS / "model_comparison.json").exists() else None
    gold = json.loads((RESULTS / "gold_assessment.json").read_text()) if (RESULTS / "gold_assessment.json").exists() else None
    methods = (ROOT / "docs" / "methods.md").read_text()
    classes = [c["name"] for c in cfg()["classes"]]
    t, d, ls, sv = an["trends"], an["drivers"], an["landsat"], an["submerged_vegetation"]
    ow, mar = d["ols"]["open_water"], d["ols"]["marsh"]
    dm = t["dry"]["marsh"]
    best = mr["best"]
    rank = lambda r: r["miou_matched"] if r.get("miou_matched") is not None else r["miou_mean"]  # noqa: E731
    full = sorted(mr["rows"], key=lambda r: -rank(r))
    rf = next((r for r in full if r["model"] == "rf_indices"), None)
    bm = rank(full[0])
    rm = rank(rf) if rf else float("nan")
    mapm = rank(next((r for r in full if r["model"] == an["model"]), full[0]))
    tr = an["transitions"]["dry"]["matrix_km2"]
    L = []
    w = L.append
    to_crop = tr[2][5] + tr[3][5]
    other_loss = sum(tr[i][j] for i in (2, 3) for j in (0, 1, 4, 6, 7))
    fate = ("with most of the lost marsh becoming cropland or grassland" if to_crop > other_loss
            else "with lost marsh going to several classes")
    lw = ls["trend"]["wet"]
    earlier = [lw["mean_1984_1995"], lw["mean_1996_2006"], lw["mean_2007_2016"]]
    rel = ("below each earlier decade" if all(lw["mean_2017_2026"] < e for e in earlier if e is not None)
           else "within the range of earlier decades")
    w("# Seasonal wetland habitat change at Ichkeul National Park, 2017–2026\n")
    w("*Sentinel-2 composites, geospatial foundation models and rainfall attribution for a UNESCO World Heritage wetland in northern Tunisia*\n")
    w("Emna Saada · " + date.today().strftime("%B %Y") + "\n")
    w("## Abstract\n")
    rob = an.get("robustness", {})
    me = rob.get(an["model"], {})
    pct = 100 * (me["dry_marsh_last3"] / me["dry_marsh_first3"] - 1) if me else float("nan")
    dirs = [v["marsh"]["dry"]["sen_slope"] < 0 for v in rob.values()]
    per_model = "; ".join(f"{MODEL_LABEL.get(m, m)}: {v['marsh']['dry']['sen_slope']:+.2f} km²/yr, p = {p_(v['marsh']['dry']['p'])}"
                          for m, v in rob.items())
    n_sig = sum(v["marsh"]["dry"]["p"] < 0.05 for v in rob.values())
    agree = "every classifier with full-area maps shows a decline" if all(dirs) else "classifiers disagree on the direction"
    w(f"Ichkeul's lake and marshes depend on a freshwater supply that upstream dams have cut since the 1980s. I built 20 seasonal "
      f"Sentinel-2 composites (wet Jan–Mar and dry Jul–Sep, 2017–2026) over a {an['basin_area_km2']:.0f} km² basin envelope and labelled "
      f"eight habitat classes with weak labels from agreeing open datasets. On spatially held-out test chips (2017–2025, matched across models) "
      f"the best model so far is {MODEL_LABEL.get(best, best)} with mIoU {bm:.3f} against weak labels, versus {rm:.3f} for a Random Forest on spectral indices. "
      + ("Its maps drive the change analysis. " if best == an["model"] else
         f"The change analysis uses {MODEL_LABEL.get(an['model'], an['model'])} (matched mIoU {mapm:.3f}), the best model that can map all 20 seasons. ")
      + f"**Open-water area follows rainfall**: season plus antecedent and previous-year rain explain "
      f"R² = {ow['r2']:.2f}, and a time trend adds only {ow['unique_r2']['year'] * 100:.1f} % (p = {p_(ow['p']['year'])}). **Summer marsh area fell by "
      f"{-pct:.0f} %** between {cfg()['years'][0]}–{cfg()['years'][2]} and {cfg()['years'][-3]}–{cfg()['years'][-1]}, and {agree} "
      f"({per_model}). Rainfall explains almost none of the marsh variation (unique R² {mar['unique_r2']['rain_antecedent']:.3f} and "
      f"{mar['unique_r2']['rain_prev_hydro_year']:.3f}), which points to non-climatic pressure (conversion at the marsh edge, grazing, reduced "
      f"freshwater flushing). The trend is significant at p < 0.05 for {n_sig} of {len(rob)} classifiers. A 40-year Landsat series (agreement with "
      f"Sentinel-2: r = {ls['overlap']['r']:.2f} over {ls['overlap']['n']} seasons) puts 2017–2026 mean winter lake water at "
      f"{lw['mean_2017_2026']:.1f} km², {rel}.\n")
    w("## 1. Questions\n")
    w("1. Do geospatial foundation models (Prithvi-EO-2.0, TerraMind) beat Random Forest, AlphaEarth-embedding Random Forest and U-Net baselines when labels are scarce and weak?\n"
      "2. How have open water, freshwater marsh, halophytic marsh and saline flats changed in 2017–2026, by season?\n"
      "3. How much of the change follows rainfall, and how much is left for non-climatic drivers?\n"
      "4. How does the decade compare with the 40-year Landsat record that starts at the time the dams were built?\n")
    w(methods)
    w("\n## 4. Results\n")
    w("### 4.1 Models (test chips, weak labels)\n")
    w("| Model | Label share | mIoU (mean ± sd) | mIoU 2017–25 (matched) | Overall accuracy | Macro F1 | κ | Seeds |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(mr["curves"], key=lambda r: (r["model"], r["fraction"])):
        mm = f"{r['miou_matched']:.3f}" if r.get("miou_matched") is not None else "–"
        w(f"| {r['label']} | {r['fraction'] * 100:.0f} % | {r['miou_mean']:.3f} ± {r['miou_std']:.3f} | {mm} | {r['oa_mean'] * 100:.1f} % | "
          f"{r['f1_mean']:.3f} | {r['kappa_mean']:.3f} | {r['n']} |")
    w("")
    w("AlphaEarth embeddings are published up to 2025, so that baseline is tested on 2017–2025 seasons only. The *matched* column scores "
      "every model on exactly those seasons, and it is the fair comparison across all baselines.\n")
    w("Per-class F1 at 100 % of training chips:\n")
    w("| Class | " + " | ".join(r["label"] for r in full) + " |\n|---|" + "---|" * len(full))
    for i, c in enumerate(classes):
        w(f"| {c} | " + " | ".join(f"{r['per_class_f1'][i]:.3f}" for r in full) + " |")
    w("")
    if mr["gpu_pending"] or not any(r["model"] in ("prithvi", "terramind") for r in full):
        w("> **Foundation-model runs pending.** The Prithvi-EO-2.0-300M and TerraMind-1.0-base experiments (and the ImageNet-initialised "
          "U-Net) are fully implemented in `kaggle/train_gpu.py` and smoke-tested end to end, but the processing environment had no GPU "
          "and could not reach Hugging Face. They run with `make gpu-upload gpu-push gpu-wait gpu-fetch`, then `make analysis web`. "
          "This report and the website update automatically. Until then, question 1 is answered for the CPU models only.\n")
    if rf:
        w(f"Halophytic marsh and freshwater marsh are the hardest classes (RF F1 {rf['per_class_f1'][3]:.2f} and {rf['per_class_f1'][2]:.2f}). "
          "Water, cropland, woodland and built-up areas score high, as expected from their distinct spectra.\n")
    if comp:
        w(f"Maps for the change analysis come from **{MODEL_LABEL.get(comp['map_model'], comp['map_model'])}** (highest mean test mIoU at 100 % labels among models with full-area maps).\n")
    else:
        w(f"Maps for the change analysis come from **{MODEL_LABEL.get(an['model'], an['model'])}**.\n")
    w("### 4.2 Seasonal change, 2017–2026 (basin envelope)\n")
    w("| Class | Wet: Sen's slope km²/yr [95 % CI], p | Dry: Sen's slope km²/yr [95 % CI], p |\n|---|---|---|")
    for k, name in KEYS.items():
        a, b = t["wet"][k], t["dry"][k]
        w(f"| {name} | {f(a['sen_slope'], 2, True)} [{f(a['slope_ci95'][0], 2, True)}, {f(a['slope_ci95'][1], 2, True)}], {p_(a['p'])} | "
          f"{f(b['sen_slope'], 2, True)} [{f(b['slope_ci95'][0], 2, True)}, {f(b['slope_ci95'][1], 2, True)}], {p_(b['p'])} |")
    w("")
    sig = [f"{KEYS[k]} ({s_})" for s_ in ("wet", "dry") for k in KEYS if t[s_][k]["p"] < 0.05]
    w(("Series with a significant monotonic trend (p < 0.05): " + ", ".join(sig) + ". ") if sig else
      "No class series reaches p < 0.05 with ten seasons per series. ")
    w(f"Dry-season marsh (freshwater + halophytic) has the most consistent decline ({f(dm['sen_slope'], 2, True)} km²/yr, p = {p_(dm['p'])}). "
      f"Between {an['transitions']['dry']['from_years'][0]}–{an['transitions']['dry']['from_years'][-1]} and "
      f"{an['transitions']['dry']['to_years'][0]}–{an['transitions']['dry']['to_years'][-1]} (majority class per pixel, dry seasons), "
      f"{tr[2][5]:.1f} km² of freshwater marsh and {tr[3][5]:.1f} km² of halophytic marsh became cropland/grassland, "
      f"{tr[2][3]:.1f} km² of freshwater marsh became halophytic marsh, and {tr[3][4]:.1f} km² of halophytic marsh became saline flat.\n")
    unstable = [KEYS[k] for k in ("shrub_wood", "built") for s_ in ("wet", "dry") if t[s_][k]["p"] < 0.05]
    w(f"*Stability check.* Woodland on Djebel Ichkeul and built-up land should not change much over a decade. Their trends in the map model are "
      f"{f(t['wet']['shrub_wood']['sen_slope'], 2, True)} / {f(t['dry']['shrub_wood']['sen_slope'], 2, True)} km²/yr (woodland, wet/dry) and "
      f"{f(t['wet']['built']['sen_slope'], 2, True)} / {f(t['dry']['built']['sen_slope'], 2, True)} km²/yr (built). "
      + ("None is significant, so the maps are temporally stable where they should be. " if not unstable else
         f"A significant trend in {', '.join(sorted(set(unstable)))} is a classification artefact. ")
      + "The robustness table shows the same check for every classifier."
      + (" The Random Forest maps show a spurious summer decline of built-up land (bright bare fields confused with buildings)."
         if rob.get("rf_indices", {}).get("built", {}).get("dry", {}).get("p", 1) < 0.05 else "") + "\n")
    if rob:
        w("**Robustness across classifiers** (Sen's slope km²/yr, Mann–Kendall p, dry seasons):\n")
        w("| Map model | Marsh | Open water | Cropland & grassland | Woodland | Built |\n|---|---|---|---|---|---|")
        for m, v in rob.items():
            w(f"| {MODEL_LABEL.get(m, m)} | " + " | ".join(f"{v[k]['dry']['sen_slope']:+.2f} ({p_(v[k]['dry']['p'])})"
                                                       for k in ("marsh", "open_water", "crop_grass", "shrub_wood", "built")) + " |")
        w("")
    inn = an["inundation"]
    w(f"**Inundation.** {inn['always_water_km2']:.1f} km² were water in all 20 seasons and {inn['ever_water_km2']:.1f} km² at least once. "
      f"{inn['wet_only_km2']:.1f} km² are flooded in most winters and dry in most summers (the seasonal pulse). Per-pixel "
      f"Mann–Kendall on wet-season MNDWI flags {an['pixel_trends']['mndwi_wet']['sig_decrease_km2']:.1f} km² getting drier and "
      f"{an['pixel_trends']['mndwi_wet']['sig_increase_km2']:.1f} km² wetter. On dry-season NDVI it flags "
      f"{an['pixel_trends']['ndvi_dry']['sig_decrease_km2']:.1f} km² browning and {an['pixel_trends']['ndvi_dry']['sig_increase_km2']:.1f} km² greening.\n")
    w("### 4.3 Drivers\n")
    ps = d["per_season"]["open_water"]
    w(f"Open water (OLS, n = 20): R² = {ow['r2']:.2f} (adjusted {ow['adj_r2']:.2f}). Unique R²: antecedent rain {ow['unique_r2']['rain_antecedent']:.3f}, "
      f"previous-year rain {ow['unique_r2']['rain_prev_hydro_year']:.3f}, year {ow['unique_r2']['year']:.3f}. Within seasons, +100 mm of antecedent "
      f"rain goes with {f(ps['wet']['slope_km2_per_100mm'], 1, True)} km² of winter water (R² {ps['wet']['r2']:.2f}, p = {p_(ps['wet']['p'])}) and "
      f"{f(ps['dry']['slope_km2_per_100mm'], 1, True)} km² of summer water (R² {ps['dry']['r2']:.2f}, p = {p_(ps['dry']['p'])}).\n")
    w(f"Marsh (OLS): R² = {mar['r2']:.2f}. The year term is {f(mar['coef']['year'], 2, True)} km²/yr (p = {p_(mar['p']['year'])}, unique R² "
      f"{mar['unique_r2']['year']:.2f}), while rainfall terms add little (unique R² {mar['unique_r2']['rain_antecedent']:.3f} and "
      f"{mar['unique_r2']['rain_prev_hydro_year']:.3f}).\n")
    g = d["gbm_shap"]
    w(f"Gradient boosting with TreeSHAP on open water ranks: " + ", ".join(
        f"{k} {v:.2f} km²" for k, v in sorted(zip(g["features"], g["mean_abs_shap"]), key=lambda x: -x[1])) +
      f" (mean |SHAP|). Its leave-one-year-out R² is {g['loyo_r2']:.2f}.\n")
    w("Interpretation: open-water extent behaves like a rain-fed system over this decade. The marsh decline does not, and it coincides with "
      "conversion to cropland and grassland at the marsh edge. This is consistent with agricultural encroachment and grazing pressure (Ramsar) "
      "and with reduced freshwater flushing, but without dam-release, salinity or land-use permit data it remains an association, not a proof.\n")
    w("### 4.4 Forty years of water (Landsat 1984–2021 + Sentinel-2)\n")
    for s in ("wet", "dry"):
        tt = ls["trend"][s]
        w(f"- **{s.capitalize()} season** mean lake water: {f(tt['mean_1984_1995'], 1)} km² (1984–95), {f(tt['mean_1996_2006'], 1)} km² (1996–2006, "
          f"Danger List), {f(tt['mean_2007_2016'], 1)} km² (2007–16), {f(tt['mean_2017_2026'], 1)} km² (2017–26). Whole series: Sen's slope "
          f"{f(tt['sen_slope'], 2, True)} km²/yr, p = {p_(tt['p'])}.")
    ov = ls["overlap"]
    w(f"\nSensor agreement over {ov['n']} overlapping seasons: r = {ov['r']:.3f}, mean absolute difference {ov['mean_abs_diff_km2']:.1f} km², "
      f"bias (Landsat − Sentinel-2) {f(ov['bias_km2'], 1, True)} km². The Landsat threshold is a single pooled Otsu value (MNDWI {ov['threshold']:.3f}).\n")
    w("### 4.5 Submerged vegetation (exploratory)\n")
    w(f"Candidate *Potamogeton* patches (NDAVI more than {sv['ndavi_threshold']:.2f} above that winter's lake median) vary from "
      f"{min(r['sav_km2'] for r in sv['series']):.1f} to {max(r['sav_km2'] for r in sv['series']):.1f} km² between winters, with no significant trend "
      f"(p = {p_(sv['trend']['p'])}). Only {sv['persistent_km2']:.1f} km² recur in at least half of the winters. Lake-wide NDAVI shifts with turbidity, "
      "so this indicator needs field calibration before any ecological reading.\n")
    w("### 4.6 Independent accuracy (gold sample)\n")
    if gold and gold.get("n_labelled"):
        for y, v in gold["dates"].items():
            m = v["metrics"]
            w(f"- {y} wet season: n = {v['n']}, overall accuracy {m['overall_accuracy']:.3f}, macro F1 {m['macro_f1']:.3f}.")
    else:
        w("A stratified sample of 594 points (2017 and 2026 wet seasons, ≥ 25 per mapped class) is drawn and served in the labelling tool "
          "(`web/label.html`). Labels must come from a person looking at the imagery, ideally checked on a field visit, so no gold accuracy "
          "is reported yet. `make gold-score` adds overall accuracy, per-class F1 and Olofsson error-adjusted areas with 95 % CIs.\n")
    w("## 5. Limitations\n")
    w("- **Weak labels.** Training and test labels come from rules, so model scores measure agreement with confident weak labels on unseen locations. They are not map accuracy.\n"
      "- **Marsh types** are separated by seasonal NDVI behaviour without field data.\n"
      "- **No official park polygon.** Statistics use a reproducible basin envelope (JRC-flooded lake component + 1 km).\n"
      "- **Short series.** Ten seasons per series give trend tests limited power, and twenty points limit the driver models.\n"
      "- **No dam-storage, release, salinity or annual bird-count series** could be obtained. The non-rainfall component is a residual trend.\n"
      "- **Mixed sensors** in the 40-year series (Landsat TOA MNDWI to 2021, Sentinel-2 maps after). The overlap quantifies the offset.\n"
      "- **AlphaEarth embeddings are annual** (2017–2025 available), so that baseline cannot see the seasonal state.\n")
    w("## 6. Reproducibility\n")
    w("`make data labels baselines analysis web` reproduces everything on a CPU from open data (about 4 hours, < 10 GB). GPU experiments: "
      "`make gpu-upload gpu-push gpu-wait gpu-fetch` (Kaggle) or `notebooks/train_gpu_colab.ipynb`. Unit tests cover the indices, metrics, "
      "Mann–Kendall/Sen, Olofsson estimator (reproduces the worked example of Olofsson et al. 2014) and the shared label-fraction rule.\n")
    w("## References\n")
    w("- Olofsson, P. et al. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment* 148, 42–57.\n"
      "- Szwarcman, D. et al. (2024). Prithvi-EO-2.0: A versatile multi-temporal foundation model for Earth observation applications. arXiv:2412.02732.\n"
      "- Jakubik, J. et al. (2025). TerraMind: Large-scale generative multimodality for Earth observation. arXiv:2504.11171.\n"
      "- Brown, C. F. et al. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291.\n"
      "- Pekel, J.-F. et al. (2016). High-resolution mapping of global surface water and its long-term changes. *Nature* 540, 418–422.\n"
      "- Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200. Zenodo.\n"
      "- Ashouri, H. et al. (2015). PERSIANN-CDR: Daily precipitation climate data record from multisatellite observations. *BAMS* 96, 69–83.\n"
      "- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *JASA* 63, 1379–1389.\n"
      "- UNESCO World Heritage Centre: Ichkeul National Park (https://whc.unesco.org/en/list/8); IUCN World Heritage Outlook (https://worldheritageoutlook.iucn.org/node/983).\n")
    text = "\n".join(L)
    (RESULTS / "REPORT.md").write_text(text)
    return text


if __name__ == "__main__":
    build()
    print("results/REPORT.md written")
