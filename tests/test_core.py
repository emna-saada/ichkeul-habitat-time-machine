import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ichkeul.baselines import fraction_chips
from ichkeul.config import grid
from ichkeul.indices import FEATURE_NAMES, feature_stack, indices
from ichkeul.metrics import confusion, scores
from ichkeul.stats import mann_kendall, mk_pixels, olofsson

ROOT = Path(__file__).resolve().parents[1]


def _gpu_script():
    spec = importlib.util.spec_from_file_location("train_gpu", ROOT / "kaggle" / "train_gpu.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError:  # torch not installed in a minimal environment
        pytest.skip("torch not installed")
    return mod


def test_grid_is_aligned_to_sentinel2_lattice():
    g = grid()
    assert (g.left - 499980) % 20 == 0 and (4200000 - g.top) % 20 == 0
    assert g.shape == (8 * 224, 10 * 224)
    lon0, lat0, lon1, lat1 = g.lonlat_bounds()
    assert lon0 < 9.6 < 9.7 < lon1 and lat0 < 37.1 < 37.2 < lat1


def test_indices_on_pure_spectra():
    x = np.zeros((10, 1, 2), np.float32)
    x[:, 0, 0] = [0.05, 0.08, 0.04, 0.09, 0.25, 0.30, 0.35, 0.36, 0.20, 0.10]  # vegetation
    x[:, 0, 1] = [0.06, 0.07, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]  # clear water
    ix = indices(x)
    assert ix["NDVI"][0, 0] > 0.7 and ix["NDVI"][0, 1] < 0
    assert ix["MNDWI"][0, 1] > 0.7 and ix["MNDWI"][0, 0] < 0
    assert feature_stack(x).shape[0] == len(FEATURE_NAMES) == 18


def test_scores_perfect_and_ignore():
    y = np.array([0, 1, 2, 2, -1])
    p = np.array([0, 1, 2, 2, 5])
    s = scores(confusion(y, p))
    assert s["overall_accuracy"] == 1.0 and s["miou"] == 1.0 and s["kappa"] == 1.0


def test_scores_known_values():
    cm = np.array([[8, 2], [1, 9]])
    full = np.zeros((8, 8), int)
    full[:2, :2] = cm
    s = scores(full)
    assert s["overall_accuracy"] == pytest.approx(17 / 20)
    assert s["per_class_iou"][0] == pytest.approx(8 / 11, abs=1e-4)
    assert s["miou"] == pytest.approx((8 / 11 + 9 / 12) / 2)


def test_mann_kendall_detects_monotonic_trend():
    y = np.array([10, 9.5, 9.1, 8.0, 7.7, 7.1, 6.0, 5.5, 5.2, 4.0])
    r = mann_kendall(y, np.arange(2017, 2027))
    assert r["trend"] == "decreasing" and r["p"] < 0.001
    assert r["slope_ci95"][0] <= r["sen_slope"] <= r["slope_ci95"][1]
    assert mann_kendall(np.array([1, 3, 2, 3, 1, 2, 3, 1, 2, 2.0]))["trend"] == "no significant trend"


def test_mk_pixels_matches_series_version():
    rng = np.random.default_rng(0)
    st = rng.normal(size=(10, 3, 3)).astype(np.float32) + np.arange(10)[:, None, None] * 0.5
    z, slope = mk_pixels(st)
    one = mann_kendall(st[:, 1, 1])
    assert slope[1, 1] == pytest.approx(one["sen_slope"], rel=1e-5)
    assert z[1, 1] == pytest.approx(one["z"], rel=1e-5)


def test_olofsson_perfect_map_returns_mapped_area():
    cm = np.diag([50, 30, 20])
    area = np.array([100.0, 40.0, 10.0])
    r = olofsson(cm, area)
    assert np.allclose(r["area"], area) and np.allclose(r["ci"], 0) and r["overall_accuracy"] == pytest.approx(1)


def test_olofsson_textbook_example():
    # Olofsson et al. 2014, Table 8/9 (deforestation example): forest-loss area 22,353 ha +- 10,215 ha
    cm = np.array([[66, 0, 5, 4], [0, 55, 8, 12], [1, 0, 153, 11], [2, 1, 9, 313]])
    area = np.array([200000, 150000, 3200000, 6450000], float) * 0.09  # pixels -> ha
    r = olofsson(cm, area)
    assert r["area"][0] == pytest.approx(21158, rel=0.02)
    assert r["overall_accuracy"] == pytest.approx(0.947, abs=0.002)


def test_fraction_subsets_identical_on_cpu_and_gpu_paths():
    m = _gpu_script()
    chips = [f"{y}_{s}_r{r}_c{c}" for y in (2017, 2018) for s in ("wet", "dry") for r in range(3) for c in range(4)]
    for frac in (0.05, 0.25, 1.0):
        for seed in (0, 1, 2):
            assert fraction_chips(chips, frac, seed) == m.fraction_chips(chips, frac, seed)
    assert set(fraction_chips(chips, 0.25, 0)) <= set(chips)
    assert len(fraction_chips(chips, 1.0, 0)) == len(chips)
