"""Spectral indices from a (10, H, W) Sentinel-2 reflectance stack.

Band order: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12.
"""
from __future__ import annotations

import numpy as np

B = {b: i for i, b in enumerate(["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"])}
INDEX_NAMES = ["NDVI", "NDWI", "MNDWI", "NDMI", "AWEI", "NDRE", "NDCI", "BSI"]


def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip((a - b) / (a + b + 1e-6), -1, 1)


def indices(x: np.ndarray) -> dict[str, np.ndarray]:
    g = lambda k: x[B[k]]  # noqa: E731
    return {
        "NDVI": _nd(g("B08"), g("B04")),
        "NDWI": _nd(g("B03"), g("B08")),
        "MNDWI": _nd(g("B03"), g("B11")),
        "NDMI": _nd(g("B08"), g("B11")),
        "AWEI": g("B02") + 2.5 * g("B03") - 1.5 * (g("B08") + g("B11")) - 0.25 * g("B12"),
        "NDRE": _nd(g("B8A"), g("B05")),
        "NDCI": _nd(g("B05"), g("B04")),
        "BSI": _nd(g("B11") + g("B04"), g("B08") + g("B02")),
    }


def feature_stack(x: np.ndarray) -> np.ndarray:
    """(18, H, W): 10 reflectance bands + 8 indices (the RF baseline features)."""
    ix = indices(x)
    return np.concatenate([x, np.stack([ix[k] for k in INDEX_NAMES])]).astype(np.float32)


FEATURE_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", *INDEX_NAMES]
