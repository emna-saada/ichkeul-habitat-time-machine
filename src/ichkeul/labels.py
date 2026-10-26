"""Weak habitat labels from agreement between independent sources.

A pixel receives a label only when ESA WorldCover (2020 and 2021), JRC Global
Surface Water occurrence, the Copernicus DEM and the season's own spectral
evidence all point to the same class, with margins around every threshold.
Everything else is left unlabeled (-1). Labels are produced per (year, season)
because water and marsh states change between composites.

Classes: 0 permanent water, 1 seasonal shallow water, 2 freshwater emergent
marsh, 3 halophytic marsh, 4 bare saline flat / mudflat, 5 cropland & grassland,
6 shrub & woodland, 7 built.
"""
from __future__ import annotations

import numpy as np

from . import aux
from .indices import indices
from .s2 import load_composite

IGNORE = -1

# thresholds (documented in REPORT.md; margins keep only confident pixels)
T = {
    "water_mndwi": 0.20, "water_ndvi": 0.05,
    "dry_land_mndwi": -0.05,
    "perm_occ": 80, "seas_occ_max": 60,
    "wet_zone_dem": 5.0,
    "fresh_ndvi_wet": 0.42,
    "halo_ndvi_lo": 0.12, "halo_ndvi_hi": 0.35, "halo_amp": 0.15,
    "flat_ndvi_max": 0.10,
    "wood_ndvi_dry": 0.30,
    "crop_occ_max": 5,
}


class StaticLayers:
    """Year-independent layers, loaded once."""

    def __init__(self):
        wc = aux.worldcover()
        self.wc20, self.wc21 = wc[2020], wc[2021]
        j = aux.jrc()
        self.occ = j["occurrence"].astype(np.float32)
        self.occ[j["occurrence"] == 255] = 0
        self.dem = aux.dem()
        self.wc_wetland = (self.wc20 == 90) | (self.wc21 == 90)
        self.wet_zone = (self.dem < T["wet_zone_dem"]) & (
            np.isin(self.wc20, (80, 90, 60)) | np.isin(self.wc21, (80, 90, 60)) | (self.occ > 0))

    def wc_both(self, *codes):
        return np.isin(self.wc20, codes) & np.isin(self.wc21, codes)


def weak_labels(year: int, season: str, st: StaticLayers) -> np.ndarray:
    """Habitat type comes from the year's wet+dry pair; the water state from this season."""
    xw, xd = load_composite(year, "wet"), load_composite(year, "dry")
    x = xw if season == "wet" else xd
    ix, iw, idr = indices(x), indices(xw), indices(xd)
    ndvi, mndwi = ix["NDVI"], ix["MNDWI"]
    nw, nd = iw["NDVI"], idr["NDVI"]

    lab = np.full(ndvi.shape, IGNORE, np.int8)
    water = (mndwi > T["water_mndwi"]) & (ndvi < T["water_ndvi"])
    land = mndwi < T["dry_land_mndwi"]

    # 0 / 1 - water, split by long-term JRC occurrence
    lab[water & (st.occ >= T["perm_occ"])] = 0
    lab[water & (st.occ < T["seas_occ_max"]) & st.wet_zone] = 1

    wz = st.wet_zone & land
    # 2 - freshwater emergent marsh: herbaceous wetland with vigorous winter-spring growth
    lab[wz & st.wc_wetland & (nw > T["fresh_ndvi_wet"]) & (nd > 0.15)] = 2
    # 3 - halophytic marsh: perennial, sparse, low-amplitude vegetation on saline lowland
    halo = (wz & (nd > T["halo_ndvi_lo"]) & (nd < T["halo_ndvi_hi"]) & (nw > T["halo_ndvi_lo"])
            & (nw < T["halo_ndvi_hi"] + 0.05) & (np.abs(nw - nd) < T["halo_amp"]) & (st.occ > 0)
            & ~st.wc_both(40))
    lab[halo] = 3
    # 4 - bare saline flat / mudflat: unvegetated in both seasons, inside the flood zone
    lab[wz & (ndvi < T["flat_ndvi_max"]) & (nd < T["flat_ndvi_max"]) & (nw < 0.15) & (st.occ > 0)] = 4

    # 5 - cropland & grassland (WorldCover agreement, rarely flooded)
    crop = land & (st.occ <= T["crop_occ_max"]) & (st.wc_both(40) | (st.wc_both(30) & ~st.wet_zone))
    lab[crop & (lab < 0)] = 5
    # 6 - shrub & woodland (evergreen in summer)
    lab[land & st.wc_both(10, 20) & (nd > T["wood_ndvi_dry"])] = 6
    # 7 - built-up
    lab[st.wc_both(50) & land] = 7
    return lab
