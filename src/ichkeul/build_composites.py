"""Step 1: seasonal Sentinel-2 composites for every year (cached)."""
import json
import time

from .config import PROC, grid, season_keys
from .s2 import build_season


def main():
    g = grid()
    print("grid", g.bounds, g.shape, g.lonlat_bounds(), flush=True)
    metas = []
    for y, s in season_keys():
        t = time.time()
        m = build_season(y, s, log=lambda *a: print(*a, flush=True))
        print(f"  done {y} {s} in {time.time()-t:.0f}s  obs/pixel={m['mean_valid_obs']:.1f}", flush=True)
        metas.append(m)
    (PROC / "composites_summary.json").write_text(json.dumps(metas, indent=1))


if __name__ == "__main__":
    main()
