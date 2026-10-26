"""GPU experiments for the Ichkeul habitat study (runs on Kaggle, Colab or any CUDA box).

    python train_gpu.py --data <chips dir> --out <output dir> [--plan plan.json] [--part 0 --parts 2]

Models
  unet            U-Net, ResNet-34 encoder (ImageNet init)               [baseline B3]
  prithvi         Prithvi-EO-2.0-300M-TL, single date, fine-tuned          [M1]
  prithvi_frozen  same, encoder frozen (decoder only)                      [ablation]
  prithvi_mt      same, 3 dates per season as the temporal dimension      [ablation]
  terramind       TerraMind-1.0-base, S2 L2A + Copernicus DEM, fine-tuned  [M2]

Every run writes one JSON line to runs_gpu.jsonl (metrics on the spatially held-out
test chips, plus a confusion matrix per season so subsets can be re-scored later).
Runs flagged "predict" also write full-area maps for all 20 seasons.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 8
IGNORE = -1
PRITHVI_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


# ------------------------------------------------------------------ data
def read(p: Path) -> np.ndarray:
    with rasterio.open(p) as s:
        return s.read()


class Chips:
    """All chips held in memory (about 1 GB single-date, 3 GB multi-temporal)."""

    def __init__(self, root: Path, multitemporal: bool, need_dem: bool):
        self.root = root
        self.stats = json.loads((root / "stats.json").read_text())
        self.mean = np.array(self.stats["means"], np.float32)[:, None, None]
        self.std = np.array(self.stats["stds"], np.float32)[:, None, None]
        self.mt = multitemporal
        self.need_dem = need_dem
        self.split = {k: [ln.strip() for ln in (root / f"{k}.txt").read_text().splitlines() if ln.strip()]
                      for k in ("train", "val", "test")}
        if need_dem:
            d = read(root / "full" / "dem.tif")[0].astype(np.float32)
            self.dem_full = (d - d.mean()) / (d.std() + 1e-6)
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def norm(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.mt:  # (T*6, H, W) -> (6, T, H, W)
            x = x.reshape(-1, 6, *x.shape[1:]).transpose(1, 0, 2, 3)
            return (x - self.mean[:, None]) / self.std[:, None]
        return (x - self.mean) / self.std

    def get(self, stem: str):
        if stem not in self.cache:
            img = read(self.root / ("images_mt" if self.mt else "images") / f"{stem}_img.tif")
            msk = read(self.root / "masks" / f"{stem}_mask.tif")[0].astype(np.int64)
            self.cache[stem] = (self.norm(img), msk)
        return self.cache[stem]

    def dem_chip(self, stem: str) -> np.ndarray:
        r, c = (int(t[1:]) for t in stem.split("_")[2:4])
        n = 224
        return self.dem_full[None, r * n:(r + 1) * n, c * n:(c + 1) * n]


def augment(x: np.ndarray, y: np.ndarray, d: np.ndarray | None, rng: np.random.Generator):
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))

    def tf(a):
        a = np.rot90(a, k, axes=(-2, -1))
        return np.flip(a, -1) if flip else a

    return (np.ascontiguousarray(tf(x)), np.ascontiguousarray(tf(y)),
            None if d is None else np.ascontiguousarray(tf(d)))


def fraction_chips(all_chips: list[str], frac: float, seed: int) -> list[str]:
    """Identical subset rule to the CPU baselines (src/ichkeul/baselines.py)."""
    rng = np.random.default_rng(1000 + seed)
    k = max(1, int(round(len(all_chips) * frac)))
    return sorted(rng.permutation(sorted(all_chips))[:k].tolist())


# ------------------------------------------------------------------ models
def build_model(name: str, smoke: bool = False) -> tuple[nn.Module, str]:
    """Returns (model, input kind) where kind is 'img', 'mt' or 'tm'."""
    if name in ("unet", "unet_cpu"):
        import segmentation_models_pytorch as smp
        # unet_cpu: random initialisation, for environments without access to pretrained weights
        weights = None if (smoke or name == "unet_cpu") else "imagenet"
        return smp.Unet("resnet34", encoder_weights=weights, in_channels=6, classes=N_CLASSES), "img"
    from terratorch.models import EncoderDecoderFactory
    f = EncoderDecoderFactory()
    common = dict(task="segmentation", backbone_pretrained=not smoke, decoder="UNetDecoder",
                  decoder_channels=[512, 256, 128, 64], head_dropout=0.1, num_classes=N_CLASSES)
    if name.startswith("prithvi"):
        frames = 3 if name == "prithvi_mt" else 1
        neck_reshape = {"name": "ReshapeTokensToImage"}
        if frames > 1:
            neck_reshape["effective_time_dim"] = frames
        m = f.build_model(backbone="prithvi_eo_v2_tiny_tl" if smoke else "prithvi_eo_v2_300_tl",
                          backbone_bands=PRITHVI_BANDS, backbone_num_frames=frames, backbone_drop_path=0.1,
                          necks=[{"name": "SelectIndices", "indices": [2, 5, 8, 11] if smoke else [5, 11, 17, 23]},
                                 neck_reshape, {"name": "LearnedInterpolateToPyramidal"}], **common)
        if name == "prithvi_frozen":
            for p in m.encoder.parameters():
                p.requires_grad = False
        return m, "mt" if frames > 1 else "img"
    if name == "terramind":
        m = f.build_model(backbone="terramind_v1_tiny" if smoke else "terramind_v1_base",
                          backbone_modalities=["S2L2A", "DEM"], backbone_bands={"S2L2A": PRITHVI_BANDS},
                          necks=[{"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                                 {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                                 {"name": "LearnedInterpolateToPyramidal"}], **common)
        return m, "tm"
    raise ValueError(name)


def forward(model, kind, x, d):
    if kind == "tm":
        out = model({"S2L2A": x, "DEM": d})
    else:
        out = model(x)
    return out.output if hasattr(out, "output") else out


# ------------------------------------------------------------------ train / eval
def confusion(y, p):
    m = y >= 0
    return np.bincount(N_CLASSES * y[m] + p[m], minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def miou(cm):
    tp = np.diag(cm).astype(float)
    ref, pred = cm.sum(1), cm.sum(0)
    present = ref > 0
    iou = tp / np.maximum(ref + pred - tp, 1)
    return float(iou[present].mean())


@torch.no_grad()
def evaluate(model, kind, data: Chips, stems, dev, stride=2, bs=16):
    model.eval()
    per_season: dict[str, np.ndarray] = {}
    for i in range(0, len(stems), bs):
        batch = stems[i:i + bs]
        xs, ys, ds = zip(*[(*data.get(s), data.dem_chip(s) if kind == "tm" else None) for s in batch])
        x = torch.from_numpy(np.stack(xs)).to(dev)
        d = torch.from_numpy(np.stack(ds)).to(dev) if kind == "tm" else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            p = forward(model, kind, x, d).argmax(1).cpu().numpy()
        for s, yy, pp in zip(batch, ys, p):
            key = "_".join(s.split("_")[:2])
            cm = confusion(yy[::stride, ::stride].ravel(), pp[::stride, ::stride].ravel())
            per_season[key] = per_season.get(key, 0) + cm
    total = sum(per_season.values())
    return total, per_season


def train_one(run: dict, data: Chips, dev, out: Path, smoke=False):
    seed = run["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model, kind = build_model(run["model"], smoke)
    model.to(dev)
    stems = fraction_chips(data.split["train"], run["fraction"], seed)
    w = torch.tensor(data.stats["class_weights"], dtype=torch.float32, device=dev)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = run.get("lr", 3e-4 if run["model"].startswith("unet") else 1e-4)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    steps, bs = run.get("steps", 1500), run.get("batch", 8)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    best, best_state, t0 = -1.0, None, time.time()
    eval_every = run.get("eval_every", max(steps // 6, 1))
    val_stems = data.split["val"]
    for step in range(steps):
        model.train()
        batch = rng.choice(stems, bs, replace=len(stems) < bs)
        xs, ys, ds = [], [], []
        for s in batch:
            x, y = data.get(s)
            d = data.dem_chip(s) if kind == "tm" else None
            x, y, d = augment(x, y, d, rng)
            xs.append(x), ys.append(y), ds.append(d)
        x = torch.from_numpy(np.stack(xs)).to(dev)
        y = torch.from_numpy(np.stack(ys)).to(dev)
        d = torch.from_numpy(np.stack(ds)).to(dev) if kind == "tm" else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = forward(model, kind, x, d)
            loss = F.cross_entropy(logits.float(), y, weight=w, ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            cm, _ = evaluate(model, kind, data, val_stems, dev)
            v = miou(cm)
            print(f"  [{run['model']} f={run['fraction']} s={seed}] step {step+1}/{steps} "
                  f"loss {loss.item():.3f} val mIoU {v:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if v > best:
                best = v
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    cm, per = evaluate(model, kind, data, data.split["test"], dev)
    rec = {**run, "n_train_chips": len(stems), "val_miou": best, "test_miou": miou(cm),
           "test_oa": float(np.trace(cm) / cm.sum()), "confusion": cm.tolist(),
           "confusion_by_season": {k: v.tolist() for k, v in per.items()},
           "train_seconds": round(time.time() - t0, 1),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
    with open(out / "runs_gpu.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"DONE {run['model']} f={run['fraction']} s={seed}: test mIoU {rec['test_miou']:.3f}", flush=True)
    if run.get("predict"):
        predict_full(model, kind, data, dev, out / "predictions", run["model"], max_seasons=1 if smoke else None)
    del model, opt
    torch.cuda.empty_cache()
    return rec


@torch.no_grad()
def predict_full(model, kind, data: Chips, dev, out_dir: Path, name: str, tile=224, stride=176, max_seasons=None):
    """Overlapping sliding-window inference over the whole study area for every season."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    full = data.root / "full"
    for p in sorted(full.glob("s2_*.tif"))[:max_seasons]:
        _, y, s = p.stem.split("_")
        src_p = full / f"mt_{y}_{s}.tif" if kind == "mt" else p
        with rasterio.open(src_p) as src:
            img, prof = src.read(), src.profile
        x = data.norm(img)
        H, W = x.shape[-2:]
        acc = np.zeros((N_CLASSES, H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)
        ys = list(range(0, H - tile, stride)) + [H - tile]
        xs_ = list(range(0, W - tile, stride)) + [W - tile]
        wins = [(i, j) for i in ys for j in xs_]
        for k in range(0, len(wins), 16):
            wb = wins[k:k + 16]
            xb = torch.from_numpy(np.stack([x[..., i:i + tile, j:j + tile] for i, j in wb])).to(dev)
            db = (torch.from_numpy(np.stack([data.dem_full[None, i:i + tile, j:j + tile] for i, j in wb])).to(dev)
                  if kind == "tm" else None)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                pr = torch.softmax(forward(model, kind, xb, db).float(), 1).cpu().numpy()
            for (i, j), q in zip(wb, pr):
                acc[:, i:i + tile, j:j + tile] += q
                cnt[i:i + tile, j:j + tile] += 1
        pred = (acc / cnt).argmax(0).astype(np.uint8)
        prof.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        with rasterio.open(out_dir / f"{name}_{y}_{s}.tif", "w", **prof) as dst:
            dst.write(pred[None])
        print(f"  predicted {name} {y} {s}", flush=True)


def default_plan() -> list[dict]:
    plan = []
    for model in ("unet", "terramind", "prithvi"):
        for frac in (0.05, 0.10, 0.25, 0.50, 1.00):
            for seed in (0, 1, 2):
                plan.append({"model": model, "fraction": frac, "seed": seed,
                             "predict": frac == 1.0 and seed == 0})
    for model in ("prithvi_frozen", "prithvi_mt"):
        for seed in (0, 1, 2):
            plan.append({"model": model, "fraction": 1.0, "seed": seed, "predict": model == "prithvi_mt" and seed == 0})
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="tiny random-init models, few steps (CPU test)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(a.plan.read_text()) if a.plan else default_plan()
    plan = plan[a.part::a.parts]
    done = set()
    if (a.out / "runs_gpu.jsonl").exists():
        for ln in (a.out / "runs_gpu.jsonl").read_text().splitlines():
            r = json.loads(ln)
            done.add((r["model"], r["fraction"], r["seed"]))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    datasets: dict[tuple, Chips] = {}
    for run in plan:
        if (run["model"], run["fraction"], run["seed"]) in done:
            continue
        if a.smoke:
            run = {**run, "steps": 4, "batch": 2, "eval_every": 4}
        key = (run["model"] == "prithvi_mt", run["model"] == "terramind")
        if key not in datasets:
            datasets[key] = Chips(a.data, *key)
        train_one(run, datasets[key], dev, a.out, a.smoke)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
