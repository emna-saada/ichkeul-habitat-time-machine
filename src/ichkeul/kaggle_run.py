"""Drive Kaggle's free GPU from the command line (needs KAGGLE_USERNAME / KAGGLE_KEY).

    python -m ichkeul.kaggle_run upload        # chips dataset (+ training script, plan)
    python -m ichkeul.kaggle_run push [name]    # start a GPU kernel
    python -m ichkeul.kaggle_run wait [name]    # poll every 5 min until done
    python -m ichkeul.kaggle_run fetch [name]   # download outputs to results/gpu/<name>
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

from .config import DATA, RESULTS, ROOT

DATASET_SLUG = "ichkeul-chips"


def user() -> str:
    u = os.environ.get("KAGGLE_USERNAME")
    if not u:
        cfg = os.path.expanduser("~/.kaggle/kaggle.json")
        u = json.load(open(cfg))["username"]
    return u


def kaggle(*args) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    if r.returncode:
        raise SystemExit(r.returncode)
    return r.stdout


def upload():
    d = DATA / "chips"
    shutil.copy(ROOT / "kaggle" / "train_gpu.py", d / "train_gpu.py")
    plan = ROOT / "kaggle" / "plan.json"
    if plan.exists():
        shutil.copy(plan, d / "plan.json")
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "Ichkeul habitat chips", "id": f"{user()}/{DATASET_SLUG}", "licenses": [{"name": "CC-BY-SA-4.0"}]}))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{user()}/{DATASET_SLUG}"],
                            capture_output=True, text=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(d), "-m", "update", "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(d), "--dir-mode", "zip")


def push(name: str = "ichkeul-gpu"):
    k = ROOT / "kaggle" / "kernel"
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user()}/{name}", "title": name, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user()}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": [],
        "machine_shape": "NvidiaTeslaT4"}, indent=1))
    kaggle("kernels", "push", "-p", str(k))


def wait(name: str = "ichkeul-gpu"):
    t0 = time.time()
    while True:
        s = kaggle("kernels", "status", f"{user()}/{name}")
        if any(w in s.lower() for w in ("complete", "error", "cancel")):
            log = RESULTS / "compute_log.md"
            with open(log, "a") as f:
                f.write(f"- kernel `{name}`: {s.strip()} after {(time.time()-t0)/3600:.2f} h of polling\n")
            return
        time.sleep(300)


def fetch(name: str = "ichkeul-gpu"):
    out = RESULTS / "gpu" / name
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user()}/{name}", "-p", str(out))


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"upload": upload, "push": push, "wait": wait, "fetch": fetch}[cmd](*rest)
