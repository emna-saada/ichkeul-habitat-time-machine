"""Kaggle GPU kernel: installs the pinned training stack and runs one shard of the
experiment plan per visible GPU, then merges the results into /kaggle/working."""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "terratorch==1.2.13",
                "segmentation-models-pytorch"], check=True)
import torch  # noqa: E402

data = Path(glob.glob("/kaggle/input/**/stats.json", recursive=True)[0]).parent
out = Path("/kaggle/working")
plan = data / "plan.json"
n = max(1, torch.cuda.device_count())
print("data", data, "gpus", n, [torch.cuda.get_device_name(i) for i in range(n)], flush=True)
procs = []
for i in range(n):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
    cmd = [sys.executable, str(data / "train_gpu.py"), "--data", str(data), "--out", str(out / f"part{i}"),
           "--part", str(i), "--parts", str(n)]
    if plan.exists():
        cmd += ["--plan", str(plan)]
    procs.append(subprocess.Popen(cmd, env=env, stdout=open(out / f"log_part{i}.txt", "w"), stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]
rows = []
for f in sorted(out.glob("part*/runs_gpu.jsonl")):
    rows += [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
(out / "runs_gpu.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print("exit codes", codes, "runs", len(rows), flush=True)
for i in range(n):
    print(f"---- log part{i} (tail)\n" + (out / f"log_part{i}.txt").read_text()[-3000:], flush=True)
