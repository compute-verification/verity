"""Run a preset of train.py configs in parallel, each in its own process with a fixed thread count.

Results: $OUT/runs/<name>.json, logs: $OUT/logs/<name>.log, summary: $OUT/sweep.json. A config whose result already
exists in --resume-from (a previous run's out dir) is copied, not rerun.
"""

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bench():
    base = {"tokens": 60 * 32 * 256, "eval_every": 10_000, "final_eval_batches": 4, "warmup": 10}
    return [
        {**base, "name": "bench-s256-dense-bf16-t4", "size": "s256", "threads": 4},
        {**base, "name": "bench-s256-dense-fp32-t4", "size": "s256", "threads": 4, "bf16": False},
        {**base, "name": "bench-s256-dense-bf16-t8", "size": "s256", "threads": 8},
        {**base, "name": "bench-s256-A8-int8k16-t4", "size": "s256", "threads": 4, "design": "A", "diag": True,
         "int8": True, "noise_k": 16, "r_dist": "int8"},
        {**base, "name": "bench-m384-dense-bf16-t8", "size": "m384", "threads": 8},
        {**base, "name": "bench-t128-dense-bf16-t2", "size": "t128", "threads": 2},
    ]


def grid(size, tokens, lr=2e-3, tag=None):
    """The first-signal grid at one size: baseline token ladder and lr check, design A by r/n with and without
    D₁RD₂, design B by ρ, controls, and the int8 noise sweep. MLP matrices hardened unless the name says `all`."""
    tag = tag or size
    b = {"size": size, "tokens": tokens, "lr": lr}
    A8 = {**b, "design": "A", "rank_frac": 1 / 8, "diag": True}
    cfgs = [{**b, "name": f"{tag}-base-D{tokens // f}", "tokens": tokens // f} for f in (1, 2, 4, 8, 16, 32)]
    cfgs += [{**b, "name": f"{tag}-base-lr{x:g}", "lr": x} for x in (lr / 2, lr * 2)]
    for inv in (32, 16, 8, 4):
        for diag in (False, True):
            cfgs.append({**b, "name": f"{tag}-A{inv}{'-diag' if diag else ''}", "design": "A", "rank_frac": 1 / inv,
                         "diag": diag})
    cfgs += [
        {**A8, "name": f"{tag}-A8-diag-uvlr4", "uv_lr_mult": 4.0},
        {**A8, "name": f"{tag}-A8-diag-lr{lr * 2:g}", "lr": lr * 2},
        {**A8, "name": f"{tag}-A8-diag-all", "harden": "all"},
        {**b, "name": f"{tag}-lowrank8", "design": "lowrank", "rank_frac": 1 / 8},
        {**b, "name": f"{tag}-frozen-diag", "design": "frozen", "diag": True},
    ]
    cfgs += [{**b, "name": f"{tag}-B{rho:g}", "design": "B", "rho": rho} for rho in (0.1, 0.25, 0.5, 1.0)]
    cfgs += [{**b, "name": f"{tag}-B0.5-all", "design": "B", "rho": 0.5, "harden": "all"}]
    cfgs += [{**b, "name": f"{tag}-int8-k{k}", "int8": True, "noise_k": k} for k in (0, 16, 64)]
    cfgs += [{**A8, "name": f"{tag}-A8-diag-int8-k{k}", "int8": True, "noise_k": k, "r_dist": "int8"}
             for k in (0, 1, 16, 64, 96)]
    return cfgs


PRESETS = {"bench": bench, "w256": lambda: grid("w256", 20_000_000)}


def run_one(cfg, data, out, threads):
    res = out / "runs" / f"{cfg['name']}.json"
    log = out / "logs" / f"{cfg['name']}.log"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    cfg = {**cfg, "threads": threads}
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run([sys.executable, str(HERE / "train.py"), "--config", json.dumps(cfg), "--data", str(data),
                            "--out", str(res)], stdout=fh, stderr=subprocess.STDOUT, env=env)
    row = {"name": cfg["name"], "rc": p.returncode, "wall_s": time.time() - t0}
    if p.returncode == 0:
        r = json.loads(res.read_text())
        row.update(final_val=r["final_val"], tok_per_s=r["tok_per_s"])
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None, help="override every config's thread count")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--resume-from", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cfgs = PRESETS[a.preset]()
    if a.only:
        cfgs = [c for c in cfgs if c["name"] in a.only]
    todo, rows = [], []
    for c in cfgs:
        prev = [Path(d) / "runs" / f"{c['name']}.json" for d in a.resume_from]
        prev = [p for p in prev if p.exists()]
        if prev:
            shutil.copy(prev[0], out / "runs" / prev[0].name)
            rows.append({"name": c["name"], "rc": 0, "copied_from": str(prev[0])})
        else:
            todo.append(c)
    print(f"sweep {a.preset}: {len(todo)} to run, {len(rows)} copied, workers {a.workers}", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, c, a.data, out, a.threads or c.get("threads", 4)) for c in todo]
        rows += [f.result() for f in futs]
    (out / "sweep.json").write_text(json.dumps({"preset": a.preset, "rows": rows}, indent=1))
    bad = [r for r in rows if r["rc"] != 0]
    print(f"sweep {a.preset}: {len(rows) - len(bad)} ok, {len(bad)} failed", flush=True)


if __name__ == "__main__":
    main()
