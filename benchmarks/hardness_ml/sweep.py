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


def lite(size, tokens, threads, lr=2e-3):
    """The scaling subset at one size: baseline ladder, A at r/n 1/32, 1/8, 1/4 with D₁RD₂, B at ρ 0.25, 0.5, 1,
    and live noise k = 16 on the baseline and on A 1/8."""
    b = {"size": size, "tokens": tokens, "lr": lr, "threads": threads}
    cfgs = [{**b, "name": f"{size}-base-D{tokens // f}", "tokens": tokens // f} for f in (1, 2, 4, 8, 16)]
    cfgs += [{**b, "name": f"{size}-A{inv}-diag", "design": "A", "rank_frac": 1 / inv, "diag": True}
             for inv in (32, 8, 4)]
    cfgs += [{**b, "name": f"{size}-B{rho:g}", "design": "B", "rho": rho} for rho in (0.25, 0.5, 1.0)]
    cfgs += [{**b, "name": f"{size}-int8-k16", "int8": True, "noise_k": 16},
             {**b, "name": f"{size}-A8-diag-int8-k16", "design": "A", "rank_frac": 1 / 8, "diag": True,
              "int8": True, "noise_k": 16, "r_dist": "int8"}]
    return cfgs


def replan1(size="w256", tokens=20_000_000, lr=2e-3, threads=4):
    """After the first signal: the noise knee on the baseline shape and on B, B's int8 R and small ρ, B on all
    linears, and seeds for the baseline, B ρ = 0.5 and A 1/8 (the measurement's noise floor)."""
    b = {"size": size, "tokens": tokens, "lr": lr, "threads": threads}
    B5 = {**b, "design": "B", "rho": 0.5}
    cfgs = [{**b, "name": f"{size}-int8-k{k}", "int8": True, "noise_k": k} for k in (2, 4, 8)]
    cfgs += [{**B5, "name": f"{size}-B0.5-int8-k{k}", "int8": True, "noise_k": k, "r_dist": "int8"}
             for k in (0, 4, 16)]
    cfgs += [{**b, "name": f"{size}-B0.05", "design": "B", "rho": 0.05},
             {**b, "name": f"{size}-B0.25-all", "design": "B", "rho": 0.25, "harden": "all"},
             {**B5, "name": f"{size}-B0.5-lr{lr * 2:g}", "lr": lr * 2}]
    cfgs += [{**b, "name": f"{size}-base-D{tokens}-s{s}", "seed": s} for s in (1, 2)]
    cfgs += [{**B5, "name": f"{size}-B0.5-s{s}", "seed": s} for s in (1, 2)]
    cfgs += [{**b, "name": f"{size}-A8-diag-s1", "design": "A", "rank_frac": 1 / 8, "diag": True, "seed": 1}]
    return cfgs


def replan2(lr=2e-3):
    """B with D₁(R+Δ)D₂ by ρ, on all linears and under int8 noise; A 1/4 on all linears; which half of the int8 arm
    (int8 R or int8 activations) helps A 1/8; whether B's growing gap saturates past compute-optimal tokens (w128 at
    60M, ~45 tokens per parameter); and B with diagonals at w512."""
    b = {"size": "w256", "tokens": 20_000_000, "lr": lr, "threads": 3}
    Bd = {**b, "design": "B", "diag": True}
    A8 = {**b, "design": "A", "rank_frac": 1 / 8, "diag": True}
    cfgs = [{**Bd, "name": f"w256-Bd{rho:g}", "rho": rho} for rho in (0.1, 0.25, 0.5)]
    cfgs += [{**Bd, "name": f"w256-Bd{rho:g}-all", "rho": rho, "harden": "all"} for rho in (0.25, 0.5)]
    cfgs += [{**Bd, "name": f"w256-Bd0.25-int8-k{k}", "rho": 0.25, "int8": True, "noise_k": k, "r_dist": "int8"}
             for k in (4, 16)]
    cfgs += [{**b, "name": "w256-A4-diag-all", "design": "A", "rank_frac": 1 / 4, "diag": True, "harden": "all"},
             {**A8, "name": "w256-A8-diag-rint8", "r_dist": "int8"},
             {**A8, "name": "w256-A8-diag-int8-k0-rgauss", "int8": True},
             {**A8, "name": "w256-A8-diag-int8-k0-s1", "int8": True, "r_dist": "int8", "seed": 1}]
    lng = {"size": "w128", "tokens": 60_000_000, "lr": lr, "threads": 4}
    cfgs += [{**lng, "name": f"w128-base-D{t}", "tokens": t} for t in (60_000_000, 40_000_000)]
    cfgs += [{**lng, "name": "w128-B0.5-D60M", "design": "B", "rho": 0.5},
             {**lng, "name": "w128-Bd0.5-D60M", "design": "B", "rho": 0.5, "diag": True},
             {**lng, "name": "w128-A8-diag-D60M", "design": "A", "rank_frac": 1 / 8, "diag": True}]
    w5 = {"size": "w512", "tokens": 20_000_000, "lr": lr, "threads": 6, "design": "B", "diag": True}
    cfgs += [{**w5, "name": f"w512-Bd{rho:g}", "rho": rho} for rho in (0.25, 0.5)]
    return cfgs


def replan3():
    """Whether B's penalty growing with width at one lr is the lr's doing: the baseline and B ρ = 0.25 at lr ∝ 1/n
    (2e-3 at w256; the μP rule for Adam on hidden matrices), where the baseline's weights move less as width grows."""
    cfgs = []
    for size, lr, threads in (("w512", 1e-3, 6), ("w128", 4e-3, 2)):
        b = {"size": size, "tokens": 20_000_000, "lr": lr, "threads": threads}
        cfgs += [{**b, "name": f"{size}-base-lr{lr:g}"},
                 {**b, "name": f"{size}-B0.25-lr{lr:g}", "design": "B", "rho": 0.25}]
    return cfgs


def replan4():
    """The baseline's lr one step past each width's best so far (4e-3 at w128 and w256, 2e-3 at w512), so the best-lr
    penalty compares B with a baseline whose lr optimum is bracketed; and B ρ = 0.25 one step past its own highest."""
    cfgs = []
    for size, base_lr, b_lr, threads in (("w128", 8e-3, 8e-3, 2), ("w256", 8e-3, 4e-3, 2), ("w512", 4e-3, 4e-3, 4)):
        b = {"size": size, "tokens": 20_000_000, "threads": threads}
        cfgs += [{**b, "name": f"{size}-base-lr{base_lr:g}", "lr": base_lr},
                 {**b, "name": f"{size}-B0.25-lr{b_lr:g}", "lr": b_lr, "design": "B", "rho": 0.25}]
    return cfgs


def replan5():
    """Width at fixed tokens per parameter: w256 at 80M tokens has w128-at-20M's 25.4 tokens per non-embedding
    parameter, since the penalty grows with tokens per parameter (w128 at 60M) and 20M tokens puts w512 at 1.6. A lr-2e-3
    ladder at 40M and 80M (with a 4e-3 bracket), B ρ = 0.25 and A 1/8 at their best 20M-token lrs; and w128's lr bracket."""
    w = {"size": "w256", "tokens": 80_000_000, "lr": 2e-3, "threads": 4}
    cfgs = [{**w, "name": "w256-base-D80000000"},
            {**w, "name": "w256-base-D40000000", "tokens": 40_000_000},
            {**w, "name": "w256-base-D80M-lr0.004", "lr": 4e-3},
            {**w, "name": "w256-B0.25-D80M", "design": "B", "rho": 0.25},
            {**w, "name": "w256-A8-diag-D80M-lr0.004", "lr": 4e-3, "design": "A", "rank_frac": 1 / 8, "diag": True}]
    b = {"size": "w128", "tokens": 20_000_000, "lr": 1.6e-2, "threads": 2}
    cfgs += [{**b, "name": "w128-base-lr0.016"}, {**b, "name": "w128-B0.25-lr0.016", "design": "B", "rho": 0.25}]
    return cfgs


PRESETS = {
    "bench": bench,
    "replan1": replan1,
    "replan2": replan2,
    "replan3": replan3,
    "replan4": replan4,
    "replan5": replan5,
    "w256": lambda: grid("w256", 20_000_000),
    "scale1": lambda: lite("w512", 20_000_000, 6) + lite("w128", 20_000_000, 2),
}


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
