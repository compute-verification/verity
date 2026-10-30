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


def redml():
    """HT-REDML's first signal: public random low bits (``lsb``) against A at the T1 rank cap, r(1/n + 1/m) ≤ 1%,
    which at d 256 is r = 2, and at HT-ML's r = d/8; every hardened linear int8 with noise k = 1."""
    tokens = int(os.environ.get("REDML_TOKENS", 4_000_000))
    base = {"size": "s256", "tokens": tokens, "harden": "all", "int8": True, "noise_k": 1, "r_dist": "int8",
            "eval_every": 100, "final_eval_batches": 32}
    cap = 2 / 256
    return [
        {**base, "name": "redml-dense-float", "int8": False, "noise_k": 0},
        {**base, "name": "redml-int8w-k1", "design": "lsb", "lsb_bits": 0},
        {**base, "name": "redml-lsb2-k1", "design": "lsb", "lsb_bits": 2},
        {**base, "name": "redml-lsb4-k1", "design": "lsb", "lsb_bits": 4},
        {**base, "name": "redml-lsb6-k1", "design": "lsb", "lsb_bits": 6},
        {**base, "name": "redml-A-cap-k1", "design": "A", "diag": True, "rank_frac": cap},
        {**base, "name": "redml-lowrank-cap-k1", "design": "lowrank", "rank_frac": cap},
        {**base, "name": "redml-A-r32-k1", "design": "A", "diag": True, "rank_frac": 0.125},
    ]


PRESETS = {"bench": bench, "redml": redml}


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
