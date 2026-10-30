"""Design D's sweep: one clean int8 pretrain, then iter.py runs from it in parallel processes.

Results: $OUT/base.json, $OUT/base.pt (the pretrained state, about 17 MB at w256), $OUT/runs/<name>.json,
$OUT/logs/<name>.log, $OUT/sweep.json. ``--base`` reuses a pretrained state instead of training one.
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

# (n_E, n_R): values in E's and R's alphabets. The design point the regimes are compared at:
DESIGN = (3, 19)


def d1(size, tokens, ft_tokens, post_tokens):
    pre = {"name": "base", "size": size, "tokens": tokens, "lr": 2e-3, "warmup": 200, "eval_every": 250}
    ft = {"size": size, "tokens": ft_tokens, "post_tokens": post_tokens}
    ne, nr = DESIGN
    tag = f"e{ne}r{nr}"
    runs = [{**ft, "name": "clean", "regime": "clean"}]
    for reg in ("none", "eonly", "absorb1", "absorb8", "absorb64", "keep1", "keep8", "keep64"):
        runs.append({**ft, "name": f"{tag}-{reg}", "regime": reg, "n_e": ne, "n_r": nr})
    runs.append({**ft, "name": f"{tag}-perstep", "regime": "perstep", "n_e": ne, "n_r": nr, "post": True,
                 "d2_ranks": [8, 32]})
    for ne2, nr2 in ((2, 1), (1, 19), (2, 16), (3, 10), (4, 16), (3, 32), (3, 64), (19, 3), (10, 10), (2, 129)):
        runs.append({**ft, "name": f"e{ne2}r{nr2}-perstep", "regime": "perstep", "n_e": ne2, "n_r": nr2})
    for rf in (1 / 8, 1 / 4):
        runs.append({**ft, "name": f"designA-r{round(1 / rf)}-e2", "regime": "designA", "n_e": 2, "n_r": 256,
                     "rank_frac": rf})
    return pre, runs


PRESETS = {
    "d1": lambda: d1("w256", 20_000_000, 4_000_000, 1_000_000),
    "smoke": lambda: d1("w128", 200_000, 65_536, 32_768),
}


def run_one(cmd, cfg, data, out, threads, base=None, save=None):
    res = out / ("base.json" if cmd == "pretrain" else f"runs/{cfg['name']}.json")
    log = out / "logs" / f"{cfg['name']}.log"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    cfg = {**cfg, "threads": threads}
    argv = [sys.executable, "-u", str(HERE / "iter.py"), cmd, "--config", json.dumps(cfg), "--data", str(data),
            "--out", str(res)]
    argv += ["--save", str(save)] if save else ["--base", str(base)]
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env)
    row = {"name": cfg["name"], "rc": p.returncode, "wall_s": time.time() - t0}
    if p.returncode == 0:
        r = json.loads(res.read_text())
        row.update({k: v for k, v in r.items() if k.startswith("val") and not isinstance(v, list)})
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--base", default=None, help="a pretrained base.pt to reuse")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pre-threads", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    pre, runs = PRESETS[a.preset]()
    if a.only:
        runs = [r for r in runs if r["name"] in a.only]
    base = out / "base.pt"
    rows = []
    if a.base:
        shutil.copy(a.base, base)
    else:
        rows.append(run_one("pretrain", pre, a.data, out, a.pre_threads, save=base))
        if rows[-1]["rc"] != 0:
            sys.exit("pretrain failed")
    print(f"sweep {a.preset}: {len(runs)} runs, workers {a.workers} x {a.threads} threads", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        rows += list(ex.map(lambda c: run_one("run", c, a.data, out, a.threads, base=base), runs))
    (out / "sweep.json").write_text(json.dumps({"preset": a.preset, "rows": rows}, indent=1))
    bad = [r for r in rows if r["rc"] != 0]
    print(f"sweep {a.preset}: {len(rows) - len(bad)} ok, {len(bad)} failed", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
