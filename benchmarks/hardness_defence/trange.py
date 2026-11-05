"""HT-GENEROUS: the range the exact split's T = (E − 2)·B reaches, against the worst case its digits are priced for.

    python benchmarks/hardness_defence/trange.py --hf REPO --layers L --match q_proj --ranks R,... --out DIR

`rotated.integer_split` gives T d int8 digits for 2·Σ|B_j|, the most |T| can be over every E on design U's 4 values.
E is drawn live on those values, so a server that holds E when it forms T can split T into the digits its drawn range
needs, and run another digit pass only when that range exceeds them: exact for every E. Per matrix and rank this
records B's worst case (width, d), T's measured range over --rows drawn rows of E, and Hoeffding's bound on one entry
exceeding a digit count's range: each term (E − 2 + ½)·B_kj lies in an interval of length 3·|B_kj|, so
P(|T − μ| ≥ t) ≤ 2·exp(−2t²/(9·‖B_j‖²)) with μ = −½·Σ_k B_kj. `log10_tail` is that bound's log10 at the two-digit
range less |μ|, for the worst column.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hardness_public"))
import rotated  # noqa: E402

TWO = 32639


def digits(width: float) -> int:
    return 1 if width <= 127 else 2 if width <= TWO else 3


def work(job) -> dict:
    name, src, ranks, rows = job
    t0 = time.time()
    q, c = rotated.load_codes(src)
    n, K = q.shape
    _U, s, Vt = np.linalg.svd(q.astype(np.float32), full_matrices=False)
    del _U
    rng = np.random.default_rng(0)
    E2 = rng.integers(0, rotated.A_NOISE, (rows, K), dtype=np.int64) - 2
    rec = {"tensor": name, "shape": [n, K], "c": c, "rows": rows, "ranks": {}}
    ranks = [r for r in ranks if 0 < r < min(n, K)]
    if not ranks:
        return rec | {"seconds": time.time() - t0}
    # B's column j is rint(β_j·V_j), β_j from σ_j alone, so B at rank r is the first r columns of B at the top rank
    R = max(ranks)
    _S, _A, B, d_top = rotated.integer_split(q, s, Vt, R)
    del _S, _A
    T = np.abs(E2 @ B).max(axis=0)
    absum = np.abs(B).sum(axis=0)
    mu = np.abs(-0.5 * B.sum(axis=0))
    norm2 = (B.astype(np.float64) ** 2).sum(axis=0)
    t = np.maximum(TWO - mu, 0.0)
    tail = np.log10(2.0) - 2 * t**2 / (9 * np.maximum(norm2, 1e-9)) / math.log(10)
    for r in ranks:
        width = int(2 * absum[:r].max())
        tmax = int(T[:r].max())
        rec["ranks"][str(r)] = {"d": digits(width), "width": width, "Bmax": int(np.abs(B[:, :r]).max()),
                                "T_absmax": tmax, "d_drawn": digits(tmax), "mu_absmax": float(mu[:r].max()),
                                "sd_max": float(math.sqrt(1.25 * norm2[:r].max())), "log10_tail": float(tail[:r].max())}
    assert rec["ranks"][str(R)]["d"] == d_top
    rec["seconds"] = time.time() - t0
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", required=True)
    ap.add_argument("--target", type=float, default=0.02)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--match", default=None)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--rows", type=int, default=1024, help="rows of E drawn")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    jobs = rotated.hf_jobs(args.hf, args.target, layers)
    if args.match:
        jobs = [j for j in jobs if rotated.keep(j[0], None, args.match.split(","))]
    ranks = [int(x) for x in args.ranks.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    with open(args.out / "trange.jsonl", "w") as fh, concurrent.futures.ProcessPoolExecutor(args.workers) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(work, (t, src, ranks, args.rows)) for t, src in jobs]):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            worst = max(rec["ranks"].values(), key=lambda v: v["T_absmax"], default=None)
            print(rec["tensor"], rec["shape"], {r: (v["d"], v["d_drawn"], v["T_absmax"]) for r, v in rec["ranks"].items()},
                  f"worst log10 tail {max((v['log10_tail'] for v in rec['ranks'].values()), default=None)}",
                  f"{rec['seconds']:.0f}s", flush=True)
            assert worst is None or worst["d_drawn"] <= worst["d"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
