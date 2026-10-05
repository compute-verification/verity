"""Time the fused logUp round (``round_logup``) on the pod: kernel time, GB/s, fraction of the HBM floor.

Usage (pod): ``python -m tensor.fused.bench_logup --out results/fused_logup.json``.
Bytes per kernel of ``n`` input elements (fold+eval): read ``5 x 24 n``, write ``5 x 24 n/2`` = ``180 n``.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time

import torch

from .. import reference as ref
from . import round_logup, round_simt

HBM = 1008e9
ELEM = 24


def time_ms(fn, reps: int = 11) -> float:
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fused_logup.json")
    ap.add_argument("--sizes", default="2097152,1048576,262144,65536,16384,4096")
    args = ap.parse_args()
    dev = "cuda"
    rng = random.Random(7)
    rows = []
    for n in (int(x) for x in args.sizes.split(",")):
        tabs = [round_simt.FusedTable(torch.randint(0, ref.P, (6, n), dtype=torch.int64, device=dev).to(torch.int32)) for _ in range(5)]
        lam = ref.random_ext(rng)
        r = ref.random_ext(rng)
        for block, nw in ((256, 4), (128, 4), (512, 8)):
            kern = round_logup.LogUpRound(dev, block=block, num_warps=nw)
            lam_m, r_m = kern.challenge(lam), kern.challenge(r)
            outs = [kern.alloc_out(n // 2) for _ in range(5)]
            s_buf = torch.zeros((kern.grid_size(n, True), 24), dtype=torch.int64, device=dev)
            t0 = time.time()
            ms = time_ms(lambda: kern.launch(tabs, lam_m, r_m, True, outs, s_buf))
            ms_fold = time_ms(lambda: kern.launch(tabs, lam_m, r_m, False, outs, s_buf))
            by = 180 * n
            row = {"n": n, "block": block, "num_warps": nw, "fold_eval_us": ms * 1e3, "fold_only_us": ms_fold * 1e3,
                   "bytes": by, "gbps": by / (ms * 1e-3) / 1e9, "hbm_floor_us": by / HBM * 1e6,
                   "frac_of_floor": (by / HBM * 1e6) / (ms * 1e3), "ext_muls": 7.5 * n,
                   "gmul_per_s": 7.5 * n / (ms * 1e-3) / 1e9, "wall_s": time.time() - t0}
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(args.out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
