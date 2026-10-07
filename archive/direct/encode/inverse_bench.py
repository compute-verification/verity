"""Blocked Montgomery batch inversion (b-v2, task 4): exactness at small N against ``ref.batch_inverse`` and timing at the v1
and v2 helper counts, for ``BatchInverter`` (5 passes) vs ``BatchInverterBlocked`` (3 + 2/G passes).

    python inverse_bench.py --out /workspace/out/inverse_blocked.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "ml" / "kernels"))

from encode import babybear as bb  # noqa: E402
from encode import buckets as bk  # noqa: E402
from encode import buckets_ref as ref  # noqa: E402
from encode.buckets_v2 import D, BatchInverterBlocked  # noqa: E402

PEAK_BW = 1008e9  # RTX 4090 GDDR6X, B/s (same convention as buckets_bench)

def timed(fn, reps: int):
    import cupy as cp

    fn()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), ts


def exactness(gs) -> dict:
    import cupy as cp

    rng = np.random.default_rng(3)
    out = {}
    for N in (1000, 4097, 70000):
        d = rng.integers(1, bb.P, (D, N)).astype(np.uint64)
        want = ref.batch_inverse(d)
        dd = cp.asarray(bk.mont(d))
        rinv = np.uint64(pow(bb.R_MOD_P, -1, bb.P))
        for g in gs:
            for T, blk in ((64, 32), (4096, 256)):
                got = cp.asnumpy(BatchInverterBlocked(g=g, threads_total=T, block=blk)(dd)).astype(np.uint64)
                got = (got * rinv) % np.uint64(bb.P)
                out[f"N{N}_G{g}_T{T}"] = bool(np.array_equal(got, want))
    return out


def bench(N: int, reps: int, gs, threads_list) -> list[dict]:
    import cupy as cp

    rng = cp.random.default_rng(9)
    d = rng.integers(1, bb.P, (D, N), dtype=cp.uint32)
    out = cp.empty_like(d)
    res = []
    for T in threads_list:
        inv = bk.BatchInverter(threads_total=T, block=256)
        med, ts = timed(lambda: inv(d, out), reps)
        res.append({"name": f"batch_inverse_T{T}", "n": N, "threads": T, "g": 1, "ms_median": med, "ms_all": ts,
                    "bytes_moved": 5 * D * N * 4, "traffic_floor_ms": 5 * D * N * 4 / PEAK_BW * 1e3})
        print(f"N={N} {res[-1]['name']}: {med:.2f} ms (floor {res[-1]['traffic_floor_ms']:.2f})", flush=True)
        for g in gs:
            invb = BatchInverterBlocked(g=g, threads_total=T, block=256)
            med, ts = timed(lambda: invb(d, out), reps)
            bm = invb.bytes_moved(N)
            res.append({"name": f"batch_inverse_blocked_G{g}_T{T}", "n": N, "threads": T, "g": g, "ms_median": med, "ms_all": ts,
                        "bytes_moved": bm, "traffic_floor_ms": bm / PEAK_BW * 1e3})
            print(f"N={N} {res[-1]['name']}: {med:.2f} ms (floor {res[-1]['traffic_floor_ms']:.2f})", flush=True)
            del invb
    del d, out
    cp.get_default_memory_pool().free_all_blocks()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="inverse_blocked.json")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--n1", type=int, default=0, help="v1 helper count (b_measured_4090.json: 125,895,743)")
    ap.add_argument("--n2", type=int, default=43_613_350)
    ap.add_argument("--gs", default="2,4,8")
    args = ap.parse_args()
    gs = [int(x) for x in args.gs.split(",")]
    result = {"exact": exactness(gs)}
    print(result["exact"], flush=True)
    assert all(result["exact"].values()), result["exact"]
    result["bench"] = {}
    ns = {"v2": args.n2}
    if args.n1:
        ns["v1"] = args.n1
    for k, N in ns.items():
        result["bench"][k] = bench(N, args.reps, gs, (131072, 262144))
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
