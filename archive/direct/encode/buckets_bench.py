"""Measure B-Ligero's remaining prover buckets (``buckets.py``) at the campaign shapes on the pod.

    python3 -m encode.buckets_bench --reps 7 --out buckets_bench.json [--quick]

Buckets and shapes (B = 4096, K = 1536, 2^-128, BabyBear / BabyBear^6; ``explore.tensor`` counts):
  irs_row_combination   B: 310,602 rows x 4096 (5.09 GB read once), A: 48,873 rows      scalar F_c coefficient per row
  quadratic_test        B: 33,485 product rows (x, y, z triples re-encoded on coset 1, 8,192-point domain)
  linear_test           B: 310,602 rows; r^T A rows = 6 coordinate rows per witness row, re-encoded on coset 1, times the
                        re-encoded witness row.  The r^T A slice is a RESIDENT PSEUDO-RANDOM buffer reused for every
                        slice: the sparse constraint combination that produces it is NOT measured (labelled).
  batch_inverse         B: 125,895,743 helpers in F_c (3.0 GB in, 3.0 GB out, 3.0 GB scratch)
  open_columns          t = 192 columns of the codeword: gather from the kept cosets store / re-encode with the
                        gather-mode encoder (no store); Merkle paths (host side) from a 16,384-leaf tree.
Every bucket is checked bit-exactly against ``buckets_ref`` on a small shape before it is timed.  CUDA events, warm,
median of ``--reps``; slice loops captured as CUDA graphs.  Inputs are resident pseudo-random data (data-independent
cost).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "ml" / "kernels"))

import cupy as cp
from hash_gpu import merkle

from encode import babybear as bb
from encode import buckets as bk
from encode import buckets_ref as ref
from encode.encode_ntt_simt import RSEncoderSIMT

ROWS = {"B": 310_602, "A": 48_873}
QUAD_ROWS = {"B": 33_485}
HELPERS = {"B": 125_895_743}
T_OPEN = 192
PEAK_BW = 1.008e12
K, D = bb.K_CODE, 6


def device_info() -> dict:
    props = cp.cuda.runtime.getDeviceProperties(0)
    _free, total = cp.cuda.runtime.memGetInfo()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                         timeout=10, check=False).stdout.strip()
    return {"gpu": props["name"].decode(), "sms": props["multiProcessorCount"], "l2_bytes": props["l2CacheSize"],
            "device_memory_bytes": total, "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
            "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(), "cupy": cp.__version__, "numpy": np.__version__,
            "nvidia_smi_driver": smi}


def timed(fn, reps: int):
    fn()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(cp.cuda.get_elapsed_time(s, e))
    return statistics.median(ts), ts


def rand_u32(shape, seed: int, lo: int = 0):
    return cp.random.RandomState(seed).randint(lo, bb.P, size=shape, dtype=cp.int32).view(cp.uint32)


def mem() -> int:
    free, total = cp.cuda.runtime.memGetInfo()
    return total - free


def free_all():
    cp.get_default_memory_pool().free_all_blocks()


def graph_of(fn, stream):
    fn()                                  # eager warm-up (compiles)
    stream.synchronize()
    stream.begin_capture()
    fn()
    return stream.end_capture()


# -- checks -------------------------------------------------------------------------------------------------------
def check_all(enc: RSEncoderSIMT) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    cb = bk.Combiner(chunk_rows=32)
    rows = 300
    W = bb.fill_rows(rows, seed=4)
    r = rng.integers(0, bb.P, (rows, D)).astype(np.uint64)
    acc = cb.acc(rows, K)
    cb.scalar(cp.asarray(W), cp.asarray(bk.mont(r)), acc)
    out["irs_row_combination"] = bool(np.array_equal(cp.asnumpy(cb.reduce(acc)).astype(np.uint64), ref.irs_row_combination(r, W)))
    rows = 70
    x, y, z = (bb.fill_rows(rows, seed=s) for s in (6, 7, 8))
    r = rng.integers(0, bb.P, (rows, D)).astype(np.uint64)
    xd, yd, zd = (cp.asarray(a) for a in (x, y, z))
    c1 = [enc(a, cp.empty((rows, K), dtype=cp.uint32), write_systematic=False, ncosets=1) for a in (xd, yd, zd)]
    acc = cb.acc(rows, 2 * K)
    cb.quadratic(xd, yd, zd, *c1, cp.asarray(np.concatenate([bk.mont(r), bk.mont2(r)], axis=1)), acc)
    out["quadratic_test"] = bool(np.array_equal(cp.asnumpy(cb.reduce(acc)).astype(np.uint64), ref.quadratic_test(r, x, y, z)))
    rows = 40
    u = bb.fill_rows(rows, seed=9)
    rA = np.stack([bb.fill_rows(rows, seed=10 + c) for c in range(D)])
    ud, rAd = cp.asarray(u), cp.asarray(rA)
    u_c1 = enc(ud, cp.empty((rows, K), dtype=cp.uint32), write_systematic=False, ncosets=1)
    rA_c1 = cp.empty((D, rows, K), dtype=cp.uint32)
    for c in range(D):
        enc(rAd[c], rA_c1[c], write_systematic=False, ncosets=1)
    acc = cb.acc(rows, 2 * K)
    cb.linear(ud, u_c1, rAd, rA_c1, acc)
    out["linear_test"] = bool(np.array_equal(cp.asnumpy(cb.reduce(acc)).astype(np.uint64), ref.linear_test(rA, u)))
    N = 1000
    d = rng.integers(1, bb.P, (D, N)).astype(np.uint64)
    got = cp.asnumpy(bk.BatchInverter(threads_total=64, block=32)(cp.asarray(bk.mont(d)))).astype(np.uint64)
    got = (got * np.uint64(pow(bb.R_MOD_P, -1, bb.P))) % np.uint64(bb.P)
    out["batch_inverse"] = bool(np.array_equal(got, ref.batch_inverse(d)))
    rows = 200
    W = bb.fill_rows(rows, seed=12)
    code = bb.encode_rows(W, bb.N_CODE, layout="coset")
    cols = np.sort(rng.choice(bb.N_CODE, T_OPEN, replace=False)).astype(np.int64)
    Wd = cp.asarray(W)
    store = enc(Wd, cp.empty((rows, 3 * K), dtype=cp.uint32), write_systematic=False)
    g = bk.ColumnGather()
    cos, sysc = cols[cols >= K], cols[cols < K]
    ok = np.array_equal(cp.asnumpy(g(store, cp.asarray((cos - K).astype(np.int32)))).astype(np.uint64), code[:, cos].T)
    ok = ok and np.array_equal(cp.asnumpy(g(Wd, cp.asarray(sysc.astype(np.int32)))).astype(np.uint64), code[:, sysc].T)
    out["open_columns_gather"] = bool(ok)
    eg = RSEncoderSIMT(gather=True)
    out["open_columns_encode_gather"] = bool(np.array_equal(cp.asnumpy(eg.gather(Wd, cols)).astype(np.uint64), code[:, cols].T))
    return out


# -- buckets --------------------------------------------------------------------------------------------------------
def bench_irs(shape: str, R: int, reps: int, chunk_rows_list=(256, 1024, 4096)) -> list[dict]:
    W = rand_u32((R, K), 1)
    r = rand_u32((R, D), 2)
    res = []
    for cr in chunk_rows_list:
        cb = bk.Combiner(chunk_rows=cr)
        acc = cb.acc(R, K)
        out = cp.empty((D, K), dtype=cp.uint32)

        def run(acc=acc, cb=cb, W=W, r=r, out=out):
            acc.fill(0)
            cb.scalar(W, r, acc)
            cb.reduce(acc, out)
        med, ts = timed(run, reps)
        res.append({"name": f"irs_row_combination_c{cr}", "bucket": "irs_row_combination", "shape": shape, "rows": R, "chunk_rows": cr,
                    "ms_median": med, "ms_all": ts, "us_per_row": med * 1e3 / R, "bytes_read": R * K * 4,
                    "read_floor_ms": R * K * 4 / PEAK_BW * 1e3, "mont_muls": R * K * D, "g_mont_muls_per_s": R * K * D / (med / 1e3) / 1e9,
                    "device_memory_in_use_bytes": mem()})
        print(f"{shape} {res[-1]['name']}: {med:.2f} ms ({res[-1]['us_per_row']:.3f} us/row), read floor {res[-1]['read_floor_ms']:.1f} ms, "
              f"{res[-1]['g_mont_muls_per_s']:.0f} G mont-mul/s", flush=True)
    free_all()
    return res


def bench_quadratic(shape: str, Q: int, reps: int, enc: RSEncoderSIMT, slice_list=(256, 512, 1024)) -> list[dict]:
    x, y, z = (rand_u32((Q, K), s) for s in (3, 4, 5))
    r12 = rand_u32((Q, 2 * D), 6)
    res = []
    stream = cp.cuda.Stream(non_blocking=True)
    for sl in slice_list:
        cb = bk.Combiner(chunk_rows=32)
        acc = cb.acc(sl, 2 * K)
        out = cp.empty((D, 2 * K), dtype=cp.uint32)
        bufs = [cp.empty((sl, K), dtype=cp.uint32) for _ in range(3)]

        def loop(acc=acc, cb=cb, bufs=bufs, out=out, sl=sl, x=x, y=y, z=z, r12=r12):
            acc.fill(0)
            for r0 in range(0, Q, sl):
                s = min(sl, Q - r0)
                for src, b in zip((x, y, z), bufs):
                    enc(src[r0:r0 + s], b[:s], write_systematic=False, ncosets=1)
                cb.quadratic(x[r0:r0 + s], y[r0:r0 + s], z[r0:r0 + s], *bufs, r12[r0:r0 + s], acc)
            cb.reduce(acc, out)
        with stream:
            g = graph_of(loop, stream)
            med, ts = timed(lambda g=g: g.launch(stream=stream), reps)
        res.append({"name": f"quadratic_test_s{sl}", "bucket": "quadratic_test", "shape": shape, "rows": Q, "slice_rows": sl,
                    "ms_median": med, "ms_all": ts, "us_per_row": med * 1e3 / Q, "encoder_calls_per_row": 3, "transforms_per_row": 6,
                    "device_memory_in_use_bytes": mem()})
        print(f"{shape} {res[-1]['name']}: {med:.2f} ms ({res[-1]['us_per_row']:.3f} us/product-row)", flush=True)
    free_all()
    return res


def bench_linear(shape: str, R: int, reps: int, enc: RSEncoderSIMT, slice_list=(128, 256, 512)) -> list[dict]:
    W = rand_u32((R, K), 7)
    res = []
    stream = cp.cuda.Stream(non_blocking=True)
    for sl in slice_list:
        rA = rand_u32((D, sl, K), 8)                    # resident pseudo-random r^T A slice, reused (NOT generated per slice)
        cb = bk.Combiner(chunk_rows=32)
        acc = cb.acc(sl, 2 * K)
        out = cp.empty((D, 2 * K), dtype=cp.uint32)
        u_c1 = cp.empty((sl, K), dtype=cp.uint32)
        rA_c1 = cp.empty((D, sl, K), dtype=cp.uint32)

        def loop(acc=acc, cb=cb, out=out, sl=sl, W=W, rA=rA, u_c1=u_c1, rA_c1=rA_c1):
            acc.fill(0)
            for r0 in range(0, R, sl):
                s = min(sl, R - r0)
                enc(W[r0:r0 + s], u_c1[:s], write_systematic=False, ncosets=1)
                for c in range(D):
                    enc(rA[c, :s], rA_c1[c, :s], write_systematic=False, ncosets=1)
                cb.linear(W[r0:r0 + s], u_c1, rA[:, :s], rA_c1[:, :s], acc)
            cb.reduce(acc, out)
        with stream:
            g = graph_of(loop, stream)
            med, ts = timed(lambda g=g: g.launch(stream=stream), reps)
        res.append({"name": f"linear_test_s{sl}", "bucket": "linear_test", "shape": shape, "rows": R, "slice_rows": sl,
                    "ms_median": med, "ms_all": ts, "us_per_row": med * 1e3 / R, "encoder_calls_per_row": 7, "transforms_per_row": 14,
                    "rA_generation_included": False, "device_memory_in_use_bytes": mem()})
        print(f"{shape} {res[-1]['name']}: {med:.2f} ms ({res[-1]['us_per_row']:.3f} us/row)", flush=True)
    free_all()
    return res


def bench_inverse(shape: str, N: int, reps: int, threads_list=(32768, 65536, 131072, 262144)) -> list[dict]:
    d = rand_u32((D, N), 9, lo=1)
    out = cp.empty_like(d)
    res = []
    for T in threads_list:
        inv = bk.BatchInverter(threads_total=T, block=256)
        med, ts = timed(lambda inv=inv: inv(d, out), reps)
        res.append({"name": f"batch_inverse_T{T}", "bucket": "batch_inverse", "shape": shape, "n": N, "threads": T, "segment": -(-N // T),
                    "ms_median": med, "ms_all": ts, "ns_per_element": med * 1e6 / N, "bytes_moved": 5 * D * N * 4,
                    "traffic_floor_ms": 5 * D * N * 4 / PEAK_BW * 1e3, "device_memory_in_use_bytes": mem()})
        print(f"{shape} {res[-1]['name']}: {med:.2f} ms ({res[-1]['ns_per_element']:.2f} ns/element), traffic floor {res[-1]['traffic_floor_ms']:.1f} ms", flush=True)
    free_all()
    return res


def bench_openings(shape: str, R: int, reps: int, enc: RSEncoderSIMT, hash: str = "blake3") -> list[dict]:
    rng = np.random.default_rng(13)
    cols = np.sort(rng.choice(bb.N_CODE, T_OPEN, replace=False)).astype(np.int64)
    W = rand_u32((R, K), 10)
    res = []
    # (ii) re-encode with the gather-mode encoder: no store needed
    eg = RSEncoderSIMT(gather=True)
    out = cp.empty((T_OPEN, R), dtype=cp.uint32)
    med, ts = timed(lambda: eg.gather(W, cols, out), reps)
    res.append({"name": "open_columns_encode_gather", "bucket": "open_columns", "shape": shape, "rows": R, "t": T_OPEN, "ms_median": med,
                "ms_all": ts, "us_per_row": med * 1e3 / R, "opened_bytes": T_OPEN * R * 4, "device_memory_in_use_bytes": mem()})
    print(f"{shape} {res[-1]['name']}: {med:.2f} ms", flush=True)
    # (i) gather from the kept cosets store (needs R x 3k words resident: 15.3 GB at B)
    need = R * 3 * K * 4 + R * K * 4 + 2 * T_OPEN * R * 4
    if need < cp.cuda.runtime.memGetInfo()[1] - 1.5e9:
        store = cp.empty((R, 3 * K), dtype=cp.uint32)
        for r0 in range(0, R, 65536):
            enc(W[r0:r0 + 65536], store[r0:r0 + 65536], write_systematic=False)
        g = bk.ColumnGather()
        cos = cp.asarray((cols[cols >= K] - K).astype(np.int32))
        sysc = cp.asarray(cols[cols < K].astype(np.int32))
        o1 = cp.empty((int(cos.shape[0]), R), dtype=cp.uint32)
        o2 = cp.empty((int(sysc.shape[0]), R), dtype=cp.uint32)

        def run(g=g, store=store, cos=cos, o1=o1, W=W, sysc=sysc, o2=o2):
            g(store, cos, o1)
            g(W, sysc, o2)
        med, ts = timed(run, reps)
        res.append({"name": "open_columns_gather_from_store", "bucket": "open_columns", "shape": shape, "rows": R, "t": T_OPEN, "ms_median": med,
                    "ms_all": ts, "us_per_row": med * 1e3 / R, "opened_bytes": T_OPEN * R * 4, "store_bytes": R * 3 * K * 4,
                    "device_memory_in_use_bytes": mem()})
        print(f"{shape} {res[-1]['name']}: {med:.2f} ms (store {R * 3 * K * 4 / 1e9:.1f} GB resident)", flush=True)
    else:
        print(f"{shape} open_columns_gather_from_store: skipped (needs {need / 1e9:.1f} GB)", flush=True)
    # (iii) Merkle paths for the t columns (host side, from the device tree levels): t x depth digests
    leaves = rand_u32((bb.N_CODE, 8), 11)
    levels = [leaves]
    while levels[-1].shape[0] > 1:
        levels.append(merkle.node_level(levels[-1], hash))
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        paths = bk.merkle_paths(levels, cols)
    wall = (time.perf_counter() - t0) / reps * 1e3
    res.append({"name": "open_columns_merkle_paths_host", "bucket": "open_columns", "shape": shape, "rows": R, "t": T_OPEN, "ms_median": wall,
                "ms_all": [wall], "depth": len(levels) - 1, "path_bytes": T_OPEN * (len(levels) - 1) * 32, "timing": "wall clock incl. D2H",
                "paths_returned": len(paths)})
    print(f"{shape} {res[-1]['name']}: {wall:.2f} ms wall ({len(levels) - 1} levels)", flush=True)
    free_all()
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="buckets_bench.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--buckets", default="irs,quadratic,linear,inverse,openings")
    a = ap.parse_args(argv)
    if a.quick:
        ROWS.update({"B": 20_000, "A": 8_192})
        QUAD_ROWS["B"] = 4_000
        HELPERS["B"] = 4_000_000
        a.reps = min(a.reps, 3)
    res = {"device": device_info(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(a), "rows": dict(ROWS),
           "quad_rows": dict(QUAD_ROWS), "helpers": dict(HELPERS), "t_open": T_OPEN, "results": [], "bench_kind": "bucket",
           "scope": "component: B-Ligero prover buckets (IRS row combination, quadratic and linear tests, helper batch inversion, column "
                    "openings) measured standalone on resident pseudo-random data; r^T A generation, witness generation and the transcript "
                    "are not included",
           "bit_exact_note": "every bucket kernel equals buckets_ref (numpy / Python-int F_c) on a small shape (bench.checks)"}
    print(json.dumps(res["device"]), flush=True)
    enc = RSEncoderSIMT()
    res["checks"] = check_all(enc)
    print("checks:", res["checks"], flush=True)
    want = a.buckets.split(",")
    for shape in ("B", "A"):
        if "irs" in want:
            res["results"] += bench_irs(shape, ROWS[shape], a.reps)
            free_all()
    if "quadratic" in want:
        res["results"] += bench_quadratic("B", QUAD_ROWS["B"], a.reps, enc)
        free_all()
    if "linear" in want:
        res["results"] += bench_linear("B", ROWS["B"], a.reps, enc)
        free_all()
    if "inverse" in want:
        res["results"] += bench_inverse("B", HELPERS["B"], a.reps)
        free_all()
    if "openings" in want:
        for shape in ("B", "A"):
            res["results"] += bench_openings(shape, ROWS[shape], a.reps, enc)
            free_all()
    res["all_bit_exact"] = all(res["checks"].values())
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["run_id"] = os.environ.get("RESEARCH_RUN_ID")
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        Path(rd, "buckets_bench.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
