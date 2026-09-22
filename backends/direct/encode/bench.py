"""Measured RS-encoder rates at the campaign's Ligero shapes (b-encode lane).  Run on the pod:

    python -m encode.bench --reps 7 --out result.json [--encoders simt,cublas,triton] [--shapes B,A] [--quick]

Shapes (BabyBear, [n=16384, k=4096], B = 4096 K = 1536 at 2^-128): Candidate B 310,602 rows (5.09 GB in, 20.36 GB
out), Candidate A 48,873 rows (0.80 GB in, 3.20 GB out).  The B batch does not fit beside its input on a 24 GB card,
so every encoder streams the batch in row slices (``--slice-rows``, default 65,536: 1.07 GB in + 4.29 GB out per
slice); the batch time is the CUDA-event time of the whole slice loop (kernel launches only; the slices' input is a
resident buffer, the cost is data-independent).  The unfused cuBLAS matrix-DFT is measured on ``--cublas-rows`` rows
and scaled linearly (labelled ``scaled_from_rows``).  Every encoder is checked bit-exactly against the numpy reference
on 64 rows before it is timed.  Timing: one warm-up, then ``reps`` timings, median.

Writes the raw JSON to ``--out`` and ``$RESEARCH_RUN_DIR/bench.json``; ``envelope.py`` wraps it in the campaign result
envelope (``verity_numerical.bench.contract``) as ``result.json``.
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

import cupy as cp
import torch

from encode import babybear as bb
from encode.encode_matdft_int8 import RSEncoderMatDFT
from encode.encode_ntt_simt import RSEncoderSIMT

ROWS = {"B": 310_602, "A": 48_873}
PEAK_BW = 1.008e12
PEAK_INT8_MACS = 330.3e12
PEAK_FP32 = 82.6e12


def device_info() -> dict:
    props = cp.cuda.runtime.getDeviceProperties(0)
    _free, total = cp.cuda.runtime.memGetInfo()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                         timeout=10, check=False).stdout.strip()
    return {"gpu": props["name"].decode(), "sms": props["multiProcessorCount"], "device_memory_bytes": total,
            "cuda_driver_version": cp.cuda.runtime.driverGetVersion(), "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cupy": cp.__version__, "numpy": np.__version__,
            "nvidia_smi_driver": smi, "triton": _triton_version()}


def _triton_version():
    try:
        import triton

        return triton.__version__
    except ImportError:
        return None


def timed(fn, reps: int) -> tuple[float, list[float]]:
    fn()
    cp.cuda.Device().synchronize()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = cp.cuda.Event()
        e = cp.cuda.Event()
        s.record()
        fn()
        e.record()
        e.synchronize()
        torch.cuda.synchronize()
        ts.append(cp.cuda.get_elapsed_time(s, e))
    return statistics.median(ts), ts


def mem_in_use() -> int:
    free, total = cp.cuda.runtime.memGetInfo()
    return total - free


def check(encoder, kind: str, rows: int = 64) -> bool:
    x = bb.fill_rows(rows, bb.K_CODE, seed=rows)
    ref = bb.encode_rows(x, bb.N_CODE, layout="coset")
    if kind == "simt":
        got = cp.asnumpy(encoder(cp.asarray(x))).astype(np.uint64)
    else:
        got = encoder(torch.from_numpy(x.view(np.int32)).cuda()).cpu().numpy().view(np.uint32).astype(np.uint64)
    return bool(np.array_equal(got, ref))


def slices(rows: int, slice_rows: int):
    out = []
    r0 = 0
    while r0 < rows:
        out.append(min(slice_rows, rows - r0))
        r0 += out[-1]
    return out


def bench_encoder(name: str, kind: str, encoder, rows: int, slice_rows: int, reps: int, write_systematic: bool = True,
                  measured_rows: int | None = None) -> dict:
    """Stream ``rows`` rows through ``encoder`` in slices; if ``measured_rows`` is given, time that many rows once
    and scale linearly."""
    mrows = measured_rows or rows
    sl = slices(mrows, slice_rows)
    big = max(sl)
    bufs = {}
    if kind == "simt":
        bufs["in"] = cp.asarray(bb.fill_rows(big, bb.K_CODE, seed=1))
        bufs["out"] = cp.empty((big, bb.N_CODE), dtype=cp.uint32)

        def run():
            for r in sl:
                encoder(bufs["in"][:r], bufs["out"][:r], write_systematic=write_systematic)
    else:
        bufs["in"] = torch.from_numpy(bb.fill_rows(big, bb.K_CODE, seed=1).view(np.int32)).cuda()
        bufs["out"] = torch.empty(big, bb.N_CODE, dtype=torch.int32, device="cuda")

        def run():
            for r in sl:
                encoder(bufs["in"][:r], bufs["out"][:r])
    med, ts = timed(run, reps)
    mem = mem_in_use()
    bufs.clear()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    scale = rows / mrows
    ms = med * scale
    words_out = rows * (bb.N_CODE if write_systematic else bb.N_CODE - bb.K_CODE)
    bytes_out = words_out * 4
    bytes_in = rows * bb.K_CODE * 4
    res = {"encoder": name, "kind": kind, "rows": rows, "measured_rows": mrows, "scaled_from_rows": None if scale == 1 else mrows,
           "slice_rows": slice_rows, "n_slices": len(sl), "write_systematic": write_systematic, "reps": reps,
           "ms_median": ms, "ms_all_measured": ts, "us_per_row": ms * 1e3 / rows,
           "bytes_in": bytes_in, "bytes_out": bytes_out,
           "codeword_gb_per_s": rows * bb.N_CODE * 4 / (ms / 1e3) / 1e9,          # full codeword produced per second
           "hbm_gb_per_s": (bytes_in + bytes_out) / (ms / 1e3) / 1e9,
           "hbm_floor_ms": (bytes_in + bytes_out) / PEAK_BW * 1e3,
           "device_memory_in_use_bytes": mem}
    res["fraction_of_hbm_floor"] = res["hbm_floor_ms"] / ms
    if kind == "simt":
        bf = RSEncoderSIMT.butterflies_per_row(bb.K_CODE) * rows
        res["butterflies"] = bf
        res["g_butterflies_per_s"] = bf / (ms / 1e3) / 1e9
        res["mont_muls_per_s"] = (bf + 3 * bb.K_CODE * rows) / (ms / 1e3)
    else:
        macs = RSEncoderMatDFT.limb_macs_per_row(bb.K_CODE) * rows
        res["limb_macs"] = macs
        res["t_macs_per_s"] = macs / (ms / 1e3) / 1e12
        res["fraction_of_int8_peak"] = res["t_macs_per_s"] * 1e12 / PEAK_INT8_MACS
        res["tensor_floor_ms"] = macs / PEAK_INT8_MACS * 1e3
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="bench.json")
    ap.add_argument("--encoders", default="simt,cublas,triton")
    ap.add_argument("--shapes", default="B,A")
    ap.add_argument("--slice-rows", type=int, default=65_536)
    ap.add_argument("--cublas-rows", type=int, default=8_192)
    ap.add_argument("--triton-rows", type=int, default=0, help="0 = full shape")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        ROWS.update({"B": 16_384, "A": 8_192})
        a.reps = min(a.reps, 3)
    encs = a.encoders.split(",")
    res = {"device": device_info(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(a),
           "rows": dict(ROWS), "encoders": [], "checks": {}, "peaks": {"hbm_bytes_per_s": PEAK_BW, "int8_macs_per_s": PEAK_INT8_MACS}}
    print(json.dumps(res["device"]), flush=True)
    todo = []
    if "simt" in encs:
        for variant, threads in (("v1", 256), ("v1", 512), ("v2", 128), ("v2", 256), ("v2", 512), ("v2", 1024)):
            e = RSEncoderSIMT(threads=threads, variant=variant)
            key = f"simt_{variant}_t{threads}"
            ok = check(e, "simt")
            res["checks"][key] = ok
            print(f"simt {variant} threads={threads} bit-exact: {ok}", flush=True)
            todo.append((f"ntt_simt_{variant}_t{threads}", "simt", e, True, None, key))
            if (variant, threads) == ("v2", 512):
                todo.append((f"ntt_simt_{variant}_t{threads}_nosys", "simt", e, False, None, key))
    if "cublas" in encs:
        e = RSEncoderMatDFT(variant="cublas")
        ok = check(e, "cublas")
        res["checks"]["cublas"] = ok
        print(f"matdft cublas bit-exact: {ok}", flush=True)
        todo.append(("matdft_int8_cublas", "cublas", e, True, a.cublas_rows, "cublas"))
    if "triton" in encs and "triton" in RSEncoderMatDFT.available():
        e = RSEncoderMatDFT(variant="triton")
        ok = check(e, "triton")
        res["checks"]["triton"] = ok
        print(f"matdft triton bit-exact: {ok}", flush=True)
        todo.append(("matdft_int8_triton", "triton", e, True, a.triton_rows or None, "triton"))
    for shape in a.shapes.split(","):
        rows = ROWS[shape]
        for name, kind, e, wsys, mrows, key in todo:
            if not res["checks"].get(key, False):
                print(f"skip {name}: failed bit-exactness", flush=True)
                continue
            if mrows is not None and a.quick:
                mrows = min(mrows, rows)
            slice_rows = a.slice_rows if kind != "cublas" else min(a.slice_rows, 2048)
            r = bench_encoder(name, kind, e, rows, slice_rows, a.reps, write_systematic=wsys, measured_rows=mrows)
            r["shape"] = shape
            res["encoders"].append(r)
            extra = (f"{r['g_butterflies_per_s']:.0f} G butterflies/s" if kind == "simt"
                     else f"{r['t_macs_per_s']:.1f} T MAC/s = {r['fraction_of_int8_peak']:.3f} of INT8 peak")
            print(f"{shape} {name}: {r['ms_median']:.1f} ms/batch ({r['us_per_row']:.2f} us/row), codeword {r['codeword_gb_per_s']:.0f} GB/s, "
                  f"HBM {r['hbm_gb_per_s']:.0f} GB/s = {r['fraction_of_hbm_floor']:.2f} of floor, {extra}, mem {r['device_memory_in_use_bytes'] / 2**30:.1f} GiB"
                  + (f" [scaled from {r['scaled_from_rows']} rows]" if r["scaled_from_rows"] else ""), flush=True)
    res["all_bit_exact"] = all(res["checks"].values())
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["run_id"] = os.environ.get("RESEARCH_RUN_ID")
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        Path(rd, "bench.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
