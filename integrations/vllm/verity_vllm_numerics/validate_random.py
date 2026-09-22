"""Millions of random and edge-family tiles through one ``mma.sync`` vs the CPU models (GPU host only).

Families follow gpu/tensor-core-semantics/validate_tiles.py.  Every element goes through
the C++ twin (``cpp/tc_model.cpp``); a random subsample of ``--python-elements`` goes
through the pure-Python model as well (``veritor.core.silicon.tc_dot`` for BF16,
``verity_vllm_numerics.mma.tc_dot`` for FP16), so the validated object is the Python
semantics and the C++ twin is checked against both the GPU and the Python model.

Usage: python -m verity_vllm_numerics.validate_random bf16 --n-random 1000000 --out results/validate_bf16.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from . import cpu_model as CM
from . import probe as G
from .mma import TCModel, model_for, tc_dot

K = 16


def random_finite_fp32_bits(rng, n, exp_lo=1, exp_hi=254):
    e = rng.integers(exp_lo, exp_hi + 1, size=n).astype(np.uint32)
    m = rng.integers(0, 1 << 23, size=n).astype(np.uint32)
    s = rng.integers(0, 2, size=n).astype(np.uint32)
    return (s << 31) | (e << 23) | m


def random_finite_operand_bits(dtype, rng, shape, exp_lo=None, exp_hi=None):
    n = int(np.prod(shape))
    if dtype == "bf16":
        lo, hi = (1 if exp_lo is None else exp_lo), (254 if exp_hi is None else exp_hi)
        e = rng.integers(lo, hi + 1, size=n).astype(np.uint16)
        m = rng.integers(0, 128, size=n).astype(np.uint16)
        s = rng.integers(0, 2, size=n).astype(np.uint16)
        return ((s << 15) | (e << 7) | m).reshape(shape)
    lo, hi = (1 if exp_lo is None else exp_lo), (30 if exp_hi is None else exp_hi)
    e = rng.integers(lo, hi + 1, size=n).astype(np.uint16)
    m = rng.integers(0, 1024, size=n).astype(np.uint16)
    s = rng.integers(0, 2, size=n).astype(np.uint16)
    return ((s << 15) | (e << 10) | m).reshape(shape)


def families(dtype, rng, n_random):
    """Yield (name, A (n,16,16) words, Bt (n,8,16) words, C (n,16,8) f32 bits)."""

    cast = lambda x: G.to_operand(dtype, x.astype(np.float32))
    zeros_acc = lambda n: np.zeros((n, 16, 8), np.uint32)
    max_e = 254 if dtype == "bf16" else 30

    n = n_random
    yield "randn_zero_acc", cast(rng.standard_normal((n, 16, K))), cast(rng.standard_normal((n, 8, K))), zeros_acc(n)

    n = n_random // 4
    acc = rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-20, 21, (n, 16, 8)))
    yield "randn_random_acc", cast(rng.standard_normal((n, 16, K))), cast(rng.standard_normal((n, 8, K))), G.f32_bits(acc.astype(np.float32))

    n = n_random // 4
    yield (
        "uniform_bits",
        random_finite_operand_bits(dtype, rng, (n, 16, K)),
        random_finite_operand_bits(dtype, rng, (n, 8, K)),
        random_finite_fp32_bits(rng, n * 128).reshape(n, 16, 8),
    )

    n = n_random // 8
    a = random_finite_operand_bits(dtype, rng, (n, 16, K), 0, 3)
    b = random_finite_operand_bits(dtype, rng, (n, 8, K), 0, 3)
    acc = random_finite_fp32_bits(rng, n * 128, 0, 3).reshape(n, 16, 8)
    yield "subnormal", a, b, acc

    n = n_random // 8
    scale = 2.0**8 if dtype == "bf16" else 2.0**4
    a = rng.standard_normal((n, 16, K)) * scale
    b = rng.standard_normal((n, 8, K)) * scale
    a[:, :, 1::2] = a[:, :, 0::2]
    b[:, :, 1::2] = -b[:, :, 0::2]
    small = rng.standard_normal((n, 16, K)) * 2.0**-6
    mask = rng.random((n, 16, K)) < 0.15
    a = np.where(mask, small, a)
    acc = rng.standard_normal((n, 16, 8)) * 2.0**-4
    yield "cancellation", cast(a), cast(b), G.f32_bits(acc.astype(np.float32))

    n = n_random // 8
    yield (
        "mixed_magnitude",
        random_finite_operand_bits(dtype, rng, (n, 16, K)),
        random_finite_operand_bits(dtype, rng, (n, 8, K)),
        G.f32_bits((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-60, 61, (n, 16, 8)))).astype(np.float32)),
    )

    n = n_random // 8
    acc = rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-40, 41, (n, 16, 8)))
    yield "accumulator_magnitude", cast(rng.standard_normal((n, 16, K))), cast(rng.standard_normal((n, 8, K))), G.f32_bits(acc.astype(np.float32))

    n = n_random // 16
    acc = random_finite_fp32_bits(rng, n * 128, 250, 254).reshape(n, 16, 8)
    a = random_finite_operand_bits(dtype, rng, (n, 16, K), max_e - 9, max_e)
    b = random_finite_operand_bits(dtype, rng, (n, 8, K), max_e - 9, max_e)
    yield "near_overflow", a, b, acc

    n = 256
    a = np.zeros((n, 16, K), np.float32)
    b = np.zeros((n, 8, K), np.float32)
    a[n // 2 :, :, 0] = 1.0
    a[n // 2 :, :, 1] = 1.0
    b[n // 2 :, :, 0] = 3.0
    b[n // 2 :, :, 1] = -3.0
    acc = np.where(rng.random((n, 16, 8)) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
    yield "signed_zero", cast(a), cast(b), acc

    n = 512
    acc = random_finite_fp32_bits(rng, n * 128, 0, 30).reshape(n, 16, 8)
    yield "tiny_acc_zero_products", np.zeros((n, 16, K), np.uint16), np.zeros((n, 8, K), np.uint16), acc


def records_of(A, Bt, C, D):
    n = A.shape[0]
    a = np.broadcast_to(A[:, :, None, :], (n, 16, 8, K)).reshape(-1, K)
    b = np.broadcast_to(Bt[:, None, :, :], (n, 16, 8, K)).reshape(-1, K)
    return C.reshape(-1), a, b, D.reshape(-1)


_MODEL: TCModel | None = None


def _init(name):
    global _MODEL
    _MODEL = model_for("bf16" if "bf16" in name else "f16")


def _eval_chunk(args):
    acc, a, b = args
    out = np.zeros(len(acc), np.uint32)
    err = np.zeros(len(acc), np.bool_)
    for i in range(len(acc)):
        try:
            out[i] = tc_dot(_MODEL, int(acc[i]), a[i].tolist(), b[i].tolist())
        except Exception:  # noqa: BLE001
            err[i] = True
    return out, err


def python_eval(model, acc, a, b, procs):
    n = len(acc)
    chunks = max(1, min(n // 2000 + 1, procs * 8))
    idx = np.array_split(np.arange(n), chunks)
    with mp.Pool(procs, initializer=_init, initargs=(model.name,)) as pool:
        parts = pool.map(_eval_chunk, [(acc[i], a[i], b[i]) for i in idx])
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def hexrec(acc, a, b, d):
    return {"acc": f"{int(acc):08x}", "a": "".join(f"{int(x):04x}" for x in a), "b": "".join(f"{int(x):04x}" for x in b), "d": f"{int(d):08x}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dtype", choices=["bf16", "f16"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--n-random", type=int, default=1_000_000)
    ap.add_argument("--python-elements", type=int, default=200_000, help="per family, random subsample through the Python model")
    ap.add_argument("--procs", type=int, default=max(1, min(96, os.cpu_count() - 4)))
    ap.add_argument("--chunk-tiles", type=int, default=250_000)
    args = ap.parse_args()

    rng = np.random.default_rng(20260907)
    model = model_for(args.dtype)
    out = {"device": G.device_info(), "dtype": args.dtype, "model": {"name": model.name, "groups": list(model.groups), "width": model.width, "zero_exponent": model.zero_exponent, "instruction": model.instruction}, "families": {}}
    golden = {"model": model.name, "dtype": args.dtype, "k": K, "records": []}
    t_all = time.time()
    for name, A, Bt, C in families(args.dtype, rng, args.n_random):
        fam = {"tiles": int(A.shape[0]), "elements": int(A.shape[0]) * 128, "mismatches_vs_cpp": 0, "python": {"elements": 0, "mismatches": 0, "domain_errors": 0, "cpp_vs_python_mismatches": 0}, "gpu_seconds": 0.0, "cpp_seconds": 0.0, "python_seconds": 0.0, "first_mismatches": []}
        for start in range(0, A.shape[0], args.chunk_tiles):
            sl = slice(start, min(A.shape[0], start + args.chunk_tiles))
            t0 = time.time()
            D = G.mma_tiles(args.dtype, A[sl], Bt[sl], C[sl])
            fam["gpu_seconds"] += time.time() - t0
            acc, a, b, d = records_of(A[sl], Bt[sl], C[sl], D)
            t0 = time.time()
            cpp = CM.tc_records(model, acc, a, b)
            fam["cpp_seconds"] += time.time() - t0
            mism = cpp != d
            fam["mismatches_vs_cpp"] += int(mism.sum())
            for i in np.flatnonzero(mism)[: max(0, 5 - len(fam["first_mismatches"]))]:
                fam["first_mismatches"].append(dict(hexrec(acc[i], a[i], b[i], d[i]), cpp=f"{int(cpp[i]):08x}"))
            # Python subsample
            n_py = min(args.python_elements - fam["python"]["elements"], len(acc))
            if n_py > 0:
                pick = rng.choice(len(acc), size=n_py, replace=False)
                t0 = time.time()
                py, err = python_eval(model, acc[pick], a[pick], b[pick], args.procs)
                fam["python_seconds"] += time.time() - t0
                fam["python"]["elements"] += n_py
                fam["python"]["mismatches"] += int(((py != d[pick]) | err).sum())
                fam["python"]["domain_errors"] += int(err.sum())
                fam["python"]["cpp_vs_python_mismatches"] += int(((py != cpp[pick]) & ~err).sum())
                if args.golden and start == 0:
                    for i in pick[:40]:
                        golden["records"].append(dict(hexrec(acc[i], a[i], b[i], d[i]), family=name))
        out["families"][name] = fam
        print(name, {k_: v for k_, v in fam.items() if k_ != "first_mismatches"}, flush=True)
    out["seconds"] = time.time() - t_all
    out["total_elements"] = int(sum(f["elements"] for f in out["families"].values()))
    out["total_mismatches_vs_cpp"] = int(sum(f["mismatches_vs_cpp"] for f in out["families"].values()))
    out["total_python_elements"] = int(sum(f["python"]["elements"] for f in out["families"].values()))
    out["total_python_mismatches"] = int(sum(f["python"]["mismatches"] for f in out["families"].values()))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    if args.golden:
        with open(args.golden, "w") as fh:
            json.dump(golden, fh, separators=(",", ":"))
    print("TOTAL", out["total_elements"], "cpp mismatches", out["total_mismatches_vs_cpp"], "python", out["total_python_elements"], "mismatches", out["total_python_mismatches"], "in", round(out["seconds"], 1), "s", flush=True)


if __name__ == "__main__":
    main()
