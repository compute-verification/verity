"""Ligerito PCS benchmark (lane ligerito-proto): per N, prover wall split into commit / recursion+sumcheck / serialization,
proof bytes, verify time, peak device memory, the three negatives, and -- on the same device -- today's Ligero
encode+commit of the same number of cells (``encode_simt`` SIMT NTT l -> 4l + ``merkle.commit_device`` fused Blake3, in
<= 4096-row slabs as the prover commits them).  Writes bench-result-style JSON.

    python -m backends.direct.ligerito.proto.bench --n 26 28 29 --queries 192 --out /workspace/lgp/ligerito_bench.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone

import numpy as np
import torch

from . import ext
from .dims import Dims, choose
from .ext import D, P
from .pcs import Proof, Timer, prove, verify

try:
    import cupy as cp
except Exception:  # noqa: BLE001
    cp = None


COMMIT = ("commit_r1_alloc", "commit_r1_encode", "commit_r1_merkle")
RECURSION = ("sumcheck_r1_contract", "sumcheck_host", "sumcheck_r1_fold", "recursion_encode", "recursion_merkle",
             "recursion_contract", "recursion_fold")
SERIAL = ("serialize_openings", "serialize_final", "serialize_bytes")


def _sync():
    torch.cuda.synchronize()


def _mem_reset():
    torch.cuda.reset_peak_memory_stats()


def _mem_peak() -> dict:
    out = {"torch_peak_allocated": torch.cuda.max_memory_allocated(), "torch_peak_reserved": torch.cuda.max_memory_reserved()}
    if cp is not None:
        pool = cp.get_default_memory_pool()
        out["cupy_pool_total"] = pool.total_bytes()
    free, total = torch.cuda.mem_get_info()
    out["device_total"] = total
    out["peak_estimate"] = out["torch_peak_reserved"] + out.get("cupy_pool_total", 0)
    return out


def _negatives(pf: Proof, z: np.ndarray, buf: bytes) -> dict:
    res = {}
    # wrong evaluation
    ok, why = verify(Proof.from_bytes(buf), z, v=(pf.v + 1) % P)
    res["wrong_evaluation"] = {"rejected": not ok, "reason": why}
    # tampered opened row (round 1, one word)
    p = Proof.from_bytes(buf)
    p.opens[0].rows[3, 1] = (p.opens[0].rows[3, 1] + 1) % P
    ok, why = verify(p, z)
    res["tampered_opened_row"] = {"rejected": not ok, "reason": why}
    # tampered opened row of the last round
    p = Proof.from_bytes(buf)
    p.opens[-1].rows[0, 0] = (p.opens[-1].rows[0, 0] + 1) % P
    ok, why = verify(p, z)
    res["tampered_opened_row_last"] = {"rejected": not ok, "reason": why}
    # wrong final vector
    p = Proof.from_bytes(buf)
    p.final_y[1, 2] = (p.final_y[1, 2] + 1) % P
    ok, why = verify(p, z)
    res["wrong_final_vector"] = {"rejected": not ok, "reason": why}
    # tampered sumcheck message
    p = Proof.from_bytes(buf)
    p.sumcheck[0][0][1, 0] = (p.sumcheck[0][0][1, 0] + 1) % P
    ok, why = verify(p, z)
    res["tampered_sumcheck"] = {"rejected": not ok, "reason": why}
    res["all_rejected"] = all(v["rejected"] for v in res.values() if isinstance(v, dict))
    return res


def run_one(n: int, dims: Dims, device, reps: int, negatives: bool, seed: int) -> dict:
    N = 1 << n
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    rng = np.random.default_rng(seed)
    z = rng.integers(0, P, size=(n, D), dtype=np.int64)
    rows = []
    proof_bytes = None
    neg = None
    # untimed warm run at THIS shape: compiles the size-specialised NTT / contraction kernels, fills the twiddle caches
    fw = torch.randint(0, P, (N,), dtype=torch.int32, device=device, generator=g)
    prove(fw, z, dims)
    del fw
    _sync()
    for rep in range(reps):
        f = torch.randint(0, P, (N,), dtype=torch.int32, device=device, generator=g)
        _sync()
        _mem_reset()
        tm = Timer(torch.device(device))
        t0 = time.perf_counter()
        pf = prove(f, z, dims, timer=tm)
        _sync()
        t_prove = time.perf_counter() - t0
        tm.start("serialize_bytes")
        buf = pf.to_bytes()
        tm.stop()
        mem = _mem_peak()
        del f
        t0 = time.perf_counter()
        pf2 = Proof.from_bytes(buf)
        ok, why = verify(pf2, z)
        t_verify = time.perf_counter() - t0
        assert ok, why
        b = dict(tm.buckets)
        row = {"rep": rep, "prover_wall_s": t_prove + b.get("serialize_bytes", 0.0),
               "commit_s": sum(b.get(k, 0.0) for k in COMMIT),
               "recursion_sumcheck_s": sum(b.get(k, 0.0) for k in RECURSION),
               "serialization_s": sum(b.get(k, 0.0) for k in SERIAL),
               "buckets": b, "verify_s": t_verify, "proof_bytes": len(buf), "coins": pf.coins, "memory": mem}
        rows.append(row)
        proof_bytes = len(buf)
        if negatives and rep == 0:
            t0 = time.perf_counter()
            neg = _negatives(pf, z, buf)
            neg["wall_s"] = time.perf_counter() - t0
        del pf, pf2, buf
        # (the torch caching allocator keeps its blocks between reps: a fresh 8-16 GB cudaMalloc would otherwise land
        #  inside commit_r1_encode -- an artefact of the benchmark, not of the prover)
    med = {k: statistics.median(r[k] for r in rows) for k in
           ("prover_wall_s", "commit_s", "recursion_sumcheck_s", "serialization_s", "verify_s")}
    pred = dims.predicted_bytes()
    return {"n": n, "cells": N, "dims": {"kprime": dims.kprime, "k": dims.k, "queries": dims.queries, "rate_log2": dims.rates,
                                        "rounds": dims.rounds, "final_log": dims.final_log, "describe": dims.describe()},
            "predicted_bytes": pred, "proof_bytes": proof_bytes, "median": med, "reps": rows, "negatives": neg}


def ligero_baseline(cells: int, device, reps: int, l: int = 4096, rate_log2: int = 2) -> dict:
    """Today's encode + commit of ``cells`` cells: SIMT NTT (m x l -> m x 4l) + fused Blake3 Merkle per <= 4096-row slab."""
    from ._ligero import encode_simt, merkle
    n_code = l << rate_log2
    m = cells // l
    slab = 4096
    enc = encode_simt.encoder_for(l, n_code, 0, device)
    W = torch.randint(0, P, (m, l), dtype=torch.int64, device=device)
    ts_enc, ts_com, ts_tot = [], [], []
    mem = None
    for _ in range(reps):
        _sync()
        _mem_reset()
        t0 = time.perf_counter()
        t_enc = t_com = 0.0
        roots = []
        for r0 in range(0, m, slab):
            a = time.perf_counter()
            U, _ = enc.encode(W[r0:r0 + slab], None, False)
            _sync()
            b = time.perf_counter()
            root, tree = merkle.commit_device(U)
            _sync()
            c = time.perf_counter()
            roots.append(root)
            t_enc += b - a
            t_com += c - b
            del U, tree
        _sync()
        ts_tot.append(time.perf_counter() - t0)
        ts_enc.append(t_enc)
        ts_com.append(t_com)
        mem = _mem_peak()
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    return {"cells": m * l, "shape": [m, l], "codeword_len": n_code, "slabs": (m + slab - 1) // slab,
            "encode_s": statistics.median(ts_enc), "commit_s": statistics.median(ts_com), "encode_commit_s": statistics.median(ts_tot),
            "memory": mem, "note": "SIMT NTT l->4l (encode_simt) + fused Blake3 Merkle (merkle.commit_device) per 4096-row slab; no tests"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[26])
    ap.add_argument("--queries", type=int, nargs="+", default=[192], help="|S| (one value, or one per committed round with --kprime)")
    ap.add_argument("--rates", type=int, nargs="*", default=None, help="rate log2 per committed round with --kprime (default 2)")
    ap.add_argument("--tag", default="", help="free-text tag recorded in the JSON (e.g. 'ligerito-design F point')")
    ap.add_argument("--ell", type=int, default=0, help="force ell (0 = the predicted-bytes optimum among 3..6)")
    ap.add_argument("--kprime", type=int, nargs="*", default=None, help="override the k'_i (one value per committed round)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--no-negatives", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    results = []

    def doc_of(results):
        return {"schema": "bench-result/v1", "kind": "ligerito-pcs-proto", "lane": "ligerito-proto",
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "device": {"name": props.name, "total_memory": props.total_memory, "torch": torch.__version__, "cuda": torch.version.cuda,
                           "cupy": getattr(cp, "__version__", None), "host": platform.node()},
                "field": {"p": P, "ext_degree": D, "beta": ext.BETA}, "hash": "blake3", "tag": args.tag,
                "argv": vars(args), "results": results}

    # warm-up: compiles the cupy kernels / torch paths once (excluded from every timing)
    warm = Dims(16, [4, 3], [16, 16])
    fw = torch.randint(0, P, (1 << 16,), dtype=torch.int32, device=device)
    prove(fw, np.random.default_rng(0).integers(0, P, size=(16, D), dtype=np.int64), warm)
    if not args.no_baseline:
        ligero_baseline(1 << 20, device, 1)
    del fw
    _sync()
    for n in args.n:
        q0 = args.queries[0]
        if args.kprime:
            L = len(args.kprime)
            qs = list(args.queries) if len(args.queries) == L else [q0] * L
            rates = list(args.rates) if args.rates else [2] * L
            assert len(rates) == L
            dims = Dims(n, list(args.kprime), qs, rates)
        elif args.ell:
            from .dims import make
            dims = make(n, args.ell, q0)
        else:
            dims = choose(n, q0)
        print(f"== n = {n}: {dims.describe()}; predicted {dims.predicted_bytes()['total_nodedup'] / 1e6:.3f} MB", flush=True)
        try:
            r = run_one(n, dims, device, args.reps, not args.no_negatives, args.seed)
        except torch.OutOfMemoryError as e:
            free, total = torch.cuda.mem_get_info()
            msg = str(e).split("See documentation")[0].strip()
            print(f"   OOM: {msg}", flush=True)
            results.append({"n": n, "cells": 1 << n, "dims": {"kprime": dims.kprime, "k": dims.k, "queries": dims.queries,
                            "describe": dims.describe()}, "oom": msg, "device_total": total,
                            "budget_note": f"f {(4 << n) / 2**30:.1f} GiB + round-1 codeword {(16 << n) / 2**30:.1f} GiB + Merkle tree ~1 GiB"})
            torch.cuda.empty_cache()
            _write(args.out, doc_of(results))
            continue
        med = r["median"]
        print(f"   prover {med['prover_wall_s']:.3f} s = commit {med['commit_s']:.3f} + recursion/sumcheck {med['recursion_sumcheck_s']:.3f}"
              f" + serialization {med['serialization_s']:.3f}; proof {r['proof_bytes'] / 1e6:.3f} MB; verify {med['verify_s']:.3f} s;"
              f" peak {r['reps'][-1]['memory']['peak_estimate'] / 2**30:.2f} GiB; coins {r['reps'][-1]['coins']}", flush=True)
        for k, v in sorted(r["reps"][-1]["buckets"].items(), key=lambda kv: -kv[1]):
            print(f"      {k:24s} {v:.3f} s")
        if r["negatives"]:
            print(f"   negatives all rejected: {r['negatives']['all_rejected']}", flush=True)
        if not args.no_baseline:
            base = ligero_baseline(1 << n, device, args.reps)
            r["ligero_baseline"] = base
            print(f"   ligero encode+commit ({base['shape'][0]} x 4096): {base['encode_commit_s']:.3f} s = encode {base['encode_s']:.3f}"
                  f" + commit {base['commit_s']:.3f}; ratio prover/baseline {med['prover_wall_s'] / base['encode_commit_s']:.2f}x", flush=True)
        results.append(r)
        _write(args.out, doc_of(results))
    _write(args.out, doc_of(results), loud=True)


def _write(path: str, doc: dict, loud: bool = False) -> None:
    if path:
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=1)
        if loud:
            print("wrote", path)


if __name__ == "__main__":
    main()
