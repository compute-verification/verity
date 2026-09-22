"""probe (lane ov-moe): which `exp` does `_moe_C::topk_softmax` compute?  Emulates the kernel's arithmetic on the stored
difftest cases (`difftest/moe_router_topk/case_*/in_0.npy` bf16 logits, `out_0.npy` f32 weights, `out_1.npy` i32 ids) with the
recorded reduction geometry (E/VPT threads x VPT values ascending, xor butterfly T/2..1, `p = expf(x - max) * (1.f / sum)`,
greedy argmax with -10000.f masking) and three candidate exponentials:

    rn      correctly rounded f32 exp (float64 exp rounded once) -- lane ov-moe's stand-in `ref_prims.F32ExpRn`
    verity  `quarantine.ov_moe.moe.nv_expf_bits`: the CUDA math-library expf sequence (cvt.sat / fma.rm magic-number range
            reduction, hi/lo log2e fma, MUFU.EX2 table, exponent-field scaling) = `NvExpf_v1`, the router body's exp (cov-moe)
    torch   `torch.exp` on CUDA f32 (libdevice `__nv_expf`, <= 2 ulp)           -- needs a GPU
    fast    `exp2f(x * log2e)` via `torch.exp2` on CUDA f32 (the `--use_fast_math` `__expf` shape)

For each candidate: how many of the cases' weight words match the kernel bit-for-bit, and the ulp histogram of the misses.
The ids are compared once (they depend on the ordering of the probabilities only).

    python -m verity_capture.bench.probe_ov_moe_router_expf DIR [--vpt 8] [--no-gpu] [--out probe.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def butterfly(vals: np.ndarray, op) -> np.float32:
    """xor butterfly over a power-of-two array of f32 lane values, mask T/2 .. 1 (every lane ends with the same value)."""
    v = vals.astype(np.float32).copy()
    m = len(v) // 2
    while m >= 1:
        v = np.array([op(v[i], v[i ^ m]) for i in range(len(v))], dtype=np.float32)
        m //= 2
    return v[0]


def f32_add(a, b):
    return np.float32(np.float32(a) + np.float32(b))


def kernel_geometry(x: np.ndarray, ex_fn, VPT: int, TOPK: int) -> tuple[np.ndarray, np.ndarray]:
    E = x.size
    T = E // VPT
    x = x.astype(np.float32)
    tmax = np.array([x[t * VPT:(t + 1) * VPT].max() for t in range(T)], dtype=np.float32)   # ascending max is order-free
    rmax = butterfly(tmax, max)
    ex = ex_fn((x - rmax).astype(np.float32)).astype(np.float32)
    tsum = []
    for t in range(T):
        s = np.float32(0.0)
        for j in range(VPT):
            s = f32_add(s, ex[t * VPT + j])
        tsum.append(s)
    rsum = butterfly(np.array(tsum, dtype=np.float32), f32_add)
    rcp = np.float32(np.float32(1.0) / rsum)
    p = (ex * rcp).astype(np.float32)
    p = np.where(np.isfinite(p), p, np.float32(0.0)).astype(np.float32)
    w, ids = [], []
    for _ in range(TOPK):
        best, best_i = p[0], 0
        for i in range(1, E):
            if p[i] > best:        # strict: first index wins ties
                best, best_i = p[i], i
        w.append(best); ids.append(best_i)
        p[best_i] = np.float32(-10000.0)
    return np.array(w, dtype=np.float32), np.array(ids, dtype=np.int32)


def bf16_words_to_f32(w: np.ndarray) -> np.ndarray:
    return (w.astype(np.uint32) << 16).view(np.float32)


def ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ia = a.view(np.int32).astype(np.int64); ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia); ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--vpt", type=int, default=8); ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = Path(a.dir)
    man = json.loads((d / "manifest.json").read_text())
    cands = {"rn": lambda v: np.exp(v.astype(np.float64)).astype(np.float32)}
    try:
        from verity_vllm.registry.quarantine.ov_moe.moe import nv_expf_bits
        cands["verity"] = lambda v: np.array([nv_expf_bits(int(b)) for b in v.astype(np.float32).view(np.uint32)], dtype=np.uint32).view(np.float32)
    except ImportError as e:  # noqa: BLE001
        print(f"[probe] verity candidate unavailable: {e}")
    if not a.no_gpu:
        import torch
        cands["torch"] = lambda v: torch.exp(torch.from_numpy(v).cuda()).cpu().numpy()
        cands["fast"] = lambda v: torch.exp2(torch.from_numpy(v).cuda() * np.float32(1.4426950408889634)).cpu().numpy()
    rep: dict = {"dir": str(d), "n_cases": len(man["cases"]), "candidates": {}}
    ids_ok = 0
    for name, fn in cands.items():
        exact_cases = exact_words = total_words = 0
        hist: dict[int, int] = {}
        per_case = []
        for rec in man["cases"]:
            cdir = d / f"case_{rec['i']}"
            x = bf16_words_to_f32(np.load(cdir / "in_0.npy").reshape(-1))
            w_prod = np.load(cdir / "out_0.npy").reshape(-1).view(np.float32)
            ids_prod = np.load(cdir / "out_1.npy").reshape(-1).view(np.int32)
            w, ids = kernel_geometry(x, fn, a.vpt, w_prod.size)
            if name == "rn":
                ids_ok += int(np.array_equal(ids, ids_prod))
            u = ulps(w, w_prod)
            exact_cases += int((u == 0).all()); exact_words += int((u == 0).sum()); total_words += u.size
            for k in u[u > 0].tolist():
                hist[int(k)] = hist.get(int(k), 0) + 1
            per_case.append({"i": rec["i"], "kind": rec["inputs"][0].get("kind"), "ulps": u.tolist()})
        rep["candidates"][name] = {"exact_cases": exact_cases, "exact_words": exact_words, "total_words": total_words,
                                   "miss_ulp_histogram": dict(sorted(hist.items())), "per_case": per_case}
        print(f"[probe] {name:6s}: {exact_cases}/{len(man['cases'])} cases, {exact_words}/{total_words} weight words bit-exact; misses by ulp: {dict(sorted(hist.items()))}")
    rep["ids_bit_exact_cases"] = ids_ok
    print(f"[probe] ids (rn candidate, ordering only): {ids_ok}/{len(man['cases'])} cases identical")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
