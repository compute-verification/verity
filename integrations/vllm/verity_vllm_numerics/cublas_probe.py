"""Targeted cuBLAS follow-ups (GPU host only), written after the first gemm_production sweep:

1. ``down_proj`` at M=512 (``ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn``) and
   ``gate_up_proj`` at M=1 (``..._128x64_ldg8_f2f_stages_32x6_tn``) launch as ``Memset`` + one GEMM
   kernel with no reduce kernel and no single-chain / workspace split-K candidate matched:
   sweep split counts, block granularities, ceil/floor bounds with the serial split-K reduce
   modes 7/8 (and 3/6/0 as controls).
2. The biased ``sliced1x2`` kernel (``qkv_proj``) matched ``bias_mode=1`` at small M and
   ``bias_mode=4`` at larger M: evaluate every epilogue variant on *all* coordinates of every
   case and report which are consistent with all of them.

Usage: python -m verity_vllm_numerics.cublas_probe --acts ... --out cublas_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import cpu_model as CM
from .gemm_production import from_words, kernel_names, sliced_k_spec, stress_families, words
from .mma import AMPERE_BF16


def split_bounds(K: int, S: int, blk: int, rule: str) -> tuple[int, ...] | None:
    nb = K // blk
    if S > nb:
        return None
    if rule == "ceil":
        per = math.ceil(nb / S)
        b = [min(nb, s * per) for s in range(S + 1)]
    elif rule == "floor":
        per = nb // S
        rem = nb - per * S
        b = [0]
        for s in range(S):
            b.append(b[-1] + per + (1 if s < rem else 0))
    else:
        raise ValueError(rule)
    if len(set(b)) != S + 1:
        return None
    return tuple(v * blk // 16 for v in b)


def probe_serial_splitk(acts: str, layers: list[int], which: tuple[tuple[str, int], ...]) -> list[dict]:
    """Sweep split-K descriptions for the kernels that launch as ``Memset`` + one GEMM kernel and
    no reduce kernel (CUTLASS-style *serial* split-K: the splits reduce through the BF16 output in
    semaphore order).  Reduce modes 7/8 model that; 3/6/0 are kept as controls."""
    out = []
    for li in layers:
        data = np.load(os.path.join(acts, f"layer{li}.npz"))
        for name, M in which:
            x, w = data[f"{name}.x"][:M], data[f"{name}.w"]
            K, N = w.shape[1], w.shape[0]
            names, y = kernel_names(lambda: F.linear(from_words(x), from_words(w)))
            gpu = words(y)
            rows = np.arange(0, M, max(1, M // 64))  # <= 64 rows for the sweep, confirm on all
            best = None
            tried = []
            t0 = time.time()
            for red in (7, 8, 3, 6, 0):
                for blk in (32, 64, 128, 256):
                    for rule in ("ceil", "floor"):
                        for S in range(2, 17):
                            bounds = split_bounds(K, S, blk, rule)
                            if bounds is None:
                                continue
                            spec = CM.GemmModelSpec(AMPERE_BF16.name, nsplit=S, bounds=bounds, reduce_mode=red, bias_mode=0, note=f"split-K {S}, {blk}-wide blocks, {rule} bounds, reduce {red}")
                            mb, _ = CM.gemm_model(AMPERE_BF16, x[rows], w, None, spec, want_f32=False)
                            n = int(np.sum(mb != gpu[rows]))
                            tried.append((n, spec.note))
                            if n == 0:
                                t1 = time.time()
                                mb, _ = CM.gemm_model(AMPERE_BF16, x, w, None, spec, want_f32=False)
                                t_full = time.time() - t1
                                if np.array_equal(mb, gpu):
                                    best = spec
                                    break
                        if best:
                            break
                    if best:
                        break
                if best:
                    break
            tried.sort()
            out.append({
                "label": f"layers.{li}.{name}", "M": M, "K": K, "N": N, "kernels": sorted(set(names)),
                "attempted": int(gpu.size), "exact": int(gpu.size) if best is not None else None,
                "model_found": None if best is None else best.to_json(),
                "cpu_seconds_full_gemm": t_full if best is not None else None,
                "candidates_tried": len(tried), "best_candidates": [{"mismatches_on_sweep_rows": n, "note": s} for n, s in tried[:8]],
                "sweep_rows": int(len(rows)), "seconds": time.time() - t0,
            })
            print(json.dumps(out[-1]), flush=True)
    return out


def probe_sliced_bias(acts: str, layers: list[int], Ms: list[int]) -> list[dict]:
    out = []
    variants = {
        "bias_after_reduce (bm1)": lambda K: sliced_k_spec(K, 2, 2, 1),
        "bias_into_slice0_then_add_slice1 (bm4)": lambda K: sliced_k_spec(K, 2, 2, 4, 0, 0),
        "bias_into_slice1_then_add_slice0 (bm4 swapped)": lambda K: sliced_k_spec(K, 2, 2, 4, 0, 1),
    }
    cases = []
    for li in layers:
        data = np.load(os.path.join(acts, f"layer{li}.npz"))
        for M in Ms:
            cases.append((f"layers.{li}.qkv_proj natural", data["qkv_proj.x"][:M], data["qkv_proj.w"], data["qkv_proj.b"]))
    rng = np.random.default_rng(11)
    for M in (7, 64):
        for fam, x, w, b in stress_families(rng, M, 1536, 2048, True):
            if fam in ("cancellation", "rounding_boundaries", "randn", "extreme_magnitudes"):
                cases.append((f"stress.qkv_proj {fam}", x, w, b))
    for label, x, w, b in cases:
        K = w.shape[1]
        names, y = kernel_names(lambda: F.linear(from_words(x), from_words(w), from_words(b)))
        gpu = words(y)
        if not any("sliced1x2" in n for n in names):
            continue
        row = {"label": label, "M": int(x.shape[0]), "kernels": sorted(set(names)), "attempted": int(gpu.size), "mismatches": {}}
        for name, mk in variants.items():
            mb, _ = CM.gemm_model(AMPERE_BF16, x, w, b, mk(K), want_f32=False)
            row["mismatches"][name] = int(np.sum(mb != gpu))
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma list of probes to run: sliced_bias,serial_splitk")
    args = ap.parse_args()
    only = None if args.only is None else set(args.only.split(","))
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    if only is None or "sliced_bias" in only:
        res["sliced1x2_bias_epilogue"] = probe_sliced_bias(args.acts, [0, 13, 27], [7, 16, 64, 100])
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
    if only is None or "serial_splitk" in only:
        res["serial_splitk"] = probe_serial_splitk(args.acts, [0, 13, 27], (("gate_up_proj", 1), ("down_proj", 512)))
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
