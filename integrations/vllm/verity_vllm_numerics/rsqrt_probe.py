"""Characterise the approximate reciprocal-square-root instructions RMSNorm ends in (GPU host only).

vLLM's CUDA ``rms_norm_kernel`` / ``fused_add_rms_norm_kernel`` compute ``s = rsqrtf(v)`` -- CUDA's
``rsqrtf`` is the hardware approximation ``MUFU.RSQ`` (documented 2 ulp), not an IEEE operation.  The
Triton ``_rms_norm_kernel`` computes ``s = 1.0 / tl.sqrt(v)`` (``sqrt.approx.f32`` then ``div.full.f32``).
Both are deterministic functions of the FP32 input, and ``rsqrt(4x) = rsqrt(x) / 2`` exactly in FP32, so
if the hardware honours that scaling the whole function is fixed by its values on two binades
([1, 2) and [2, 4)): 2^24 entries.  This probe

1. tabulates both functions on those binades and on several other binades (scaling check),
2. records each as a signed ulp offset from the correctly rounded IEEE result (int8),
3. writes the tables to ``--table-out`` (npz, stays on the pod; sha256 goes in the JSON) and a summary
   (error histogram, scaling invariance, sha256) to ``--out``.

``rsqrt_from_table`` then evaluates the characterised function for arbitrary FP32 inputs (normal range).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch


def ieee_rsqrt(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(x.astype(np.float64))).astype(np.float32)


def ulp_delta(approx: np.ndarray, exact: np.ndarray) -> np.ndarray:
    d = approx.view(np.int32).astype(np.int64) - exact.view(np.int32).astype(np.int64)
    if np.abs(d).max() > 127:
        raise RuntimeError(f"ulp delta {np.abs(d).max()} exceeds int8")
    return d.astype(np.int8)


def binade(e: int) -> np.ndarray:
    """All 2^23 FP32 values in [2^e, 2^(e+1))."""
    bits = ((e + 127) << 23) + np.arange(1 << 23, dtype=np.uint32)
    return bits.view(np.float32)


def gpu_rsqrtf(x: np.ndarray) -> np.ndarray:
    return torch.rsqrt(torch.from_numpy(x).cuda()).cpu().numpy()


_triton_kernel = None


def gpu_triton_inv_sqrt(x: np.ndarray) -> np.ndarray:
    """1.0 / tl.sqrt(x) with the same Triton the vLLM kernel is compiled with."""
    global _triton_kernel
    import triton
    import triton.language as tl

    if _triton_kernel is None:

        @triton.jit
        def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            v = tl.load(x_ptr + offs, mask=m, other=1.0)
            rms = tl.sqrt(v)
            inv = 1.0 / rms
            tl.store(y_ptr + offs, inv, mask=m)

        _triton_kernel = _k
    xt = torch.from_numpy(x).cuda()
    yt = torch.empty_like(xt)
    n = xt.numel()
    _triton_kernel[(triton.cdiv(n, 1024),)](xt, yt, n, BLOCK=1024)
    return yt.cpu().numpy()


def characterise(fn, name: str) -> tuple[dict, np.ndarray]:
    t0 = time.time()
    tables = {}
    for e in (0, 1):
        x = binade(e)
        tables[e] = ulp_delta(fn(x), ieee_rsqrt(x))
    table = np.concatenate([tables[0], tables[1]])  # index: parity*2^23 + significand
    # scaling invariance: other binades must reproduce the two reference binades exactly
    invariance = {}
    for e in (-40, -21, -20, -3, -2, -1, 2, 3, 20, 21, 60, 61):
        x = binade(e)
        got = fn(x)
        ref = rsqrt_from_table(table, x)
        invariance[str(e)] = int(np.sum(got != ref))
    vals, counts = np.unique(table, return_counts=True)
    summary = {
        "function": name,
        "entries": int(table.size),
        "ulp_error_histogram": {str(int(v)): int(c) for v, c in zip(vals, counts)},
        "max_abs_ulp_error": int(np.abs(table).max()),
        "fraction_correctly_rounded": float(np.mean(table == 0)),
        "scaling_invariance_mismatches_per_binade": invariance,
        "sha256_int8_table": hashlib.sha256(table.tobytes()).hexdigest(),
        "seconds": time.time() - t0,
    }
    return summary, table


def rsqrt_from_table(table: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The characterised function on normal positive FP32 inputs (subnormals / zero / inf not modelled)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    exp = ((bits >> 23) & 0xFF).astype(np.int64) - 127
    sig = (bits & 0x7FFFFF).astype(np.int64)
    parity = exp & 1
    delta = table[parity * (1 << 23) + sig].astype(np.int64)
    exact = ieee_rsqrt(x)
    return (exact.view(np.int32).astype(np.int64) + delta).astype(np.int32).view(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--table-out", required=True)
    args = ap.parse_args()
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    s1, t1 = characterise(gpu_rsqrtf, "cuda rsqrtf (MUFU.RSQ) via torch.rsqrt")
    res["rsqrtf"] = s1
    print(json.dumps(s1, indent=1), flush=True)
    try:
        import triton

        s2, t2 = characterise(gpu_triton_inv_sqrt, f"triton {triton.__version__}: 1.0 / tl.sqrt(x)")
        res["triton_inv_sqrt"] = s2
        print(json.dumps(s2, indent=1), flush=True)
    except Exception as e:  # noqa: BLE001
        res["triton_inv_sqrt"] = {"error": f"{type(e).__name__}: {e}"}
        t2 = np.zeros(1, np.int8)
    np.savez_compressed(args.table_out, rsqrtf=t1, triton_inv_sqrt=t2)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
