"""[ov-sampling] Bit-exact, torch-free transcription of the two libdevice functions inside vLLM's Gumbel-max sampler:
`__nv_logf` and `__nv_log1pf` (CUDA libdevice.10.bc as shipped with Triton 3.7.1, FTZ variants — Triton compiles kernels with
`nvvm-reflect-ftz=1`, and the cached PTX of `_gumbel_sample_kernel` shows exactly the instruction mix the IR predicts:
`fma.rn.ftz.f32` ×192, `add.rz` ×8, `cvt.rn.f32.s32` ×24, `mul.f32` ×40, `add.f32` ×24, `max.f32` ×8, no `lg2.approx`).

The LLVM IR both functions were transcribed from is `out/gen/lanes/ov-sampling/probe/libdevice_logf_log1pf.ll` (dumped with
llvmlite from `triton/backends/nvidia/lib/libdevice.10.bc`).  Every fp32 operation is reproduced with a single IEEE rounding:

    fma.rn.ftz.f32   -> `fma_f32`     exact product in float64 (24+24 bits) + TwoSum error term + midpoint-aware final rounding
    add.rz.ftz.f32   -> `add_rz_f32`  float64 sum + TwoSum error term, truncated toward zero
    mul.f32 / add.f32 / cvt.rn.f32.s32 -> numpy float32 arithmetic (correctly rounded, RN-even)
    bit tricks       -> int32 wraparound arithmetic on the float bit patterns, exactly as the IR does

With `gen_sampling_rng` (Philox4x32-10 + `uint_to_uniform_float`, already bit-exact) this makes the whole Gumbel noise vector
`g(seed, pos)[v] = -logf(-log1pf(-max(u, 2^-31ish)))` a pure integer/fp32 function that the laptop reproduces bit for bit —
the `rng` root class needs no tolerance and no margin certificate (verified against the GPU by `gen_ov_sampling_probe.py`).

    python -m verity_vllm.program.kernels.libdevice_sampling --compare PROBE_DIR   # PROBE_DIR/g_<seed>_<pos>.npy from the pod probe
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

from verity_vllm.program.kernels import sampling_rng as R

F32 = np.float32
FLT_MIN = np.float32(1.1754943508222875e-38)          # 2**-126, IR constant 0x3810000000000000
INF = np.float32(np.inf)


def _c(hex_double: str) -> np.float32:
    """An LLVM float constant printed as a double hex literal (`0x3FE62E4300000000`) -> the float32 it denotes exactly."""
    return np.float32(struct.unpack(">d", bytes.fromhex(hex_double[2:]))[0])


LN2 = _c("0x3FE62E4300000000")                          # 0.693147182 (bits 0x3F317218)
TWO_M23 = _c("0x3E80000000000000")                      # 2**-23
TWO_P23 = _c("0x4160000000000000")                      # 2**23
# __nv_logf polynomial (Horner, highest degree first), then -0.5
LOGF_POLY = [_c("0xBFC0AA04E0000000"), _c("0x3FC2073EC0000000"), _c("0xBFBF19B980000000"), _c("0x3FC1E52AA0000000"),
             _c("0xBFC55B1720000000"), _c("0x3FC99DA160000000"), _c("0xBFCFFFE440000000"), _c("0x3FD5554F00000000")]
# __nv_log1pf polynomial
LOG1PF_POLY = [_c("0xBFA737EF00000000"), _c("0x3FBB000240000000"), _c("0xBFC0EF1C00000000"), _c("0x3FC28C8EA0000000"),
               _c("0xBFC54D1BA0000000"), _c("0x3FC995F3C0000000"), _c("0xBFD0000840000000"), _c("0x3FD5555CC0000000")]


def ftz(x: np.ndarray) -> np.ndarray:
    """Flush subnormals to a sign-preserving zero (`.ftz` on inputs and results)."""
    x = np.asarray(x, dtype=F32)
    sub = (np.abs(x) < FLT_MIN) & (x != 0)
    return np.where(sub, np.copysign(F32(0.0), x), x).astype(F32)


def _two_sum(p: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Knuth TwoSum in float64: s = fl(p + c), e = (p + c) - s exactly."""
    s = p + c
    bb = s - p
    e = (p - (s - bb)) + (c - bb)
    return s, e


def _round_exact_to_f32(s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """RN-even float32 of the exact value s + e, where s is float64 and |e| < ulp64(s)/2 (TwoSum error).  Rounding s alone
    is wrong only when s is exactly a float32 midpoint and e != 0 decides the side."""
    r = s.astype(F32)
    r64 = r.astype(np.float64)
    hazard = (e != 0) & (r64 != s)
    if np.any(hazard):
        lo = np.where(r64 > s, np.nextafter(r, F32(-np.inf)), r).astype(F32)
        hi = np.where(r64 > s, r, np.nextafter(r, F32(np.inf))).astype(F32)
        mid = (lo.astype(np.float64) + hi.astype(np.float64)) * 0.5
        is_mid = hazard & (mid == s)
        r = np.where(is_mid & (e > 0), hi, np.where(is_mid & (e < 0), lo, r)).astype(F32)
    return r


def fma_f32(a, b, c) -> np.ndarray:
    """`fma.rn.ftz.f32`: one rounding of a*b + c (float32 inputs), subnormals flushed on input and output."""
    a, b, c = (ftz(np.broadcast_to(np.asarray(x, dtype=F32), np.broadcast_shapes(np.shape(a), np.shape(b), np.shape(c))))
               for x in (a, b, c))
    with np.errstate(invalid="ignore", over="ignore"):
        p = a.astype(np.float64) * b.astype(np.float64)                # exact: 24 + 24 bits
        s, e = _two_sum(p, c.astype(np.float64))
        return ftz(_round_exact_to_f32(s, e))


def add_rz_f32(a, b) -> np.ndarray:
    """`add.rz.ftz.f32`: a + b rounded toward zero."""
    a, b = (ftz(np.broadcast_to(np.asarray(x, dtype=F32), np.broadcast_shapes(np.shape(a), np.shape(b)))) for x in (a, b))
    s, e = _two_sum(a.astype(np.float64), b.astype(np.float64))
    r = s.astype(F32)
    r64 = r.astype(np.float64)
    t = np.where(np.abs(r64) > np.abs(s), np.nextafter(r, F32(0.0)), r).astype(F32)      # truncate s toward zero
    back = (r64 == s) & (e != 0) & (np.sign(e) == -np.sign(s))                        # exact value just inside s
    t = np.where(back, np.nextafter(t, F32(0.0)), t).astype(F32)
    return ftz(t)


def nv_logf(a) -> np.ndarray:
    """libdevice `__nv_logf` (accurate, FTZ), element-wise on float32."""
    a = np.asarray(a, dtype=F32)
    small = a < FLT_MIN
    a2 = np.where(small, (a * TWO_P23).astype(F32), a).astype(F32)
    i0 = np.where(small, F32(-23.0), F32(0.0)).astype(F32)
    ai = a2.view(np.int32)
    with np.errstate(over="ignore"):
        e = ((ai - np.int32(1059760811)) & np.int32(-8388608)).astype(np.int32)
        m = (ai - e).view(F32)
    i2 = fma_f32(e.astype(F32), TWO_M23, i0)
    f = (m - F32(1.0)).astype(F32)
    r = fma_f32(LOGF_POLY[0], f, LOGF_POLY[1])
    for k in LOGF_POLY[2:]:
        r = fma_f32(r, f, k)
    r = fma_f32(r, f, F32(-0.5))
    r = (r * f).astype(F32)
    r = fma_f32(r, f, f)
    r = fma_f32(i2, LN2, r)
    bad = a2.view(np.uint32) >= np.uint32(0x7F800000)                  # inf, nan, negative
    if np.any(bad):
        r = np.where(bad, fma_f32(a2, INF, INF), r).astype(F32)
    r = np.where(a2 == 0, F32(-np.inf), r).astype(F32)
    return r


def nv_log1pf(a) -> np.ndarray:
    """libdevice `__nv_log1pf` (accurate, FTZ), element-wise on float32."""
    a = np.asarray(a, dtype=F32)
    s = add_rz_f32(a, F32(1.0))
    si = s.view(np.int32)
    with np.errstate(over="ignore"):
        e = ((si - np.int32(1061158912)) & np.int32(-8388608)).astype(np.int32)
        m1 = (a.view(np.int32) - e).view(F32)
        t = (np.int32(1082130432) - e).view(F32)
    r0 = fma_f32(F32(0.25), t, F32(-1.0))
    u = (m1 + r0).astype(F32)
    ef = (e.astype(F32) * TWO_M23).astype(F32)
    r = fma_f32(LOG1PF_POLY[0], u, LOG1PF_POLY[1])
    for k in LOG1PF_POLY[2:]:
        r = fma_f32(r, u, k)
    r = fma_f32(r, u, F32(-0.5))
    r = (r * u).astype(F32)
    r = fma_f32(r, u, u)
    r = fma_f32(ef, LN2, r)
    ab = a.view(np.uint32)
    bad = ab >= np.uint32(0x7F800000)
    if np.any(bad):
        big = a.view(np.int32) > np.int32(-1082130432)
        r = np.where(bad & big, fma_f32(a, INF, INF), r).astype(F32)
        r = np.where(bad & (a == 0), F32(-0.0), r).astype(F32)
    return r


def gumbel_noise_exact(seed: int, pos: int, vocab_size: int, *, is_drafting: bool = False) -> np.ndarray:
    """vLLM `gumbel_noised_argmax`'s fp32 noise row, bit-exact: u = max(tl.rand(randint(seed, pos), v), 2^-31ish);
    g = -logf(-log1pf(-u))."""
    gs = R.gumbel_seed_of(seed, pos, is_drafting=is_drafting)
    keys = np.arange(int(vocab_size), dtype=np.int32)
    u = np.maximum(R.tl_rand(gs, keys, seed_bits=32), R.TL_RAND_MIN_F32).astype(F32)
    return (-nv_logf((-nv_log1pf((-u).astype(F32))).astype(F32))).astype(F32)


def compare_dir(probe_dir: Path) -> dict:
    """PROBE_DIR/g_<seed>_<pos>_<V>.npy (GPU) vs the transcription; also the GPU `u`/`l1` dumps when present."""
    out: dict = {"schema": "verity-gen/ov-sampling-libdevice-compare/v1", "probes": []}
    for p in sorted(probe_dir.glob("g_*.npy")):
        _, seed, pos, V = p.stem.split("_")
        seed, pos, V = int(seed), int(pos), int(V)
        g_gpu = np.load(p)
        g_cpu = gumbel_noise_exact(seed, pos, V)
        neq = int((g_gpu.view(np.uint32) != g_cpu.view(np.uint32)).sum())
        rec = {"seed": seed, "pos": pos, "V": V, "n_neq_g": neq, "bit_exact_g": neq == 0}
        l1 = probe_dir / f"l1_{seed}_{pos}_{V}.npy"
        if l1.exists():
            gs = R.gumbel_seed_of(seed, pos)
            u = np.maximum(R.tl_rand(gs, np.arange(V, dtype=np.int32), seed_bits=32), R.TL_RAND_MIN_F32).astype(F32)
            l1_cpu = nv_log1pf((-u).astype(F32))
            rec["n_neq_log1pf"] = int((np.load(l1).view(np.uint32) != l1_cpu.view(np.uint32)).sum())
        if neq:
            idx = np.flatnonzero(g_gpu.view(np.uint32) != g_cpu.view(np.uint32))[:5]
            rec["examples"] = [{"v": int(i), "gpu": float(g_gpu[i]), "cpu": float(g_cpu[i])} for i in idx]
        out["probes"].append(rec)
    out["all_bit_exact"] = bool(out["probes"]) and all(r["bit_exact_g"] for r in out["probes"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True, help="directory with the pod probe's g_<seed>_<pos>_<V>.npy dumps")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = compare_dir(Path(a.compare))
    txt = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(txt)
    return 0 if rep["all_bit_exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
