#!/usr/bin/env python3
"""The exact-run census for block-scaled FP4 (E2M1) on sm_120: when a chain's words stop depending on its order.

On the block-scaled step (``verity.ml.tc.models.BlockScaledAlignAdd``) every 16-term group dot of E2M1 products is an exact
integer, the block scales multiply it losslessly, and one align-add per k64 step truncates the four scaled dots and the
FP32 accumulator onto a common grid and normalises RZ.  A step is *exact* when no bit is lost anywhere in it (no addend
truncated, no bit dropped by the normalisation): its word is then the exact sum, which any exact method (int8 IMMA over
E2M1's integers, a 2:4 sparse instruction, Strassen, any order) reproduces bit for bit.  This script measures how often
that happens, on adversarial families and on real activations and weights, and how much structurally skippable work
(all-zero blocks, groups truncated away, 2:4-sparse steps) the operands carry.

The atom is parametric: ``--atom`` names any ``BlockScaledAlignAdd`` in ``verity.ml.tc.models.MODELS`` (the stand-ins
``blackwell_sm120_nvf4_m16n8k64`` and ``blackwell_sm120_mxf4_m16n8k64``, pinned from the RTX 5090, until the RTX PRO recheck),
and ``--frac-bits`` / ``--acc-window`` override its adder for sensitivity runs.  The noise modes are stand-ins until the FP4
PoUW design fixes its own (``none``: the clean path; ``row``: Pearl's FP8 rule, RMS delta*rms(row); ``block``: RMS
sigma_c code units per block, after the block scale).

    python benchmarks/pouw/fp4_census.py selftest
    python benchmarks/pouw/fp4_census.py synth --k 8192 --pairs 128 --out synth.json
    python benchmarks/pouw/fp4_census.py real --acts acts_dir --out real.json      # npz files from fp4_quality.py --dump-acts
    python benchmarks/pouw/fp4_census.py breakeven
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_result  # noqa: E402

E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_NUM = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int64)      # |value| in halves
E2M1_MAX = 6.0
E4M3_MAX = 448.0
NEG = -(1 << 40)                                                      # "minus infinity" for integer exponent maxima


# ---- the atom, vectorised ---------------------------------------------------------------------------------------------

class Atom:
    """``BlockScaledAlignAdd.step_scaled`` over P independent chains at once, with a per-step loss flag.

    Operands are signed E2M1 numerators (halves) and per-block scale mantissas / exponents (``_scale_dyadic``'s pairs).
    """

    def __init__(self, model, frac_bits: int | None = None, acc_window: int | None = None):
        self.name = model.name + ("" if frac_bits is None and acc_window is None else f"[frac={frac_bits},acc={acc_window}]")
        self.k, self.group, self.block = model.k_per_instruction, model.group, model.block
        self.scale = model.scale
        self.frac_bits = model.frac_bits if frac_bits is None else frac_bits
        self.acc_window = model.acc_window if acc_window is None else acc_window
        self.grid_floor = model.grid_floor
        self.signed_underflow = model.underflow_zero == "signed"
        self.groups = self.k // self.group
        self._gidx = (np.arange(self.groups) * self.group) // self.block
        self.raise_overflow = True

    def step(self, acc: np.ndarray, an: np.ndarray, bn: np.ndarray, sa: tuple, sb: tuple) -> tuple[np.ndarray, dict]:
        """acc: (P,) exact FP32 values (float64); an, bn: (P, k) numerators; sa, sb: ((P, k/block) mantissas, exponents).

        Returns the new accumulators and the step's diagnostics: ``lost`` (P,), and per group (P, groups) ``part``
        (participates), ``nonzero`` (some product nonzero), ``cancel`` (participates, dot 0), ``trunc`` (nonzero scaled dot
        truncated to 0 by the grid), ``glost`` (the group's scaled dot lost bits).
        """
        P, G, g = an.shape[0], self.groups, self.group
        prod = an * bn
        gd = prod.reshape(P, G, g).sum(-1)
        nonzero = (prod != 0).reshape(P, G, g).any(-1)
        ma = sa[0][:, self._gidx] * sb[0][:, self._gidx]
        X = -2 + sa[1][:, self._gidx] + sb[1][:, self._gidx]
        mant = gd * ma
        part = nonzero & (ma != 0)
        # the accumulator as (mantissa, exponent) and its exponent-field anchor
        a = np.abs(acc)
        _, e = np.frexp(a)
        normal = a >= 2.0 ** -126
        efield = np.where(normal, e - 1, -126).astype(np.int64)
        acc_e = np.where(normal, efield - 23, -149)
        acc_m = np.rint(np.ldexp(a, -acc_e)).astype(np.int64)
        acc_m = np.where(acc < 0, -acc_m, acc_m)
        anchor_g = np.where(part, X - self.frac_bits, NEG).max(-1)
        anchor_a = np.where(acc_m != 0, efield - (self.acc_window - 1), NEG)
        live = (anchor_g > NEG) | (acc_m != 0)
        lsb = np.maximum(np.maximum(anchor_g, anchor_a), self.grid_floor)
        # addends truncated toward zero onto 2^lsb
        sh = lsb[:, None] - X
        absm = np.abs(mant)
        up = np.clip(-sh, 0, 62)
        dn = np.clip(sh, 0, 62)
        v = np.where(sh <= 0, absm << up, absm >> dn)
        glost = (sh > 0) & (mant != 0) & ((absm & ((np.int64(1) << dn) - 1)) != 0) | ((sh >= 62) & (mant != 0))
        trunc = (mant != 0) & (v == 0)
        v = np.where(mant < 0, -v, v)
        sha = lsb - acc_e
        absa = np.abs(acc_m)
        va = np.where(sha <= 0, absa << np.clip(-sha, 0, 62), absa >> np.clip(sha, 0, 62))
        alost = (sha > 0) & (acc_m != 0) & ((absa & ((np.int64(1) << np.clip(sha, 0, 62)) - 1)) != 0)
        va = np.where(acc_m < 0, -va, va)
        total = v.sum(-1) + va
        # RZ normalisation to binary32
        mag = np.abs(total)
        _, bl = np.frexp(mag.astype(np.float64))
        e_val = bl.astype(np.int64) - 1 + lsb
        grid = np.where(e_val >= -126, e_val - 23, -149)
        shn = grid - lsb
        q = np.where(shn > 0, mag >> np.clip(shn, 0, 62), mag << np.clip(-shn, 0, 62))
        nlost = (shn > 0) & (mag != 0) & ((mag & ((np.int64(1) << np.clip(shn, 0, 62)) - 1)) != 0)
        _, qb = np.frexp(q.astype(np.float64))
        overflow = (q > 0) & (qb.astype(np.int64) - 1 + grid > 127)
        if self.raise_overflow and overflow.any():
            raise OverflowError("sum leaves the binary32 range (outside the model)")
        out = np.ldexp(q.astype(np.float64), np.minimum(grid, 127))
        out = np.where(total < 0, -out, out)
        out = np.where(q == 0, -0.0 if self.signed_underflow else 0.0, out)
        out = np.where((q == 0) & (total >= 0), 0.0, out)
        out = np.where(live, out, 0.0)
        lost = live & (glost.any(-1) | alost | nlost)
        return out, {"lost": lost, "part": part, "nonzero": nonzero, "cancel": part & (gd == 0), "trunc": trunc, "glost": glost,
                     "overflow": overflow}


def scale_pairs(fmt: str, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale bytes -> (mantissa, exponent) as ``models._scale_dyadic`` (UE4M3 for nvf4, UE8M0 for mxf4)."""
    codes = codes.astype(np.int64)
    if fmt == "ue8m0":
        return np.full_like(codes, 8), codes - 130
    e, m = (codes >> 3) & 0xF, codes & 7
    return np.where(e == 0, m, 8 + m), np.where(e == 0, -9, e - 10)


def numerators(codes: np.ndarray) -> np.ndarray:
    n = E2M1_NUM[codes & 7]
    return np.where(codes & 8, -n, n)


def f32_word(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def selftest(cases: int = 4000, seed: int = 7) -> dict:
    """Bit-exact against ``verity.ml.tc`` on random steps with arbitrary accumulators, and on chains."""
    from verity.ml.tc.models import BLACKWELL_SM120_MXF4, BLACKWELL_SM120_NVF4
    rng = np.random.default_rng(seed)
    report = {}
    for model in (BLACKWELL_SM120_NVF4, BLACKWELL_SM120_MXF4):
        atom = Atom(model)
        ns = model.k // model.block
        a = rng.integers(0, 16, (cases, model.k))
        b = rng.integers(0, 16, (cases, model.k))
        a[: cases // 4, ::3] = 0                                        # sparse operands: non-participating groups
        if model.scale == "ue4m3":
            sa, sb = rng.integers(0, 0x7F, (cases, ns)), rng.integers(0, 0x7F, (cases, ns))
            sa[cases // 2: 3 * cases // 4] = rng.integers(0, 12, (cases // 4, ns))      # subnormal / zero scales
        else:
            sa, sb = rng.integers(60, 190, (cases, ns)), rng.integers(60, 190, (cases, ns))
            sa[cases // 2: 3 * cases // 4] = rng.integers(0, 40, (cases // 4, ns))      # deep: the grid floor, underflow
        mag = np.exp2(rng.integers(-150, 60, cases).astype(np.float64)) * rng.standard_normal(cases)
        acc = np.where(rng.random(cases) < 0.2, 0.0, mag).astype(np.float32).astype(np.float64)
        atom.raise_overflow = False
        got, fl = atom.step(acc, numerators(a), numerators(b), scale_pairs(model.scale, sa), scale_pairs(model.scale, sb))
        bad = skipped = 0
        for i in range(cases):
            try:
                want = model.step_scaled(f32_word(acc[i]), [int(x) for x in a[i]], [int(x) for x in b[i]],
                                         [int(x) for x in sa[i]], [int(x) for x in sb[i]])
            except Exception:                                           # outside the model: the flag must say so
                skipped += 1
                bad += int(not fl["overflow"][i])
                continue
            if want != f32_word(got[i]) or fl["overflow"][i]:
                bad += 1
        # chains: 8 steps from +0
        P, S = 200, 8
        a = rng.integers(0, 16, (P, S * model.k))
        b = rng.integers(0, 16, (P, S * model.k))
        lo, hi = (100, 140) if model.scale == "ue8m0" else (0x30, 0x50)
        sa, sb = rng.integers(lo, hi, (P, S * ns)), rng.integers(lo, hi, (P, S * ns))
        acc = np.zeros(P)
        k = model.k
        for s in range(S):
            acc, _ = atom.step(acc, numerators(a[:, s * k:(s + 1) * k]), numerators(b[:, s * k:(s + 1) * k]),
                               scale_pairs(model.scale, sa[:, s * ns:(s + 1) * ns]), scale_pairs(model.scale, sb[:, s * ns:(s + 1) * ns]))
        cbad = sum(model.chain(0, [int(x) for x in a[i]], [int(x) for x in b[i]], [int(x) for x in sa[i]],
                               [int(x) for x in sb[i]]) != f32_word(acc[i]) for i in range(P))
        report[model.name] = {"step_cases": cases, "step_outside_model": skipped, "step_mismatches": bad, "chain_cases": P,
                              "chain_steps": S, "chain_mismatches": int(cbad)}
    return report


# ---- quantizers ------------------------------------------------------------------------------------------------------

def e2m1_codes(y: np.ndarray) -> np.ndarray:
    """RNE onto E2M1 with saturation at +-6 (ties to the even mantissa); returns 4-bit codes."""
    a = np.minimum(np.abs(y), E2M1_MAX)
    hi = np.searchsorted(E2M1_VALUES, a, side="left").clip(0, 7)
    lo = (hi - 1).clip(0, 7)
    dlo, dhi = a - E2M1_VALUES[lo], E2M1_VALUES[hi] - a
    pick = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    pick = np.where(a == E2M1_VALUES[hi], hi, pick)
    return (pick | np.where((y < 0) & (pick > 0), 8, 0)).astype(np.int64)


E4M3_FIN = np.array([(m * 2.0 ** -9 if e == 0 else (8 + m) * 2.0 ** (e - 10)) for e in range(16) for m in range(8)][:127])


def e4m3_codes_unsigned(x: np.ndarray) -> np.ndarray:
    """RNE onto UE4M3 (codes 0..126), saturating at 448."""
    a = np.minimum(np.abs(x), E4M3_MAX)
    hi = np.searchsorted(E4M3_FIN, a, side="left").clip(0, 126)
    lo = (hi - 1).clip(0, 126)
    dlo, dhi = a - E4M3_FIN[lo], E4M3_FIN[hi] - a
    pick = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    return np.where(a == E4M3_FIN[hi], hi, pick).astype(np.int64)


def quantize(x: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(rows, k) -> (E2M1 codes, scale bytes per block, the global factor g), x ~= g * scale(block) * value(code).

    nvf4: NVIDIA's two-level recipe -- a per-tensor FP32 scale g = amax/(6*448), per 16-block UE4M3 scale
          e4m3(amax_b/(6g)), elements RNE(x/(g*s_b)) saturating at 6.
    mxf4: OCP MX v1.0 -- per 32-block UE8M0 scale 2^(floor(log2 amax_b) - 2), elements RNE(x/2^e) saturating at 6.
    """
    rows, k = x.shape
    if fmt == "nvf4":
        g = float(np.abs(x).max()) / (E2M1_MAX * E4M3_MAX) or 1.0
        xb = x.reshape(rows, k // 16, 16) / g
        sc = e4m3_codes_unsigned(np.abs(xb).max(-1) / E2M1_MAX)
        m, e = scale_pairs("ue4m3", sc)
        s = np.ldexp(m.astype(np.float64), e)
        codes = e2m1_codes(np.where(s[..., None] > 0, xb / np.where(s > 0, s, 1.0)[..., None], 0.0))
        return codes.reshape(rows, k), sc, g
    xb = x.reshape(rows, k // 32, 32)
    amax = np.abs(xb).max(-1)
    _, e = np.frexp(np.where(amax > 0, amax, 1.0))
    ex = np.clip(e - 1 - 2, -127, 127)
    ex = np.where(amax > 0, ex, -127)
    codes = e2m1_codes(xb / np.ldexp(1.0, ex)[..., None])
    return codes.reshape(rows, k), (ex + 127).astype(np.int64), 1.0


def dequantize(codes: np.ndarray, sc: np.ndarray, g: float, fmt: str) -> np.ndarray:
    rows, k = codes.shape
    blk = 16 if fmt == "nvf4" else 32
    m, e = scale_pairs("ue4m3" if fmt == "nvf4" else "ue8m0", sc)
    s = np.ldexp(m.astype(np.float64), e)
    v = numerators(codes).astype(np.float64) / 2
    return (v.reshape(rows, k // blk, blk) * s[..., None]).reshape(rows, k) * g


def noise_lines(rows: int, k: int, rng) -> np.ndarray:
    """Pearl's rank-32 noise rows of unit RMS (lines of +-(1..128))."""
    E = rng.integers(1, 129, (rows, 32)) * rng.choice([-1, 1], (rows, 32))
    F = rng.integers(1, 129, (k, 32)) * rng.choice([-1, 1], (k, 32))
    Z = (E @ F.T).astype(np.float64)
    return Z / np.sqrt((Z ** 2).mean(1, keepdims=True))


def noisy(x: np.ndarray, fmt: str, mode: str, delta: float, rng) -> np.ndarray:
    """The stand-in noise, added before the quantizer: ``row`` RMS delta*rms(row); ``block`` RMS delta*0.5 code units of
    the block's clean scale (amax_b/6), so it survives any in-block outlier."""
    if mode == "none":
        return x
    Z = noise_lines(x.shape[0], x.shape[1], rng)
    if mode == "row":
        return x + delta * np.sqrt((x ** 2).mean(1, keepdims=True)) * Z
    blk = 16 if fmt == "nvf4" else 32
    xb = x.reshape(x.shape[0], -1, blk)
    s = np.abs(xb).max(-1, keepdims=True) / E2M1_MAX
    s = np.where(s > 0, s, np.sqrt((x ** 2).mean(1, keepdims=True))[..., None] / E2M1_MAX + 1e-30)
    return (xb + delta * 0.5 * s * Z.reshape(xb.shape)).reshape(x.shape)


# ---- chains ------------------------------------------------------------------------------------------------------------

def runs_of(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """flags (P, T) bool -> (longest run of True per row, run length of the run containing each position)."""
    P, T = flags.shape
    longest = np.zeros(P, dtype=np.int64)
    at = np.zeros((P, T), dtype=np.int64)
    for p in range(P):
        f = flags[p]
        t = 0
        while t < T:
            if not f[t]:
                t += 1
                continue
            u = t
            while u < T and f[u]:
                u += 1
            at[p, t:u] = u - t
            longest[p] = max(longest[p], u - t)
            t = u
    return longest, at


def chains(atom: Atom, fmt: str, qa, qb, pairs: np.ndarray) -> dict:
    """Run the chains for the given (row of A, row of B) pairs from +0 over all k; returns per-step flags."""
    (ca, sa), (cb, sb) = qa, qb
    ia, ib = pairs[:, 0], pairs[:, 1]
    P, K = len(pairs), ca.shape[1]
    T, k, ns = K // atom.k, atom.k, atom.k // atom.block
    scale = atom.scale
    na, nb = numerators(ca[ia]), numerators(cb[ib])
    pa, pb = scale_pairs(scale, sa[ia]), scale_pairs(scale, sb[ib])
    acc = np.zeros(P)
    out = {f: np.zeros((P, T), dtype=bool) for f in ("lost", "ident")}
    gflags = {f: np.zeros((P, T, atom.groups), dtype=bool) for f in ("part", "nonzero", "cancel", "trunc", "glost")}
    for t in range(T):
        d, fl = atom.step(acc, na[:, t * k:(t + 1) * k], nb[:, t * k:(t + 1) * k],
                          (pa[0][:, t * ns:(t + 1) * ns], pa[1][:, t * ns:(t + 1) * ns]),
                          (pb[0][:, t * ns:(t + 1) * ns], pb[1][:, t * ns:(t + 1) * ns]))
        out["lost"][:, t] = fl["lost"]
        out["ident"][:, t] = d == acc
        for f in gflags:
            gflags[f][:, t] = fl[f]
        acc = d
    out.update(gflags)
    out["final"] = acc
    return out


RUN_LENGTHS = (1, 2, 4, 8, 14, 32, 64, 128)


def summarize_chains(fl: dict, tile: tuple[int, int] | None = None) -> dict:
    exact = ~fl["lost"]
    P, T = exact.shape
    longest, at = runs_of(exact)
    prefix = np.where(exact.all(1), T, np.argmin(exact, axis=1))
    nz = fl["nonzero"]
    res = {
        "steps": int(T),
        "exact_steps": float(exact.mean()),
        "exact_chains": float(exact.all(1).mean()),
        "exact_prefix_mean_steps": float(prefix.mean()),
        "longest_exact_run_steps": {"mean": float(longest.mean()), "p50": float(np.median(longest)), "max": int(longest.max())},
        "steps_in_exact_runs_ge": {str(L): float((at >= L).mean()) for L in RUN_LENGTHS if L <= T},
        "identity_steps": float(fl["ident"].mean()),
        "groups_all_products_zero": float((~nz).mean()),
        "groups_cancel_to_zero": float(fl["cancel"].mean()),
        "groups_truncated_to_zero": float(fl["trunc"].mean()),
        "groups_losing_bits": float(fl["glost"].mean()),
    }
    if tile is not None:
        tm, tn = tile
        te = exact.reshape(-1, tm * tn, T).all(1)                      # every word of the tile exact at that step
        tl, tat = runs_of(te)
        res["tile"] = {"shape": [tm, tn], "tiles": int(te.shape[0]), "exact_steps": float(te.mean()),
                       "longest_exact_run_steps": {"mean": float(tl.mean()), "max": int(tl.max())},
                       "steps_in_exact_runs_ge": {str(L): float((tat >= L).mean()) for L in RUN_LENGTHS if L <= T},
                       "identity_steps": float(fl["ident"].reshape(-1, tm * tn, T).all(1).mean()),
                       "groups_all_products_zero": float((~nz).reshape(-1, tm * tn, T, nz.shape[-1]).all(1).mean())}
    return res


def operand_stats(codes: np.ndarray, sc: np.ndarray, fmt: str, clean_codes: np.ndarray | None, tile_rows: int = 16) -> dict:
    """Structure an adversary can see without computing products: zero blocks, 2:4 compatibility, duplicate blocks."""
    rows, k = codes.shape
    blk = 16 if fmt == "nvf4" else 32
    z = (codes & 7) == 0
    zb = z.reshape(rows, k // 16, 16).all(-1)
    quad_ok = z.reshape(rows, k // 4, 4).sum(-1) >= 2
    steps = k // 64
    tr = rows // tile_rows
    q = quad_ok[: tr * tile_rows].reshape(tr, tile_rows, steps, 16)
    sparse_tile_steps = q.all(-1).all(1)                               # mma.sp usable on the whole 16 x 64 fragment
    # duplicate 16-blocks (codes and scale) among the rows of a tile at the same k position
    s16 = np.repeat(sc, blk // 16, axis=1)
    keyv = np.concatenate([codes.reshape(rows, k // 16, 16), s16[..., None]], -1)
    dup = 0
    for t in range(tr):
        kb = keyv[t * tile_rows:(t + 1) * tile_rows]
        for j in range(k // 16):
            u = np.unique(kb[:, j], axis=0)
            dup += tile_rows - len(u)
    res = {"zero_elements": float(z.mean()), "zero_blocks16": float(zb.mean()), "quads_2of4": float(quad_ok.mean()),
           "tile_steps_2of4": float(sparse_tile_steps.mean()) if tr else None,
           "dup_blocks16_in_tile": dup / max(1, tr * tile_rows * (k // 16)), "tile_rows": tile_rows}
    if clean_codes is not None:
        res["codes_changed_by_noise"] = float((codes != clean_codes).mean())
    return res


# ---- families --------------------------------------------------------------------------------------------------------

FAMILIES = ("gaussian", "student-t4", "grid-aligned", "uniform-block-amax", "outlier-in-block", "sparse-2of4", "zero-blocks",
            "scale-spread-30", "acc-freeze", "coherent", "rank-1")


def family(name: str, rows: int, k: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """(A, B), both (rows, k): useful activations and weights before noise and quantization."""
    if name == "gaussian":
        return rng.standard_normal((rows, k)), 0.02 * rng.standard_normal((rows, k))
    if name == "student-t4":
        return rng.standard_t(4, (rows, k)), 0.02 * rng.standard_t(4, (rows, k))
    if name == "grid-aligned":                          # already E2M1 values, one power of two: quantization is lossless
        v = lambda: E2M1_VALUES[rng.integers(0, 8, (rows, k))] * rng.choice([-1.0, 1.0], (rows, k))  # noqa: E731
        A, B = v(), v() * 2.0 ** -6
        A[:, ::16], B[:, ::16] = 6.0, 6.0 * 2.0 ** -6    # every block's amax is 6: every scale equal
        return A, B
    if name == "uniform-block-amax":                    # random values, every 16-block normalised to the same amax
        A, B = rng.standard_normal((rows, k)), rng.standard_normal((rows, k))
        for M in (A, B):
            Mb = M.reshape(rows, -1, 16)
            Mb /= np.abs(Mb).max(-1, keepdims=True)
        return A, B * 0.02
    if name == "outlier-in-block":                      # one entry per 16-block at 64x the rest: the rest round to 0
        A, B = rng.standard_normal((rows, k)), 0.02 * rng.standard_normal((rows, k))
        A.reshape(rows, -1, 16)[:, :, 0] = 64.0
        B.reshape(rows, -1, 16)[:, :, 0] = 64.0 * 0.02
        return A, B
    if name == "sparse-2of4":
        A, B = rng.standard_normal((rows, k)), 0.02 * rng.standard_normal((rows, k))
        A.reshape(rows, -1, 4)[:, :, :2] = 0.0
        return A, B
    if name == "zero-blocks":                           # half of A's 16-blocks are exactly zero
        A, B = rng.standard_normal((rows, k)), 0.02 * rng.standard_normal((rows, k))
        A.reshape(rows, -1, 32)[:, :, :16] = 0.0
        return A, B
    if name == "scale-spread-30":                       # block amaxes spread over 2^30 within each row
        A, B = rng.standard_normal((rows, k)), 0.02 * rng.standard_normal((rows, k))
        A.reshape(rows, -1, 32)[...] *= np.exp2(rng.uniform(-30, 0, (rows, k // 32)))[..., None]
        return A, B
    if name == "acc-freeze":                            # one huge first block, the rest 2^36 below
        A, B = rng.standard_normal((rows, k)), 0.02 * rng.standard_normal((rows, k))
        A[:, :32] *= 2.0 ** 18
        B[:, :32] *= 2.0 ** 18
        return A, B
    if name == "coherent":                              # B's row equals A's row: every product positive
        A = rng.standard_normal((rows, k))
        return A, 0.02 * A.copy()
    if name == "rank-1":
        u = rng.standard_normal(k)
        return rng.standard_normal((rows, 1)) * u, 0.02 * rng.standard_normal((rows, k))
    raise ValueError(name)


def census_matrix(atom: Atom, fmt: str, A: np.ndarray, B: np.ndarray, noise: str, delta: float, rng, pairs: int,
                  tile: tuple[int, int] = (16, 8)) -> dict:
    """Quantize A (activations) and B (weights) with the stand-in noise, then: paired words, one all-pairs tile, operands."""
    ya, yb = noisy(A, fmt, noise, delta, rng), noisy(B, fmt, noise, delta, rng)
    ca, sa, _ = quantize(ya, fmt)
    cb, sb, _ = quantize(yb, fmt)
    clean_a = quantize(A, fmt)[0] if noise != "none" else None
    n = min(pairs, A.shape[0], B.shape[0])
    words = summarize_chains(chains(atom, fmt, (ca, sa), (cb, sb), np.stack([np.arange(n), np.arange(n)], 1)))
    tm, tn = tile
    tp = np.array([(i, j) for i in range(tm) for j in range(tn)])
    ntile = min(A.shape[0] // tm, B.shape[0] // tn, 4)
    tp = np.concatenate([tp + [t * tm, t * tn] for t in range(ntile)])
    tiles = summarize_chains(chains(atom, fmt, (ca, sa), (cb, sb), tp), tile=tile)["tile"]
    return {"words": words, "tile": tiles, "A": operand_stats(ca, sa, fmt, clean_a), "B": operand_stats(cb, sb, fmt, None)}


def base_split(fmt: str, A: np.ndarray, noise: str, delta: float, rng, tile_rows: int = 16) -> dict:
    """The salt-free base split (design X-FP4-1), as a cheaper-computation cost.

    The cheapest word-reproducing prover precomputes ``base @ base.T`` before the salt (free, salt-independent: ``base``
    is the clean-path codes it can guess from the known input) and, per salt, corrects with only the changed codes
    ``Delta = codes(salt) != base``.  Because chains are exact, this split is exact.  What it *saves* is set by how dense
    ``Delta`` is: a 16x64 A fragment whose ``Delta`` is 2:4 everywhere (<=2 changed per 4 consecutive k, per row) is one
    ``mma.sp`` pass (0.5x); a denser fragment needs the dense route or two sparse passes (1.0x); an all-unchanged fragment
    is skipped (0x).  So the per-call A-side arithmetic the cheater pays, in dense-FP4-MAC units, is the mean over
    fragments of {0, 0.5, 1.0}.  It *ties* the honest 1.0x when ``Delta`` is dense, and *beats* it (breaks gamma) if many
    fragments are <=2:4.  (B's ``Delta`` is per weight-epoch, amortized; this reports the per-call A side.)
    """
    base = quantize(A, fmt)[0]
    salt = quantize(noisy(A, fmt, noise, delta, rng), fmt)[0]
    delta_map = base != salt
    rows, K = delta_map.shape
    nr, nk = rows - rows % tile_rows, K - K % 64
    density = float(delta_map.mean())
    # Theory §5 Q2/Q3's statistic: per (row, 4-group of k), a code is salt-dead when the base already holds it (Delta = 0).
    # >= 2 salt-dead <=> <= 2 salt-dependent <=> that group's Delta fits one 2:4 half (the per-4-group 0.5x debit).
    g4 = delta_map[:, : K - K % 4].reshape(rows, -1, 4).sum(-1)
    share_ge2_dead = float((g4 <= 2).mean())
    iid_ge2_dead = sum(math.comb(4, c) * density ** c * (1 - density) ** (4 - c) for c in range(3))
    salt_dead = {"share_4groups_ge2_salt_dead": share_ge2_dead,
                 "share_4groups_ge2_salt_dead_iid": float(iid_ge2_dead),
                 "debit_2of4_per_mac": 0.5 * share_ge2_dead}
    if nr == 0 or nk == 0:
        return {"code_change_density": density, "fragments": 0, **salt_dead}
    d = delta_map[:nr, :nk].reshape(nr // tile_rows, tile_rows, nk // 64, 64)
    d = d.transpose(0, 2, 1, 3).reshape(-1, tile_rows, 16, 4)          # [frag, row, group-of-4 along k, 4]
    changed_per_group = d.sum(-1)                                       # [frag, row, group] in 0..4
    frag_changed = changed_per_group.reshape(len(changed_per_group), -1)
    skippable = frag_changed.sum(1) == 0
    one_pass = (~skippable) & (changed_per_group.max((1, 2)) <= 2)      # every (row,group) has <=2 changed
    dense = ~(skippable | one_pass)
    cost = np.where(skippable, 0.0, np.where(one_pass, 0.5, 1.0))
    return {"code_change_density": density,
            "fragments": int(len(cost)),
            "frag_skippable": float(skippable.mean()),
            "frag_one_pass_2of4": float(one_pass.mean()),
            "frag_dense": float(dense.mean()),
            "cheater_A_cost_per_dense_mac": float(cost.mean()),
            "beats_honest": bool(cost.mean() < 0.999),
            **salt_dead}


def atoms_from(args) -> dict[str, tuple[Atom, str]]:
    from verity.ml.tc.models import MODELS
    out = {}
    for name in args.atom:
        m = MODELS[name]
        fmt = "nvf4" if m.scale == "ue4m3" else "mxf4"
        out[name] = (Atom(m, args.frac_bits, args.acc_window), fmt)
    return out


def cmd_synth(args) -> dict:
    rng = np.random.default_rng(args.seed)
    res = {"k": args.k, "pairs": args.pairs, "seed": args.seed, "noise": args.noise, "delta": args.delta, "results": []}
    fams = FAMILIES if args.families == "all" else tuple(args.families.split(","))
    for name, (atom, fmt) in atoms_from(args).items():
        for fam in fams:
            A, B = family(fam, max(args.pairs, 64), args.k, rng)
            for noise in args.noise.split(","):
                try:
                    r = census_matrix(atom, fmt, A, B, noise, args.delta, rng, args.pairs)
                except OverflowError as e:
                    r = {"outside_model": str(e)}
                row = {"atom": atom.name, "format": fmt, "family": fam, "noise": noise, **r}
                res["results"].append(row)
                w = r.get("words", {})
                print(json.dumps({"atom": atom.name, "family": fam, "noise": noise, "exact_steps": w.get("exact_steps"),
                                  "exact_chains": w.get("exact_chains"), "longest": w.get("longest_exact_run_steps"),
                                  "tile_exact": r.get("tile", {}).get("exact_steps"),
                                  "zero_blocks_A": r.get("A", {}).get("zero_blocks16"),
                                  "tile_steps_2of4_A": r.get("A", {}).get("tile_steps_2of4")}), flush=True)
    return res


def cmd_real(args) -> dict:
    """Real activations (npz: per linear, 'x' token rows) against real weight rows ('w'), from fp4_quality.py --dump-acts."""
    rng = np.random.default_rng(args.seed)
    res = {"acts": str(args.acts), "noise": args.noise, "delta": args.delta, "results": []}
    files = sorted(Path(args.acts).glob("*.npz"))
    for name, (atom, fmt) in atoms_from(args).items():
        for f in files:
            d = np.load(f)
            for lin in sorted({key.rsplit("/", 1)[0] for key in d.files}):
                X, W = d[f"{lin}/x"].astype(np.float64), d[f"{lin}/w"].astype(np.float64)
                if args.max_k and X.shape[1] > args.max_k:
                    X, W = X[:, : args.max_k], W[:, : args.max_k]
                for noise in args.noise.split(","):
                    try:
                        r = census_matrix(atom, fmt, X, W, noise, args.delta, rng, args.pairs)
                    except OverflowError as e:
                        r = {"outside_model": str(e)}
                    row = {"atom": atom.name, "format": fmt, "source": f.stem, "linear": lin, "k": int(X.shape[1]), "noise": noise, **r}
                    res["results"].append(row)
                    w = r.get("words", {})
                    print(json.dumps({"atom": atom.name, "src": f.stem, "linear": lin, "noise": noise, "exact_steps": w.get("exact_steps"),
                                      "exact_chains": w.get("exact_chains"), "longest": w.get("longest_exact_run_steps"),
                                      "tile_exact": r.get("tile", {}).get("exact_steps"),
                                      "zero_blocks_A": r.get("A", {}).get("zero_blocks16"),
                                      "quads_2of4_A": r.get("A", {}).get("quads_2of4")}), flush=True)
    return res


def cmd_basesplit(args) -> dict:
    """The base-split falsifier (design X-FP4-1): per family/format/noise, how cheap the salt-free split is (A side)."""
    rng = np.random.default_rng(args.seed)
    res = {"k": args.k, "pairs": args.pairs, "seed": args.seed, "noise": args.noise, "delta": args.delta, "results": []}
    fams = FAMILIES if args.families == "all" else tuple(args.families.split(","))
    for name, (atom, fmt) in atoms_from(args).items():
        for fam in fams:
            A, _ = family(fam, max(args.pairs, 64), args.k, rng)
            for noise in args.noise.split(","):
                if noise == "none":
                    continue                                            # the split needs a salt: noise is what the base can't guess
                r = base_split(fmt, A, noise, args.delta, rng)
                row = {"atom": atom.name, "format": fmt, "family": fam, "noise": noise, **r}
                res["results"].append(row)
                print(json.dumps({"format": fmt, "family": fam, "noise": noise, "density": round(r["code_change_density"], 3),
                                  "cheater_A_cost": round(r.get("cheater_A_cost_per_dense_mac", 1.0), 3),
                                  "one_pass_2of4": round(r.get("frag_one_pass_2of4", 0.0), 3),
                                  "ge2_salt_dead": round(r["share_4groups_ge2_salt_dead"], 3),
                                  "ge2_salt_dead_iid": round(r["share_4groups_ge2_salt_dead_iid"], 3),
                                  "beats_honest": r.get("beats_honest")}), flush=True)
    return res


# ---- break-even ------------------------------------------------------------------------------------------------------

#: Placeholder prices in units of one dense FP4 MAC (mma.sync kind::mxf4nvf4), from the RTX PRO Blackwell whitepaper v1.1
#: Table 4 (RTX PRO 6000 Workstation, dense, boost clock): FP4 2015.2, FP8 and INT8 1007.6, FP16/BF16 503.8 TFLOPS with FP32
#: accumulate, FP32 126.0 TFLOPS non-tensor.  Public, until GPU 4 measures the instruction rates on the card.
PRICES = {"fp4": 1.0, "fp4_sparse": 0.5, "fp8": 2.0, "int8": 2.0, "int8_sparse": 1.0, "bf16": 4.0, "fp32_op": 16.0}


def breakeven(prices: dict) -> dict:
    """Strassen-Winograd (7 products, 15 additions) on an exact m x k x n region, sub-products at price r per MAC.

    Cost per honest FP4 MAC after L levels: (7/8)^L * r, plus the output post-adds 7*(mn/4) per level at p each (per
    exact region of depth k_e: 7p/(4 k_e) per level) and the per-call A-side adds (q/n); B-side adds are per epoch.  The
    operand sums of L levels are sums of 2^L E2M1 values: they fit int8 while 12 * 2^L <= 127 (L <= 3), E4M3 for L = 1
    (every sum of two E2M1 values is an E4M3 value), BF16 while 12 * 2^(L+1) <= 256 in halves (L <= 3; wider sums need
    FP32).  No format holding a sum of two E2M1 values runs at the FP4 rate.
    """
    p = prices["fp32_op"]
    rows = []
    for fmt, r, Lmax in (("fp8 (E4M3 via kind::f8f6f4)", prices["fp8"], 1), ("int8 IMMA", prices["int8"], 3),
                         ("int8 IMMA 2:4 sparse (needs sparse sums)", prices["int8_sparse"], 3), ("bf16", prices["bf16"], 3),
                         ("hypothetical: sums at the FP4 rate", prices["fp4"], 6)):
        for L in range(1, Lmax + 1):
            mult = (7 / 8) ** L * r
            # the depth of an exact region at which the post-adds are paid back (n large): (1 - mult) = 7 p L / (4 k_e)
            ke = (7 * p * L / 4) / (1 - mult) if mult < 1 else None
            rows.append({"sub_products": fmt, "levels": L, "mac_cost_ratio": round(mult, 4),
                         "pays": mult < 1, "exact_depth_to_break_even": None if ke is None else round(ke, 1),
                         "exact_k64_steps_to_break_even": None if ke is None else round(ke / 64, 2)})
    return {"prices_fp4_mac_units": prices, "label": "Derived from Public rates (placeholders until GPU 4's microbenchmarks)",
            "strassen": rows,
            "sparse_2of4": {"cost_ratio_where_A_fragment_is_2of4": prices["fp4_sparse"] / prices["fp4"],
                            "note": "pays on every step where the whole 16x64 A fragment (or B's, with the product transposed) is 2:4; "
                                    "reproduces the words when the chain is exact there, or when mma.sp's align-add equals the dense one"},
            "int8_exact_route": {"cost_ratio": prices["int8"] / prices["fp4"] + prices["fp32_op"] / 16,
                                 "note": "IMMA k16 group dots (2 units/MAC) plus one scale multiply-add per 16 MACs per output (p/16)"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "synth", "real", "basesplit", "breakeven"])
    ap.add_argument("--atom", action="append", default=None)
    ap.add_argument("--frac-bits", type=int, default=None)
    ap.add_argument("--acc-window", type=int, default=None)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--families", default="all")
    ap.add_argument("--noise", default="none,row,block")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--acts", default=None)
    ap.add_argument("--max-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.atom = args.atom or ["blackwell_sm120_nvf4_m16n8k64", "blackwell_sm120_mxf4_m16n8k64"]
    t0 = time.time()
    if args.cmd == "selftest":
        rep = selftest()
        print(json.dumps(rep))
        ok = all(v["step_mismatches"] == 0 and v["chain_mismatches"] == 0 for v in rep.values())
        if args.out:
            Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
            run_result.write(Path(args.out), ok, "fp4-census-selftest", json.dumps(rep), [])
        sys.exit(0 if ok else 1)
    if args.cmd == "breakeven":
        doc = breakeven(PRICES)
        print(json.dumps(doc, indent=1))
        if args.out:
            Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
        return
    rep = selftest(cases=500)
    if not all(v["step_mismatches"] == 0 and v["chain_mismatches"] == 0 for v in rep.values()):
        print(json.dumps({"selftest": rep}))
        sys.exit("the vectorised atom disagrees with verity.ml.tc: no census")
    doc = {"synth": cmd_synth, "real": cmd_real, "basesplit": cmd_basesplit}[args.cmd](args)
    doc.update({"tool": "pouw_fp4_census", "cmd": args.cmd, "atoms": args.atom, "selftest": rep, "seconds": round(time.time() - t0, 1)})
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        es = [r["words"]["exact_steps"] for r in doc["results"] if "words" in r]
        if es:
            run_result.write(out, True, f"fp4-census-{args.cmd}", f"{len(doc['results'])} cells; exact steps {min(es):.4f}..{max(es):.4f}",
                             [("cells", len(doc["results"]), "count"), ("exact_steps_max", max(es), "fraction")])
        else:
            cc = [r["cheater_A_cost_per_dense_mac"] for r in doc["results"] if "cheater_A_cost_per_dense_mac" in r]
            run_result.write(out, bool(cc) and min(cc) >= 0.999, f"fp4-census-{args.cmd}",
                             f"{len(doc['results'])} cells; cheater A cost {min(cc):.3f}..{max(cc):.3f} (>=1 => no saving)" if cc else "no cells",
                             [("cells", len(doc["results"]), "count"), ("cheater_A_cost_min", min(cc) if cc else 0.0, "ratio")])


if __name__ == "__main__":
    main()
