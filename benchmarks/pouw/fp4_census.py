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
    python benchmarks/pouw/fp4_census.py basesplit --noise row --delta 0.25 [--acts acts_dir] --out split.json
    python benchmarks/pouw/fp4_census.py merge --noise none,row --delta 0.25 [--acts acts_dir] --out merge.json
    python benchmarks/pouw/fp4_census.py debit --atom blackwell_sm120_nvf4_m16n8k64 --noise row --delta 0.25 [--acts acts_dir] --out debit.json
    python benchmarks/pouw/fp4_census.py breakeven
"""
from __future__ import annotations

import argparse
import copy
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
        # below the step's grid whatever the codes: the largest group dot (16 x 144 quarters) times the scales < 2^lsb
        subgrid = (ma != 0) & (sh > 0) & ((sh >= 40) | (2304 * ma < (np.int64(1) << np.clip(sh, 0, 40))))
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
                     "overflow": overflow, "subgrid": subgrid, "exact_sum": (np.ldexp(mant.astype(np.float64), X)).sum(-1)}


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
            "scale-spread-30", "acc-freeze", "coherent", "rank-1", "spiky-row", "stride-row", "stride-row-tight",
            "stride-row-grid", "stride-row-grid-edge")


def _spiky(rows: int, k: int, rng) -> np.ndarray:
    """One entry per 16-block at +-U(0.8, 1), the rest 0.08 N(0, 1) (``t1_closures_attack.py``'s spiky rows)."""
    x = 0.08 * rng.standard_normal((rows, k))
    nb = k // 16
    pos = rng.integers(0, 16, (rows, nb)) + 16 * np.arange(nb)[None]
    x[np.arange(rows)[:, None], pos] = rng.choice([-1.0, 1.0], (rows, nb)) * rng.uniform(0.8, 1.0, (rows, nb))
    return x


def _stride(rows: int, k: int, rng, div: float) -> np.ndarray:
    """+-U(0.5, 1) except every 8th position, where rho is sampled, at N(0, 1) / div: rho ~ 1/div of the block amax."""
    x = rng.choice([-1.0, 1.0], (rows, k)) * rng.uniform(0.5, 1.0, (rows, k))
    x[:, ::8] = rng.standard_normal((rows, k // 8)) / div
    return x


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
    if name == "spiky-row":
        return _spiky(rows, k, rng), 0.02 * _spiky(rows, k, rng)
    if name in ("stride-row", "stride-row-tight"):          # the proposer's rho = amax/10 sits on the 10 rho screen; /9 inside it
        div = 10.0 if name == "stride-row" else 9.0
        return _stride(rows, k, rng, div), 0.02 * _stride(rows, k, rng, div)
    if name in ("stride-row-grid", "stride-row-grid-edge"):
        # the top two E2M1 codes of amax 1, and rho = 1/10 exactly (the 10 rho screen's edge) or 1/9.45 (1/9.44 in BF16,
        # inside the sqrt(90) screen)
        g = lambda: rng.choice([-1.0, 1.0], (rows, k)) * rng.choice([2 / 3, 1.0], (rows, k))  # noqa: E731
        A, B = g(), g()
        r = 0.1 if name == "stride-row-grid" else 1 / 9.45
        A[:, ::8], B[:, ::8] = (r * rng.choice([-1.0, 1.0], (rows, k // 8)) for _ in range(2))
        return A, 2.0 ** -6 * B                             # a power of two keeps B's BF16 rho at A's, on the screen's side
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


# ---- the replayed debit (theory-pearl-c4-domain.md §2: identity atoms, item 2; sub-grid blocks, item 5) ---------------

SUBGRID_UNITS, IDENTITY_UNITS = 16, 64            # MACs charged per sub-grid (word, group) and per identity atom-word


def _step_inputs(atom: Atom, qa, qb, ia, ib, t: int):
    (na, pa), (nb, pb) = qa, qb
    k, ns = atom.k, atom.k // atom.block
    return (na[ia, t * k:(t + 1) * k], nb[ib, t * k:(t + 1) * k],
            (pa[0][ia, t * ns:(t + 1) * ns], pa[1][ia, t * ns:(t + 1) * ns]),
            (pb[0][ib, t * ns:(t + 1) * ns], pb[1][ib, t * ns:(t + 1) * ns]))


def debit_chain(atom: Atom, qa, qb, ia, ib, acc0: np.ndarray, keep: np.ndarray) -> dict:
    """One chain per word (A row ia, B row ib) from acc0: per word the sub-grid groups, the identity steps (the step's
    output equals its input, so skipping it alone leaves the word unchanged) and the RMS exact atom sum; the whole
    trajectory for the words ``keep`` indexes."""
    T = qa[0].shape[1] // atom.k
    acc = acc0.astype(np.float64).copy()
    sub, ident, lost = (np.zeros(len(ia), np.int64) for _ in range(3))
    ss = np.zeros(len(ia))
    traj = np.zeros((len(keep), T + 1))
    traj[:, 0] = acc[keep]
    for t in range(T):
        d, fl = atom.step(acc, *_step_inputs(atom, qa, qb, ia, ib, t))
        sub += fl["subgrid"].sum(-1)
        ident += d == acc
        lost += fl["lost"]
        ss += fl["exact_sum"] ** 2
        acc = d
        traj[:, t + 1] = acc[keep]
    return {"subgrid": sub, "ident": ident, "lost": lost, "rms": np.sqrt(ss / T), "traj": traj, "steps": T, "final": acc}


def merged(atom: Atom, factor: int = 2) -> Atom:
    """``factor`` atoms merged into one align-add over all their groups: the 2:4-sparse m16n8k128 route on a 2:4 span as
    ``fp4_emulation.py merged_k128`` models it (a hypothesis until GPU 4's capture)."""
    m = copy.copy(atom)
    m.name, m.k = f"{atom.name}[merged x{factor}]", atom.k * factor
    m.groups = m.k // m.group
    m._gidx = (np.arange(m.groups) * m.group) // m.block
    return m


def hot_start(rms: np.ndarray, ia: np.ndarray, rows: int, hot_bits: int, rule: str = "row") -> np.ndarray:
    """T1's H = 1.5 * 2^E_H per word, its grid 2^(E_H - 23) at 2^-hot_bits of the RMS atom sum: the row's largest over the
    given words, one H per A row (``row``, ``fp4_emulation.py --hot-rule row``), or the word's own (``word``, the ideal
    that GPU 5's product-free per-word estimate aims at)."""
    if rule == "row":
        rms_row = np.zeros(rows)
        np.maximum.at(rms_row, ia, rms)
        rms = rms_row[ia]
    eh = np.floor(np.log2(np.maximum(rms, 2.0 ** -100))).astype(np.int64) + 23 - hot_bits
    return np.ldexp(1.5, eh)


def lone_skip_identities(atom: Atom, qa, qb, ia, ib, traj: np.ndarray) -> np.ndarray:
    """Per word, the atoms whose lone skip leaves the final word unchanged (the debit's exact test): the chain without
    step s, replayed from its state before s until it rejoins the original trajectory or ends."""
    W, T = traj.shape[0], traj.shape[1] - 1
    Z = traj[:, :T].copy()                            # Z[w, s]: the state after skipping step s, at step s
    done = Z == traj[:, 1:]
    for u in range(1, T):
        w, s = np.nonzero(~done[:, :u])
        if not len(w):
            continue
        d, _ = atom.step(Z[w, s], *_step_inputs(atom, qa, qb, ia[w], ib[w], u))
        Z[w, s] = d
        done[w, s] = d == traj[w, u + 1]
    return done.sum(1)


F_S = 98.0                                        # A's credited forming per element (theory §2: 64 + 1 + 32.8)
#: Qwen2.5-7B's output width n per linear: the forming term charges f_s per dead A element times |cols| / n.
QWEN25_7B_N = {"q_proj": 3584, "k_proj": 512, "v_proj": 512, "o_proj": 3584, "gate_proj": 18944, "up_proj": 18944,
               "down_proj": 3584}


def span_2of4(live: np.ndarray, rows: int = 16, span: int = 128) -> dict:
    """Both 2:4 predicates on a boolean map (a formed operand's nonzero codes, or a correction's salt-dependent entries):
    D-24's per 4-group (<= 2 live of 4 consecutive k) and the instruction's per 8 of k (<= 2 live element pairs), each
    per group and per aligned span (``rows`` rows x ``span`` of k, the A operand of one mma.sp m16n8k128)."""
    r, K = live.shape
    nr, nk = r - r % rows, K - K % span
    z = live[:nr, :nk]
    g4, p8 = z.reshape(nr, nk // 4, 4).sum(-1), z.reshape(nr, nk // 8, 4, 2).any(-1).sum(-1)
    spans = lambda ok, per: ok.reshape(nr // rows, rows, nk // span, span // per).all((1, 3))
    return {"g4": g4 <= 2, "g4_dead": g4 == 0, "p8": p8 <= 2, "p8_dead": p8 == 0,
            "span_d24": spans(g4 <= 2, 4), "span_pairs": spans(p8 <= 2, 8), "span_dead": spans(g4 == 0, 4)}


TWO_FOUR_UNITS = ("g4", "p8", "span_d24", "span_pairs", "span_either")


def two_four_term(sp: dict, rows: int = 16) -> dict[str, np.ndarray]:
    """§2 item 1 per 16-row block, as a share of its MACs: 0.5 per MAC where the correction fits one 2:4 pass, 1.0 where it
    is all salt-dead, per unit: the 4-group (theory as written), the 8-of-k pair group, or the whole span (D-24's
    predicate, the instruction's, or either)."""
    nb = sp["span_d24"].shape[0]
    per_row = lambda ok, dead: (0.5 * ok + 0.5 * dead).mean(1)[: nb * rows].reshape(nb, rows).mean(1)
    per_span = lambda ok: (0.5 * ok + 0.5 * sp["span_dead"]).mean(1)
    return {"g4": per_row(sp["g4"], sp["g4_dead"]), "p8": per_row(sp["p8"], sp["p8_dead"]),
            "span_d24": per_span(sp["span_d24"]), "span_pairs": per_span(sp["span_pairs"]),
            "span_either": per_span(sp["span_d24"] | sp["span_pairs"])}


def forming_scale(X: np.ndarray, fmt: str, delta: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row, the Pearl-C4 reference's (rho, alpha, a_E byte) in float64: rho the RMS over every 8th position,
    alpha = 2688 / (s + 4 delta rho) (a power of two for MXFP4), a_E = UE4M3(delta rho alpha 16 / sqrt(2346.125))."""
    rho = np.sqrt((X[:, ::8] ** 2).mean(1))
    alpha = 2688.0 / np.maximum(np.abs(X).max(1) + 4 * delta * rho, 2.0 ** -32)
    if fmt == "mxf4":
        alpha = np.exp2(np.floor(np.log2(alpha)))
    return rho, alpha, e4m3_codes_unsigned(delta * rho * alpha * 16 / np.sqrt(2346.125))


def witness_salt_dead(X: np.ndarray, fmt: str, delta: float = 0.25) -> np.ndarray:
    """Per element, the Pearl-C4 reference's ``salt_dead`` (``pearl_c4.py`` at 524d0c2b), in float64: an element is
    salt-live when two of the noise support's draws give it two codes, the draws being the noise atom at +-72 a_E (E = +6
    on the row, F_t = +-6) and 0, with every other element of its block unmoved.  The row is at the reference's forming
    scale (``forming_scale``), block scales by its rule (UE4M3 of amax/6 with floor 0x01; UE8M0 rounded up)."""
    blk = 16 if fmt == "nvf4" else 32
    rows, k = X.shape
    _, alpha, ae = forming_scale(X, fmt, delta)
    m, e = scale_pairs("ue4m3", ae)
    amp = 72.0 * np.ldexp(m.astype(np.float64), e)[:, None, None]
    c = (alpha[:, None] * X).reshape(rows, k // blk, blk)
    mags = np.abs(c)
    top = mags.max(-1, keepdims=True)
    rest = np.where(mags == top, np.sort(mags, -1)[..., -2:-1], top)

    def code(v):
        amax = np.maximum(np.abs(v), rest) / E2M1_MAX
        if fmt == "nvf4":
            sm, se = scale_pairs("ue4m3", np.maximum(e4m3_codes_unsigned(amax), 1))
            s = np.ldexp(sm.astype(np.float64), se)
        else:
            s = np.exp2(np.ceil(np.log2(np.maximum(amax, 2.0 ** -127))))
        q = e2m1_codes(v / s)
        return np.where(q & 7, q, 0)

    mid = code(c)
    return ((code(c - amp) == mid) & (code(c + amp) == mid)).reshape(rows, k)


DEAD_RHO = float(np.sqrt(90.0))                   # the screen (clean amax_b)^2 > 90 rho^2 (DEAD_RHO_SQ, #534); 10 at 5f0a1529


def dss_dead(X: np.ndarray, fmt: str, dead_rho: float = DEAD_RHO) -> np.ndarray:
    """D-SS's salt-free screen per (row, block): clean amax_b^2 > dead_rho^2 * rho^2, rho^2 the mean square over every 8th
    position (sqrt(90)^2 is exactly 90 in float64)."""
    blk = 16 if fmt == "nvf4" else 32
    rho_sq = (X[:, ::8] ** 2).mean(1)
    return np.abs(X).reshape(X.shape[0], -1, blk).max(-1) ** 2 > dead_rho ** 2 * rho_sq[:, None]


def rows_admitted(X: np.ndarray, fmt: str, delta: float = 0.25, dead_rho: float = DEAD_RHO) -> dict[str, np.ndarray]:
    """The reference's ``row_passes`` per row: D-NF (a_E a normal UE4M3, byte >= 0x08) and D-SS (at most one dead block
    in 64), each and both."""
    dnf = forming_scale(X, fmt, delta)[2] >= 0x08
    d = dss_dead(X, fmt, dead_rho)
    dss = 64 * d.sum(1) <= d.shape[1]
    return {"dnf": dnf, "dss": dss, "both": dnf & dss}


def span_merged_equal(atom: Atom, qa, qb, ia, ib, traj: np.ndarray) -> np.ndarray:
    """Per (word, span of two atoms): the merged k128 align-add from the chain's state before the span gives the chain's
    state after it (GPU 4's span-word test)."""
    m = merged(atom)
    S = (traj.shape[1] - 1) // 2
    eq = np.zeros((traj.shape[0], S), bool)
    for s in range(S):
        d, _ = m.step(traj[:, 2 * s], *_step_inputs(m, qa, qb, ia, ib, s))
        eq[:, s] = d == traj[:, 2 * s + 2]
    return eq


def _tile_stats(x: np.ndarray, rho: float) -> dict:
    return {"mean": float(x.mean()), "max": float(x.max()), "over_rho": float((x > rho).mean())}


def debit_census(atom: Atom, fmt: str, A: np.ndarray, B: np.ndarray, noise: str, delta: float, rng, rows: int,
                 tile: int = 16, skip_tiles: int = 2, hot_bits: int = 14, rho: float = 1 / 400,
                 hot_rule: str = "row", n: int | None = None, dead_rho: float = DEAD_RHO, screen: str = "debit") -> dict:
    """The per-tile debit of §2 on every word of ``rows`` x ``rows`` (A rows x B rows), as a share of the tile's chain MACs
    against the cap rho, for v1's chains from +0 and v2's (T1) from H (``hot_start``).

    Replayed per chain version: identity atoms (item 2) and sub-grid groups (item 5).  On every tile identity atoms are
    counted as identity steps (a lower bound on the debit's lone-skip test); on ``skip_tiles`` whole tiles (the worst by
    that count, and random ones) by the lone-skip test itself.  Per operand, the same for v1 and v2, as the Pearl-C4
    reference's ``tile_debit`` charges them: the 2:4 term (item 1) on its salt-dead codes (``witness_salt_dead``) at each
    unit of ``two_four_term`` (its own is ``g4``); dead x dead pairs of D-SS's dead blocks (item 3, one block's MACs per
    pair-word); the forming of salt-dead or unneeded A elements (item 4, f_s times |cols| / n, n the output width,
    default k); and D-24's row windows on the formed codes (0.5 per MAC per row and aligned 128 of k), with the
    instruction's pair rule beside it.  Item 1 is also read on one salt's changes (``code``; ``value``: code times block
    scale, as the MMA sees it), which TT_OUT-FP4 carries rather than debits.  Both 2:4 predicates are counted on the
    formed 16-row spans of A' and B', and per version, whether the merged k128 step reproduces the chain's words over
    each span, on all spans and on 2:4 spans.  ``tile_debit_total`` adds items 2 and 5 (identity steps), 3, 4 and the
    D-24 windows to item 1 at each level and unit; the reference's total is ``witness``/``g4``.  Dead blocks are D-SS's at
    ``dead_rho``. With ``screen="cap"`` a tile is admitted when every row of both its sides passes the reference's row
    rule at 5f0a1529 (``rows_admitted``: D-NF and at most one dead block in 64). With ``screen="debit"`` (the 10:35Z rule,
    item 4 on the witness since 11:32Z) the cap is gone and the screen is charged in item 3 alone: a tile is admitted when
    every row passes D-NF, and ``tiles_admitted_under_rho`` is the share whose total stays within rho (the rest are
    rejected).  ``tiles`` keeps each tile's admission and item 4, and each version's ``tile_total_ref`` the reference's
    total per tile, so another cap, or another n (item 4 scales as 1/n, no other term reads n), is exact offline.
    Each side's dead share and row admission at 12 rho is reported beside."""
    ya, yb = noisy(A, fmt, noise, delta, rng), noisy(B, fmt, noise, delta, rng)
    ca, sa, _ = quantize(ya, fmt)
    cb, sb, _ = quantize(yb, fmt)
    scale = atom.scale
    qa, qb = (numerators(ca), scale_pairs(scale, sa)), (numerators(cb), scale_pairs(scale, sb))
    ra, rb = min(rows, A.shape[0]) // tile * tile, min(rows, B.shape[0]) // tile * tile
    ia, ib = (x.ravel() for x in np.meshgrid(np.arange(ra), np.arange(rb), indexing="ij"))
    K, nt = ca.shape[1], (ra // tile) * (rb // tile)
    n = n or K
    blk = 16 if fmt == "nvf4" else 32
    out = {"n_words": int(len(ia)), "tile": [tile, tile], "tiles": int(nt), "k": int(K), "n": int(n), "rho": rho,
           "hot_bits": hot_bits, "hot_rule": hot_rule, "skip_tiles": skip_tiles, "dead_rho": dead_rho, "screen": screen,
           "operands": {}}
    formed, terms, dead, sdead, windows, admit = {}, {}, {}, {}, {}, {}
    share = lambda sp, live: {"change_density": float(live.mean()), "g4_ge2_dead": float(sp["g4"].mean()),
                              "g4_all_dead": float(sp["g4_dead"].mean()), "p8_le2_live_pairs": float(sp["p8"].mean()),
                              "span_d24": float(sp["span_d24"].mean()), "span_pairs": float(sp["span_pairs"].mean())}
    for side, c, s, X, rr in (("A", ca, sa, A, ra), ("B", cb, sb, B, rb)):
        c0, s0, _ = quantize(X, fmt)
        c, s, c0, s0 = c[:rr], s[:rr], c0[:rr], s0[:rr]
        f = formed[side] = span_2of4((c & 7) != 0, tile)
        windows[side] = {p: f[p].reshape(rr, -1, 128 // w).all(-1).sum(1) for p, w in (("g4", 4), ("p8", 8))}
        dead[side] = dss_dead(X[:rr], fmt, dead_rho)
        adm, d12 = rows_admitted(X[:rr], fmt, delta, dead_rho), dss_dead(X[:rr], fmt, 12.0)
        admit[side] = adm["both"] if screen == "cap" else adm["dnf"]
        sdead[side] = witness_salt_dead(X[:rr], fmt, delta)
        sp = span_2of4(~sdead[side], tile)
        terms[side] = {"witness": two_four_term(sp, tile)}
        o = {"spans": int(f["span_d24"].size), "formed_span_d24": float(f["span_d24"].mean()),
             "formed_span_pairs": float(f["span_pairs"].mean()),
             "formed_row_windows_d24": float(windows[side]["g4"].sum() / f["span_d24"].size / tile),
             "formed_row_windows_pairs": float(windows[side]["p8"].sum() / f["span_d24"].size / tile),
             "dss_dead_blocks": float(dead[side].mean()), "dss_rows_admitted": float(adm["dss"].mean()),
             "dnf_rows_admitted": float(adm["dnf"].mean()), "rows_admitted": float(adm["both"].mean()),
             "dss_dead_blocks_12rho": float(d12.mean()),
             "dss_rows_admitted_12rho": float((64 * d12.sum(1) <= d12.shape[1]).mean()),
             "witness": share(sp, ~sdead[side])}
        if noise != "none":
            (N1, E1), (N0, E0) = block_ints(c, s, fmt), block_ints(c0, s0, fmt)
            val = lambda N, E: np.ldexp(N.astype(np.float64), E[..., None]).reshape(rr, -1)
            for lvl, live in (("code", c != c0), ("value", val(N1, E1) != val(N0, E0))):
                sp = span_2of4(live, tile)
                o[lvl] = share(sp, live)
                terms[side][lvl] = two_four_term(sp, tile)
        out["operands"][side] = o
    nta, ntb = ra // tile, rb // tile
    admitted = (admit["A"].reshape(-1, tile).all(1)[:, None] & admit["B"].reshape(-1, tile).all(1)[None, :]).ravel()
    dA, dB = (dead[x].reshape(-1, tile, dead[x].shape[1]).sum(1).astype(np.float64) for x in "AB")
    item3 = (dA @ dB.T) * blk / (tile * tile * K)
    sd = sdead["A"].reshape(nta, tile, K).sum(1).astype(np.float64)                 # salt-dead A elements per position
    unneeded = ((cb[:rb] & 7) == 0).reshape(ntb, tile, K).all(1).astype(np.float64)   # no B' row of the tile needs t
    item4 = F_S * (sd.sum(1)[:, None] + (tile - sd) @ unneeded.T) * tile / n / (tile * tile * K)
    win = {p: 4.0 * (windows["A"][p].reshape(nta, tile).sum(1)[:, None]
                     + windows["B"][p].reshape(ntb, tile).sum(1)[None, :]) / K for p in ("g4", "p8")}
    levels = ("witness", "code", "value") if noise != "none" else ("witness",)
    out["item1"] = {lvl: {u: _tile_stats((terms["A"][lvl][u][:, None] + terms["B"][lvl][u][None, :]).ravel(), rho)
                          for u in TWO_FOUR_UNITS} for lvl in levels}
    out["item3"], out["item4"] = _tile_stats(item3.ravel(), rho), _tile_stats(item4.ravel(), rho)
    out["windows_d24"], out["windows_pairs"] = _tile_stats(win["g4"].ravel(), rho), _tile_stats(win["p8"].ravel(), rho)
    out["tiles_admitted"] = float(admitted.mean())
    out["tiles"] = {"admitted": admitted.tolist(), "item4": item4.ravel().tolist()}
    operand_units = {lvl: {u: (terms["A"][lvl][u][:, None] + terms["B"][lvl][u][None, :] + item3 + item4
                               + win["g4"]).ravel() for u in TWO_FOUR_UNITS} for lvl in levels}
    cold = None
    for name in ("v1", "v2"):
        acc0 = np.zeros(len(ia)) if name == "v1" else hot_start(cold["rms"], ia, ra, hot_bits, hot_rule)
        r = debit_chain(atom, qa, qb, ia, ib, acc0, np.arange(len(ia)))
        cold = r if name == "v1" else cold
        k128 = debit_chain(merged(atom), qa, qb, ia, ib, acc0, ia[:0])["final"] if K % (2 * atom.k) == 0 else None
        units = SUBGRID_UNITS * r["subgrid"] + IDENTITY_UNITS * r["ident"]
        per_tile = units.reshape(ra // tile, tile, rb // tile, tile).sum((1, 3)).ravel() / (tile * tile * K)
        worst = int(per_tile.argmax())
        picked = [worst, *map(int, rng.choice(np.delete(np.arange(nt), worst), min(skip_tiles, nt) - 1, replace=False))]
        exact = []
        for t in picked:
            ti, tj = divmod(t, rb // tile)
            w = ((ti * tile + np.arange(tile))[:, None] * rb + tj * tile + np.arange(tile)).ravel()
            skip = lone_skip_identities(atom, qa, qb, ia[w], ib[w], r["traj"][w])
            exact.append({"tile": [ti, tj], "step_identity_share": float(per_tile[t]),
                          "lone_skip_share": float((SUBGRID_UNITS * r["subgrid"][w] + IDENTITY_UNITS * skip).sum()
                                                   / (tile * tile * K)),
                          "lone_skip_identities": float(skip.sum() / (len(w) * r["steps"]))})
        out[name] = {"subgrid_groups": float(r["subgrid"].sum() / (len(ia) * r["steps"] * atom.groups)),
                     "identity_steps": float(r["ident"].sum() / (len(ia) * r["steps"])),
                     "steps_losing_bits": float(r["lost"].sum() / (len(ia) * r["steps"])),
                     "words_exact": float((r["lost"] == 0).mean()),
                     "tile_debit_share": {"mean": float(per_tile.mean()), "max": float(per_tile.max()),
                                          "over_rho": float((per_tile > rho).mean())},
                     "tile_debit_exact": exact, "tile_debit_exact_max": max(e["lone_skip_share"] for e in exact),
                     "merged_k128_words_equal": None if k128 is None else float((k128 == r["final"]).mean())}
        if k128 is not None:
            eq = span_merged_equal(atom, qa, qb, ia, ib, r["traj"])
            out[name]["k128_span_words_equal"] = float(eq.mean())
            out[name]["k128_span_words_equal_on_2of4"] = {}
            for pred in ("span_d24", "span_pairs"):
                sel = formed["A"][pred][ia // tile] | formed["B"][pred][ib // tile]
                out[name]["k128_span_words_equal_on_2of4"][pred] = {
                    "span_steps": int(sel.sum()), "equal": float(eq[sel].mean()) if sel.any() else None}
        if operand_units:
            tot = {lvl: {u: per_tile + x for u, x in d.items()} for lvl, d in operand_units.items()}
            out[name]["tile_debit_total"] = {lvl: {u: {
                "max": float(x.max()), "over_rho": float((x > rho).mean()),
                "max_admitted": float(x[admitted].max()) if admitted.any() else None,
                "over_rho_admitted": float((x[admitted] > rho).mean()) if admitted.any() else None,
                "tiles_admitted_under_rho": float((admitted & (x <= rho)).mean())}
                for u, x in d.items()} for lvl, d in tot.items()}
            out[name]["tile_total_ref"] = tot["witness"]["g4"].tolist()
    return out


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


# ---- merge rate (the FP4-tile model's epsilon_R) ---------------------------------------------------------------------

#: Two-block pre-add patterns: X-blocks (row, k-block) paired at the finest level (adjacent) and at one Strassen level
#: (halves of the rows or of the k-blocks, and the diagonal A11 + A22).
MERGE_PATTERNS = ("adjacent-k", "adjacent-row", "strassen-k", "strassen-row", "strassen-diag")


def block_ints(codes: np.ndarray, sc: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray]:
    """Blocks as exact integers: (rows, nb, blk) N and (rows, nb) E with value = N * 2^E (in units of the global g)."""
    rows, k = codes.shape
    blk = 16 if fmt == "nvf4" else 32
    m, e = scale_pairs("ue4m3" if fmt == "nvf4" else "ue8m0", sc)
    return numerators(codes).reshape(rows, k // blk, blk) * m[..., None], e - 1   # value = (n/2) * m * 2^e


def fp4_representable(N: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray]:
    """Per block (P, blk) of values N * 2^E (E drops out: a power of two is free): (on the E2M1 grid under one scale of the format, under any scale > 0).

    The block is q_j/2 * s for codes q_j in E2M1's halves {0,1,2,3,4,6,8,12}; the largest |N_j| = M fixes q_max, hence
    s = 2 M 2^E / q_max, so each of the 7 choices of q_max is one exact integer test.  "One scale of the format" is a
    UE4M3 value up to a power of two, i.e. s's odd part is at most 15 (nvf4: a prover forming a merged operand sets its
    own FP32 global scale, so a power-of-two shift is free), or a power of two (mxf4's UE8M0).  An all-zero block passes.
    """
    M = np.abs(N).max(-1)
    zero = M == 0
    Ms = np.where(zero, 1, M)
    strict, anyscale = zero.copy(), zero.copy()
    odd_max = 15 if fmt == "nvf4" else 1
    for qmax in (1, 2, 3, 4, 6, 8, 12):
        num = N * qmax
        ok = ~zero & (num % Ms[:, None] == 0).all(-1) & np.isin(np.abs(num // Ms[:, None]), E2M1_NUM).all(-1)
        anyscale |= ok
        three = 3 if qmax % 3 == 0 else 1
        t = np.where(Ms % three == 0, 2 * Ms // three, 0)             # s = t * 2^(E - log2(qmax / three)); 0: s not dyadic
        odd = t // np.where(t > 0, t & -t, 1)
        strict |= ok & (t > 0) & (odd <= odd_max)
    return strict, anyscale


def merge_rate(fmt: str, A: np.ndarray, noise: str, delta: float, rng, partner_blocks: int = 256) -> dict:
    """epsilon_R: how often a sum or difference of two salted input blocks is itself one FP4 value (FP4-tile model §2.3).

    Per pattern and sign, over the pairs whose two blocks are both nonzero: ``eps_strict`` = the merged block is
    FP4-representable under one scale of the format; ``eps_any_scale`` = on the E2M1 grid under any common factor (an
    upper bound);     ``cancel`` = the blocks cancel exactly (a duplicate up to sign); ``dup`` = v1 = +-v2 (either sign).
    """
    codes, sc, g = quantize(noisy(A, fmt, noise, delta, rng), fmt)
    return merge_rate_codes(codes, sc, fmt, partner_blocks)


def merge_rate_codes(codes: np.ndarray, sc: np.ndarray, fmt: str, partner_blocks: int = 256) -> dict:
    """``merge_rate`` on operands already formed: E2M1 codes (rows, k) and their block scale bytes."""
    N, E = block_ints(codes, sc, fmt)
    rows, nb, blk = N.shape
    live = np.abs(N).max(-1) > 0
    out = {"zero_codes": float(((codes & 7) == 0).mean()), "zero_blocks": float((~live).mean()), "blocks": int(rows * nb),
           "patterns": {}}
    h, hk = rows // 2, nb // 2
    idx = {"adjacent-k": ((slice(None), slice(0, 2 * hk, 2)), (slice(None), slice(1, 2 * hk, 2))),
           "adjacent-row": ((slice(0, 2 * h, 2), slice(None)), (slice(1, 2 * h, 2), slice(None))),
           "strassen-k": ((slice(None), slice(0, hk)), (slice(None), slice(hk, 2 * hk))),
           "strassen-row": ((slice(0, h), slice(None)), (slice(h, 2 * h), slice(None))),
           "strassen-diag": ((slice(0, h), slice(0, hk)), (slice(h, 2 * h), slice(hk, 2 * hk)))}
    worst = 0.0
    for pat in MERGE_PATTERNS:
        a, b = idx[pat]
        N1, E1, N2, E2 = N[a].reshape(-1, blk), E[a].reshape(-1), N[b].reshape(-1, blk), E[b].reshape(-1)
        both = live[a].reshape(-1) & live[b].reshape(-1)
        lo = np.minimum(E1, E2)
        far = np.abs(E1 - E2) > 40                                      # the smaller block sits below 40 bits: never on a 2-bit grid
        s1, s2 = np.minimum(E1 - lo, 40), np.minimum(E2 - lo, 40)
        A1, A2 = N1 << s1[:, None], N2 << s2[:, None]
        dup = both & ~far & ((A1 == A2).all(-1) | (A1 == -A2).all(-1))
        row = {"pairs": int(both.sum())}
        for sign, S in (("+", A1 + A2), ("-", A1 - A2)):
            strict, anyscale = fp4_representable(S, fmt)
            nz = np.abs(S).max(-1) > 0
            n = max(1, int(both.sum()))
            r = {"eps_strict": float((both & ~far & nz & strict).sum() / n),
                 "eps_any_scale": float((both & ~far & nz & anyscale).sum() / n),
                 "cancel": float((both & ~nz).sum() / n)}
            worst = max(worst, r["eps_strict"])
            row[sign] = r
        row["dup"] = float(dup.sum() / max(1, int(both.sum())))
        out["patterns"][pat] = row
    out["eps_R_strict_max"] = worst
    out["eps_R_any_scale_max"] = max(out["patterns"][p][s]["eps_any_scale"] for p in MERGE_PATTERNS for s in "+-")
    out["partners"] = merge_partners(N, E, live, fmt, partner_blocks)
    return out


def _mergeable(N1, E1, N2, E2, fmt) -> np.ndarray:
    """Pairs (P, blk) whose sum or difference is one nonzero FP4 value under one scale of the format."""
    lo = np.minimum(E1, E2)
    far = np.abs(E1 - E2) > 40
    A1, A2 = N1 << np.minimum(E1 - lo, 40)[:, None], N2 << np.minimum(E2 - lo, 40)[:, None]
    ok = np.zeros(len(N1), bool)
    for S in (A1 + A2, A1 - A2):
        ok |= fp4_representable(S, fmt)[0] & (np.abs(S).max(-1) > 0)
    return ok & ~far


def merge_partners(N: np.ndarray, E: np.ndarray, live: np.ndarray, fmt: str, max_blocks: int = 256) -> dict:
    """The adaptive side of epsilon_R: over the first ``max_blocks`` k-blocks, the share of live blocks with at least one
    mergeable partner (sum or difference one FP4 value), searching every other row at the same k-block, and every other
    of those k-blocks in the same row.  A prover free to choose its pairs can merge at most this share of blocks."""
    rows, nb, _ = N.shape
    kb = min(nb, max_blocks)
    col = np.zeros((rows, kb), bool)
    i, j = np.triu_indices(rows, 1)
    for b in range(kb):
        ok = _mergeable(N[i, b], E[i, b], N[j, b], E[j, b], fmt) & live[i, b] & live[j, b]
        np.logical_or.at(col[:, b], i[ok], True)
        np.logical_or.at(col[:, b], j[ok], True)
    row = np.zeros((rows, kb), bool)
    p, q = np.triu_indices(kb, 1)
    for r in range(rows):
        ok = _mergeable(N[r, p], E[r, p], N[r, q], E[r, q], fmt) & live[r, p] & live[r, q]
        np.logical_or.at(row[r], p[ok], True)
        np.logical_or.at(row[r], q[ok], True)
    n_live = max(1, int(live[:, :kb].sum()))
    return {"same_kblock_any_row": float((col & live[:, :kb]).sum() / n_live),
            "same_row_any_kblock": float((row & live[:, :kb]).sum() / n_live),
            "rows": int(rows), "kblocks_searched": int(kb)}


def cmd_merge(args) -> dict:
    """epsilon_R per format, operand and noise: the FP4-tile model's merge rate, on the confirmed atom's formats."""
    rng = np.random.default_rng(args.seed)
    res = {"k": args.k, "pairs": args.pairs, "seed": args.seed, "noise": args.noise, "delta": args.delta,
           "acts": str(args.acts) if args.acts else None, "operand": args.operand, "patterns": list(MERGE_PATTERNS),
           "results": []}
    for name, (atom, fmt) in atoms_from(args).items():
        for label, A in operands(args, rng):
            for noise in args.noise.split(","):
                r = merge_rate(fmt, A, noise, args.delta, rng, args.partner_blocks)
                res["results"].append({"atom": atom.name, "format": fmt, **label, "noise": noise, **r})
                print(json.dumps({"format": fmt, **label, "noise": noise, "zero_codes": round(r["zero_codes"], 4),
                                  "zero_blocks": round(r["zero_blocks"], 4), "eps_R_strict_max": round(r["eps_R_strict_max"], 5),
                                  "eps_R_any_scale_max": round(r["eps_R_any_scale_max"], 5),
                                  "partner_col": round(r["partners"]["same_kblock_any_row"], 4),
                                  "partner_row": round(r["partners"]["same_row_any_kblock"], 4)}), flush=True)
    return res


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


def cmd_debit(args) -> dict:
    """The replayed debit per 16 x 16 tile (identity atoms, sub-grid groups) for v1 (+0) and v2 (T1's H) chains, on the
    families (A, B from ``family``) or on real (activation, weight) rows."""
    rng = np.random.default_rng(args.seed)
    res = {"k": args.k, "rows": args.pairs, "seed": args.seed, "noise": args.noise, "delta": args.delta,
           "acts": str(args.acts) if args.acts else None, "hot_bits": args.hot_bits, "results": []}
    if args.acts:
        acts = Path(args.acts)
        cells = []
        for f in [acts] if acts.is_file() else sorted(acts.glob("*.npz")):
            d = np.load(f)
            for lin in sorted({key.rsplit("/", 1)[0] for key in d.files}):
                X, W = d[f"{lin}/x"].astype(np.float64), d[f"{lin}/w"].astype(np.float64)
                if args.max_k:
                    X, W = X[:, : args.max_k], W[:, : args.max_k]
                cells.append(({"source": f.stem, "linear": lin, "k": int(X.shape[1]),
                               "n": args.n or QWEN25_7B_N.get(lin.rsplit(".", 1)[-1])}, X, W))
    else:
        fams = FAMILIES if args.families == "all" else tuple(args.families.split(","))
        cells = [({"family": fam, "n": args.n or args.k}, *family(fam, args.pairs, args.k, rng)) for fam in fams]
    for name, (atom, fmt) in atoms_from(args).items():
        for label, A, B in cells:
            for noise in args.noise.split(","):
                try:
                    r = debit_census(atom, fmt, A, B, noise, args.delta, rng, args.pairs, hot_bits=args.hot_bits,
                                     hot_rule=args.hot_rule, n=label["n"], dead_rho=args.dead_rho, screen=args.screen)
                except OverflowError as e:
                    r = {"outside_model": str(e)}
                res["results"].append({"atom": atom.name, "format": fmt, **label, "noise": noise, **r})
                tot = lambda v, lvl, u: r[v]["tile_debit_total"][lvl][u]["max"] if "tile_debit_total" in r.get(v, {}) else None
                print(json.dumps({"format": fmt, **label, "noise": noise, **{
                    f"{v}_{q}": round(r[v][key], 6) for v in ("v1", "v2") if v in r
                    for q, key in (("subgrid", "subgrid_groups"), ("ident", "identity_steps"), ("words_exact", "words_exact"),
                                   ("tile_max", "tile_debit_exact_max"), ("k128_span_equal", "k128_span_words_equal"))
                    if r[v].get(key) is not None},
                    **{f"{v}_k128_equal": r[v]["merged_k128_words_equal"] for v in ("v1", "v2") if v in r},
                    **{f"formed_{p}_{s}": r["operands"][s][f"formed_span_{p}"] for p in ("d24", "pairs") for s in "AB"
                       if "operands" in r},
                    **({"item1_witness_g4_max": round(r["item1"]["witness"]["g4"]["max"], 6),
                        "item1_witness_span_max": round(r["item1"]["witness"]["span_either"]["max"], 6),
                        **({"item1_code_g4_max": round(r["item1"]["code"]["g4"]["max"], 4)} if "code" in r["item1"] else {}),
                        "item3_max": round(r["item3"]["max"], 6), "item4_max": round(r["item4"]["max"], 6),
                        "windows_d24_max": round(r["windows_d24"]["max"], 6), "tiles_admitted": r["tiles_admitted"],
                        **{f"rows_admitted_{s}": round(r["operands"][s]["rows_admitted"], 4) for s in "AB"},
                        **{f"{v}_total_ref_max": round(tot(v, "witness", "g4"), 6) for v in ("v1", "v2")},
                        **{f"{v}_tiles_admitted_under_rho": r[v]["tile_debit_total"]["witness"]["g4"]["tiles_admitted_under_rho"]
                           for v in ("v1", "v2") if "tile_debit_total" in r.get(v, {})}}
                       if "item1" in r else {})}),
                    flush=True)
    return res


def operands(args, rng):
    """One side's operands (--operand a: activations, b: weights): the synthetic families, or with --acts the real rows
    fp4_quality.py --dump-acts wrote."""
    side = getattr(args, "operand", "a")
    if not args.acts:
        for fam in FAMILIES if args.families == "all" else tuple(args.families.split(",")):
            yield {"family": fam}, family(fam, max(args.pairs, 64), args.k, rng)[0 if side == "a" else 1]
        return
    acts = Path(args.acts)
    for f in [acts] if acts.is_file() else sorted(acts.glob("*.npz")):
        d = np.load(f)
        for lin in sorted({key.rsplit("/", 1)[0] for key in d.files}):
            X = d[f"{lin}/{'x' if side == 'a' else 'w'}"].astype(np.float64)
            X = X[:, : args.max_k] if args.max_k else X
            yield {"source": f.stem, "linear": lin, "k": int(X.shape[1])}, X


def cmd_basesplit(args) -> dict:
    """The base-split falsifier (design X-FP4-1): per family/format/noise, how cheap the salt-free split is (A side)."""
    rng = np.random.default_rng(args.seed)
    res = {"k": args.k, "pairs": args.pairs, "seed": args.seed, "noise": args.noise, "delta": args.delta,
           "acts": str(args.acts) if args.acts else None, "results": []}
    for name, (atom, fmt) in atoms_from(args).items():
        for label, A in operands(args, rng):
            for noise in args.noise.split(","):
                if noise == "none":
                    continue                                            # the split needs a salt: noise is what the base can't guess
                r = base_split(fmt, A, noise, args.delta, rng)
                row = {"atom": atom.name, "format": fmt, **label, "noise": noise, **r}
                res["results"].append(row)
                print(json.dumps({"format": fmt, **label, "noise": noise, "density": round(r["code_change_density"], 3),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "synth", "real", "basesplit", "merge", "debit", "breakeven"])
    ap.add_argument("--atom", action="append", default=None)
    ap.add_argument("--frac-bits", type=int, default=None)
    ap.add_argument("--acc-window", type=int, default=None)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--families", default="all")
    ap.add_argument("--noise", default="none,row,block")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--acts", default=None)
    ap.add_argument("--operand", choices=["a", "b"], default="a", help="basesplit/merge: activations (a) or weights (b)")
    ap.add_argument("--max-k", type=int, default=0)
    ap.add_argument("--partner-blocks", type=int, default=256, help="merge: k-blocks the adaptive partner search covers")
    ap.add_argument("--hot-bits", type=int, default=14, help="debit: v2's H has its grid at 2^-hot_bits of the RMS atom sum")
    ap.add_argument("--hot-rule", choices=["row", "word"], default="row", help="debit: whose RMS atom sum sets v2's H")
    ap.add_argument("--dead-rho", type=float, default=DEAD_RHO,
                    help="debit: D-SS's dead block is clean amax_b^2 > dead_rho^2 * rho^2 (default sqrt(90))")
    ap.add_argument("--screen", choices=["debit", "cap"], default="debit",
                    help="debit: no row cap, the screen charged in item 3 (10:35Z; item 4 on the witness, 11:32Z); "
                         "cap: at most one dead block in 64")
    ap.add_argument("--n", type=int, default=0,
                    help="debit: output width for the forming term (default: Qwen2.5-7B's per linear with --acts, else k)")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    args.atom = args.atom or ["blackwell_sm120_nvf4_m16n8k64", "blackwell_sm120_mxf4_m16n8k64"]
    return args


def main() -> None:
    args = parse_args()
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
    doc = {"synth": cmd_synth, "real": cmd_real, "basesplit": cmd_basesplit, "merge": cmd_merge, "debit": cmd_debit}[args.cmd](args)
    doc.update({"tool": "pouw_fp4_census", "cmd": args.cmd, "atoms": args.atom, "selftest": rep, "seconds": round(time.time() - t0, 1)})
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        es = [r["words"]["exact_steps"] for r in doc["results"] if "words" in r]
        eps = [r["eps_R_strict_max"] for r in doc["results"] if "eps_R_strict_max" in r]
        tiles = [r[v]["tile_debit_exact_max"] for r in doc["results"] for v in ("v1", "v2") if v in r]
        if tiles:
            run_result.write(out, all("v1" in r for r in doc["results"]), f"fp4-census-{args.cmd}",
                             f"{len(doc['results'])} cells; worst tile debit share {max(tiles):.6f}",
                             [("cells", len(doc["results"]), "count"), ("tile_debit_share_max", max(tiles), "fraction")])
        elif eps:
            run_result.write(out, True, f"fp4-census-{args.cmd}", f"{len(eps)} cells; eps_R (strict) max {max(eps):.5f}",
                             [("cells", len(eps), "count"), ("eps_R_strict_max", max(eps), "fraction")])
        elif es:
            run_result.write(out, True, f"fp4-census-{args.cmd}", f"{len(doc['results'])} cells; exact steps {min(es):.4f}..{max(es):.4f}",
                             [("cells", len(doc["results"]), "count"), ("exact_steps_max", max(es), "fraction")])
        else:
            cc = [r["cheater_A_cost_per_dense_mac"] for r in doc["results"] if "cheater_A_cost_per_dense_mac" in r]
            run_result.write(out, bool(cc) and min(cc) >= 0.999, f"fp4-census-{args.cmd}",
                             f"{len(doc['results'])} cells; cheater A cost {min(cc):.3f}..{max(cc):.3f} (>=1 => no saving)" if cc else "no cells",
                             [("cells", len(doc["results"]), "count"), ("cheater_A_cost_min", min(cc) if cc else 0.0, "ratio")])


if __name__ == "__main__":
    main()
