"""Exact checks of the cheater routes behind `model.py`'s floors, at A = B = 0, on small integer shapes.

Each check computes a segment's noised increment Δ_s = Σ_{k in s} X[:, k] W'[:, k]ᵀ directly and by the route `model.py`
prices, and asserts they agree exactly (over the integers, or mod 2^b for a b-bit word):

- `hyb`: Dither v3.2's split, with U's columns permuted per noise tile, per segment, or not at all;
- `limbs`: the merged factor's limb count, and that an 8-bit word needs one u8 limb of it;
- `planes`: the bit-plane route for a word over fresh noise, k(k+1)/2-style counts for a multi-bit dither;
- `wraps`: wrap rates of absmax-quantized operands under ±a noise (synthetic operands, as `v31.operands("llm")`).

`uv run benchmarks/pouw/noise_structures/check.py --out DIR` writes DIR/check.json and exits nonzero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.int64) @ b.astype(np.int64).T


def noise(rng: np.random.Generator, shape: tuple[int, ...], amp: int) -> np.ndarray:
    return rng.integers(-amp, amp + 1, size=shape, dtype=np.int64)


def perms(rng: np.random.Generator, lp: int, tiles: int, t: int, mode: str) -> list[np.ndarray]:
    if mode == "none":
        return [np.arange(lp)] * tiles
    if mode == "segment":
        per_seg = [rng.permutation(lp) for _ in range(-(-tiles // t))]
        return [per_seg[tau // t] for tau in range(tiles)]
    return [rng.permutation(lp) for _ in range(tiles)]


def hyb(rng: np.random.Generator, m: int, n: int, lp: int, t: int, segs: int, mode: str, amp: int = 80) -> dict:
    """Δ_s = d_A,s d_B,sᵀ + (D_s + V M_s) Uᵀ + V Y_sᵀ with U's columns read through π_τ, checked against the direct sum.

    Under `none` the route is d_A,s d_B,sᵀ + D_s Uᵀ + V Y_sᵀ + t·(V Uᵀ), with V Uᵀ computed once for every segment."""
    k = lp * t * segs
    tiles = k // lp
    dA = rng.integers(0, 2, size=(m, k), dtype=np.int64)
    dB = rng.integers(0, 2, size=(n, k), dtype=np.int64)
    V, U = noise(rng, (m, lp), amp), noise(rng, (n, lp), amp)
    pis = perms(rng, lp, tiles, t, mode)
    X = dA + np.tile(V, (1, tiles))
    Wp = dB + np.concatenate([U[:, p] for p in pis], axis=1)
    assert np.abs(X).max() <= 127 and np.abs(Wp).max() <= 127, "A = B = 0 never wraps"
    out = {"m": m, "n": n, "lp": lp, "t": t, "segs": segs, "mode": mode, "amp": amp, "ok": True}
    VU = mm(V, U)
    f_max = 0
    for s in range(segs):
        cols = slice(s * t * lp, (s + 1) * t * lp)
        direct = mm(X[:, cols], Wp[:, cols])
        binary = mm(dA[:, cols], dB[:, cols])
        D = np.zeros((m, lp), dtype=np.int64)
        VM = np.zeros((m, lp), dtype=np.int64)
        Y = np.zeros((n, lp), dtype=np.int64)
        for tau in range(s * t, (s + 1) * t):
            inv = np.argsort(pis[tau])
            D += dA[:, tau * lp:(tau + 1) * lp][:, inv]
            VM += V[:, inv]
            Y += dB[:, tau * lp:(tau + 1) * lp]
        if mode == "none":
            route = binary + mm(D, U) + mm(V, Y) + t * VU
        else:
            F = D + VM
            f_max = max(f_max, int(np.abs(F).max()))
            route = binary + mm(F, U) + mm(V, Y)
        out["ok"] &= bool(np.array_equal(direct, route))
    out["merged_factor_max"] = f_max
    return out


def limbs(rng: np.random.Generator, m: int, n: int, lp: int, t: int, amp: int = 80) -> dict:
    """The merged factor F = D_s + V M_s needs two s8 limbs for a 16-bit word at t ≥ 2, one u8 limb for an 8-bit word."""
    dA = rng.integers(0, 2, size=(m, t * lp), dtype=np.int64)
    V, U = noise(rng, (m, lp), amp), noise(rng, (n, lp), amp)
    pis = [rng.permutation(lp) for _ in range(t)]
    F = np.zeros((m, lp), dtype=np.int64)
    for tau, p in enumerate(pis):
        inv = np.argsort(p)
        F += dA[:, tau * lp:(tau + 1) * lp][:, inv] + V[:, inv]
    exact = mm(F, U)
    lo = ((F + 128) % 256) - 128
    hi = (F - lo) // 256
    two = mm(lo, U) + 256 * mm(hi, U)
    u8 = F % 256
    one = mm(u8, U)
    return {"t": t, "amp": amp, "F_range": [int(F.min()), int(F.max())],
            "fits_one_s8_limb": bool(np.abs(F).max() <= 127),
            "two_s8_limbs_exact": bool(np.array_equal(exact, two)),
            "hi_limb_fits_s8": bool(np.abs(hi).max() <= 127),
            "one_u8_limb_exact_mod_2^8": bool(np.array_equal(exact % 256, one % 256)),
            "one_u8_limb_exact_mod_2^16": bool(np.array_equal(exact % 65536, one % 65536))}


def split24(hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """hi = S + R with S 2:4 sparse along the inner (last) axis, groups of 4 consecutive entries; R takes each group's
    nonzeros past its first two."""
    m, k = hi.shape
    g = hi.reshape(m, k // 4, 4)
    nz = np.cumsum(g != 0, axis=2)
    S = np.where(nz <= 2, g, 0).reshape(m, k)
    return S, hi - S


def sparse_hi(rng: np.random.Generator, m: int, n: int, lp: int, t: int, mode: str, amp: int = 80) -> dict:
    """The merged factor's high limb as a 2:4 sparse operand: F = lo + 256·hi with lo on [-128, 127], hi = S + R, S 2:4
    along L' (the left operand's inner axis, as wgmma.sp takes it), R a residual on the CUDA cores. Checks
    F Uᵀ = lo Uᵀ + 256 (S Uᵀ + R Uᵀ) exactly and reports hi's density and R's.

    `tile`: F = D_s + Σ_τ V π_τ⁻¹ (a sum of t independent permuted copies, concentrated). `segment`: π_τ = π_s, so F's V
    part is t·Vπ_s⁻¹, which the cheater never merges: it prices D_s Uᵀ and (Vπ_s⁻¹)Uᵀ as one dense limb each instead."""
    dA = rng.integers(0, 2, size=(m, t * lp), dtype=np.int64)
    V, U = noise(rng, (m, lp), amp), noise(rng, (n, lp), amp)
    pis = perms(rng, lp, t, t, mode)
    F = np.zeros((m, lp), dtype=np.int64)
    for tau, p in enumerate(pis):
        inv = np.argsort(p)
        F += dA[:, tau * lp:(tau + 1) * lp][:, inv] + V[:, inv]
    lo = ((F + 128) % 256) - 128
    hi = (F - lo) // 256
    S, R = split24(hi)
    exact = np.array_equal(mm(F, U), mm(lo, U) + 256 * (mm(S, U) + mm(R, U)))
    return {"t": t, "mode": mode, "amp": amp, "hi_density": float(np.mean(hi != 0)), "residual_density": float(np.mean(R != 0)),
            "hi_max": int(np.abs(hi).max()), "exact": bool(exact)}


def planes(rng: np.random.Generator, m: int, n: int, k: int, dither_bits: int, word_bits: int) -> dict:
    """Fresh int8 noise R (n×K) against a fresh d-bit dither (m×K) at A = 0: T mod 2^w from b1 AND-popcount plane pairs
    (i, j) with i + j < w. The count of pairs is the route's b1 products; the check shows the pairs suffice exactly."""
    d = rng.integers(0, 2 ** dither_bits, size=(m, k), dtype=np.int64)
    R = rng.integers(-128, 128, size=(n, k), dtype=np.int64)
    T = mm(d, R)
    Ru = R % 256
    acc = np.zeros((m, n), dtype=np.int64)
    pairs = 0
    for i in range(dither_bits):
        for j in range(8):
            if i + j < word_bits:
                weight = -(1 << (i + j)) if j == 7 else 1 << (i + j)
                acc += weight * mm((d >> i) & 1, (Ru >> j) & 1)
                pairs += 1
    mod = 1 << word_bits
    return {"dither_bits": dither_bits, "word_bits": word_bits, "b1_products": pairs, "b1_cost_over_G_at_rb_1/8": pairs / 8,
            "exact": bool(np.array_equal(T % mod, acc % mod))}


def q127(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(127 * x / np.abs(x).max(axis=1, keepdims=True)), -127, 127).astype(np.int64)


def wraps(rng: np.random.Generator, rows: int, k: int, amps: list[int]) -> list[dict]:
    """Share of operand entries with A + d + noise outside int8, for absmax-quantized Student-t(3) activations and Gaussian
    weights (synthetic, `v31.operands("llm")`'s model), noise uniform on ±a, d binary."""
    A = q127(rng.standard_t(3, size=(rows, k)))
    B = q127(rng.standard_normal(size=(rows, k)))
    res = []
    for a in amps:
        out = {"amp": a}
        for name, M in (("p_X", A), ("p_W", B)):
            v = M + rng.integers(0, 2, size=M.shape) + rng.integers(-a, a + 1, size=M.shape)
            out[name] = float(np.mean((v < -128) | (v > 127)))
        res.append(out)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    rep = {
        "hyb": [hyb(rng, 6, 10, 32, t, 3, mode) for mode in ("tile", "segment", "none") for t in (1, 2, 3, 4, 8)],
        "limbs": [limbs(rng, 8, 8, 64, t, amp) for t in (1, 2, 3, 4, 8) for amp in (40, 48, 80)],
        "sparse_hi": [sparse_hi(rng, 256, 8, 256, t, mode, amp) for mode in ("tile", "segment")
                      for t in (2, 3, 4, 6, 8) for amp in (80, 127)],
        "planes": [planes(rng, 6, 9, 256, db, wb) for db in (1, 2, 3) for wb in (4, 5, 8, 16)],
        "wraps": wraps(rng, 64, 4096, [8, 16, 32, 48, 64, 80]),
    }
    ok = (all(r["ok"] for r in rep["hyb"]) and all(r["two_s8_limbs_exact"] and r["one_u8_limb_exact_mod_2^8"] for r in rep["limbs"])
          and all(r["exact"] for r in rep["planes"]) and all(r["exact"] for r in rep["sparse_hi"]))
    rep["all_ok"] = ok
    rep["failures"] = ([r for r in rep["hyb"] if not r["ok"]] + [r for r in rep["planes"] + rep["sparse_hi"] if not r["exact"]]
                       + [r for r in rep["limbs"] if not (r["two_s8_limbs_exact"] and r["one_u8_limb_exact_mod_2^8"])])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"all_ok": ok, "failures": rep["failures"], "limbs": [(r["t"], r["amp"], r["F_range"], r["fits_one_s8_limb"]) for r in rep["limbs"]],
                      "planes": [(r["dither_bits"], r["word_bits"], r["b1_products"]) for r in rep["planes"]],
                      "sparse_hi": [(r["mode"], r["t"], r["amp"], round(r["hi_density"], 4), round(r["residual_density"], 5), r["hi_max"])
                                    for r in rep["sparse_hi"]],
                      "wraps": rep["wraps"]}, indent=None))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
