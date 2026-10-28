"""ATTN-NOISE's exact check: attention's committed words at every operand value (B3: K = V = 0), and its noise's h.

Coordinates are checked_strip.py's (Lean's `Definitions.Pouw.RelativeSecurity`): the peeled side P carries U (and, in
two-sided designs, a fresh bit c), the served side V carries a fresh bit b an entry. The forms:
- adopted P·V (H1's `peel`, apv_ref.py): P̃ = P + U(·, κ mod 32), X = V + b; words T_t and the served O
  (checked_strip.py's `checked1` at r = L = 32);
- cohalf QKᵀ (H1's `cohalf`, cgemm_ref.py's check_half): Q̃ = Q + c + U(·, κ mod k/2) with U on ±16, X = K + b; words
  T_t and O (narrow_u.py's `co` at r = k/2).

Parts:
- B. B3's instances: F₂ cross ranks of one cell's words at the zero instance, P = 0 alone, V = 0 alone, a one-hot P
  row, a uniform P row and a random instance. Adopted P·V has k (the chain lemma's T·L) and cohalf (k, k + r) at
  every one: the cross block doesn't read the operands (`Proofs.Pouw.AttnNoise.pairDiff_dPrefix_operands`). The
  served O adds no rank to P·V's T (`crossRankLe_dClean`: the peeled output is bit-affine).
- C. B3's premises: one-sided noise (U's weights 0) has rank 0 at every instance and every word constant at the zero
  instance (`not_chainBound_noNoise`); row dither (U of period r = 1) has rank k/L, one a word. Mutations put each
  premise on the adopted form and are caught by part B's check.
- G. Per unit (one output coordinate), the noise bits a cheat must match per unit of work it saves (h, GRIND),
  from how units share an operand row's noise: matching one side's noise leaves a unit's words
  affine in the other side's coins, which have no floor (`crossRankLe_dClean`'s case). Then the SHA-512 compressions
  a drawn unit spends deriving its noise in its proof on the deployed NS 1 patch layout (4 rows × 128
  columns a compression; V's over Vᵀ) and on a layout with each unit's operand noise contiguous, and GRIND's
  bound a seed at prefill and at decode.

Run from the repository root: `uv run python -B benchmarks/pouw/theory/attn_noise.py --mutations`.
`benchmarks/pouw/tests/test_theory_attn_noise.py` runs it in the pouw suite.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from verity.core.protocols.pouw.credit import DELTA

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checked_strip import W_RE, Block, cross_ranks  # noqa: E402
from exact_kit.checks import Checks  # noqa: E402
from narrow_u import NarrowBlock, planes  # noqa: E402

INSTANCES = ("zero", "P = 0", "V = 0", "one-hot P", "uniform P", "random")
#: SHA-512's output bits a compression in counter mode (one block a counter value).
SHA_BITS = 8 * hashlib.sha512().digest_size
#: The NS 1 patch: 4 rows × 128 columns of 1-bit noise a compression (sha_noise.cuh, cgemm_ref.patch_bits).
PATCH_ROWS, PATCH_COLS = 4, 128
#: A drawn unit's seed: 2 compressions (SHA-512 of the window coin and the two roots).
SEED = 2


def operands(B: Block, kind, rng):
    """(P, V) of one block at instance `kind`; P in Lean coordinates, a probability row as P_prob − 64."""
    P = rng.integers(-63, 63, (B.n, B.k))
    V = rng.integers(-128, 127, (B.k, B.n2))
    if kind == "zero":
        return np.zeros_like(P), np.zeros_like(V)
    if kind == "P = 0":
        return np.zeros_like(P), V
    if kind == "V = 0":
        return P, np.zeros_like(V)
    if kind == "one-hot P":
        P = np.full((B.n, B.k), -64, np.int64)
        P[np.arange(B.n), rng.integers(0, B.k, B.n)] = 62
        return P, V
    if kind == "uniform P":
        return np.full((B.n, B.k), round(126 / B.k) - 64, np.int64), V
    if kind == "random":
        return P, V
    raise ValueError(kind)


def ranks(B: Block, rng):
    return {kind: cross_ranks(B, *operands(B, kind, rng), mod=1) for kind in INSTANCES}


def h_table(L, d=128, u_bits_qk=64 * 5, u_bits_pv=32 * 7):
    """GRIND's h a form and operand: the bits of one operand row's noise over the units that read it (every unit of
    a row, or of a column, of one A·B with its own seed), at the worst row (key 0 or query L − 1: L units)."""
    return {"QKᵀ, K row (b)": d / L, "QKᵀ, Q row (c and U)": (d + u_bits_qk) / L,
            "P·V, V column (b)": L / L, "P·V, P row (U)": u_bits_pv / d}


def h_decode(L, d=128, u_bits_qk=64 * 5, u_bits_pv=32 * 7):
    """h at a decode step of one query against L keys, K and V noised afresh by the step's matmuls: Q's row is read by
    the step's L units, each K row by one, P's row by d units and each V column's L bits by one."""
    return {"QKᵀ, K row (b)": d, "QKᵀ, Q row (c and U)": (d + u_bits_qk) / L,
            "P·V, V column (b)": L, "P·V, P row (U)": u_bits_pv / d}


#: The per-window δ's bits: δ = 2^-DELTA_LOG2 (PoUW's `credit.DELTA`).
DELTA_LOG2 = DELTA.denominator.bit_length() - 1


def grind_eps(L, delta_log2=DELTA_LOG2, t_log2=128, d=128):
    """Decision 112's (iv): a seed's grind premium is at most log₂(T/δ)/h units, so ε_grind ≤ log₂(T/δ)/(h·N) over
    the N units the seed covers, one seed a matmul (one head). Prefill: QKᵀ covers the causal L(L + 1)/2 scores,
    P·V L·d outputs; decode: L scores and d outputs."""
    bits, hp, hd = t_log2 + delta_log2, h_table(L, d), h_decode(L, d)
    return {"prefill QKᵀ": bits / (min(hp["QKᵀ, K row (b)"], hp["QKᵀ, Q row (c and U)"]) * L * (L + 1) / 2),
            "prefill P·V": bits / (min(hp["P·V, V column (b)"], hp["P·V, P row (U)"]) * L * d),
            "decode QKᵀ": bits / (min(hd["QKᵀ, K row (b)"], hd["QKᵀ, Q row (c and U)"]) * L),
            "decode P·V": bits / (min(hd["P·V, V column (b)"], hd["P·V, P row (U)"]) * d)}


def unit_compressions(k, d=128):
    """SHA-512 compressions a drawn unit spends on its noise: (deployed patch layout, contiguous layout). QKᵀ's unit
    reads Q's row (c in one patch, U from its tag-11 digest) and K's row (one patch); P·V's reads P's row's U (one
    digest a row pair) and V's column of b over k keys, which the patch layout over Vᵀ serves 128 keys a compression."""
    qk = (SEED + 1 + 1 + 1, SEED + math.ceil((d + 64 * 5 + d) / SHA_BITS))
    pv = (SEED + 1 + math.ceil(k / PATCH_COLS), SEED + math.ceil((32 * 7 + k) / SHA_BITS))
    return {"QKᵀ": qk, "P·V": pv}


def main():
    checks = Checks()
    rng = np.random.default_rng(20261018)
    out = {}

    # ------------------------------------------------------------ B
    k, L = 64, 32
    pv_T, pv_TO = Block("dither", 1, k, 1, 32, L), Block("checked1", 1, k, 1, 32, L)
    got_T, got_TO = ranks(pv_T, rng), ranks(pv_TO, rng)
    out["P·V adopted, T"] = {x: list(v[:2]) for x, v in got_T.items()}
    out["P·V adopted, T and O"] = {x: list(v[:2]) for x, v in got_TO.items()}
    checks.check(f"B1 adopted P·V (k, r, L) = ({k}, 32, {L}): T's cross rank k = {k} at every instance (zero, P = 0, "
                 f"V = 0, one-hot P, uniform P, random), and committing O beside T adds none",
                 all(v[0] == k for v in got_T.values()) and all(v[:2] == got_T[x][:2] for x, v in got_TO.items()),
                 str(out["P·V adopted, T and O"]))
    for kk in (64, 96, 128):
        rr = kk // 2
        qk = NarrowBlock("co", 1, kk, 1, rr, L, planes(16))
        got = ranks(qk, rng)
        out[f"QKᵀ cohalf k={kk}"] = {x: list(v[:2]) for x, v in got.items()}
        checks.check(f"B2 cohalf QKᵀ (k, r, L) = ({kk}, {rr}, {L}), U on ±16: cross ranks (k, k + r) = ({kk}, "
                     f"{kk + rr}) at every instance, same-side blocks 0",
                     all(v[:2] == (kk, kk + rr) and v[2] == 0 for v in got.values()), str(out[f"QKᵀ cohalf k={kk}"]))

    # ------------------------------------------------------------ C
    one = Block("checked1", 1, k, 1, 32, L, w=np.zeros(7, np.int64))
    got = ranks(one, rng)
    bare = Block("dither", 1, k, 1, 32, L, w=np.zeros(7, np.int64))
    zw = bare.words(np.full((1, k), 64, np.int64), np.zeros((k, 1), np.int64), rng.integers(0, 2, (64, bare.nbits)))
    out["one-sided"] = {x: list(v[:2]) for x, v in got.items()}
    checks.check("C1 one-sided noise (U's weights 0, so U = −64, and b on V only): cross rank 0 at every instance, and "
                 "at P̃ = 0, V = 0 every T word is 0 at every coin point (B3's collapse)",
                 all(v[:2] == (0, 0) for v in got.values()) and not np.any(zw), str(out["one-sided"]))
    row = Block("checked1", 1, k, 1, 1, L)
    got = ranks(row, rng)
    out["row dither"] = {x: list(v[:2]) for x, v in got.items()}
    checks.check(f"C2 row dither (U of period 1 a row): cross rank k/L = {k // L}, one a word, at every instance "
                 f"(B3's rank-1 cross terms)", all(v[0] == k // L for v in got.values()), str(out["row dither"]))
    if checks.mutations:
        for name, B in (("one-sided noise on adopted P·V", Block("dither", 1, k, 1, 32, L, w=np.zeros(7, np.int64))),
                        ("row dither on adopted P·V", Block("dither", 1, k, 1, 1, L, w=W_RE))):
            got = ranks(B, rng)
            with checks.mutant(name):
                checks.check(f"M B1's check on {name}", all(v[0] == k for v in got.values()), str(got))

    # ------------------------------------------------------------ G
    h = {f"L={n}": h_table(n) for n in (2048, 8192, 32768)}
    comp = {f"k={n}": unit_compressions(n) for n in (2048, 8192, 32768)}
    out["h"] = {key: {f: round(v, 5) for f, v in row_.items()} for key, row_ in h.items()}
    out["compressions (patch, contiguous)"] = comp
    worst = {key: min(row_, key=row_.get) for key, row_ in h.items()}
    checks.check("G1 GRIND's h: the least is QKᵀ's K row, d/L bits a unit (one key's 128 bits, read by every query of "
                 "its head), 1/16 at 2K and 1/256 at 32K; P·V's V column holds 1 bit a unit, its P row 1.75",
                 all(w_ == "QKᵀ, K row (b)" for w_ in worst.values()) and h["L=2048"]["QKᵀ, K row (b)"] == 1 / 16
                 and h["L=32768"]["QKᵀ, K row (b)"] == 1 / 256 and h["L=8192"]["P·V, V column (b)"] == 1
                 and h["L=2048"]["P·V, P row (U)"] == 1.75, str(out["h"]))
    checks.check("G2 the noise per drawn unit: QKᵀ 5 compressions (4 contiguous); P·V 3 + k/128 on the "
                 "patch layout over Vᵀ (19, 67, 259 at k = 2K, 8K, 32K) against 2 + ⌈(224 + k)/512⌉ contiguous (7, 19, "
                 "67)", comp["k=2048"] == {"QKᵀ": (5, 4), "P·V": (19, 7)} and comp["k=8192"]["P·V"] == (67, 19)
                 and comp["k=32768"]["P·V"] == (259, 67), str(comp))
    grind = {f"L={n}": grind_eps(n) for n in (2048, 8192, 32768)}
    out["grind ε bound a seed (T = 2^128, δ = 2^-40)"] = {key: {f: round(v, 6) for f, v in g.items()}
                                                          for key, g in grind.items()}
    checks.check("G3 GRIND's bound a seed, one seed a matmul: prefill QKᵀ 2.625/(L + 1) of its work "
                 "(0.128% at 2K, 0.008% at 32K) and P·V 1.3125/L; at decode one query's Q row and P row bind, "
                 "37.5% of the step's QKᵀ and 75% of its P·V at any L: decode needs more units a seed",
                 abs(grind["L=2048"]["prefill QKᵀ"] - 2.625 / 2049) < 1e-6
                 and abs(grind["L=32768"]["prefill P·V"] - 1.3125 / 32768) < 1e-6
                 and all(abs(g["decode QKᵀ"] - 0.375) < 1e-9 and abs(g["decode P·V"] - 0.75) < 1e-9
                         for g in grind.values()), str(out["grind ε bound a seed (T = 2^128, δ = 2^-40)"]))

    print("\nRESULTS " + json.dumps(out, default=str, sort_keys=True))
    sys.exit(checks.finish())


if __name__ == "__main__":
    main()
