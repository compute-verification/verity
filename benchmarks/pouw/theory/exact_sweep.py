"""Exact-output designs under DistinctWrites with H32: each one's T1 gap terms in W1, T2 and T3, and checks.

noise_sweep.py prices served and stacked noise; this prices the designs whose served output is A·Bᵀ exactly, on the same
terms (a share of stock's m·n·K int8 MACs in priced work, H32 = 32 a register write, moves free):

    gap = forming − forced + de-noise + fallback + [final word free]·32/K + dup·α (+ coins when uncredited)

Two results shape the list.

- **Exact in the GEMM costs width (`minimal_width`).** If X = A·P + E·Q and Y = R·B + T·F + H equal A·Bᵀ for all A, B, E,
  F, then P·R = I, Q·R = 0, Q·T = 0 and Q·H = 0, so col(R) and col([T|H]) are independent subspaces of ker Q and
  W ≥ K + rank Q + rank [T|H]. Full-rank noise on both sides needs 3K (NCP's ncp3 is tight); one-sided needs 2K (os2) or
  K + rank Q (per2, lowrank1). Every such design's final word is A·Bᵀ, free under M0's free set.
- **De-noising in the stock epilogue is free (`epilogue`).** Stock converts each accumulator with I2F, scales it by the
  column's then the row's scale (FMUL, FMUL) and packs bf16. Replacing I2F by IADD3(D, 0x4B400000 − r_i, −c_j) gives the
  float 1.5·2²³ + (D − r_i − c_j) while |A·Bᵀ| < 2²², and replacing the column FMUL by FFMA(v, s_j, −1.5·2²³·s_j)
  removes the offset; when 1.5·2²³·s_j is exact in fp32 the output is stock's bit for bit. A separable noise r_i + c_j
  is removed with the epilogue's own instruction count. A column-only c_j folds into the FFMA alone while
  |A·Bᵀ + c_j| < 2²⁴. A noise that isn't separable (SSN's code a₁·C_j) costs a write an output (32/K) to remove.

`rc_magic` carries a row bit every k32 step and a fresh column bit in step 1 and removes both at the epilogue; its words
are distinct on the salt grid for every data case. A program that branches on the coins (W1 prices a branch at 0) sees
more: per salt, a row-step whose n words equal an earlier row-step's is copied (`branch_collapse`). One bit a step
merges chosen zero rows' steps and chosen identical rows' prefixes, so rc_magic, rc_ffma, ssn_xr and ssn_x collapse as
ssn_1l does. `rc_hw` hardens it: a 28-bit row code against the column code in step 1, odd step coins, and step S
trading the code for a row offset of about 2¹⁸ values, so chosen rows lose only final words whose offsets collide
(`hw_final_bound`). Chosen weights see more again: a step lane against a public column weight (1, or ssn_1l_hw's
signs) is cancelled by one weight lane equal to −weight in each step's block and an activation lane of 1, so a step
with a_t = 1 adds nothing and repeats the row's previous row-step (about half of them; `cancel`). `rc_hg` weights the
step lanes by G ∈ {1, 2}ⁿ drawn from coins once the weights are registered and cancels the walk w_i·G_j in step S, so
no committed data can cancel a step; ssn_1l_hg does the same for the served lead. Two limits remain for every exact
design:

- a final word with no row noise (A·Bᵀ, or A·Bᵀ + c_j) is the same on chosen identical rows, so it is lost: (m−1)/(m·S);
- a final word with row noise needs the magic, whose range a per-row bound must certify: free after an RMSNorm (from
  stock's own row scale), K/4 IDP4A a row elsewhere (8/n), and on chosen out-of-range rows the fallback write (32/K)
  on every row, unless the weight matrix is statically safe (127·max_j ‖B_j‖₁ < 2²²).

`rc_hw_mix` (`rc_hg_mix`) takes rc_hw (rc_hg) where the bound is free and rc_hw_ffma (rc_hg_ffma) elsewhere. Served
noise has neither limit.

Data models: M0 (free set: constants, salt-independent functions, raw coin words), M1u (the call's own words on a grid
uniform in A: the final-word lemma, under PeriodInventory and the atomic call) and M1s (M1 on a grid that keeps the
served data's zero blocks and identical rows, priced on noise_data.py's measurement). `--data` reads exact_data.py's
output (final-word repeats, the magic range's fallback rows and static safety, the scale fold's exactness),
`--noise-data` noise_data.py's. Every row carries `gap` (served data), `gap_chosen` (the worst of prover-chosen
identical, zero and out-of-range rows, the weights public) and `gap_chosen_weights` (chosen weights too).
Output: JSON (`--json`) and a text table (`--table`); exit 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_sweep as ns  # noqa: E402

H32 = ns.H32
MAGIC_BITS = 0x4B400000
MAGIC = 1.5 * 2 ** 23
MAGIC_RANGE = 2 ** 22
LANE_MAX = 127

MODELS = ("M0", "M1u", "M1s")
EXACT: dict = {}
NOISE: dict = {}


def steps(k: int, first: int, extra_last: Callable[[int], int] = lambda s: 0) -> int:
    """k32 steps for K data entries: `first` in step 1, 31 in every later step, and `extra_last(S)` more noise lanes in
    the last step (beyond its one)."""
    s = 1 + math.ceil(max(0, k - first) / 31)
    while first + 31 * (s - 1) - extra_last(s) < k:
        s += 1
    return s


def s_rc(k: int) -> int:
    return steps(k, 30)


def s_rcf(k: int) -> int:
    return steps(k, 30, lambda s: math.ceil((s - 1) / LANE_MAX) - 1)


def s_x(k: int) -> int:
    return steps(k, 28, lambda s: 3 + math.ceil(max(1, s - 2) / LANE_MAX))


def s_xr(k: int) -> int:
    return steps(k, 28, lambda s: 4)


HW_KAPPA = (1, 121, 121, 121)
HW_EXTRA_MAX = 24
HW_FIXED = 9
#: rc_hg's two more step-S lanes: the walk w_i = Σ a_t as d0 + 63·d1 against −G_j and −63·G_j
HG_FIXED = HW_FIXED + 2
HG_BASE = 63


def hw_steps(k: int, fixed: int = HW_FIXED) -> int:
    """rc_hw's k32 steps: step 1 holds 27 data, the code a1 (4 lanes) and c_j's constant lane; steps 2..S hold 31
    data and a_t, less 5 + e in step S (the code's cancellation, a1₂'s lane and e lanes of a1₃); rc_hg (`HG_FIXED`)
    has two more lanes there."""
    s = 2
    while 31 * s - fixed < k:
        s += 1
    return s


def hg_steps(k: int) -> int:
    return hw_steps(k, HG_FIXED)


def hw_extra(k: int, fixed: int = HW_FIXED) -> int:
    return min(HW_EXTRA_MAX - (fixed - HW_FIXED), 31 * hw_steps(k, fixed) - fixed - k)


def hw_weights_e(e: int) -> tuple:
    """The row offset a1·W step S leaves: κ from the code's cancellation against κ − C_j, a1₂'s lane against 7 (so
    a1₁ + 128·a1₂ is uniform on [0, 2¹⁴)) and e lanes of a1₃ against 127."""
    return (HW_KAPPA[0], HW_KAPPA[1] + 7, HW_KAPPA[2] + 127 * e, HW_KAPPA[3])


def hw_weights(k: int, fixed: int = HW_FIXED) -> tuple:
    return hw_weights_e(hw_extra(k, fixed))


def hw_offset_q(k: int, fixed: int = HW_FIXED) -> float:
    """1 / Σ p² of the final row offset (a1 uniform on [0, 127]⁴, a_t = 1 + 2·b_t): rc_hw's r_i = a1·W + Σ_{t≥2} a_t,
    rc_hg's a1·W (its walk cancels in step S). The effective number of values a pair of chosen-equal rows' final words
    must miss."""
    s, w = hw_steps(k, fixed), hw_weights(k, fixed)
    size = 127 * sum(w) + 2 * s + 1
    nfft = 1 << (size - 1).bit_length()
    spec = np.ones(nfft // 2 + 1, dtype=complex)
    for wl in w:
        lane = np.zeros(nfft)
        lane[:127 * wl + 1:wl] = 1 / 128
        spec *= np.fft.rfft(lane)
    walk = np.zeros(nfft)
    if fixed == HW_FIXED:
        walk[:2 * (s - 1) + 1:2] = [math.comb(s - 1, i) / 2 ** (s - 1) for i in range(s)]
    else:
        walk[0] = 1.0
    p = np.clip(np.fft.irfft(spec * np.fft.rfft(walk), nfft)[:size], 0, None)
    return float(1 / (p ** 2).sum() * p.sum() ** 2)


def hg_g(n: int, seed: int = 2026) -> np.ndarray:
    """rc_hg's step-lane column weights G ∈ {1, 2}ⁿ, drawn from coins once the weights are registered (here a fixed
    generator the data never reads)."""
    return 1 + np.random.default_rng(seed).integers(0, 2, n)


def row_words(bits: int) -> int:
    return math.ceil(bits / 32)


def row_bit_writes(s: int) -> int:
    """A row's noise-lane forming a call: a LOP3 a step and a funnel-shift rotate for every step not at bit 24."""
    return s + s - row_words(s)


def popc_writes(bits: int) -> int:
    """r_i = popcount of a row's noise bits: a mask and a POPC a coin word, summed, then one IADD for the magic."""
    return 2 * row_words(bits) + row_words(bits)


def lanes_writes(s: int) -> int:
    """−r_i spread over ⌈r_max/127⌉ int8 lanes and merged into the last step's register."""
    return 2 * math.ceil(s / LANE_MAX)


@dataclass(frozen=True)
class Design:
    name: str
    noise: str
    alpha: Callable[[int], float]
    form_x: Callable[[int, int, int], float]
    form_y: Callable[[int, int, int], float]
    forced_x: Callable[[int, int, int], float]
    forced_y: Callable[[int, int, int], float]
    denoise: Callable[[int, int, int], float]
    final_free_m0: bool
    coin_bits: Callable[[int, int, int], float]
    c1_bytes: Callable[[int, int, int], float]
    dup: Callable[[str, int, int, str], float]
    fallback: bool
    t3: str
    premises: str
    verdict: str
    exact: bool = True
    notes: list[str] = field(default_factory=list)


def _z(m: int, n: int, k: int) -> float:
    return 0.0


def _no_dup(model: str, m: int, k: int, lin: str) -> float:
    return 0.0


def scope(m: int) -> str:
    return "m64" if m <= 128 else "ctx"


def exact_share(field_: str, m: int, lin: str) -> float | None:
    if not EXACT:
        return None
    per = EXACT.get("by_linear", {}).get(lin, {}).get(scope(m))
    src = per if per is not None else EXACT["summary"][scope(m)]
    return src.get(field_)


def noise_share(field_: str, m: int) -> float | None:
    if not NOISE:
        return None
    sc = ns.DATA_SCOPE[min(ns.DATA_SCOPE, key=lambda r: abs(math.log(max(m, 1) / r)))]
    return NOISE["summary"][sc][field_]


#: under M1s a final word repeats only where the data's structure makes two words one function of A: a zero row's
#: final words for a row-common noise, identical rows' for a column-common one
M1S_FINAL = {"final_row_dup_share": "zero_row_share", "final_col_dup_share": "identical_row_share"}


def _final_dup(field_: str, steps_fn: Callable[[int], int]):
    """A design whose only data-dependent loss is its final word: the repeated share of final words over S words
    (M0: equal values on the served data; M1s: the structural repeats; M1u: none)."""
    def f(model: str, m: int, k: int, lin: str) -> float:
        if model == "M1u":
            return 0.0
        v = exact_share(field_, m, lin) if model == "M0" else noise_share(M1S_FINAL[field_], m)
        return float("nan") if v is None else v / steps_fn(k)
    return f


def _m1_final(model: str, m: int, k: int, lin: str, s: int = 1) -> float:
    """An in-GEMM exact design's final word under M1s: repeated on identical rows, free on zero rows (per word)."""
    if model != "M1s":
        return 0.0
    ident = noise_share("identical_row_share", m)
    return float("nan") if ident is None else ident / max(1, s)


def _plain_dup(model: str, m: int, k: int, lin: str) -> float:
    if model == "M0":
        return 1.0
    if model == "M1u":
        return 0.0
    v = noise_share("perm_dup_share", m)
    return float("nan") if v is None else v


def designs() -> list[Design]:
    p = ns.P_ENTRY
    s1 = ns.steps1l
    w1 = lambda k: ns.ssn_1l_words(s1(k))  # noqa: E731
    premises_m0 = "DistinctWritesMH100Floor (H32, rank 1) on W1 (MH100w faithful, rank 3)"
    coins = "coins credited: ShaSpeed + hash-compose (rank 4)"
    out = [
        Design("ssn_1l", "served lead (noise_sweep): 28 data + 4-lane column code a1·C in step 1, then 31 data + 1 row bit a "
               "step; noise served, β_j by accumulator init", lambda k: H32 * s1(k) / k,
               lambda m, n, k: H32 * ns.ssn_1l_forming_writes(s1(k)) / (n * k), _z,
               lambda m, n, k: H32 * s1(k) / (n * k), _z, _z, False,
               lambda m, n, k: 32 * w1(k) * m, lambda m, n, k: 4 * s1(k) / (n * k), _no_dup, False,
               "served noise (std ~6 / ~9 units at K 2,048 / 8,192); measured on benchmarks/ppl", f"{premises_m0}; {coins}",
               "reference (served, not exact)", exact=False),
        Design("ssn_1l_mov", "ssn_1l with its −β_j accumulator init priced as a MOV an output (32/K)",
               lambda k: H32 * s1(k) / k, lambda m, n, k: H32 * ns.ssn_1l_forming_writes(s1(k)) / (n * k), _z,
               lambda m, n, k: H32 * s1(k) / (n * k), _z, lambda m, n, k: H32 / k, False,
               lambda m, n, k: 32 * w1(k) * m, lambda m, n, k: 4 * s1(k) / (n * k), _no_dup, False,
               "~0 (estimate)", f"{premises_m0}; {coins}",
               "smell: the init is a register write; fold β_j into the column FFMA instead (free)", exact=False),
        Design("ssn_pd", "ssn_1l partly de-noised in the stock epilogue: r_i (the later steps' row bits) at the magic "
               "IADD3, β_j in the column FFMA's addend; the code a1·C_j stays served", lambda k: H32 * s1(k) / k,
               lambda m, n, k: H32 * ns.ssn_1l_forming_writes(s1(k)) / (n * k), _z,
               lambda m, n, k: H32 * s1(k) / (n * k), _z, lambda m, n, k: H32 * popc_writes(s1(k)) / (n * k), False,
               lambda m, n, k: 32 * w1(k) * m, lambda m, n, k: 4 * s1(k) / (n * k), _no_dup, True,
               "served code noise only (std ~3.6 units, zero mean)", f"{premises_m0}; {coins}",
               "the served lead with no init and a third of its noise; committed words unchanged", exact=False),
        Design("ncp3", "noise_sweep's ncp3: three blocks, full-rank noise both sides (tight at 3K)", lambda k: 3.0,
               lambda m, n, k: 32 / n, lambda m, n, k: 32 / m, lambda m, n, k: 24 / n, lambda m, n, k: 24 / m, _z, True,
               lambda m, n, k: 6 * (m + n) * k + k / 4 * math.log2(k / 4), lambda m, n, k: 3 / n + 3 / m,
               lambda model, m, k, lin: _m1_final(model, m, k, lin, 3 * k // 32), False,
               "int7 activations; measured on benchmarks/ppl", f"{premises_m0}; M1 for the final word: PeriodInventory "
               "+ atomic call (rank 3)", "minimal for two-sided full-rank noise; dominated"),
        Design("os2", "X=[A+E|E], Y=[B;-B] (tight at 2K)", lambda k: 2.0, lambda m, n, k: 2 * p / n, _z,
               lambda m, n, k: 2 * p / n * (1 - (noise_share("zero_reg_share", m) or 0.0)), _z, _z, True,
               lambda m, n, k: 3 * m * k, lambda m, n, k: 2 / n,
               lambda model, m, k, lin: _m1_final(model, m, k, lin, 2 * k // 32), False,
               "activations clamped to ±124; measured on benchmarks/ppl", f"{premises_m0}; M1 (rank 3) for the final word", "dominated by per2"),
        Design("per2", "X=[A+V.P | V x L], Y=[B; -limbs(fold32 B)] (K + 32·limbs; the bound is K + 32)",
               lambda k: 1 + H32 * ns.per2_limbs(k) / k, lambda m, n, k: p / n + 256 / (n * k), _z,
               lambda m, n, k: (p / n + 256 / (n * k)) * (1 - (noise_share("zero_reg_share", m) or 0.0)), _z, _z, True,
               lambda m, n, k: 3 * 32 * m, lambda m, n, k: 1 / n,
               lambda model, m, k, lin: _m1_final(model, m, k, lin, k // 32), False,
               "activations clamped to ±124; measured on benchmarks/ppl", f"{premises_m0}; M1 (rank 3) for the final word",
               "the best in-GEMM exact design; needs M1 (final word) and C1 at T 512"),
        Design("lowrank1", "X=[A+u vᵀ | u x L], v public, u fresh a row", lambda k: 1 + H32 * ns.per2_limbs(k) / k,
               lambda m, n, k: 2 * p / n, _z, lambda m, n, k: p / n, _z, _z, True, lambda m, n, k: 8 * m,
               lambda m, n, k: 1 / n, lambda model, m, k, lin: _m1_final(model, m, k, lin, k // 32), False,
               "~0 (estimate)", f"{premises_m0}; M1 (rank 3)", "dominated by per2"),
        Design("plain", "no noise: the committed GEMM itself", lambda k: 1.0, _z, _z, _z, _z, _z, False,
               lambda m, n, k: 0.0, _z, _plain_dup, False, "0 (stock)",
               "M1: PeriodInventory + atomic call (rank 3); fails M0 outright",
               "M1 makes noise redundant: 0 gap on a uniform-A grid; data's zero blocks and identical rows in M1s"),
        Design("pair", "os2 split over two calls ((A+E)·B and E·B), subtracted in the epilogue", lambda k: 2.0,
               lambda m, n, k: 2 * p / n, _z, lambda m, n, k: 2 * p / n, _z, lambda m, n, k: H32 / k, False,
               lambda m, n, k: 3 * m * k, lambda m, n, k: 2 / n, _no_dup, False, "0 (exact)",
               f"{premises_m0}; one atomic call over both (fresh noise a call otherwise fails)",
               "dominated by os2 + one write an output"),
        Design("ssn_x", "ssn_1l, the last step cancelling the code (−a1 against C) and the row bits (−r_i in "
               "⌈(S−2)/127⌉ lanes): the final word is A·Bᵀ", lambda k: H32 * s_x(k) / k,
               lambda m, n, k: H32 * (ns.ssn_1l_forming_writes(s_x(k)) + 2 + popc_writes(s_x(k)) + lanes_writes(s_x(k)))
               / (n * k), _z, lambda m, n, k: H32 * s_x(k) / (n * k), _z, _z, True,
               lambda m, n, k: 32 * ns.ssn_1l_words(s_x(k)) * m, lambda m, n, k: 4 * s_x(k) / (n * k),
               lambda model, m, k, lin: _m1_final(model, m, k, lin, s_x(k)), False, "0 (exact)",
               f"{premises_m0}; M1 (rank 3) for the final word", "fails M0 on Llama (32/K); passes V4 dense"),
        Design("ssn_xd", "ssn_1l served, de-noised exactly: IDP4A(a1, C_j) an output for the code, r_i at the magic",
               lambda k: H32 * s1(k) / k, lambda m, n, k: H32 * (ns.ssn_1l_forming_writes(s1(k)) + popc_writes(s1(k)))
               / (n * k), _z, lambda m, n, k: H32 * s1(k) / (n * k), _z, lambda m, n, k: H32 / k, False,
               lambda m, n, k: 32 * w1(k) * m, lambda m, n, k: 4 * s1(k) / (n * k), _no_dup, True, "0 (exact)",
               f"{premises_m0}; {coins}", "dominated: a write an output for the code"),
        Design("ssn_xr", "ssn_1l, the last step cancelling the code; r_i removed at the magic IADD3",
               lambda k: H32 * s_xr(k) / k,
               lambda m, n, k: H32 * (ns.ssn_1l_forming_writes(s_xr(k)) + 2) / (n * k), _z,
               lambda m, n, k: H32 * s_xr(k) / (n * k), _z, lambda m, n, k: H32 * popc_writes(s_xr(k)) / (n * k),
               False, lambda m, n, k: 32 * ns.ssn_1l_words(s_xr(k)) * m, lambda m, n, k: 4 * s_xr(k) / (n * k),
               _final_dup("final_row_dup_share", s_xr), True, "0 (exact; see scale_fold)",
               f"{premises_m0}; {coins}", "final word repeats on equal outputs of a row (data-dependent)"),
        Design("rc_ffma", "31 data + 1 row bit a step (step 1: 30 data, a row bit, a fresh column bit c_j against X = 1); "
               "the last step cancels the row bits (−r_i lanes); c_j folded into the column FFMA's addend",
               lambda k: H32 * s_rcf(k) / k,
               lambda m, n, k: H32 * (row_bit_writes(s_rcf(k)) + 1 + popc_writes(s_rcf(k)) + lanes_writes(s_rcf(k)))
               / (n * k), lambda m, n, k: 2 * H32 / (m * k),
               lambda m, n, k: H32 * s_rcf(k) / (n * k), lambda m, n, k: H32 / (m * k), lambda m, n, k: H32 / (m * k),
               False, lambda m, n, k: 32 * row_words(s_rcf(k)) * m + 32 * row_words(n),
               lambda m, n, k: 4 * s_rcf(k) / (n * k) + 4 / (m * k), _final_dup("final_col_dup_share", s_rcf), False,
               "0 (exact while |A·Bᵀ + c_j| < 2^24)", f"{premises_m0}; {coins}",
               "final word repeats on equal outputs of a column (data-dependent); no row check"),
        Design("rc_magic", "31 data + 1 row bit a step (step 1: 30 data, a row bit, a fresh column bit c_j against X = 1); "
               "r_i + c_j removed by the stock epilogue's IADD3 (magic) and FFMA", lambda k: H32 * s_rc(k) / k,
               lambda m, n, k: H32 * (row_bit_writes(s_rc(k)) + 1) / (n * k), lambda m, n, k: 2 * H32 / (m * k),
               lambda m, n, k: H32 * s_rc(k) / (n * k), lambda m, n, k: H32 / (m * k),
               lambda m, n, k: H32 * popc_writes(s_rc(k)) / (n * k) + H32 / (m * k), False,
               lambda m, n, k: 32 * row_words(s_rc(k)) * m + 32 * row_words(n),
               lambda m, n, k: 4 * s_rc(k) / (n * k) + 4 / (m * k), _no_dup, True,
               "0 (exact while |A·Bᵀ| < 2^22; bit-identical to stock where 1.5·2^23·s_j is exact)",
               f"{premises_m0}; {coins}", "every word distinct and non-free on any data; fallback rows pay 32/K"),
        Design("rc_hw", "hardened rc_magic: step 1 a 28-bit row code a1 ∈ [0,127]⁴ against the column code C_j and a "
               "fresh column bit c_j; steps 2..S an odd coin a_t = 1 + 2b_t against 1; step S cancels the code against "
               "κ − C_j and spends its spare lanes on a1₂ and a1₃, leaving the row offset a1·W (Q ≈ 2^18); r_i + c_j "
               "removed by the magic IADD3 and FFMA", lambda k: H32 * hw_steps(k) / k,
               lambda m, n, k: H32 * (row_bit_writes(hw_steps(k)) + 1 + math.ceil((1 + hw_extra(k)) / 4)) / (n * k),
               lambda m, n, k: 2 * H32 / (m * k), lambda m, n, k: H32 * hw_steps(k) / (n * k),
               lambda m, n, k: H32 / (m * k),
               lambda m, n, k: H32 * (popc_writes(hw_steps(k)) + 6) / (n * k) + H32 / (m * k), False,
               lambda m, n, k: 32 * row_words(27 + hw_steps(k)) * m + 32 * row_words(n),
               lambda m, n, k: 4 * hw_steps(k) / (n * k) + 4 / (m * k), _no_dup, True,
               "0 (exact while |A·Bᵀ| < 2^22)", f"{premises_m0}; {coins}; rc-hw-distinct (open)",
               "chosen identical or zero rows lose only final words with equal offsets; chosen out-of-range rows "
               "force the fallback write"),
        Design("rc_hw_ffma", "rc_hw's code and odd coins, step S cancelling the code against C_j and the walk R_i in "
               "two lanes (−⌊R_i/127⌋ against 127, −(R_i mod 127) against 1); only c_j is folded, in the FFMA: no range "
               "limit below 2^24 and no row check",
               lambda k: H32 * hw_steps(k) / k,
               lambda m, n, k: H32 * (row_bit_writes(hw_steps(k)) + 1 + popc_writes(hw_steps(k)) + 2) / (n * k),
               lambda m, n, k: 2 * H32 / (m * k), lambda m, n, k: H32 * hw_steps(k) / (n * k),
               lambda m, n, k: H32 / (m * k), lambda m, n, k: H32 / (m * k), False,
               lambda m, n, k: 32 * row_words(27 + hw_steps(k)) * m + 32 * row_words(n),
               lambda m, n, k: 4 * hw_steps(k) / (n * k) + 4 / (m * k), _final_dup("final_col_dup_share", hw_steps),
               False, "0 (exact while |A·Bᵀ| < 2^24)", f"{premises_m0}; {coins}; rc-hw-distinct (open)",
               "final word A·Bᵀ + c_j: lost on equal outputs of a column, and on chosen identical rows"),
        Design("rc_hg", "rc_hw with its step lanes against G_j ∈ {1, 2}, drawn from coins once the weights are "
               "registered, and the walk w_i = Σ a_t cancelled in step S (d0 + 63·d1 against −G_j and −63·G_j): "
               "chosen weights can't cancel a step's noise; r_i = a1·W and c_j removed by the magic IADD3 and FFMA",
               lambda k: H32 * hg_steps(k) / k,
               lambda m, n, k: H32 * (row_bit_writes(hg_steps(k)) + 1 + math.ceil((3 + hw_extra(k, HG_FIXED)) / 4)
                                      + popc_writes(hg_steps(k)) + 3) / (n * k),
               lambda m, n, k: 2 * H32 / (m * k), lambda m, n, k: H32 * hg_steps(k) / (n * k),
               lambda m, n, k: H32 / (m * k), lambda m, n, k: H32 * 6 / (n * k) + H32 / (m * k), False,
               lambda m, n, k: 32 * row_words(27 + hg_steps(k)) * m + 32 * row_words(n),
               lambda m, n, k: 4 * hg_steps(k) / (n * k) + 4 / (m * k), _no_dup, True,
               "0 (exact while |A·Bᵀ| < 2^22)", f"{premises_m0}; {coins}; rc-hg-distinct (open); G drawn after the "
               "weights' registration", "chosen weights cancel nothing; chosen identical or zero rows lose only final "
               "words with equal offsets; chosen out-of-range rows force the fallback write"),
        Design("rc_hg_ffma", "rc_hg's code, odd coins and G, step S cancelling the code against C_j and the walk "
               "against G_j; only c_j is folded, in the FFMA: no range limit below 2^24 and no row check",
               lambda k: H32 * hw_steps(k) / k,
               lambda m, n, k: H32 * (row_bit_writes(hw_steps(k)) + 1 + popc_writes(hw_steps(k)) + 3) / (n * k),
               lambda m, n, k: 2 * H32 / (m * k), lambda m, n, k: H32 * hw_steps(k) / (n * k),
               lambda m, n, k: H32 / (m * k), lambda m, n, k: H32 / (m * k), False,
               lambda m, n, k: 32 * row_words(27 + hw_steps(k)) * m + 32 * row_words(n),
               lambda m, n, k: 4 * hw_steps(k) / (n * k) + 4 / (m * k), _final_dup("final_col_dup_share", hw_steps),
               False, "0 (exact while |A·Bᵀ| < 2^24)", f"{premises_m0}; {coins}; rc-hg-distinct (open); G drawn "
               "after the weights' registration",
               "final word A·Bᵀ + c_j: lost on equal outputs of a column, and on chosen identical rows"),
        Design("ssn_1l_hw", "the served lead hardened against the coin branch (premise_ledger.py's ssn_1l_hw): "
               "a1 ∈ [0,15]⁴ against the code, a_t = 1 + 2b_t "
               "against public column signs; noise served", lambda k: H32 * s1(k) / k,
               lambda m, n, k: H32 * ns.ssn_1l_forming_writes(s1(k)) / (n * k), _z,
               lambda m, n, k: H32 * s1(k) / (n * k), _z, _z, False,
               lambda m, n, k: 32 * w1(k) * m, lambda m, n, k: 4 * s1(k) / (n * k), _no_dup, False,
               "served noise std 41–43 units; ≤ ~+0.2% (bracketed by a measured run, estimate)",
               f"{premises_m0}; {coins}; ssn-hw-distinct (open)", "reference: the served lead hardened against the "
               "coin branch; chosen weights cancel its steps through the public signs", exact=False),
        Design("ssn_1l_hg", "ssn_1l_hw with its step signs drawn from coins once the weights are registered", lambda k: H32 * s1(k) / k,
               lambda m, n, k: H32 * ns.ssn_1l_forming_writes(s1(k)) / (n * k), _z,
               lambda m, n, k: H32 * s1(k) / (n * k), _z, _z, False,
               lambda m, n, k: 32 * w1(k) * m, lambda m, n, k: 4 * s1(k) / (n * k), _no_dup, False,
               "as ssn_1l_hw", f"{premises_m0}; {coins}; ssn-hw-distinct (open); signs drawn after the weights' "
               "registration", "reference: the served lead hardened against the coin branch and chosen weights",
               exact=False),
        Design("rc_hybrid", "rc_magic on rows whose Cauchy–Schwarz bound is below 2^22, rc_ffma on the rest",
               lambda k: H32 * s_rcf(k) / k,
               lambda m, n, k: H32 * (row_bit_writes(s_rcf(k)) + 1 + lanes_writes(s_rcf(k))) / (n * k),
               lambda m, n, k: 2 * H32 / (m * k), lambda m, n, k: H32 * s_rcf(k) / (n * k), lambda m, n, k: H32 / (m * k),
               lambda m, n, k: H32 * popc_writes(s_rcf(k)) / (n * k) + 2 * H32 / (m * k), False,
               lambda m, n, k: 32 * row_words(s_rcf(k)) * m + 32 * row_words(n),
               lambda m, n, k: 4 * s_rcf(k) / (n * k) + 4 / (m * k), "hybrid", False,
               "0 (exact)", f"{premises_m0}; {coins}", "rc_ffma's loss only on rows past the bound (estimate: independent)"),
    ]
    return [_hybrid(d) if d.dup == "hybrid" else d for d in out] + [
        Design("rc_hw_mix", "rc_hw on linears fed by an RMSNorm (a free per-row range bound), rc_hw_ffma on the rest",
               *([lambda *a: float("nan")] * 3), *([_z] * 3), False, _z, _z, _no_dup, True, "0 (exact)",
               f"{premises_m0}; {coins}; rc-hw-distinct (open)", "each linear takes the cheaper path; chosen weights "
               "cancel its steps"),
        Design("rc_hg_mix", "rc_hg on linears fed by an RMSNorm, rc_hg_ffma on the rest",
               *([lambda *a: float("nan")] * 3), *([_z] * 3), False, _z, _z, _no_dup, True, "0 (exact)",
               f"{premises_m0}; {coins}; rc-hg-distinct (open); G drawn after the weights' registration",
               "the exact lead: each linear takes the cheaper path")]


def _per2_words(k: int) -> int:
    return k // 32 + ns.per2_limbs(k)


#: design → (coin-branch model, its k32 steps); `rc_hybrid` is the worse of its two paths
BRANCH = {
    "ssn_1l": ("ssn_1l", ns.steps1l), "ssn_1l_mov": ("ssn_1l", ns.steps1l), "ssn_pd": ("ssn_1l", ns.steps1l),
    "ssn_xd": ("ssn_1l", ns.steps1l), "ssn_xr": ("ssn_xr", s_xr), "ssn_x": ("ssn_x", s_x),
    "ssn_1l_hw": ("ssn_1l_hw", ns.steps1l), "ssn_1l_hg": ("ssn_1l_hg", ns.steps1l), "rc_magic": ("rc_magic", s_rc),
    "rc_ffma": ("rc_ffma", s_rcf),
    "rc_hybrid": ("rc_hybrid", s_rcf), "rc_hw": ("rc_hw", hw_steps), "rc_hw_ffma": ("rc_hw_ffma", hw_steps),
    "rc_hg": ("rc_hg", hg_steps), "rc_hg_ffma": ("rc_hg_ffma", hw_steps),
    "ncp3": ("dense", lambda k: 3 * k // 32), "os2": ("dense", lambda k: 2 * k // 32), "per2": ("dense", _per2_words),
    "lowrank1": ("dense", _per2_words), "pair": ("none", lambda k: 2 * k // 32), "plain": ("plain", lambda k: k // 32),
}
#: linears whose input leaves an RMSNorm: ‖q_i‖ ≤ 127·max|g|·√K / max|x_i| + √K/2 from stock's own row max, so the
#: magic's per-row range check is free; elsewhere it needs ‖q_i‖ (K/4 IDP4A a row)
POST_NORM = {"qkv", "gate_up"}
#: per linear, the path a mixed design takes
MIX = {"rc_hw_mix": lambda lin: "rc_hw" if lin in POST_NORM else "rc_hw_ffma",
       "rc_hg_mix": lambda lin: "rc_hg" if lin in POST_NORM else "rc_hg_ffma"}
MAGIC_DESIGNS = {"rc_magic", "rc_hybrid", "rc_hw", "rc_hg", "ssn_pd", "ssn_xd", "ssn_xr"}


def _hybrid(d: Design) -> Design:
    def dup(model: str, m: int, k: int, lin: str) -> float:
        if model == "M1u":
            return 0.0
        u, c = exact_share("cs_unsafe_rows_share", m, lin), exact_share("final_col_dup_share", m, lin)
        return float("nan") if u is None or c is None else u * c / s_rcf(k)
    return Design(**{**d.__dict__, "dup": dup})


def gap_terms(d: Design, m: int, n: int, k: int, model: str, c1: bool, lin: str = "") -> dict:
    a = d.alpha(k)
    cx, cy = (d.forced_x(m, n, k), d.forced_y(m, n, k)) if c1 else (0.0, 0.0)
    final = H32 / k if d.final_free_m0 and model == "M0" else 0.0
    dup = d.dup(model, m, k, lin)
    fb, check, chosen_range = range_terms(d, m, n, k, lin)
    cb = coin_branch(d.name, m, k, model)
    t = {"alpha": a, "forming_x": d.form_x(m, n, k) - cx, "forming_y": d.form_y(m, n, k) - cy,
         "denoise": d.denoise(m, n, k), "fallback": fb, "sign_fix": 0.0, "final_word": final, "dup": a * dup,
         "range_check": check, "coin_branch_natural": a * cb["natural"]}
    t["gap"] = sum(v for key, v in t.items() if key != "alpha")
    t["coin_branch_chosen"] = a * cb["chosen"]
    t["coin_branch_weights"] = a * cb["weights"]
    t["range_chosen"] = chosen_range
    data_terms = t["dup"] + t["coin_branch_natural"] + t["fallback"]
    t["gap_chosen"] = t["gap"] - data_terms + max(data_terms, t["coin_branch_chosen"], t["range_chosen"])
    t["gap_chosen_weights"] = t["gap"] - data_terms + max(data_terms, t["coin_branch_chosen"], t["range_chosen"],
                                                          t["coin_branch_weights"])
    t["coins"] = d.coin_bits(m, n, k) / 8 * ns.COIN_PRICED_BYTE / (m * n * k)
    t["gap_coins_uncredited"] = t["gap"] + t["coins"]
    t["forced_share_of_forming"] = (cx + cy) / max(1e-30, d.form_x(m, n, k) + d.form_y(m, n, k))
    return t


def range_terms(d: Design, m: int, n: int, k: int, lin: str) -> tuple[float, float, float]:
    """The magic's range, for a design that decides per row tile whether |A·Bᵀ| < 2²²: the fallback write on unsafe
    tiles of the served data, the decision's own cost (free from stock's row scale after an RMSNorm, else ‖q_i‖ by
    K/4 IDP4A a row, 8/n) and, on prover-chosen out-of-range rows, the fallback on every row (32/K). A weight matrix
    whose 127·max_j ‖B_j‖₁ is below 2²² in every layer needs none of them."""
    if not d.fallback and d.name not in MAGIC_DESIGNS:
        return 0.0, 0.0, 0.0
    st = EXACT.get("static", {}).get(lin)
    if st and st["static_safe_layers"] == st["layers"]:
        return 0.0, 0.0, 0.0
    field_ = "free_unsafe" if lin in POST_NORM else "cs_unsafe"
    u = exact_share(f"{field_}_row_tiles_share", m, lin)
    if u is None:
        u = exact_share(f"{field_}_rows_share", m, lin) if field_ == "cs_unsafe" else None
    fb = (float("nan") if u is None else H32 / k * u) if d.fallback else 0.0
    return fb, 0.0 if lin in POST_NORM else 8 / n, H32 / k if d.fallback else 0.0


def coin_branch(name: str, m: int, k: int, model: str) -> dict:
    """The coin-branch share (before α) at a call of m rows: the worse of all-identical and all-zero chosen rows
    (`chosen`, the requests chosen and the weights public), chosen weights cancelling the step lanes' public weights
    (`weights`: a row's own steps merge, so at most 512 rows measure it), and the served data's zero and identical rows
    (none under M1u's uniform grid)."""
    bm, sf = BRANCH.get(name, ("none", lambda k: 1))
    s = sf(k)
    models = ("rc_magic", "rc_ffma") if bm == "rc_hybrid" else (bm,)
    salts = 1 if m * s > 400_000 else 2

    def f(case: str, rows: int) -> float:
        return max(branch_collapse(mo, case, rows, s, k, salts=salts) for mo in models)
    chosen = max(f("identical", m), f("zero", m))
    weights = f("cancel", min(m, 512))
    if model == "M1u":
        return {"chosen": chosen, "weights": weights, "natural": 0.0}
    zr, idr = noise_share("zero_row_share", m), noise_share("identical_row_share", m)
    natural = float("nan") if zr is None or idr is None else zr * f("zero", max(1, round(zr * m))) + idr * f("pair", 2)
    return {"chosen": chosen, "weights": weights, "natural": natural}


_BY_NAME: dict = {}


def weighted(d: Design, shapes: dict, m: int, model: str, c1: bool) -> dict:
    acc, tot = {}, 0.0
    if not _BY_NAME:
        _BY_NAME.update({x.name: x for x in designs()})
    for lin, (n, k) in shapes.items():
        w = m * n * k
        part = _BY_NAME[MIX[d.name](lin)] if d.name in MIX else d
        for key, v in gap_terms(part, m, n, k, model, c1, lin).items():
            acc[key] = acc.get(key, 0.0) + w * v
        tot += w
    return {key: v / tot for key, v in acc.items()}


POINTS = {"T512_64tok": 64, "T512_512tok": 512, "T8192": 8192}


def load_exact(path: str) -> None:
    with open(path) as fh:
        d = json.load(fh)
    EXACT["summary"] = d["summary"]
    by: dict = {}
    for key, st in d["per_linear"].items():
        _, lin, sc = key.split(".")
        e = by.setdefault(lin, {}).setdefault(sc, {"n": 0})
        e["n"] += 1
        for f, base in (("final_row_dup", "outputs"), ("final_col_dup", "outputs"), ("cs_unsafe_rows", "rows"),
                        ("cs_unsafe_row_tiles", "row_tiles"), ("free_unsafe_rows", "rows"),
                        ("free_unsafe_row_tiles", "row_tiles"), ("ge_range", "outputs"), ("rowonly_dup", "words"),
                        ("colonly_dup", "words")):
            if f in st:
                e[f"{f}_share"] = e.get(f"{f}_share", 0.0) + st[f] / st[base]
    for lin, scs in by.items():
        for sc, e in scs.items():
            cnt = e.pop("n")
            for f in list(e):
                e[f] /= cnt
    EXACT["by_linear"] = by
    EXACT["scale_fold_exact_share"] = d["summary"]["scale_fold_exact_share"]
    if "static" in d["summary"]:
        EXACT["static"] = d["summary"]["static"]


def load_noise(path: str) -> None:
    """noise_data.py's summary, with the MAC-weighted share of all-zero rows a scope added from its per-linear counts."""
    ns.load_data(path)
    with open(path) as fh:
        d = json.load(fh)
    summ = {sc: dict(v) for sc, v in d["summary"].items()}
    for sc in summ:
        macs = zr = 0.0
        for key, a in d["per_linear"].items():
            if key.endswith("." + sc):
                w = d["weights"][key.rsplit(".", 1)[0]]
                mac = a["rows"] * w["n"] * w["K"]
                macs += mac
                zr += mac * a["zero_rows"] / a["rows"]
        summ[sc]["zero_row_share"] = zr / macs
    NOISE["summary"] = summ


# ---------------------------------------------------------------- the minimality bound

def minimal_width(k: int, rank_q: int, rank_th: int) -> int:
    """The least stacked width of an exact X·Y = A·Bᵀ whose noise has these ranks (the theorem in the docstring)."""
    return k + rank_q + rank_th


def _rank(mat: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(mat.astype(float)))


def stacked_maps(name: str, k: int, rng: np.random.Generator) -> dict:
    """P, Q (noise into X) and R, T (into Y) of each in-GEMM exact design, as X = A·P + E·Q and Y = R·B + T·F."""
    eye = np.eye(k, dtype=np.int64)
    zero = np.zeros((k, k), dtype=np.int64)
    if name == "ncp3":
        perm = rng.permutation(k)
        r = eye[perm]
        return {"P": np.hstack([eye, eye, -eye]), "Q": np.hstack([eye, r, -(eye + r)]),
                "R": np.vstack([eye, eye, eye]), "T": np.vstack([eye, -r, eye - r])}
    if name == "os2":
        return {"P": np.hstack([eye, zero]), "Q": np.hstack([eye, eye]), "R": np.vstack([eye, -eye]),
                "T": np.zeros((2 * k, 1), dtype=np.int64)}
    if name == "per2":
        g = 4
        tile = np.hstack([np.eye(g, dtype=np.int64)] * (k // g))
        return {"P": np.hstack([eye, np.zeros((k, g), dtype=np.int64)]), "Q": np.hstack([tile, np.eye(g, dtype=np.int64)]),
                "R": np.vstack([eye, -tile]), "T": np.zeros((k + g, 1), dtype=np.int64)}
    raise ValueError(name)


def check_minimality(rng: np.random.Generator) -> dict:
    out = {}
    for name in ("ncp3", "os2", "per2"):
        mp = stacked_maps(name, 8, rng)
        p_, q, r, t = mp["P"], mp["Q"], mp["R"], mp["T"]
        exact = ((p_ @ r == np.eye(8)).all() and not (p_ @ t).any() and not (q @ r).any() and not (q @ t).any())
        w = p_.shape[1]
        out[name] = {"W": w, "bound": minimal_width(8, _rank(q), _rank(t)), "exact": bool(exact)}
    return out


def check_bound_tight(rng: np.random.Generator, trials: int = 100) -> dict:
    """The bound is attained: for random K, rank Q, rank T, an exact map at W = K + rank Q + rank T (R and T spanning
    independent subspaces of ker Q, P solving P·[R|T] = [I|0]) exists, and on it rank [R|T] = K + rank T."""
    built = 0
    for _ in range(trials):
        k, q, t = (int(v) for v in rng.integers(1, 4, 3))
        w = minimal_width(k, q, t)
        qm = rng.integers(-3, 4, (q, w))
        if _rank(qm) < q:
            continue
        ker = np.linalg.svd(qm.astype(float))[2][q:].T
        c = ker @ rng.normal(size=(ker.shape[1], k + t))
        r, tt = c[:, :k], c[:, k:]
        p_ = np.hstack([np.eye(k), np.zeros((k, t))]) @ np.linalg.pinv(c)
        built += bool(np.allclose(p_ @ r, np.eye(k)) and np.allclose(p_ @ tt, 0) and np.allclose(qm @ c, 0)
                      and np.linalg.matrix_rank(c) == k + np.linalg.matrix_rank(tt))
    return {"trials": trials, "built": built}


def check_bound_exhaustive(lim: int = 2) -> dict:
    """K = 1 with noise on both sides at W = 2, one below the bound 3: every integer P, Q, R, T in [−lim, lim]², Q and T
    nonzero, searched for P·R = 1, P·T = 0, Q·R = 0, Q·T = 0. The bound says none exists; at W = 3 ncp3's map does."""
    v = np.array([(a, b) for a in range(-lim, lim + 1) for b in range(-lim, lim + 1)])
    nz = v[(v != 0).any(1)]
    pr = v @ v.T
    ok_rt = (pr[:, :, None] == 1) & (v @ nz.T == 0)[:, None, :]
    qr = (nz @ v.T == 0)
    qt = (nz @ nz.T == 0)
    hits = 0
    for pi in range(len(v)):
        rt = ok_rt[pi]
        if not rt.any():
            continue
        r_idx, t_idx = np.nonzero(rt)
        hits += int((qr[:, r_idx] & qt[:, t_idx]).sum())
    return {"W": 2, "entries": [-lim, lim], "solutions": hits}


# ---------------------------------------------------------------- the epilogue

def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).astype(np.float32)


def _fma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """fp32 FMA: a·b + c is exact in float64 here (24-bit a, b; c an fp32 of the same scale), then one rounding."""
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)


def _bf16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def stock_epilogue(acc: np.ndarray, s_col: np.ndarray, s_row: np.ndarray) -> np.ndarray:
    f = acc.astype(np.float32)
    return _bf16(_f32(_f32(f.astype(np.float64) * s_col[None, :]).astype(np.float64) * s_row[:, None]))


def magic_epilogue(acc: np.ndarray, r: np.ndarray, c: np.ndarray, s_col: np.ndarray, s_row: np.ndarray) -> np.ndarray:
    """IADD3(D, MAGIC_BITS − r_i, −c_j) read as fp32, FFMA(v, s_j, −MAGIC·s_j), FMUL(s_i), bf16."""
    bits = (acc.astype(np.int64) + (MAGIC_BITS - r)[:, None] - c[None, :]) & 0xFFFFFFFF
    v = bits.astype(np.uint32).view(np.float32)
    u = _fma(v, np.broadcast_to(s_col[None, :], v.shape), np.broadcast_to(_f32(-MAGIC * s_col)[None, :], v.shape))
    return _bf16(_f32(u.astype(np.float64) * s_row[:, None]))


def ffma_epilogue(acc: np.ndarray, c: np.ndarray, s_col: np.ndarray, s_row: np.ndarray) -> np.ndarray:
    """I2F(D), FFMA(f, s_j, −c_j·s_j), FMUL(s_i), bf16."""
    f = acc.astype(np.float32)
    u = _fma(f, np.broadcast_to(s_col[None, :], f.shape), np.broadcast_to(_f32(-c * s_col)[None, :], f.shape))
    return _bf16(_f32(u.astype(np.float64) * s_row[:, None]))


def fold_exact(s: np.ndarray) -> np.ndarray:
    x = s.astype(np.float64) * MAGIC
    return x.astype(np.float32).astype(np.float64) == x


def round_scale(s: np.ndarray, bits: int = 22) -> np.ndarray:
    m, e = np.frexp(s.astype(np.float64))
    return np.ldexp(np.round(m * 2 ** bits) / 2 ** bits, e).astype(np.float32)


def check_epilogue(rng: np.random.Generator) -> dict:
    m, n = 64, 256
    p = rng.integers(-MAGIC_RANGE + 1, MAGIC_RANGE, (m, n))
    p[0, :8] = [0, 1, -1, MAGIC_RANGE - 1, -MAGIC_RANGE + 1, 2 ** 20, -(2 ** 20), 12345]
    r = rng.integers(0, 266, m)
    c = rng.integers(0, 2, n)
    acc = p + r[:, None] + c[None, :]
    s_raw = _f32(rng.uniform(1e-4, 1e-2, n))
    s_col = round_scale(s_raw)
    s_row = _f32(rng.uniform(1e-3, 1.0, m))
    stock = stock_epilogue(p, s_col, s_row)
    out = {"fold_exact_rounded": bool(fold_exact(s_col).all()),
           "magic_equal": bool((magic_epilogue(acc, r, c, s_col, s_row) == stock).all()),
           "ffma_equal": bool((ffma_epilogue(p + c[None, :], c, s_col, s_row) == stock).all())}
    stock_raw = stock_epilogue(p, s_raw, s_row)
    mag_raw = magic_epilogue(acc, r, c, s_raw, s_row)
    diff = mag_raw != stock_raw
    out["raw_scale_fold_exact_share"] = float(fold_exact(s_raw).mean())
    out["raw_scale_mismatch_share"] = float(diff.mean())
    val = lambda h: (h.astype(np.uint32) << 16).view(np.float32).astype(np.float64)  # noqa: E731
    err = np.abs(val(mag_raw) - val(stock_raw)) / (s_raw[None, :].astype(np.float64) * s_row[:, None])
    rel = np.abs(val(stock_raw)) / (s_raw[None, :].astype(np.float64) * s_row[:, None])
    out["raw_scale_mismatch_max_units"] = float(err.max())
    out["raw_scale_mismatch_max_units_rel_bf16"] = float((err / np.maximum(rel * 2 ** -8, 1)).max())
    big = p.copy()
    big[1, 0] = 2 * MAGIC_RANGE
    out["out_of_range_breaks"] = bool((magic_epilogue(big + r[:, None] + c[None, :], r, c, s_col, s_row)
                                       != stock_epilogue(big, s_col, s_row)).any())
    return out


# ---------------------------------------------------------------- small instances: distinctness on a salt grid

def _rows_bits(h: np.ndarray, s: int) -> np.ndarray:
    """A row's S noise bits, each its own (coin word, bit): bit p of word w is bit 32w + p."""
    return np.stack([(h[..., t // 32] >> (t % 32)) & 1 for t in range(s)], axis=-1)


def sim_rc(name: str, case: str, salts: int = 24, m: int = 4, n: int = 6, seed: int = 1, s: int = 4,
           drop_col: bool = False) -> dict:
    """rc_magic / rc_ffma / ssn_xr / ssn_x on S steps: the committed words, the formed registers, the epilogue's output
    against A·Bᵀ, and distinctness on the salt grid with raw coin words free. `drop_col` removes c_j (a mutation)."""
    rng = np.random.default_rng(seed)
    first = 30 if name.startswith("rc") else 28
    k = first + 31 * (s - 1) - (4 if name in ("ssn_xr", "ssn_x") else 0)
    a, b = ns._data(case, m, n, 160, rng)
    a, b = a[:, :k], b[:k]
    code = ns.ssn_a_w0(n, ns.DATA_PER_STEP)[0]
    xs, ys, raw, formed, rs, cs = [], [], [], [], [], []
    for _ in range(salts):
        hr = rng.integers(0, 2 ** 32, (m, 1), dtype=np.int64)
        hc = rng.integers(0, 2 ** 32, (1,), dtype=np.int64)
        bits = _rows_bits(hr, s + 4)
        cbit = np.array([(hc[0] >> j) & 1 for j in range(n)]) * (0 if drop_col else 1)
        xl, yl = [], []
        lo = 0
        for t in range(s):
            d = (first if t == 0 else 31)
            lanes_x, lanes_y = [], []
            if name.startswith("rc"):
                if t == 0:
                    lanes_x += [bits[:, 0:1], np.ones((m, 1), dtype=np.int64)]
                    lanes_y += [np.ones((1, n), dtype=np.int64), cbit[None, :]]
                elif name == "rc_ffma" and t == s - 1:
                    rsum = bits[:, :s - 1].sum(1, keepdims=True)
                    lanes_x += [-rsum]
                    lanes_y += [np.ones((1, n), dtype=np.int64)]
                else:
                    lanes_x += [bits[:, t:t + 1]]
                    lanes_y += [np.ones((1, n), dtype=np.int64)]
            else:
                if t == 0:
                    lanes_x += [bits[:, s:s + 4]]
                    lanes_y += [code]
                elif t == s - 1:
                    lanes_x += [-bits[:, s:s + 4]]
                    lanes_y += [code]
                    if name == "ssn_x":
                        lanes_x += [-bits[:, 1:s - 1].sum(1, keepdims=True)]
                        lanes_y += [np.ones((1, n), dtype=np.int64)]
                    else:
                        lanes_x += [bits[:, t:t + 1]]
                        lanes_y += [np.ones((1, n), dtype=np.int64)]
                else:
                    lanes_x += [bits[:, t:t + 1]]
                    lanes_y += [np.ones((1, n), dtype=np.int64)]
            lx = np.concatenate(lanes_x, axis=1)
            d = 32 - lx.shape[1]
            xl += [a[:, lo:lo + d], lx]
            yl += [b[lo:lo + d], np.concatenate(lanes_y, axis=0)]
            lo += d
        assert lo == k, (lo, k)
        x, y = np.concatenate(xl, axis=1), np.concatenate(yl, axis=0)
        xs.append(x)
        ys.append(y)
        raw.append(np.concatenate([hr.ravel(), hc]))
        formed.append(np.concatenate([ns._pack(x[:, 32 * t + 28:32 * t + 32]).ravel() for t in range(s)]
                                     + ([ns._pack(y[28:32].T).ravel()] if name.startswith("rc") else [])))
        if name == "rc_magic":
            rs.append(bits[:, :s].sum(1))
        elif name == "ssn_xr":
            rs.append(bits[:, 1:s].sum(1))
        else:
            rs.append(np.zeros(m, dtype=np.int64))
        cs.append(cbit if name.startswith("rc") else np.zeros(n, dtype=np.int64))
    x, y = np.stack(xs), np.stack(ys)
    words = ns._words(x, y)
    out = ns._distinct([words, np.stack(formed)], [np.stack(raw)])
    final = words[:, :, :, -1]
    p = a @ b
    resid = final - np.stack(rs)[:, :, None] - np.stack(cs)[:, None, :]
    out["exact_after_epilogue"] = bool((resid == p[None]).all())
    out["final_free"] = bool((final == p[None]).all())
    return out


def sim_hw(case: str, salts: int = 24, m: int = 4, n: int = 6, seed: int = 1, s: int = 4, e: int = 2,
           variant: str = "hw") -> dict:
    """rc_hw (or rc_hg, `variant="hg"`: step lanes against G_j, the walk cancelled in step S) on S steps with e offset
    lanes: its committed words on the salt grid (raw coin words free) and the magic epilogue's output against A·Bᵀ."""
    rng = np.random.default_rng(seed)
    hg = variant == "hg"
    k = 27 + 31 * (s - 2) + 26 - e - (HG_FIXED - HW_FIXED if hg else 0)
    a, b = ns._data(case, m, n, 160, rng)
    a, b = a[:, :k], b[:k]
    code = ns.ssn_a_w0(n, ns.DATA_PER_STEP)[0]
    kappa = np.array(HW_KAPPA)[:, None]
    g = (hg_g(n) if hg else np.ones(n, dtype=np.int64))[None, :]
    xs, ys, raw, rs, cs = [], [], [], [], []
    for _ in range(salts):
        hr = rng.integers(0, 2 ** 32, (m,), dtype=np.int64)
        hc = rng.integers(0, 2 ** 32, (1,), dtype=np.int64)
        a1 = np.stack([(hr >> (8 * q)) & 0x7F for q in range(4)], axis=1)
        at = np.stack([np.zeros(m, dtype=np.int64)] + [1 + 2 * ((hr >> (8 * t + 7)) & 1) for t in range(s - 1)], axis=1)
        walk = at.sum(1)[:, None]
        cbit = np.array([(hc[0] >> j) & 1 for j in range(n)])
        xl, yl, lo = [], [], 0
        for t in range(s):
            if t == 0:
                lx, ly = [a1, np.ones((m, 1), dtype=np.int64)], [code, cbit[None, :]]
            elif t < s - 1:
                lx, ly = [at[:, t:t + 1]], [g]
            else:
                lx = [at[:, t:t + 1], a1, a1[:, 1:2]] + [a1[:, 2:3]] * e
                ly = [g, kappa - code, np.full((1, n), 7)] + [np.full((1, n), 127)] * e
                if hg:
                    lx += [walk % HG_BASE, walk // HG_BASE]
                    ly += [-g, -HG_BASE * g]
            lx, ly = np.concatenate(lx, axis=1), np.concatenate(ly, axis=0)
            assert np.abs(lx).max() <= 127 and np.abs(ly).max() <= 127
            d = 32 - lx.shape[1]
            xl += [a[:, lo:lo + d], lx]
            yl += [b[lo:lo + d], ly]
            lo += d
        assert lo == k, (lo, k)
        xs.append(np.concatenate(xl, axis=1))
        ys.append(np.concatenate(yl, axis=0))
        raw.append(np.concatenate([hr, hc]))
        w = np.array(hw_weights_e(e))
        rs.append(a1 @ w + (0 if hg else at.sum(1)))
        cs.append(cbit)
    words = ns._words(np.stack(xs), np.stack(ys))
    out = ns._distinct([words], [np.stack(raw)])
    resid = words[:, :, :, -1] - np.stack(rs)[:, :, None] - np.stack(cs)[:, None, :]
    out["exact_after_epilogue"] = bool((resid == (a @ b)[None]).all())
    return out


def sim_plain_m1(case: str, grids: int = 24, m: int = 4, n: int = 6, seed: int = 1) -> dict:
    """The plain GEMM's words on a grid over A that keeps the case's structure (M1s); `random` is M1u."""
    rng = np.random.default_rng(seed)
    _, b = ns._data(case, m, n, 128, rng)
    xs = []
    for _ in range(grids):
        a, _ = ns._data(case, m, n, 128, rng)
        xs.append(a)
    x = np.stack(xs)
    y = np.broadcast_to(b, (grids, *b.shape))
    return ns._distinct([ns._words(x, y)], [])


# ---------------------------------------------------------------- the coin branch (prover-chosen rows)
#
# A W1 program may branch on the coins at no cost: per salt, it computes each class of equal row-steps (a row's n words
# after step t) once and copies the rest. A design's loss is the share of row-steps equal to an earlier one at the same
# salt, on prover-chosen rows (all identical, all zero) and on the served data's zero and identical rows.

BRANCH_N = 16
BRANCH_N_FULL = 2048
BRANCH_CASES = ("identical", "zero", "pair", "random", "cancel")
_SIGNS_SEED = 12345
_KEY = np.random.default_rng(99).integers(1, 2 ** 62, (BRANCH_N, 2), dtype=np.int64)
_CODE: dict = {}
_COLLAPSE: dict = {}


def branch_code(n: int = BRANCH_N, n_full: int = BRANCH_N_FULL) -> np.ndarray:
    """ssn_1l's 4-lane column code at n columns spread over a layer of n_full, so every digit varies as in the layer."""
    if n_full not in _CODE:
        _CODE[n_full] = ns.ssn_a_w0(n_full, ns.DATA_PER_STEP)[0]
    return _CODE[n_full][:, np.linspace(0, n_full - 1, n).round().astype(int)]


def branch_noise(model: str, m: int, s: int, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """The noise in every row-step's words at one salt (m × S × n); the data part is added by `branch_collapse`."""
    code = branch_code(n)
    if model in ("ssn_1l", "ssn_xr", "ssn_x"):
        a1 = rng.integers(0, 2, (m, 4))
        at = rng.integers(0, 2, (m, s))
        at[:, 0] = 0
        nz = (a1 @ code)[:, None, :] + at.cumsum(1)[:, :, None]
        if model == "ssn_xr":
            nz[:, -1] -= a1 @ code
        elif model == "ssn_x":
            nz[:, -1] = 0
        return nz
    if model in ("ssn_1l_hw", "ssn_1l_hg"):
        a1 = rng.integers(0, 16, (m, 4))
        at = 1 + 2 * rng.integers(0, 2, (m, s))
        at[:, 0] = 0
        seed = _SIGNS_SEED if model == "ssn_1l_hw" else _SIGNS_SEED + 1
        signs = np.random.default_rng(seed).choice(np.array([-1, 1]), size=(s, n))
        return (a1 @ code)[:, None, :] + (at[:, :, None] * signs[None]).cumsum(1)
    if model.startswith("rc_hg"):
        a1 = rng.integers(0, 128, (m, 4))
        at = 1 + 2 * rng.integers(0, 2, (m, s))
        at[:, 0] = 0
        c = rng.integers(0, 2, n)
        g = np.ones(n, dtype=np.int64) if model == "rc_hg_public" else hg_g(n)
        nz = (a1 @ code)[:, None, :] + at.cumsum(1)[:, :, None] * g + c
        nz[:, -1] = c if model == "rc_hg_ffma" else (a1 @ np.array(hw_weights(k, HG_FIXED)))[:, None] + c
        return nz
    if model in ("rc_magic", "rc_ffma"):
        c = rng.integers(0, 2, n)
        nz = np.broadcast_to(rng.integers(0, 2, (m, s)).cumsum(1)[:, :, None] + c, (m, s, n)).copy()
        if model == "rc_ffma":
            nz[:, -1] = c
        return nz
    if model.startswith("rc_hw"):
        a1 = rng.integers(0, 16 if model == "rc_hw_nibble" else 128, (m, 4))
        b = rng.integers(0, 2, (m, s))
        at = b if model == "rc_hw_even" else 1 + 2 * b
        at[:, 0] = 0
        c = rng.integers(0, 2, n)
        w = np.zeros(4, dtype=np.int64) if model == "rc_hw_nooffset" else np.array(hw_weights(k))
        nz = (a1 @ code)[:, None, :] + at.cumsum(1)[:, :, None] + c
        nz[:, -1] = c if model == "rc_hw_ffma" else (a1 @ w + at.sum(1))[:, None] + c
        return nz
    if model == "dense":
        nz = rng.integers(-2 ** 40, 2 ** 40, (m, s, n))
        nz[:, -1] = 0
        return nz
    raise ValueError(model)


def step_weights(model: str, s: int, n: int) -> np.ndarray:
    """The column weights of a design's step lanes as chosen weights can copy them (S × n): public signs for
    ssn_1l_hw, 1 for the rest. ssn_1l_hg's signs and rc_hg's G are drawn after the weights are registered, so the data
    copies the public guess (ssn_1l_hw's signs, 1) and misses."""
    if model in ("ssn_1l_hw", "ssn_1l_hg"):
        return np.random.default_rng(_SIGNS_SEED).choice(np.array([-1, 1]), size=(s, n))
    return np.ones((s, n), dtype=np.int64)


def _data_dots(case: str, m: int, s: int, n: int, rng: np.random.Generator, sd: float = 300.0,
               public: np.ndarray | None = None) -> np.ndarray:
    """Data partial sums D_{≤t} (m × S × n). `cancel`: distinct rows whose every later step's dot is −1 times the
    step lanes' public weight (chosen weights: one lane of B equal to −weight in each step's block, A's lane 1), so a
    step with a_t = 1 adds nothing."""
    if case == "zero":
        return np.zeros((m, s, n), dtype=np.int64)
    if case == "cancel":
        d = np.zeros((m, s, n), dtype=np.int64)
        d[:, 0] = np.rint(rng.normal(0, sd, (m, n))).astype(np.int64)
        d[:, 1:] = -public[None, 1:]
        return d.cumsum(axis=1)
    if case == "identical":
        d = np.broadcast_to(np.rint(rng.normal(0, sd, (1, s, n))).astype(np.int64), (m, s, n))
    elif case == "pair":
        d = np.repeat(np.rint(rng.normal(0, sd, ((m + 1) // 2, s, n))).astype(np.int64), 2, axis=0)[:m]
    else:
        d = np.rint(rng.normal(0, sd, (m, s, n))).astype(np.int64)
    return d.cumsum(axis=1)


def _unique_rows(words: np.ndarray) -> int:
    """Distinct rows by two 64-bit random linear keys (wrapping); a false merge has probability about 2⁻¹²⁰."""
    key = words @ _KEY[:words.shape[1]]
    return int(np.unique(key.view([("a", np.int64), ("b", np.int64)]).ravel()).shape[0])


def branch_collapse(model: str, case: str, m: int, s: int, k: int, n: int = BRANCH_N, salts: int = 2,
                    seed: int = 7) -> float:
    """Expected share of row-steps equal to an earlier row-step at the same salt (a call of m rows, S steps).

    `plain` (no noise) and `dense` (fresh full-entropy noise, the final word A·Bᵀ: ncp3, os2, per2, lowrank1) are
    exact in closed form; `dense_sim` simulates `dense`. `none` is a design with no data-only word (pair)."""
    key = (model, case, m, s, k, n, salts, seed)
    if key in _COLLAPSE:
        return _COLLAPSE[key]
    if model == "none" or case in ("random", "cancel") and model in ("dense", "plain"):
        v = 0.0
    elif model == "plain":
        v = {"zero": 1 - 1 / (m * s), "identical": 1 - 1 / m, "pair": (m // 2) / m}[case]
    elif model == "dense":
        v = {"zero": (m - 1) / (m * s), "identical": (m - 1) / (m * s), "pair": (m // 2) / (m * s)}[case]
    else:
        rng = np.random.default_rng(seed)
        d = _data_dots(case, m, s, n, rng, public=step_weights(model, s, n))
        shares = []
        for _ in range(salts):
            words = (d + branch_noise("dense" if model == "dense_sim" else model, m, s, n, k, rng)).reshape(m * s, n)
            shares.append(1 - _unique_rows(words) / (m * s))
        v = float(np.mean(shares))
    _COLLAPSE[key] = v
    return v


def hw_final_bound(m: int, k: int, fixed: int = HW_FIXED) -> float:
    """rc_hw (rc_hg: `HG_FIXED`) on m chosen-equal rows: the expected share of row-steps lost to final words with equal
    offsets, at most C(m, 2) / Q of the m·S row-steps (Q = hw_offset_q); its other row-steps differ through the 28-bit
    code."""
    return (m - 1) / (2 * hw_offset_q(k, fixed) * hw_steps(k, fixed))


def check() -> list[dict]:
    res = []
    rng = np.random.default_rng(7)

    def ok(name: str, cond: bool, detail) -> None:
        res.append({"check": name, "pass": bool(cond), "detail": detail})

    mn = check_minimality(rng)
    ok("ncp3, os2 and per2 cancel exactly and meet W = K + rank Q + rank T (ncp3 3K, os2 2K, per2 K + g)",
       all(v["exact"] and v["W"] == v["bound"] for v in mn.values()), mn)
    rb = check_bound_tight(rng)
    ok("the width bound is attained for random ranks (exact maps at W = K + rank Q + rank T)", rb["built"] >= 90, rb)
    ex = check_bound_exhaustive()
    ok("no exact map with two-sided noise at K = 1, W = 2 (exhaustive over [-2, 2])", ex["solutions"] == 0, ex)
    ep = check_epilogue(rng)
    ok("magic IADD3 + FFMA epilogue equals stock's bf16 output bit for bit (scales with 22-bit significands)",
       ep["fold_exact_rounded"] and ep["magic_equal"], ep)
    ok("column-only fold (I2F, FFMA with −c_j·s_j) equals stock's output bit for bit", ep["ffma_equal"], ep)
    ok("an output at 2^22 breaks the magic epilogue (mutation: the range bound is needed)", ep["out_of_range_breaks"], ep)
    ok("unrounded fp32 scales: mismatches are rare and within max(2 accumulator units, 1 bf16 ulp) of stock",
       ep["raw_scale_mismatch_share"] < 0.01 and ep["raw_scale_mismatch_max_units_rel_bf16"] <= 2, ep)
    for name in ("rc_magic", "rc_ffma", "ssn_xr", "ssn_x"):
        r = sim_rc(name, "random")
        ok(f"{name}: the epilogue's output is A·Bᵀ", r["exact_after_epilogue"], r)
    rcm = {c: sim_rc("rc_magic", c)["bad_share"] for c in ns.CASES}
    ok("rc_magic: every word distinct and non-free on every data case (M0)", all(v == 0 for v in rcm.values()), rcm)
    rcf = {c: sim_rc("rc_ffma", c)["bad_share"] for c in ns.CASES}
    ok("rc_ffma: robust except the final word on identical rows (a column's equal outputs)",
       rcf["identical_rows"] > 0 and all(v == 0 for c, v in rcf.items() if c != "identical_rows"), rcf)
    xr = {c: sim_rc("ssn_xr", c)["bad_share"] for c in ns.CASES}
    ok("ssn_xr: loses final words on identical columns and zero rows (a row's equal outputs)",
       xr["identical_cols"] > 0 and xr["zero_row"] > 0 and xr["random"] == 0, xr)
    sx = sim_rc("ssn_x", "random")
    ok("ssn_x: the final word is A·Bᵀ and the only free word (M0)", sx["final_free"] and sx["free"] == 4 * 6
       and sx["dup"] == 0, sx)
    dc = sim_rc("rc_magic", "identical_cols", drop_col=True)
    ok("rc_magic without its column bit loses words on identical columns (mutation)", dc["bad_share"] > 0, dc)
    pu = sim_plain_m1("random")
    ok("plain GEMM on a uniform-A grid (M1u): every word distinct (noise redundant under M1)", pu["bad_share"] == 0, pu)
    pz = {c: sim_plain_m1(c)["bad_share"] for c in ("zero_blocks", "identical_rows", "zero_row")}
    ok("plain GEMM on a structured grid (M1s) loses words on zero blocks, identical rows and zero rows",
       all(v > 0 for v in pz.values()), pz)
    hw = {c: sim_hw(c) for c in ns.CASES}
    ok("rc_hw: the magic epilogue's output is A·Bᵀ on every data case", all(v["exact_after_epilogue"] for v in hw.values()),
       {c: v["exact_after_epilogue"] for c, v in hw.items()})
    ok("rc_hw: every word distinct and non-free on every data case (M0 on the salt grid)",
       all(v["bad_share"] == 0 for v in hw.values()), {c: v["bad_share"] for c, v in hw.items()})
    ok("rc_hw keeps ssn_1l's steps at K 2,048, 7,168 and 8,192 (spare lanes 20, 15, 14 for the offset)",
       all(hw_steps(k) == ns.steps1l(k) for k in (2048, 7168, 8192)), {k: (hw_steps(k), hw_extra(k)) for k in
                                                                     (2048, 7168, 8192)})
    rng_q = np.random.default_rng(5)
    a1 = rng_q.integers(0, 128, (400_000, 4))
    off = a1 @ np.array(hw_weights(2048)) + 2 * rng_q.binomial(hw_steps(2048) - 1, 0.5, 400_000)
    _, cnt = np.unique(off, return_counts=True)
    q_mc = len(off) * (len(off) - 1) / (cnt * (cnt - 1)).sum()
    ok("rc_hw's offset Q (FFT) agrees with a Monte Carlo pair-collision estimate within 15%",
       abs(q_mc / hw_offset_q(2048) - 1) < 0.15, {"fft": hw_offset_q(2048), "mc": q_mc})
    br = {mo: {c: branch_collapse(mo, c, 512, 67, 2048) for c in ("identical", "zero", "random")}
          for mo in ("rc_magic", "rc_ffma", "ssn_xr", "rc_hw", "rc_hw_ffma", "ssn_1l_hw")}
    ok("coin branch: rc_magic, rc_ffma and ssn_xr collapse on chosen zero rows (≥ 50% of row-steps), as ssn_1l does",
       all(br[mo]["zero"] >= 0.5 for mo in ("rc_magic", "rc_ffma", "ssn_xr")), br)
    ok("coin branch: rc_hw loses on chosen rows at most 2× its final-word bound + 10⁻⁴, and nothing on random rows",
       max(br["rc_hw"]["identical"], br["rc_hw"]["zero"]) <= 2 * hw_final_bound(512, 2048) + 1e-4
       and br["rc_hw"]["random"] == 0, {"rc_hw": br["rc_hw"], "bound": hw_final_bound(512, 2048)})
    ok("coin branch: rc_hw_ffma loses exactly the final row-steps of chosen identical rows ((m−1)/(m·S))",
       abs(br["rc_hw_ffma"]["identical"] - 511 / (512 * 67)) < 1e-4, br["rc_hw_ffma"])
    big = {c: branch_collapse("rc_hw", c, 8192, 67, 2048, salts=1) for c in ("identical", "zero")}
    ok("coin branch at 8,192 rows: rc_hw within 2× its final-word bound + 10⁻⁴",
       max(big.values()) <= 2 * hw_final_bound(8192, 2048) + 1e-4, {**big, "bound": hw_final_bound(8192, 2048)})
    ds = {c: (branch_collapse("dense_sim", c, 64, 12, 384), branch_collapse("dense", c, 64, 12, 384))
          for c in ("identical", "zero", "pair")}
    ok("coin branch: the dense designs' closed form (final word A·Bᵀ only) matches its simulation",
       all(abs(a - b) < 1e-9 for a, b in ds.values()), ds)
    mut = {"even": branch_collapse("rc_hw_even", "zero", 512, 67, 2048),
           "nooffset": branch_collapse("rc_hw_nooffset", "zero", 512, 67, 2048),
           "nibble_8192": branch_collapse("rc_hw_nibble", "identical", 8192, 67, 2048, salts=1)}
    ok("mutations caught: even step coins (zero rows' steps merge), no row offset (final words merge), 4-bit code "
       "(identical rows merge at 8,192 rows)", mut["even"] > 0.1 and mut["nooffset"] > 10 * hw_final_bound(512, 2048)
       and mut["nibble_8192"] > 3 * big["identical"], mut)
    hg = {c: sim_hw(c, variant="hg") for c in ns.CASES}
    ok("rc_hg: the magic epilogue's output is A·Bᵀ and every word is distinct and non-free on every data case",
       all(v["exact_after_epilogue"] and v["bad_share"] == 0 for v in hg.values()),
       {c: (v["exact_after_epilogue"], v["bad_share"]) for c, v in hg.items()})
    ok("rc_hg keeps ssn_1l's steps at K 2,048, 7,168 and 8,192, and its walk digits fit int8 (d1 ≤ 3(S−1)/63)",
       all(hg_steps(k) == ns.steps1l(k) and 3 * (hg_steps(k) - 1) // HG_BASE <= LANE_MAX and hw_extra(k, HG_FIXED) >= 0
           for k in (2048, 7168, 8192)), {k: (hg_steps(k), hw_extra(k, HG_FIXED)) for k in (2048, 7168, 8192)})
    cw = {mo: branch_collapse(mo, "cancel", 512, 67, 2048)
          for mo in ("ssn_1l", "ssn_1l_hw", "rc_magic", "rc_hw", "rc_hw_ffma", "rc_hg", "rc_hg_ffma", "ssn_1l_hg",
                     "rc_hg_public")}
    ok("chosen weights cancelling the public step weights: ssn_1l, ssn_1l_hw, rc_magic, rc_hw and rc_hw_ffma lose ≥ 40% "
       "of row-steps; rc_hg, rc_hg_ffma and ssn_1l_hg (weights drawn after registration) lose none; a public G "
       "(mutation) loses ≥ 40%", all(cw[mo] >= 0.4 for mo in ("ssn_1l", "ssn_1l_hw", "rc_magic", "rc_hw", "rc_hw_ffma",
                                                             "rc_hg_public"))
       and all(cw[mo] == 0 for mo in ("rc_hg", "rc_hg_ffma", "ssn_1l_hg")), cw)
    hgb = {c: branch_collapse("rc_hg", c, 512, 67, 2048) for c in ("identical", "zero", "random")}
    ok("coin branch: rc_hg loses on chosen rows at most 2× its final-word bound + 10⁻⁴, and nothing on random rows",
       max(hgb["identical"], hgb["zero"]) <= 2 * hw_final_bound(512, 2048, HG_FIXED) + 1e-4 and hgb["random"] == 0,
       {**hgb, "bound": hw_final_bound(512, 2048, HG_FIXED)})
    ok("rc steps: K 2,048 and 8,192 take as many steps as ssn_1l (67, 265)",
       (s_rc(2048), s_rc(8192)) == (ns.steps1l(2048), ns.steps1l(8192)), (s_rc(2048), s_rc(8192)))
    ok("in-GEMM cancellation lanes fit the last step's spare lanes at Llama's and V4's K",
       all(s_x(k) == ns.steps1l(k) and s_rcf(k) == s_rc(k) for k in (2048, 7168, 8192)),
       {k: (s_x(k), ns.steps1l(k), s_rcf(k), s_rc(k)) for k in (2048, 7168, 8192)})
    return res


def report() -> dict:
    ds = designs()
    rows = []
    for d in ds:
        for model in MODELS:
            for c1 in (False, True):
                e = {"design": d.name, "model": model, "c1": c1}
                for point, m in POINTS.items():
                    e[f"llama_{point}"] = weighted(d, ns.LLAMA, m, model, c1)
                    e[f"v4dense_{point}"] = weighted(d, ns.V4_DENSE, m, model, c1)
                for point in ns.STEP:
                    t2 = ns.t2_estimate(d, point, c1)
                    e[f"t2_{point}"], e[f"t2_overlapped_{point}"] = t2["t2"], t2["t2_overlapped"]
                rows.append(e)
    per_dot = {d.name: {lin: {"alpha": d.alpha(k), "extra_macs": (d.alpha(k) - 1) * k,
                              "forming_x_writes_row": d.form_x(1, n, k) * n * k / H32,
                              "forming_y_writes_col": d.form_y(1, n, k) * k / H32,
                              "denoise_writes_row_call": d.denoise(10 ** 12, n, k) * n * k / H32,
                              "coin_bits_row": d.coin_bits(1, 0, k), "coin_bits_col_call": d.coin_bits(0, n, k) / n}
                        for lin, (n, k) in ns.LLAMA.items()} for d in ds}
    return {"designs": [{k: v for k, v in d.__dict__.items() if not callable(v)} for d in ds], "rows": rows,
            "per_dot_product": per_dot, "checks": check(),
            "data": {"exact": EXACT or None, "noise_summary": NOISE.get("summary")},
            "params": {"H32": H32, "magic_bits": MAGIC_BITS, "magic_range": MAGIC_RANGE, "points": POINTS,
                       "llama": ns.LLAMA, "v4_dense": ns.V4_DENSE, "coin_priced_byte": ns.COIN_PRICED_BYTE}}


def best(rep: dict) -> dict:
    """At each point and under each data model, the exact designs whose gap (coins credited, and uncredited) is within
    T1's 1%, best first."""
    exact = {x["name"] for x in rep["designs"] if x["exact"]}
    out = {}
    for point in POINTS:
        for basis in ("gap", "gap_coins_uncredited", "gap_chosen", "gap_chosen_weights"):
            for shape in ("llama", "v4dense"):
                for model in MODELS:
                    cands = sorted((r[f"{shape}_{point}"][basis], r["design"], r["c1"]) for r in rep["rows"]
                                   if r["design"] in exact and r["model"] == model
                                   and not math.isnan(r[f"{shape}_{point}"][basis]))
                    out[f"{shape}_{point}_{model}_{basis}"] = [{"gap": g, "design": n, "c1": c}
                                                               for g, n, c in cands if g <= 0.01][:6]
    return out


def table(rep: dict) -> str:
    hdr = ("design", "model", "C1", "L T512(64)", "L T512(512)", "L T8192", "V4d T512(64)", "V4d T8192",
           "L T512(64) +coins", "L T8192 +coins", "V4d T8192 +coins", "L T512(512) chosen", "L T8192 chosen",
           "V4d T8192 chosen", "L T8192 chosen+W", "alpha L", "T2 8192", "T2 512(64)")
    lines = [" | ".join(hdr)]
    for r in rep["rows"]:
        def f(key: str, basis: str = "gap") -> str:
            v = r[key][basis]
            return "n/a" if math.isnan(v) else f"{100 * v:.3f}%"
        lines.append(" | ".join([r["design"], r["model"], "y" if r["c1"] else "n", f("llama_T512_64tok"),
                                 f("llama_T512_512tok"), f("llama_T8192"), f("v4dense_T512_64tok"), f("v4dense_T8192"),
                                 f("llama_T512_64tok", "gap_coins_uncredited"), f("llama_T8192", "gap_coins_uncredited"),
                                 f("v4dense_T8192", "gap_coins_uncredited"), f("llama_T512_512tok", "gap_chosen"),
                                 f("llama_T8192", "gap_chosen"), f("v4dense_T8192", "gap_chosen"),
                                 f("llama_T8192", "gap_chosen_weights"),
                                 f"{r['llama_T8192']['alpha']:.4f}", f"{r['t2_T8192']:.1f}x",
                                 f"{r['t2_T512_64tok']:.1f}x"]))
    lines.append("")
    for c in rep["checks"]:
        lines.append(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--table", action="store_true", help="print the text table")
    ap.add_argument("--data", help="exact_data.py's JSON (final-word repeats, fallback rows, scale fold)")
    ap.add_argument("--noise-data", help="noise_data.py's JSON (zero registers, identical rows, M1s duplicates)")
    args = ap.parse_args(argv)
    if args.data:
        load_exact(args.data)
    if args.noise_data:
        load_noise(args.noise_data)
    rep = report()
    rep["best"] = best(rep)
    for name in ("rc_magic", "rc_ffma", "rc_hybrid", "rc_hw", "rc_hw_mix", "rc_hg", "rc_hg_mix", "ssn_1l",
                 "ssn_1l_hw", "ssn_1l_hg", "per2"):
        for model in ("M0", "M1u"):
            r = next(r for r in rep["rows"] if r["design"] == name and r["model"] == model and not r["c1"])
            for basis in ("gap", "gap_coins_uncredited", "gap_chosen", "gap_chosen_weights"):
                g = r["llama_T8192"][basis]
                if not math.isnan(g):
                    rep[f"periods_{name}_{model}_noC1_llama_T8192_{basis}"] = ns.periods(g, r["llama_T8192"]["alpha"])
    text = json.dumps(rep, indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    if args.table:
        print(table(rep))
    elif not args.json:
        print(text)
    return 0 if all(c["pass"] for c in rep["checks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
