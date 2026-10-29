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
committed weights can't cancel a whole row-step; ssn_1l_hg does the same for the served lead. At word grain they still
can (`word_cancel`): requests chosen once G is known give a step dot −v on each column class, and the words where
a_t·G_j = v repeat (a quarter of them), which free moves regroup into tiles and skip. What resists is a step
increment the data can't predict: `rc_hgc_ffma` draws G ∈ [−127, 127] ∖ {0} from the call's seed (≤ 1/254, a Y
register a column a step a call to form it), and `rc_hws_ffma` makes each step's row coin a signed coin byte against
the public weight 1 (1/256, 8 coin bits a row-step). An exact design can afford noise that wide because none of it is
served; a served design's T3 can't. Two limits remain for every exact design:

- a final word with no row noise (A·Bᵀ, or A·Bᵀ + c_j) is the same on chosen identical rows, so it is lost: (m−1)/(m·S);
- a final word with row noise needs the magic, whose range a per-row bound must certify: free after an RMSNorm (from
  stock's own row scale), K/4 IDP4A a row elsewhere (8/n), and on chosen out-of-range rows the fallback write (32/K)
  on every row, unless the weight matrix is statically safe (127·max_j ‖B_j‖₁ < 2²²).

`rc_hw_mix` (`rc_hg_mix`) takes rc_hw (rc_hg) where the bound is free and rc_hw_ffma (rc_hg_ffma) elsewhere. Served
noise has neither limit.

`ssn_1l_tag_x` serves the tag lead (`ssn_1l_tag`) exactly with its committed words noisy: its words to step S − 1 are
the lead's, and step S cancels the code (a1 against −C_j) and the tag (κ − T(i) against G_j, leaving κ·G_j) and adds
coin bytes against public weights. Its final word A·Bᵀ + ρ_i + κ·G_j is salted, so the middle-segment skip, which runs
back from a final word the prover knows, has none to start from and each of the S words still costs a write; the
magic removes ρ_i + κ·G_j. Chosen identical or zero rows lose only final words whose offsets meet (`tagx_final_bound`).
`ssn_1l_tag_xf` folds only G_0j, in the FFMA, and loses identical rows' final words; `ssn_1l_tag_mix` takes each
where its range bound is free and the other elsewhere. `ssn_1l_tag_xh` writes no fallback: a row the free bound fails
gets ‖q_i‖₂ (8/n), and a row neither bound certifies writes zeros to step S's added lanes, so its final word and served
output are the lead's. Its words are each row's x or lead words, and chosen out-of-range rows cost the bound rather than
a write an output; `ssn_1l_tag_xh_mix` takes it after an RMSNorm and xf elsewhere.

Data models: M0 (free set: constants, salt-independent functions, raw coin words), M1u (the call's own words on a grid
uniform in A: the final-word lemma, under PeriodInventory and the atomic call) and M1s (M1 on a grid that keeps the
served data's zero blocks and identical rows, priced on noise_data.py's measurement). `--data` reads exact_data.py's
output (final-word repeats, the magic range's fallback rows and static safety, the scale fold's exactness),
`--noise-data` noise_data.py's. Every row carries `gap` (served data), `gap_chosen` (the worst of prover-chosen
identical, zero and out-of-range rows, the weights public) and `gap_chosen_weights` (chosen weights too, and the word
cancel with chosen identical rows' final words).
Output: JSON (`--json`) and a text table (`--table`); exit 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import functools
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


def hgc_g(n: int, rng: np.random.Generator) -> np.ndarray:
    """rc_hgc's step-lane column weights, G ∈ [−127, 127] ∖ {0}, drawn from the call's seed (after A and B are fixed)."""
    g = rng.integers(1, LANE_MAX + 1, n)
    return np.where(rng.integers(0, 2, n) == 1, g, -g)


def hgc_fixed(k: int) -> int:
    """rc_hgc's step-S lanes beyond rc_hw's: the walk w_i = Σ_{t≥2} a_t ≤ 3(S − 1) in unary chunks of at most 127, each
    against −G_j (G is too wide for rc_hg's base-63 digits)."""
    fixed = HG_FIXED
    while (nxt := HW_FIXED + math.ceil(3 * (hw_steps(k, fixed) - 1) / LANE_MAX)) != fixed:
        fixed = nxt
    return fixed


def hgc_steps(k: int) -> int:
    return hw_steps(k, hgc_fixed(k))


def hws_walk_lanes(s: int) -> int:
    """rc_hws's walk lanes in step S: w_i = Σ_{t≥2} a_t with a_t a signed coin byte, as d0 ∈ [−63, 64] against −1 and
    d1 = round(w_i / 127) in chunks of at most 127 against −127."""
    return 1 + math.ceil((math.ceil(128 * (s - 1) / LANE_MAX) + 1) / LANE_MAX)


def hws_fixed(k: int) -> int:
    fixed = HW_FIXED + 2
    while (nxt := HW_FIXED + hws_walk_lanes(hw_steps(k, fixed))) != fixed:
        fixed = nxt
    return fixed


def hws_steps(k: int) -> int:
    return hw_steps(k, hws_fixed(k))


def _hgc_lanes(k: int) -> int:
    return hgc_fixed(k) - HW_FIXED


def _hgc_y_regs(k: int) -> int:
    """Y registers a column rc_hgc forms a call: G merged into steps 2..S−1's register, and step S's G lane beside the
    walk's −G_j lanes."""
    return hgc_steps(k) - 2 + math.ceil((1 + _hgc_lanes(k)) / 4)


#: the tag lead's step 1: 24 data, the row tag T(i) against G_j ∈ [−3, 3]⁴ and the code's 4 bits a1 against C_j
TAG_FIRST = 24
#: ssn_1l_tag_x's step-S lanes beyond the lead's: a1 against −C_j, κ − T(i) against G_j, u0 against 1 and u1 against
#: 127 (ssn_1l_tag_xf: the walk's two digits in place of u0, u1)
TAGX_FIXED = 10
TAGX_KAPPA = (1, 7, 49, 121)
TAGX_EXTRA_MAX = 12


def tag_steps(k: int) -> int:
    return steps(k, TAG_FIRST)


def tagx_steps(k: int) -> int:
    return steps(k, TAG_FIRST, lambda s: TAGX_FIXED)


def tagx_extra(k: int) -> int:
    """Lanes of the coin byte u2 against 127 that fit step S's spare lanes."""
    return min(TAGX_EXTRA_MAX, TAG_FIRST + 31 * (tagx_steps(k) - 1) - TAGX_FIXED - k)


def tag_digits(m: int) -> np.ndarray:
    """The row tag T(i): balanced base-13 digits of z_i = (⌊i/2⌋ + 1)·(−1)^i, injective for m ≤ 28,560 (m × 4)."""
    i = np.arange(m)
    z = (i // 2 + 1) * np.where(i % 2 == 0, 1, -1)
    out = np.zeros((m, 4), dtype=np.int64)
    for e in range(4):
        out[:, e] = (z + 6) % 13 - 6
        z = (z - out[:, e]) // 13
    assert (z == 0).all(), "m too large for a 4-digit base-13 tag"
    return out


def _q(lanes: list[tuple[int, np.ndarray]], walk: np.ndarray | None = None) -> float:
    """1 / Σ p² of Σ w·x over independent lanes (weight w, x uniform on the given values), plus a walk's pmf."""
    size = sum(w * int(np.ptp(v)) for w, v in lanes) + (len(walk) if walk is not None else 1)
    nfft = 1 << size.bit_length()
    spec = np.ones(nfft // 2 + 1, dtype=complex)
    for w, v in lanes:
        lane = np.zeros(nfft)
        lane[w * (v - v.min())] = 1 / len(v)
        spec *= np.fft.rfft(lane)
    if walk is not None:
        spec *= np.fft.rfft(walk, nfft)
    p = np.clip(np.fft.irfft(spec, nfft)[:size], 0, None)
    return float(p.sum() ** 2 / (p ** 2).sum())


def tagx_offset_q(k: int) -> float:
    """The effective number of values of ssn_1l_tag_x's final row offset ρ_i = u0 + 127·u1 + 127e·u2 + w_i (u0 on
    [0, 127], u1 and u2 signed coin bytes, w_i the walk Σ_{t≥2} (1 + a_t)): no data shift makes two rows' offsets
    meet with probability above 1/Q."""
    s, e, byte = tagx_steps(k), tagx_extra(k), np.arange(-128, 128)
    walk = np.array([math.comb(s - 1, i) / 2 ** (s - 1) for i in range(s)])
    return _q([(1, np.arange(128)), (LANE_MAX, byte)] + ([(LANE_MAX * e, byte)] if e else []), walk)


def tagx_col_q() -> float:
    """The effective number of values of ssn_1l_tag_x's column offset κ·G_j (G_j uniform on [−3, 3]⁴)."""
    return _q([(w, np.arange(-3, 4)) for w in TAGX_KAPPA])


def tagx_final_bound(m: int, k: int) -> float:
    """ssn_1l_tag_x on m chosen-equal (identical or zero) rows: the expected share of row-steps lost to final words
    with equal row offsets, at most C(m, 2) / Q of the m·S row-steps; its other row-steps are the lead's."""
    return (m - 1) / (2 * tagx_offset_q(k) * tagx_steps(k))


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
    #: what a row past the magic's range costs: the fallback write ("fallback"), or the lead's step S and served output
    #: once the free and Cauchy–Schwarz bounds both fail ("lead")
    range_mode: str = "fallback"


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
        Design("rc_hgc_ffma", "rc_hg_ffma with G ∈ [−127, 127] ∖ {0} drawn from the call's seed, so after A and B, "
               "merged into each step's Y register every call; the walk cancelled in step S in unary chunks of ≤ 127 "
               "against −G_j", lambda k: H32 * hgc_steps(k) / k,
               lambda m, n, k: H32 * (row_bit_writes(hgc_steps(k)) + 1 + popc_writes(hgc_steps(k))
                                      + 2 * _hgc_lanes(k) + math.ceil(_hgc_lanes(k) / 4)) / (n * k),
               lambda m, n, k: H32 * (2 + _hgc_y_regs(k)) / (m * k), lambda m, n, k: H32 * hgc_steps(k) / (n * k),
               lambda m, n, k: H32 * (1 + _hgc_y_regs(k)) / (m * k), lambda m, n, k: H32 / (m * k), False,
               lambda m, n, k: 32 * row_words(27 + hgc_steps(k)) * m + 32 * row_words(n) + 8 * n,
               lambda m, n, k: 4 * hgc_steps(k) / (n * k) + 4 * (1 + _hgc_y_regs(k)) / (m * k),
               _final_dup("final_col_dup_share", hgc_steps), False, "0 (exact while |A·Bᵀ| < 2^24)",
               f"{premises_m0}; {coins}; rc-hgc-distinct (open); G from the call's seed",
               "chosen data repeat a later step's word with probability ≤ 1/254; G costs a Y write a column a step a "
               "call (≈ 1/m)"),
        Design("rc_hws_ffma", "rc_hw_ffma with each later step's coin a signed coin byte a_t ∈ [−128, 127] against the "
               "public weight 1 (the noise cancels, so its size costs no T3); step S cancels the code against C_j and "
               "the walk as d0 against −1 and d1 = round(w_i/127) in ≤ 127 chunks against −127; c_j folded in the FFMA",
               lambda k: H32 * hws_steps(k) / k,
               lambda m, n, k: H32 * (hws_steps(k) + 1 + math.ceil((hws_steps(k) - 1) / 4) + 3
                                      + 2 * (hws_walk_lanes(hws_steps(k)) - 1)
                                      + math.ceil(hws_walk_lanes(hws_steps(k)) / 4)) / (n * k),
               lambda m, n, k: 2 * H32 / (m * k), lambda m, n, k: H32 * hws_steps(k) / (n * k),
               lambda m, n, k: H32 / (m * k), lambda m, n, k: H32 / (m * k), False,
               lambda m, n, k: 32 * row_words(28 + 8 * (hws_steps(k) - 1)) * m + 32 * row_words(n),
               lambda m, n, k: 4 * hws_steps(k) / (n * k) + 4 / (m * k), _final_dup("final_col_dup_share", hws_steps),
               False, "0 (exact while |A·Bᵀ| < 2^24)", f"{premises_m0}; {coins}; rc-hws-distinct (open)",
               "chosen data repeat a later step's word with probability ≤ 1/256 with the weights public; 8 coin bits a "
               "row-step"),
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
        Design("ssn_1l_tag", "the served lead with row tags: step 1 holds 24 data, the row tag T(i) (balanced base-13 "
               "digits of the row) against G_j ∈ [−3, 3]⁴ drawn once the weights are registered, and the code's bits "
               "a1 against C_j; every later step 31 data and 1 + a_t against 1; noise served",
               lambda k: H32 * tag_steps(k) / k,
               lambda m, n, k: H32 * (ns.ssn_1l_forming_writes(tag_steps(k)) + 1) / (n * k),
               lambda m, n, k: H32 / (m * k), lambda m, n, k: H32 * tag_steps(k) / (n * k),
               lambda m, n, k: H32 / (m * k), _z, False, lambda m, n, k: 32 * ns.ssn_1l_words(tag_steps(k)) * m,
               lambda m, n, k: 4 * tag_steps(k) / (n * k) + 4 / (m * k), _no_dup, False,
               "served noise std 10–16 units (estimate); unmeasured",
               f"{premises_m0}; {coins}; ssn-tag-distinct, g-private, copy-floor (open)", "reference: the served lead",
               exact=False),
        Design("ssn_1l_tag_x", "the tag lead de-noised in the stock epilogue: steps 1..S−1 are the lead's; step S adds "
               "a1 against −C_j and κ − T(i) against G_j (the code and the tag cancel, leaving κ·G_j) and coin bytes u0 "
               "against 1, u1 against 127 and u2 in e lanes against 127; ρ_i = u0 + 127·u1 + 127e·u2 + w_i and κ·G_j "
               "are removed by the magic IADD3 and FFMA, so the served output is A·Bᵀ and the committed words stay "
               "noisy", lambda k: H32 * tagx_steps(k) / k,
               lambda m, n, k: H32 * (ns.ssn_1l_forming_writes(tagx_steps(k)) + 3
                                      + math.ceil((2 + tagx_extra(k)) / 4)) / (n * k),
               lambda m, n, k: 2 * H32 / (m * k), lambda m, n, k: H32 * tagx_steps(k) / (n * k),
               lambda m, n, k: H32 / (m * k),
               lambda m, n, k: H32 * (popc_writes(tagx_steps(k)) + 4) / (n * k) + 2 * H32 / (m * k), False,
               lambda m, n, k: 32 * row_words(tagx_steps(k) + 26) * m,
               lambda m, n, k: 4 * tagx_steps(k) / (n * k) + 4 / (m * k), _no_dup, True,
               "0: stock's output bit for bit where 1.5·2^23·s_j is exact (exact while |A·Bᵀ| < 2^22)",
               f"{premises_m0}; {coins}; ssn-tag-distinct, g-private, copy-floor (open, the lead's); the final row "
               "offset's collision bound (counting)", "the lead's committed words to step S − 1 and a non-free final "
               "word, so no middle-segment skip; chosen identical or zero rows lose only final words with equal "
               "offsets; chosen out-of-range rows force the fallback write"),
        Design("ssn_1l_tag_xh", "ssn_1l_tag_x where a per-row bound certifies the magic's range (the free bound from "
               "stock's row scale after an RMSNorm, else ‖q_i‖₂ by K/4 IDP4A against each tile's largest ‖B_j‖₂); a row "
               "neither certifies writes zeros to step S's added lanes, so its final word and served output are the "
               "lead's: no fallback write", lambda k: H32 * tagx_steps(k) / k,
               lambda m, n, k: H32 * (ns.ssn_1l_forming_writes(tagx_steps(k)) + 3
                                      + math.ceil((2 + tagx_extra(k)) / 4) + 2) / (n * k),
               lambda m, n, k: 2 * H32 / (m * k), lambda m, n, k: H32 * tagx_steps(k) / (n * k),
               lambda m, n, k: H32 / (m * k),
               lambda m, n, k: H32 * (popc_writes(tagx_steps(k)) + 4) / (n * k) + 2 * H32 / (m * k), False,
               lambda m, n, k: 32 * row_words(tagx_steps(k) + 26) * m,
               lambda m, n, k: 4 * tagx_steps(k) / (n * k) + 4 / (m * k), _no_dup, False,
               "0 on rows a bound certifies (stock's bit for bit where 1.5·2^23·s_j is exact); the lead's served noise "
               "on the rest (`served_noisy_rows`)",
               f"{premises_m0}; {coins}; ssn-tag-distinct, g-private, copy-floor (open, the lead's); the final row "
               "offset's collision bound (counting)", "each row's words are ssn_1l_tag_x's or the lead's, so no "
               "middle-segment skip; chosen out-of-range rows cost the bound (8/n), not a write an output",
               range_mode="lead"),
        Design("ssn_1l_tag_xf", "the tag lead with step S cancelling the code (a1 against −C_j), the tag (κ − T(i) "
               "against G_j with κ = (1, 0, 0, 0), leaving G_0j) and the walk (its base-127 digits against −1 and "
               "−127); only G_0j is folded, in the FFMA: no range limit below 2^24 and no row check",
               lambda k: H32 * tagx_steps(k) / k,
               lambda m, n, k: H32 * (ns.ssn_1l_forming_writes(tagx_steps(k)) + 4 + popc_writes(tagx_steps(k)))
               / (n * k),
               lambda m, n, k: 2 * H32 / (m * k), lambda m, n, k: H32 * tagx_steps(k) / (n * k),
               lambda m, n, k: H32 / (m * k), lambda m, n, k: H32 / (m * k), False,
               lambda m, n, k: 32 * ns.ssn_1l_words(tagx_steps(k)) * m,
               lambda m, n, k: 4 * tagx_steps(k) / (n * k) + 4 / (m * k),
               _final_dup("final_col_dup_share", tagx_steps), False, "0 (exact while |A·Bᵀ + G_0j| < 2^24)",
               f"{premises_m0}; {coins}; ssn-tag-distinct, g-private, copy-floor (open, the lead's)",
               "final word A·Bᵀ + G_0j: lost on equal outputs of a column, and on chosen identical or zero rows"),
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
               "the exact lead: each linear takes the cheaper path"),
        Design("ssn_1l_tag_mix", "ssn_1l_tag_x on linears fed by an RMSNorm, ssn_1l_tag_xf on the rest",
               *([lambda *a: float("nan")] * 3), *([_z] * 3), False, _z, _z, _no_dup, True, "0 (exact)",
               f"{premises_m0}; {coins}; ssn-tag-distinct, g-private, copy-floor (open, the lead's)",
               "the served lead de-noised: each linear takes the cheaper path"),
        Design("ssn_1l_tag_xh_mix", "ssn_1l_tag_xh on linears fed by an RMSNorm, ssn_1l_tag_xf on the rest",
               *([lambda *a: float("nan")] * 3), *([_z] * 3), False, _z, _z, _no_dup, False,
               "0 on rows a bound certifies; the lead's served noise on the rest",
               f"{premises_m0}; {coins}; ssn-tag-distinct, g-private, copy-floor (open, the lead's)",
               "the served lead de-noised with no fallback write after an RMSNorm", range_mode="lead")]


def _per2_words(k: int) -> int:
    return k // 32 + ns.per2_limbs(k)


#: design → (coin-branch model, its k32 steps); `rc_hybrid` is the worse of its two paths
BRANCH = {
    "ssn_1l": ("ssn_1l", ns.steps1l), "ssn_1l_mov": ("ssn_1l", ns.steps1l), "ssn_pd": ("ssn_1l", ns.steps1l),
    "ssn_xd": ("ssn_1l", ns.steps1l), "ssn_xr": ("ssn_xr", s_xr), "ssn_x": ("ssn_x", s_x),
    "ssn_1l_hw": ("ssn_1l_hw", ns.steps1l), "ssn_1l_hg": ("ssn_1l_hg", ns.steps1l), "rc_magic": ("rc_magic", s_rc),
    "rc_ffma": ("rc_ffma", s_rcf),
    "rc_hybrid": ("rc_hybrid", s_rcf), "rc_hw": ("rc_hw", hw_steps), "rc_hw_ffma": ("rc_hw_ffma", hw_steps),
    "rc_hg": ("rc_hg", hg_steps), "rc_hg_ffma": ("rc_hg_ffma", hw_steps), "rc_hgc_ffma": ("rc_hgc_ffma", hgc_steps),
    "rc_hws_ffma": ("rc_hws_ffma", hws_steps), "ssn_1l_tag": ("ssn_1l_tag", tag_steps),
    "ssn_1l_tag_x": ("ssn_1l_tag_x", tagx_steps), "ssn_1l_tag_xf": ("ssn_1l_tag_xf", tagx_steps),
    "ssn_1l_tag_xh": ("ssn_1l_tag_x", tagx_steps),
    "ncp3": ("dense", lambda k: 3 * k // 32), "os2": ("dense", lambda k: 2 * k // 32), "per2": ("dense", _per2_words),
    "lowrank1": ("dense", _per2_words), "pair": ("none", lambda k: 2 * k // 32), "plain": ("plain", lambda k: k // 32),
}
#: linears whose input leaves an RMSNorm: ‖q_i‖ ≤ 127·max|g|·√K / max|x_i| + √K/2 from stock's own row max, so the
#: magic's per-row range check is free; elsewhere it needs ‖q_i‖ (K/4 IDP4A a row)
POST_NORM = {"qkv", "gate_up"}
#: per linear, the path a mixed design takes
MIX = {"rc_hw_mix": lambda lin: "rc_hw" if lin in POST_NORM else "rc_hw_ffma",
       "rc_hg_mix": lambda lin: "rc_hg" if lin in POST_NORM else "rc_hg_ffma",
       "ssn_1l_tag_mix": lambda lin: "ssn_1l_tag_x" if lin in POST_NORM else "ssn_1l_tag_xf",
       "ssn_1l_tag_xh_mix": lambda lin: "ssn_1l_tag_xh" if lin in POST_NORM else "ssn_1l_tag_xf"}
MAGIC_DESIGNS = {"rc_magic", "rc_hybrid", "rc_hw", "rc_hg", "ssn_pd", "ssn_xd", "ssn_xr", "ssn_1l_tag_x"}


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
    fb, check, chosen_range, noisy = range_terms(d, m, n, k, lin)
    cb = coin_branch(d.name, m, k, model)
    bm, sf = BRANCH.get(d.name, ("none", lambda k: 1))
    wc = word_cancel(bm, n) * max(0, sf(k) - 2) / sf(k)
    t = {"alpha": a, "forming_x": d.form_x(m, n, k) - cx, "forming_y": d.form_y(m, n, k) - cy,
         "denoise": d.denoise(m, n, k), "fallback": fb, "sign_fix": 0.0, "final_word": final, "dup": a * dup,
         "range_check": check, "coin_branch_natural": a * cb["natural"]}
    t["gap"] = sum(v for key, v in t.items() if key != "alpha")
    t["coin_branch_chosen"] = a * cb["chosen"]
    t["coin_branch_weights"] = a * cb["weights"]
    t["word_cancel"] = a * wc
    t["range_chosen"] = chosen_range
    data_terms = t["dup"] + t["coin_branch_natural"] + t["fallback"]
    lead = d.range_mode == "lead"
    if lead:
        # the bound's cost depends on the data, and chosen rows can fail it and be identical at once
        data_terms += check
        t["range_chosen"] = chosen_range + t["coin_branch_chosen"]
    t["gap_chosen"] = t["gap"] - data_terms + max(data_terms, t["coin_branch_chosen"], t["range_chosen"])
    # the word cancel repeats steps 2..S−1 and chosen identical rows copy final words: one choice of data does both
    t["gap_chosen_weights"] = t["gap"] - data_terms + max(
        data_terms, t["range_chosen"], t["coin_branch_weights"],
        min(a, t["coin_branch_chosen"] + t["word_cancel"]) + (chosen_range if lead else 0.0))
    t["served_noisy_rows"] = noisy if d.exact else 1.0
    t["coins"] = d.coin_bits(m, n, k) / 8 * ns.COIN_PRICED_BYTE / (m * n * k)
    t["gap_coins_uncredited"] = t["gap"] + t["coins"]
    t["forced_share_of_forming"] = (cx + cy) / max(1e-30, d.form_x(m, n, k) + d.form_y(m, n, k))
    return t


def range_terms(d: Design, m: int, n: int, k: int, lin: str) -> tuple[float, float, float, float]:
    """The magic's range, for a design that decides per row tile whether |A·Bᵀ| < 2²²: the fallback write on unsafe
    tiles of the served data, the decision's own cost (free from stock's row scale after an RMSNorm, else ‖q_i‖ by
    K/4 IDP4A a row, 8/n), on prover-chosen out-of-range rows the fallback on every row (32/K), and the share of rows
    served noisy. A weight matrix whose 127·max_j ‖B_j‖₁ is below 2²² in every layer needs none of them.

    `range_mode="lead"` writes no fallback: after an RMSNorm it computes ‖q_i‖ only on rows the free bound fails, and a
    row that neither bound certifies takes the lead's step S and serves the lead's output; chosen rows fail both, so
    they cost the bound on every row (8/n) and are served noisy."""
    if d.range_mode == "lead":
        st = EXACT.get("static", {}).get(lin)
        if st and st["static_safe_layers"] == st["layers"]:
            return 0.0, 0.0, 0.0, 0.0
        cs = exact_share("cs_unsafe_rows_share", m, lin)
        free = exact_share("free_unsafe_rows_share", m, lin) if lin in POST_NORM else 1.0
        if cs is None or free is None:
            return 0.0, float("nan"), 8 / n, float("nan")
        return 0.0, 8 / n * free, 8 / n, min(cs, free)
    if not d.fallback and d.name not in MAGIC_DESIGNS:
        return 0.0, 0.0, 0.0, 0.0
    st = EXACT.get("static", {}).get(lin)
    if st and st["static_safe_layers"] == st["layers"]:
        return 0.0, 0.0, 0.0, 0.0
    field_ = "free_unsafe" if lin in POST_NORM else "cs_unsafe"
    u = exact_share(f"{field_}_row_tiles_share", m, lin)
    if u is None:
        u = exact_share(f"{field_}_rows_share", m, lin) if field_ == "cs_unsafe" else None
    fb = (float("nan") if u is None else H32 / k * u) if d.fallback else 0.0
    return fb, 0.0 if lin in POST_NORM else 8 / n, H32 / k if d.fallback else 0.0, 0.0


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
           variant: str = "hw", cap: int = LANE_MAX) -> dict:
    """rc_hw (or rc_hg, `variant="hg"`: step lanes against G_j, the walk cancelled in step S; rc_hgc, `"hgc"`: G wide
    and fresh every salt, the walk in unary chunks of at most `cap`) on S steps with e offset lanes: its committed words
    on the salt grid (raw coin words free) and the magic epilogue's output against A·Bᵀ."""
    rng = np.random.default_rng(seed)
    hg = variant in ("hg", "hgc")
    walk_lanes = {"hw": 0, "hg": HG_FIXED - HW_FIXED, "hgc": math.ceil(3 * (s - 1) / cap)}[variant]
    k = 27 + 31 * (s - 2) + 26 - e - walk_lanes
    a, b = ns._data(case, m, n, 160, rng)
    a, b = a[:, :k], b[:k]
    code = ns.ssn_a_w0(n, ns.DATA_PER_STEP)[0]
    kappa = np.array(HW_KAPPA)[:, None]
    g = (hg_g(n) if hg else np.ones(n, dtype=np.int64))[None, :]
    xs, ys, raw, rs, cs = [], [], [], [], []
    for _ in range(salts):
        hr = rng.integers(0, 2 ** 32, (m,), dtype=np.int64)
        hc = rng.integers(0, 2 ** 32, (1,), dtype=np.int64)
        if variant == "hgc":
            g = hgc_g(n, rng)[None, :]
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
                if variant == "hg":
                    lx += [walk % HG_BASE, walk // HG_BASE]
                    ly += [-g, -HG_BASE * g]
                elif variant == "hgc":
                    lx += [np.clip(walk - cap * q, 0, cap) for q in range(walk_lanes)]
                    ly += [-g] * walk_lanes
            lx, ly = np.concatenate(lx, axis=1), np.concatenate(ly, axis=0)
            assert np.abs(lx).max() <= 127 and np.abs(ly).max() <= 127
            d = 32 - lx.shape[1]
            xl += [a[:, lo:lo + d], lx]
            yl += [b[lo:lo + d], ly]
            lo += d
        assert lo == k, (lo, k)
        xs.append(np.concatenate(xl, axis=1))
        ys.append(np.concatenate(yl, axis=0))
        raw.append(np.concatenate([hr, hc] + ([g[0]] if variant == "hgc" else [])))
        w = np.array(hw_weights_e(e))
        rs.append(a1 @ w + (0 if hg else at.sum(1)))
        cs.append(cbit)
    words = ns._words(np.stack(xs), np.stack(ys))
    out = ns._distinct([words], [np.stack(raw)])
    resid = words[:, :, :, -1] - np.stack(rs)[:, :, None] - np.stack(cs)[:, None, :]
    out["exact_after_epilogue"] = bool((resid == (a @ b)[None]).all())
    return out


def sim_hws(case: str, salts: int = 24, m: int = 4, n: int = 6, seed: int = 1, s: int = 4, cap: int = 4) -> dict:
    """rc_hws_ffma on S steps: signed coin bytes against 1, step S cancelling the code against −C_j and the walk (d0
    against −1, d1 in chunks of at most `cap` against −127). Its committed words on the salt grid (raw coin words free)
    and the final word less c_j against A·Bᵀ."""
    rng = np.random.default_rng(seed)
    hi = math.ceil((math.ceil(128 * (s - 1) / LANE_MAX) + 1) / cap)
    k = 27 + 31 * (s - 2) + 27 - 1 - hi
    a, b = ns._data(case, m, n, 160, rng)
    a, b = a[:, :k], b[:k]
    code = ns.ssn_a_w0(n, ns.DATA_PER_STEP)[0]
    one = np.ones((1, n), dtype=np.int64)
    xs, ys, raw, cs = [], [], [], []
    for _ in range(salts):
        hr = rng.integers(0, 2 ** 32, (m, s), dtype=np.int64)
        hc = rng.integers(0, 2 ** 32, (1,), dtype=np.int64)
        a1 = np.stack([(hr[:, 0] >> (8 * q)) & 0x7F for q in range(4)], axis=1)
        at = np.concatenate([np.zeros((m, 1), dtype=np.int64), ((hr[:, 1:] & 0xFF) ^ 0x80) - 0x80], axis=1)
        walk = at.sum(1, keepdims=True)
        d1 = np.rint(walk / LANE_MAX).astype(np.int64)
        d0 = walk - LANE_MAX * d1
        chunks = [np.sign(d1) * np.clip(np.abs(d1) - cap * q, 0, cap) for q in range(hi)]
        cbit = np.array([(hc[0] >> j) & 1 for j in range(n)])
        xl, yl, lo = [], [], 0
        for t in range(s):
            if t == 0:
                lx, ly = [a1, np.ones((m, 1), dtype=np.int64)], [code, cbit[None, :]]
            elif t < s - 1:
                lx, ly = [at[:, t:t + 1]], [one]
            else:
                lx, ly = [at[:, t:t + 1], a1, d0] + chunks, [one, -code, -one] + [-LANE_MAX * one] * hi
            lx, ly = np.concatenate(lx, axis=1), np.concatenate(ly, axis=0)
            assert lx.min() >= -128 and lx.max() <= LANE_MAX and np.abs(ly).max() <= LANE_MAX
            d = 32 - lx.shape[1]
            xl += [a[:, lo:lo + d], lx]
            yl += [b[lo:lo + d], ly]
            lo += d
        assert lo == k, (lo, k)
        xs.append(np.concatenate(xl, axis=1))
        ys.append(np.concatenate(yl, axis=0))
        raw.append(np.concatenate([hr.ravel(), hc]))
        cs.append(cbit)
    words = ns._words(np.stack(xs), np.stack(ys))
    out = ns._distinct([words], [np.stack(raw)])
    out["exact_after_epilogue"] = bool(((words[:, :, :, -1] - np.stack(cs)[:, None, :]) == (a @ b)[None]).all())
    return out


def sim_tag(case: str, variant: str = "x", salts: int = 24, m: int = 4, n: int = 6, seed: int = 1, s: int = 4,
            e: int = 2) -> dict:
    """The tag lead (`variant="lead"`, served), ssn_1l_tag_x (`"x"`, e lanes of u2), ssn_1l_tag_xh (`"xh"`: x with
    the odd rows taking the lead's step S) or ssn_1l_tag_xf (`"xf"`) on S steps: its committed words on the salt grid
    (raw coin words and G free, T(i) a constant) and, de-noised, the epilogue's output against A·Bᵀ (`"xh"`: the
    de-noised rows' output, and whether the others' final words are the lead's)."""
    rng = np.random.default_rng(seed)
    extra = {"lead": 0, "x": TAGX_FIXED + e, "xh": TAGX_FIXED + e, "xf": TAGX_FIXED}[variant]
    k = TAG_FIRST + 31 * (s - 2) + 31 - extra
    a, b = ns._data(case, m, n, 160, rng)
    a, b = a[:, :k], b[:k]
    code = ns.ssn_a_w0(n, ns.DATA_PER_STEP)[0]
    tag = tag_digits(m)
    one = np.ones((1, n), dtype=np.int64)
    kappa = np.array(TAGX_KAPPA if variant in ("x", "xh") else (1, 0, 0, 0))
    lead_rows = (np.arange(m) % 2 == 1) if variant == "xh" else np.zeros(m, dtype=bool)
    xs, ys, raw, rs, cs, leads = [], [], [], [], [], []
    for _ in range(salts):
        hr = rng.integers(0, 2 ** 32, (m,), dtype=np.int64)
        hu = rng.integers(0, 2 ** 32, (m,), dtype=np.int64)
        g = rng.integers(-3, 4, (4, n))
        a1 = np.stack([(hr >> q) & 1 for q in range(4)], axis=1)
        at = np.stack([np.zeros(m, dtype=np.int64)] + [1 + ((hr >> (4 + t)) & 1) for t in range(s - 1)], axis=1)
        walk = at.sum(1, keepdims=True)
        u0, u1, u2 = hu[:, None] & 0x7F, ((hu[:, None] >> 8 & 0xFF) ^ 0x80) - 0x80, ((hu[:, None] >> 16 & 0xFF) ^ 0x80) - 0x80
        xl, yl, lo = [], [], 0
        for t in range(s):
            if t == 0:
                lx, ly = [tag, a1], [g, code]
            elif t < s - 1:
                lx, ly = [at[:, t:t + 1]], [one]
            else:
                lx, ly = [at[:, t:t + 1]], [one]
                if variant in ("x", "xh"):
                    added = np.concatenate([a1, kappa - tag, u0, u1] + [u2] * e, axis=1)
                    lx += [np.where(lead_rows[:, None], 0, added)]
                    ly += [-code, g, one, LANE_MAX * one] + [LANE_MAX * one] * e
                elif variant == "xf":
                    lx += [a1, kappa - tag, -(walk % LANE_MAX), -(walk // LANE_MAX)]
                    ly += [-code, g, one, LANE_MAX * one]
            lx, ly = np.concatenate(lx, axis=1), np.concatenate(ly, axis=0)
            assert lx.min() >= -128 and lx.max() <= LANE_MAX and np.abs(ly).max() <= LANE_MAX
            d = 32 - lx.shape[1]
            xl += [a[:, lo:lo + d], lx]
            yl += [b[lo:lo + d], ly]
            lo += d
        assert lo == k, (lo, k)
        xs.append(np.concatenate(xl, axis=1))
        ys.append(np.concatenate(yl, axis=0))
        raw.append(np.concatenate([hr, hu, g.ravel()]))
        rs.append((u0 + LANE_MAX * u1 + LANE_MAX * e * u2 + walk)[:, 0] if variant in ("x", "xh")
                  else np.zeros(m, np.int64))
        cs.append(kappa @ g)
        leads.append(a1 @ code + tag @ g + walk)
    words = ns._words(np.stack(xs), np.stack(ys))
    out = ns._distinct([words], [np.stack(raw)])
    resid = words[:, :, :, -1] - np.stack(rs)[:, :, None] - np.stack(cs)[:, None, :]
    out["exact_after_epilogue"] = bool((resid[:, ~lead_rows] == (a @ b)[None, ~lead_rows]).all())
    if variant == "xh":
        out["lead_rows_lead_final"] = bool((words[..., -1][:, lead_rows] - np.stack(leads)[:, lead_rows]
                                            == (a @ b)[None, lead_rows]).all())
    out["final_bad"] = ns._distinct([words[:, :, :, -1]], [np.stack(raw)])["bad_share"]
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
    if model.startswith("ssn_1l_tag"):
        a1 = rng.integers(0, 2, (m, 4))
        g = rng.integers(-3, 4, (4, n))
        at = 1 + rng.integers(0, 2, (m, s))
        at[:, 0] = 0
        nz = (a1 @ code + tag_digits(m) @ g)[:, None, :] + at.cumsum(1)[:, :, None]
        if model == "ssn_1l_tag_x":
            u = rng.integers(-128, 128, (m, 3))
            rho = (u[:, 0] & 0x7F) + LANE_MAX * u[:, 1] + LANE_MAX * tagx_extra(k) * u[:, 2] + at.sum(1)
            nz[:, -1] = rho[:, None] + np.array(TAGX_KAPPA) @ g
        elif model == "ssn_1l_tag_xf":
            nz[:, -1] = g[0]
        return nz
    if model.startswith("rc_hg"):
        a1 = rng.integers(0, 128, (m, 4))
        at = 1 + 2 * rng.integers(0, 2, (m, s))
        at[:, 0] = 0
        c = rng.integers(0, 2, n)
        g = (np.ones(n, dtype=np.int64) if model == "rc_hg_public" else hgc_g(n, rng) if model == "rc_hgc_ffma"
             else hg_g(n))
        nz = (a1 @ code)[:, None, :] + at.cumsum(1)[:, :, None] * g + c
        nz[:, -1] = c if model.endswith("_ffma") else (a1 @ np.array(hw_weights(k, HG_FIXED)))[:, None] + c
        return nz
    if model in ("rc_magic", "rc_ffma"):
        c = rng.integers(0, 2, n)
        nz = np.broadcast_to(rng.integers(0, 2, (m, s)).cumsum(1)[:, :, None] + c, (m, s, n)).copy()
        if model == "rc_ffma":
            nz[:, -1] = c
        return nz
    if model == "rc_hws_ffma":
        a1 = rng.integers(0, 128, (m, 4))
        at = rng.integers(-128, 128, (m, s))
        at[:, 0] = 0
        c = rng.integers(0, 2, n)
        nz = (a1 @ code)[:, None, :] + at.cumsum(1)[:, :, None] + c
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


#: per coin-branch model, a later step's noise increment a_t[i]·w_j: a_t's values (uniform), and when the column
#: weights w are drawn: "public" (known before the weights are registered), "registered" (after the weights, before
#: the requests) or "call" (from the call's seed, after A and B)
STEP_NOISE = {"ssn_1l": ((0, 1), "public"), "ssn_xr": ((0, 1), "public"), "ssn_x": ((0, 1), "public"),
              "rc_magic": ((0, 1), "public"), "rc_ffma": ((0, 1), "public"), "rc_hybrid": ((0, 1), "public"),
              "rc_hw": ((1, 3), "public"), "rc_hw_ffma": ((1, 3), "public"), "ssn_1l_hw": ((1, 3), "public"),
              "ssn_1l_hg": ((1, 3), "registered"), "rc_hg": ((1, 3), "registered"),
              "rc_hg_ffma": ((1, 3), "registered"), "rc_hgw_ffma": ((1, 3), "registered"),
              "rc_hgc_ffma": ((1, 3), "call"), "rc_hws_ffma": (tuple(range(-128, 128)), "public"),
              "ssn_1l_tag": ((1, 2), "public"), "ssn_1l_tag_x": ((1, 2), "public"),
              "ssn_1l_tag_xf": ((1, 2), "public")}


def step_column_weights(model: str, n: int) -> np.ndarray:
    """A registered design's step-lane column weights (`rc_hgw_ffma`: rc_hgc's wide G, but drawn at registration)."""
    if model == "ssn_1l_hg":
        return np.random.default_rng(_SIGNS_SEED + 1).choice(np.array([-1, 1]), size=n)
    if model in ("rc_hg", "rc_hg_ffma"):
        return hg_g(n)
    return hgc_g(n, np.random.default_rng(2027))


@functools.lru_cache(maxsize=None)
def word_cancel(model: str, n: int, classes: int = 31) -> float:
    """Share of a later step's words that chosen data leave equal to the step before (the cancel case at word grain).

    The step's dot is −v_q on each of `classes` classes of columns (one-hot B lanes, fixed when B is registered), with
    v_q set once A is chosen, so a word repeats where a_t[i]·w_j = v_q. A program with free moves (W1) regroups those
    rows (per step) and columns (once) into tiles and skips them. Public weights are met column by column (B's lane
    is −w_j), so the share is a_t's likeliest value's; registered weights are known when A is chosen, so each class
    takes its likeliest product; weights from the call's seed aren't, so one v meets the products' distribution."""
    if model not in STEP_NOISE:
        return 0.0
    a_vals, drawn = STEP_NOISE[model]
    a_vals = np.array(a_vals)
    if drawn == "public":
        return 1 / len(a_vals)
    if drawn == "call":
        support = np.concatenate([np.arange(-LANE_MAX, 0), np.arange(1, LANE_MAX + 1)])
        return float(np.unique(np.multiply.outer(a_vals, support), return_counts=True)[1].max()
                     / (len(a_vals) * len(support)))
    w = step_column_weights(model, n)
    hit = sum(np.unique(np.multiply.outer(a_vals, w[q]), return_counts=True)[1].max()
              for q in np.array_split(np.arange(n), min(classes, n)))
    return float(hit / (len(a_vals) * n))


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
    hgc = {c: sim_hw(c, variant="hgc", cap=4) for c in ns.CASES}
    ok("rc_hgc (G fresh every salt, the walk in three unary chunks): the epilogue's output is A·Bᵀ and every word is "
       "distinct and non-free on every data case", all(v["exact_after_epilogue"] and v["bad_share"] == 0
                                                        for v in hgc.values()),
       {c: (v["exact_after_epilogue"], v["bad_share"]) for c, v in hgc.items()})
    ok("rc_hgc keeps ssn_1l's steps at K 2,048, 7,168 and 8,192 with its walk chunks in step S",
       all(hgc_steps(k) == ns.steps1l(k) and hw_extra(k, hgc_fixed(k)) >= 0 for k in (2048, 7168, 8192)),
       {k: (hgc_steps(k), _hgc_lanes(k), hw_extra(k, hgc_fixed(k))) for k in (2048, 7168, 8192)})
    hws = {c: sim_hws(c) for c in ns.CASES}
    ok("rc_hws (signed coin bytes, the walk's d1 in two chunks): the final word less c_j is A·Bᵀ on every data case; "
       "every word is distinct and non-free except identical rows' final words (no row noise; at most 1/S)",
       all(v["exact_after_epilogue"] for v in hws.values()) and 0 < hws["identical_rows"]["bad_share"] <= 1 / 4
       and all(v["bad_share"] == 0 for c, v in hws.items() if c != "identical_rows"),
       {c: (v["exact_after_epilogue"], v["bad_share"]) for c, v in hws.items()})
    ok("rc_hws keeps ssn_1l's steps at K 2,048, 7,168 and 8,192 with its walk lanes in step S",
       all(hws_steps(k) == ns.steps1l(k) for k in (2048, 7168, 8192)),
       {k: (hws_steps(k), hws_walk_lanes(hws_steps(k))) for k in (2048, 7168, 8192)})
    wcs = {mo: word_cancel(mo, 2048) for mo in ("ssn_1l", "rc_hw_ffma", "ssn_1l_hw", "ssn_1l_hg", "rc_hg_ffma",
                                                "rc_hgw_ffma", "rc_hgc_ffma", "rc_hws_ffma")}
    ok("word-level cancel at n 2,048: public step weights lose 1/2 of a later step's words, rc_hg's and ssn_1l_hg's "
       "registered narrow weights ≥ 1/4, rc_hgc's per-call G ≤ 1/254 and rc_hws's coin bytes 1/256; G drawn at "
       "registration (mutation) loses ≥ 2× rc_hgc's to per-class choices",
       all(wcs[mo] == 0.5 for mo in ("ssn_1l", "rc_hw_ffma", "ssn_1l_hw"))
       and min(wcs["ssn_1l_hg"], wcs["rc_hg_ffma"]) >= 0.25 and wcs["rc_hgc_ffma"] <= 1 / 254 + 1e-12
       and wcs["rc_hws_ffma"] == 1 / 256 and wcs["rc_hgw_ffma"] >= 2 * wcs["rc_hgc_ffma"], wcs)

    def sim_wc(model: str, m: int = 64, s: int = 12, n: int = 248, calls: int = 32) -> float:
        """The repeated share of steps 2..S−1's words over `calls` calls (a per-call G is redrawn in each)."""
        rng = np.random.default_rng(3)
        v = np.full(n, 3, dtype=np.int64)
        if STEP_NOISE[model][1] == "registered":
            w = step_column_weights(model, n)
            for q in np.array_split(np.arange(n), 31):
                vals, cnt = np.unique(np.multiply.outer(np.array(STEP_NOISE[model][0]), w[q]), return_counts=True)
                v[q] = vals[cnt.argmax()]
        hits = []
        for _ in range(calls):
            words = branch_noise(model, m, s, n, 2048, rng) - np.arange(s)[None, :, None] * v
            hits.append((words[:, 1:s - 1] == words[:, :s - 2]).mean())
        return float(np.mean(hits))
    swc = {mo: (sim_wc(mo), word_cancel(mo, 248)) for mo in ("rc_hg_ffma", "rc_hgc_ffma", "rc_hws_ffma")}
    ok("word-level cancel: simulated repeats through the coin-branch noise match the closed form (rc_hg_ffma within "
       "0.02, rc_hgc_ffma and rc_hws_ffma within 0.002)", abs(swc["rc_hg_ffma"][0] - swc["rc_hg_ffma"][1]) <= 0.02
       and all(abs(swc[mo][0] - swc[mo][1]) <= 0.002 for mo in ("rc_hgc_ffma", "rc_hws_ffma")), swc)
    tg = {v: {c: sim_tag(c, v) for c in ns.CASES} for v in ("lead", "x", "xf")}
    ok("ssn_1l_tag (served) and ssn_1l_tag_x: every word distinct and non-free on every data case, the final word "
       "included, so the middle-segment skip finds no free word to run back from; ssn_1l_tag_x's epilogue output is "
       "A·Bᵀ", all(r["bad_share"] == 0 for v in ("lead", "x") for r in tg[v].values())
       and all(r["exact_after_epilogue"] for r in tg["x"].values()),
       {v: {c: (r["exact_after_epilogue"], r["bad_share"]) for c, r in tg[v].items()} for v in ("lead", "x")})
    ok("ssn_1l_tag_xf: the epilogue output is A·Bᵀ; every word distinct and non-free except identical or zero rows' "
       "final words (no row noise; at most 1/S)", all(r["exact_after_epilogue"] for r in tg["xf"].values())
       and all(0 < tg["xf"][c]["bad_share"] <= 1 / 4 for c in ("identical_rows", "zero_row"))
       and all(r["bad_share"] == 0 for c, r in tg["xf"].items() if c not in ("identical_rows", "zero_row")),
       {c: (r["exact_after_epilogue"], r["bad_share"]) for c, r in tg["xf"].items()})
    th = {c: sim_tag(c, "xh") for c in ns.CASES}
    ok("ssn_1l_tag_xh with half its rows past the bound: every word distinct and non-free on every data case; the "
       "certified rows' output is A·Bᵀ and the others' final words are the lead's",
       all(r["bad_share"] == 0 and r["exact_after_epilogue"] and r["lead_rows_lead_final"] for r in th.values()),
       {c: (r["bad_share"], r["exact_after_epilogue"], r["lead_rows_lead_final"]) for c, r in th.items()})
    dn = {d.name: d for d in designs()}
    rx = {lin: (gap_terms(dn["ssn_1l_tag_xh"], 8192, n, k, "M0", False, lin),
                gap_terms(dn["ssn_1l_tag_x"], 8192, n, k, "M0", False, lin)) for lin, (n, k) in ns.LLAMA.items()}
    ok("ssn_1l_tag_xh writes no fallback, and on chosen out-of-range rows its bound (8/n) and coin branch cost less "
       "than ssn_1l_tag_x's check and write an output at every Llama linear",
       all(h["fallback"] == 0 and h["range_chosen"] < x["range_chosen"] + x["range_check"] for h, x in rx.values()),
       {lin: (h["range_chosen"], x["range_chosen"] + x["range_check"]) for lin, (h, x) in rx.items()})
    ok("ssn_1l_tag_x keeps the lead's steps (67 / 232 / 265 at K 2,048 / 7,168 / 8,192) with its lanes in step S, and "
       "its final row offset has Q ≥ 2^17", all(tagx_steps(k) == tag_steps(k) == ns.steps1l(k) and tagx_extra(k) >= 0
                                                and tagx_offset_q(k) >= 2 ** 17 for k in (2048, 7168, 8192)),
       {k: (tagx_steps(k), tagx_extra(k), round(math.log2(tagx_offset_q(k)), 2)) for k in (2048, 7168, 8192)})
    tb = {c: branch_collapse("ssn_1l_tag_x", c, 8192, 67, 2048) for c in ("identical", "zero", "random")}
    tl = {c: branch_collapse("ssn_1l_tag", c, 8192, 67, 2048) for c in ("identical", "zero")}
    bound = tagx_final_bound(8192, 2048)
    ok("coin branch at 8,192 rows, K 2,048: ssn_1l_tag_x loses on chosen identical or zero rows within 0.5–2× its "
       "final-word bound and nothing on random rows; the served lead loses nothing",
       all(0.5 * bound <= tb[c] <= 2 * bound for c in ("identical", "zero")) and tb["random"] == 0
       and all(v == 0 for v in tl.values()), {**tb, "lead": tl, "bound": bound})
    ok("the de-noise changes no later step: chosen weights cancel half of the served lead's and ssn_1l_tag_x's "
       "row-steps alike", word_cancel("ssn_1l_tag", 2048) == word_cancel("ssn_1l_tag_x", 2048) == 0.5,
       {mo: word_cancel(mo, 2048) for mo in ("ssn_1l_tag", "ssn_1l_tag_x")})
    ok("rc steps: K 2,048 and 8,192 take as many steps as ssn_1l (67, 265)",
       (s_rc(2048), s_rc(8192)) == (ns.steps1l(2048), ns.steps1l(8192)), (s_rc(2048), s_rc(8192)))
    ok("in-GEMM cancellation lanes fit the last step's spare lanes at Llama's and V4's K",
       all(s_x(k) == ns.steps1l(k) and s_rcf(k) == s_rc(k) for k in (2048, 7168, 8192)),
       {k: (s_x(k), ns.steps1l(k), s_rcf(k), s_rc(k)) for k in (2048, 7168, 8192)})
    return res


def report() -> dict:
    ds = designs()
    by_name = {d.name: d for d in ds}
    rows = []
    for d in ds:
        for model in MODELS:
            for c1 in (False, True):
                e = {"design": d.name, "model": model, "c1": c1}
                for point, m in POINTS.items():
                    e[f"llama_{point}"] = weighted(d, ns.LLAMA, m, model, c1)
                    e[f"v4dense_{point}"] = weighted(d, ns.V4_DENSE, m, model, c1)
                for point in ns.STEP:
                    # a mixed design's T2 is the worse of its paths' (an upper estimate)
                    parts = [ns.t2_estimate(by_name[path], point, c1)
                             for path in ({MIX[d.name](lin) for lin in ns.LLAMA} if d.name in MIX else {d.name})]
                    e[f"t2_{point}"] = max(p["t2"] for p in parts)
                    e[f"t2_overlapped_{point}"] = max(p["t2_overlapped"] for p in parts)
                rows.append(e)
    per_dot = {d.name: {lin: {"alpha": d.alpha(k), "extra_macs": (d.alpha(k) - 1) * k,
                              "forming_x_writes_row": d.form_x(1, n, k) * n * k / H32,
                              "forming_y_writes_col": d.form_y(1, n, k) * k / H32,
                              "denoise_writes_row_call": d.denoise(10 ** 12, n, k) * n * k / H32,
                              "coin_bits_row": d.coin_bits(1, 0, k), "coin_bits_col_call": d.coin_bits(0, n, k) / n}
                        for lin, (n, k) in ns.LLAMA.items()} for d in ds if d.name not in MIX}
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
           "V4d T8192 chosen", "L T8192 chosen+W", "alpha L", "T2 8192", "T2 512(64)", "L noisy rows T512(64) / T8192")
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
                                 f"{r['t2_T512_64tok']:.1f}x",
                                 f"{f('llama_T512_64tok', 'served_noisy_rows')} / {f('llama_T8192', 'served_noisy_rows')}"]))
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
    for name in ("rc_magic", "rc_ffma", "rc_hybrid", "rc_hw", "rc_hw_mix", "rc_hw_ffma", "rc_hg", "rc_hg_mix",
                 "rc_hg_ffma", "rc_hgc_ffma", "rc_hws_ffma", "ssn_1l", "ssn_1l_hw", "ssn_1l_hg", "ssn_1l_tag",
                 "ssn_1l_tag_x", "ssn_1l_tag_xf", "ssn_1l_tag_mix", "ssn_1l_tag_xh", "ssn_1l_tag_xh_mix", "per2"):
        for model in ("M0", "M1u"):
            r = next(r for r in rep["rows"] if r["design"] == name and r["model"] == model and not r["c1"])
            for basis in ("gap", "gap_coins_uncredited", "gap_chosen", "gap_chosen_weights"):
                for shape in ("llama", "v4dense"):
                    g = r[f"{shape}_T8192"][basis]
                    if not math.isnan(g):
                        rep[f"periods_{name}_{model}_noC1_{shape}_T8192_{basis}"] = ns.periods(
                            g, r[f"{shape}_T8192"]["alpha"])
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
