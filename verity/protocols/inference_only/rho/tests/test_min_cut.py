"""The min-cut bound: `rho.min_cut` against the Lean text it computes, its rows of `β(k)`, the bound at a policy, the
bound over a period's blocks with untied words, with a threshold and with none, beside the work no block is credited
for, the class in `CLASSES`, and `periods.py --bounds`."""

import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from rho import instances, lean, min_cut, theorem
from rho.classes import CLASSES, FlopBudget, Outside, premise_status, premises
from rho.families import family
from rho.params import Registry
from rho.scenario import build

ROOT = Path(__file__).resolve().parents[5]
REG = Registry.load()
LOCK = json.loads((ROOT / "security" / "lean-audit.json").read_text())


def lean_text(name: str) -> str:
    """The Lean file that defines `name`, the one module whose definitions the lock lists it among, as one line."""
    (module,) = (m for m, read in LOCK["reads"].items() if name in read["definitions"])
    return " ".join((ROOT / "security" / f"{module.replace('.', '/')}.lean").read_text().split())


#: Lean's `alphaHE` and `shareF` as their file defines them, and both conjuncts of `HE` and `HEShared` as theirs
#: concludes them
ALPHA = "noncomputable def alphaHE (k : ℕ) (q b : ℝ) : ℝ := √(k ^ 3 / (q * b)) / 2"
SHARE = "noncomputable def shareF (ω : ℕ) : ℝ := √(2 * (1 - 1 / ω))"
HE = ("(W.N : ℝ) ≤ m * (k + alphaHE k (βw * βx) b) + k / 2 * √(m * W.O / (βw * βx)) ∧ "
      "(c / d.D.size d.D.computed : ℝ) * (W.N / (k + alphaHE k (βw * βx) b) - "
      "k / 2 * √(W.O * W.N / (βw * βx)) / (k + alphaHE k (βw * βx) b) ^ (3 / 2 : ℝ)) ≤ "
      "m * c / d.D.size d.D.computed")
HE_SHARED = ("(W.N : ℝ) ≤ m * (k + shareF ω * alphaHE k q b) + shareF ω * (k / 2 * √(m * W.O / q)) ∧ "
             "(c / d.D.size d.D.computed : ℝ) * (W.N / (k + shareF ω * alphaHE k q b) - "
             "shareF ω * (k / 2 * √(W.O * W.N / q)) / (k + shareF ω * alphaHE k q b) ^ (3 / 2 : ℝ)) ≤ "
             "m * c / d.D.size d.D.computed")


def alpha_he(k, q, b):
    """`ALPHA`, transcribed."""
    return math.sqrt(k ** 3 / (q * b)) / 2


def share_f(w):
    """`SHARE`, transcribed."""
    return math.sqrt(2 * (1 - 1 / w))


def he(m, c, size, n, o, k, bw, bx, b):
    """`HE`'s two conjuncts, transcribed: the right side of the first, and the left side of the second."""
    return (m * (k + alpha_he(k, (bw * bx), b)) + k / 2 * math.sqrt(m * o / (bw * bx)),
            (c / size) * (n / (k + alpha_he(k, (bw * bx), b))
                          - k / 2 * math.sqrt(o * n / (bw * bx)) / (k + alpha_he(k, (bw * bx), b)) ** (3 / 2)))


def he_shared(m, c, size, n, o, k, q, b, w):
    """`HE_SHARED`'s two conjuncts, transcribed likewise."""
    return (m * (k + share_f(w) * alpha_he(k, q, b)) + share_f(w) * (k / 2 * math.sqrt(m * o / q)),
            (c / size) * (n / (k + share_f(w) * alpha_he(k, q, b))
                          - share_f(w) * (k / 2 * math.sqrt(o * n / q))
                          / (k + share_f(w) * alpha_he(k, q, b)) ** (3 / 2)))


#: (N, O, |C|, c, k, β_w, β_x, b): a small work, a traced step's, a block of the served policy's size, and the
#: session policy's at a profile's costs
POINTS = ((2.0e9, 3.0e7, 5.0e12, 1.0e9, 6.5e4, 15.9, 15.9, 16.0),
          (7.2986e14, 3.5673e11, 1.2017e19, 4.6566e14, 65_536, 15.9, 15.9, 16.0),
          (3.6e10, 2.8e8, 1.4e13, 1.4e13, 4.2e8, 16.0, 0.37, 16.0),
          (1.0e18, 2.7e14, 3.0e21, 2.45e19, 2.0e7, 2.5, 4.0, 4.0))


def test_the_arithmetic_is_the_lean_statements():
    """The module's functions against each statement's sides, transcribed from the text pinned here: `HE` at `F = 1`
    and `q = β_w β_x`, `HEShared` at `F = shareF ω`."""
    assert ALPHA in lean_text(f"{min_cut.AREA}.alphaHE") and SHARE in lean_text(f"{min_cut.AREA}.shareF")
    assert HE in lean_text(min_cut.HE) and HE_SHARED in lean_text(min_cut.HE_SHARED)
    assert lean.status(min_cut.HE) == lean.status(min_cut.HE_SHARED) == "proved"
    assert lean.status(min_cut.HARDNESS) == lean.status(min_cut.EXPORTS) == "assumption"
    assert min_cut.share(2) == share_f(2) == 1 and min_cut.share(3) == share_f(3) == pytest.approx(2 / math.sqrt(3))
    for (n, o, size, c, k, bw, bx, b), m in itertools.product(POINTS, (1.0, 7.0, 1e6)):
        q = bw * bx
        assert min_cut.alpha(k, q, b) == alpha_he(k, q, b)
        count, slowdown = he(m, c, size, n, o, k, bw, bx, b)
        assert min_cut.macs_at_most(m, k, o, q, b) == pytest.approx(count, rel=1e-14)
        assert min_cut.slowdown_at_least(n, o, size, c, k, q, b) == pytest.approx(slowdown, rel=1e-12)
        for w in (2, 3):
            count, slowdown = he_shared(m, c, size, n, o, k, q, b, w)
            assert min_cut.macs_at_most(m, k, o, q, b, min_cut.share(w)) == pytest.approx(count, rel=1e-14)
            assert min_cut.slowdown_at_least(n, o, size, c, k, q, b, min_cut.share(w)) == pytest.approx(
                slowdown, rel=1e-12)


def test_the_least_blocks_and_the_macs_a_block_are_the_count_solved():
    """`blocks_at_least` is where the count meets `N`, `macs_a_block` is `N` over it whatever `N`, and the second
    conjunct is never above the first solved exactly: below 0 where the outputs' term outweighs the first."""
    for (n, o, size, c, k, bw, bx, b), share in itertools.product(POINTS, (1.0, min_cut.share(3))):
        q = bw * bx
        m = min_cut.blocks_at_least(n, o, k, q, b, share)
        assert min_cut.macs_at_most(m, k, o, q, b, share) == pytest.approx(n, rel=1e-9)
        below, above = (min_cut.macs_at_most(m * x, k, o, q, b, share) for x in (0.999, 1.001))
        assert below < n < above
        for scale in (1.0, 1e-3, 1e4):
            assert min_cut.macs_a_block(k, o / n, q, b, share) == pytest.approx(
                n * scale / min_cut.blocks_at_least(n * scale, o * scale, k, q, b, share), rel=1e-9)
        assert min_cut.slowdown_at_least(n, o, size, c, k, q, b, share) <= m * c / size * (1 + 1e-12)
    n, o, size, c, k, bw, bx, b = POINTS[2]
    assert min_cut.slowdown_at_least(n, o, size, c, k, bw * bx, b) < 0 < min_cut.blocks_at_least(n, o, k, bw * bx, b)


def test_beta_is_the_hardness_rows_the_next_rows_between_two_and_there_is_none_past_the_last():
    """The rows are `premise.training-cut-hardness.beta`'s, with a source; `k` rises and `β` falls along them; (H) is
    credible up to the last row's `k`, and past it there is no `β`. A `k` strictly between two rows reads the next
    row's `β`, a step: a row is the least ratio found up to its `k`, so the rows allow no more between them, and
    nothing between two rows reads above the next one."""
    p = REG.param("premise.training-cut-hardness.beta")
    assert min_cut.BETA == tuple(map(tuple, p.value)) and p.source and p.status == "assumed"
    ks, betas = zip(*min_cut.BETA, strict=True)
    assert list(ks) == sorted(ks) and list(betas) == sorted(betas, reverse=True) and betas[-1] > 0
    assert REG["premise.training-cut-hardness.credible_bits"] == ks[-1]
    assert min_cut.beta(1) == min_cut.beta(ks[0]) == betas[0]
    for (k0, b0), (k1, b1) in itertools.pairwise(min_cut.BETA):
        assert min_cut.beta(k0) == b0 > b1 == min_cut.beta(k1)
        for k in (math.nextafter(k0, math.inf), k0 + 1, math.sqrt(k0 * k1), (k0 + k1) / 2, math.nextafter(k1, 0)):
            assert k0 < k < k1 and min_cut.beta(k) == b1
    assert "next row" in p.source and "linear" not in p.source
    assert min_cut.beta(ks[-1] * (1 + 1e-9)) is None and min_cut.beta(1e12) is None


#: the served policy's public parameters: gates, fresh input bits and output bits a block, and the θ-MACs a block holds
C, K, N_OUT, MACS = 13_818_131_814_643, 422_197_408, 25_165_824, 36_241_670_144
#: a position of Llama 3.2 1B's forward, from its shape: MACs and output words. 16 layers of q and o (2048 → 2048),
#: k and v (2048 → 512), gate and up (2048 → 8192) and down (8192 → 2048), and the head (2048 → 128,256)
POSITION = 16 * (2 * 2048 * 2048 + 2 * 2048 * 512 + 3 * 2048 * 8192) + 2048 * 128_256
WORDS = 16 * (2 * 2048 + 2 * 512 + 2 * 8192 + 2048) + 128_256


def test_the_bound_at_the_served_policy_is_one():
    """A block of the served policy may finish about 5.0e13 MACs of the served model's forward, one bf16 output word
    per 2,447 MACs, and holds 3.6e10: one block does it, and the slowdown is 1. Its `k` is between the last two rows,
    so its `β` is the last row's. At the first row's `k` a block may finish under a million MACs."""
    assert (POSITION, WORDS) == (1_235_746_816, 505_088)
    o = WORDS * 16 / POSITION
    r = min_cut.bound(C, K, N_OUT, N=MACS, O=MACS * o, size=C)
    assert r.slowdown == 1 and r.blocks == pytest.approx(MACS / r.macs_a_block) and r.blocks < 1
    assert r.statement == min_cut.HE and r.premises == (min_cut.HARDNESS, min_cut.EXPORTS)
    assert (r.beta_w, r.b, r.F) == (16, 16, 1) and r.beta_x == min_cut.beta(K) == min_cut.BETA[-1][1] == 0.37
    assert r.alpha == min_cut.alpha(K, 16 * r.beta_x, 16)
    per_block = min_cut.macs_a_block(K, o, 16 * r.beta_x, 16)
    assert r.macs_a_block == pytest.approx(per_block) and per_block == pytest.approx(5.012e13, rel=5e-4)
    assert min_cut.macs_a_block(K, o, 16 * r.beta_x, 16, min_cut.share(3)) == pytest.approx(6.666e13, rel=5e-4)
    assert min_cut.macs_a_block(145_568, o, 16 * min_cut.beta(145_568), 16) == pytest.approx(9.383e5, rel=5e-4)
    assert all(s in r.rests_on for s in (min_cut.HE, min_cut.HARDNESS, min_cut.EXPORTS, "β_x = 0.37"))
    assert min_cut.bound(C, K, N=MACS, O=MACS * o, size=C) == min_cut.bound(C, K, None, N=MACS, O=MACS * o, size=C) == r


def test_the_bound_is_the_least_blocks_priced_at_c_and_never_below_one():
    n, o, size, c, k, bw, bx, b = POINTS[1]
    r = min_cut.bound(c, k, 0, N=n, O=o, size=size, beta_w=bw, beta_x=bx, b=b)
    m = min_cut.blocks_at_least(n, o, k, bw * bx, b)
    assert r.blocks == m and r.slowdown == m * c / size > 1e5 and r.macs_a_block == pytest.approx(n / m)
    assert r.slowdown >= min_cut.slowdown_at_least(n, o, size, c, k, bw * bx, b) > 0.99 * r.slowdown
    shared = min_cut.bound(c, k, 0, N=n, O=o, size=size, beta_w=bw, beta_x=bx, b=b, F=min_cut.share(3))
    assert shared.statement == min_cut.HE_SHARED and 1 < shared.slowdown < r.slowdown
    cheap = min_cut.bound(c, k, 0, N=n, O=o, size=size, beta_w=2.5, beta_x=bx, b=b)
    assert 1 < cheap.slowdown < r.slowdown
    assert min_cut.bound(c, k, 0, N=n, O=o, size=size).beta_x == min_cut.beta(k) == 15.9


def test_past_the_last_row_or_on_no_work_the_bound_says_nothing():
    r = min_cut.bound(C, 1e9, N_OUT, N=MACS, O=MACS / 128, size=C)
    assert (r.slowdown, r.blocks, r.macs_a_block, r.beta_x) == (1, 0, math.inf, None)
    assert "no bound at k = 1e+09" in r.rests_on
    assert min_cut.bound(C, 1e9, N_OUT, N=MACS, O=MACS / 128, size=C, beta_x=0.3).macs_a_block < math.inf
    assert min_cut.bound(C, 1e6, N_OUT, N=0, O=0, size=C).slowdown == 1


#: Lean's `workHEOn` as its file defines it, and both bounds of `HECharged` as its file concludes them
WORK_ON = "noncomputable def workHEOn (T S : ℕ) (q b O : ℝ) : ℝ := T + √(S * O / q) / 2 + √(S * T / (q * b)) / 2"
HE_CHARGED = ("(W.N : ℝ) ≤ ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) + workHEOn T S (βw * βx) b W.O ∧ "
              "((∀ s, γ ≤ (W.mac s).card) → (∀ u ∈ Gᶜ, (P.gates u).card ≤ c) → "
              "(γ : ℝ) * W.N ≤ (Gᶜ.card : ℝ) * c + γ * workHEOn T S (βw * βx) b W.O)")


def work_he_on(t, s, q, b, o):
    """`WORK_ON`, transcribed."""
    return t + math.sqrt(s * o / q) / 2 + math.sqrt(s * t / (q * b)) / 2


def test_the_charged_count_is_the_lean_statement_which_no_lock_lists_a_proof_of():
    """`macs_on` against `workHEOn`, transcribed from the text pinned here, and `HECharged`'s two bounds as its file
    concludes them: `N` is at most the MACs the charged blocks hold whole and `workHEOn`, and with MACs of at least
    `γ` gates and charged blocks of at most `c`, `γ N ≤ x c + γ workHEOn`, which `bound_charged` solves at
    `x c / γ`. At every block under one budget it is `HE`'s count. The statement is in Lean and no lock lists a proof of
    it, so the lock records neither it nor `workHEOn`: their text is read from the files that define `workHE` and
    state `HE`, where they are. The period's bound on the untied words and an untied word's bits are no Lean
    statement."""
    assert WORK_ON in lean_text(f"{min_cut.AREA}.workHE") and HE_CHARGED in lean_text(min_cut.HE)
    assert f"def {min_cut.HE_CHARGED.rsplit('.', 1)[1]} : Prop :=" in lean_text(min_cut.HE)
    assert lean.status(min_cut.HE_CHARGED) == "absent"
    assert lean.status(min_cut.UNTIED_WORDS) == lean.status(min_cut.WORD_BITS) == "draft"
    for (_, o, _, _, k, bw, bx, b), m in itertools.product(POINTS, (1.0, 7.0, 1e6)):
        q = bw * bx
        for t, sq in ((m * k, m * k * k), (3 * m * k, 5 * m * k * k), (0.0, 0.0)):
            assert min_cut.macs_on(t, sq, o, q, b) == work_he_on(t, sq, q, b, o)
        assert min_cut.macs_on(m * k, m * k * k, o, q, b) == pytest.approx(min_cut.macs_at_most(m, k, o, q, b),
                                                                         rel=1e-13)
    r = min_cut.bound_charged(C, TIED, blocks=54_300, untied=UNTIED, untied_macs=FEED, w=20_435, **SERVED)
    q = r.beta_w * r.beta_x
    assert r.charged == 32 and r.charged_macs == 32 * C / r.mac_gates and r.mac_gates == C / MACS
    assert r.untied_macs == FEED and r.macs == pytest.approx(
        FEED + r.charged_macs + work_he_on(r.total, r.squares_by_q * q, q, r.b,
                                           SERVED["O"] / MACS * (r.macs - FEED)), rel=1e-12)
    assert r.slowdown == pytest.approx(54_300 * C * MACS / (r.macs * C), rel=1e-12)


#: the served policy with every tied value right: the fresh input bits a block reads, a salt's among them
TIED = 145_568 + 512
#: the served model's work a block: its MACs, their output wires at a bf16 output word a position's share, its gates
SERVED = {"N": MACS, "O": MACS * WORDS * 16 / POSITION, "size": C}
#: a bound on the untied words of a monitoring period's accepted execution phases: the function's input, a sampled
#: proof's bound over the period to give
UNTIED = 671_369
#: the MACs that feed them, at most: the most a unit of ticket weight buys, 2,048 for each word of the least weight
FEED = 2_048 * UNTIED
#: the blocks of the execution phase `UNTIED` is the bound of: the served record's run, a block a slot
RUN_BLOCKS = 4
#: a block's y words, 35 positions of the linears' output words: the most of them one block has untied
BLOCK_WORDS = 35 * (WORDS - 128_256)
#: the execution phases of a week, a month of 30 days and a year of 365, at 600 s a phase
WEEK, MONTH, YEAR = 1_008, 4_320, 52_560


def untied_over(blocks, phases):
    """The bound on the untied words of `blocks` blocks served over `phases` execution phases, every node at its
    bound and all accepted. A node's bound is a share of one phase's ticket weight whatever the period's length, and
    `UNTIED` is it for a phase of `RUN_BLOCKS` blocks. So a node has `UNTIED / RUN_BLOCKS` for each block of one of
    its phases, and so have any number of nodes together: the untied words a block grow with nothing but the
    period's shortness, and a row that sets a node's `UNTIED` beside the blocks of a faster node understates them."""
    return UNTIED / RUN_BLOCKS * blocks / phases


def squares(r):
    """`S` where (H) is asked once: `Q` at that one `q`."""
    return pytest.approx(r.squares_by_q * r.beta_w * r.beta_x, rel=1e-12)


def charged(blocks, w=math.inf, **over):
    """The count alone at the served policy's block, (H) once: none of the MACs that feed the untied words."""
    return min_cut.bound_charged(C, TIED, blocks=blocks, w=w,
                                 **{"untied": UNTIED, "untied_macs": 0, **SERVED, **over})


def test_with_no_untied_word_the_charged_bound_is_the_bounds_and_with_every_block_charged_it_is_one():
    """The limits. No untied word: `bound`'s slowdown, whatever the threshold and the block count. Every block
    charged: 1. At `w = 0` a block that isn't charged reads `k`, so (H) is taken at `k` and `S` is `blocks · k²`.
    With no threshold no block is charged and every word may be in one block."""
    tied = min_cut.bound(C, TIED, **SERVED)
    for blocks, w in itertools.product((1, 54_300, 1e15), (0, 5, math.inf)):
        r = min_cut.bound_charged(C, TIED, N_OUT, blocks=blocks, untied=0, untied_macs=0, w=w, **SERVED)
        assert r.slowdown == pytest.approx(tied.slowdown, rel=1e-12) and r.charged == 0 == r.charged_macs
        assert (r.budget, r.beta_x, r.total, squares(r)) == (TIED, tied.beta_x, blocks * TIED, blocks * TIED ** 2)
    for n, o, size, c, k, bw, bx, b in POINTS:
        work = {"N": n, "O": o, "size": size, "beta_w": bw, "beta_x": bx, "b": b}
        assert min_cut.bound_charged(c, k, blocks=7.0, untied=0, untied_macs=0, **work).slowdown == pytest.approx(
            min_cut.bound(c, k, **work).slowdown, rel=1e-12)
    every = charged(100, 0)
    assert every.charged == 100 and every.slowdown == 1 == every.charged_of_blocks
    assert every.charged_macs == 100 * MACS and 0.99 < every.charged_of_macs < 1
    zero = charged(1e6, 0)
    assert zero.charged == UNTIED and zero.budget == TIED and zero.beta_x == min_cut.beta(TIED)
    assert zero.total == 1e6 * TIED + 32 * UNTIED and squares(zero) == 1e6 * TIED ** 2
    none = charged(54_300)
    assert none.w == math.inf and none.charged == 0 == none.charged_macs == none.charged_of_macs
    assert none.budget == TIED + 32 * UNTIED and none.beta_x == min_cut.beta(none.budget) == 2.7
    assert squares(none) == 54_300 * TIED ** 2 + 32 * UNTIED * (2 * TIED + 32 * UNTIED)
    assert none.slowdown == charged(54_300, UNTIED).slowdown > charged(54_300, UNTIED - 1).slowdown
    assert charged(54_300, UNTIED).charged == 0 < charged(54_300, UNTIED - 1).charged == 1
    assert none.statement == min_cut.HE_CHARGED
    assert none.premises == (min_cut.HARDNESS, min_cut.EXPORTS, min_cut.UNTIED_WORDS, min_cut.WORD_BITS)
    assert all(x in none.rests_on for x in (*none.premises, min_cut.HE_CHARGED, "no threshold"))
    assert "w = 20435, fixed before the period" in charged(54_300, 20_435).rests_on


def test_a_stated_most_a_block_reads_lowers_the_budget_and_charges_no_block_below_the_threshold():
    """`most`, the most untied words one block reads where its caller states one. Unstated it is `untied`. Below the
    threshold no block is charged, (H) is taken at `k + word · most` and `S` at `most` a block; above it the count is
    the threshold's. It never lowers the bound, and the best threshold is searched up to it."""
    assert charged(54_300).most == UNTIED == charged(54_300, most=10 * UNTIED).most
    under = charged(54_300, 20_435, most=20_000)
    assert under.charged == 0 == under.charged_macs and under.budget == TIED + 32 * 20_000
    assert squares(under) == 54_300 * TIED ** 2 + 32 * UNTIED * (2 * TIED + 32 * 20_000)
    assert under.slowdown == charged(54_300, most=20_000).slowdown > charged(54_300, 20_435).slowdown
    assert "no block charged, none reading more than the threshold w = 20435" in under.rests_on
    assert "at most 20000 of them in one block" in under.rests_on
    assert charged(54_300, 20_435, most=20_435.5).charged == 0 == charged(54_300, 20_435.5, most=20_436.5).charged - 32
    assert min_cut.best_w(C, TIED, blocks=2_379, untied=UNTIED, untied_macs=0, most=335_684.5, **SERVED) == 335_684
    over = charged(54_300, 20_435, most=20_436)
    assert (over.charged, over.budget, over.squares_by_q) == (32, TIED + 32 * 20_435, charged(54_300, 20_435).squares_by_q)
    few = charged(54_300, most=10)
    assert squares(few) == 54_300 * TIED ** 2 + 32 * 543_000 * (2 * TIED + 320)
    for blocks, w, most in itertools.product((1e4, 54_300, 1e15), (0, 1_582, math.inf), (0, 10, 20_000, UNTIED)):
        assert charged(blocks, w, most=most).slowdown >= charged(blocks, w).slowdown
    assert min_cut.best_w(C, TIED, blocks=54_300, untied=UNTIED, untied_macs=0, most=20_000, **SERVED) == 20_000
    assert charged(54_300, most=10).slowdown == pytest.approx(37_518.7, abs=0.05)


def test_the_charged_bound_falls_with_the_untied_words_and_says_nothing_past_the_last_row():
    """More untied words never raise it, and it is never above `bound`'s. A block charged whole stays in the count:
    its MACs are the most of what the period holds where few blocks are charged. A MAC of more gates leaves a charged
    block fewer of them. Where a block that isn't charged may read past (H)'s last row there is no bound."""
    tied = min_cut.bound(C, TIED, **SERVED).slowdown
    for blocks, w in itertools.product((1e4, 54_300, 25_165_824, 1e15), (0, 1_582, 20_435, math.inf)):
        rows = [min_cut.bound_charged(C, TIED, blocks=blocks, untied=e, untied_macs=0, w=w, **SERVED).slowdown
                for e in (0, 1, 1e3, UNTIED, 10 * UNTIED)]
        assert rows == sorted(rows, reverse=True) and rows[0] == pytest.approx(tied, rel=1e-12) and rows[-1] >= 1
    few = charged(54_300, 20_435)
    assert few.charged_of_blocks == 32 / 54_300 and 0.89 < few.charged_of_macs < 0.90
    assert charged(54_300, 20_435, mac_gates=2 * C / MACS).slowdown > few.slowdown
    assert charged(54_300, 20_435, mac_gates=2 * C / MACS).charged_macs == few.charged_macs / 2
    past = min_cut.bound_charged(C, TIED, blocks=54_300, untied=1e8, untied_macs=0, **SERVED)
    assert past.budget > min_cut.BETA[-1][0] and (past.slowdown, past.macs, past.beta_x) == (1, math.inf, None)
    assert "no bound with no threshold" in past.rests_on
    assert min_cut.bound_charged(C, TIED, blocks=0, untied=UNTIED, untied_macs=FEED, **SERVED).slowdown == 1
    assert min_cut.bound_charged(C, TIED, blocks=54_300, untied=UNTIED, untied_macs=0, N=0, O=0, size=C).slowdown == 1


def test_the_charged_count_alone_at_the_served_policy():
    """The count alone at the served policy's block: every untied word possibly in one block, and none of the MACs
    that feed them. Every tied value right: 38 thousand, `bound`'s. One budget for every block at the bits half the words
    are: 6. Charging a block for one untied word leaves nothing below a million blocks and everything from 1e17.
    With no threshold a period of ten thousand blocks keeps six thousand; a threshold adds to it from about fifty
    thousand blocks, where two blocks are charged. Every row is one node's at the run's rate, `RUN_BLOCKS` blocks a
    phase, where its bound is `UNTIED`: more blocks are a longer period, never a larger or a faster fleet. At a
    fleet's rate the untied words outnumber the blocks, and charging a block for one untied word leaves nothing."""
    assert min_cut.bound(C, TIED, **SERVED).slowdown == pytest.approx(37_653.4, abs=0.05)
    assert min_cut.bound(C, TIED + 10_688_197, **SERVED).slowdown == pytest.approx(6.3, abs=0.05)
    for blocks, at_zero in ((1e4, 1.0), (54_300, 1.0), (1e6, 1.5), (1e17, 37_653.4), (1e19, 37_653.4)):
        assert charged(blocks, 0).slowdown == pytest.approx(at_zero, abs=0.05)
    for blocks, w, slowdown in ((1e4, math.inf, 6_051.8), (54_300, math.inf, 10_795.6), (54_300, 20_435, 1_521.9),
                                (54_300, 223_789, 12_675.9), (1e6, 223_789, 28_826.5),
                                (25_165_824, 1_582, 21_392.6), (25_165_824, 20_344, 35_552.1),
                                (25_165_824, math.inf, 13_521.7), (1e15, 0, 37_652.2),
                                (1e15, math.inf, 13_529.4)):
        assert charged(blocks, w).slowdown == pytest.approx(slowdown, abs=0.05), (blocks, w)
    assert charged(1.651e15, 0, untied=untied_over(1.651e15, WEEK), most=BLOCK_WORDS).slowdown == 1
    assert min_cut.best_w(C, TIED, blocks=1e4, untied=UNTIED, untied_macs=0, **SERVED) == UNTIED
    for blocks in (54_300, 1e6):
        w = min_cut.best_w(C, TIED, blocks=blocks, untied=UNTIED, untied_macs=0, **SERVED)
        assert w == 223_789 == UNTIED // 3 and charged(blocks, w).charged == 2
        assert all(charged(blocks, w).slowdown >= charged(blocks, x).slowdown for x in (0, 1_582, 20_435, math.inf))
    assert min_cut.best_w(C, TIED, blocks=1e4, untied=0, untied_macs=0, **SERVED) == 0


#: the period's rows at the served class: (blocks, the execution phases they are served over, the most untied words
#: one block reads, the best threshold with one β_x, the blocks charged at it, the slowdown at it, the slowdown with
#: one β_x at the top budget and no block charged, and the slowdown with a β_x a block and no threshold). Every
#: row's untied words are `untied_over` its blocks and phases, and nothing else. A node's rows are a node at the
#: run's rate, `RUN_BLOCKS` blocks a phase, whose bound is `UNTIED` however long it serves; that one block of it
#: reads at most half the node's untied words is these rows' statement of their record, which the function takes as
#: `most`: a sampled tile's rows are two blocks' or more. A datacenter's and the world's are scaled: every machine
#: at its data-sheet peak for a week, a month and a year, every node at its bound and all accepted, where a node's
#: bound is more than a block's words, so one block may be untied whole.
PERIOD = ((2_379, 2_379 / RUN_BLOCKS, UNTIED // 2, 335_684, 0, 3_677.1, 3_677.1, 4_107.8),
          (1e4, 1e4 / RUN_BLOCKS, UNTIED // 2, 335_684, 0, 7_883.9, 7_883.9, 10_923.1),
          (54_300, 54_300 / RUN_BLOCKS, UNTIED // 2, 223_789, 2, 12_564.7, 11_860.1, 24_122.5),
          (1e6, 1e6 / RUN_BLOCKS, UNTIED // 2, 223_789, 2, 28_795.0, 13_424.7, 36_407.0),
          (25_165_824, 25_165_824 / RUN_BLOCKS, UNTIED // 2, 20_344, 32, 35_550.2, 13_525.2, 37_601.8),
          (1e10, 1e10 / RUN_BLOCKS, UNTIED // 2, 20_344, 32, 37_647.8, 13_529.4, 37_653.3),
          (1.651e15, WEEK, BLOCK_WORDS, 911_841, 3.01487e11, 1_614.1, 52.8, 53.2),
          (7.077e15, MONTH, BLOCK_WORDS, 825_882, 3.32926e11, 4_787.8, 202.2, 209.6),
          (8.61e16, YEAR, BLOCK_WORDS, 245_435, 1.12024e12, 20_422.0, 1_336.6, 1_844.3),
          (6.191e18, WEEK, BLOCK_WORDS, 911_841, 1.13053e15, 1_614.1, 52.8, 53.2),
          (2.653e19, MONTH, BLOCK_WORDS, 825_882, 1.24806e15, 4_787.8, 202.2, 209.6),
          (3.228e20, YEAR, BLOCK_WORDS, 245_435, 4.19992e15, 20_422.0, 1_336.6, 1_844.3))
#: a β_x a block with its own best threshold at a fleet's week, month and year: the threshold and the slowdown
PERIOD_ROWS_W = ((893_962, 1_694.1), (762_988, 5_779.8), (601_611, 24_820.5))


def test_the_period_at_the_served_class_under_each_reading():
    """The three readings side by side at the served class, each under its premise's name, each row with its block
    count and its threshold, the MACs that feed the untied words charged whole. A node at the run's rate. One β_x at
    the top budget, no block charged: (H) is asked of every block where the block that reads the most untied words
    reads, and the figure stops at 13.5 thousand however long the period. One β_x with the best threshold: the same
    up to ten thousand blocks, where no block is charged, and near `bound`'s 38 thousand once the period is long. A
    β_x a block: no threshold is needed, and it is the largest of the three. A fleet at the census's rate: it holds
    166.5 untied words a block over a week, 38.85 over a month and 3.19 over a year, whatever its nodes, and a block
    may be untied whole. With no threshold both readings fall to 53, about 205 and under two thousand. With the
    best threshold one β_x gives 1.6, 4.8 and 20 thousand, the charged blocks under a five-thousandth of the blocks,
    and the year's threshold is the last whose budget (H)'s third row covers; a β_x a block needs its own threshold
    there too, and then gives 1.7, 5.8 and 25 thousand. At `w = 0` every block of a fleet may read an untied word,
    every block is charged and the bound is 1: there the threshold is a parameter the figure needs, never a default.
    One bound for every block at what the worst block reads is 1 as well: a block untied whole is the served policy's
    with nothing tied. A β_x a block is no statement of `HECharged`'s at any row."""
    for blocks, phases, most, w, over, at_w, at_none, at_rows in PERIOD:
        untied, fleet = untied_over(blocks, phases), most == BLOCK_WORDS
        assert untied == (pytest.approx(UNTIED / RUN_BLOCKS / phases * blocks, rel=1e-12) if fleet else UNTIED)
        terms = {"blocks": blocks, "untied": untied, "untied_macs": 2_048 * untied, "most": most, **SERVED}
        rows = min_cut.bound_charged(C, TIED, rows=True, **terms)
        assert rows.slowdown == pytest.approx(at_rows, abs=0.05) and rows.charged == 0 and rows.budget == TIED + 32 * most
        assert rows.statement == min_cut.HE_ROWS and "which no guarantee states" in rows.rests_on
        assert "at what each block reads" in rows.rests_on
        if fleet:
            at, with_w = PERIOD_ROWS_W[(WEEK, MONTH, YEAR).index(phases)]
            assert min_cut.best_w(C, TIED, rows=True, **terms) == at and at_w > at_rows > at_none
            assert min_cut.bound_charged(C, TIED, w=at, rows=True, **terms).slowdown == pytest.approx(with_w, abs=0.05)
            assert with_w > at_w and untied / blocks == pytest.approx(UNTIED / RUN_BLOCKS / phases, rel=1e-12)
            zero = min_cut.bound_charged(C, TIED, w=0, **terms)
            assert untied >= blocks and (zero.slowdown, zero.charged) == (1, blocks)
        else:
            assert rows.slowdown >= max(at_w, at_none) - 0.05
            assert min_cut.best_w(C, TIED, rows=True, **terms) == most
        assert min_cut.best_w(C, TIED, **terms) == w, blocks
        two, none = min_cut.bound_charged(C, TIED, w=w, **terms), min_cut.bound_charged(C, TIED, **terms)
        assert two.charged == pytest.approx(over, rel=1e-5) and none.charged == 0, blocks
        assert two.slowdown == pytest.approx(at_w, abs=0.05) and none.slowdown == pytest.approx(at_none, abs=0.05)
        assert f"w = {w}" in two.rests_on and "no threshold" in none.rests_on
        assert two.statement == none.statement == min_cut.HE_CHARGED and "once, at k = " in two.rests_on
        assert (two.reading, none.reading, rows.reading) == (
            f"one β_x, with the threshold w = {w}", "one β_x at the top budget, no block charged",
            "a β_x a block, no threshold")
        assert all(f"fed by at most {2_048 * untied:.6g} MACs" in r.rests_on for r in (two, none, rows))
        assert two.slowdown >= none.slowdown and (over > 0 or two.slowdown == none.slowdown)
        assert two.slowdown <= charged(blocks, w, untied=untied, most=most).slowdown
        for r in (two, none, rows):
            assert r.gates == pytest.approx(r.macs * C / MACS, rel=1e-12) == pytest.approx(blocks * C / r.slowdown)
            assert r.share == 1 / r.slowdown and r.gates <= blocks * C
    assert TIED + 32 * 245_435 == min_cut.BETA[2][0] and BLOCK_WORDS == 13_189_120
    assert [round(UNTIED / RUN_BLOCKS / p, 2) for p in (WEEK, MONTH, YEAR)] == [166.51, 38.85, 3.19]
    assert all(1e-5 < row[4] / row[0] < 2e-4 for row in PERIOD[6:]) and PERIOD[6][5:] == PERIOD[9][5:]
    assert TIED + 32 * BLOCK_WORDS == K + 512 and min_cut.bound(C, K + 512, **SERVED).slowdown == 1
    assert min_cut.bound(C, TIED, **SERVED).slowdown < 37_653.45
    assert min_cut.bound_charged(C, TIED, w=5, rows=True, **terms).reading == "a β_x a block, with the threshold w = 5"
    assert lean.status(min_cut.HE_ROWS) == "draft"


def test_a_beta_a_block_covers_every_way_the_untied_words_fall():
    """`rows`, (H) asked of each block at the table's row for what it reads: `Q` is at least the sum of each block's
    square over its own `q`, however the untied words fall among the blocks, a whole number each. Every such way is
    tried, at a `k` below a row, on one, a word short of one and just under one, and with words wide enough that a
    few cross several rows or pass the last, where there is no bound. With one `β_x` the same sum at the budget's `q`
    is under its `Q` too, and a `β_x` a block is never the lower figure. `rows` takes no `β_x`, and with no untied
    word it is `bound`'s."""
    row = min_cut.BETA[1][0]
    for k, word in ((TIED, 32), (row, 32), (row - 32, 32), (row - 10, 32), (min_cut.BETA[0][0], 32), (TIED, 4e5),
                    (TIED, 7e6), (TIED, 3e8)):
        for blocks, untied, most in itertools.product((1, 3), (1, 2, 5), (1, 2, 5)):
            terms = {"blocks": blocks, "untied": untied, "untied_macs": 0, "most": most, "word": word, **SERVED}
            ways = [x for x in itertools.product(range(min(most, untied) + 1), repeat=blocks) if sum(x) <= untied]
            by_row, once = (min_cut.bound_charged(C, k, rows=r, **terms) for r in (True, False))
            if by_row.beta_x is None:
                assert k + word * min(most, untied) > min_cut.BETA[-1][0] and by_row.slowdown == 1 == once.slowdown
                continue
            assert by_row.squares_by_q >= (1 - 1e-12) * max(
                sum((k + word * x) ** 2 / (16 * min_cut.beta(k + word * x)) for x in way) for way in ways), (k, word)
            assert once.squares_by_q >= (1 - 1e-12) * max(
                sum((k + word * x) ** 2 / (16 * once.beta_x) for x in way) for way in ways), (k, word)
            assert by_row.squares_by_q <= once.squares_by_q * (1 + 1e-12) and by_row.slowdown >= once.slowdown
    with pytest.raises(ValueError, match="takes no `beta_x`"):
        min_cut.bound_charged(C, TIED, blocks=1, untied=1, untied_macs=0, rows=True, beta_x=2.7, **SERVED)
    tied = min_cut.bound(C, TIED, **SERVED)
    none = min_cut.bound_charged(C, TIED, blocks=54_300, untied=0, untied_macs=0, rows=True, **SERVED)
    assert none.slowdown == pytest.approx(tied.slowdown, rel=1e-12) and none.beta_x == tied.beta_x


def test_a_slowdown_is_quoted_beside_the_work_no_block_is_credited_for():
    """`capped`: a share of a block's work that nothing checks is the developer's whole, so the slowdown is at most
    one over that share whatever the count gives: a thousandth leaves under a thousand of `bound`'s 38 thousand. It is
    `theorem.multiple_bound` at that dark share and the count's rate, and with no such share the count's own."""
    tied = min_cut.bound(C, TIED, **SERVED).slowdown
    assert min_cut.capped(tied, 0) == pytest.approx(tied, rel=1e-12)
    assert min_cut.capped(tied, 0.001) == pytest.approx(974.1, abs=0.05)
    assert min_cut.capped(tied, 0.001) < 1_000 < min_cut.capped(tied, 0.00097)
    assert min_cut.capped(math.inf, 0.002) == 500 and min_cut.capped(math.inf, 0) == math.inf
    for slowdown, share in itertools.product((1.0, 6.3, tied, 1e30), (0.0, 1e-4, 0.002, 0.5, 1.0)):
        got = min_cut.capped(slowdown, share)
        assert got == pytest.approx(theorem.multiple_bound(0.4, share, 1 / slowdown), rel=1e-12)
        assert 1 <= got <= slowdown and (share == 0 or got <= 1 / share)
    assert "is in no term here (`capped`)" in charged(54_300).rests_on


def test_the_class_reads_the_files_rows_so_other_rows_in_the_registry_are_refused():
    with pytest.raises(ValueError, match="edit the file"):
        premises(REG.override({"premise.training-cut-hardness.beta": [[1e5, 16.0], [1e9, 1.0]]}))
    with pytest.raises(ValueError, match="edit the file"):
        build(REG.override({"premise.training-cut-hardness.beta": [[1e5, 16.0]]}), "reference-remote-h200")


def reference_profile():
    s = build(REG, "reference-remote-h200")
    return s, s.families[0].profile(s.model, s.run)


def test_the_class_is_the_macs_a_block_at_the_profiles_costs_and_shape():
    """A weight value at the profile's `b_w`, an activation value at `β(k)` and at most its `b_a` bits, an output at
    `b_a` bits and wires, over the model's own outputs a product; nothing past the last row."""
    arg = CLASSES["min-cut"]
    assert arg.cites == (min_cut.HE, min_cut.HARDNESS, min_cut.EXPORTS) and arg.counts == "kappa"
    assert lean.status(arg.cites[0]) == "proved" and lean.weakest(arg.cites) == "conditional"
    assert premise_status(arg.cites, 2e7) == "assumed" and premise_status(arg.cites, 1e9) == "contested"
    _, p = reference_profile()
    a = arg.allowance(p)
    assert min_cut.beta(4e8) < min_cut.beta(2e7) < p.b_a < min_cut.beta(8e6)
    for k, beta in ((1e5, p.b_a), (8e6, p.b_a), (2e7, min_cut.beta(2e7)), (4e8, min_cut.beta(4e8))):
        assert a(FlopBudget(1e30, k)) == min_cut.macs_a_block(k, p.outputs_per_product * p.b_a, p.b_w * beta, p.b_a)
    assert a(FlopBudget(1e30, 8e8)) < math.inf and a(FlopBudget(1e30, 8.1e8)) == math.inf
    with pytest.raises(Outside, match="outputs_per_product"):
        arg.allowance(replace(p, outputs_per_product=None))


def test_outputs_per_product_is_the_models_shape():
    """A matrix's forward product gives `d_out` values a token and its input gradient `d_in`, the head's among them,
    over the products `κ` counts; a kind that doesn't count whole matrices' products states none, and is outside."""
    s, p = reference_profile()
    layers = [m for g in instances.LAYERED for m in s.model.matrices.get(g, ())]
    head = list(s.model.matrices.get("head", ()))
    products = 2 * sum(m.size * m.active for m in layers) + sum(m.size * m.active for m in head)
    values = sum((m.d_out + m.d_in) * m.active for m in layers) + sum(m.d_in * m.active for m in head)
    assert p.outputs_per_product == pytest.approx(values / products, rel=1e-12)
    assert 1 / max(m.d_in for m in layers) < p.outputs_per_product < 1 / min(m.d_out for m in layers)
    lora = family(REG, "lora-r16").profile(s.model, s.run)
    assert lora.outputs_per_product is None
    with pytest.raises(Outside, match="outputs_per_product"):
        CLASSES["min-cut"].allowance(lora)


def test_periods_bounds_gives_the_rows_of_one_class_alone_and_says_what_they_cover(tmp_path, monkeypatch, capsys):
    """On the min-cut bound alone the declared blocks hold almost none of the rate, so onsite's figure is remote's:
    the uncertified routes' and the unregistered pool's, which the header says beside the rows. Without the flag the
    header has no such sentence."""
    import periods

    names = ("reference-onsite-h200", "reference-remote-h200")
    alone = periods.only(REG, names, ["min-cut"])
    assert [b.name for b in build(alone, names[0]).bounds] == ["min-cut"]
    assert len(build(REG, names[0]).bounds) == len(CLASSES)
    with pytest.raises(ValueError, match="not classes"):
        periods.only(REG, names, ["min-cut", "no-such-class"])

    argv = ["periods.py", *(a for n in names for a in ("--scenario", n))]
    monkeypatch.setattr(sys, "argv", [*argv, "--bounds", "min-cut", "--out", str(tmp_path)])
    periods.main()
    text = capsys.readouterr().out
    assert "Each row is bounded by `min-cut` alone (`--bounds`)" in text
    doc = json.loads((tmp_path / "periods.json").read_text())
    assert doc["bounds"] == ["min-cut"] and len(doc["rows"]) == 2 * len(periods.FLEETS) * len(periods.PERIODS)
    onsite, remote = ([x for x in doc["rows"] if x["scenario"] == n] for n in names)
    for x, y in zip(onsite, remote, strict=True):
        assert x["parts"]["setter"] == y["parts"]["setter"] == "min-cut"
        assert x["slowdown"] == pytest.approx(y["slowdown"], rel=1e-12) and 2.9 < x["slowdown"] < 3.1
        assert x["scaled"]["per_window_share"] < 1e-5 < 0.99 < x["scaled"]["routes_share"]
    monkeypatch.setattr(sys, "argv", ["periods.py", "--scenario", names[1]])
    periods.main()
    assert "--bounds" not in capsys.readouterr().out
