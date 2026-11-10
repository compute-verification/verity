"""The min-cut bound: `rho.min_cut` against the Lean text it computes, its rows of `β(k)`, and the bound at a
policy."""

import itertools
import math
from pathlib import Path

import pytest
from rho import min_cut
from rho.params import Registry

ROOT = Path(__file__).resolve().parents[3]
REG = Registry.load()

#: `Nci.Cut.alphaHE` and `shareF` as Security/Definitions/Nci/Cut.lean defines them, and both conjuncts of
#: `Nci.Cut.Guarantees.HE` and `HEShared` as Security/Definitions/Nci/Guarantees/Cut.lean concludes them
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
          (3.6e10, 2.8e8, 1.4e13, 1.4e13, 4.2e8, 16.0, 1.02, 16.0),
          (1.0e18, 2.7e14, 3.0e21, 2.45e19, 2.0e7, 2.5, 4.0, 4.0))


def test_the_arithmetic_is_the_lean_statements():
    """The module's functions against each statement's sides, transcribed from the text pinned here: `HE` at `F = 1`
    and `q = β_w β_x`, `HEShared` at `F = shareF ω`."""
    defs = " ".join((ROOT / "Security" / "Definitions" / "Nci" / "Cut.lean").read_text().split())
    statements = " ".join((ROOT / "Security" / "Definitions" / "Nci" / "Guarantees" / "Cut.lean").read_text().split())
    assert ALPHA in defs and SHARE in defs
    assert HE in statements and HE_SHARED in statements
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


def test_the_least_blocks_and_the_cap_a_block_are_the_count_solved():
    """`blocks_at_least` is where the count meets `N`, `cap` is `N` over it whatever `N`, and the second conjunct is
    never above the first solved exactly: below 0 where the outputs' term outweighs the first."""
    for (n, o, size, c, k, bw, bx, b), share in itertools.product(POINTS, (1.0, min_cut.share(3))):
        q = bw * bx
        m = min_cut.blocks_at_least(n, o, k, q, b, share)
        assert min_cut.macs_at_most(m, k, o, q, b, share) == pytest.approx(n, rel=1e-9)
        below, above = (min_cut.macs_at_most(m * x, k, o, q, b, share) for x in (0.999, 1.001))
        assert below < n < above
        for scale in (1.0, 1e-3, 1e4):
            assert min_cut.cap(k, o / n, q, b, share) == pytest.approx(
                n * scale / min_cut.blocks_at_least(n * scale, o * scale, k, q, b, share), rel=1e-9)
        assert min_cut.slowdown_at_least(n, o, size, c, k, q, b, share) <= m * c / size * (1 + 1e-12)
    n, o, size, c, k, bw, bx, b = POINTS[2]
    assert min_cut.slowdown_at_least(n, o, size, c, k, bw * bx, b) < 0 < min_cut.blocks_at_least(n, o, k, bw * bx, b)


def test_beta_is_the_hardness_rows_and_there_is_none_past_the_last():
    """The rows are `premise.training-cut-hardness.beta`'s, with a source; `k` rises and `β` falls along them; (H) is
    credible up to the last row's `k`, and past it there is no `β`."""
    p = REG.param("premise.training-cut-hardness.beta")
    assert min_cut.BETA == tuple(map(tuple, p.value)) and p.source and p.status == "assumed"
    ks, betas = zip(*min_cut.BETA, strict=True)
    assert list(ks) == sorted(ks) and list(betas) == sorted(betas, reverse=True) and betas[-1] > 0
    assert REG["premise.training-cut-hardness.credible_bits"] == ks[-1]
    assert min_cut.beta(1) == min_cut.beta(ks[0]) == betas[0]
    for (k0, b0), (k1, b1) in itertools.pairwise(min_cut.BETA):
        assert min_cut.beta(k1) == b1
        assert min_cut.beta(math.sqrt(k0 * k1)) == pytest.approx((b0 + b1) / 2, rel=1e-12)
    assert min_cut.beta(ks[-1] * (1 + 1e-9)) is None and min_cut.beta(1e12) is None


#: the served policy's public parameters: gates, fresh input bits and output bits a block, and the θ-MACs a block holds
C, K, N_OUT, MACS = 13_818_131_814_643, 422_197_408, 25_165_824, 36_241_670_144
#: a position of Llama 3.2 1B's forward, from its shape: MACs and output words. 16 layers of q and o (2048 → 2048),
#: k and v (2048 → 512), gate and up (2048 → 8192) and down (8192 → 2048), and the head (2048 → 128,256)
POSITION = 16 * (2 * 2048 * 2048 + 2 * 2048 * 512 + 3 * 2048 * 8192) + 2048 * 128_256
WORDS = 16 * (2 * 2048 + 2 * 512 + 2 * 8192 + 2048) + 128_256


def test_the_bound_at_the_served_policy_is_one():
    """A block of the served policy may finish about 1.8e13 MACs of the served model's forward, one bf16 output word
    per 2,447 MACs, and holds 3.6e10: one block does it, and the slowdown is 1. At the first row's `k` the cap is
    under a million MACs."""
    assert (POSITION, WORDS) == (1_235_746_816, 505_088)
    o = WORDS * 16 / POSITION
    r = min_cut.bound(C, K, N_OUT, N=MACS, O=MACS * o, size=C)
    assert r.slowdown == 1 and r.blocks == pytest.approx(MACS / r.cap) and r.blocks < 1
    assert r.statement == min_cut.HE and r.premises == (min_cut.HARDNESS, min_cut.EXPORTS)
    assert (r.beta_w, r.b, r.F) == (16, 16, 1) and r.beta_x == min_cut.beta(K) == pytest.approx(1.017, rel=1e-3)
    assert r.alpha == min_cut.alpha(K, 16 * r.beta_x, 16)
    assert r.cap == pytest.approx(min_cut.cap(K, o, 16 * r.beta_x, 16)) == pytest.approx(1.845e13, rel=5e-4)
    assert min_cut.cap(K, o, 16 * r.beta_x, 16, min_cut.share(3)) == pytest.approx(2.450e13, rel=5e-4)
    assert min_cut.cap(145_568, o, 16 * min_cut.beta(145_568), 16) == pytest.approx(9.383e5, rel=5e-4)
    assert all(s in r.rests_on for s in (min_cut.HE, min_cut.HARDNESS, min_cut.EXPORTS, "β_x = 1.017"))
    assert min_cut.bound(C, K, 0, N=MACS, O=MACS * o, size=C) == r


def test_the_bound_is_the_least_blocks_priced_at_c_and_never_below_one():
    n, o, size, c, k, bw, bx, b = POINTS[1]
    r = min_cut.bound(c, k, 0, N=n, O=o, size=size, beta_w=bw, beta_x=bx, b=b)
    m = min_cut.blocks_at_least(n, o, k, bw * bx, b)
    assert r.blocks == m and r.slowdown == m * c / size > 1e5 and r.cap == pytest.approx(n / m)
    assert r.slowdown >= min_cut.slowdown_at_least(n, o, size, c, k, bw * bx, b) > 0.99 * r.slowdown
    shared = min_cut.bound(c, k, 0, N=n, O=o, size=size, beta_w=bw, beta_x=bx, b=b, F=min_cut.share(3))
    assert shared.statement == min_cut.HE_SHARED and 1 < shared.slowdown < r.slowdown
    cheap = min_cut.bound(c, k, 0, N=n, O=o, size=size, beta_w=2.5, beta_x=bx, b=b)
    assert 1 < cheap.slowdown < r.slowdown
    assert min_cut.bound(c, k, 0, N=n, O=o, size=size).beta_x == min_cut.beta(k) == 15.9


def test_past_the_last_row_or_on_no_work_the_bound_says_nothing():
    r = min_cut.bound(C, 1e9, N_OUT, N=MACS, O=MACS / 128, size=C)
    assert (r.slowdown, r.blocks, r.cap, r.beta_x) == (1, 0, math.inf, None) and "no bound at k = 1e+09" in r.rests_on
    assert min_cut.bound(C, 1e9, N_OUT, N=MACS, O=MACS / 128, size=C, beta_x=0.3).cap < math.inf
    assert min_cut.bound(C, 1e6, N_OUT, N=0, O=0, size=C).slowdown == 1
