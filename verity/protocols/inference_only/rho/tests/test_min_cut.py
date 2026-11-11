"""The min-cut bound: `rho.min_cut` against the Lean text it computes, its rows of `β(k)`, the bound at a policy, the
class in `CLASSES`, and `periods.py --bounds`."""

import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from rho import instances, lean, min_cut
from rho.classes import CLASSES, FlopBudget, Outside, premise_status, premises
from rho.families import family
from rho.params import Registry
from rho.scenario import build

ROOT = Path(__file__).resolve().parents[5]
REG = Registry.load()
LOCK = json.loads((ROOT / "Security" / "lean-audit.json").read_text())


def lean_text(name: str) -> str:
    """The Lean file that defines `name`, the one module whose definitions the lock lists it among, as one line."""
    (module,) = (m for m, read in LOCK["reads"].items() if name in read["definitions"])
    return " ".join((ROOT / "Security" / f"{module.replace('.', '/')}.lean").read_text().split())


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
          (3.6e10, 2.8e8, 1.4e13, 1.4e13, 4.2e8, 16.0, 1.02, 16.0),
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
    per 2,447 MACs, and holds 3.6e10: one block does it, and the slowdown is 1. At the first row's `k` a block may
    finish under a million MACs."""
    assert (POSITION, WORDS) == (1_235_746_816, 505_088)
    o = WORDS * 16 / POSITION
    r = min_cut.bound(C, K, N_OUT, N=MACS, O=MACS * o, size=C)
    assert r.slowdown == 1 and r.blocks == pytest.approx(MACS / r.macs_a_block) and r.blocks < 1
    assert r.statement == min_cut.HE and r.premises == (min_cut.HARDNESS, min_cut.EXPORTS)
    assert (r.beta_w, r.b, r.F) == (16, 16, 1) and r.beta_x == min_cut.beta(K) == pytest.approx(1.017, rel=1e-3)
    assert r.alpha == min_cut.alpha(K, 16 * r.beta_x, 16)
    per_block = min_cut.macs_a_block(K, o, 16 * r.beta_x, 16)
    assert r.macs_a_block == pytest.approx(per_block) and per_block == pytest.approx(1.845e13, rel=5e-4)
    assert min_cut.macs_a_block(K, o, 16 * r.beta_x, 16, min_cut.share(3)) == pytest.approx(2.450e13, rel=5e-4)
    assert min_cut.macs_a_block(145_568, o, 16 * min_cut.beta(145_568), 16) == pytest.approx(9.383e5, rel=5e-4)
    assert all(s in r.rests_on for s in (min_cut.HE, min_cut.HARDNESS, min_cut.EXPORTS, "β_x = 1.017"))
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
    assert min_cut.beta(4e8) < p.b_a < min_cut.beta(2e7)
    for k, beta in ((1e5, p.b_a), (2e7, p.b_a), (4e8, min_cut.beta(4e8))):
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
