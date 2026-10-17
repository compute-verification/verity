"""eps_budget: Dither's skippable share as served, per component, for each design of the registry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from verity_machine_model.eps import design, designs

spec = importlib.util.spec_from_file_location("pouw_eps_budget", Path(__file__).parents[1] / "eps_budget.py")
E = importlib.util.module_from_spec(spec)
sys.modules["pouw_eps_budget"] = E
spec.loader.exec_module(E)

CELLS = [(m, c, ph) for m in E.MODELS for c in E.CONTEXTS for ph in E.PHASES]


def test_the_strips_are_their_rank_over_depth():
    assert E.products("one-sided", "QKᵀ", 128) == (128, 64, 64)        # L/d = 25% logical, two limbs executed
    assert E.products("two-sided", "linears", 4096) == (4096, 0, 0)
    assert E.products("cohalf", "QKᵀ", 128) == (128, 64, 0)              # r = k/2 on one limb: checked
    assert E.products("co32c", "linears", 4096) == (4096, 3 * 32, 0)     # every digit limb is a committed word
    assert E.products("peel", "P·V", 1024) == (1024, 66, 66)             # a one-hot P row frees both limbs
    assert E.products("pv", "P·V", 1024) == (1024, 3 * 32, 0)
    assert E.products("pv", "P·V", 8192) == (8192, 4 * 32, 0)            # Ycum takes a third digit past 2K keys


@pytest.mark.parametrize(("leaf", "comp"), [("cohalf", "P·V"), ("pv", "linears"), ("peel", "QKᵀ")])
def test_a_leaf_with_no_price_on_a_product_is_refused(leaf, comp):
    with pytest.raises(ValueError, match="no price"):
        E.products(leaf, comp, 1024)


def test_a_checked_leaf_with_no_tensor_scalar_price_is_refused():
    noise = {**design("dither-c").noise, "linears": "cohalf"}
    with pytest.raises(ValueError, match="no tensor–scalar price for cohalf on linears"):
        E.parts("llama-3.1-8b", 2048, "prefill", noise)


def test_dither_1s_skips_its_peels_and_the_clean_lm_head_but_not_forming():
    s = E.summary("llama-3.1-8b", 2048, "prefill", design("dither-1s"))
    skipped = {r["component"] for r in s.rows if r["eps_mac"] or r["eps_time"]}
    assert skipped == {"linears", "lm_head", "QKᵀ", "P·V"}               # softmax and forming checked by C1′
    assert s.eps_mac == pytest.approx(0.0226, abs=5e-4)


def test_two_sided_meets_the_target_everywhere():
    for m, c, ph in CELLS:
        s = E.summary(m, c, ph, design("dither-2s"))
        assert s.eps_mac < 3e-3 and s.eps_time < 3e-3, (m, c, ph)
        assert s.gamma < 2.2, (m, c, ph)
        skipped = {r["component"] for r in s.rows if r["eps_mac"] or r["eps_time"]}
        assert skipped == {"P·V"}, (m, c, ph)


@pytest.mark.parametrize("name", ["dither-c", "dither-2s-pv"])
def test_checked_designs_skip_no_product_only_vector_work(name):
    d = design(name)
    for m, c, ph in CELLS:
        s, a = E.summary(m, c, ph, d), E.summary(m, c, ph, design("dither-2s"))
        assert s.eps_mac == 0, (m, c, ph)
        assert (s.eps_time > 0) == (ph == "prefill") and s.eps_time < 3e-3, (m, c, ph)   # decode prices products only
        if ph == "decode":
            assert s.gamma >= a.gamma - 1e-9, (m, c, ph)       # by products, committed digit limbs cost at least the peel


def test_the_designs_reproduce_skips_figures():
    """SKIP's calculator before the registry (#2234): A 0.131% at 8B 2K, Ap 0.078% at 8B 2K and Bc 0.145% at 8B 8K by
    honest time, P in one block and every served output free."""
    for name, eps, ctx in (("dither-2s", 0.0013098, 2048), ("dither-2s-pv", 0.00077950, 2048),
                           ("dither-c", 0.0014506, 8192)):
        q = E.design_skip(design(name))
        assert (q.model, q.ctx, q.phase) == ("llama-3.1-8b", ctx, "prefill"), name
        assert q.eps_time == pytest.approx(eps, rel=1e-4), name


def test_the_registry_skip_is_eps_budgets_number():
    """One ε_skip a design: its `[skip]` in benchmarks/pouw/machine_model/data/designs/ is design_skip's, to the three
    figures the registry writes."""
    for name in designs():
        d = design(name)
        got, want = sum(p.eps for p in d.skip.values()), E.design_skip(d).eps_time
        assert got == pytest.approx(want, rel=5e-3), (
            f"{name}: [skip] is {got:.3g}, eps_budget.py's design_skip gives {want:.3g}: write "
            f"[skip.served] eps = {float(f'{want:.3g}')} in data/designs/{name}.toml (or its base's)")


def test_pv_units_price_as_checked_strip_without_the_scaling_and_the_sink(monkeypatch):
    monkeypatch.setattr(E, "PV_SCALE_OPS", 0.0)
    for m, c, eps, gamma in (("llama-3.1-8b", 2048, 0.019613, 1.3753), ("llama-3.1-70b", 8192, 0.005081, 1.2920)):
        e, g = E.pv_ts(m, c, 0, sink=False)
        assert eps <= e < eps + 1e-4 and g == pytest.approx(gamma, abs=1e-4)   # the zero-instance routes add < 0.01%


def test_smaller_p_blocks_cost_more_eps_and_gamma():
    m = "llama-3.1-8b"
    got = {b: E.pv_ts(m, 2048, b) for b in (0, 256, 512, 1024)}
    assert got[256] > got[512] > got[1024] > got[0]
    assert E.pv_ts(m, 2048, 2048) == got[0]
    ap, bc = design("dither-2s-pv"), design("dither-c")
    for b in (256, 512, 1024):
        a, c = E.pv_block_point(m, 2048, ap, b), E.pv_block_point(m, 2048, bc, b)
        e = a["eps_pv"] / (1 + a["eps_pv"]) * a["pv_share"]
        assert a["eps"] == pytest.approx(e / (1 - e))                    # two-sided skips nothing off P V
        assert a["eps"] < c["eps"] and a["gamma"] < c["gamma"]


def test_decode_hides_the_edit_under_dram_and_the_zero_weight_gap_is_reported():
    for m, c, _ in CELLS:
        assert E.summary(m, c, "decode", design("dither-2s")).gamma < 1.05
        assert E.decode_dram(m, c) > 1


def test_main_writes_both_tables(tmp_path):
    assert E.main(["--md", str(tmp_path / "t.md"), "--json", str(tmp_path / "t.json")]) == 0
    text = (tmp_path / "t.md").read_text()
    assert "| dither-1s (Dither-1S) |" in text and "dither-2s: ε_skip MAC / time, γ" in text and "| **whole** |" in text
