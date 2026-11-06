"""eps_budget: Dither's skippable share as served, per component, for each design of the registry."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from verity_machine_model.eps import design, designs, eps, machine
from verity_machine_model.model import device, devices

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
    """The program's target, ε < 1%, as `design_skip` quotes it."""
    for m, c, ph in CELLS:
        s = E.summary(m, c, ph, design("dither-2s"), credit="floor")
        assert s.eps_mac < 1e-2 and s.eps_time < 1e-2, (m, c, ph)
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
    """Corollary D3's ε_skip, P in one block and every served output free: A's 0.327% at 8B 2K on VF3q-16 with vsum's
    sums (SPEC §2.9's vector, its peel at the checked kernels' apv_l7qs) and 0.292% with vfold's, the served V producer
    of every peel, and each design's on VF3r-16, the vector of record, whose ♭ is about 10% cheaper, each at its own ♭
    (dither-2s-fill's filler takes ♭ from 17.221 to 34.000 and halves its share). Dither-1S binds at 8B 32K decode:
    QKᵀ's one-sided strip is 2L of its d = 128 products a word."""
    two = design("dither-2s")
    for sums, eps in (("vsum", 0.0032750), ("vfold", 0.0029227)):
        q = E.design_skip(replace(two, sums=sums), vector="VF3q-16")
        assert (q.ctx, q.phase) == (2048, "prefill") and q.eps_time == pytest.approx(eps, rel=1e-4), sums
    assert E.design_skip(replace(two, sums="vsum")).eps_time == pytest.approx(0.0036468, rel=1e-4)
    for name, eps, ctx, phase in (("dither-2s", 0.0032546, 2048, "prefill"),
                                  ("dither-2s-fill", 0.0016511, 2048, "prefill"),
                                  ("dither-2s-pv", 0.0024783, 2048, "prefill"),
                                  ("dither-c", 0.0034120, 8192, "prefill"),
                                  ("dither-c-peel", 0.0041746, 2048, "prefill"),
                                  ("co-dither-l16", 0.0015656, 2048, "prefill"),
                                  ("dither-1s", 0.1230973, 32768, "decode")):
        q = E.design_skip(design(name))
        assert (q.model, q.ctx, q.phase) == ("llama-3.1-8b", ctx, phase), name
        assert q.eps_time == pytest.approx(eps, rel=1e-4) and q.credit == "floor", name


def test_the_registry_skip_is_eps_budgets_number():
    """One ε_skip a design on each device: its `[skip]` in benchmarks/pouw/machine_model/data/designs/ (sm_120) or
    data/devices/sm_NN/designs/ is design_skip's, to the three figures the registry writes."""
    for dev in devices():
        data = device(dev)
        for name in designs(data):
            d = design(name, data)
            got, want = sum(p.eps for p in d.skip.values()), E.design_skip(d, data=data).eps_time
            assert got == pytest.approx(want, rel=5e-3), (
                f"{dev} {name}: [skip] is {got:.3g}, eps_budget.py's design_skip gives {want:.3g}: write "
                f"[skip.served] eps = {want:#.3g} in {data / 'designs' / name}.toml")


def test_a_skip_at_another_vector_is_eps_budgets_number_there():
    """♭ moves with Π, so a design read on a vector other than its device's current records ε_skip there as
    `[skip.served] at = { V = … }` (`machine-model headline` refuses a reading without it), design_skip's at V."""
    seen = 0
    for dev in devices():
        data = device(dev)
        for name in designs(data):
            d = design(name, data)
            for vec in {v for p in d.skip.values() for v in p.at}:
                got = sum(p.at.get(vec, p.eps) for p in d.skip.values())
                want = E.design_skip(d, data=data, vector=vec).eps_time
                assert got == pytest.approx(want, rel=5e-3), (
                    f"{dev} {name}: [skip] at {vec} is {got:.4g}, design_skip(d, vector={vec!r}) gives {want:.4g}")
                seen += 1
    assert seen >= 9


def test_each_device_prices_its_own_floor_and_the_same_skippable_work():
    """♭ on each device's own Π (`prices`): SPEC's 19.049 and 19.183 on VF3q-16, the LI designs' 5.961 (H200) and 4.035
    (B200), and each variant's own, dither-2s-fill's with its 4.5 filler instructions a word. The peel's tail is Π's
    price of the checked kernels' instructions where that exceeds 1.57 of ♭'s MACs, as on the H200 and B200, whose
    mma.sync m16n8k32 takes 5.98 and 7.85 SMSP-cycles. The checked leaves' vector work is the same cycles on every
    device (a vector instruction is one issue cycle on each Π), so only B_♭ moves it."""
    assert E.prices("dither-c-peel", vector="VF3q-16").mma == pytest.approx(19.049, abs=1e-3)
    assert E.prices("dither-2s", vector="VF3q-16").mma == pytest.approx(19.183, abs=1e-3)
    assert E.prices("dither-2s").vector == "VF3r-16" and E.prices("dither-2s").mma == pytest.approx(17.221, abs=1e-3)
    assert E.prices("dither-2s-fill").mma == pytest.approx(34.000, abs=1e-3)
    sm90, sm100 = device("sm_90"), device("sm_100")
    assert E.prices("dither-2s-li-l32", sm90).mma == pytest.approx(5.961, abs=1e-3)
    p = E.prices("dither-2s-li-l32", sm100)
    assert p.mma == pytest.approx(4.035, abs=1e-3) and p.tail * 128 * 66 == pytest.approx(31.386, abs=1e-3)
    assert p.tail > E.RATES["apv_tail_macs"] * p.mma / 4096
    w0, f0, _ = E.floor_skip("llama-3.1-8b", 8192, design("dither-c"))
    w1, f1, _ = E.floor_skip("llama-3.1-8b", 8192, design("dither-c-li-l32", sm90), data=sm90)
    assert f0 == f1 and w0 / w1 == pytest.approx(17.158 / 5.961, abs=1e-3)


def test_every_family_keeps_a_design_under_one_percent_and_vfold_takes_the_b200_li_2s_under_it():
    """Under D3 every device has a design whose ε_skip is under 1%; Dither-1S is above it everywhere (QKᵀ's one-sided
    strip). The B200's LI Dither-2S was at it with vsum's sums, 1.005% at 8B 2K with its served output free, its sums
    and tail each about half; vfold's take it to 0.839% (0.666% checked), the tail 0.446% and the sums
    0.331% (Round 10)."""
    for dev in devices():
        data = device(dev)
        skips = {n: E.design_skip(design(n, data), data=data).eps_time for n in designs(data)}
        assert min(skips.values()) < 1e-2 and all(v > 0.06 for n, v in skips.items() if n.startswith("dither-1s")), dev
    sm100 = device("sm_100")
    li = design("dither-2s-li-l32", sm100)
    assert li.sums == "vfold" and E.design_skip(li, data=sm100).eps_time == pytest.approx(0.0083948, abs=1e-6)
    assert E.design_skip(replace(li, sums="vsum"), data=sm100).eps_time == pytest.approx(0.010049, abs=1e-5)
    chk = E.floor_eps("llama-3.1-8b", 2048, "prefill", li, checked=True, data=sm100)[0]
    assert chk == pytest.approx(0.0066649, abs=1e-6)


def test_the_b200s_filled_li_design_is_under_one_percent_with_the_operands_free():
    """The free-operand reading (operands free, q* = 1.22, D3's ε_skip): the free-operand cheater skips LI's 8
    tcgen05.ld.x128 a MMA, 8 of ♭'s 1033 issue cycles at every q* (the lane and uniform routes write no word), beside
    its ε_skip of 0.839% with vfold's sums (1.005% with vsum's): core 1.61%. ¾ verified lane instruction a
    word adds 768 to ♭ and to the cheater, diluting both terms over one B_♭, and vfold's sums cost 0.7265 an
    entry against vsum's 35/32: core 0.927% at γ 3.90, 0.952% with its ε_luck (Corollary D3; the filler at
    p = 0). Re-sized with ε_luck and each size's own ε_skip, 0.661 a word is the least that reaches 1.000%
    (γ 3.72), and 0.755 gets to 0.950% (γ 3.91)."""
    sm100 = device("sm_100")
    pi = machine("", sm100)
    li, fill = design("dither-2s-li-l32", sm100), design("dither-2s-li-fill-l32", sm100)
    assert (li.sums, fill.sums, fill.noise, fill.honest) == ("vfold", "vfold", li.noise, li.honest)
    bare = eps(li, pi, 1.22, sm100, operands="free", credit="floor")
    assert bare.machine_eps == pytest.approx(8 / 1033, abs=1e-9) and bare.skip == pytest.approx(0.00839, abs=5e-6)
    assert bare.core() - bare.luck == pytest.approx(0.01613, abs=1e-5) and bare.luck < 1e-5
    e = eps(fill, pi, 1.22, sm100, operands="free", credit="floor")
    assert e.machine_eps == pytest.approx(8 / 1801, abs=1e-9) and e.skip == pytest.approx(0.00483, abs=5e-6)
    assert e.core() - e.luck == pytest.approx(0.00927, abs=1e-5) and e.gamma == pytest.approx(3.900, abs=1e-3)
    assert e.core() == pytest.approx(0.00952, abs=1e-5)
    assert E.prices(fill.name, sm100).mma == pytest.approx(1801 / 256, abs=1e-9)
    m = "llama-3.1-8b"
    _, vfold, _ = E.floor_skip(m, 2048, fill, data=sm100)
    _, vsum, _ = E.floor_skip(m, 2048, replace(fill, sums="vsum"), data=sm100)
    kv_d = E.MODELS[m].kv * E.MODELS[m].d
    assert vsum["P·V"] - vfold["P·V"] == pytest.approx((E.RATES["c_vsum"] - E.RATES["c_vfold"]) * kv_d)


def test_h1_under_the_floor_credit_is_specs_table(monkeypatch):
    """SPEC §2.9's H1 table: Corollary D3's ε_skip, prefill, served free (served
    checked), each piece at its model price over B_♭, on VF3q-16 with co32c's and cohalf's vector work at TS_EPS of ♭'s
    component (SPEC's reading), its sums by vsum as SPEC prices them. Its "line 209 alone" row, vsum at its issue rate
    over honest's measured time, understates it. At the TS model's own gap (TS_CHEAT, the larger) 8B 2K is 0.411%, and
    on VF3r-16 0.457%; vfold's sums, the served V producer, take VF3r-16's to 0.417%."""
    m, v = "llama-3.1-8b", "VF3q-16"
    d = replace(design("dither-c-peel"), sums="vsum")
    for leaf in ("co32c", "cohalf"):
        monkeypatch.setitem(E.TS_CHEAT, leaf, E.prices(d.name, vector=v).mma)
    spec = {("llama-3.1-8b", 2048): (0.405, 0.340), ("llama-3.1-8b", 8192): (0.373, 0.315),
            ("llama-3.1-8b", 32768): (0.292, 0.252), ("llama-3.1-70b", 2048): (0.184, 0.150),
            ("llama-3.1-70b", 8192): (0.180, 0.148), ("llama-3.1-70b", 32768): (0.166, 0.140)}
    for (mm, c), (free, chk) in spec.items():
        f = E.summary(mm, c, "prefill", d, credit="floor", vector=v)
        k = E.summary(mm, c, "prefill", d, checked=True, credit="floor", vector=v)
        assert round(100 * f.eps_time, 3) == free and round(100 * k.eps_time, 3) == chk, (mm, c)
        assert f.gamma == E.summary(mm, c, "prefill", d).gamma, (mm, c)        # honest's latency stays in γ
    rows = E.summary(m, 2048, "prefill", d, credit="floor", vector=v).rows
    split = {r["component"]: round(100 * r["eps_time"], 3) for r in rows if round(100 * r["eps_time"], 3)}
    assert split == {"linears": 0.073, "QKᵀ": 0.005, "P·V": 0.328}      # tail 0.209, sums 0.105, served f 0.013
    mixed = E.summary(m, 2048, "prefill", d, {**E.RATES, "t_vsum": 0.135 * E.RATES["t_vsum"]})
    assert round(100 * mixed.eps_time, 3) == 0.307
    monkeypatch.undo()
    f, r = E.summary(m, 2048, "prefill", d, credit="floor", vector=v), E.summary(m, 2048, "prefill", d, credit="floor")
    assert round(100 * f.eps_time, 3) == 0.411 and round(100 * r.eps_time, 3) == 0.457
    assert round(100 * E.summary(m, 2048, "prefill", design("dither-c-peel"), credit="floor").eps_time, 3) == 0.417
    assert E.summary(m, 2048, "decode", d, credit="floor").eps_time < f.eps_time    # decode's ♭ at its compute alone


def test_the_peel_is_priced_at_bc_kernels_measured_kernel():
    """By honest time (γ's input), the checked kernels' apv_l7qs: the tail at 2.8 mainloop MACs a strip product, the
    sums a one-hot P row frees, and the served epilogue. Checked downstream, the cheater pays the 4th IMMA and the
    epilogue and still skips the sums; at the model's tail with no sums, H1's 0.193% (8B 2K) comes back."""
    d, m, c = design("dither-c-peel"), "llama-3.1-8b", 2048
    free, chk = E.summary(m, c, "prefill", d), E.summary(m, c, "prefill", d, checked=True)
    assert chk.eps_time < free.eps_time and chk.gamma == free.gamma
    assert free.eps_time == pytest.approx(0.0060859, rel=1e-4) and chk.eps_time == pytest.approx(0.0055510, rel=1e-4)
    model = {**E.RATES, "tail_pv": E.RATES["tail"], "t_vsum": 0.0, "t_apv_serve": 0.0}
    assert E.summary(m, c, "prefill", d, model).eps_time == pytest.approx(0.0019336, rel=1e-4)
    s = E.MODELS[m]
    with_sums = next(p for p in E.parts(m, c, "prefill", d.noise) if p.component == "P·V")
    no_sums = next(p for p in E.parts(m, c, "prefill", d.noise, {**E.RATES, "t_vsum": 0.0}) if p.component == "P·V")
    assert with_sums.seconds_skip - no_sums.seconds_skip == pytest.approx(E.RATES["t_vsum"] * s.kv * s.d)
    assert E.summary(m, c, "decode", d).eps_time == E.summary(m, c, "decode", d, {**E.RATES, "t_vsum": 0.0}).eps_time


def test_pv_units_price_as_checked_strip_without_the_scaling_and_the_sink(monkeypatch):
    monkeypatch.setattr(E, "PV_SCALE_OPS", 0.0)
    monkeypatch.setattr(E, "pv_ts", E.pv_ts.__wrapped__)
    for m, c, eps, gamma in (("llama-3.1-8b", 2048, 0.043914, 1.4080), ("llama-3.1-70b", 8192, 0.010158, 1.2985)):
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
    assert "sm_100 on DC-100: ε_skip under the floor credit" in text and "| dither-2s-li-l32 | 32 | 4.035 |" in text
