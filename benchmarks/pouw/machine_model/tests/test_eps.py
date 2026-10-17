"""`machine-model eps`: the ε budget's terms, ε_machine's LP against hand arithmetic on the census, and the data it reads."""
from __future__ import annotations

import json
import math
import tomllib
from dataclasses import replace

import pytest

from verity_machine_model import eps as E
from verity_machine_model.cli import main
from verity_machine_model.model import census, model, vector

PIN = "VF3l-16"                    # the vector these hand-checked numbers were read on, whatever is current
PI = E.machine(PIN)
H = E.honest("dither-duo-l32")
M = E.honest("e-l7qs")
JI = PI.c["MMA"]["JI3e_0"]          # a MMA's SMSP-cycles on the joint issue row that binds honest
UP = PI.c["UNI"]["UP"]              # a uniform instruction's on the uniform pipe
OPERANDS = {"free": 0.0, "floor": 0.2, "honest": 0.375}   # the cheater's issue-cycles a MMA besides the MMA and fold


def provisional(name: str) -> E.Design:
    """A design on the provisional profile (no overhead beyond its operands), whose arithmetic is by hand."""
    return replace(E.design(name), honest=H.name, floor=H.floor)


DUO = provisional("dither-duo")


def test_the_honest_profile():
    """Honest's MMA is counted under its own form's census kind, never the generic MMA."""
    assert H.form == M.form == "IMMA16832" and "MMA" not in H.per_mma
    assert H.base() == {"IMMA16832": 1, "LDS128": 0.25, "ALU": 0.125, "TMA512": 0.125}
    assert sum(H.per_mma.values()) - sum(H.base().values()) == 12   # duo: 12 warp instructions a MMA
    assert H.words == 128 and H.depth == 32 and H.L == 32
    assert H.tensor("free") == {"IMMA16832": 1} and H.tensor("floor") == {"IMMA16832": 1, "LDS128": 0.2}
    assert H.tensor("floor", {"LDSM_X4": 0.1}) == {"IMMA16832": 1, "LDSM_X4": 0.1} and H.tensor("honest") == H.base()
    with pytest.raises(SystemExit, match="operands: one of"):
        H.tensor("none")


def test_the_measured_profile():
    """PROFILE's e_l7qs: the model reproduces its work bound on JI3e_0, and the measured time above it is ε_sched."""
    assert M.status == "measured" and M.measured == 26.1221 and "WAIT" not in M.per_mma
    assert max(PI.load(M.per_mma).items(), key=lambda t: t[1]) == ("JI3e_0", pytest.approx(22.4334, abs=1e-4))
    e = E.eps(E.design("dither-duo"), PI, 8)
    assert e.T_honest_source == "measured" and e.T_honest * 128 == M.measured
    assert e.sched == pytest.approx(1 - 22.4334 / 26.1221, abs=1e-5)
    assert e.clock == pytest.approx(1 - 2092 / 2430, abs=1e-4)   # SPEC's P1: the prover sets its own clocks
    assert e.total() == pytest.approx(1 - (1 - e.skip - e.machine_eps - e.luck) * (1 - e.sched) * (1 - e.clock))
    assert e.floor == pytest.approx((22.4334 - JI - 12) / 22.4334, abs=1e-5)   # honest's overhead, under H's credit
    m = E.eps(E.design("dither-duo"), PI, 8, credit="model")   # PROFILE's B(H)/f_max: ε_sched, ε_clock leave ε for γ
    assert m.sched is None and m.clock == 0 and m.machine_eps == e.machine_eps and m.total() < e.total()
    assert m.credit_ratio == pytest.approx(1 / ((1 - e.sched) * (1 - e.clock)), rel=1e-9)
    xin = E.eps(E.design("dither-xin"), PI, 8)   # another fold: H's measured overhead over its model carries over
    assert xin.sched is None and xin.T_honest_source.startswith("measured +")
    assert xin.T_honest - xin.baseline.T == pytest.approx(e.T_honest - e.baseline.T, rel=1e-9)
    up = E.eps(E.design("dither-duo").with_upfill(), PI, 8)
    assert up.as_built.T == pytest.approx(e.as_built.T, rel=1e-12)   # UPFILL fills UP's slack beside honest's own UNI
    assert up.T_honest_source.startswith("measured +") and up.sched == pytest.approx(e.sched, abs=1e-12)
    fill = E.eps(E.design("dither-two-sided-fill"), PI, 8)
    assert fill.T_honest - fill.baseline.T == pytest.approx(e.T_honest - e.baseline.T, rel=1e-9)


def test_the_noise_sets_the_operand_floor():
    """SPEC §3.4: 1/39 a MMA where B̃ is resident at the zero instance, 0.20 where both operands are fresh."""
    assert E.design("dither-duo").floor == {"LDSM_X4": pytest.approx(1 / 39, abs=1e-6)}
    assert E.design("dither-two-sided").floor == {"LDSM_X4": 0.2} == M.floor
    for n, phi in (("dither-duo", 1 / 39), ("dither-two-sided", 0.2)):
        s = E.setup(E.design(n).with_upfill(), PI, operands="floor", credit="floor")
        assert E.threshold(s) == pytest.approx((JI + phi) / 4, abs=1e-5)


def test_honest_binds_the_joint_issue_row():
    """Honest a MMA: one MMA on JI3e_0 plus 12.375 other warp instructions at one SMSP-cycle each."""
    e = E.eps(DUO, PI, 8)
    assert e.as_built.binding == ("JI3e_0",)
    assert e.as_built.T * 128 == pytest.approx(JI + 12.375, rel=1e-12)
    assert e.baseline.T == pytest.approx(e.as_built.T, rel=1e-9)   # dp4a doesn't pay beside an issue-bound kernel


@pytest.mark.parametrize("operands", OPERANDS)
def test_the_operand_floor(operands):
    """With the scalar routes closed the cheater saves what honest pays on JI3e_0 for operands above its own."""
    s = E.setup(DUO, PI, operands=operands)
    assert s.floor() == pytest.approx((0.375 - OPERANDS[operands]) / (JI + 12.375), abs=1e-12)


@pytest.mark.parametrize("operands", OPERANDS)
def test_the_lane_route_pays_below_the_mma_s_issue_cost(operands):
    """A word on lanes costs q*/32 issue-cycles; on the tensor pipe (JI3e_0 + operands)/128 besides the fold both pay:
    SPEC's Corollary D1, 32α/(βω) = 2.029 with operands free."""
    s = E.setup(DUO, PI, operands=operands)
    t = E.threshold(s, uniform=False)
    assert t == pytest.approx((JI + OPERANDS[operands]) / 4, abs=1e-5)
    assert s.machine_eps(t - 0.01, uniform=False) > s.floor() + 1e-6
    assert s.machine_eps(t + 0.01, uniform=False) == pytest.approx(s.floor(), abs=1e-12)


@pytest.mark.parametrize("q", [4.0, 8.0, 16.0])
def test_the_uniform_route_in_closed_form(q):
    """Words written and folded on the uniform datapath at q* + 3 uniform instructions: the cheater balances honest's
    JI3e_0 time on the rest against UP, so ε_machine = T_H/(T_H + (q* + 3)·UP)."""
    e = E.eps(DUO, PI, q, operands="honest", lanes=False)
    th = e.as_built.T
    assert e.machine_eps == pytest.approx(th / (th + (q + 3) * UP), rel=1e-9)
    assert set(e.cheat.binding) == {"JI3e_0", "UP"}


def test_upfill_closes_the_uniform_route():
    d = DUO.with_upfill()
    e = E.eps(d, PI, 8, operands="honest")
    assert e.fillers[-1][1] == pytest.approx(e.as_built.T / UP)
    assert e.as_built.T == pytest.approx(E.eps(DUO, PI, 8).as_built.T)   # it costs honest nothing
    assert e.machine_eps == 0 and E.eps(DUO, PI, 8, operands="honest").machine_eps > 0.009
    s = E.setup(d, PI, operands="honest")
    assert E.threshold(s) == pytest.approx(E.threshold(E.setup(DUO, PI, operands="honest"), uniform=False), abs=1e-5)
    free = E.setup(d, PI)
    assert free.machine_eps(8) == pytest.approx(free.floor(), abs=1e-12) and E.threshold(free) == pytest.approx(
        JI / 4, abs=1e-5)
    with pytest.raises(SystemExit, match="has UPFILL already"):
        d.with_upfill()


def test_a_uniform_instruction_on_the_issue_rows():
    """Charged an issue slot, the uniform route costs the cheater at least the lane route at q* ≥ the threshold without
    UPFILL. UPFILL then costs honest issue time, and the cheater does its checked work on lanes, 32 values a warp
    instruction, so it opens more than it closes."""
    pi = E.uniform_on_issue(PI, 1.0)
    assert pi.c["UNI"]["issue"] == pi.c["UNI"]["JI3e_0"] == 1.0 and pi.c["UNI"]["UP"] == UP
    assert "issue" not in PI.c["UNI"] and "UNI on issue at 1" in pi.vector
    refit = E.Machine(vector="X", rows=("issue", "JI3n_0", "JI3n_1", "A", "UP"), cuda=(),
                      c={"ALU": {"issue": 1.0, "JI3n_0": 0.9346, "JI3n_1": 0.2963, "A": 2.0}, "UNI": {"UP": UP}})
    assert E.uniform_on_issue(refit, 0.92).c["UNI"] == pytest.approx({"UP": UP, "issue": 0.92, "JI3n_0": 0.92 * 0.9346})
    old, new = E.setup(DUO, PI, credit="floor"), E.setup(DUO, pi, credit="floor")
    assert new.baseline.T == pytest.approx(old.baseline.T, rel=1e-9)
    assert E.threshold(old) is None and E.threshold(new) == pytest.approx(E.threshold(old, uniform=False), abs=1e-5)
    assert new.machine_eps(8) == pytest.approx(new.floor(), abs=1e-9)
    assert new.machine_eps(8) < old.machine_eps(8) - 0.005
    up = E.setup(DUO.with_upfill(), pi, credit="floor")
    assert up.baseline.T > 1.5 * new.baseline.T and up.machine_eps(8) > 0.3


def test_lane_filler_dilutes_the_operand_floor():
    """x lane instructions a word of verified filler add 4x issue-cycles a MMA to honest, and the cheater must run them
    too, so what it saves on operands is a smaller share. The design file is DUO's two-sided twin with 4.5 of them."""
    two = provisional("dither-two-sided").with_upfill()
    fill = provisional("dither-two-sided-fill").with_upfill()
    f = fill.filler[0]
    assert (f.per_word, f.mix, f.route) == (4.5, E.lane_fill(4.5).mix, "lanes")
    assert two.with_fill(4.5).filler == (E.UPFILL, E.lane_fill(4.5))
    e0, e = E.eps(two, PI, 8), E.eps(fill, PI, 8)
    assert e.as_built.T * 128 == pytest.approx(JI + 12.375 + 4 * 4.5, rel=1e-12)
    assert e.machine_eps == pytest.approx(e.floor, abs=1e-12) and e.machine_eps < e0.machine_eps
    assert e.total() < e0.total() and e.gamma < 3
    assert E.threshold(E.setup(fill, PI)) == pytest.approx(JI / 4, abs=1e-5)


@pytest.mark.parametrize("n, phi", [("dither-duo", 1 / 39), ("dither-two-sided", 0.2)])
def test_the_floor_credit(n, phi):
    """SPEC §0.7a: ♭ is the tensor route at φ_del with the fold, B_♭ = JI3e_0's MMA + 12 + φ_del a MMA. Under it ε_sched,
    ε_clock and honest's overhead leave ε for γ, the cheater paying the floor saves nothing once UPFILL closes the
    uniform route and q* is past the threshold. Paying nothing, it saves φ_del on JI3e_0 but UPFILL holds UP for B_♭,
    so it moves filler to lanes (1/32 an issue-cycle against UP's cost) until the two rows balance."""
    d = E.design(n).with_upfill()
    e = E.eps(d, PI, 8, operands="floor", credit="floor")
    assert e.baseline.T * 128 == pytest.approx(JI + 12 + phi, abs=1e-6) and "JI3e_0" in e.baseline.binding
    assert e.sched is None and e.clock == 0 and e.machine_eps == 0
    assert e.credit_ratio == pytest.approx(M.measured / (JI + 12 + phi) / (1 - d.clock.eps), abs=1e-6)
    assert e.total() == pytest.approx(e.skip + e.luck, abs=1e-12)
    free = E.eps(d, PI, 8, credit="floor")
    assert free.machine_eps == pytest.approx(phi * (1 - 1 / (32 * UP + 1)) / (JI + 12 + phi), abs=1e-8)
    assert free.gamma == e.gamma == E.eps(d, PI, 8).gamma   # the credit doesn't change what honest runs
    with pytest.raises(SystemExit, match="the floor program delivers operands"):
        E.setup(d, PI, operands="honest", credit="floor")
    with pytest.raises(SystemExit, match="credit: one of"):
        E.setup(d, PI, credit="none")


def test_plain_work_counts_in_gamma_only():
    """A design that runs plain (unfolded) MMAs beside its work, as CO-Dither at r = k/2 runs a k/2 strip on one limb,
    pays them in γ and nowhere else: γ + 0.5."""
    duo = E.design("dither-duo")
    co = replace(duo, plain=0.5)
    assert co.work == duo.work == 1
    a, b = E.eps(co, PI, 8), E.eps(duo, PI, 8)
    assert a.gamma == pytest.approx(b.gamma + 0.5, abs=1e-12) and a.machine_eps == b.machine_eps


def test_the_curve_falls_with_q():
    s = E.setup(DUO, PI)
    xs = [s.machine_eps(q) for q in E.QS]
    assert E.QS[0] == 0.5 and E.QS[-1] == 16 and len(E.QS) == 32
    assert all(a >= b - 1e-12 for a, b in zip(xs, xs[1:])) and xs[0] > 0.2 and s.floor() < xs[-1] < 0.025


def test_fold_cost():
    assert DUO.with_fold_cost(3).fold == DUO.fold
    d = DUO.with_fold_cost(1.5)
    assert d.fold == {"ALU": 1.0, "IMAD": 0.5} and d.fold_cost() == 1.5
    xin = provisional("dither-xin")
    assert E.eps(d, PI, 4).machine_eps == pytest.approx(E.eps(xin, PI, 4).machine_eps, abs=1e-12)
    assert E.eps(d, PI, 4).gamma < E.eps(DUO, PI, 4).gamma


def test_the_certificate_and_the_terms_add_up():
    """π* is the LP's dual: a convex combination of rows pricing the cheater's mix at its time. At π*, ε_machine splits
    into ε_cert and ε_route, and the lucky cheater's time is 1 − ε_machine − ε_luck of honest's."""
    cases = [(d, o, c) for d in (DUO, DUO.with_upfill(), E.design("dither-duo").with_upfill()) for o in OPERANDS
             for c in E.CREDITS if not (c == "floor" and o == "honest")]
    for d, operands, credit in cases:
        e = E.eps(d, PI, 8, operands=operands, credit=credit)
        assert sum(e.cheat.pi.values()) == pytest.approx(1, abs=1e-9)
        assert e.cheat.price(PI, e.cheat.n) == pytest.approx(e.cheat.T, rel=1e-9)
        assert e.cert + e.route == pytest.approx(e.machine_eps, abs=1e-12) and e.cert >= 0 and e.route >= -1e-12
        assert 1 - e.lucky.T / e.baseline.T == pytest.approx(e.machine_eps + e.luck, abs=1e-12)


def test_luck_and_the_total():
    """Theorem S inside the LP: the cheater skips d̄ of the words, saving their MMA and operands and not their fold."""
    assert E.deficit(DUO) == pytest.approx(41 * math.log(2) / (0.25 * 48000))
    e = E.eps(DUO, PI, 8, operands="honest", lanes=False, uniform=False)
    assert e.cheat.binding == ("JI3e_0",)
    assert e.luck == pytest.approx(E.deficit(DUO) * (JI + 0.375) / (JI + 12.375), rel=1e-6)
    e = E.eps(DUO, PI, 8)
    assert e.sched is None and e.mand == 0 and e.skip == 0.0063
    assert e.total() == pytest.approx(1 - (1 - e.skip - e.machine_eps - e.luck) * (1 - e.clock))


def test_adapt_is_a_proof_units_term():
    """ε_adapt (SPEC's Theorem D) sits in the first factor beside ε_skip, ε_machine and ε_luck, and is the proof unit's."""
    assert set(E.adapts()) == {"segment", "whole", "whole-recompute", "per-stage"}
    assert E.adapt().name == "segment" and E.adapt().eps == 0.21 and E.adapt("whole").eps == 0.0035
    e0 = E.eps(DUO, PI, 8)
    e = E.eps(DUO, PI, 8, adapt=E.adapt("whole"))
    assert e0.terms()["adapt"] is None and e.terms()["adapt"] == 0.0035
    assert e.total() == pytest.approx(1 - (1 - e.skip - e.machine_eps - e.luck - 0.0035) * (1 - e.clock))
    assert E.eps(DUO, PI, 8, adapt=E.adapt("per-stage")).total() == pytest.approx(e0.total())
    with pytest.raises(SystemExit, match="no proof unit 'none'"):
        E.adapt("none")
    assert all(E.adapt(n).selection for n in E.adapts())


def test_core_and_sample_are_two_layers():
    """Daniel (06:29Z): the core ε is the protocol with 100% checking, ε_luck = ε_adapt = 0; ε_sample = ε_luck + ε_adapt
    is the sampling compiler's and never enters it."""
    e = E.eps(DUO, PI, 8, adapt=E.adapt("whole"))
    assert e.core() == pytest.approx(1 - (1 - e.skip - e.mand - e.machine_eps) * (1 - (e.sched or 0)) * (1 - e.clock))
    assert e.sample() == pytest.approx(e.luck + 0.0035) and e.luck > 0
    assert e.core() == pytest.approx(E.eps(DUO, PI, 8).core()) and e.core() < e.total()
    assert e.core(sched=False) <= e.core()


def test_luck_spends_one_budget_on_words_and_filler():
    """Theorem S's budget ln(1/δ)/S weights each deficient block by 1 − p (¾ a word, 0.854 a filler block): a lane
    filler is cheaper to skip than a word, so the lucky cheater leaves filler undone and saves more than d̄ of words."""
    d = E.design("dither-two-sided-fill").with_upfill()
    assert all(f.p == 0.854 for f in d.filler)
    e = E.eps(d, PI, 8, operands="floor", credit="floor")
    s = e.lucky.short
    assert s["filler 1"] > 0 and (1 - d.p) * s["words"] + sum(
        (1 - f.p) * s[f"filler {j + 1}"] for j, f in enumerate(d.filler)) == pytest.approx(E.budget(d), rel=1e-6)
    words_only = E.eps(replace(d, filler=tuple(replace(f, p=0.0) for f in d.filler)), PI, 8, operands="floor",
                       credit="floor")
    assert e.luck > words_only.luck and words_only.lucky.short["filler 1"] == 0


def test_the_table_route_is_bound_by_L2():
    d = E.design("dither-duo").with_upfill()
    base = E.setup(d, PI, credit="floor")
    wide = E.Table(q=1.25, bytes=8.0, l2=1e18, sms=188, mhz=2092, source="")    # L2 never binds
    unbound = E.setup(d, PI, credit="floor", table=wide)
    assert unbound.cheat(2.16, lanes=False).T == pytest.approx(base.cheat(1.25).T, rel=1e-9)
    shut = E.setup(d, PI, credit="floor", table=replace(wide, l2=1e9))         # 1 GB/s: next to no table words
    assert shut.machine_eps(2.16) == pytest.approx(base.machine_eps(2.16), abs=1e-5)
    t = E.table()
    assert t.per_cycle() == pytest.approx(7.244e12 / (188 * 4 * 2092e6))
    s = E.setup(d, PI, credit="floor", table=t)
    m = s.cheat(2.16)
    assert "L2" in m.binding and m.words["table"] * t.bytes / t.per_cycle() == pytest.approx(m.T, rel=1e-9)
    assert 0 < m.words["table"] < 0.25 and s.machine_eps(2.16) > base.machine_eps(2.16)
    assert s.floor() == pytest.approx(s.machine_eps(8.0), abs=1e-9)        # the floor keeps the table route
    measured = E.setup(d, PI, credit="floor", table=E.table(q=1.906))
    assert base.machine_eps(2.16) < measured.machine_eps(2.16) < s.machine_eps(2.16)


def test_designs_read_what_exists():
    c = census(model(), vector(PIN).epoch)
    for n in E.designs():
        d = E.design(n)
        assert d.name == n and d.honest in E.profiles() and d.floor and "SPEC" in d.floor_source
        assert set(d.fold) | set(d.baseline) | set(d.floor) | {k for f in d.filler for k in f.mix} <= set(c), n
        assert all("note:" in p.source or "CALC" in p.source or "assumption" in p.source
                   for p in (*d.skip.values(), *d.mand.values())), n
        assert 0 <= E.eps(d, PI, 8).total() < 1
    for n in E.profiles():
        h = E.honest(n)
        assert set(h.per_mma) - {h.form} | set(h.floor) <= set(c) and h.form in PI.c, n
    with pytest.raises(SystemExit, match="no design 'none'"):
        E.design("none")


def test_cli(capsys):
    assert main(["eps", "--vector", PIN]) == 0
    out = capsys.readouterr().out
    assert out.startswith("dither-duo: ") and "binding JI3e_0" in out and "ε_machine  11.171%" in out
    assert "q* threshold (ε_machine = its floor): none ≤ 64; lanes alone 2.029" in out and "ε_sched    14.121%" in out
    assert "ε core     " in out and "ε_sample" not in out and "ε_luck" not in out
    assert main(["eps", "--vector", PIN, "--checking", "sampled"]) == 0
    out = capsys.readouterr().out
    assert "ε_adapt 21.000%: proof unit segment" in out and "ε_sample   21.153%" in out
    with pytest.raises(SystemExit, match="--checking sampled"):
        main(["eps", "--adapt", "whole"])
    assert main(["eps", "--vector", PIN, "dither-two-sided", "--upfill", "--operands", "floor", "--credit", "floor",
                 "--checking", "sampled", "--adapt", "per-stage", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["credit"] == "floor" and d["terms"]["machine"] == 0 and d["terms"]["sched"] is None
    assert d["threshold"] == pytest.approx((JI + 0.2) / 4, abs=1e-5) and d["core"] < 0.002
    assert d["adapt"]["unit"] == "per-stage" and d["terms"]["adapt"] == 0
    assert d["sample"]["eps"] == pytest.approx(d["terms"]["luck"]) and d["sample"]["budget"]["checks"] == 48000
    assert main(["eps", "--vector", PIN, "dither-two-sided", "--upfill", "--q", "4", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-two-sided+up" and d["operands"] == "free"
    assert d["terms"]["machine"] == pytest.approx(d["floor"]) and d["threshold"] == pytest.approx(2.029, abs=1e-3)
    assert len(d["curve"]) == 32 and d["curve"][0][0] == 0.5 and d["curve"][-1][0] == 16
    assert d["curve_uniform"][7] == pytest.approx([4, d["terms"]["machine"]])   # q*_U = q* = 4
    assert main(["eps", "--vector", PIN, "--q-uniform", "32", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-duo@qu32"
    assert d["floor"] < d["terms"]["machine"] < E.eps(E.design("dither-duo"), PI, 8).machine_eps
    assert main(["eps", "--vector", PIN, "dither-two-sided", "--upfill", "--operands", "honest", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    base = PI.load(M.base())["JI3e_0"]   # honest's MMA and its other issue a MMA, the fold aside
    assert d["terms"]["machine"] == 0 and d["threshold_lanes"] == pytest.approx(base / 4, abs=1e-5)
    assert d["threshold"] >= d["threshold_lanes"] - 1e-6
    assert main(["eps", "--vector", PIN, "--upfill", "--credit", "floor", "--q", "2.16", "--lookups", "--lookup-q",
                 "1.906", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["lookups"]["q"] == 1.906 and d["cheat"]["words"]["table"] > 0 and "L2" in d["cheat"]["binding"]
    with pytest.raises(SystemExit):
        main(["eps", "--vector", PIN, "--l2", "1e12"])
    assert main(["eps", "--vector", PIN, "--credit", "floor", "--uni-issue", "0.92", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["vector"] == "VF3l-16 (UNI on issue at 0.92)" and d["threshold"] == pytest.approx(2.029, abs=1e-3)
    assert main(["eps", "--vector", PIN, "--table", "dither-duo", "--fold-sweep", "1.5,3", "--operands", "floor", "--q",
                 "2.16"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Π VF3l-16, operands floor")
    assert lines[1].split()[-6:-3] == ["core@2", "core@2.16", "core@8"]
    assert [x.split()[0] for x in lines[2:]] == ["dither-duo@r1.5", "dither-duo@r3"]
    assert main(["eps", "--vector", PIN, "--table", "co-dither-pv", "--operands", "floor", "--credit", "floor",
                 "--checking", "sampled", "--adapt", "whole"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "checking sampled" in lines[0] and "proof unit whole" in lines[0]
    assert lines[1].split()[-4:] == ["|", "luck", "adapt", "sample"] and lines[2].split()[-4] == "|"
    luck, adapt, sample = (float(x.rstrip("%")) for x in lines[2].split()[-3:])
    assert adapt == 0.35 and sample == pytest.approx(luck + adapt, abs=1e-3)


QPIN = "VF3q-16"                   # the per-form numbers below were read on it (FACET-BREAK's JI3q rows)
QPI = E.machine(QPIN)
FORMS = E.forms()
I = next(f for f in FORMS if f.kind == "IMMA16832")
DESIGNS = ("dither-duo", "dither-two-sided", "dither-two-sided-pv", "co-dither", "co-dither-pv")


def test_the_form_table():
    """data/forms.toml: tensor_kinds' 19 native forms as the census's 14 per-form kinds, each with its word yield."""
    t = tomllib.loads((E.DATA / "forms.toml").read_text())
    assert sum(len(v["ptx"]) for v in t.values() if isinstance(v, dict)) == 19
    c = census(model(), vector(QPIN).epoch)
    assert {f.kind for f in FORMS} == {k for k in c if k != "MMA" and "MMA" in k}
    assert all(set(c[f.kind]) >= {r for r in c["MMA"] if r.startswith("JI3q")} for f in FORMS)
    by = {f.kind: f for f in FORMS}
    assert (I.words, I.depth, I.split, I.per_word(32)) == (128, 32, 1, 1 / 128)
    assert by["IMMA8816"].words == 64 and by["IMMASP16864"].depth == 32 and by["IMMASP16832"].depth == 16
    assert by["HMMA16816F16"].depth == 4 and by["QMMA16832"].depth == pytest.approx(32 / 3)
    assert by["OMMASF16864"].depth == pytest.approx(64 / 3) and all(f.split >= 1 and f.exact for f in FORMS)


def test_a_form_reads_the_generic_line_where_it_has_none():
    """VF3q-16's EFFMA rows were fitted for the generic MMA alone, and before VF3n-16 no form has a line."""
    effma = [r for r in QPI.rows if r.startswith("EFFMA")]
    assert len(effma) == 7 and all(QPI.c[f.kind][r] == QPI.c["MMA"][r] for f in FORMS for r in effma)
    assert QPI.c["IMMA16832"]["JI3q_10"] > QPI.c["MMA"]["JI3q_10"]
    assert all(PI.c[f.kind] == PI.c["MMA"] for f in FORMS)


def test_honest_is_credited_at_its_own_form():
    """The floor program ♭ on VF3q-16 pays IMMA16832's own coefficients: JI3q_10 binds, not the generic MMA's JI3q_11."""
    s = E.setup(E.design("dither-two-sided"), QPI, operands="floor", credit="floor")
    assert s.baseline.T * 128 == pytest.approx(19.183, abs=1e-3) and s.baseline.binding == ("JI3q_10",)
    assert s.baseline.forms == {"IMMA16832": 1.0} and "MMA" not in s.baseline.n
    assert s.forms == (I,)   # every other form `undominated` drops


def test_t1_every_form_is_dominated():
    """T1 on VF3q-16 with a class per form: on every row each form's instructions a word cost at least IMMA16832's, so
    the tensor route alone (the lane, uniform and table routes closed) saves nothing, whatever the operands cost. The
    ties: IMMASP16864 everywhere it has the same line; QMMA16832 would undercut IMMA16832 below split 1.0083 (JI3o_1),
    OMMASF16864 below split 2."""
    tie = {}
    for f in FORMS:
        ratio = {r: QPI.c[f.kind][r] * f.per_word(32) / (QPI.c[I.kind][r] * I.per_word(32))
                 for r, v in QPI.c[I.kind].items() if v}
        assert min(ratio.values()) >= 1 - 1e-12, f.kind
        tie[f.kind] = f.split / min(ratio.values())
    assert tie["QMMA16832"] == pytest.approx(1.0083, abs=1e-4) and tie["OMMASF16864"] == pytest.approx(2)
    assert max(v for k, v in tie.items() if k not in ("IMMA16832", "IMMASP16864", "QMMA16832", "OMMASF16864")) <= 1 + 1e-9
    assert E.undominated(list(FORMS), QPI) == (I,)
    for n in DESIGNS:
        for operands, credit in (("floor", "floor"), ("honest", "H")):
            s = E.setup(E.design(n), QPI, operands=operands, credit=credit)
            every = replace(s, forms=FORMS)   # the LP over all 14 forms: what `undominated` drops never helps
            assert s.floor() == 0 and every.floor() == pytest.approx(0, abs=1e-12), n
            assert every.cheat(0.0, lanes=False, uniform=False).forms == {"IMMA16832": 1.0}, n
            assert every.machine_eps(2) == pytest.approx(s.machine_eps(2), abs=1e-12), n


def test_the_cheater_mixes_forms():
    """The LP takes a form cheaper than honest's when Π has one: IMMA16816 at a quarter of its own line writes a word
    for at most half of IMMA16832's on every row, so the tensor route alone saves about half the MMA's share."""
    c = dict(QPI.c, IMMA16816={r: v / 4 for r, v in QPI.c["IMMA16816"].items()})
    pi = replace(QPI, c=c)
    s = E.setup(E.design("dither-two-sided"), pi, operands="floor", credit="floor")
    m = s.cheat(0.0, lanes=False, uniform=False)
    assert m.forms == {"IMMA16816": 1.0} and s.floor() > 0.15
    assert E.setup(E.design("dither-two-sided"), QPI, operands="floor", credit="floor").cheat(2.0).forms == {
        "IMMA16832": pytest.approx(0.7194, abs=1e-4)}


@pytest.mark.parametrize("n, eps2, eps3, q0, lookups", [
    ("dither-two-sided", 0.03502, 0.00532, 3.2428, 0.05983),
    ("dither-duo", 0.03447, 0.00457, 3.1993, 0.05883),
])
def test_eps_with_a_class_per_form(n, eps2, eps3, q0, lookups):
    """Core ε's machine term on VF3q-16, credit and operands at the floor: FACET-BREAK's IMMA16832-own counterfactual
    (#2244: 3.50%/3.45% at q* = 2, 0 at q* = 4, the floor at q* = 3.2), now with the cheater free to mix forms. With
    the table route the floor is 5.98%/5.88% (4.81%/5.11% under the generic MMA at the minimum over forms)."""
    s = E.setup(E.design(n), QPI, operands="floor", credit="floor")
    assert s.machine_eps(2) == pytest.approx(eps2, abs=1e-5) and s.machine_eps(3) == pytest.approx(eps3, abs=1e-5)
    assert s.machine_eps(4) == 0 and E.threshold(s) == pytest.approx(q0, abs=1e-4)
    t = E.setup(E.design(n), QPI, operands="floor", credit="floor", table=E.table())
    assert t.floor() == pytest.approx(lookups, abs=1e-5) and t.machine_eps(4) == pytest.approx(lookups, abs=1e-5)
    assert all(set(t.cheat(q).forms) == {"IMMA16832"} and "MMA" not in t.cheat(q).n for q in (2, 3, 4))


def test_cli_names_the_forms(capsys):
    assert main(["eps", "--vector", QPIN, "dither-two-sided", "--credit", "floor", "--operands", "floor", "--q", "2"]) == 0
    out = capsys.readouterr().out
    assert "IMMA.16832.S32.S8.S8 (IMMA16832)" in out and "on any of 14 MMA forms" in out
    assert "tensor on IMMA16832 71.9%" in out
    assert main(["eps", "--vector", QPIN, "dither-two-sided", "--credit", "floor", "--operands", "floor", "--q", "2",
                 "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["honest_form"] == "IMMA16832" and d["cheat"]["forms"] == {"IMMA16832": pytest.approx(0.7194, abs=1e-4)}
