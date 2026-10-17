"""`machine-model eps`: the ε budget's terms, ε_machine's LP against hand arithmetic on the census, and the data it reads."""
from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from verity_machine_model import eps as E
from verity_machine_model.cli import main
from verity_machine_model.model import census, model, vector

PI = E.machine()
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
    assert H.base() == {"MMA": 1, "LDS128": 0.25, "ALU": 0.125, "TMA512": 0.125}
    assert sum(H.per_mma.values()) - sum(H.base().values()) == 12   # duo: 12 warp instructions a MMA
    assert H.words == 128 and H.depth == 32 and H.L == 32
    assert H.tensor("free") == {"MMA": 1} and H.tensor("floor") == {"MMA": 1, "LDS128": 0.2}
    assert H.tensor("floor", {"LDSM_X4": 0.1}) == {"MMA": 1, "LDSM_X4": 0.1} and H.tensor("honest") == H.base()
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
    xin = E.eps(E.design("dither-xin"), PI, 8)
    assert xin.sched is None and xin.T_honest_source == "model"   # another fold: unmeasured
    up = E.eps(E.design("dither-duo").with_upfill(), PI, 8)
    assert up.as_built.T == pytest.approx(e.as_built.T, rel=1e-12)   # UPFILL fills UP's slack beside honest's own UNI
    assert up.T_honest_source.startswith("measured ×") and up.sched == pytest.approx(e.sched, abs=1e-12)


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
    assert t.per_cycle() == pytest.approx(6e12 / (188 * 4 * 2092e6))
    s = E.setup(d, PI, credit="floor", table=t)
    m = s.cheat(2.16)
    assert "L2" in m.binding and m.words["table"] * t.bytes / t.per_cycle() == pytest.approx(m.T, rel=1e-9)
    assert 0 < m.words["table"] < 0.1 and s.machine_eps(2.16) > base.machine_eps(2.16)
    assert s.floor() == pytest.approx(s.machine_eps(8.0), abs=1e-9)        # the floor keeps the table route
    measured = E.setup(d, PI, credit="floor", table=E.table(q=1.906))
    assert base.machine_eps(2.16) < measured.machine_eps(2.16) < s.machine_eps(2.16)


def test_designs_read_what_exists():
    c = census(model(), vector(model().current).epoch)
    for n in E.designs():
        d = E.design(n)
        assert d.name == n and d.honest in E.profiles() and d.floor and "SPEC" in d.floor_source
        assert set(d.fold) | set(d.baseline) | set(d.floor) | {k for f in d.filler for k in f.mix} <= set(c), n
        assert all("note:" in p.source or "CALC" in p.source or "assumption" in p.source
                   for p in (*d.skip.values(), *d.mand.values())), n
        assert 0 <= E.eps(d, PI, 8).total() < 1
    for n in E.profiles():
        assert set(E.honest(n).per_mma) | set(E.honest(n).floor) <= set(c), n
    with pytest.raises(SystemExit, match="no design 'none'"):
        E.design("none")


def test_cli(capsys):
    assert main(["eps"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("dither-duo: ") and "binding JI3e_0" in out and "ε_machine  11.171%" in out
    assert "q* threshold (ε_machine = its floor): none ≤ 64; lanes alone 2.029" in out and "ε_sched    14.121%" in out
    assert main(["eps", "dither-two-sided", "--upfill", "--operands", "floor", "--credit", "floor", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["credit"] == "floor" and d["terms"]["machine"] == 0 and d["terms"]["sched"] is None
    assert d["threshold"] == pytest.approx((JI + 0.2) / 4, abs=1e-5) and d["total"] < 0.002
    assert main(["eps", "dither-two-sided", "--upfill", "--q", "4", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-two-sided+up" and d["operands"] == "free"
    assert d["terms"]["machine"] == pytest.approx(d["floor"]) and d["threshold"] == pytest.approx(2.029, abs=1e-3)
    assert len(d["curve"]) == 32 and d["curve"][0][0] == 0.5 and d["curve"][-1][0] == 16
    assert d["curve_uniform"][7] == pytest.approx([4, d["terms"]["machine"]])   # q*_U = q* = 4
    assert main(["eps", "--q-uniform", "32", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-duo@qu32"
    assert d["floor"] < d["terms"]["machine"] < E.eps(E.design("dither-duo"), PI, 8).machine_eps
    assert main(["eps", "dither-two-sided", "--upfill", "--operands", "honest", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    base = PI.load(M.base())["JI3e_0"]   # honest's MMA and its other issue a MMA, the fold aside
    assert d["terms"]["machine"] == 0 and d["threshold_lanes"] == pytest.approx(base / 4, abs=1e-5)
    assert d["threshold"] >= d["threshold_lanes"] - 1e-6
    assert main(["eps", "--upfill", "--credit", "floor", "--q", "2.16", "--lookups", "--lookup-q", "1.906",
                 "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["lookups"]["q"] == 1.906 and d["cheat"]["words"]["table"] > 0 and "L2" in d["cheat"]["binding"]
    with pytest.raises(SystemExit):
        main(["eps", "--l2", "1e12"])
    assert main(["eps", "--table", "dither-duo", "--fold-sweep", "1.5,3", "--operands", "floor", "--q", "2.16"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Π VF3l-16, operands floor")
    assert lines[1].split()[-6:-3] == ["ε@2", "ε@2.16", "ε@8"]
    assert [x.split()[0] for x in lines[2:]] == ["dither-duo@r1.5", "dither-duo@r3"]
