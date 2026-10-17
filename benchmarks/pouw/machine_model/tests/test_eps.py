"""`machine-model eps`: the ε budget's terms, ε_machine's LP against hand arithmetic on the census, and the data it reads."""
from __future__ import annotations

import json
import math
import shutil

import pytest

from verity_machine_model import eps as E
from verity_machine_model.cli import main
from verity_machine_model.model import DATA, census, model, vector

PI = E.machine()
DUO = E.design("dither-duo")
H = E.honest("dither-duo-l32")
JI = PI.c["MMA"]["JI3e_0"]          # a MMA's SMSP-cycles on the joint issue row that binds honest
UP = PI.c["UNI"]["UP"]              # a uniform instruction's on the uniform pipe
OPERANDS = {"free": 0.0, "floor": 0.2, "honest": 0.375}   # the cheater's issue-cycles a MMA besides the MMA and fold


def test_the_honest_profile():
    assert H.base() == {"MMA": 1, "LDS128": 0.25, "ALU": 0.125, "TMA512": 0.125}
    assert sum(H.per_mma.values()) - sum(H.base().values()) == 12   # duo: 12 warp instructions a MMA
    assert H.words == 128 and H.depth == 32 and H.L == 32
    assert H.tensor("free") == {"MMA": 1} and H.tensor("floor") == {"MMA": 1, "LDS128": 0.2}
    assert H.tensor("honest") == H.base()
    with pytest.raises(SystemExit, match="operands: one of"):
        H.tensor("none")


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
    two = E.design("dither-two-sided").with_upfill()
    fill = E.design("dither-two-sided-fill").with_upfill()
    f = fill.filler[0]
    assert (f.per_word, f.mix, f.route) == (4.5, E.lane_fill(4.5).mix, "lanes")
    assert two.with_fill(4.5).filler == (E.UPFILL, E.lane_fill(4.5))
    e0, e = E.eps(two, PI, 8), E.eps(fill, PI, 8)
    assert e.as_built.T * 128 == pytest.approx(JI + 12.375 + 4 * 4.5, rel=1e-12)
    assert e.machine_eps == pytest.approx(e.floor, abs=1e-12) and e.machine_eps < e0.machine_eps
    assert e.total() < 0.01 < e0.total() and e.gamma < 3
    assert E.threshold(E.setup(fill, PI)) == pytest.approx(JI / 4, abs=1e-5)


def test_the_curve_falls_with_q():
    s = E.setup(DUO, PI)
    xs = [s.machine_eps(q) for q in E.QS]
    assert E.QS[0] == 0.5 and E.QS[-1] == 16 and len(E.QS) == 32
    assert all(a >= b - 1e-12 for a, b in zip(xs, xs[1:])) and xs[0] > 0.2 and s.floor() < xs[-1] < 0.025


def test_fold_cost():
    assert DUO.with_fold_cost(3).fold == DUO.fold
    d = DUO.with_fold_cost(1.5)
    assert d.fold == {"ALU": 1.0, "IMAD": 0.5} and d.fold_cost() == 1.5
    xin = E.design("dither-xin")
    assert E.eps(d, PI, 4).machine_eps == pytest.approx(E.eps(xin, PI, 4).machine_eps, abs=1e-12)
    assert E.eps(d, PI, 4).gamma < E.eps(DUO, PI, 4).gamma


def test_the_certificate_and_the_terms_add_up():
    """π* is the LP's dual: a convex combination of rows pricing the cheater's mix at its time. At π*, ε_machine splits
    into ε_cert and ε_route, and the lucky cheater's time is 1 − ε_machine − ε_luck of honest's."""
    for d in (DUO, DUO.with_upfill()):
        for operands in OPERANDS:
            e = E.eps(d, PI, 8, operands=operands)
            assert sum(e.cheat.pi.values()) == pytest.approx(1, abs=1e-9)
            assert e.cheat.price(PI, e.cheat.n) == pytest.approx(e.cheat.T, rel=1e-9)
            assert e.cert + e.route == pytest.approx(e.machine_eps, abs=1e-12) and e.cert >= 0 and e.route >= -1e-12
            assert 1 - e.lucky.T / e.baseline.T == pytest.approx(e.machine_eps + e.luck, abs=1e-12)


def test_luck_and_the_total():
    """Theorem S inside the LP: the cheater skips d̄ of the words, saving their MMA and operands and not their fold."""
    assert E.deficit(DUO) == pytest.approx(40 * math.log(2) / (0.25 * 121180))
    e = E.eps(DUO, PI, 8, operands="honest", lanes=False, uniform=False)
    assert e.cheat.binding == ("JI3e_0",)
    assert e.luck == pytest.approx(E.deficit(DUO) * (JI + 0.375) / (JI + 12.375), rel=1e-6)
    e = E.eps(DUO, PI, 8)
    assert e.sched is None and e.mand == 0 and e.clock == 0 and e.skip == 0.0063
    assert e.total() == pytest.approx(e.skip + e.machine_eps + e.luck)


def test_a_measured_profile_gives_eps_sched(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    p = data / "honest" / "dither-duo-l32.toml"
    p.write_text(p.read_text() + '\n[measured]\ncycles_per_mma = 25.0\nsource = "a test"\n')
    e = E.eps(E.design("dither-duo", data), PI, 8, data)
    assert e.sched == pytest.approx(1 - (JI + 12.375) / 25.0)
    assert e.total() == pytest.approx(1 - (1 - e.skip - e.machine_eps - e.luck) * (1 - e.sched))
    assert E.eps(E.design("dither-xin", data), PI, 8, data).sched is None   # another fold: unmeasured


def test_designs_read_what_exists():
    c = census(model(), vector(model().current).epoch)
    for n in E.designs():
        d = E.design(n)
        assert d.name == n and d.honest in E.profiles()
        assert set(d.fold) | set(d.baseline) | {k for f in d.filler for k in f.mix} <= set(c), n
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
    assert out.startswith("dither-duo: ") and "binding JI3e_0" in out and "ε_machine  2.747%" in out
    assert "q* threshold (ε_machine = its floor): none ≤ 64; lanes alone 2.029" in out
    assert main(["eps", "dither-two-sided", "--upfill", "--q", "4", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-two-sided+up" and d["operands"] == "free"
    assert d["terms"]["machine"] == pytest.approx(d["floor"]) and d["threshold"] == pytest.approx(2.029, abs=1e-3)
    assert len(d["curve"]) == 32 and d["curve"][0][0] == 0.5 and d["curve"][-1][0] == 16
    assert d["curve_uniform"][7] == pytest.approx([4, d["terms"]["machine"]])   # q*_U = q* = 4
    assert main(["eps", "--q-uniform", "32", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-duo@qu32" and d["floor"] < d["terms"]["machine"] < E.eps(DUO, PI, 8).machine_eps
    assert main(["eps", "dither-two-sided", "--upfill", "--operands", "honest", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["terms"]["machine"] == 0 and d["threshold"] == pytest.approx(2.123, abs=1e-3)
    assert main(["eps", "--table", "dither-duo", "--fold-sweep", "1.5,3", "--operands", "floor"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Π VF3l-16, operands floor")
    assert [x.split()[0] for x in lines[2:]] == ["dither-duo@r1.5", "dither-duo@r3"]
