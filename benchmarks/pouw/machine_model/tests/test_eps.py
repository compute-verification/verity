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


def test_the_honest_profile():
    assert H.base() == {"MMA": 1, "LDS128": 0.25, "ALU": 0.125, "TMA512": 0.125}
    assert sum(H.per_mma.values()) - sum(H.base().values()) == 12   # duo: 12 warp instructions a MMA
    assert H.words == 128 and H.depth == 32 and H.L == 32


def test_honest_binds_the_joint_issue_row():
    """Honest a MMA: one MMA on JI3e_0 plus 12.375 other warp instructions at one SMSP-cycle each."""
    e = E.eps(DUO, PI, 8)
    assert e.as_built.binding == ("JI3e_0",)
    assert e.as_built.T * 128 == pytest.approx(JI + 12.375, rel=1e-12)
    assert e.baseline.T == pytest.approx(e.as_built.T, rel=1e-9)   # dp4a doesn't pay beside an issue-bound kernel


def test_the_lane_route_pays_below_the_mma_s_issue_cost():
    """A word on lanes costs q*/32 issue-cycles; on the tensor pipe (JI3e_0 + 0.375)/128 besides the fold both pay."""
    t = E.threshold(DUO, PI, uniform=False)
    assert t == pytest.approx((JI + 0.375) / 4, abs=1e-5)
    assert E.eps(DUO, PI, t - 0.01, uniform=False).machine_eps > 0
    assert E.eps(DUO, PI, t + 0.01, uniform=False).machine_eps == 0


@pytest.mark.parametrize("q", [4.0, 8.0, 16.0])
def test_the_uniform_route_in_closed_form(q):
    """Words written and folded on the uniform datapath at q* + 3 uniform instructions: the cheater balances honest's
    JI3e_0 time on the rest against UP, so ε_machine = T_H/(T_H + (q* + 3)·UP)."""
    e = E.eps(DUO, PI, q, lanes=False)
    th = e.as_built.T
    assert e.machine_eps == pytest.approx(th / (th + (q + 3) * UP), rel=1e-9)
    assert set(e.cheat.binding) == {"JI3e_0", "UP"}


def test_upfill_closes_the_uniform_route():
    d = DUO.with_upfill()
    e = E.eps(d, PI, 8)
    assert e.filler_per_word == pytest.approx(e.as_built.T / UP)
    assert e.as_built.T == pytest.approx(E.eps(DUO, PI, 8).as_built.T)   # it costs honest nothing
    assert e.machine_eps == 0 and E.eps(DUO, PI, 8).machine_eps > 0.009
    assert E.threshold(d, PI) == pytest.approx(E.threshold(DUO, PI, uniform=False), abs=1e-5)
    with pytest.raises(SystemExit, match="has a filler already"):
        d.with_upfill()


def test_the_curve_falls_with_q():
    xs = [E.eps(DUO, PI, q).machine_eps for q in E.QS]
    assert all(a >= b - 1e-12 for a, b in zip(xs, xs[1:])) and xs[0] > 0.2 and 0 < xs[-1] < 0.006


def test_fold_cost():
    assert DUO.with_fold_cost(3).fold == DUO.fold
    d = DUO.with_fold_cost(1.5)
    assert d.fold == {"ALU": 1.0, "IMAD": 0.5} and d.fold_cost() == 1.5
    xin = E.design("dither-xin")
    assert E.eps(d, PI, 4).machine_eps == pytest.approx(E.eps(xin, PI, 4).machine_eps, abs=1e-12)
    assert E.eps(d, PI, 4).gamma < E.eps(DUO, PI, 4).gamma


def test_luck_and_the_total():
    assert E.luck(DUO) == pytest.approx(40 * math.log(2) / (0.25 * 121180))
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
        assert set(d.fold) | set(d.baseline) | set((d.filler.mix or {})) <= set(c), n
        assert all("note:" in p.source or "CALC" in p.source or "assumption" in p.source
                   for p in (*d.skip.values(), *d.mand.values())), n
        assert 0 <= E.eps(d, PI, 8).total() < 1
    with pytest.raises(SystemExit, match="no design 'none'"):
        E.design("none")


def test_cli(capsys):
    assert main(["eps"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("dither-duo: ") and "binding JI3e_0" in out and "ε_machine  0.951%" in out
    assert "q* threshold (ε_machine = 0): none ≤ 64; lanes alone 2.123" in out
    assert main(["eps", "dither-two-sided", "--upfill", "--q", "4", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-two-sided+up" and d["terms"]["machine"] == 0 and d["threshold"] == pytest.approx(
        2.123, abs=1e-3)
    assert len(d["curve"]) == 31 and d["curve"][0][0] == 1 and d["curve"][-1][0] == 16
    assert main(["eps", "--table", "dither-duo", "--fold-sweep", "1.5,3"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [x.split()[0] for x in lines[2:]] == ["dither-duo@r1.5", "dither-duo@r3"]
