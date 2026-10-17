"""`machine-model eps`: the ε budget's terms, ε_machine's LP against hand arithmetic on the census, and the data it reads."""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from verity_machine_model import eps as E
from verity_machine_model.cli import main
from verity_machine_model.model import DATA, census, model, vector

PIN = "VF3l-16"                    # the vector these hand-checked numbers were read on, whatever is current
PI = E.machine(PIN)
PIU = E.uniform_on_issue(PI, 1.0)   # decision 73: a uniform instruction takes an issue slot, which VF3l-16 doesn't price
SPECS = Path(__file__).resolve().parents[3] / "ppl" / "specs"   # the perplexity arms a design names
H = E.honest("dither-duo-l32")
M = E.honest("e-l7qs")
JI = PI.c["MMA"]["JI3e_0"]          # a MMA's SMSP-cycles on the joint issue row that binds honest
UP = PI.c["UNI"]["UP"]              # a uniform instruction's on the uniform pipe
OPERANDS = {"free": 0.0, "floor": 0.2, "honest": 0.375}   # the cheater's issue-cycles a MMA besides the MMA and fold


def provisional(name: str) -> E.Design:
    """A design on the provisional profile (no overhead beyond its operands), whose arithmetic is by hand."""
    return replace(E.design(name), honest=H.name, floor=H.floor)


DUO = provisional("dither-1s")


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
    e = E.eps(E.design("dither-1s"), PI, 8)
    assert e.T_honest_source == "measured" and e.T_honest * 128 == M.measured
    assert e.sched == pytest.approx(1 - 22.4334 / 26.1221, abs=1e-5)
    assert e.clock == pytest.approx(1 - 2092 / 2430, abs=1e-4)   # SPEC's P1: the prover sets its own clocks
    assert e.total() == pytest.approx(1 - (1 - e.skip - e.machine_eps - e.luck) * (1 - e.sched) * (1 - e.clock))
    assert e.floor == pytest.approx((22.4334 - JI - 12) / 22.4334, abs=1e-5)   # honest's overhead, under H's credit
    m = E.eps(E.design("dither-1s"), PI, 8, credit="model")   # PROFILE's B(H)/f_max: ε_sched, ε_clock leave ε for γ
    assert m.sched is None and m.clock == 0 and m.machine_eps == e.machine_eps and m.total() < e.total()
    assert m.credit_ratio == pytest.approx(1 / ((1 - e.sched) * (1 - e.clock)), rel=1e-9)
    r = E.eps(E.design("dither-1s").with_fold_cost(1.5), PI, 8)   # another fold: H's measured overhead carries over
    assert r.sched is None and r.T_honest_source.startswith("measured +")
    assert r.T_honest - r.baseline.T == pytest.approx(e.T_honest - e.baseline.T, rel=1e-9)
    fill = E.eps(E.design("dither-2s-fill"), PI, 8)
    assert fill.T_honest - fill.baseline.T == pytest.approx(e.T_honest - e.baseline.T, rel=1e-9)


def test_the_noise_sets_the_operand_floor():
    """SPEC §3.4: 1/39 a MMA where B̃ is resident at the zero instance, 0.20 where both operands are fresh."""
    assert E.design("dither-1s").floor == {"LDSM_X4": pytest.approx(1 / 39, abs=1e-6)}
    assert E.design("dither-2s").floor == {"LDSM_X4": 0.2} == M.floor
    for n, phi in (("dither-1s", 1 / 39), ("dither-2s", 0.2)):
        s = E.setup(E.design(n), PIU, operands="floor", credit="floor")
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


def test_a_uniform_instruction_on_the_issue_rows():
    """Charged an issue slot (decision 73), a uniform instruction does one value where a lane instruction does 32, so
    past the lanes' threshold the uniform route saves the cheater nothing, and no filler is needed to close it."""
    pi = PIU
    assert pi.c["UNI"]["issue"] == pi.c["UNI"]["JI3e_0"] == 1.0 and pi.c["UNI"]["UP"] == UP
    assert "issue" not in PI.c["UNI"] and "UNI on issue at 1" in pi.vector
    refit = E.Machine(vector="X", rows=("issue", "JI3n_0", "JI3n_1", "A", "UP"), cuda=(),
                      c={"ALU": {"issue": 1.0, "JI3n_0": 0.9346, "JI3n_1": 0.2963, "A": 2.0}, "UNI": {"UP": UP}})
    assert E.uniform_on_issue(refit, 0.92).c["UNI"] == pytest.approx({"UP": UP, "issue": 0.92, "JI3n_0": 0.92 * 0.9346})
    charged = replace(refit, c={**refit.c, "UNI": {"UP": UP, "issue": 1.0, "JI3n_0": 0.69}})   # VF3p-16's kind of census
    assert not E.uniform_issue_charged(refit) and E.uniform_issue_charged(charged)
    assert E.uniform_on_issue(charged, 0.92) is charged
    old, new = E.setup(DUO, PI, credit="floor"), E.setup(DUO, pi, credit="floor")
    assert new.baseline.T == pytest.approx(old.baseline.T, rel=1e-9)
    assert E.threshold(old) is None and E.threshold(new) == pytest.approx(E.threshold(old, uniform=False), abs=1e-5)
    assert new.machine_eps(8) == pytest.approx(new.floor(), abs=1e-9)
    assert new.machine_eps(8) < old.machine_eps(8) - 0.005


def test_lane_filler_dilutes_the_operand_floor():
    """x lane instructions a word of verified filler add 4x issue-cycles a MMA to honest, and the cheater must run them
    too, so what it saves on operands is a smaller share. The design file is Dither-2S with 4.5 of them."""
    two = provisional("dither-2s")
    fill = provisional("dither-2s-fill")
    f = fill.filler[0]
    assert (f.per_word, f.mix, f.route) == (4.5, E.lane_fill(4.5).mix, "lanes")
    assert two.with_fill(4.5).filler == (E.lane_fill(4.5),)
    e0, e = E.eps(two, PIU, 8), E.eps(fill, PIU, 8)
    assert e.as_built.T * 128 == pytest.approx(JI + 12.375 + 4 * 4.5, rel=1e-12)
    assert e.machine_eps == pytest.approx(e.floor, abs=1e-12) and e.machine_eps < e0.machine_eps
    assert e.total() < e0.total() and e.gamma < 3
    assert E.threshold(E.setup(fill, PIU)) == pytest.approx(JI / 4, abs=1e-5)


@pytest.mark.parametrize("n, phi", [("dither-1s", 1 / 39), ("dither-2s", 0.2)])
def test_the_floor_credit(n, phi):
    """SPEC §0.7a: ♭ is the tensor route at φ_del with the fold, B_♭ = JI3e_0's MMA + 12 + φ_del a MMA. Under it ε_sched,
    ε_clock and honest's overhead leave ε for γ, and with a uniform instruction on the issue rows the cheater paying
    the floor saves nothing past the threshold. Paying nothing, it saves φ_del on JI3e_0."""
    d = E.design(n)
    e = E.eps(d, PIU, 8, operands="floor", credit="floor")
    assert e.baseline.T * 128 == pytest.approx(JI + 12 + phi, abs=1e-6) and "JI3e_0" in e.baseline.binding
    assert e.sched is None and e.clock == 0 and e.machine_eps == 0
    assert e.credit_ratio == pytest.approx(M.measured / (JI + 12 + phi) / (1 - d.clock.eps), abs=1e-6)
    assert e.total() == pytest.approx(e.skip + e.luck, abs=1e-12)
    free = E.eps(d, PIU, 8, credit="floor")
    assert free.machine_eps == pytest.approx(phi / (JI + 12 + phi), abs=1e-8)
    assert free.gamma == e.gamma == E.eps(d, PIU, 8).gamma   # the credit doesn't change what honest runs
    with pytest.raises(SystemExit, match="the floor program delivers operands"):
        E.setup(d, PI, operands="honest", credit="floor")
    with pytest.raises(SystemExit, match="credit: one of"):
        E.setup(d, PI, credit="none")


def test_plain_work_counts_in_gamma_only():
    """A design that runs plain (unfolded) MMAs beside its work, as CO-Dither at r = k/2 runs a k/2 strip on one limb,
    pays them in γ and nowhere else: γ + 0.5."""
    duo = E.design("dither-1s")
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
    assert E.eps(d, PI, 4).gamma < E.eps(DUO, PI, 4).gamma


def test_a_cheaper_fold_raises_the_threshold():
    """Why XOR-in (1.5 a word) and LI (1) left the ε designs: a cheaper fold leaves lane time idle beside the tensor
    route, the credited program spends it on dp4a words (8 IMAD and 0.5 LDS.128 a word), and a cheater's lane words
    save time until q* reaches dp4a's 8.5, against duo's 2.03."""
    for r in (1.0, 1.5):
        s = E.setup(E.design("dither-1s").with_fold_cost(r), PI, operands="floor", credit="floor")
        assert E.threshold(s, uniform=False) == pytest.approx(8.5, abs=1e-3)
    assert E.threshold(E.setup(E.design("dither-1s"), PI, operands="floor", credit="floor"), uniform=False) < 2.1


def test_the_certificate_and_the_terms_add_up():
    """π* is the LP's dual: a convex combination of rows pricing the cheater's mix at its time. At π*, ε_machine splits
    into ε_cert and ε_route, and the lucky cheater's time is 1 − ε_machine − ε_luck of honest's."""
    cases = [(d, pi, o, c) for d in (DUO, E.design("dither-1s"), E.design("dither-2s-fill")) for pi in (PI, PIU)
             for o in OPERANDS for c in E.CREDITS if not (c == "floor" and o == "honest")]
    for d, pi, operands, credit in cases:
        e = E.eps(d, pi, 8, operands=operands, credit=credit)
        assert sum(e.cheat.pi.values()) == pytest.approx(1, abs=1e-9)
        assert e.cheat.price(pi, e.cheat.n) == pytest.approx(e.cheat.T, rel=1e-9)
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
    assert E.adapt("segment").eps == 0.21 and E.adapt("whole").eps == 0.0035
    assert {E.design(n).unit for n in E.designs()} == {"whole"}   # decision 72: whole dot products
    e0 = E.eps(DUO, PI, 8)
    e = E.eps(DUO, PI, 8, adapt=E.adapt("whole"))
    assert e0.terms()["adapt"] is None and e.terms()["adapt"] == 0.0035
    assert e.total() == pytest.approx(1 - (1 - e.skip - e.machine_eps - e.luck - 0.0035) * (1 - e.clock))
    assert E.eps(DUO, PI, 8, adapt=E.adapt("per-stage")).total() == pytest.approx(e0.total())
    with pytest.raises(SystemExit, match="no proof unit 'none'"):
        E.adapt("none")
    assert all(E.adapt(n).selection for n in E.adapts())


def test_two_bit_entries_on_the_indexed_operand(tmp_path):
    """b-bit noise entries on the operand the table route indexes cost it b times the 1-bit lane instructions and bytes;
    at b = 2 (2.44 slots) the route is above the lane threshold, so the core ε is the closed route's."""
    t = E.table()
    d = E.design("dither-c")
    assert d.index_bits == 1 and d.table(t) is t and d.table(None) is None
    b2 = d.with_index_bits(2)
    assert b2.name == "dither-c@b2" and (b2.table(t).q, b2.table(t).bytes) == (2 * t.q, 2 * t.bytes)
    pi = E.uniform_on_issue(PI, 0.92)
    closed, one = (E.eps(x, pi, 8, operands="floor", credit="floor", table=tb) for x, tb in ((d, None), (d, t)))
    two = E.eps(b2, pi, 8, operands="floor", credit="floor", table=t)
    assert one.core() > closed.core() + 0.01 and two.core() == pytest.approx(closed.core(), abs=1e-9)
    data = tmp_path / "data"
    shutil.copytree(E.DATA, data)
    f = data / "designs" / "dither-c.toml"
    f.write_text(f.read_text() + "\n[table]\nbits = 2\n")
    assert E.design("dither-c", data).index_bits == 2 and E.design("dither-c-peel", data).index_bits == 2


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
    d = E.design("dither-2s-fill")
    assert all(f.p == 0.854 for f in d.filler)
    e = E.eps(d, PIU, 8, operands="floor", credit="floor")
    s = e.lucky.short
    assert s["filler 1"] > 0 and (1 - d.p) * s["words"] + sum(
        (1 - f.p) * s[f"filler {j + 1}"] for j, f in enumerate(d.filler)) == pytest.approx(E.budget(d), rel=1e-6)
    words_only = E.eps(replace(d, filler=tuple(replace(f, p=0.0) for f in d.filler)), PI, 8, operands="floor",
                       credit="floor")
    assert e.luck > words_only.luck and words_only.lucky.short["filler 1"] == 0


def test_the_table_route_is_bound_by_L2():
    d = E.design("dither-1s")
    base = E.setup(d, PIU, credit="floor")
    wide = E.Table(q=1.25, bytes=8.0, l2=1e18, sms=188, mhz=2092, source="")    # L2 never binds
    unbound = E.setup(d, PIU, credit="floor", table=wide)
    assert unbound.cheat(2.16, lanes=False).T == pytest.approx(base.cheat(1.25).T, rel=1e-9)
    shut = E.setup(d, PIU, credit="floor", table=replace(wide, l2=1e9))         # 1 GB/s: next to no table words
    assert shut.machine_eps(2.16) == pytest.approx(base.machine_eps(2.16), abs=1e-5)
    t = E.table()
    assert t.per_cycle() == pytest.approx(6e12 / (188 * 4 * 2092e6))
    s = E.setup(d, PIU, credit="floor", table=t)
    m = s.cheat(2.16)
    assert "L2" in m.binding and m.words["table"] * t.bytes / t.per_cycle() == pytest.approx(m.T, rel=1e-9)
    assert 0 < m.words["table"] < 0.2 and s.machine_eps(2.16) > base.machine_eps(2.16)
    assert s.floor() == pytest.approx(s.machine_eps(8.0), abs=1e-9)        # the floor keeps the table route
    measured = E.setup(d, PIU, credit="floor", table=E.table(q=1.906))
    assert base.machine_eps(2.16) < measured.machine_eps(2.16) < s.machine_eps(2.16)


def test_designs_read_what_exists():
    """Every design reads the census, names a leaf of NOISE for each product, a proof unit and, where one is on main,
    its perplexity arm; decision 79's three names are designs, and each alias names one design."""
    c = census(model(), vector(PIN).epoch)
    for n in E.designs():
        d = E.design(n)
        assert d.name == n and d.honest in E.profiles() and d.floor and "SPEC" in d.floor_source
        assert set(d.fold) | set(d.baseline) | set(d.floor) | {k for f in d.filler for k in f.mix} <= set(c), n
        assert all("note:" in p.source or "CALC" in p.source or "assumption" in p.source
                   for p in (*d.skip.values(), *d.mand.values())), n
        assert 0 <= E.eps(d, PI, 8).total() < 1
        assert d.title.startswith("Dither-") and set(d.noise) == set(E.PRODUCTS) and d.unit in E.adapts(), n
        assert d.coin_bits >= 1 and "note:" in d.coins_source, n
        assert not d.ppl or (SPECS / f"{d.ppl}.toml").read_text(), n
        assert all(E.design(a).name == n for a in d.aliases), n
    assert {E.design(n).title for n in E.designs()} >= {"Dither-1S", "Dither-2S", "Dither-C"}
    for n in E.profiles():
        assert set(E.honest(n).per_mma) | set(E.honest(n).floor) <= set(c), n
    with pytest.raises(SystemExit, match="no design 'none'"):
        E.design("none")


def test_a_design_reads_its_base_and_the_defaults(tmp_path):
    """A variant states what it changes: the rest comes from its base and data/designs.toml, but never its name, title,
    summary or aliases, which would collide."""
    two, fill = E.design("dither-2s"), E.design("dither-2s-fill")
    assert fill.base == "dither-2s" and fill.skip == two.skip and fill.floor == two.floor and fill.ppl == two.ppl
    assert fill.aliases == ("dither-two-sided-fill",) and fill.title != two.title and fill.filler and not two.filler
    assert E.design("dither-2s-pv").ppl == "" and E.design("dither-c-peel").noise["pv"] == "peel"
    assert E.design("A") == two and E.design("dither-duo").name == "dither-1s" and E.aliases()["Bc"] == "dither-c-peel"
    assert two.clock.eps == 0.1391 and two.checks == 48000 and two.baseline_name == "dp4a" and two.fold_cost() == 3
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    bad = data / "designs" / "x.toml"
    bad.write_text('base = "x"\nname = "x"\ntitle = "Dither-X"\nsummary = "x"\n')
    with pytest.raises(SystemExit, match="its bases loop"):
        E.design("x", data)
    bad.write_text('base = "dither-2s"\nname = "x"\nsummary = "x"\n')
    with pytest.raises(SystemExit, match="own name, title, summary"):
        E.design("x", data)
    bad.write_text('base = "dither-2s"\nname = "x"\ntitle = "Dither-X"\nsummary = "x"\naliases = ["A"]\n')
    with pytest.raises(SystemExit, match="alias 'A' names two designs"):
        E.design("x", data)
    bad.write_text('base = "dither-2s"\nname = "x"\ntitle = "Dither-X"\nsummary = "x"\n[noise]\nlinears = "ncp"\n')
    with pytest.raises(SystemExit, match=r"\[noise\] names a leaf"):
        E.design("x", data)
    bad.write_text('base = "dither-2s"\nname = "x"\ntitle = "Dither-X"\nsummary = "x"\n'
                   '[filler]\nper_word = 4\nmix = { ALU = 1 }\nroute = "uniform"\n')
    with pytest.raises(SystemExit, match="decision 73"):
        E.design("x", data)
    bad.write_text('base = "dither-2s"\nname = "x"\ntitle = "Dither-X"\nsummary = "x"\n[coins]\nbits = 2\nsource = "x"\n')
    assert E.design("x", data).index_bits == 2 and two.index_bits == 1
    bad.write_text(bad.read_text() + "[table]\nbits = 1\n")
    with pytest.raises(SystemExit, match="below the design's 2-bit coins"):
        E.design("x", data)


def test_cli(capsys):
    assert main(["eps", "--vector", PIN]) == 0
    out = capsys.readouterr().out
    assert out.startswith("dither-1s: ") and "binding JI3e_0" in out and "ε_machine  11.171%" in out
    assert "Dither-1S (also dither-duo, record, adopted): noise linears one-sided" in out
    assert "q* threshold (ε_machine = its floor): none ≤ 64; lanes alone 2.029" in out and "ε_sched    14.121%" in out
    assert "ε core     " in out and "ε_sample" not in out and "ε_luck" not in out
    assert main(["eps", "--vector", PIN, "--checking", "sampled"]) == 0
    out = capsys.readouterr().out
    assert "ε_adapt 0.350%: proof unit whole" in out and "ε_sample   0.503%" in out
    assert main(["eps", "--vector", PIN, "--checking", "sampled", "--adapt", "segment"]) == 0
    assert "ε_sample   21.153%" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--checking sampled"):
        main(["eps", "--adapt", "whole"])
    assert main(["eps", "--vector", PIN, "dither-2s", "--uni-issue", "1", "--operands", "floor", "--credit",
                 "floor", "--checking", "sampled", "--adapt", "per-stage", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-2s" and d["registry"]["aliases"] == ["dither-two-sided", "A", "adopted-bd-lm"]
    assert d["credit"] == "floor" and d["terms"]["machine"] == 0 and d["terms"]["sched"] is None
    assert d["threshold"] == pytest.approx((JI + 0.2) / 4, abs=1e-5) and d["core"] < 0.002
    assert d["adapt"]["unit"] == "per-stage" and d["terms"]["adapt"] == 0
    assert d["sample"]["eps"] == pytest.approx(d["terms"]["luck"]) and d["sample"]["budget"]["checks"] == 48000
    assert main(["eps", "--vector", PIN, "dither-2s", "--uni-issue", "1", "--q", "4", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-2s" and d["operands"] == "free"
    assert d["terms"]["machine"] == pytest.approx(d["floor"]) and d["threshold"] == pytest.approx(2.029, abs=1e-3)
    assert len(d["curve"]) == 32 and d["curve"][0][0] == 0.5 and d["curve"][-1][0] == 16
    assert d["curve_uniform"][7] == pytest.approx([4, d["terms"]["machine"]])   # q*_U = q* = 4
    assert main(["eps", "--vector", PIN, "--q-uniform", "32", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-1s@qu32"
    assert d["floor"] < d["terms"]["machine"] < E.eps(E.design("dither-1s"), PI, 8).machine_eps
    assert main(["eps", "--vector", PIN, "dither-2s", "--uni-issue", "1", "--operands", "honest", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    base = PIU.load(M.base())["JI3e_0"]   # honest's MMA and its other issue a MMA, the fold aside
    assert d["terms"]["machine"] == 0 and d["threshold_lanes"] == pytest.approx(base / 4, abs=1e-5)
    assert d["threshold"] >= d["threshold_lanes"] - 1e-6
    assert main(["eps", "--vector", PIN, "--uni-issue", "1", "--credit", "floor", "--q", "2.16", "--lookups",
                 "--lookup-q", "1.906", "--json"]) == 0
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
    assert [x.split()[0] for x in lines[2:]] == ["dither-1s@r1.5", "dither-1s@r3"]
    assert main(["eps", "--vector", PIN, "--table", "dither-c", "dither-2s", "--operands", "floor", "--credit",
                 "floor", "--checking", "sampled"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "checking sampled" in lines[0] and "the proof unit (each design its own)" in lines[0]
    assert lines[1].split()[-5:] == ["|", "unit", "luck", "adapt", "sample"] and lines[2].split()[-5:-3] == ["|", "whole"]
    luck, adapt, sample = (float(x.rstrip("%")) for x in lines[2].split()[-3:])
    assert adapt == 0.35 and sample == pytest.approx(luck + adapt, abs=1e-3)
    assert main(["designs"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [x.split()[0] for x in lines[1:1 + len(E.designs())]] == E.designs()
    assert main(["designs", "--json"]) == 0
    reg = {r["name"]: r for r in json.loads(capsys.readouterr().out)}
    assert reg["dither-c"]["noise"] == {"linears": "co32c", "lm_head": "co32c", "qk": "cohalf", "pv": "pv"}
    assert reg["dither-1s"]["ppl"] == "adopted" and reg["dither-c"]["ppl"] is None
