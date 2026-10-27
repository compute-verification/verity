"""`machine-model eps`: the ε budget's terms, ε_machine's LP against hand arithmetic on the census, and the data it reads."""
from __future__ import annotations

import json
import math
import shutil
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from verity_machine_model import eps as E
from verity_machine_model.cli import main
from verity_machine_model.headline import spec as headline_spec
from verity_machine_model.model import DATA, DEVICES, census, device, devices, model, rows, vector, vectors

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
    assert (f.per_word, f.mix, f.route, f.p) == (4.5, E.lane_fill(4.5).mix, "lanes", E.lane_fill(4.5).p)
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
    assert e.sched is None and e.mand == 0 and e.skip == 0.123
    assert e.total() == pytest.approx(1 - (1 - e.skip - e.machine_eps - e.luck) * (1 - e.clock))


def test_adapt_is_a_proof_units_term():
    """ε_adapt (SPEC's Theorem D) sits in the first factor beside ε_skip, ε_machine and ε_luck, and is the proof block's."""
    assert set(E.adapts()) == {"segment", "whole", "whole-recompute", "per-stage"}
    assert E.adapt("segment").eps == 0.21 and E.adapt("whole").eps == 0.0035
    assert {E.design(n).unit for n in E.designs()} == {"whole"}   # decision 72: whole dot products
    e0 = E.eps(DUO, PI, 8)
    e = E.eps(DUO, PI, 8, adapt=E.adapt("whole"))
    assert e0.terms()["adapt"] is None and e.terms()["adapt"] == 0.0035
    assert e.total() == pytest.approx(1 - (1 - e.skip - e.machine_eps - e.luck - 0.0035) * (1 - e.clock))
    assert E.eps(DUO, PI, 8, adapt=E.adapt("per-stage")).total() == pytest.approx(e0.total())
    with pytest.raises(SystemExit, match="no proof block 'none'"):
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
    """The core ε is the protocol with 100% checking, Corollary D3's: every check is one draw, so ε_luck
    is in it (LEDGER; JL is not assumed) and ε_adapt = 0; ε_sample = ε_adapt is the sampling compiler's and never
    enters it."""
    e = E.eps(DUO, PI, 8, adapt=E.adapt("whole"))
    assert e.core() == pytest.approx(
        1 - (1 - e.skip - e.mand - e.machine_eps - e.luck) * (1 - (e.sched or 0)) * (1 - e.clock))
    assert e.sample() == pytest.approx(0.0035) and e.luck > 0
    assert e.core() == pytest.approx(E.eps(DUO, PI, 8).core()) and e.core() < e.total()
    assert e.core(sched=False) <= e.core()


def test_luck_spends_one_budget_on_words_and_filler():
    """Theorem S's budget ln(1/δ)/S weights each deficient block by 1 − p: ¾ a word, and 0 a lane filler block, whose
    receipt the check recomputes (PinnedFillBlocksDeficit), so the lucky cheater skips words alone. At pfill's 0.854 a
    filler block is cheaper to skip than a word and the cheater leaves filler undone; with no luck credit (p = 1) it
    leaves the whole filler undone, even with every block checked."""
    d = E.design("dither-2s-fill")
    assert [(f.p, f.luck.split(":")[0]) for f in d.filler] == [(0.0, E.PINNED.split(" ")[0])]
    e = E.eps(d, PIU, 8, operands="floor", credit="floor")
    assert e.lucky.short["filler 1"] == 0
    assert (1 - d.p) * e.lucky.short["words"] == pytest.approx(E.budget(d), rel=1e-6)

    def at(p: float, operands: str = "floor") -> E.Eps:
        return E.eps(replace(d, filler=tuple(replace(f, p=p) for f in d.filler)), PIU, 8, operands=operands,
                     credit="floor")

    s = at(0.854).lucky.short
    assert s["filler 1"] > 0 and (1 - d.p) * s["words"] + (1 - 0.854) * s["filler 1"] == pytest.approx(
        E.budget(d), rel=1e-6)
    assert at(0.854).luck > e.luck
    assert at(1.0).lucky.short["filler 1"] == pytest.approx(1)
    assert at(1.0, "free").core() > at(0.0, "free").core() > at(0.0).core()


def test_the_table_route_is_bound_by_L2():
    d = E.design("dither-1s")
    base = E.setup(d, PIU, credit="floor")
    wide = E.Table(q=1.25, bytes=8.0, l2=1e18, sms=188, mhz=2092, source="")    # L2 never binds
    unbound = E.setup(d, PIU, credit="floor", table=wide)
    assert unbound.cheat(2.16, lanes=False).T == pytest.approx(base.cheat(1.25).T, rel=1e-9)
    shut = E.setup(d, PIU, credit="floor", table=replace(wide, l2=1e9))         # 1 GB/s: next to no table words
    assert shut.machine_eps(2.16) == pytest.approx(base.machine_eps(2.16), abs=1e-5)
    t = E.table()
    assert t.per_cycle() == pytest.approx(7.244e12 / (188 * 4 * 2092e6))
    s = E.setup(d, PIU, credit="floor", table=t)
    m = s.cheat(2.16)
    assert "L2" in m.binding and m.words["table"] * t.bytes / t.per_cycle() == pytest.approx(m.T, rel=1e-9)
    assert 0 < m.words["table"] < 0.25 and s.machine_eps(2.16) > base.machine_eps(2.16)
    assert s.floor() == pytest.approx(s.machine_eps(8.0), abs=1e-9)        # the floor keeps the table route
    measured = E.setup(d, PIU, credit="floor", table=E.table(q=1.906))
    assert base.machine_eps(2.16) < measured.machine_eps(2.16) < s.machine_eps(2.16)


def test_designs_read_what_exists():
    """Every design reads the census, names a leaf of NOISE for each product, a proof block and, where one is on main,
    its perplexity arm; decision 79's three names are designs, and each alias names one design."""
    c = census(model(), vector(PIN).epoch)
    for n in E.designs():
        d = E.design(n)
        assert d.name == n and d.honest in E.profiles() and d.floor and "SPEC" in d.floor_source
        assert set(d.fold) | set(d.baseline) | set(d.floor) | {k for f in d.filler for k in f.mix} <= set(c), n
        assert all("note:" in p.source or "CALC" in p.source or "assumption" in p.source
                   for p in (*d.skip.values(), *d.mand.values())), n
        assert all(p.source.startswith("eps_budget.py") for p in d.skip.values()), n
        assert 0 <= E.eps(d, PI, 8).total() < 1
        assert d.title.startswith("Dither-") and set(d.noise) == set(E.PRODUCTS) and d.unit in E.adapts(), n
        assert d.coin_bits >= 1 and "note:" in d.coins_source, n
        assert not d.ppl or (SPECS / f"{d.ppl}.toml").read_text(), n
        assert all(E.design(a).name == n for a in d.aliases), n
    assert {E.design(n).title for n in E.designs()} >= {"Dither-1S", "Dither-2S", "Dither-C"}
    for n in E.profiles():
        h = E.honest(n)
        assert set(h.per_mma) - {h.form} | set(h.floor) <= set(c) and h.form in PI.c, n
    with pytest.raises(SystemExit, match="no design 'none'"):
        E.design("none")


def test_a_design_reads_its_base_and_the_defaults(tmp_path):
    """A variant states what it changes: the rest comes from its base and data/designs.toml, but never its name, title,
    summary or aliases, which would collide."""
    two, fill = E.design("dither-2s"), E.design("dither-2s-fill")
    assert fill.base == "dither-2s" and fill.noise == two.noise and fill.floor == two.floor and fill.ppl == two.ppl
    assert E.design("co-dither-l16").sums == E.design("dither-c-peel").sums == "vfold" != E.design("dither-c").sums
    assert fill.aliases == ("dither-two-sided-fill",) and fill.title != two.title and fill.filler and not two.filler
    assert E.design("dither-2s-pv").ppl == "adopted-bd-lm-pv" and E.design("dither-c-peel").noise["pv"] == "peel"
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
    bad.write_text('base = "dither-2s"\nname = "x"\ntitle = "Dither-X"\nsummary = "x"\nsums = "vsum"\n')
    assert E.design("x", data).sums == "vsum" and two.sums == "vfold"
    bad.write_text('base = "dither-2s"\nname = "x"\ntitle = "Dither-X"\nsummary = "x"\nsums = "vscan"\n')
    with pytest.raises(SystemExit, match="not a peel sums kernel"):
        E.design("x", data)
    bad.write_text('base = "dither-2s"\nname = "x"\ntitle = "Dither-X"\nsummary = "x"\n'
                   '[filler]\nper_word = 4\nmix = { ALU = 1 }\nroute = "uniform"\n')
    with pytest.raises(SystemExit, match="decision 73"):
        E.design("x", data)
    head = 'base = "dither-2s"\nname = "x"\ntitle = "Dither-X"\nsummary = "x"\n[filler]\nper_word = 4\nmix = { ALU = 1 }\n'
    for luck in ("", "p = 0.854\n", "p = 1.5\nluck = \"x\"\n"):
        bad.write_text(head + luck)
        with pytest.raises(SystemExit, match="theorem that proves it"):
            E.design("x", data)
    bad.write_text(head + "p = 1.0\n")
    assert [(f.p, f.luck) for f in E.design("x", data).filler] == [(1.0, "")]
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
    assert "ε core     " in out and "ε_sample" not in out and "ε_luck     0.153%" in out
    assert main(["eps", "--vector", PIN, "--checking", "sampled"]) == 0
    out = capsys.readouterr().out
    assert "ε_adapt 0.350%: proof block whole" in out and "ε_sample   0.350%" in out
    assert main(["eps", "--vector", PIN, "--checking", "sampled", "--adapt", "segment"]) == 0
    assert "ε_sample   21.000%" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--checking sampled"):
        main(["eps", "--adapt", "whole"])
    assert main(["eps", "--vector", PIN, "dither-2s", "--uni-issue", "1", "--operands", "floor", "--credit",
                 "floor", "--checking", "sampled", "--adapt", "per-stage", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["design"] == "dither-2s" and d["registry"]["aliases"] == ["dither-two-sided", "A", "adopted-bd-lm"]
    assert d["credit"] == "floor" and d["terms"]["machine"] == 0 and d["terms"]["sched"] is None
    assert d["threshold"] == pytest.approx((JI + 0.2) / 4, abs=1e-5) and d["core"] < 0.0045
    assert d["adapt"]["unit"] == "per-stage" and d["terms"]["adapt"] == 0
    assert d["sample"]["eps"] == d["terms"]["adapt"] == 0 and d["luck_budget"]["checks"] == 48000
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
    assert "checking sampled" in lines[0] and "the proof block (each design its own)" in lines[0]
    head = lines[1].split()
    assert head[-4:] == ["|", "unit", "adapt", "sample"] and lines[2].split()[-4:-2] == ["|", "whole"]
    assert head[head.index("floor") + 1:head.index("floor") + 3] == ["luck", "core@2"]
    adapt, sample = (float(x.rstrip("%")) for x in lines[2].split()[-2:])
    assert adapt == sample == 0.35
    assert main(["designs"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [x.split()[0] for x in lines[1:1 + len(E.designs())]] == E.designs()
    assert main(["designs", "--json"]) == 0
    reg = {r["name"]: r for r in json.loads(capsys.readouterr().out)}
    assert reg["dither-c"]["noise"] == {"linears": "co32c", "lm_head": "co32c", "qk": "cohalf", "pv": "pv"}
    assert reg["dither-1s"]["ppl"] == "adopted" and reg["dither-c"]["ppl"] == "co32c-pv"


SM89 = DEVICES / "sm_89"


def test_devices_follow_the_layout():
    """README "Devices": sm_120 is data/ and the default; any other device is data/devices/sm_NN/, priced once it has a
    model.toml, every row of its census some vector's, and every design's honest MMA a class of its census."""
    assert device() == device("sm_120") == device("sm120") == DATA and devices()[0] == "sm_120"
    assert device("sm89") == device("sm_89") == SM89 and "sm_89" in devices()
    for name in devices()[1:]:
        d = device(name)
        pi = E.machine("", d)
        assert set(pi.rows) <= set(rows(d)) == {r for v in vectors(d) for r in vector(v, d).rows}, name
        for n in E.designs(d):
            ds = E.design(n, d)
            h = E.honest(ds.honest, d)
            assert h.cls in pi.c and h.L == ds.L and ds.name == n, (name, n)
            assert set(ds.fold) | set(ds.baseline) | set(ds.floor) | set(h.per_mma) <= set(pi.c), (name, n)
    with pytest.raises(SystemExit, match="isn't a device directory"):
        device("sm_7")


def test_sm89_the_fold_binds_honest():
    """The L40S (EPS1 ADA): at L = 16 duo's two ALU-pipe instructions a word hold the floor program on row A, 8 a MMA at
    ALU's measured rate, above IMMA.16816's issue, so no lane route saves time at any q*. The uniform route does: UNI
    runs on UP at 2.0 a SM-clock (uniform.cu, every form) and costs issue and JI89 its measured cost beside honest16, so
    a cheater writes words there in JI89's slack, less of them as q* grows, and ε_machine never reaches its floor. At
    L = 32, with IMMA.16832's ALU and FFMA facets refitted to FACET-BREAK's schedule (EPS1 ADA r3, opcode_probe/sm89/
    ji.py), JI89 binds the floor program and the cheater alike: lane words save time only below JI89's q*0 (IMMA.16832's
    coefficient over a thread's 4 words, 1.9), and UNI's JI89 cost leaves the uniform route nothing."""
    pi = E.machine("", SM89)
    assert pi.vector == "ADA-89" and set(pi.c[E.UNI]) == {"UP", "issue", "JI89"}
    assert pi.c[E.UNI]["UP"] == pytest.approx(4 / (2.0 * 1.02), rel=1e-3)   # UNI's rate + 2%
    assert pi.c[E.UNI]["JI89"] < pi.c[E.UNI]["issue"] < pi.c["ALU"]["issue"]
    alu = 4 / (1.9985 * 1.02)                            # an ALU instruction's SMSP-cycles on A, its rate + 2%
    d16 = E.design("dither-1s-l16", SM89)
    s16 = E.setup(d16, pi, SM89, "floor", "floor")
    assert s16.as_built.T * 128 == pytest.approx(8 * alu, rel=1e-9) and "A" in s16.as_built.binding
    assert all(s16.machine_eps(q, uniform=False) == 0 for q in (2, 8, 16)) and E.threshold(s16, uniform=False) == 0
    m = [s16.machine_eps(q) for q in (2, 8, 16)]
    assert 1e-3 > m[0] > m[1] > m[2] > 0 and E.threshold(s16) is None
    assert {"JI89", "A"} <= set(s16.cheat(8).binding) and s16.cheat(8).words["lanes"] == 0
    e = E.eps(d16, pi, 2.0, SM89, operands="floor", credit="floor")
    assert e.machine_eps == pytest.approx(m[0]) and e.core() > e.skip
    assert e.T_honest * 128 == pytest.approx(19.069) and e.T_honest_source == "measured"
    assert e.gamma_clock == pytest.approx(e.gamma * 2520 / 1680, rel=1e-6)
    s32 = E.setup(E.design("dither-1s-l32", SM89), pi, SM89, "floor", "floor")
    assert "JI89" in s32.as_built.binding and 1.8 < E.threshold(s32) < 2
    assert all(s32.machine_eps(q) == pytest.approx(0, abs=1e-9) for q in (2, 8, 16)) and s32.machine_eps(1.5) > 0
    assert all(s32.machine_eps(q) == pytest.approx(s32.machine_eps(q, uniform=False), abs=1e-12) for q in (1.5, 2, 8))
    t = E.table(SM89)
    assert t.l2 is None and t.per_cycle() is None and t.mhz == 2520 and t.sms == 142
    open32 = E.setup(E.design("dither-1s-l32", SM89), pi, SM89, "floor", "floor", table=t)
    assert open32.machine_eps(8) == pytest.approx(s32.machine_eps(t.q), abs=1e-9)   # unbounded L2: lanes at q*_T


def test_dc_the_tensor_couples_with_lanes_on_sm90_not_sm100():
    """EPS1 DC's H200 and B200 at L = 32. On sm_90 duo binds the floor program on the joint ALU facet: a wgmma takes
    62.77 SMSP-cycles of it and 62.74 of issue, so lane words save time at every q* (no threshold) and ε_machine is 4%
    at q* = 2.03. On sm_100 the tensor holds no lane pipe row: beside the saturated tcgen05.mma the exact tclaneU_lop3
    keeps the tensor at 0.999 and LOP3 at 0.991 of alone, so census.py writes no JPA100 row, duo binds the ALU pipe A
    alone, and a word on lanes saves nothing there (threshold 0 without the uniform route). The uniform datapath (UP,
    and 0.92 of an issue slot a uniform instruction, its price beside an issue-saturated stream: DC r5) takes 0.59% of
    the B200's words off A at q* = 2.03, so it has no threshold; on the H200 JI90_0 charges a uniform instruction 0.96,
    which closes it. At L = 256 the B200 stays ALU-bound: 8 MMAs hold the tensor pipe 4015 SMSP-cycles a 32768 words,
    under the fold's 4016 of ALU (RT-BLACKWELL-16: JPA100_1, fitted to the block-timed tclane_lop3_r128, coupled the
    two, read this floor tensor-bound at 5410 and the cheat at over 20%)."""
    sm90, sm100 = DEVICES / "sm_90", DEVICES / "sm_100"
    duo = [n for n in E.designs(sm90) if "-li" not in n]
    assert duo == [f"dither-{n}-l{L}" for n in ("1s", "2s", "c") for L in (32, 64, 128, 256)]
    p90 = E.machine("", sm90)
    s = E.setup(E.design("dither-2s-l32", sm90), p90, sm90, "floor", "floor")
    assert s.as_built.T * 4096 == pytest.approx(62.7735919 + 256 * 1.99077535, rel=1e-6)
    assert s.as_built.binding == ("JPA90_0",) and E.threshold(s) is None
    assert s.machine_eps(2.03) > 0.04 and s.machine_eps(8) < 0.001
    assert s.cheat(2.03).words["uniform"] == pytest.approx(0, abs=1e-9)
    p100 = E.machine("", sm100)
    s = E.setup(E.design("dither-2s-l32", sm100), p100, sm100, "floor", "floor")
    load = p100.load(s.as_built.n)
    assert not any(r.startswith("JPA100") for r in load)
    assert s.as_built.binding == ("A",) and s.as_built.T == pytest.approx(load["A"], rel=1e-12)
    assert E.threshold(s, uniform=False) == 0 and s.machine_eps(0.5, uniform=False) == 0
    assert load["TM100_0"] * 32768 == pytest.approx(19.9132141 + 32 * 16.5392944, rel=1e-6)
    assert E.threshold(s) is None and s.cheat(2.03).words["uniform"] == pytest.approx(0.0059, abs=1e-4)
    assert s.machine_eps(2.03) == pytest.approx(0.00595, abs=1e-4)
    far = E.setup(E.design("dither-2s-l256", sm100), p100, sm100, "floor", "floor")
    fl = p100.load(far.as_built.n)
    assert far.as_built.binding == ("A",) and 0.999 < fl["M"] / fl["A"] < 1 and far.machine_eps(2.03) < 0.01


def test_dc_li_binds_honest_on_the_lane_facet():
    """DC-DESIGN's LI designs at L = 32 (one lane instruction a word, LOP3 and IMAD alternating): a balanced fold binds
    honest where duo binds one pipe with issue slack, so relocating a word to lanes stops paying below output counting's
    q* = ½. On sm_90 LI alone leaves the ALU pipe 1.5% short of half issue; 0.02 verified FFMA a word puts honest on
    JI90_0, whose threshold is 0.4999, no margin. On sm_100 the x128 loads are the tensor route's, honest binds issue,
    and γ is the measured tcliU_x128 (107.7 words a SM-clock beside the saturated MMA)."""
    for dev, binds, thr, t in (("sm_90", ("JI90_0",), 0.4999, 190.761), ("sm_100", ("issue",), 0.0088, 1216.6)):
        d = DEVICES / dev
        pi = E.machine("", d)
        for n in ("1s", "2s", "c"):
            ds = E.design(f"dither-{n}-li-l32", d)
            h = E.honest(ds.honest, d)
            s = E.setup(ds, pi, d, "floor", "floor")
            assert h.fold_name == ds.fold_name == "li" and ds.fold == h.fold == {"ALU": 0.5, "IMAD": 0.5}, dev
            assert s.as_built.binding == binds and E.threshold(s) == pytest.approx(thr, abs=1e-4), dev
            assert s.machine_eps(0.5) == pytest.approx(0, abs=1e-9), dev
            e = E.eps(ds, pi, 0.5, d, operands="floor", credit="floor")
            assert e.T_honest * h.words == pytest.approx(t, rel=1e-4), dev
            assert e.core() == pytest.approx(e.skip + e.luck), dev
    assert [(f.mix, f.p) for f in E.design("dither-2s-li-l32", DEVICES / "sm_90").filler] == [({"FFMA": 1}, 0.0)]
    assert E.honest("dc100-li-l32", DEVICES / "sm_100").base() == {"UTCIMMA_M128_N256": 1, "LDTM_32X32B_X128": 8}


def test_sm90_ffma_filler_has_no_uniform_form():
    """sm_90's uniform datapath is integer only (model.toml's `uniform_forms`), so a filler with FFMA in its mix stays on
    lanes even on DC-90-ri, where a uniform instruction is nearly free on JI90_0; an IMAD filler may move there, and a
    device that names no forms lets every filler."""
    d = DEVICES / "sm_90"
    pi = E.machine("DC-90-ri", d)
    assert pi.uniform_forms == ("IMAD", "ALU") and E.machine("", DEVICES / "sm_100").uniform_forms is None
    assert not pi.uniform_form(E.Filler(per_word=0.07, mix={"FFMA": 2, "IMAD": 1}))
    assert pi.uniform_form(E.Filler(per_word=0.07, mix={"IMAD": 1}))
    ds = E.design("dither-2s-licpw-redux-l32", d)
    for q in (1.22, 0.5):
        assert E.eps(ds, pi, q, d, operands="free", credit="floor").cheat.filler_uniform == (0.0,)
    imad = replace(ds, filler=(replace(ds.filler[0], mix={"IMAD": 1.0}),))
    assert E.eps(imad, pi, 1.22, d, operands="free", credit="floor").cheat.filler_uniform[0] > 0.0


def test_the_fold_routes_are_the_cheaters_and_the_floor_programs():
    """RT2-HOPPER-9, -11 and RT3-DC-1, -4: the bit-exact checkpoint routes (four segments' words in one STG.128, an XOR on
    the uniform datapath, shared-memory and L2 atomics, a transpose through shared memory), each instruction at its
    census class, and the check reads the checkpoint word, not its program, so the cheater folds by any mix of the
    design's fold and its `[fold_routes]`, RT4-SHEET-7's generic-address L2 XORs (ATOM.E, no AT90 line) among them, and
    after them the counted folds, none built: RT4-SHEET-13's and -14's moves of LI on li and each route, and li-bulk's
    BT1024 and trimmed W = ∞ members (RT4-SHEET-22); li-bulk itself (RT4-SHEET-17) is built, at its measured cost. With
    ♭ at REDUX they beat it by 1.47% on DS-90 and 2.25% on DC-90-ri, where the cheater folds by the generic CP_RED2 and
    li-bulk's trimmed W = ∞ member (ATOME32, REDUX and the stores on the JI90 lines the H200 measured beside the IGMMA,
    r20261021-070616-d06c, r20261021-074303-da7f and r20261021-161119-62ac). ♭ folds by the built routes alone: by
    cp-red2 (3 LOP3 and 2 REDG a segment) on DS-90, as the cheater does, and on DC-90-ri by CP_RED2 and CP_RED8 (global
    and generic tie there), so ε_machine is 0 on DS-90 and 0.330% on DC-90-ri. Honest's kernel folds by REDUX, so γ
    doesn't move."""
    d = DEVICES / "sm_90"
    ds = E.design("dither-2s-licpw-redux-p1-l32", d)
    built = ["cp-seg4", "cp-seg4u", "cp-ats", "cp-ats2", "cp-ats8", "cp-red", "cp-red2", "cp-red8", "cp-red-generic",
             "cp-red2-generic", "cp-red8-generic", "cp-xt"]
    counted = [r + m for r in ("li", *built) for m in ("-rt3", "-wideh", "-rt3-wideh")]
    assert list(ds.fold_routes) == [*built, *counted, "li-bulk", "li-bulk-bt1024", "li-bulk-trim-inf"]
    assert ds.counted_routes == (*counted, "li-bulk-bt1024", "li-bulk-trim-inf")
    assert list(ds.built_routes()) == [*built, "li-bulk"]
    assert ds.fold_routes_source
    assert ds.with_fold_cost(0.5).fold_routes["cp-seg4u"]["UNI"] == pytest.approx(0.0078125 * 0.5 / ds.fold_cost())
    plain = replace(ds, fold_routes={}, counted_routes=())
    for vec, q, attack, eps in (("DS-90", 0.5, 0.014706, 0.0), ("DC-90-ri", 1.22, 0.022499, 0.003303)):
        pi = E.machine(vec, d)
        s, s0 = E.setup(ds, pi, d, "free", "floor"), E.setup(plain, pi, d, "free", "floor")
        assert not s0.baseline.folds and s.baseline.T < s0.baseline.T
        assert set(s.baseline.folds) <= set(ds.built_routes()), vec
        assert replace(s0, design=ds).machine_eps(q) == pytest.approx(attack, abs=1e-6), vec
        assert s.machine_eps(q) == pytest.approx(eps, abs=1e-6), vec
        h = E.setup(ds, pi, d, "free", "H")
        assert not h.as_built.folds and not h.baseline.folds and h.cheat(q).folds == s.cheat(q).folds
        e, e0 = (E.eps(x, pi, q, d, operands="free", credit="floor") for x in (ds, plain))
        assert e.gamma == e0.gamma and e.T_honest == e0.T_honest
    s = E.setup(ds, E.machine("DS-90", d), d, "free", "floor")
    assert s.baseline.folds == s.cheat(0.5).folds == pytest.approx({"cp-red2": 1.0})
    assert s.baseline.T * 128 == pytest.approx(4.1875)
    assert E.setup(plain, s.machine, d, "free", "floor").baseline.T * 128 == pytest.approx(4.25)


def test_flat_never_folds_by_a_counted_route(tmp_path):
    """♭ is credited only with programs a run has built, so it folds by the design's fold and its built routes alone,
    and a design's `counted` routes are the cheater's only: priced at a thousandth of their counts, they leave ♭ where
    it is on every H200 vector and lower the cheater's time. A `counted` name off the table is refused."""
    d = DEVICES / "sm_90"
    ds = E.design("dither-2s-licpw-redux-p1-l32", d)
    cheap = replace(ds, fold_routes={n: {k: v / 1000 if n in ds.counted_routes else v for k, v in r.items()}
                                     for n, r in ds.fold_routes.items()})
    for vec in ("DS-90", "DC-90", "DC-90-ri"):
        pi = E.machine(vec, d)
        s, c = E.setup(ds, pi, d, "free", "floor"), E.setup(cheap, pi, d, "free", "floor")
        assert set(s.baseline.folds) <= set(ds.built_routes()) and set(c.baseline.folds) <= set(ds.built_routes()), vec
        assert c.baseline.T == pytest.approx(s.baseline.T, rel=1e-12), vec
        assert set(c.cheat(0.5).folds) & set(ds.counted_routes) and c.cheat(0.5).T < s.cheat(0.5).T, vec
    data = tmp_path / "sm_90"
    shutil.copytree(d, data)
    f = data / "designs" / "dither-2s-licpw-redux-l32.toml"
    f.write_text(f.read_text().replace('"li-bulk-trim-inf",\n]', '"li-bulk-trim-inf", "li-bulk-bt2048",\n]', 1))
    with pytest.raises(SystemExit, match="`counted` lists the routes no run has built"):
        E.design("dither-2s-licpw-redux-l32", data)


def test_the_l2_xor_checkpoints_on_the_menu_span_their_family():
    """RT2-HOPPER-1 and RT3-DC-1: the checkpoint's lane XOR in memory, a LOP3 and b red.global.xor (REDG) a segment,
    2a + b ≥ 8, is bit-exact. CP_RED, CP_RED2 and CP_RED8, (a, b) = (4, 1), (3, 2) and (0, 8), are on the menu, a REDG at
    REDG32: one issue slot, and on DC-90 and DC-90-ri AT90's 290 SMSP-cycles too, the L2's atomic row. The family's
    (2, 4) and (1, 6) are mixes of CP_RED2 and CP_RED8, so with them on the menu ♭ and ε_machine don't move."""
    d = DEVICES / "sm_90"
    ds = E.design("dither-2s-licpw-redux-p1-l32", d)
    assert {"cp-red", "cp-red2", "cp-red8"} <= set(ds.fold_routes)
    red = {f"cp-red{b}": {"ALU": (64 + a) / 128, "IMAD": 0.5, "REDG32": b / 128} for a, b in ((2, 4), (1, 6))}
    assert set(E.machine("DS-90", d).c["REDG32"]) == {"DS_issue"}
    for vec in ("DC-90", "DC-90-ri"):
        pi = E.machine(vec, d)
        assert {"issue", "AT90"} <= set(pi.c["REDG32"]), vec
        s, r = (E.setup(x, pi, d, "free", "floor") for x in (ds, replace(ds, fold_routes={**ds.fold_routes, **red})))
        assert r.baseline.T == pytest.approx(s.baseline.T), vec
        assert r.machine_eps(1.22) == pytest.approx(s.machine_eps(1.22), abs=1e-9), vec


HEADLINE_Q = 1.22   # decision 89's reading: the 1.22-slot issue floor (Theorem C, unproved in Lean; CALC r10, #2325)
HEADLINE = {fam: (f["design"], f["measured"]["vector"]) for fam, f in headline_spec()["family"].items()}   # headline.toml's


def _worst_core_under_margin(data, pi, d, m):
    """Core ε with every near measured facet (one the cheater loads to within 5% at the headline q*) moved m in the
    cheater's favour at once: decision 76's margin on the rows the headline reads. The axiomatic issue slot (every warp
    instruction = 1, decision 73) is not a measured row, and the exact operand floor is an architectural count, so both
    are left alone; a facet's measured lane coefficient is the small one it shares (not the tensor-pipe MMA value)."""
    def core(p):
        return E.eps(d, p, HEADLINE_Q, data, operands="free", credit="floor").core()

    c = E.setup(d, pi, data, "free", "floor").cheat(HEADLINE_Q)
    load = pi.load(c.n)
    near = [r for r in pi.rows if r != "issue" and c.T and load[r] / c.T >= 0.95]
    cc = {k: dict(v) for k, v in pi.c.items()}
    for r in near:
        small = min(pi.c[k][r] for k in pi.c if r in pi.c[k])
        hit = [k for k in pi.c if r in pi.c[k] and abs(pi.c[k][r] - small) < 1e-9]
        scaled = lambda f: replace(pi, c={k: ({**v, r: v[r] * f} if k in hit else v) for k, v in cc.items()})
        f = (1 - m) if core(scaled(1 - m)) >= core(scaled(1 + m)) else (1 + m)
        for k in hit:
            cc[k][r] *= f
    return core(replace(pi, c=cc))


def test_dc_the_headline_holds_decision_76s_margin():
    """EPS1 DC's margin audit (note:20261017T0452Z-draft-eps1-dc Round 8): each device family's headline core ε, on
    data/headline.toml's design at its measured reading's vector, stays under 1% when each measured facet the headline
    reads moves 2% in the cheater's favour (decision 76), at the decision-89 reading (operands free, q* = 1.22), ε_luck
    in the core (Corollary D3). A family whose core is at or over 1% before the margin is the headline's own row
    (test_headline pins it, with its reason), so the audit holds only the families under 1% to 1%; the margin never
    lowers a core. It doesn't move the A100 or the RTX PRO 6000, and it moves the L40S by under 0.01 points. How far it
    moves the H200 and the B200 depends on their designs, which the audit reads rather than pins.
    dither-2s-li-l32, the H200's headline before LI-CP-W, is the thin design: its 0.02 FFMA filler puts honest on JI90_0
    by only the ALU pipe's 1.5% issue slack, so a 2% rise in JPA90_0 (ALU lanes beside the tensor) moves honest's
    binding off JI90_0 and opens the lane route, taking core to 1.084% with the peel at sm_90a's own counts (RT-ISA-7;
    0.98% at sm_120's): it crosses 1% past a 1.7% margin. A 0.05 FFMA filler a word restores the full margin (core back
    to its 0.679% floor) at γ 1.589, the cheapest add-on; its ε_skip at the larger ♭ is CALC's to re-price."""
    worst = {}
    for dev, (name, vec) in HEADLINE.items():
        data = device(dev)
        pi = E.machine(vec, data)
        d = E.design(name, data)
        base = E.eps(d, pi, HEADLINE_Q, data, operands="free", credit="floor").core()
        worst[dev] = (base, _worst_core_under_margin(data, pi, d, 0.02))
        assert worst[dev][1] >= base - 1e-12, f"{dev} {name} on {vec}: the margin lowered core"
        if base < 0.01:
            assert worst[dev][1] < 0.01, f"{dev} {name} on {vec}: {worst[dev][1]*100:.3f}% over 1% under decision 76's margin"
    assert {"sm_80", "sm_89", "sm_120"} <= set(worst)
    for dev in ("sm_80", "sm_120"):
        assert worst[dev][1] == pytest.approx(worst[dev][0], rel=1e-9), f"{dev}: the margin moved core"
    assert 0 < worst["sm_89"][1] - worst["sm_89"][0] < 1e-4
    # dither-2s-li-l32 is thin at its 0.02 filler; 0.05 FFMA a word restores the margin to the floor
    sm90 = device("sm_90")
    p90 = E.machine("", sm90)
    thin = E.design("dither-2s-li-l32", sm90)
    assert E.eps(thin, p90, HEADLINE_Q, sm90, operands="free", credit="floor").core() == pytest.approx(0.00681, abs=5e-5)
    assert _worst_core_under_margin(sm90, p90, thin, 0.02) == pytest.approx(0.01084, abs=5e-5)
    assert _worst_core_under_margin(sm90, p90, thin, 0.016) < 0.01 < _worst_core_under_margin(sm90, p90, thin, 0.018)
    wide = replace(thin, filler=(E.lane_fill(0.05, {"FFMA": 1.0}),))
    assert _worst_core_under_margin(sm90, p90, wide, 0.02) == pytest.approx(0.00679, abs=5e-5)


def test_cli_device(capsys):
    assert main(["eps", "--device", "sm89", "--table", "--credit", "floor", "--operands", "floor"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Π ADA-89") and lines[1].split()[3:5] == ["γ", "γ_f"]
    assert [x.split()[0] for x in lines[2:]] == [f"dither-{n}-l{L}" for n in ("1s", "2s", "c") for L in (16, 32)]
    assert main(["--device", "sm_89", "eps", "dither-2s-l16", "--lookups", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["lookups"]["l2"] is None and d["gamma_clock"] > d["gamma"] and d["data"] == str(SM89)
    assert main(["--device", "sm89", "eps", "--lookups"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("dither-1s-l16: ") and "L2 unbounded: unmeasured on this device" in out
    with pytest.raises(SystemExit, match="--device or --data"):
        main(["--data", str(SM89), "--device", "sm89", "eps"])


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


RPI = E.machine("VF3r-16")   # FACET-BREAK r4 (#2286), the fx numbers below were read on it


@pytest.mark.parametrize("x, row, q0", [(1.0, "JI3s_0", 1.1261), (0.5, "JI3s_2", 1.3760)])
def test_mandatory_lane_work_closes_the_lane_route(x, row, q0):
    """One IMAD a word in front of duo (SAT-120's fx1) puts honest's floor program on JI3s_0, where a lane word costs
    more than the IMMA16832 share it replaces from q* = 1.13, under the generic table route's 1.22 slots, so ε_machine
    is 0 at every q* from there, by lanes or lookups. Half of it leaves honest on JI3s_2 (q*₀ 1.38)."""
    d = E.design("dither-c-peel")
    f = d.with_fill(x, {"IMAD": 1.0})
    s = E.setup(f, RPI, operands="floor", credit="floor")
    t = E.setup(f, RPI, operands="floor", credit="floor", table=E.table())
    assert E.setup(d, RPI, operands="floor", credit="floor").machine_eps(2) > 0.04
    assert s.baseline.binding == (row,) and E.threshold(s) == pytest.approx(q0, abs=1e-4)
    zero = [s.machine_eps(q) + t.machine_eps(q) < 1e-12 for q in (1.22, 2, 3.47, 4)]
    assert zero == [x == 1.0] * 4 and (t.floor() < 1e-12) == (x == 1.0)


def test_a_filler_mix():
    """The filler's class mix decides honest's row: one LOP3 a word binds row A (ALU at half rate), one IMAD JI3s_0, and
    two IMADs row I."""
    assert E.lane_fill(1.0, {"IMAD": 1.0}).mix == {"IMAD": 1.0} and E.lane_fill(1.0).mix == {"ALU": 1.0, "IMAD": 1.0}
    d = E.design("dither-c-peel")
    bind = lambda x, mix: E.setup(d.with_fill(x, mix), RPI, operands="floor", credit="floor").baseline.binding
    assert bind(1.0, {"ALU": 1.0}) == ("A",) and bind(1.0, {"IMAD": 1.0}) == ("JI3s_0",)
    assert bind(2.0, {"IMAD": 1.0}) == ("I",) and bind(2.0, {"IMAD": 2.0, "ALU": 1.0}) == ("JI3s_0",)


def test_cli_fill_mix(capsys):
    assert main(["eps", "--vector", "VF3r-16", "dither-c-peel", "--credit", "floor", "--operands", "floor", "--q",
                 "1.22", "--fill", "1", "--fill-mix", "IMAD=1", "--lookups", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["terms"]["machine"] == 0 and d["fillers"][0]["mix"] == {"IMAD": 1.0}
    assert d["baseline"]["binding"] == ["JI3s_0"] and d["honest_form"] == "IMMA16832"
    for bad in (["--fill-mix", "IMAD=1"], ["--fill", "1", "--fill-mix", "IMAD"], ["--fill", "1", "--fill-mix", "IMAD=0"]):
        with pytest.raises(SystemExit, match="--fill-mix"):
            main(["eps", "--vector", "VF3r-16", *bad])
