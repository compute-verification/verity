"""leak_bounds, the one calculator of K. `per_link`, `datacenter` and `world` give chain_record.leak's rows on a warden
run's files, and `datacenter` is links × `per_link` for identical links, each link at δ / links.

K's terms are `NetworkCheck.Period`'s rows form's, transcribed here from the Lean (`clog2`, `rowsTerms`, `Scan.terms`,
`faultyEntries`, `eps`) rather than from the calculator, in three parts:
- timing: a link-window's jitter term ⌈log2 D_j^T⌉ over its scan's T steps, at D_j = 2, whatever the request rate;
- the side channels: the tied audit's free inputs, given for a served audit and none for the stand-in tie, whose
  content, NR·T / EVERY slot entries of `ingress_scan.entry_bits` a link-window (E_w), is unchecked and in no K;
- the checking term: 0 with every proof block sampled, E_w a link-window when the served audit is sampled in part, and
  the pooled form's F incorrect proof blocks.
The datacenter's links are the census's reference datacenter's machines, and the world's are the adversary resource
model's compute in machines of the datacenter's GPUs; a share is of one copy of the weights, labelled with what it is;
ε is the scans' δ summed, m·δ_scan over the period's link-windows and the links.

At δ = 1% for the whole datacenter, the pooled form's incorrect proof blocks alone pass the weights. A window class
stands in for its scan class and its T, and the calculator refuses what it can't count."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import leak_bounds as LB  # noqa: E402
from verity.core.protocols.certifier import ingress_scan as IS  # noqa: E402

CENSUS = HERE.parents[1] / "benchmarks" / "census"
#: the scan class of a window class of KO 1 at 4 lanes, a timestep every 16 buckets (`scan_state.scan_class(4, 1, 16)`)
KO1 = IS.Class(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=16, WITHIN=12, D=27, KO=1, KI=1024)
#: the warden's scan class, KO 64 (`scan_zk.CLASS`)
W63 = replace(KO1, KO=64)
#: 8 requests a link-window of 32 buckets of 100 ms
RATE = 2.5
#: a served audit's free inputs a link-window
FREE = 40


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reference(name: str) -> dict:
    return next(d for d in json.loads((CENSUS / name).read_text()) if d["role"] == "reference")


def clog2(x: int) -> int:
    """`NetworkCheck.Period.clog2`: if x ≤ 1 then 0 else Nat.log2 (x - 1) + 1."""
    return 0 if x <= 1 else int(math.log2(x - 1)) + 1


def test_per_link_datacenter_and_world_are_chain_records_rows(tmp_path):
    import scan_zk as sz
    assert sz.CLASS == W63
    cr = _load("chain_record")
    run = tmp_path / "c1"
    (run / "out").mkdir(parents=True)
    (run / "out/deployment.json").write_text(json.dumps({"params": asdict(sz.PARAMS)}))
    (run / "end.json").write_text(json.dumps({"run": "c1", "verdict": {"windows": 75, "requests": 100}}))
    doc = cr.leak(run, {"draw": {"honest": 64}})
    at = {"T": 32, "tau": Fraction(1, 10)}
    per_s = Fraction(100, 75) / Fraction(32, 10)

    def terms(b):
        return b.terms._asdict() | {"K": b.bits, "share": b.share}

    assert {(r["tie"], r["sampling"]) for r in doc["rows"]} == {("stand-in", "whole"), ("served", "whole"),
                                                                ("served", "in part")}
    for r in doc["rows"]:
        whole = r["sampling"] == "whole"
        link = LB.link(sz.CLASS, per_s, **at, audit_whole=whole)
        if r["scope"] == "link":
            got = {d: LB.per_link(sz.CLASS, per_s, r["period"], r["tie"], d, **at, audit_whole=whole)
                   for d in (None, *LB.DELTAS)}
        elif r["scope"] == "world":
            got = {d: LB.world(link, r["period"], r["tie"], d) for d in (None, *LB.DELTAS)}
        else:
            got = {d: LB.datacenter({link: LB.datacenter_links()}, r["period"], r["tie"], d)
                   for d in (None, *LB.DELTAS)}
        assert r["per_period"] == terms(got[None]) and Fraction(r["eps"]) == got[None].eps
        assert r["pooled"] == {f"{float(d):g}": terms(got[d]) for d in LB.DELTAS}
    for t in doc["per_link_window"]:
        one = LB.per_link(sz.CLASS, per_s, Fraction(32, 10), t["tie"], **at, audit_whole=t["sampling"] == "whole")
        assert {k: v for k, v in t.items() if k not in ("tie", "sampling")} == terms(one)
    rows = {(r["period"], r["scope"], r["tie"], r["sampling"]): r for r in doc["rows"]}
    one, many = (rows["week", s, "served", "in part"]["per_period"] for s in ("link", "datacenter"))
    assert many["checking"] == pytest.approx(12_500 * one["checking"], rel=1e-12)
    assert many["timing"] == pytest.approx(12_500 * one["timing"], rel=1e-12)
    one, many = (rows["week", s, "served", "whole"]["pooled"]["0.01"] for s in ("link", "datacenter"))
    assert many["checking"] > 12_500 * one["checking"]
    assert doc["theorems"]["datacenter"] == LB.STATEMENTS["datacenter"]


def test_datacenter_is_links_times_per_link():
    n = 12
    for reading in LB.READINGS:
        for whole in (True, False):
            one = LB.link(KO1, RATE, T=32, audit_free=FREE, audit_whole=whole)
            for period in LB.PERIODS:
                each = LB.per_link(KO1, RATE, period, reading, T=32, audit_free=FREE, audit_whole=whole)
                dc = LB.datacenter([one] * n, period, reading)
                assert dc == LB.datacenter({one: n}, period, reading)
                for field in LB.Terms._fields:
                    mine, theirs = getattr(dc.terms, field), getattr(each.terms, field)
                    assert mine is theirs is None or mine == pytest.approx(n * theirs, rel=1e-15)
                assert dc.bits is None if reading == "stand-in" else dc.bits == pytest.approx(n * each.bits)
                assert dc.delta is None and dc.eps == n * each.eps
                for delta in LB.DELTAS:
                    pooled = LB.datacenter([one] * n, period, reading, delta)
                    split = LB.per_link(KO1, RATE, period, reading, delta / n, T=32, audit_free=FREE,
                                        audit_whole=whole)
                    assert pooled.terms.checking == pytest.approx(n * split.terms.checking, rel=1e-15)
                    assert pooled.delta == delta and pooled.statement == LB.STATEMENTS["datacenter"]
                    assert pooled.terms.checking > n * LB.per_link(KO1, RATE, period, reading, delta, T=32,
                                                                   audit_whole=whole).terms.checking
    one, ko64 = LB.link(KO1, RATE, T=32, audit_free=FREE), LB.link(W63, 1, T=32, audit_free=FREE)
    mixed = LB.datacenter([one] * 3 + [ko64], "week", "served")
    assert mixed == LB.datacenter({one: 3, ko64: 1}, "week", "served")
    assert mixed.bits == pytest.approx(3 * LB.per_link(KO1, RATE, "week", "served", T=32, audit_free=FREE).bits
                                       + LB.per_link(W63, 1, "week", "served", T=32, audit_free=FREE).bits)
    assert LB.datacenter([one, LB.link(KO1, RATE, T=32)], "week", "served").terms.side is None


def test_the_stand_in_ties_content_is_unchecked_and_in_no_bound():
    for cls, entry, e_w in ((W63, 1_136, 9_088), (replace(W63, KO=4), 112, 896)):
        link = LB.link(cls, RATE, T=32, audit_free=FREE, audit_whole=False)
        assert IS.entry_bits(cls) == entry and link.entries == cls.NR * 32 // cls.EVERY == 8
        assert link.e_w == e_w == 8 * entry < 32 * IS.prog_bits(cls)
        assert link.widths == {"NS": IS.state_bits(cls), "NE": entry, "NV": IS.view_bits(cls)}
        for period in (link.block_s, "week"):
            m = link.blocks(period)
            got = link.terms(period, "stand-in")
            assert (got.side, got.unchecked, got.checking) == (None, m * e_w, 0)
            assert got.bits is None and link.bits(period, "stand-in") is None
            assert got.timing == float(m * 32)
        bound = LB.per_link(cls, RATE, "week", "stand-in", T=32, audit_free=FREE)
        assert (bound.bits, bound.share, bound.terms.unchecked) == (None, None, 189_000 * e_w)
        assert LB.per_link(cls, RATE, "week", "stand-in", Fraction(1, 100), T=32).bits is None
    assert LB.link(W63, RATE, T=64).e_w == 2 * 9_088
    assert LB.UNCHECKED == "unchecked (stand-in tie)"


def test_the_served_reading_is_its_audits_free_inputs_and_e_w_when_sampled_in_part():
    tau = Fraction(4)
    four = dict(T=32, tau=tau)
    rate = Fraction(4) / (32 * tau)
    timing = clog2(LB.JITTER_BOUND ** 32)
    assert timing == 32
    whole = LB.link(W63, rate, **four, audit_free=0)
    part = LB.link(W63, rate, **four, audit_free=0, audit_whole=False)
    assert whole.terms(whole.block_s, "served") == LB.Terms(32, 0, 0, 0)
    assert whole.bits(whole.block_s, "served") == timing == 32
    assert part.terms(part.block_s, "served") == LB.Terms(32, 0, 9_088, 0)
    assert part.bits(part.block_s, "served") == timing + 9_088 == 9_120
    free = LB.link(W63, rate, **four, audit_free=FREE)
    assert free.bits(free.block_s, "served") == timing + FREE
    assert free.terms("week", "served").side == free.blocks("week") * FREE
    unknown = LB.link(W63, rate, **four, audit_whole=False)
    got = unknown.terms(unknown.block_s, "served")
    assert (got.side, got.checking, got.bits) == (None, 9_088, None)
    assert LB.per_link(W63, rate, "week", "served", **four).share is None


def test_the_checking_term_is_charged_in_one_place():
    n = 189_000
    for reading in LB.READINGS:
        for whole in (True, False):
            link = LB.link(W63, RATE, T=32, audit_free=FREE, audit_whole=whole)
            e_w = 9_088 if reading == "served" and not whole else 0
            assert link.checking("week", reading) == link.terms("week", reading).checking == n * e_w
            for delta in LB.DELTAS:
                f = LB.fmax(delta)
                pooled = n * e_w + f * math.log2(n * 64 + 1) + link.b * f
                assert link.checking("week", reading, delta) == pytest.approx(pooled)
                assert link.terms("week", reading, delta).checking == link.checking("week", reading, delta)
                assert link.terms("week", reading, delta)[:2] == link.terms("week", reading)[:2]
    with pytest.raises(ValueError, match="stand-in, served"):
        LB.link(W63, RATE, T=32).checking("week", "registered")


def test_timing_is_the_jitter_term_at_lean_s_ceiling_whatever_the_rate():
    assert LB.JITTER_BOUND == 2
    steps = (16, 32, 48)
    assert [LB.link(W63, RATE, T=t).jitter for t in steps] == [clog2(2 ** t) for t in steps] == list(steps)
    for rate in (0, RATE, 1_000):
        assert LB.link(W63, rate, T=32).terms("week", "served").timing == 189_000 * 32


def test_the_census_sets_the_datacenter_and_the_world():
    dc, world = _reference("deployments.json"), _reference("adversary.json")
    assert LB.datacenter_links() == dc["machines"] == 12_500
    flop_s = world["flop_per_s_per_h100"] * world["monitoring_period_s"]
    machines = world["total_flop"] / flop_s / dc["gpus_per_machine"]
    assert LB.world_links() == math.ceil(machines) == 18_550_229
    one = LB.link(W63, RATE, T=32, audit_free=FREE)
    for reading in LB.READINGS:
        for delta in (None, *LB.DELTAS):
            got = LB.world(one, "year", reading, delta)
            assert got == LB.datacenter({one: 18_550_229}, "year", reading, delta)
            assert got.statement == LB.STATEMENTS["datacenter"]
    assert LB.world(one, "week", "served").bits == pytest.approx(
        18_550_229 * LB.per_link(W63, RATE, "week", "served", T=32, audit_free=FREE).bits, rel=1e-12)


def test_a_share_is_of_one_bf16_copy_and_says_so():
    assert LB.WEIGHTS == LB.Weights(1_235_814_400 * 16, "Llama-3.2-1B in BF16")
    one = LB.link(W63, RATE, T=32, audit_free=FREE)
    for got in (LB.per_link(W63, RATE, "week", "served", T=32, audit_free=FREE),
                LB.datacenter([one] * 3, "week", "served"), LB.world(one, "week", "served", Fraction(1, 100))):
        assert got.weights == LB.WEIGHTS.label and got.share == got.bits / LB.WEIGHTS.bits
    int8 = LB.Weights(1_235_814_400 * 8, "an int8 copy")
    got = LB.per_link(W63, RATE, "week", "served", T=32, audit_free=FREE, weights=int8)
    assert got.weights == "an int8 copy" and got.share == 2 * LB.per_link(W63, RATE, "week", "served", T=32,
                                                                         audit_free=FREE).share


def test_eps_is_the_scans_delta_summed():
    one = LB.link(W63, RATE, T=32)
    assert one.scan_delta == Fraction(1, 2 ** 20)
    for period, s in LB.PERIODS.items():
        m = Fraction(s) / Fraction(32, 10)
        for reading in LB.READINGS:
            per = LB.per_link(W63, RATE, period, reading, T=32)
            assert per.eps == m / 2 ** 20 and isinstance(per.eps, Fraction)
            for delta in LB.DELTAS:
                pooled = LB.per_link(W63, RATE, period, reading, delta, T=32)
                assert pooled.delta == delta and pooled.eps == per.eps
                assert LB.datacenter([one] * 5, period, reading, delta).eps == 5 * per.eps
    assert LB.per_link(W63, RATE, "week", "served", T=32, scan_delta=Fraction(1, 2 ** 40)).eps == Fraction(
        189_000, 2 ** 40)
    assert LB.world(one, "week", "served").eps == 18_550_229 * Fraction(189_000, 2 ** 20)


def test_the_pooled_datacenters_incorrect_blocks_alone_pass_the_weights():
    n, one = LB.datacenter_links(), Fraction(1, 100)
    idle = LB.link(KO1, 0, T=32, audit_free=0)
    floor = LB.datacenter({idle: n}, "week", "served", one)
    incorrect = n * 15 * (idle.b + math.log2(189_000 * 64 + 1))
    jitter = n * 189_000 * clog2(LB.JITTER_BOUND ** 32)
    assert LB.fmax(one / n) == 15 and incorrect > LB.WEIGHTS.bits
    assert floor.terms.checking == pytest.approx(incorrect) and floor.bits == pytest.approx(incorrect + jitter)
    assert LB.datacenter({idle: n}, "week", "served").bits == jitter


def test_a_window_class_and_refusals():
    window = SimpleNamespace(scan=KO1, buckets=32)
    for reading in LB.READINGS:
        assert LB.per_link(window, RATE, "month", reading) == LB.per_link(KO1, RATE, "month", reading, T=32)
    assert LB.link(window, RATE, T=16).T == 16 and LB.link(KO1, RATE, T=32).tau == Fraction(1, 10)
    assert LB.per_link(KO1, RATE, 604_800, "served", T=32) == LB.per_link(KO1, RATE, "week", "served", T=32)
    with pytest.raises(ValueError, match="needs T"):
        LB.link(KO1, RATE)
    with pytest.raises(TypeError, match="ingress_scan.Class"):
        LB.link(SimpleNamespace(buckets=32), RATE)
    with pytest.raises(ValueError, match="rate"):
        LB.link(KO1, -1, T=32)
    with pytest.raises(ValueError, match="multiple of EVERY = 16"):
        LB.link(KO1, RATE, T=24)
    with pytest.raises(ValueError, match="scan's δ"):
        LB.link(KO1, RATE, T=32, scan_delta=0)
    for free in (-1, 1.5, "40"):
        with pytest.raises(ValueError, match="free inputs"):
            LB.link(KO1, RATE, T=32, audit_free=free)
    with pytest.raises(TypeError, match="sampled whole"):
        LB.link(KO1, RATE, T=32, audit_whole=1)
    for old in ("registered", "anchors"):
        with pytest.raises(ValueError, match="stand-in, served"):
            LB.per_link(KO1, RATE, "week", old, T=32)
    with pytest.raises(ValueError, match="week, month, year"):
        LB.per_link(KO1, RATE, "decade", "served", T=32)
    for delta in (0, 1, 125):
        with pytest.raises(ValueError, match="δ is in"):
            LB.datacenter([LB.link(KO1, RATE, T=32)] * 12_500, "week", "served", delta)
    with pytest.raises(ValueError, match="at least one link"):
        LB.datacenter([], "week", "served")
    with pytest.raises(TypeError, match="LinkLoads"):
        LB.datacenter([KO1], "week", "served")
    with pytest.raises(ValueError, match="positive integers"):
        LB.datacenter({LB.link(KO1, RATE, T=32): 0}, "week", "served")
