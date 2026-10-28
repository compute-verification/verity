"""leak_bounds, the one calculator of K. `per_link`, `datacenter` and `world` give chain_record.leak's rows on a warden
run's files, and `datacenter` is links × `per_link` for identical links, each link at δ / links.

K's terms are `NetworkCheck.Period`'s rows form's, transcribed here from the Lean (`clog2`, `capsTerm`, `rowsTerms`,
`eps`) rather than from the calculator:
- the registered free inputs are the tied audit's slot entries, NR·T / EVERY entries of `ingress_scan.entry_bits`;
- a request's caps are charged twice, as schedules and as fail cells, each ⌈log2(cap + 1)⌉;
- the datacenter's links are the census's reference datacenter's machines, and the world's are the adversary resource
  model's compute in machines of the datacenter's GPUs;
- a share is of one copy of the weights, labelled with what it is;
- ε is the scans' δ summed, m·δ_scan over the period's sampling blocks and the links.

At δ = 1% for the whole datacenter, the pooled form's faulty proof blocks alone pass the weights. A window class stands
in for its scan class and its T, and the calculator refuses what it can't count."""

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
#: 8 requests a sampling block of 32 buckets of 100 ms
RATE = 2.5


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
    at = {"T": 32, "tau": Fraction(1, 10), "egress_cells": 65}
    per_s = Fraction(100, 75) / Fraction(32, 10)
    link = LB.link(sz.CLASS, per_s, **at)
    for r in doc["rows"]:
        if r["scope"] == "link":
            got = {d: LB.per_link(sz.CLASS, per_s, r["period"], r["prog"], d, **at) for d in (None, *LB.DELTAS)}
        elif r["scope"] == "world":
            got = {d: LB.world(link, r["period"], r["prog"], d) for d in (None, *LB.DELTAS)}
        else:
            got = {d: LB.datacenter({link: LB.datacenter_links()}, r["period"], r["prog"], d)
                   for d in (None, *LB.DELTAS)}
        assert r["per_period"] == got[None].bits and r["per_period_share"] == got[None].share
        assert Fraction(r["eps"]) == got[None].eps
        assert r["pooled"] == {f"{float(d):g}": got[d].bits for d in LB.DELTAS}
        assert r["pooled_share"] == {f"{float(d):g}": got[d].share for d in LB.DELTAS}
        assert r["per_link_window"] == LB.per_link(sz.CLASS, per_s, Fraction(32, 10), r["prog"], **at).bits
    rows = {(r["period"], r["scope"], r["prog"]): r for r in doc["rows"]}
    one, many = rows["week", "link", "anchors"], rows["week", "datacenter", "anchors"]
    assert many["per_period"] == pytest.approx(12_500 * one["per_period"], rel=1e-12)
    assert many["pooled"]["0.01"] > 12_500 * one["pooled"]["0.01"]
    assert doc["theorems"]["datacenter"] == LB.STATEMENTS["datacenter"]


def test_datacenter_is_links_times_per_link():
    n = 12
    one = LB.link(KO1, RATE, T=32)
    for reading in LB.READINGS:
        for period in LB.PERIODS:
            each = LB.per_link(KO1, RATE, period, reading, T=32)
            dc = LB.datacenter([one] * n, period, reading)
            assert dc == LB.datacenter({one: n}, period, reading)
            assert dc.bits == n * each.bits and dc.share == pytest.approx(n * each.share, rel=1e-15)
            assert dc.delta is None and dc.eps == n * each.eps
            for delta in LB.DELTAS:
                pooled = LB.datacenter([one] * n, period, reading, delta)
                assert pooled.bits == n * LB.per_link(KO1, RATE, period, reading, delta / n, T=32).bits
                assert pooled.delta == delta and pooled.statement == LB.STATEMENTS["datacenter"]
                assert pooled.bits > n * LB.per_link(KO1, RATE, period, reading, delta, T=32).bits
    ko64 = LB.link(W63, 1, T=32)
    mixed = LB.datacenter([one] * 3 + [ko64], "week", "anchors")
    assert mixed == LB.datacenter({one: 3, ko64: 1}, "week", "anchors")
    assert mixed.bits == pytest.approx(3 * LB.per_link(KO1, RATE, "week", "anchors", T=32).bits
                                       + LB.per_link(W63, 1, "week", "anchors", T=32).bits)


def test_the_registered_free_inputs_are_the_tied_audits_slot_entries():
    for cls, entry, block in ((W63, 1_136, 9_088), (replace(W63, KO=4), 112, 896)):
        link = LB.link(cls, RATE, T=32)
        assert IS.entry_bits(cls) == entry and link.entries == cls.NR * 32 // cls.EVERY == 8
        assert link.free("registered") == {"block": block, "request": 0} == {"block": 8 * entry, "request": 0}
        assert link.free("registered")["block"] < 32 * IS.prog_bits(cls)
        assert link.free("anchors") == {"block": 0, "request": 33 + cls.KO * cls.IDB}
        assert link.widths == {"NS": IS.state_bits(cls), "NE": entry, "NV": IS.view_bits(cls)}
    assert LB.link(W63, RATE, T=64).free("registered")["block"] == 2 * 9_088


def test_the_caps_are_charged_twice_at_lean_s_ceilings():
    for cells in range(1, W63.KO + 2):
        link = LB.link(W63, RATE, T=32, egress_cells=cells)
        assert link.cell == clog2(cells + 1) and link.cap == 2 * clog2(cells + 1)
    assert [LB.link(W63, RATE, T=32, egress_cells=c).cap for c in (1, 2, 3, 63, 64, 65)] == [2, 4, 4, 12, 14, 14]
    tau = Fraction(4)
    four = LB.link(W63, Fraction(4) / (32 * tau), T=32, tau=tau, egress_cells=65)
    schedules = cells = clog2(65 + 1) * 4
    assert four.bits(four.block_s, "registered") == schedules + cells + 9_088 == 9_144


def test_the_census_sets_the_datacenter_and_the_world():
    dc, world = _reference("deployments.json"), _reference("adversary.json")
    assert LB.datacenter_links() == dc["machines"] == 12_500
    machines = world["total_flop"] / world["flop_per_s_per_h100"] / world["monitoring_period_s"] / dc["gpus_per_machine"]
    assert LB.world_links() == math.ceil(machines) == 18_550_229
    one = LB.link(W63, RATE, T=32)
    for delta in (None, *LB.DELTAS):
        got = LB.world(one, "year", "registered", delta)
        assert got == LB.datacenter({one: 18_550_229}, "year", "registered", delta)
        assert got.statement == LB.STATEMENTS["datacenter"]
    assert LB.world(one, "week", "anchors").bits == pytest.approx(
        18_550_229 * LB.per_link(W63, RATE, "week", "anchors", T=32).bits, rel=1e-12)


def test_a_share_is_of_one_bf16_copy_and_says_so():
    assert LB.WEIGHTS == LB.Weights(1_235_814_400 * 16, "Llama-3.2-1B in BF16")
    one = LB.link(W63, RATE, T=32)
    for got in (LB.per_link(W63, RATE, "week", "registered", T=32), LB.datacenter([one] * 3, "week", "registered"),
                LB.world(one, "week", "registered", Fraction(1, 100))):
        assert got.weights == LB.WEIGHTS.label and got.share == got.bits / LB.WEIGHTS.bits
    int8 = LB.Weights(1_235_814_400 * 8, "an int8 copy")
    got = LB.per_link(W63, RATE, "week", "registered", T=32, weights=int8)
    assert got.weights == "an int8 copy" and got.share == 2 * LB.per_link(W63, RATE, "week", "registered", T=32).share


def test_eps_is_the_scans_delta_summed():
    one = LB.link(W63, RATE, T=32)
    assert one.scan_delta == Fraction(1, 2 ** 20)
    for period, s in LB.PERIODS.items():
        m = Fraction(s) / Fraction(32, 10)
        per = LB.per_link(W63, RATE, period, "registered", T=32)
        assert per.eps == m / 2 ** 20 and isinstance(per.eps, Fraction)
        for delta in LB.DELTAS:
            pooled = LB.per_link(W63, RATE, period, "registered", delta, T=32)
            assert pooled.delta == delta and pooled.eps == per.eps
            assert LB.datacenter([one] * 5, period, "registered", delta).eps == 5 * per.eps
    assert LB.per_link(W63, RATE, "week", "registered", T=32, scan_delta=Fraction(1, 2 ** 40)).eps == Fraction(
        189_000, 2 ** 40)
    assert LB.world(one, "week", "registered").eps == 18_550_229 * Fraction(189_000, 2 ** 20)


def test_the_pooled_datacenters_faulty_blocks_alone_pass_the_weights():
    n, one = LB.datacenter_links(), Fraction(1, 100)
    floor = LB.datacenter({LB.link(KO1, 0, T=32): n}, "week", "anchors", one)
    assert LB.fmax(one / n) == 15 and floor.share > 1
    assert floor.bits == pytest.approx(n * 15 * (LB.link(KO1, 0, T=32).b + math.log2(189_000 * 64 + 1)))
    assert LB.datacenter({LB.link(KO1, 0, T=32): n}, "week", "anchors").bits == 0


def test_a_window_class_and_refusals():
    window = SimpleNamespace(scan=KO1, buckets=32)
    for reading in LB.READINGS:
        assert LB.per_link(window, RATE, "month", reading) == LB.per_link(KO1, RATE, "month", reading, T=32)
    assert LB.link(window, RATE, T=16).T == 16 and LB.link(KO1, RATE, T=32).tau == Fraction(1, 10)
    assert LB.per_link(KO1, RATE, 604_800, "anchors", T=32) == LB.per_link(KO1, RATE, "week", "anchors", T=32)
    with pytest.raises(ValueError, match="needs T"):
        LB.link(KO1, RATE)
    with pytest.raises(TypeError, match="ingress_scan.Class"):
        LB.link(SimpleNamespace(buckets=32), RATE)
    with pytest.raises(ValueError, match="KO \\+ 1 = 2"):
        LB.link(KO1, RATE, T=32, egress_cells=3)
    with pytest.raises(ValueError, match="rate"):
        LB.link(KO1, -1, T=32)
    with pytest.raises(ValueError, match="multiple of EVERY = 16"):
        LB.link(KO1, RATE, T=24)
    with pytest.raises(ValueError, match="scan's δ"):
        LB.link(KO1, RATE, T=32, scan_delta=0)
    with pytest.raises(ValueError, match="registered, anchors"):
        LB.per_link(KO1, RATE, "week", "everywhere", T=32)
    with pytest.raises(ValueError, match="week, month, year"):
        LB.per_link(KO1, RATE, "decade", "anchors", T=32)
    for delta in (0, 1, 125):
        with pytest.raises(ValueError, match="δ is in"):
            LB.datacenter([LB.link(KO1, RATE, T=32)] * 12_500, "week", "anchors", delta)
    with pytest.raises(ValueError, match="at least one link"):
        LB.datacenter([], "week", "anchors")
    with pytest.raises(TypeError, match="Links"):
        LB.datacenter([KO1], "week", "anchors")
    with pytest.raises(ValueError, match="positive integers"):
        LB.datacenter({LB.link(KO1, RATE, T=32): 0}, "week", "anchors")
