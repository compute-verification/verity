"""leak_bounds, the one calculator of K. `per_link` gives chain_record.leak's one-machine rows on a warden run's files,
and `datacenter` gives its 1,000-machine rows. `datacenter` is links × `per_link` for identical links, each link at
δ / links.

The KO 1 class's points reproduce at the census's reference datacenter, with an entry of 32 + KO·IDB bits and with
`ingress_scan.entry_layout`'s read bit beside them:
- a week for one link, as registered and at the anchors, and for every link of the datacenter;
- the datacenter's rates at which K reaches the weights, summed from per-link statements each at δ = 1%, whose union
  fails with at most links · δ;
- at δ = 1% for the whole datacenter, the pooled form's faulty proof blocks alone pass the weights, while the per-period
  form keeps its rates.

A window class stands in for its scan class and its T, and the calculator refuses what it can't count."""

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

CENSUS = HERE.parents[1] / "benchmarks" / "census" / "deployments.json"
#: the scan class of a window class of KO 1 at 4 lanes, a timestep every 16 buckets (`scan_state.scan_class(4, 1, 16)`)
KO1 = IS.Class(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=16, WITHIN=12, D=27, KO=1, KI=1024)
#: 8 requests a sampling block of 32 buckets of 100 ms
RATE = 2.5


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _links() -> int:
    """The census's reference datacenter's machines, one network certifier on each one's uplink."""
    return next(d for d in json.loads(CENSUS.read_text()) if d["role"] == "reference")["machines"]


def test_per_link_and_datacenter_are_chain_records_rows(tmp_path):
    import scan_zk as sz
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
        if r["machines"] == 1:
            got = {d: LB.per_link(sz.CLASS, per_s, r["period"], r["prog"], d, **at) for d in (None, *LB.DELTAS)}
        else:
            got = {d: LB.datacenter([link] * r["machines"], r["period"], r["prog"], d) for d in (None, *LB.DELTAS)}
        assert r["per_period"] == got[None].bits and r["per_period_share"] == got[None].share
        assert r["pooled"] == {f"{float(d):g}": got[d].bits for d in LB.DELTAS}
        assert r["pooled_share"] == {f"{float(d):g}": got[d].share for d in LB.DELTAS}
        assert r["per_link_window"] == LB.per_link(sz.CLASS, per_s, Fraction(32, 10), r["prog"], **at).bits
    rows = {(r["period"], r["machines"], r["prog"]): r for r in doc["rows"]}
    one, many = rows["week", 1, "anchors"], rows["week", 1000, "anchors"]
    assert many["per_period"] == pytest.approx(1000 * one["per_period"], rel=1e-12)
    assert many["pooled"]["0.01"] > 1000 * one["pooled"]["0.01"]
    assert doc["theorems"]["datacenter"] == LB.STATEMENTS["datacenter"]


def test_datacenter_is_links_times_per_link():
    n = 12
    for reading in LB.READINGS:
        for period in LB.PERIODS:
            each = LB.per_link(KO1, RATE, period, reading, T=32)
            dc = LB.datacenter([LB.link(KO1, RATE, T=32)] * n, period, reading)
            assert dc.bits == n * each.bits and dc.share == pytest.approx(n * each.share, rel=1e-15)
            assert dc.delta is None
            for delta in LB.DELTAS:
                pooled = LB.datacenter([LB.link(KO1, RATE, T=32)] * n, period, reading, delta)
                assert pooled.bits == n * LB.per_link(KO1, RATE, period, reading, delta / n, T=32).bits
                assert pooled.delta == delta and pooled.statement == LB.STATEMENTS["datacenter"]
                assert pooled.bits > n * LB.per_link(KO1, RATE, period, reading, delta, T=32).bits
    ko64 = replace(KO1, KO=64)
    mixed = [LB.link(KO1, RATE, T=32)] * 3 + [LB.link(ko64, 1, T=32)]
    assert LB.datacenter(mixed, "week", "anchors").bits == pytest.approx(
        3 * LB.per_link(KO1, RATE, "week", "anchors", T=32).bits + LB.per_link(ko64, 1, "week", "anchors", T=32).bits)


#: the KO 1 points at the anchors, by whether an entry carries its read bit: a request's entry bits, one link's week
#: (bits, share), one link's week at 0.5 requests a second, the datacenter's week, the datacenter's rates summed from
#: per-link statements at δ = 1% (a week, a month) and its per-period rates (a week, a month, a year)
POINTS = {False: (49, 7.6485e7, "0.387%", "0.0774%", "48.4", [348, 81], [646.3, 150.8, 12.4]),
          True: (50, 7.79965e7, "0.394%", "0.0789%", "49.3", [341, 80], [633.8, 147.9, 12.2])}


@pytest.fixture(params=[False, True], ids=["no read bit", "entry_layout"])
def read(request, monkeypatch):
    """An entry of 32 + KO·IDB bits, without its read bit, and `ingress_scan.entry_layout`'s, with it."""
    if not request.param:
        layout = IS.entry_layout
        monkeypatch.setattr(IS, "entry_layout", lambda cls: tuple(e for e in layout(cls) if e[0] != "read"))
    return request.param


def test_the_ko1_points_reproduce(read):
    entry, bits, share, slow, many, _, _ = POINTS[read]
    link = LB.link(KO1, RATE, T=32)
    assert link.widths == {"NS": 145_360, "NP": 208, "NV": 608} and link.b == 145_968 and link.egress_cells == 2
    week = {reading: LB.per_link(KO1, RATE, "week", reading, T=32) for reading in LB.READINGS}
    assert f"{week['registered'].bits:.6g}" == "1.26038e+09" and f"{week['registered'].share:.2%}" == "6.37%"
    assert week["anchors"].bits == pytest.approx(bits, rel=1e-5) and f"{week['anchors'].share:.3%}" == share
    assert week["anchors"].bits == pytest.approx(RATE * 604_800 * (entry + math.log2(3)))
    assert f"{LB.per_link(KO1, 0.5, 'week', 'anchors', T=32).share:.4%}" == slow
    n = _links()
    assert n == 12_500
    dc = {reading: LB.datacenter([link] * n, "week", reading) for reading in LB.READINGS}
    assert f"{dc['registered'].share:.0f}" == "797" and f"{dc['anchors'].share:.1f}" == many


def _rate(period: str, delta: Fraction | None, n: int, *, each_at_delta: bool = False) -> float:
    """The datacenter's certified requests a second at which K over ``period`` reaches the weights, at the anchors: K is
    affine in the rate. ``each_at_delta`` sums n per-link statements each at δ, which fail together with at most n·δ."""
    def k(per_s: Fraction) -> float:
        if each_at_delta:
            return n * LB.per_link(KO1, per_s, period, "anchors", delta, T=32).bits
        return LB.datacenter([LB.link(KO1, per_s, T=32)] * n, period, "anchors", delta).bits

    floor = k(0)
    return (LB.WEIGHTS_BITS - floor) / (k(Fraction(1, n)) - floor)


def test_the_datacenter_rates_and_what_they_hold_at(read):
    *_, summed, per_period = POINTS[read]
    n, one = _links(), Fraction(1, 100)
    assert [round(_rate(p, one, n, each_at_delta=True)) for p in ("week", "month")] == summed
    assert n * one == 125
    floor = LB.datacenter([LB.link(KO1, 0, T=32)] * n, "week", "anchors", one)
    assert LB.fmax(one / n) == 15 and floor.share > 1 and f"{floor.share:.3g}" == "1.38"
    assert _rate("week", one, n) < 0
    assert [round(_rate(p, None, n), 1) for p in ("week", "month", "year")] == per_period


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
