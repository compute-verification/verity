"""The datacenter API, layer by layer: a traced circuit, a declared datacenter, a placement and its crossings, a run, the
audit's claims and the account; each dishonest run refused for its own reason."""
from __future__ import annotations

import pytest

from datacenter import (Assumption, Block, Carried, Clock, Computed, Credit, Datacenter, Deviation, Device, Entered,
                        Erased, Fleet, Input, Interval, Link, Refused, Wipe, account, audit, crossings, feasible,
                        occupancy, place, run, trace)


def two_layer(x, w1, w2):
    h = (x * w1) ^ 0x5A
    return h * w2 + x


def node(n: str) -> Block:
    gpu = Device(f"{n}.gpu", "gpu", rate={"scalar": Assumption(f"{n}.gpu.rate", 1e6, "ops/s")},
                 memory=Assumption(f"{n}.gpu.hbm", 6.4e10, "bits"), residual=Assumption(f"{n}.gpu.residual", 8e6, "bits"))
    return Block(n, [gpu])


def datacenter() -> Datacenter:
    up = Assumption("uplink", 1e8, "bit/s", "declared")
    fabric = [Device("a.cert", "certifier", trusted=True), Device("b.cert", "certifier", trusted=True),
              Device("leaf", "switch"), Device("world", "outside")]
    links = [Link("a.gpu", "a.cert", up), Link("a.cert", "leaf", up), Link("leaf", "b.cert", up),
             Link("b.cert", "b.gpu", up), Link("world", "leaf", up)]
    return Datacenter("dc0", [node("a"), node("b")], fabric, links)


def site(n: str, world: bool = False) -> Datacenter:
    """`datacenter()` with every device named `n.<name>`, the outside world only if `world`."""
    dcx = datacenter()
    rename = lambda s: f"{n}.{s}"  # noqa: E731
    blocks = [Block(rename(b.name), [Device(rename(d.name), d.kind, d.rate, d.memory, d.residual) for d in b.devices])
              for b in dcx.blocks]
    fabric = [Device(rename(d.name), d.kind, trusted=d.trusted) for d in dcx.fabric if d.kind != "outside"]
    links = [Link(rename(l.a), rename(l.b), l.capacity) for l in dcx.links if "world" not in (l.a, l.b)]
    if world:
        fabric.append(Device("world", "outside"))
        links.append(Link("world", rename("leaf"), Assumption("isp", 1e9, "bit/s")))
    return Datacenter(n, blocks, fabric, links)


@pytest.fixture
def setup():
    c = trace(two_layer, x=Input(16), w1=Input(16, source="registered"), w2=Input(16, source="registered"))
    dc = datacenter()
    p = place(c, dc, {"*": "a.gpu", "x": "world", "w2": "b.gpu"}, deliver="world")
    return c, dc, p


INPUTS = {"x": 7, "w1": 3, "w2": 5}


def test_a_traced_circuit_evaluates_as_python_does():
    c = trace(two_layer, x=Input(16), w1=Input(16), w2=Input(16))
    assert c.evaluate(INPUTS)[c.outputs[0]] == (((7 * 3) ^ 0x5A) * 5 + 7) & 0xFFFF
    assert c.boolean_size == 2 * 6 * 16 * 16 + 16 + 5 * 16


def test_a_boundary_link_without_a_certifier_breaks_coverage():
    dc = datacenter()
    dc.links.append(Link("a.gpu", "leaf", Assumption("bypass", 1e9, "bit/s")))
    assert any("passes no certifier" in p for p in dc.validate())
    assert datacenter().validate() == []


def test_crossings_are_derived_from_the_placement_including_inputs_and_delivery(setup):
    c, dc, p = setup
    assert p.paced()
    moved = {(x.value.name, x.src, x.dst) for x in crossings(p)}
    assert moved == {("x", "world", "a.gpu"), ("w2", "b.gpu", "a.gpu"), (None, "a.gpu", "world")}
    for x in crossings(p):
        assert any(dc.certifier_of(l) for l in x.path)


def test_an_honest_run_is_credited_and_its_only_freedom_is_timing_advice(setup):
    c, dc, p = setup
    profile = occupancy(p, frame_bits=32, bucket=1)
    t = run(p, INPUTS, window=4, wipes=[Wipe("b.gpu", Interval(0, 2))])
    claims = audit(t, profile, syncs=128)
    assert not [x for x in claims if isinstance(x, Refused)]
    assert Computed(c.boolean_size) in claims and Erased("b.gpu", Interval(0, 2)) in claims
    assert Entered("x", 16, "outside") in claims and not any(isinstance(x, Entered) and x.input == "w1" for x in claims)
    assert all(x.bits == 7 for x in claims if isinstance(x, Carried))
    acc = account(claims, p, Clock(1e-3), Interval(0, 16))
    a, b = acc.blocks
    assert acc.sound and a.credited == c.boolean_size - 0 and b.credited == 0
    assert b.state == 8e6 and a.state == 6.4e10
    assert acc.input_bits == {"x": 16}


@pytest.mark.parametrize("deviation, why", [
    (Deviation(extra={(("a.gpu", "a.cert"), 2): 1}), "occupancy differs"),
    (Deviation(hold={(("a.gpu", "a.cert"), 0): 3}), "occupancy differs"),
    (Deviation(failed_wipes={"b.gpu"}), "spot check failed"),
])
def test_each_deviation_is_refused_for_its_own_reason(setup, deviation, why):
    c, dc, p = setup
    t = run(p, INPUTS, window=4, wipes=[Wipe("b.gpu", Interval(0, 2))], deviation=deviation)
    refused = [x for x in audit(t, occupancy(p, 32, 1), syncs=128) if isinstance(x, Refused)]
    assert refused and all(why in r.why for r in refused)


def test_a_frame_that_is_not_a_committed_crossing_is_refused_even_at_the_declared_occupancy(setup):
    c, dc, p = setup
    t = run(p, INPUTS, window=4)
    t.values[c.outputs[0]] ^= 1                      # the frame on the wire no longer carries the committed output
    from datacenter.observe import _digest
    t.records = [type(r)(r.link, r.window, r.counts, _digest([(c.outputs[0], t.values[c.outputs[0]])]), r.certifier)
                 if r.link == ("a.gpu", "a.cert") and r.window == 1 else r for r in t.records]
    refused = [x for x in audit(t, occupancy(p, 32, 1)) if isinstance(x, Refused)]
    assert refused and "not a committed crossing" in refused[0].why


def test_work_a_helper_could_ship_in_is_bounded_by_the_certified_ingress(setup):
    """Compute's scenario 1: a helper off the block computes the products and ships them in. What the claims credit to
    the block, less its certified ingress times the circuit's intensity, must have run on the block itself."""
    c, dc, p = setup
    claims = audit(run(p, INPUTS, window=4), occupancy(p, 32, 1))
    narrow = account(claims, p, Clock(1e-9), Interval(0, 16))       # 16 ns: almost nothing can come in
    wide = account(claims, p, Clock(1e-3), Interval(0, 16))         # 16 ms at 100 Mbit/s: plenty can
    a_narrow, a_wide = narrow.blocks[0], wide.blocks[0]
    assert a_narrow.credited == a_wide.credited > 0
    assert a_narrow.local_at_least > 0.9 * a_narrow.credited and a_wide.local_at_least == 0


def test_credit_from_two_audits_on_one_device_must_fit_every_interval():
    rate = {"gpu0": 10.0}
    fits = [Credit("gpu0", Interval(0, 10), 60), Credit("gpu0", Interval(5, 15), 40)]
    assert feasible(fits, rate) == []
    double = [Credit("gpu0", Interval(0, 10), 80), Credit("gpu0", Interval(2, 8), 40)]
    assert feasible(double, rate) and feasible(double, rate)[0][1] == Interval(0, 10)


def test_telemetry_leaves_on_a_declared_offramp_and_is_charged_in_full(setup):
    from datacenter import Emitted, Offramp
    c, dc, _ = setup
    temps = Offramp("telemetry", "b.gpu", "world", bits=64, windows=2)
    p = place(c, dc, {"*": "a.gpu", "x": "world", "w2": "b.gpu"}, deliver="world", offramps=(temps,))
    t = run(p, INPUTS, window=4)
    claims = audit(t, occupancy(p, 32, 1, window=4), syncs=128)
    assert not [x for x in claims if isinstance(x, Refused)]
    assert {x.window for x in claims if isinstance(x, Emitted)} == {0, 1}
    acc = account(claims, p, Clock(1e-3), Interval(0, 16))
    assert acc.offramp_bits == {"telemetry": 128}
    undeclared = place(c, dc, {"*": "a.gpu", "x": "world", "w2": "b.gpu"}, deliver="world")
    refused = [x for x in audit(t, occupancy(undeclared, 32, 1, window=4)) if isinstance(x, Refused)]
    assert refused, "telemetry the developer didn't declare doesn't fit the declared occupancy"


def test_a_circuit_placed_across_a_fleet_crosses_the_inter_datacenter_link():
    east, west = site("east"), site("west", world=True)
    fleet = Fleet([east, west], [Link("east.a.cert", "west.a.cert", Assumption("wan", 1e10, "bit/s"), kind="wan")])
    assert fleet.validate() == []
    c = trace(two_layer, x=Input(16), w1=Input(16, source="registered"), w2=Input(16, source="registered"))
    p = place(c, fleet.merged(), {"*": "west.a.gpu", "x": "world", "w2": "east.b.gpu"}, deliver="world")
    w2 = next(x for x in crossings(p) if x.value.name == "w2")
    assert any(l.kind == "wan" for l in w2.path)
    claims = audit(run(p, INPUTS, window=4), occupancy(p, 32, 1, window=4), syncs=128)
    assert not [x for x in claims if isinstance(x, Refused)]


def test_a_fleet_links_datacenters_only_through_certifiers():
    ok = Fleet([site("east"), site("west")], [Link("east.a.cert", "west.a.cert", Assumption("wan", 1e9, "bit/s"), kind="wan")])
    assert ok.validate() == []
    bad = Fleet([site("east"), site("west")], [Link("east.a.gpu", "west.a.cert", Assumption("wan", 1e9, "bit/s"), kind="wan")])
    assert any("not a certifier" in p for p in bad.validate())
    clash = Fleet([datacenter(), datacenter()])
    assert any("is in both" in p for p in clash.validate())
