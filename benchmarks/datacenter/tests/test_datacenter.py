"""The datacenter API, layer by layer: a traced circuit, a declared datacenter, a placement and its crossings, the public
profile, a run, the audit's claims and the account; each dishonest run refused for its own reason."""
from __future__ import annotations

from dataclasses import replace

import pytest

from datacenter import (FLEET_CLOSED, Assumption, Block, Carried, Computed, Credit, Datacenter, Deviation, Device,
                        Entered, Fleet, Cadence, Held, Input, Interval, Ledger, Link, Maintenance, On, Plan, Port, Record,
                        Refused, Wipe,
                        account, audit, check_records, crossings, digest, double_credit, place, run, trace, traffic)
from datacenter.binding import hops
from datacenter.observe import digest as frames_digest

EPOCH = Interval(0, 16)
INPUTS = {"x": 7, "w1": 3, "w2": 5}
HBM, RESIDUAL = 6.4e10, 8e6


def two_layer(x, w1, w2):
    h = (x * w1) ^ 0x5A
    return h * w2 + x


def node(n: str) -> Block:
    gpu = Device(f"{n}.gpu", "gpu", rate={"gate": Assumption(f"{n}.gpu.rate", 1e6, "gate/s")},
                 memory=Assumption(f"{n}.gpu.hbm", HBM, "bits"), residual=Assumption(f"{n}.gpu.residual", RESIDUAL, "bits"))
    return Block(n, [gpu])


def datacenter(tick: float = 1e-3) -> Datacenter:
    up = Assumption("uplink", 1e8, "bit/s", "declared")
    fabric = [Device("a.cert", "certifier"), Device("b.cert", "certifier"), Device("leaf", "switch"),
              Device("challenger", "challenger"), Device("world", "outside")]
    links = [Link("a.gpu", "a.cert", up), Link("a.cert", "leaf", up), Link("leaf", "b.cert", up),
             Link("b.cert", "b.gpu", up), Link("challenger", "leaf", up), Link("world", "leaf", up)]
    cadence = Cadence(frame_bits=32, bucket_ticks=1, buckets=4, jitter_classes=128,
                      tick=Assumption("tick", tick, "s", "declared"))
    return Datacenter("dc0", [node("a"), node("b")], fabric, links, cadence)


def refusals(claims) -> list[Refused]:
    return [x for x in claims if isinstance(x, Refused)]


def wipes(*devices: str) -> list[Wipe]:
    return [Wipe(d, HBM - RESIDUAL, Interval(0, 2)) for d in devices]


@pytest.fixture
def setup():
    c = trace(two_layer, x=Input(16), w1=Input(16, "registered"), w2=Input(16, "registered"))
    dc = datacenter()
    p = place(c, dc, "a.gpu", deliver="world", x="world", w2="b.gpu")
    return c, dc, p


def test_a_traced_circuit_evaluates_as_python_does():
    c = trace(two_layer, x=Input(16), w1=Input(16), w2=Input(16))
    assert c.evaluate(INPUTS)[c.outputs[0]] == (((7 * 3) ^ 0x5A) * 5 + 7) & 0xFFFF
    assert c.work == 2 * 6 * 16 * 16 + 16 + 5 * 16


def test_a_circuit_of_another_formats_primitives_counts_work_in_its_unit():
    """An adapted circuit (verity's word Definitions, say) brings each primitive's semantics and its work in the format's
    unit, and the profile and the account credit that unit at the devices' declared rates in it."""
    from datacenter.logical import Circuit, Gate
    c = Circuit([Gate("input", (), 16, input=Input(16)), Gate("input", (), 16, input=Input(16, "registered")),
                 Gate("Bf16Mul", (0, 1), 16, fn=lambda a, b: a * b + 1, size=1)], (2,), {0: "x", 1: "w"}, unit="op")
    assert c.evaluate({"x": 3, "w": 4})[2] == 13 and c.work == 1
    dc = datacenter()
    dc.blocks[0] = Block("a", [replace(dc.blocks[0].devices[0], rate={"op": Assumption("a.gpu.ops", 10, "op/s")})])
    p = place(c, dc, "a.gpu", deliver="world", x="world")
    profile = p.public(Plan(EPOCH))
    a = account(audit(run(p, {"x": 3, "w": 4}, Plan(EPOCH)), profile), profile).blocks[0]
    assert profile.unit == "op" and a.credited == 1 and a.capacity == 10 * 16e-3


def test_a_boundary_link_without_a_certifier_breaks_coverage():
    dc = datacenter()
    dc.links.append(Link("a.gpu", "leaf", Assumption("bypass", 1e9, "bit/s")))
    assert any("passes no certifier" in p for p in dc.validate())
    assert datacenter().validate() == []


def test_trusted_devices_are_the_auditors_and_place_refuses_what_isnt_there():
    dc = datacenter()
    assert {d for d, dev in dc.devices.items() if dev.trusted} == {"a.cert", "b.cert", "challenger"}
    c = trace(two_layer, x=Input(16), w1=Input(16), w2=Input(16))
    with pytest.raises(ValueError, match="isn't there"):
        place(c, dc, "a.gpu", w3="b.gpu")
    with pytest.raises(ValueError, match="isn't there"):
        place(c, dc, "c.gpu")
    z = trace(lambda x, z: x ^ z, x=Input(16), z=Input(16, "trusted"))
    with pytest.raises(ValueError, match="trusted device"):
        place(z, dc, "a.gpu")


def test_crossings_are_derived_from_the_placement_including_inputs_and_delivery(setup):
    c, dc, p = setup
    assert p.paced()
    cells = [(p.leaf[i], p.cell[i], p.tick(i)) for i in range(len(c.gates))]
    assert len(set(cells)) == len(cells)                     # one value a cell a tick (Placed.cell_inj)
    moved = {(x.value.name, x.src, x.dst) for x in crossings(p)}
    assert moved == {("x", "world", "a.gpu"), ("w2", "b.gpu", "a.gpu"), (None, "a.gpu", "world")}
    for x in crossings(p):
        assert any(dc.port(*h) for h in hops(x.path, x.src))
    assert set(traffic(p)) == {Port("a.cert", "ingress"), Port("a.cert", "egress"), Port("b.cert", "egress")}


def test_the_profile_is_what_audit_and_account_read_of_the_placement(setup):
    c, dc, p = setup
    profile = p.public(Plan(EPOCH))
    assert profile.work == {"a": c.work, "b": 0}
    assert profile.occupancy[Port("a.cert", "egress")] == {7: 1}     # the output, one frame, at the world's read tick
    assert list(profile.windows) == [0, 1, 2, 3]


def test_an_honest_run_is_credited_and_its_only_freedom_is_timing_advice(setup):
    c, dc, p = setup
    plan = Plan(EPOCH, wipes=wipes("b.gpu"))
    profile = p.public(plan)
    claims = audit(run(p, INPUTS, plan), profile)
    assert not refusals(claims)
    assert Computed("a", c.work, EPOCH) in claims and not any(isinstance(x, Computed) and x.part == "b" for x in claims)
    assert Held("labels@b.gpu", HBM - RESIDUAL, Interval(0, 2)) in claims
    assert Entered("x", 16, "outside") in claims and not any(isinstance(x, Entered) and x.input == "w1" for x in claims)
    carried = [x for x in claims if isinstance(x, Carried)]
    assert len(carried) == len(dc.ports) * 4 and all(x.bits == 7 for x in carried)
    acc = account(claims, profile)
    a, b = acc.blocks
    assert acc.sound and a.credited == c.work and b.credited == 0
    assert b.carried == RESIDUAL and a.carried == HBM                # b was wiped down to its residual
    assert acc.input_bits == {"x": 16} and acc.located == {"labels@b.gpu": ["b.gpu"]}


@pytest.mark.parametrize("deviate, why", [
    (lambda: Deviation(extra={(Port("a.cert", "egress"), 2): 1}), "occupancy differs"),
    (lambda: Deviation(hold={(Port("a.cert", "egress"), 7): 9}), "occupancy differs"),
    (lambda: Deviation(down={(Port("a.cert", "egress"), 0)}), "no record"),
    (lambda: Deviation(failed_wipes={"b.gpu"}), "spot check"),
])
def test_each_deviation_is_refused_for_its_own_reason(setup, deviate, why):
    c, dc, p = setup
    plan = Plan(EPOCH, wipes=wipes("b.gpu"))
    refused = refusals(audit(run(p, INPUTS, plan, deviate()), p.public(plan)))
    assert refused and all(why in r.why or "committed crossing" in r.why for r in refused), refused


def test_a_frame_that_is_not_a_committed_crossing_is_refused_even_at_the_declared_occupancy(setup):
    c, dc, p = setup
    plan = Plan(EPOCH)
    t = run(p, INPUTS, plan)
    out, port = c.outputs[0], Port("a.cert", "egress")
    forged = frames_digest([(out, t.values[out] ^ 1)])           # the frame on the wire isn't the committed output
    t.records = [replace(r, digest=forged) if (r.port, r.window) == (port, 1) else r for r in t.records]
    assert refusals(audit(t, p.public(plan))) == [Refused(On("port", "a.cert:out", Interval(4, 8)),
                                                          "a frame is not a committed crossing")]


def test_records_are_checked_against_the_declared_ones_whatever_the_certifier_signs():
    """The mock datacenter's records carry the real certifier's digest, not `observe.digest`: `check_records` compares
    whatever the certifier signs with what an honest run would have it sign."""
    cadence, port = datacenter().cadence, Port("a.cert", "egress")
    declared = {(port, 0): Record(port, 0, (1, 0, 0, 0), b"one frame"), (port, 1): Record(port, 1, (0, 0, 0, 0), b"none")}
    assert [type(x) for x in check_records(list(declared.values()), declared, cadence)] == [Carried, Carried]
    late = [Record(port, 0, (0, 1, 0, 0), b"one frame"), Record(port, 5, (0, 0, 0, 0), b"another epoch's")]
    assert [x.why for x in check_records(late, declared, cadence)] == [
        "occupancy differs from the declared profile", "no record: the certifier was down or its record was withheld"]


def test_a_refused_window_is_charged_at_its_links_full_capacity(setup):
    """Fail-closed: a port-window the audit refuses could have carried anything its link can, so the covert budget
    charges it at line rate rather than dropping it."""
    c, dc, p = setup
    plan = Plan(EPOCH)
    acc = account(audit(run(p, INPUTS, plan, Deviation(down={(Port("a.cert", "egress"), 0)})), p.public(plan)), p.public(plan))
    assert not acc.sound and acc.refused_bits == pytest.approx(1e8 * 4e-3)


def test_a_block_out_for_maintenance_owes_no_records_but_what_it_held_is_charged(setup):
    """Maintenance: someone the auditor doesn't control has the block in hand. Its certifiers owe no records then, so the
    epoch stays sound, but everything it held may leave (charged to the covert budget), anything may come back in, and
    only a wipe after the maintenance bounds what it holds when it rejoins."""
    c, dc, p = setup
    off = Deviation(down={(Port("b.cert", "egress"), 2), (Port("b.cert", "ingress"), 2)})
    serviced = Maintenance("b", Interval(8, 12))                      # b's traffic is all in window 1
    for wipe, carried in ((Interval(0, 2), HBM), (Interval(12, 14), RESIDUAL)):
        plan = Plan(EPOCH, wipes=[Wipe("b.gpu", HBM - RESIDUAL, wipe)], maintenance=[serviced])
        acc = account(audit(run(p, INPUTS, plan, off), p.public(plan)), p.public(plan))
        assert acc.sound and acc.serviced == {"b": HBM} and acc.blocks[1].carried == carried
        assert acc.covert_bits == 14 * 7 + 16 + HBM
    unplanned = Plan(EPOCH)
    assert not account(audit(run(p, INPUTS, unplanned, off), p.public(unplanned)), p.public(unplanned)).sound


def test_work_is_pinned_to_a_block_only_by_data_it_must_consume_there_or_a_named_premise(setup):
    """Compute's review 1: a helper holding the registered weights needs only small inputs and outputs, so crossing bytes
    alone attribute nothing. Fresh data from a trusted device that only the block receives pins the work there: a helper
    could get it only through the block's certified egress. So does the named premise that the fleet is closed."""
    c, dc, p = setup
    profile = p.public(Plan(EPOCH))
    claims = audit(run(p, INPUTS, Plan(EPOCH)), profile)
    a = account(claims, profile).blocks[0]
    assert a.credited > 0 and a.local_at_least == 0                 # no anchor: nothing is attributed
    closed = account(claims, profile, assume=(FLEET_CLOSED,)).blocks[0]
    assert closed.local_at_least == closed.credited and "fleet-closed" in closed.assumptions

    def noised(x, w1, w2, z):
        h = ((x * w1) ^ z).named("h")                              # the challenger's noise z enters stage h
        return (h * w2 + x).named("y")

    c2 = trace(noised, x=Input(16), w1=Input(16, "registered"), w2=Input(16, "registered"), z=Input(16, "trusted"))

    def attributed(tick: float):
        p2 = place(c2, datacenter(tick), "a.gpu", deliver="world", x="world", z="challenger", w2="b.gpu")
        profile2 = p2.public(Plan(EPOCH))
        claims2 = audit(run(p2, INPUTS | {"z": 9}, Plan(EPOCH)), profile2)
        assert not any(isinstance(x, Entered) and x.input == "z" for x in claims2)   # the verifier's coins aren't charged
        return account(claims2, profile2).blocks[0]

    narrow, wide = attributed(1e-9), attributed(1e-3)               # 16 ns of egress: almost nothing out; 16 ms: plenty
    assert narrow.forced == 16 and narrow.local_at_least > 0.85 * narrow.credited and wide.local_at_least == 0


def test_credit_from_two_audits_on_one_device_must_fit_every_interval_per_resource():
    capacity = {("gpu0", "compute"): 10.0, ("gpu0", "read"): 10.0}
    fits = [Credit("gpu0", "compute", Interval(0, 10), 60), Credit("gpu0", "compute", Interval(5, 15), 40)]
    assert double_credit(fits, capacity) == []
    double = [Credit("gpu0", "compute", Interval(0, 10), 80), Credit("gpu0", "compute", Interval(2, 8), 40)]
    assert double_credit(double, capacity) and double_credit(double, capacity)[0][2] == Interval(0, 10)
    other_resource = [Credit("gpu0", "compute", Interval(0, 10), 80), Credit("gpu0", "read", Interval(2, 8), 40)]
    assert double_credit(other_resource, capacity) == []


def test_telemetry_is_an_inside_input_delivered_out_and_charged_in_full():
    def monitored(x, w1, w2, temperature):
        return two_layer(x, w1, w2), temperature

    c = trace(monitored, x=Input(16), w1=Input(16, "registered"), w2=Input(16, "registered"), temperature=Input(16, "inside"))
    dc = datacenter()
    p = place(c, dc, "a.gpu", deliver="world", x="world", w2="b.gpu", temperature="b.gpu")
    profile = p.public(Plan(EPOCH))
    assert Port("b.cert", "egress") in profile.occupancy
    acc = account(audit(run(p, INPUTS | {"temperature": 45}, Plan(EPOCH)), profile), profile)
    assert acc.sound and acc.input_bits == {"x": 16, "temperature": 16}
    silent = place(trace(two_layer, x=Input(16), w1=Input(16, "registered"), w2=Input(16, "registered")), dc, "a.gpu",
                   deliver="world", x="world", w2="b.gpu")
    refused = refusals(audit(run(p, INPUTS | {"temperature": 45}, Plan(EPOCH)), silent.public(Plan(EPOCH))))
    assert any(r.on.name == "b.cert:out" for r in refused), "telemetry the profile doesn't declare doesn't fit it"


def test_an_image_pull_is_a_registered_input_free_unless_its_bytes_differ():
    image = 0xC0FFEE
    c = trace(lambda x, image: (x * image) ^ 1, x=Input(32), image=Input(32, "registered", digest(image, 32)))
    p = place(c, datacenter(), "a.gpu", deliver="world", x="world", image="world")
    profile = p.public(Plan(EPOCH))
    honest = account(audit(run(p, {"x": 7, "image": image}, Plan(EPOCH)), profile), profile)
    assert honest.sound and honest.input_bits == {"x": 32}
    swapped = refusals(audit(run(p, {"x": 7, "image": 0xBADC0DE}, Plan(EPOCH)), profile))
    assert swapped == [Refused(On("input", "image"), "its bytes differ from the registered digest")]


def test_choosing_the_profile_from_a_registered_set_costs_its_log(setup):
    c, dc, p = setup
    plan = Plan(EPOCH, profiles=8)
    acc = account(audit(run(p, INPUTS, plan), p.public(plan)), p.public(plan))
    assert acc.advice_bits == 3 and acc.covert_bits == 3 + acc.timing_bits + 16


def test_a_circuit_placed_across_a_fleet_crosses_the_inter_datacenter_link():
    fleet = Fleet([datacenter().prefixed("east"), datacenter().prefixed("west")],
                  [Link("east.a.cert", "west.a.cert", Assumption("wan", 1e10, "bit/s"), kind="wan")])
    assert fleet.validate() == []
    c = trace(two_layer, x=Input(16), w1=Input(16, "registered"), w2=Input(16, "registered"))
    p = place(c, fleet.merged(), "west.a.gpu", deliver="world", x="world", w2="east.b.gpu")
    w2 = next(x for x in crossings(p) if x.value.name == "w2")
    assert any(l.kind == "wan" for l in w2.path)
    profile = p.public(Plan(EPOCH))
    assert not refusals(audit(run(p, INPUTS, Plan(EPOCH)), profile))


def test_a_fleet_links_datacenters_only_through_certifiers():
    east, west = datacenter().prefixed("east"), datacenter().prefixed("west")
    wan = Assumption("wan", 1e9, "bit/s")
    assert Fleet([east, west], [Link("east.a.cert", "west.a.cert", wan, kind="wan")]).validate() == []
    bad = Fleet([east, west], [Link("east.a.gpu", "west.a.cert", wan, kind="wan")])
    assert any("not a certifier" in p for p in bad.validate())
    assert any("is in both" in p for p in Fleet([datacenter(), datacenter()]).validate())


def test_a_ledger_adds_covert_bits_across_nights_and_wipes_bound_what_each_night_carries(setup):
    c, dc, p = setup
    ledger = Ledger()
    for night in range(3):
        plan = Plan(EPOCH, wipes=wipes("a.gpu", "b.gpu") if night else ())
        ledger.add(account(audit(run(p, INPUTS, plan), p.public(plan)), p.public(plan)))
    assert ledger.sound and ledger.covert_bits == 3 * ledger.epochs[0].covert_bits
    first, *wiped = ledger.carried
    assert all(w == 2 * RESIDUAL for w in wiped) and first == 2 * HBM
