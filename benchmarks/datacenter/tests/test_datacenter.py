"""The datacenter API, layer by layer: a Program built from registered Definitions, a declared datacenter, a placement and
its crossings, the public profile, a run, the audit's claims and the account; each dishonest run refused for its own
reason."""
from __future__ import annotations

import struct
from dataclasses import replace

import pytest
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.gemm import GEMM_SM120_E4M3, GemmCoordinateE4m3
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value as Word

from datacenter import (FLEET_CLOSED, SALT_CUSTODY, Assumption, Block, Cadence, Carried, Challenge, Computed, Credit,
                        Datacenter, Deviation, Device, Entered, Fleet, Held, Interval, Ledger, Link, Maintenance, On,
                        Plan, Port, Record, Refused, Store, Wipe, account, audit, check_records, crossings,
                        double_credit, evaluate, place, register, run, traffic, values, work)
from datacenter.binding import hops
from datacenter.logical import gate_work, gates
from datacenter.models import DOT, LINEAR, MAT, VEC, WEIGHTS, X, program, two_stage, words
from datacenter.observe import digest as frames_digest

EPOCH = Interval(0, 16)
PROGRAM = two_stage()
REG = register(PROGRAM, WEIGHTS)
INPUTS = {"x": X}
STAGE = {"mac:bf16": 1024, "cuda:op": 32}           # one linear: 32 × 32 multiply-adds, 32 roundings to BF16
HBM, RESIDUAL = 6.4e10, 8e6


def node(n: str) -> Block:
    gpu = Device(f"{n}.gpu", "gpu", rate={"mac:bf16": Assumption(f"{n}.gpu.bf16", 1e6, "MAC/s"),
                                          "cuda:op": Assumption(f"{n}.gpu.ops", 1e5, "op/s")},
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


def wipes(*devices: str, at: int = 0) -> list[Wipe]:
    return [Wipe(d, HBM - RESIDUAL, Interval(at, at + 2)) for d in devices]


def staged(dc: Datacenter, **kw):
    """The two stages on a and b: x enters a from the world, h crosses from a to b, y leaves b for the world."""
    return place(PROGRAM, dc, "a.gpu", registration=REG, deliver="world", x="world", h="a.gpu", y="b.gpu", **kw)


@pytest.fixture
def setup():
    dc = datacenter()
    return dc, staged(dc)


def _float(w: int) -> float:
    return struct.unpack(">f", (w << 16).to_bytes(4, "big"))[0]


def test_a_program_of_registered_definitions_has_census_values_work_and_its_evaluation():
    """The circuit is a `verity.ml.circuits` Program: its root nodes are its parameters' input nodes and its named Calls,
    its values are the census's (what each node passes the Calls that read it), each Call's work is counted by kind
    from its Definition's body, and its evaluation is the BF16 tensor-core step's, which a float matmul matches to
    BF16's precision."""
    assert [(v.name, v.bits, v.reads, v.output) for v in values(PROGRAM)] == [
        ("x", 512, (3,), False), ("w1", 16384, (3,), False), ("w2", 16384, (4,), False), ("h", 512, (4,), False),
        ("y", 512, (), True)]
    assert work(PROGRAM) == {3: STAGE, 4: STAGE}
    c = PROGRAM.circuit
    assert sum(gate_work(c, g).get("mac:bf16", 0) for g in range(c.size)) == 2048
    t = evaluate(PROGRAM, INPUTS | WEIGHTS)
    y = [_float(t[g]) for g in gates(PROGRAM, values(PROGRAM)[4])]
    w1, w2, x = ([_float(w) for w in ws] for ws in (WEIGHTS["w1"], WEIGHTS["w2"], X))
    h = [sum(x[i] * w1[32 * j + i] for i in range(32)) for j in range(32)]
    ref = [sum(h[i] * w2[32 * j + i] for i in range(32)) for j in range(32)]
    assert y == pytest.approx(ref, abs=0.05)


def test_work_counts_by_kind_is_credited_at_its_weight_and_takes_time_at_its_rate():
    """Compute's review 2: PoUW credits W_ref, multiply-adds weighted by kind (sm_120's cohalf QKᵀ at 1.25), and a
    device's rate depends on the operand format. So each primitive's gates are work of its kind: the BF16 and E4M3
    tensor-core steps' products and the CUDA cores' roundings. The profile carries per-kind counts, the audit credits
    them at their weights, and the account times each kind at the block's declared rate for it. Kinds on one facet (the
    tensor cores' `mac:*`) add up; the roundings on the CUDA cores run beside them (compute's review 3), so the block's
    busy share is the busiest facet's, and the ledger credits and caps each facet."""
    row8 = Array(32, Word(8))
    qk = bind(GemmCoordinateE4m3, K=32, **GEMM_SM120_E4M3)
    attend = program("Attend", {"x": VEC, "w1": MAT, "q": row8, "k": row8}, Tuple(VEC, Word(16)),
                     lambda B, x, w1, q, k: tuple_of(B.call(LINEAR, x, w1, name="h"), B.call(qk, q, k, name="qk")),
                     registered=("w1", "k"))
    assert work(attend) == {4: STAGE, 5: {"mac:fp8": 32, "cuda:op": 1}}
    dc = datacenter()
    rates = {"mac:bf16": Assumption("a.gpu.bf16", 1e6, "MAC/s"), "mac:fp8": Assumption("a.gpu.fp8", 5e5, "MAC/s"),
             "cuda:op": Assumption("a.gpu.ops", 1e5, "op/s")}
    dc.blocks[0] = Block("a", [replace(dc.blocks[0].devices[0], rate=rates)])
    k = [0x30 + i % 8 for i in range(32)]                            # E4M3 codes 0.5 to 0.94
    p = place(attend, dc, "a.gpu", registration=register(attend, {"w1": WEIGHTS["w1"], "k": k}), deliver="world",
              x="world", q="world")
    profile = p.public(Plan(EPOCH), weights={"mac:fp8": 1.25})
    assert profile.work == {"a": {"mac:bf16": 1024, "mac:fp8": 32, "cuda:op": 33}, "b": {}}
    acc = account(audit(run(p, {"x": X, "q": [0x38] * 32}, Plan(EPOCH)), profile), profile)
    a = acc.blocks[0]
    mac = (1024 / 1e6 + 32 / 5e5) / 16e-3
    assert a.credited == 1024 + 1.25 * 32 + 33 and a.loads == pytest.approx({"mac": mac, "cuda": 33 / 1e5 / 16e-3})
    assert a.busy == pytest.approx(max(a.loads.values())) and a.busy < sum(a.loads.values())
    assert acc.sound and {x.resource for x in acc.credits} == {"compute:mac", "compute:cuda"}
    assert {"a.gpu.bf16", "a.gpu.fp8", "a.gpu.ops"} <= set(a.assumptions)


def test_a_boundary_link_without_a_certifier_breaks_coverage():
    dc = datacenter()
    dc.links.append(Link("a.gpu", "leaf", Assumption("bypass", 1e9, "bit/s")))
    assert any("passes no certifier" in p for p in dc.validate())
    assert datacenter().validate() == []


def test_where_an_input_enters_says_where_its_bits_come_from_and_place_refuses_what_isnt_there():
    """A run input's source is the device it enters on: the outside world, a device in a block (telemetry), or a trusted
    device (the verifier's coins). A registered input's is its registration, which a Program that has one is placed
    with."""
    dc = datacenter()
    assert {d for d, dev in dc.devices.items() if dev.trusted} == {"a.cert", "b.cert", "challenger"}
    with pytest.raises(ValueError, match="isn't there"):
        place(PROGRAM, dc, "a.gpu", registration=REG, w3="b.gpu")
    with pytest.raises(ValueError, match="isn't there"):
        place(PROGRAM, dc, "c.gpu", registration=REG)
    with pytest.raises(ValueError, match="placed with their registration"):
        place(PROGRAM, dc, "a.gpu")
    sources = [place(PROGRAM, dc, "a.gpu", registration=REG, x=d).source(0) for d in ("world", "b.gpu", "challenger")]
    assert sources == ["outside", "inside", "trusted"] and staged(dc).source(1) == "registered"


def test_crossings_are_derived_from_the_placement_including_inputs_and_delivery(setup):
    dc, p = setup
    assert p.paced() and p.step == {0: 0, 3: 3, 4: 6}            # x, then h after x's transfer, then y after h's
    moved = {(x.value.name, x.src, x.dst, x.at) for x in crossings(p)}
    assert moved == {("x", "world", "a.gpu", 3), ("h", "a.gpu", "b.gpu", 6), ("y", "b.gpu", "world", 7)}
    for x in crossings(p):
        assert any(dc.port(*h) for h in hops(x.path, x.src))
    assert set(traffic(p)) == {Port(c, d) for c in ("a.cert", "b.cert") for d in ("ingress", "egress")}


def test_the_profile_is_what_audit_and_account_read_of_the_placement(setup):
    dc, p = setup
    profile = p.public(Plan(EPOCH))
    assert profile.work == {"a": STAGE, "b": STAGE} and profile.credit("a") == 1056
    assert profile.occupancy[Port("b.cert", "egress")] == {7: 16}     # y's 32 words, at the world's read tick
    assert list(profile.windows) == [0, 1, 2, 3] and profile.registered == REG.root


def test_an_honest_run_is_credited_and_its_only_freedom_is_timing_advice(setup):
    dc, p = setup
    plan = Plan(EPOCH, wipes=wipes("b.gpu"))
    profile = p.public(plan)
    claims = audit(run(p, INPUTS, plan), profile)
    assert not refusals(claims)
    assert Computed("a", 1056, EPOCH) in claims and Computed("b", 1056, EPOCH) in claims
    assert Held("labels@b.gpu", HBM - RESIDUAL, Interval(0, 2)) in claims
    assert [x for x in claims if isinstance(x, Entered)] == [Entered("x", 512, "outside")]
    carried = [x for x in claims if isinstance(x, Carried)]
    assert len(carried) == len(dc.ports) * 4 and all(x.bits == 7 for x in carried)
    acc = account(claims, profile)
    a, b = acc.blocks
    assert acc.sound and a.credited == b.credited == 1056
    assert b.carried == RESIDUAL and a.carried == HBM                # b was wiped down to its residual
    assert acc.input_bits == {"x": 512} and acc.located == {"labels@b.gpu": ["b.gpu"]}


@pytest.mark.parametrize("deviate, why", [
    (lambda: Deviation(extra={(Port("a.cert", "egress"), 2): 1}), "occupancy differs"),
    (lambda: Deviation(hold={(Port("b.cert", "egress"), 7): 9}), "occupancy differs"),
    (lambda: Deviation(down={(Port("a.cert", "egress"), 0)}), "no record"),
    (lambda: Deviation(failed_wipes={"b.gpu"}), "spot check"),
    (lambda: Deviation(swapped={"w2": words("other weights", 32 * 32)}), "differ from the registration"),
])
def test_each_deviation_is_refused_for_its_own_reason(setup, deviate, why):
    dc, p = setup
    plan = Plan(EPOCH, wipes=wipes("b.gpu"))
    refused = refusals(audit(run(p, INPUTS, plan, deviate()), p.public(plan)))
    assert refused and all(why in r.why or "committed crossing" in r.why for r in refused), refused


def test_a_frame_that_is_not_a_committed_crossing_is_refused_even_at_the_declared_occupancy(setup):
    dc, p = setup
    plan = Plan(EPOCH)
    t = run(p, INPUTS, plan)
    port, y = Port("b.cert", "egress"), p.values[4]
    forged = frames_digest([(g, t.values[g] ^ 1) for g in p.gates(y)])   # the frames on the wire aren't y's values
    t.records = [replace(r, digest=forged) if (r.port, r.window) == (port, 1) else r for r in t.records]
    assert refusals(audit(t, p.public(plan))) == [Refused(On("port", "b.cert:out", Interval(4, 8)),
                                                          "a frame is not a committed crossing")]


def test_values_that_are_not_the_programs_evaluation_are_refused(setup):
    dc, p = setup
    plan = Plan(EPOCH)
    t = run(p, INPUTS, plan)
    t.inputs = t.inputs | {"x": words("another request", 32)}        # the committed values were for other inputs
    assert {r.why for r in refusals(audit(t, p.public(plan)))} >= {"the committed values are not the Program's evaluation"}


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
    dc, p = setup
    plan = Plan(EPOCH)
    acc = account(audit(run(p, INPUTS, plan, Deviation(down={(Port("a.cert", "egress"), 0)})), p.public(plan)), p.public(plan))
    assert not acc.sound and acc.refused_bits == pytest.approx(1e8 * 4e-3)


def test_a_block_out_for_maintenance_owes_no_records_but_what_it_held_is_charged(setup):
    """Maintenance: someone the auditor doesn't control has the block in hand. Its certifiers owe no records then, so the
    epoch stays sound, but everything it held may leave (charged to the covert budget), anything may come back in, and
    only a wipe after the maintenance bounds what it holds when it rejoins."""
    dc, p = setup
    off = Deviation(down={(Port("b.cert", "egress"), 2), (Port("b.cert", "ingress"), 2)})
    serviced = Maintenance("b", Interval(8, 12))                      # b's traffic is all in window 1
    for wipe, carried in ((Interval(0, 2), HBM), (Interval(12, 14), RESIDUAL)):
        plan = Plan(EPOCH, wipes=[Wipe("b.gpu", HBM - RESIDUAL, wipe)], maintenance=[serviced])
        acc = account(audit(run(p, INPUTS, plan, off), p.public(plan)), p.public(plan))
        assert acc.sound and acc.serviced == {"b": HBM} and acc.blocks[1].carried == carried
        assert acc.covert_bits == 14 * 7 + 512 + HBM
    unplanned = Plan(EPOCH)
    assert not account(audit(run(p, INPUTS, unplanned, off), p.public(unplanned)), p.public(unplanned)).sound


def noised(B, x, w1, w2, z):
    h = B.call(LINEAR, x, w1, name="h")
    B.call(DOT, h, z, name="p")                                      # the trusted device's fresh data z enters a's work
    return B.call(LINEAR, h, w2, name="y")


NOISED = program("Noised", {"x": VEC, "w1": MAT, "w2": MAT, "z": VEC}, VEC, noised, registered=("w1", "w2"))


def test_work_is_pinned_to_a_block_only_by_data_it_must_consume_there_or_a_named_premise(setup):
    """Compute's review 1: a helper holding the registered weights needs only small inputs and outputs, so crossing bytes
    alone attribute nothing. Fresh data from a trusted device that only the block receives pins the work there: a helper
    could get it only through the block's certified egress. So does the named premise that the fleet is closed."""
    dc, p = setup
    profile = p.public(Plan(EPOCH))
    claims = audit(run(p, INPUTS, Plan(EPOCH)), profile)
    a = account(claims, profile).blocks[0]
    assert a.credited > 0 and a.local_at_least == 0                 # no anchor: nothing is attributed
    closed = account(claims, profile, assume=(FLEET_CLOSED,)).blocks[0]
    assert closed.local_at_least == closed.credited and "fleet-closed" in closed.assumptions

    def attributed(tick: float):
        p2 = place(NOISED, datacenter(tick), "a.gpu", registration=register(NOISED, WEIGHTS), deliver="world",
                   x="world", z="challenger", h="a.gpu", p="a.gpu", y="b.gpu")
        profile2 = p2.public(Plan(EPOCH))
        claims2 = audit(run(p2, INPUTS | {"z": words("noise", 32)}, Plan(EPOCH)), profile2)
        assert not refusals(claims2)
        assert not any(isinstance(x, Entered) and x.input == "z" for x in claims2)   # the verifier's coins aren't charged
        return account(claims2, profile2).blocks[0]

    narrow, wide = attributed(1e-9), attributed(1e-3)               # 16 ns of egress: almost nothing out; 16 ms: plenty
    assert narrow.forced == 512 and narrow.local_at_least > 0.85 * narrow.credited and wide.local_at_least == 0


def test_a_salt_in_its_certifiers_custody_pins_the_work_that_depends_on_it():
    """Compute's third anchor (ATTRIB). Block a's certifier releases a 512-bit salt into a alone. The noised product p
    (PoUW's proof work) depends on it and never leaves, so a helper off the block can't have computed it. That holds
    while k, the bits that could carry the salt out before registration, stays at most 384: a's egress timing advice
    (4 windows × 7 bits) and nothing else here. A 512-bit telemetry input delivered out of a voids it, and so does a
    refused port-window of a's."""
    def salted(B, x, w1, w2, s, t):
        h = B.call(LINEAR, x, w1, name="h")
        B.call(DOT, h, s, name="p")
        return tuple_of(B.call(LINEAR, h, w2, name="y"), t)

    c = program("Salted", {"x": VEC, "w1": MAT, "w2": MAT, "s": VEC, "t": VEC}, Tuple(VEC, VEC), salted,
                registered=("w1", "w2"))
    reg, inputs = register(c, WEIGHTS), INPUTS | {"s": words("salt", 32), "t": words("telemetry", 32)}

    def custody(t: str = "b.gpu", deviation: Deviation | None = None):
        p = place(c, datacenter(), "a.gpu", registration=reg, deliver="world", x="world", s="a.cert", t=t, h="a.gpu",
                  p="a.gpu", y="b.gpu")
        profile = p.public(Plan(EPOCH))
        return profile, account(audit(run(p, inputs, Plan(EPOCH), deviation), profile), profile, assume=(SALT_CUSTODY,))

    profile, acc = custody()
    a = acc.blocks[0]
    assert profile.salt == {"a": 512, "b": 0} and profile.pinned["a"] == 33 and profile.salt_out["a"] == 0
    assert a.custody == 4 * 7 and a.local_at_least == 33 and "salt-custody" in a.assumptions
    assert acc.blocks[1].custody is None and acc.blocks[1].local_at_least == 0
    leaky, acc = custody(t="a.gpu")
    assert leaky.salt_out["a"] == 512 and acc.blocks[0].custody == 4 * 7 + 512 and acc.blocks[0].local_at_least == 0
    _, acc = custody(deviation=Deviation(down={(Port("a.cert", "egress"), 0)}))
    assert not acc.sound and acc.blocks[0].local_at_least == 0


def test_the_joint_ledger_credits_each_units_time_and_memory_once(setup):
    """The joint rule on real accounts. The claims credit a block its busy share of the epoch, and a store's device the
    bits read from it and the bit-ticks it held, and `capacity` says what each can give a tick. Two 50 Gbit stores held
    on one 64 Gbit GPU at once is more than it has, so that account is unsound. And the same night entered twice in a
    ledger credits the wiped GPU's memory twice over the wipe, which `joint` refuses, though each account alone is
    sound. A block's compute is the block's only as far as attribution places it there (compute's review 3): without
    an anchor it comes out of the whole datacenter's capacity, and under one it carries that anchor as its grade."""
    dc, p = setup
    stores = (Store("s1", 5e10, "a.gpu"), Store("s2", 5e10, "a.gpu"))
    crowded = staged(dc, stores=stores)
    plan = Plan(EPOCH, challenges=[Challenge("s1", 0, 8), Challenge("s2", 0, 8)])
    acc = account(audit(run(crowded, INPUTS, plan), crowded.public(plan)), crowded.public(plan))
    assert not acc.refused and not acc.sound
    assert acc.overcredited == [("a.gpu", "held", Interval(0, 8), 2 * 5e10 * 8, HBM * 8)]
    one = Plan(EPOCH, challenges=[Challenge("s1", 0, 8)])
    assert account(audit(run(crowded, INPUTS, one), crowded.public(one)), crowded.public(one)).sound
    night = Plan(EPOCH, wipes=wipes("b.gpu"))
    honest = account(audit(run(p, INPUTS, night), p.public(night)), p.public(night))
    assert honest.sound and {(x.unit, x.resource, x.grade) for x in honest.credits} == {
        ("dc0", "compute:mac", None), ("dc0", "compute:cuda", None), ("b.gpu", "held", None)}   # the prover system's
    closed = account(audit(run(p, INPUTS, night), p.public(night)), p.public(night), assume=(FLEET_CLOSED,))
    assert {(x.unit, x.grade) for x in closed.credits if x.resource.startswith("compute")} == {
        ("a", "fleet-closed"), ("b", "fleet-closed")}
    # PoUW's proved bound, when the scheme gives it, is the block's compute credit (compute's review 3, 15:12Z): one
    # number per block per window, at 1 block-tick a tick, in place of the honest routing's per-facet loads
    timed = p.public(night, ticks={"a": 2.0})
    pouw = account(audit(run(p, INPUTS, night), timed), timed, assume=(FLEET_CLOSED,))
    a = pouw.blocks[0]
    assert pouw.sound and a.busy == 2.0 / 16
    assert [(x.unit, x.resource, x.work, x.grade, x.basis) for x in pouw.credits if x.unit == "a"] == [
        ("a", "compute", 2.0, "fleet-closed", "proved")]
    assert {x.basis for x in closed.credits if x.resource.startswith("compute")} == {"honest-routing"}
    overlap = Ledger([pouw, pouw]).joint()                      # 4 block-ticks fit in 16, but the windows overlap
    assert ("a", "compute", EPOCH, 4.0, 0.0) in overlap and not any(r == "compute" and v > 0 for _, r, _, _, v in overlap)
    twice = Ledger().add(honest).add(honest)
    assert not twice.sound and [(u, r, i) for u, r, i, _, _ in twice.joint()] == [("b.gpu", "held", Interval(0, 2))]


def test_credit_from_two_audits_on_one_device_must_fit_every_interval_per_resource():
    capacity = {("gpu0", "compute"): 10.0, ("gpu0", "read"): 10.0}
    fits = [Credit("gpu0", "compute", Interval(0, 10), 60), Credit("gpu0", "compute", Interval(5, 15), 40)]
    assert double_credit(fits, capacity) == []
    double = [Credit("gpu0", "compute", Interval(0, 10), 80), Credit("gpu0", "compute", Interval(2, 8), 40)]
    assert double_credit(double, capacity) and double_credit(double, capacity)[0][2] == Interval(0, 10)
    other_resource = [Credit("gpu0", "compute", Interval(0, 10), 80), Credit("gpu0", "read", Interval(2, 8), 40)]
    assert double_credit(other_resource, capacity) == []


def monitored(B, x, w1, w2, temperature):
    h = B.call(LINEAR, x, w1, name="h")
    return tuple_of(B.call(LINEAR, h, w2, name="y"), temperature)


def test_telemetry_is_an_inside_input_delivered_out_and_charged_in_full():
    c = program("Monitored", {"x": VEC, "w1": MAT, "w2": MAT, "temperature": Word(16)}, Tuple(VEC, Word(16)), monitored,
                registered=("w1", "w2"))
    dc = datacenter()
    p = place(c, dc, "a.gpu", registration=register(c, WEIGHTS), deliver="world", x="world", h="a.gpu", y="b.gpu",
              temperature="b.gpu")
    profile = p.public(Plan(EPOCH))
    assert p.source(3) == "inside" and Port("b.cert", "egress") in profile.occupancy
    acc = account(audit(run(p, INPUTS | {"temperature": 45}, Plan(EPOCH)), profile), profile)
    assert acc.sound and acc.input_bits == {"x": 512, "temperature": 16}
    refused = refusals(audit(run(p, INPUTS | {"temperature": 45}, Plan(EPOCH)), staged(dc).public(Plan(EPOCH))))
    assert any(r.on.name == "b.cert:out" for r in refused), "telemetry the profile doesn't declare doesn't fit it"


def test_an_image_pull_is_a_registered_input_free_unless_its_bytes_differ():
    """A container image is what the program brings, like its weights: a registered input, whose bytes the registration
    fixes. The node runs it, so no gate reads it; running other bytes is refused."""
    image = words("image", 32)
    c = program("Imaged", {"x": VEC, "w1": MAT, "image": VEC}, VEC, lambda B, x, w1, image: B.call(LINEAR, x, w1),
                registered=("w1", "image"))
    p = place(c, datacenter(), "a.gpu", registration=register(c, {"w1": WEIGHTS["w1"], "image": image}), deliver="world",
              x="world", image="a.gpu")
    profile = p.public(Plan(EPOCH))
    honest = account(audit(run(p, INPUTS, Plan(EPOCH)), profile), profile)
    assert honest.sound and honest.input_bits == {"x": 512}
    swapped = refusals(audit(run(p, INPUTS, Plan(EPOCH), Deviation(swapped={"image": words("other", 32)})), profile))
    assert swapped == [Refused(On("input", "image"), "its bytes differ from the registration")]


def test_choosing_the_profile_from_a_registered_set_costs_its_log(setup):
    dc, p = setup
    plan = Plan(EPOCH, profiles=8)
    acc = account(audit(run(p, INPUTS, plan), p.public(plan)), p.public(plan))
    assert acc.advice_bits == 3 and acc.covert_bits == 3 + acc.timing_bits + 512


def test_a_program_placed_across_a_fleet_crosses_the_inter_datacenter_link():
    fleet = Fleet([datacenter().prefixed("east"), datacenter().prefixed("west")],
                  [Link("east.leaf", "west.leaf", Assumption("wan", 1e10, "bit/s"), kind="wan")])
    assert fleet.validate() == []
    p = place(PROGRAM, fleet.merged(), "west.a.gpu", registration=REG, deliver="world", x="world", h="west.a.gpu",
              y="east.b.gpu")
    h = next(x for x in crossings(p) if x.value.name == "h")
    assert [(l.a, l.b) for l in h.path] == [("west.a.gpu", "west.a.cert"), ("west.a.cert", "west.leaf"),
                                           ("east.leaf", "west.leaf"), ("east.leaf", "east.b.cert"),
                                           ("east.b.cert", "east.b.gpu")]
    profile = p.public(Plan(EPOCH))
    assert not refusals(audit(run(p, INPUTS, Plan(EPOCH)), profile))


def test_a_link_between_datacenters_is_fabric_and_coverage_is_per_block():
    """A link between datacenters may join any two fabric devices, a certifier included: each block's boundary is
    already certified. What the fleet refuses is each datacenter's own problems, a device in two of them, a link naming
    no device, and one that leaves a block without passing a certifier."""
    east, west = datacenter().prefixed("east"), datacenter().prefixed("west")
    wan = Assumption("wan", 1e9, "bit/s")
    for a, b in (("east.leaf", "west.leaf"), ("east.leaf", "west.a.cert"), ("east.challenger", "world")):
        assert Fleet([east, west], [Link(a, b, wan, kind="wan")]).validate() == [], (a, b)
    merged = Fleet([east, west], [Link("east.leaf", "west.leaf", wan, kind="wan")]).merged()
    assert [repr(p) for p in merged.ports] == [f"{s}.{b}.cert:{d}" for s in ("east", "west") for b in "ab"
                                               for d in ("in", "out")]
    assert Fleet([east, west], [Link("east.a.gpu", "west.leaf", wan)]).validate() == [
        "inter-datacenter link east.a.gpu-west.leaf passes no certifier (coverage)"]
    assert Fleet([east, west], [Link("east.leaf", "west.spine", wan)]).validate() == [
        "inter-datacenter link east.leaf-west.spine names an unknown device 'west.spine'"]
    assert Fleet([datacenter(), datacenter()]).validate() == [f"device {d!r} is in both dc0 and dc0" for d in
                                                              ("a.gpu", "b.gpu", "a.cert", "b.cert", "leaf", "challenger")]
    leaky = replace(east, links=[*east.links, Link("east.a.gpu", "east.leaf", wan)])
    assert Fleet([leaky, west]).validate() == ["east: boundary link east.a.gpu-east.leaf passes no certifier (coverage)"]


def test_a_ledger_adds_covert_bits_across_nights_and_wipes_bound_what_each_night_carries(setup):
    dc, _ = setup
    ledger = Ledger()
    for night in range(3):
        start = 16 * night                                           # successive nights on the one trusted clock
        p = staged(dc, start=start)
        plan = Plan(Interval(start, start + 16), wipes=wipes("a.gpu", "b.gpu", at=start) if night else ())
        ledger.add(account(audit(run(p, INPUTS, plan), p.public(plan)), p.public(plan)))
    assert ledger.sound and ledger.covert_bits == 3 * ledger.epochs[0].covert_bits
    first, *wiped = ledger.carried
    assert all(w == 2 * RESIDUAL for w in wiped) and first == 2 * HBM
