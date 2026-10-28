"""The datacenter API, layer by layer: a Program built from registered Definitions, a declared datacenter, a placement and
its crossings, the public profile, a run, the audit's claims and the account; each dishonest run refused for its own
reason."""
from __future__ import annotations

import struct
from dataclasses import replace

import pytest
from verity.core.protocols.sampled_proofs.one_stage import registration as one_stage
from verity.ml.circuits import partition_object
from verity.ml.circuits.census import census
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean.gemm import GEMM_SM120_E4M3, GemmCoordinateE4m3
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple

from datacenter import (FLEET_CLOSED, SALT_CUSTODY, SERVED_PARTITION, Accounts, Assumption, Cadence, Carried, Challenge,
                        Computed, Credit, Datacenter, Deviation, DeviceEntry, Entered, Fleet, Held, Interval, LinkEntry,
                        Maintenance, On, Plan, Port, PortRecord, Refused, StoreEntry, Wipe, Zone, account, ands, audit,
                        check_records, crossings, double_credit, evaluate, names, occupancy, place, read, register, run,
                        traffic, values, work)
from datacenter.binding import hops
from datacenter.logical import census_of, committed
from datacenter.models import DOT, LINEAR, MAT, VEC, WEIGHTS, X, program, two_stage, word, words
from datacenter.observe import digest as frames_digest

EPOCH = Interval(0, 16)
PROGRAM = two_stage()
REG = register(PROGRAM, WEIGHTS)
INPUTS = {"x": X}
STAGE = {"mac:bf16": 1024, "cuda:gate": 5728}       # one linear: 32 × 32 multiply-adds, 32 roundings to BF16 of 179 gates
ANDS = 528640 + 2112                                # its ANDs on bits: the BF16 step's and the roundings'
HBM, RESIDUAL = 6.4e10, 8e6


def node(n: str) -> Zone:
    gpu = DeviceEntry(f"{n}.gpu", "gpu", rate={"mac:bf16": Assumption(f"{n}.gpu.bf16", 1e6, "MAC/s"),
                                               "cuda:gate": Assumption(f"{n}.gpu.gates", 1e7, "gate/s")},
                      memory=Assumption(f"{n}.gpu.hbm", HBM, "bits"),
                      residual=Assumption(f"{n}.gpu.residual", RESIDUAL, "bits"),
                      write=Assumption(f"{n}.gpu.write", 1e14, "bit/s"))
    return Zone(n, [gpu])


def datacenter(tick: float = 1e-3) -> Datacenter:
    up = Assumption("uplink", 1e8, "bit/s", "declared")
    fabric = [DeviceEntry("a.cert", "certifier"), DeviceEntry("b.cert", "certifier"), DeviceEntry("leaf", "switch"),
              DeviceEntry("challenger", "challenger"), DeviceEntry("world", "outside")]
    links = [LinkEntry("a.gpu", "a.cert", up), LinkEntry("a.cert", "leaf", up), LinkEntry("leaf", "b.cert", up),
             LinkEntry("b.cert", "b.gpu", up), LinkEntry("challenger", "leaf", up), LinkEntry("world", "leaf", up)]
    cadence = Cadence(frame_bits=32, bucket_ticks=1, buckets=4, jitter_classes=128,
                      tick=Assumption("tick", tick, "s", "declared"))
    return Datacenter("dc0", [node("a"), node("b")], fabric, links, cadence)


def refusals(claims) -> list[Refused]:
    return [x for x in claims if isinstance(x, Refused)]


def wipes(*devices: str, at: int = 0) -> list[Wipe]:
    return [Wipe(d, HBM - RESIDUAL, Interval(at, at + 2)) for d in devices]


def staged(dc: Datacenter, registration=REG, **kw):
    """The two stages on a and b: x enters a from the world, h crosses from a to b, y leaves b for the world."""
    return place(PROGRAM, dc, "a.gpu", registration=registration, deliver="world", x="world", h="a.gpu", y="b.gpu", **kw)


@pytest.fixture
def setup():
    dc = datacenter()
    return dc, staged(dc)


def _float(w: int) -> float:
    return struct.unpack(">f", (w << 16).to_bytes(4, "big"))[0]


def test_a_program_of_registered_definitions_has_census_values_work_and_its_evaluation():
    """The circuit is a `verity.ml.circuits` Program: its root nodes are its parameters' input nodes and its named Calls,
    its values are the census's (what each node passes the Calls that read it), each Call's work is counted by kind
    from its Definition's body and its ANDs from the census's Boolean work, and its evaluation is the BF16 tensor-core
    step's, which a float matmul matches to BF16's precision."""
    assert [(v.name, v.bits, v.reads, v.output) for v in values(PROGRAM)] == [
        ("x", 512, (3,), False), ("w1", 16384, (3,), False), ("w2", 16384, (4,), False), ("h", 512, (4,), False),
        ("y", 512, (), True)]
    assert work(PROGRAM) == {3: STAGE, 4: STAGE} and ands(PROGRAM) == {3: ANDS, 4: ANDS}
    assert len(committed(PROGRAM)) == 32 + 2 * 1024 + 2 * 32          # a run commits its census values, not a Call's wires
    doc = census_of(PROGRAM)                                         # the multiply-adds are the census's served MACs
    nodes = doc["nodes"]
    assert {k: doc["classes"][doc["callees"][nodes["callee"][k]]]["served_macs"] * nodes["n"][k] for k in (3, 4)} == {
        k: w["mac:bf16"] for k, w in work(PROGRAM).items()}
    t = evaluate(PROGRAM, INPUTS | WEIGHTS)
    y = [_float(w) for w in read(PROGRAM, t, values(PROGRAM)[4])]
    w1, w2, x = ([_float(w) for w in ws] for ws in (WEIGHTS["w1"], WEIGHTS["w2"], X))
    h = [sum(x[i] * w1[32 * j + i] for i in range(32)) for j in range(32)]
    ref = [sum(h[i] * w2[32 * j + i] for i in range(32)) for j in range(32)]
    assert y == pytest.approx(ref, abs=0.05)


@pytest.mark.slow
def test_each_blocks_computed_claim_rests_on_its_proof_units_under_the_registered_partition(setup):
    """The sampled proofs draw proof blocks from the Program's partition, which one-stage registers with the Program (the
    record names the query's digest, the partition's and the population). A block's proof blocks are that partition's
    whose gates run there, read with the verifier's copy of the query and held to the record (proofs, review 4): under
    Q_word v1 at X = 16 each linear is 32 units, one a coordinate, so each stage's block rests its `Computed` claim on its
    own 32 of the 64. Without a record they are the served query's, and the account says it assumed it. Slow: core checks
    Q_word's cut on the Boolean linear's 1.6 million gates, once for the record and once for the verifier (about 45 s),
    and no other test reads a registered partition."""
    dc, p = setup
    query = partition_object.query()
    rec = one_stage.record(PROGRAM, query, {"law": "subset", "k": 8}, "hm96-sha512", [], {"kind": "stand-in"})
    profile = p.public(Plan(EPOCH), record=rec, query=query)
    assert profile.units == {"a": 32, "b": 32} and profile.partition == "registered" and rec["population"] == 64
    claims = audit(run(p, INPUTS, Plan(EPOCH)), profile)
    assert Computed("a", ANDS, EPOCH, units=32) in claims and Computed("b", ANDS, EPOCH, units=32) in claims
    assumed = p.public(Plan(EPOCH))
    assert assumed.units == profile.units and assumed.partition == "assumed"
    assert [SERVED_PARTITION.name in account(claims, x).blocks[0].assumptions for x in (assumed, profile)] == [True, False]
    with pytest.raises(ValueError, match="copy of its query"):
        p.public(Plan(EPOCH), record=rec, query=partition_object.call_query(8))
    with pytest.raises(ValueError, match="isn't this Program's"):
        p.public(Plan(EPOCH), record=rec | {"population": 63}, query=query)


@pytest.mark.slow
def test_pouws_compiled_program_is_placed_with_its_salt_entering_at_the_blocks_certifier():
    """PoUW's compiler rewrites a served s8 matmul into sm_120's H1 tile unit, whose one coin is `pouw.salt`. The compiled
    Program is placed like any other, read at the census's grain without expanding a gate: the salt enters at the block's
    certifier, so it is trusted data the block consumes and the salt compute's custody counts, and the unit's work comes
    out by kind, its SHA-512 seed, noise and receipt fold as Boolean gates and its products, `z`'s and the lanes' segment
    dots (r·c·(2K + 32m)), as int8 multiply-adds, of which census/v1 counts none served (Lean's `Accounting.Gates.roleOf`
    puts the tile's served role on its receipt's words, which take no MACs). Its salt's cone, gate by gate, is past the explicit limit, so nothing is pinned by it. The block's compute is PoUW's
    proved bound for its census (benchmarks/pouw/proved_ticks.py), basis "proved", which credits its tile's `z` products,
    r·c·K, and not the segment dots'. Slow: compiling builds the unit's 15
    million gates (about 40 s), and no other test has a compiled Program."""
    import importlib.util
    import sys
    from pathlib import Path

    from verity.ml.protocols.pouw.circuit import compile as pouw
    from verity.ml.protocols.pouw.circuit import h1tile
    from datacenter.models import s8_grid

    spec = importlib.util.spec_from_file_location("pouw_proved_ticks", Path(__file__).parents[2] / "pouw/proved_ticks.py")
    PT = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PT)
    compiled = pouw.compile(s8_grid(), h1tile.SCHEME).program
    assert [n for n, b in zip(names(compiled), compiled.bindings) if b == "registration"] == ["w"]
    dc = datacenter()
    p = place(compiled, dc, "a.gpu", registration=register(compiled, {"w": [1] * 16 * 64 * 8}),
              deliver="world", x="world", **{"pouw.salt": "a.cert", "pouw.positions": "world", "pouw.roots": "world"})
    assert [p.source(k) for k in range(5)] == ["outside", "registered", "trusted", "outside", "outside"]
    work = p.public(Plan(EPOCH)).work
    assert work == {"a": {"cuda:gate": 9455977, "mac:int8": 4 * 16 * (64 + 64 + 32 * 2)}, "b": {}}
    assert PT.tiles(compiled) == {PT.Tile(4, 16, 64, 64): 1} and PT.placed(work["a"], PT.tiles(compiled)) == {
        "mac:linears": 4 * 16 * 64}
    ticks = PT.proved_block_ticks(PT.placed(work["a"], PT.tiles(compiled)), PT.sheet(), PT.SHEET_EPS, dc.cadence.seconds(1),
                                  gpus=sum(d.kind == "gpu" for d in dc.block("a").devices))
    profile = p.public(Plan(EPOCH), ticks={"a": ticks})
    assert census_of(compiled)["totals"]["served_macs"] == 0
    assert profile.salt["a"] == profile.forced["a"] == 512 and profile.pinned["a"] == 0
    assert {x.value.name for x in crossings(p) if x.dst == "world"} == {"pouw[0][0:2048]", "pouw[0][2048:2304]"}
    # the claim an accepted window gives, in place of `audit`'s re-execution of the unit's gates
    acc = account([Computed("a", profile.ands["a"], EPOCH)], profile, assume=(FLEET_CLOSED,))
    assert profile.ands["a"] == sum(ands(compiled).values()) > 0 and profile.uncounted is None
    assert [c for c in acc.credits if c.resource.startswith("compute")] == [
        Credit("a", "compute", EPOCH, ticks, "fleet-closed", "proved")]


@pytest.mark.slow
def test_pouws_one_stream_program_is_placed_and_credits_z_at_its_k():
    """The same s8 grid compiled to the one stream (`h1tile.SCHEME_V3`): its unit is `PouwH1Tile_v3`, whose z is the
    lanes' folded words recombined, so its block's int8 products are the segment dots' alone, r·c·(K + 32m), and the
    proved figure credits z's r·c·K as on v1. Slow: compiling builds the unit's gates (about 40 s); it catches a v3 block
    `tiles` doesn't find, which `placed` refuses."""
    import importlib.util
    import sys
    from pathlib import Path

    from verity.ml.protocols.pouw.circuit import compile as pouw
    from verity.ml.protocols.pouw.circuit import h1tile
    from datacenter.models import s8_grid

    spec = importlib.util.spec_from_file_location("pouw_proved_ticks", Path(__file__).parents[2] / "pouw/proved_ticks.py")
    PT = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PT)
    assert h1tile.SCHEME_V3 == PT.SCHEME_V3
    compiled = pouw.compile(s8_grid(), h1tile.SCHEME_V3).program
    dc = datacenter()
    p = place(compiled, dc, "a.gpu", registration=register(compiled, {"w": [1] * 16 * 64 * 8}),
              deliver="world", x="world", **{"pouw.salt": "a.cert", "pouw.positions": "world", "pouw.roots": "world"})
    work = p.public(Plan(EPOCH)).work
    assert work["b"] == {} and work["a"]["mac:int8"] == 4 * 16 * (64 + 32 * 2) and set(work["a"]) == {"cuda:gate", "mac:int8"}
    assert PT.tiles(compiled) == {PT.Tile(4, 16, 64, 64, 3): 1}
    assert PT.placed(work["a"], PT.tiles(compiled)) == {"mac:linears": 4 * 16 * 64}
    ticks = PT.proved_block_ticks(PT.placed(work["a"], PT.tiles(compiled)), PT.sheet(), PT.SHEET_EPS, dc.cadence.seconds(1),
                                  gpus=sum(d.kind == "gpu" for d in dc.block("a").devices))
    profile = p.public(Plan(EPOCH), ticks={"a": ticks})
    acc = account([Computed("a", profile.ands["a"], EPOCH)], profile, assume=(FLEET_CLOSED,))
    assert [c for c in acc.credits if c.resource.startswith("compute")] == [
        Credit("a", "compute", EPOCH, ticks, "fleet-closed", "proved")]
    with pytest.raises(ValueError, match="aren't its H1 tiles'"):
        PT.placed(work["a"], {PT.Tile(4, 16, 64, 64): 1})


def test_work_is_credited_in_ands_and_takes_time_by_kind_at_its_rate():
    """Compute credit is the declared circuit's AND count under one pinned lowering, with the
    multiply-adds reported by format beside it. A device's rate depends on the operand format, so each primitive's
    gates are work of its kind, timed at the block's rate for it. Kinds on one facet (the tensor cores' `mac:*`) add
    up; the roundings on the CUDA cores run beside them, so the block's busy share is the busiest
    facet's, and the ledger credits and caps each facet."""
    row8 = Array(32, word(8))
    qk = bind(GemmCoordinateE4m3, K=32, **GEMM_SM120_E4M3)
    attend = program("Attend", {"x": VEC, "w1": MAT, "q": row8, "k": row8}, Tuple(VEC, word(16)),
                     lambda B, x, w1, q, k: tuple_of(B.call(LINEAR, x, w1, name="h"), B.call(qk, q, k, name="qk")),
                     registered=("w1", "k"))
    assert work(attend) == {4: STAGE, 5: {"mac:fp8": 32, "cuda:gate": 179}}
    dc = datacenter()
    rates = {"mac:bf16": Assumption("a.gpu.bf16", 1e6, "MAC/s"), "mac:fp8": Assumption("a.gpu.fp8", 5e5, "MAC/s"),
             "cuda:gate": Assumption("a.gpu.gates", 1e7, "gate/s")}
    dc.blocks[0] = Zone("a", [replace(dc.blocks[0].devices[0], rate=rates)])
    k = [0x30 + i % 8 for i in range(32)]                            # E4M3 codes 0.5 to 0.94
    p = place(attend, dc, "a.gpu", registration=register(attend, {"w1": WEIGHTS["w1"], "k": k}), deliver="world",
              x="world", q="world")
    profile = p.public(Plan(EPOCH))
    assert profile.work == {"a": {"mac:bf16": 1024, "mac:fp8": 32, "cuda:gate": 5728 + 179}, "b": {}}
    assert profile.ands == {"a": ANDS + 9194, "b": 0}                # the E4M3 coordinate's 9194 ANDs on bits
    acc = account(audit(run(p, {"x": X, "q": [0x38] * 32}, Plan(EPOCH)), profile), profile)
    a = acc.blocks[0]
    mac = (1024 / 1e6 + 32 / 5e5) / 16e-3
    assert a.credited == ANDS + 9194 and a.loads == pytest.approx({"mac": mac, "cuda": 5907 / 1e7 / 16e-3})
    assert a.macs == {"mac:bf16": 1024, "mac:fp8": 32} and "MACs by format (not credited)" in repr(acc)
    assert a.busy == pytest.approx(max(a.loads.values())) and a.busy < sum(a.loads.values())
    assert acc.sound and {x.resource for x in acc.credits} == {"compute:mac", "compute:cuda"}
    assert {"a.gpu.bf16", "a.gpu.fp8", "a.gpu.gates"} <= set(a.assumptions)


def test_a_boundary_link_without_a_certifier_breaks_coverage():
    dc = datacenter()
    dc.links.append(LinkEntry("a.gpu", "leaf", Assumption("bypass", 1e9, "bit/s")))
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


def test_a_registered_value_has_a_copy_in_every_block_that_reads_it_unless_held_in_one():
    """Memory accounting, review 4: registration puts a copy of a registered value in every block that reads it, as
    serving holds its weights and the remote audit registers each copy, so a weight both stages read never crosses. Held
    in one copy instead (named in `place`), it is moved to the other block before it is read, and that crossing is
    bandwidth like any other: 512 more frames on a's uplink, which the profile declares and the run shows."""
    def shared(B, x, w):
        h = B.call(LINEAR, x, w, name="h")
        return B.call(LINEAR, h, w, name="y")

    c = program("Shared", {"x": VEC, "w": MAT}, VEC, shared, registered=("w",))
    reg, dc, plan = register(c, {"w": WEIGHTS["w1"]}), datacenter(), Plan(EPOCH)
    kw = dict(registration=reg, deliver="world", x="world", h="a.gpu", y="b.gpu")
    copies, held = place(c, dc, "a.gpu", **kw), place(c, dc, "a.gpu", **kw, w="a.gpu")
    assert {x.value.name for x in crossings(copies)} == {"x", "h", "y"}
    assert {(x.value.name, x.src, x.dst) for x in crossings(held)} >= {("w", "a.gpu", "b.gpu")}
    out = Port("a.cert", "egress")
    assert [sum(q.public(plan).occupancy[out].values()) for q in (copies, held)] == [16, 16 + 512]
    assert not refusals(audit(run(held, INPUTS, plan), held.public(plan)))


def test_the_profile_is_what_audit_and_account_read_of_the_placement(setup):
    dc, p = setup
    profile = p.public(Plan(EPOCH))
    assert profile.work == {"a": STAGE, "b": STAGE} and profile.ands == {"a": ANDS, "b": ANDS}
    assert profile.occupancy[Port("b.cert", "egress")] == {7: 16}     # y's 32 words, at the world's read tick
    assert list(profile.windows) == [0, 1, 2, 3] and profile.registered == REG.root
    assert profile.census == census(PROGRAM, digest=program_digest(PROGRAM), boolean=True)["digest"]   # names the lowering


def test_an_honest_run_is_credited_and_its_only_freedom_is_timing_advice(setup):
    dc, p = setup
    plan = Plan(EPOCH, wipes=wipes("b.gpu"))
    profile = p.public(plan)
    claims = audit(run(p, INPUTS, plan), profile)
    assert not refusals(claims)
    assert Computed("a", ANDS, EPOCH, units=32) in claims and Computed("b", ANDS, EPOCH, units=32) in claims
    assert Held("labels@b.gpu", HBM - RESIDUAL, Interval(0, 2)) in claims
    assert [x for x in claims if isinstance(x, Entered)] == [Entered("x", 512, "outside")]
    carried = [x for x in claims if isinstance(x, Carried)]
    assert len(carried) == len(dc.ports) * 4 and all(x.bits == 7 for x in carried)
    acc = account(claims, profile)
    a, b = acc.blocks
    assert acc.sound and a.credited == b.credited == ANDS
    assert b.carried == RESIDUAL and a.carried == HBM                # b was wiped down to its residual
    assert acc.input_bits == {"x": 512} and acc.located == {"labels@b.gpu": ["b.gpu"]}


def certified(tick: float = 1e-3) -> Datacenter:
    """The test datacenter with the outside world behind a certifier of its own: every link off a block ends at a
    trusted device."""
    dc = datacenter(tick)
    up = dc.link("world", "leaf").capacity
    dc.fabric.append(DeviceEntry("world.cert", "certifier"))
    dc.links = [l for l in dc.links if "world" not in (l.a, l.b)] + [LinkEntry("world", "world.cert", up),
                                                                     LinkEntry("world.cert", "leaf", up)]
    return dc


def test_the_links_where_two_machines_the_auditor_doesnt_control_meet_are_uncovered():
    """A link is covered when a trusted device ends it, a certifier or the challenger, and a link inside a block is the
    block's own. The test datacenter's world reaches the leaf switch directly, so that link is uncovered, and so is a
    bypass from a block to the leaf; with the world behind a certifier of its own, every link is covered."""
    dc = datacenter()
    assert [(l.a, l.b) for l in dc.uncovered_links()] == [("world", "leaf")]
    dc.links.append(LinkEntry("a.gpu", "leaf", Assumption("bypass", 1e9, "bit/s")))
    assert [(l.a, l.b) for l in dc.uncovered_links()] == [("world", "leaf"), ("a.gpu", "leaf")]
    assert certified().validate() == [] and certified().uncovered_links() == []


def test_the_export_is_reported_by_port_and_charged_only_when_certifiers_cover_every_link(setup):
    """The covert budget is only what goes beyond the committed computation, so committed
    values leaving a block were free, and something off the block can read them. The export, λ_out, is the committed
    bits that cross a certified boundary out of a block, to another block or outside, by egress port and within the
    epoch, read from the declared crossings: h out of a and y out of b, 512 bits each, and not x, which comes in.
    It counts as novel information read by its receivers once network certifiers cover every link,
    and not before. The test datacenter's world-to-leaf link is uncovered, so the export is reported and not
    applicable; with the world behind a certifier, the same run's export is charged. The covert budget is the same in
    both."""
    dc, p = setup
    out = {Port("a.cert", "egress"): 512.0, Port("b.cert", "egress"): 512.0}
    assert p.public(Plan(EPOCH)).exported == out
    assert p.public(Plan(Interval(0, 7))).exported == {Port("a.cert", "egress"): 512.0}   # y is read at tick 7
    accounts = []
    for q in (p, staged(certified())):
        profile = q.public(Plan(EPOCH))
        accounts.append(account(audit(run(q, INPUTS, Plan(EPOCH)), profile), profile))
    leaf, covered = accounts
    assert leaf.sound and leaf.exported == out and leaf.export_bits == 1024 and not leaf.export_charged
    assert leaf.uncovered == ["world-leaf"] and "not applicable while no certifier covers world-leaf" in repr(leaf)
    assert covered.sound and covered.exported == out and covered.export_charged and covered.uncovered == []
    assert "charged as novel information read by their receivers" in repr(covered)
    assert leaf.covert_bits == covered.covert_bits == 4 * 4 * 7 + 512


def test_with_every_pair_link_sealed_the_export_is_charged_at_the_facility_boundary_only(setup):
    """With the certifiers' pair links sealed (`Datacenter(sealed=True)`, off by default), h's crossing from a
    to b is ciphertext between a.cert and b.cert, which only they open, so it leaves the export; y, which b delivers to
    the world, stays, at b.cert's egress. The rest of the account is the unsealed one's. A fleet's pair links are
    sealed when every site's are, and a site renamed into a fleet keeps its own."""
    dc, p = setup
    assert not dc.sealed and p.public(Plan(EPOCH)).exported == {Port("a.cert", "egress"): 512.0,
                                                                Port("b.cert", "egress"): 512.0}
    accounts = {}
    for sealed in (False, True):
        q = staged(replace(certified(), sealed=sealed))
        profile = q.public(Plan(EPOCH))
        accounts[sealed] = account(audit(run(q, INPUTS, Plan(EPOCH)), profile), profile)
    plain, sealed = accounts[False], accounts[True]
    assert sealed.sound and sealed.export_charged and sealed.exported == {Port("b.cert", "egress"): 512.0}
    assert sealed.export_bits == 512 and plain.export_bits == 1024
    assert sealed.covert_bits == plain.covert_bits and sealed.credits == plain.credits
    assert replace(dc, sealed=True).prefixed("s").sealed
    one, other = dc.prefixed("s1"), replace(dc, sealed=True).prefixed("s2")
    assert not Fleet([one, other]).merged().sealed
    assert Fleet([replace(one, sealed=True), other]).merged().sealed


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
    declared = {(port, 0): PortRecord(port, 0, (1, 0, 0, 0), b"one frame"),
                (port, 1): PortRecord(port, 1, (0, 0, 0, 0), b"none")}
    assert [type(x) for x in check_records(list(declared.values()), declared, cadence)] == [Carried, Carried]
    late = [PortRecord(port, 0, (0, 1, 0, 0), b"one frame"), PortRecord(port, 5, (0, 0, 0, 0), b"another epoch's")]
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
    """A helper holding the registered weights needs only small inputs and outputs, so crossing bytes
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
    """The salt anchor. Zone a's certifier releases a 512-bit salt into a alone. The noised product p
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
    dot = 14907                                                      # p's ANDs the salt reaches: its 16586 less h's own
    assert profile.salt == {"a": 512, "b": 0} and profile.pinned["a"] == dot and profile.salt_out["a"] == 0
    assert a.custody == 4 * 7 and a.local_at_least == dot and "salt-custody" in a.assumptions
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
    sound. A block's compute is the block's only as far as attribution places it there: without
    an anchor it comes out of the whole datacenter's capacity, and under one it carries that anchor as its grade."""
    dc, p = setup
    stores = (StoreEntry("s1", 5e10, "a.gpu"), StoreEntry("s2", 5e10, "a.gpu"))
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
    # PoUW's proved bound, when the scheme gives it, is the block's compute credit: one
    # number per block per window, at 1 block-tick a tick, in place of the honest routing's per-facet loads
    timed = p.public(night, ticks={"a": 2.0})
    pouw = account(audit(run(p, INPUTS, night), timed), timed, assume=(FLEET_CLOSED,))
    a = pouw.blocks[0]
    assert pouw.sound and a.busy == 2.0 / 16
    assert [(x.unit, x.resource, x.work, x.grade, x.basis) for x in pouw.credits if x.unit == "a"] == [
        ("a", "compute", 2.0, "fleet-closed", "proved")]
    assert {x.basis for x in closed.credits if x.resource.startswith("compute")} == {"honest-routing"}
    overlap = Accounts([pouw, pouw]).joint()                      # 4 block-ticks fit in 16, but the windows overlap
    assert ("a", "compute", EPOCH, 4.0, 0.0) in overlap and not any(r == "compute" and v > 0 for _, r, _, _, v in overlap)
    twice = Accounts().add(honest).add(honest)
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
    c = program("Monitored", {"x": VEC, "w1": MAT, "w2": MAT, "temperature": word(16)}, Tuple(VEC, word(16)), monitored,
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


def test_the_occupancy_profile_is_one_of_those_the_registration_commits_and_costs_the_log_of_their_number(setup):
    """Choosing the occupancy profile is advice worth log2 of how many the developer may choose
    from, so that number is fixed before the run and public. The registration commits the profiles' digests, each
    counted from its epoch's first bucket, and the plan reads their number from it: here the night's schedule may start
    at any of 8 ticks, 3 bits, in any epoch. A declared occupancy that isn't one of them is refused, and so is a plan
    with another number; a plan takes no count of its own. With none registered the one profile is the placement's own,
    as before."""
    dc, p = setup
    eight = register(PROGRAM, WEIGHTS, profiles=[occupancy(staged(dc, start=s)) for s in range(8)])
    assert len(eight.profiles) == 8 and register(PROGRAM, WEIGHTS, eight.salts).root != eight.root

    def night(q, plan):
        return account(audit(run(q, INPUTS, plan), q.public(plan)), q.public(plan))

    chosen = staged(dc, eight, start=3)
    acc = night(chosen, Plan(EPOCH, profiles=eight))
    assert acc.sound and acc.advice_bits == 3 and acc.covert_bits == 3 + acc.timing_bits + 512
    assert night(staged(dc, eight, start=16 + 3), Plan(Interval(16, 32), profiles=eight)).sound
    unlisted = night(staged(dc, eight, start=8), Plan(EPOCH, profiles=eight))
    assert unlisted.refused == [Refused(On("profile", "occupancy"),
                                        "the declared occupancy is not one of the registered profiles")]
    assert [r.why for r in night(chosen, Plan(EPOCH)).refused] == ["the plan's number of profiles is not the registration's"]
    with pytest.raises(TypeError, match="from the registration that commits them"):
        Plan(EPOCH, profiles=8)
    own = night(p, Plan(EPOCH, profiles=REG))
    assert Plan(EPOCH, profiles=REG).profiles == 1 and own.sound and own.advice_bits == 0


def test_a_program_placed_across_a_fleet_crosses_the_inter_datacenter_link():
    fleet = Fleet([datacenter().prefixed("east"), datacenter().prefixed("west")],
                  [LinkEntry("east.leaf", "west.leaf", Assumption("wan", 1e10, "bit/s"), kind="wan")])
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
        assert Fleet([east, west], [LinkEntry(a, b, wan, kind="wan")]).validate() == [], (a, b)
    merged = Fleet([east, west], [LinkEntry("east.leaf", "west.leaf", wan, kind="wan")]).merged()
    assert [repr(p) for p in merged.ports] == [f"{s}.{b}.cert:{d}" for s in ("east", "west") for b in "ab"
                                               for d in ("in", "out")]
    assert Fleet([east, west], [LinkEntry("east.a.gpu", "west.leaf", wan)]).validate() == [
        "inter-datacenter link east.a.gpu-west.leaf passes no certifier (coverage)"]
    assert Fleet([east, west], [LinkEntry("east.leaf", "west.spine", wan)]).validate() == [
        "inter-datacenter link east.leaf-west.spine names an unknown device 'west.spine'"]
    assert Fleet([datacenter(), datacenter()]).validate() == [f"device {d!r} is in both dc0 and dc0" for d in
                                                              ("a.gpu", "b.gpu", "a.cert", "b.cert", "leaf", "challenger")]
    leaky = replace(east, links=[*east.links, LinkEntry("east.a.gpu", "east.leaf", wan)])
    assert Fleet([leaky, west]).validate() == ["east: boundary link east.a.gpu-east.leaf passes no certifier (coverage)"]


def test_a_ledger_adds_covert_bits_across_nights_and_wipes_bound_what_each_night_carries(setup):
    dc, _ = setup
    ledger = Accounts()
    for night in range(3):
        start = 16 * night                                           # successive nights on the one trusted clock
        p = staged(dc, start=start)
        plan = Plan(Interval(start, start + 16), wipes=wipes("a.gpu", "b.gpu", at=start) if night else ())
        ledger.add(account(audit(run(p, INPUTS, plan), p.public(plan)), p.public(plan)))
    assert ledger.sound and ledger.covert_bits == 3 * ledger.epochs[0].covert_bits
    first, *wiped = ledger.carried
    assert all(w == 2 * RESIDUAL for w in wiped) and first == 2 * HBM


def test_custody_is_the_share_of_the_registered_weights_the_ledgers_epochs_could_have_carried_out():
    """What could leave adds up over a horizon, so weight custody is a cumulative check. Over the ledger's
    epochs it adds each one's covert budget, its export where it is charged, and σ from each server (a block's GPU)
    over the epoch's real duration, under `DEDICATED_RACKS` (1 kbit/s a server), against the registered weights' bits:
    three 16 ms nights on two GPUs give 96 bits of side channel. On the leaf datacenter the export isn't charged and is
    reported beside the bits; with every link covered it is charged. At a co-resident DRAM channel's rate (DRAMA, about
    2 Mbit/s) the same nights could have carried more than the weights."""
    registered = 2 * 32 * 32 * 16

    def nights(dc: Datacenter) -> Accounts:
        ledger = Accounts()
        for night in range(3):
            plan, q = Plan(Interval(16 * night, 16 * night + 16)), staged(dc, start=16 * night)
            ledger.add(account(audit(run(q, INPUTS, plan), q.public(plan)), q.public(plan)))
        return ledger

    covert = 3 * (4 * 4 * 7 + 512)
    leaf, covered = nights(datacenter()).custody(registered), nights(certified()).custody(registered)
    assert (leaf.covert, leaf.export, leaf.uncharged) == (covert, 0, 3 * 1024) and leaf.side == pytest.approx(96)
    assert leaf.bits == pytest.approx(covert + 96) and leaf.share == pytest.approx((covert + 96) / registered)
    assert (covered.export, covered.uncharged) == (3 * 1024, 0) and covered.bits == pytest.approx(covert + 96 + 3 * 1024)
    assert "3072 exported bits not charged" in repr(leaf) and "not charged" not in repr(covered)
    dram = nights(datacenter()).custody(registered, Assumption("co-resident-dram", 2e6, "bit/s per server", "measured"))
    assert dram.side == pytest.approx(2e6 * 3 * 16e-3 * 2) and dram.share > 1
