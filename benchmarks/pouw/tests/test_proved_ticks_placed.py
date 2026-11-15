"""PoUW's compiled Program on a block of the datacenter model (`benchmarks/datacenter`, `import datacenter`): the s8 grid
compiled to `h1tile.SCHEME` and to its one stream `SCHEME_V3` is placed like any other Program, `proved_ticks` reads the
block's work as its H1 tiles' (`tiles`, `placed`), and the block's account credits `proved_block_ticks`' figure as its
compute, basis "proved"."""

from __future__ import annotations

import proved_ticks as PT
import pytest
from datacenter import (FLEET_CLOSED, Assumption, Cadence, Computed, Credit, Datacenter, DeviceEntry, Interval, LinkEntry,
                        Plan, Zone, account, ands, crossings, names, place, register)
from datacenter.logical import census_of
from datacenter.models import s8_grid
from verity.protocols.pouw.prover.circuit import compile as pouw
from verity.protocols.pouw.prover.circuit import h1tile

EPOCH = Interval(0, 16)
HBM, RESIDUAL = 6.4e10, 8e6


def node(n: str) -> Zone:
    gpu = DeviceEntry(f"{n}.gpu", "gpu", rate={"mac:bf16": Assumption(f"{n}.gpu.bf16", 1e6, "MAC/s"),
                                               "cuda:gate": Assumption(f"{n}.gpu.gates", 1e7, "gate/s")},
                      memory=Assumption(f"{n}.gpu.hbm", HBM, "bits"),
                      residual=Assumption(f"{n}.gpu.residual", RESIDUAL, "bits"),
                      write=Assumption(f"{n}.gpu.write", 1e14, "bit/s"))
    return Zone(n, [gpu])


def datacenter() -> Datacenter:
    """Two blocks, a and b, of one GPU each behind its certifier, on one leaf switch with the challenger and the outside
    world, at a tick of 1 ms."""
    up = Assumption("uplink", 1e8, "bit/s", "declared")
    fabric = [DeviceEntry("a.cert", "certifier"), DeviceEntry("b.cert", "certifier"), DeviceEntry("leaf", "switch"),
              DeviceEntry("challenger", "challenger"), DeviceEntry("world", "outside")]
    links = [LinkEntry("a.gpu", "a.cert", up), LinkEntry("a.cert", "leaf", up), LinkEntry("leaf", "b.cert", up),
             LinkEntry("b.cert", "b.gpu", up), LinkEntry("challenger", "leaf", up), LinkEntry("world", "leaf", up)]
    cadence = Cadence(frame_bits=32, bucket_ticks=1, buckets=4, jitter_classes=128,
                      tick=Assumption("tick", 1e-3, "s", "declared"))
    return Datacenter("dc0", [node("a"), node("b")], fabric, links, cadence)


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
