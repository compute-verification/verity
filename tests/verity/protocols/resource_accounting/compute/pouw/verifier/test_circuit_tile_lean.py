"""The Lean tile circuits are `pc8`'s tile templates: the same computing gates in the same order, their primitives, widths,
reads and outputs, with the template's parameter leaves as its input gates. `lean_table` restates `TileCircuit.tileGate`
(`security/Definitions/Pouw/PearlC/TileCircuit.lean`, `tileSpec p K TM TN`; `pc4`'s, `tile4Gate`, is
`test_circuit_pc4_lean`'s); the Lean proofs (`tile_correct` in `Pouw.SecurityProofs.PearlC`) are about those tables, so this test and the pinned descriptor digests are what tie them to
the Programs. Each gate's operand order and its primitive's semantics are the per-primitive trusted step (the atom is
`p.step` or `nvf4Word`, `F32Add_v2` is `f32Add`, the K16 step is `k16Step`)."""
from __future__ import annotations

import pytest

from verity.circuits import Program
from verity.circuits.codec import program_digest
from verity.circuits.cut import CallGraph, call_scope

pc8 = pytest.importorskip("verity.protocols.resource_accounting.compute.pouw.prover.circuit.pc8", reason="the tile template is `verity.protocols.resource_accounting.compute.pouw.prover.circuit.pc8`'s")

#: `program_digest` of the templates the Lean table was checked against; a change to any needs the Lean re-read
DIGESTS = {
    ("sm120", 128, (1, 1)): "04f0809f2e0f72610890d2beb518badbc1c6f7b01ad3baf529ce5457ca563b23ee5ccacaa97e41bdc242bf93ffdee9f2bc68284f83824f853225b185f2da0e49",
    ("h100", 128, (1, 1)): "3e526ab30132be923fa9808e278f8190e5b9a93384a100c71f5d1e07e886de0dfc866f29014f587d28705abd1976431445461c8590907ded2627710465d4d8fe",
    ("sm120", 128, (64, 64)): "4481154ab194ae5e72d0b79bbfa93a6e54af9bf6dce303864fc9d6d9b39ff0f8e235b7601287063e14c30807717032577c1999cb94ad163aa7ca6d7232822fc4",
    ("h100", 128, (64, 64)): "6f79801eb52aca36b9300d269e250e427e031d573e2dfb7f7e042ce517f7e737978395d1aa23c896dcd40b9a1630caed5053323f79ac0725323d9fe422a4d5a0",
}

#: the deployed tile (`pc8.TILE`) at sm_120's depth, each digest about a minute to compute
DEPLOYED = {
    ("sm120", 8192, (64, 64)): "7ea9b0a3eed5537b2ab06d611b657ad059a0255187cb36ecd1db4f88fefbd946656bd0d7b5c4618097bec56a9a25eb77d07979160dfc3f3ac74fbb2cd459c0f7",
    ("h100", 8192, (64, 64)): "7e96c81a8d91726ad068700e0fbfb92275df226ca691609010cec0916031756e0c14d5cc5dadcb996ec377120d40ecce25bbce003d444042ef3fd85b239116bb",
}


def lean_table(k: int, tm: int, tn: int):
    """`tileGate p K TM TN`'s computed gates as (kind, reads) after its `tileIn` input gates, and its outputs in the
    template's output-leaf order (every `cGate`, then every `uGate`, row-major)."""
    nw = k // 128
    base, ln = (tm + tn) * k + 64 * (tm + tn), 5 * nw + 4
    gates, cs, us = [], [], []
    for i in range(tm):
        for j in range(tn):
            e = base + (i * tn + j) * ln
            a, b = i * k, tm * k + j * k
            pa, pb = (tm + tn) * k + 64 * i, (tm + tn) * k + 64 * tm + 64 * j
            for r in range(ln):
                g = e + r
                if r < 5 * nw and r % 5 < 4:
                    t = 4 * (r // 5) + r % 5
                    gates.append(("atom", ([] if r % 5 == 0 else [g - 1]) + [a + 32 * t + p for p in range(32)]
                                  + [b + 32 * t + p for p in range(32)]))
                elif r < 5 * nw:
                    gates.append(("add", ([] if r // 5 == 0 else [g - 5]) + [g - 1]))
                else:
                    s = r - 5 * nw
                    gates.append(("peel", [g - 1] + [pa + 16 * s + p for p in range(16)]
                                  + [pb + 16 * s + p for p in range(16)]))
            cs.append(e + 5 * nw - 1)
            us.append(e + ln - 1)
    return base, gates, cs + us


CASES = [(dev, k, (1, 1)) for dev in ("sm120", "h100") for k in (128, 256, 1024, 8192)] + [
    ("sm120", 128, (2, 3)), ("h100", 256, (3, 2)), ("sm120", 1024, (2, 2)), ("sm120", 128, (64, 64))]


@pytest.mark.parametrize("dev,k,tile", CASES)
def test_the_lean_tile_table_is_the_templates_call_graph(dev, k, tile):
    tm, tn = tile
    fn = pc8.templates(k, tile, dev)["tile"]
    g = CallGraph(fn)
    c, _ = call_scope(fn)
    base, table, outs = lean_table(k, tm, tn)
    assert sorted(g.input_ids) == list(range(base))
    assert [c.gate(r).prim.ret.w for r in range(base)] == [8] * ((tm + tn) * k) + [16] * (64 * (tm + tn))
    names = {v: n for n, v in g.prims.items()}
    kind = {pc8.ATOMS[pc8.DEVICES[dev].atom.name].id: "atom", "F32Add_v2": "add", "HopperBF16WgmmaDot16_v1": "peel"}
    reads = [set() for _ in range(g.n)]
    for s, d in zip(g.src, g.dst):
        reads[d].add(base + s)
    for ix, d in g.input_reads:
        reads[d].add(g.input_ids[ix])
    assert g.n == len(table)
    assert [kind[names[g.prim[i]]] for i in range(g.n)] == [t for t, _ in table]
    assert list(g.width) == [32] * g.n
    assert reads == [set(r) for _, r in table]
    assert all(len(set(r)) == len(r) for _, r in table)
    assert [base + o for o in g.outputs] == outs


@pytest.mark.parametrize("dev,k,tile", sorted(DIGESTS))
def test_the_checked_templates_are_pinned(dev, k, tile):
    assert program_digest(Program(pc8.templates(k, tile, dev)["tile"])) == DIGESTS[dev, k, tile]


@pytest.mark.slow
@pytest.mark.parametrize("dev,k,tile", sorted(DEPLOYED))
def test_the_deployed_tile_is_pinned(dev, k, tile):
    """Catches a change to the deployed template at K = 8192 that the table check and the K = 128 pins don't see,
    such as one gated on the depth or the tile."""
    assert tile == pc8.TILE
    assert program_digest(Program(pc8.templates(k, tile, dev)["tile"])) == DEPLOYED[dev, k, tile]
