"""The Lean tile circuits are `pc8`'s and `pc4`'s tile templates: the same computing gates in the same order, their
primitives, widths, reads and outputs, with the template's parameter leaves as its input gates. `lean_table` restates
`TileCircuit.tileGate` (`Security/Definitions/Pouw/PearlC/TileCircuit.lean`, `tileSpec p K TM TN`) and `lean_table4` restates
`TileCircuit.tile4Gate` (`Tile4Circuit.lean`, `tile4Spec K TM TN`); the Lean proofs (`tile_correct`, `tile4_correct` in
`Pouw.SecurityProofs.PearlC`) are about those tables, so this test and the pinned descriptor digests are what tie them to
the Programs. Each gate's operand order and its primitive's semantics are the per-primitive trusted step (the atom is
`p.step` or `nvf4Word`, `F32Add_v2` is `f32Add`, the K16 step is `k16Step`)."""
from __future__ import annotations

import pytest

from verity.ml.circuits import Program
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.cut import CallGraph, call_scope

pc8 = pytest.importorskip("verity.core.protocols.pouw.circuit.pc8", reason="the tile template is `verity.core.protocols.pouw.circuit.pc8`'s")
pc4 = pytest.importorskip("verity.core.protocols.pouw.circuit.pc4", reason="the tile template is `verity.core.protocols.pouw.circuit.pc4`'s")

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
                    gates.append(("atom", ([] if r % 5 == 0 else [g - 1]) + [a + 32 * t + l for l in range(32)]
                                  + [b + 32 * t + l for l in range(32)]))
                elif r < 5 * nw:
                    gates.append(("add", ([] if r // 5 == 0 else [g - 5]) + [g - 1]))
                else:
                    s = r - 5 * nw
                    gates.append(("peel", [g - 1] + [pa + 16 * s + l for l in range(16)]
                                  + [pb + 16 * s + l for l in range(16)]))
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
    assert pc8.TILE == tile
    assert program_digest(Program(pc8.templates(k, tile, dev)["tile"])) == DEPLOYED[dev, k, tile]


#: `program_digest` of the `pc4` templates the Lean table was checked against
DIGESTS4 = {
    (128, (1, 1)): "7cc96f4aa6a2b68a676404f3be86c997cb6d5889d67b4336adcb2301bb2b42dbf59011ee318ce0af96bcf2135df08607339dce9c7128b64f64e865e064749476",
    (128, (64, 64)): "81ac3d445c5518b223309462102f5a1a4381c91b2830d7a92937ecfb97e323b50ebf3fa186579bb5234075477c1a34f64616e4934173483f89bd780d8c8c9516",
}

#: the deployed tile (`pc4.TILE`) at K = 8192
DEPLOYED4 = {(8192, (64, 64)): "5d52d991c237f460928a26c00019c8ad906d0b50a95a9fa6ee5c74a95b8e0ac06d22d1e0315317b91dc87878e3f01796cd59d0d697cc3f641491a8c309dd9a19"}


def lean_table4(k: int, tm: int, tn: int):
    """`tile4Gate K TM TN`'s computed gates as (kind, reads) after its `tile4In` input gates, its input widths, and its
    outputs in the template's output-leaf order (every `c4Gate`, then every `u4Gate`, row-major)."""
    nc, row = k // 64, k + k // 16
    base, ln = (tm + tn) * row + 64 * (tm + tn), nc + 4
    widths = [4] * (tm * k) + [8] * (tm * (k // 16)) + [4] * (tn * k) + [8] * (tn * (k // 16)) + [16] * (64 * (tm + tn))
    gates, cs, us = [], [], []
    for i in range(tm):
        for j in range(tn):
            e = base + (i * tn + j) * ln
            a, b = i * k, tm * row + j * k
            sa, sb = tm * k + i * (k // 16), tm * row + tn * k + j * (k // 16)
            pa, pb = (tm + tn) * row + 64 * i, (tm + tn) * row + 64 * tm + 64 * j
            for r in range(ln):
                g = e + r
                if r < nc:
                    gates.append(("atom", ([] if r == 0 else [g - 1]) + [a + 64 * r + l for l in range(64)]
                                  + [b + 64 * r + l for l in range(64)] + [sa + 4 * r + l for l in range(4)]
                                  + [sb + 4 * r + l for l in range(4)]))
                else:
                    s = r - nc
                    gates.append(("peel", [g - 1] + [pa + 16 * s + l for l in range(16)]
                                  + [pb + 16 * s + l for l in range(16)]))
            cs.append(e + nc - 1)
            us.append(e + ln - 1)
    return base, widths, gates, cs + us


CASES4 = [(k, (1, 1)) for k in (64, 128, 1024, 8192)] + [(128, (2, 3)), (256, (3, 2)), (1024, (2, 2)), (128, (64, 64))]


@pytest.mark.parametrize("k,tile", CASES4)
def test_the_lean_tile4_table_is_the_templates_call_graph(k, tile):
    tm, tn = tile
    fn = pc4.templates(k, tile)["tile"]
    g = CallGraph(fn)
    c, _ = call_scope(fn)
    base, widths, table, outs = lean_table4(k, tm, tn)
    assert sorted(g.input_ids) == list(range(base))
    assert [c.gate(r).prim.ret.w for r in range(base)] == widths
    names = {v: n for n, v in g.prims.items()}
    kind = {"BlackwellNvf4OmmaDot64_v1": "atom", "HopperBF16WgmmaDot16_v1": "peel"}
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


@pytest.mark.parametrize("k,tile", sorted(DIGESTS4))
def test_the_checked_tile4_templates_are_pinned(k, tile):
    assert program_digest(Program(pc4.templates(k, tile)["tile"])) == DIGESTS4[k, tile]


@pytest.mark.slow
@pytest.mark.parametrize("k,tile", sorted(DEPLOYED4))
def test_the_deployed_tile4_is_pinned(k, tile):
    """Catches a change to `pc4`'s deployed template at K = 8192 that the table check and the K = 128 pins don't see."""
    assert pc4.TILE == tile
    assert program_digest(Program(pc4.templates(k, tile)["tile"])) == DEPLOYED4[k, tile]
