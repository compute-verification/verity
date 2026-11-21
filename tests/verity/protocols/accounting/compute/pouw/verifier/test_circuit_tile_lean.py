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

pc8 = pytest.importorskip("verity.protocols.accounting.compute.pouw.prover.circuit.pc8", reason="the tile template is `verity.protocols.accounting.compute.pouw.prover.circuit.pc8`'s")

#: `program_digest` of the templates the Lean table was checked against; a change to any needs the Lean re-read
DIGESTS = {
    ("sm120", 128, (1, 1)): "5161648ddbabe3b52bbdc3aed280d4824f157130fb89cbbcdd315f50eb93621a71651d3b11913e470056024d1303a8ebf7d3485259f3d9ee43f6653b30a75754",
    ("h100", 128, (1, 1)): "6aef0aa9678a41e8dc9c008ee95f89718eb8e60db51d51ebc1f19ef9edf71b2e5c7751125ac59582ed2cf8b21e070e2b6869f155b9be8928cd5182905bc4aaeb",
    ("sm120", 128, (64, 64)): "27faa42d38a32eb817a959b9e0a706ce0bcc1a4f73baa797d408a3542ed5dec29a7434dea365c33b72fa42fd809ad18e653924d6c7fe31bda4cddc2cdc8fb1b9",
    ("h100", 128, (64, 64)): "6093643487681ac574c505ca202608f6ac42feeab289e4047df0e8ee315c2fb4b7ae6c660286226789e027c2a1ecca47bdafd5cc390de01a60ead4cf344e81cd",
}

#: the deployed tile (`pc8.TILE`) at sm_120's depth, each digest about a minute to compute
DEPLOYED = {
    ("sm120", 8192, (64, 64)): "8f147b67a529efc57b35186ca4ce726ceb2c399862cb44f855b3216744231b142c6d1fe4486182623691c29522d9bcfaf3123b997f517c40b8bdfde2a33a9655",
    ("h100", 8192, (64, 64)): "94927ab5a7b1af4081f42bc1794fce5719b047fe968905dc7854173b6dc9ab8d78495927b9a0b497785dc8c1b5b72e25f46ff7ab3f824cefa586d581f6730232",
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
