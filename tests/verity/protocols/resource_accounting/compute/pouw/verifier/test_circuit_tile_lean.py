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
    ("sm120", 128, (1, 1)): "b615882ad409a1d1a12282202ffa84786bbd5eb960407ebcf7bc17df1bff2c3ccd699331453a3a2ae2db8211bc81b32ef86ff18afa9f05936b2518a2e03fb25a",
    ("h100", 128, (1, 1)): "58f33e76ef947b1871ad22c60b847d982534f7eb31eebc3781e2f83bb5cc8728c7b4585e7f046bd3be55154ffbaa1ee6acaf10de55721442e5856a8ef06f4b8f",
    ("sm120", 128, (64, 64)): "c89ff74d9a43756c968d65f282d5d19439c7d17f6a23b37b43b28aad4f6623491581489176d21ed274bdf315ca14ff1c1acd579c1cbc7f99028d8fe974f7db34",
    ("h100", 128, (64, 64)): "b31837029e02d6f3d95aa5e0d37b714bebea95fb40e328f1f79fe9adeb36c418d72e28ff76946c0ad46986ac1b5eb31bfeaa931e8007f881693afa886539890e",
}

#: the deployed tile (`pc8.TILE`) at sm_120's depth, each digest about a minute to compute
DEPLOYED = {
    ("sm120", 8192, (64, 64)): "9b3a88a5348af3e9b6bd3e8e6b33d7831d265873fede19cab5a91c76a58c3ccda2905aa4d9286035f703ac74aa84c37c15d39b98998a149cbdfb9c78d8e3a9ae",
    ("h100", 8192, (64, 64)): "3e319a24be47bd8cf4cc553287e5b388e47c7cc034bd7a006bf7eaf442e844686e036697c778d268999b400479d4fb1a1bff5f8819da3aa6002891e677ac8ff4",
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
