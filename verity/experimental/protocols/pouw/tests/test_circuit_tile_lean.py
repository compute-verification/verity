"""The Lean tile circuits are `pc8`'s and `pc4`'s tile templates: the same computing gates in the same order, their
primitives, widths, reads and outputs, with the template's parameter leaves as its input gates. `lean_table` restates
`TileCircuit.tileGate` (`verity/Security/Definitions/Pouw/PearlC/TileCircuit.lean`, `tileSpec p K TM TN`) and `lean_table4` restates
`TileCircuit.tile4Gate` (`Tile4Circuit.lean`, `tile4Spec K TM TN`); the Lean proofs (`tile_correct`, `tile4_correct` in
`Pouw.SecurityProofs.PearlC`) are about those tables, so this test and the pinned descriptor digests are what tie them to
the Programs. Each gate's operand order and its primitive's semantics are the per-primitive trusted step (the atom is
`p.step` or `nvf4Word`, `F32Add_v2` is `f32Add`, the K16 step is `k16Step`)."""
from __future__ import annotations

import pytest

from verity.ml.circuits import Program
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.cut import CallGraph, call_scope

pc8 = pytest.importorskip("verity.experimental.protocols.pouw.circuit.pc8", reason="the tile template is `verity.experimental.protocols.pouw.circuit.pc8`'s")
pc4 = pytest.importorskip("verity.experimental.protocols.pouw.circuit.pc4", reason="the tile template is `verity.experimental.protocols.pouw.circuit.pc4`'s")

#: `program_digest` of the templates the Lean table was checked against; a change to any needs the Lean re-read
DIGESTS = {
    ("sm120", 128, (1, 1)): "60e44c42fb163b9d86be70de09a94ac30ce0dcc8a1903ca1a3dcc1e268db3992",
    ("h100", 128, (1, 1)): "16500d4572bdf76ad277cabe2ba5192e4a167871e501e9c575a590a1516b9a4e",
    ("sm120", 128, (64, 64)): "72573caa5130511ee1dbe6634960e363b4f2fd75fd602070ce0360f2eee53e08",
    ("h100", 128, (64, 64)): "dc0d069eb9864a5cd2506faef104b85b6c0d40c9dad86733632f2e94ab29f1fe",
}

#: the deployed tile (`pc8.TILE`) at sm_120's depth, each digest about a minute to compute
DEPLOYED = {
    ("sm120", 8192, (64, 64)): "2846d5e7eed7bfa51adad62015ff5889b98b06571e76e4231e33f0423a8132e6",
    ("h100", 8192, (64, 64)): "14f7b954d979d302d75a89088c747b06b557fd65b2c661d9f264a864388e154f",
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
    (128, (1, 1)): "f04828d5567a398e799ce9afd180e18537a049885c86e36b1ed1b1e14ddef5cd",
    (128, (64, 64)): "3d60f4cd4d3554745c42bf1d1a08f53c194d53f6fa4da5a6cc812bb2f62313c7",
}

#: the deployed tile (`pc4.TILE`) at K = 8192
DEPLOYED4 = {(8192, (64, 64)): "36ea3001612e01cf8fea36de9c83a2af9617232a00e9d3a981bf7762b8327574"}


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
