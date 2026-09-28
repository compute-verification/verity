"""`verity.ml.commitments.gates.frame_v3`: the hidden-position read of a dense frame-v3-sha512 tree reproduces the roots
of ``frame_v3/vectors_sha512.json`` at every position of its trees and opens its hiding leaves (``hm96-sha512/row/v2``,
``hm96-sha512/row-seg/v1``) at their rank; it refuses a wrong position, root, sibling, value or domain id; its leaf and node
agree with `CommitmentDomain.leaf` and `.node` at every length of uint the deployed depths take (one to three bytes); and its
parts cost the gates pinned here."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.frame_v3.vectors_sha512 import domain
from verity.ml.commitments.gates import frame_v3 as F
from verity.ml.commitments.gates.sha512 import words_be
from verity.ml.commitments.merkle import MerkleTree, merkle_depth

V = json.loads((Path(F.__file__).parents[1] / "frame_v3" / "vectors_sha512.json").read_text())
ROW, SEG = "hm96-sha512/row/v2", "hm96-sha512/row-seg/v1"
#: (part, its statics) -> (And, Xor, Not) gates of its traced circuit: the leaf at D = 1 and at the deployed depths' widths
#: (D_A = 13 over row-seg rows, D_B = 17 over v2 rows), and the node at both index widths a read of D <= 17 takes
PARTS = {
    ("FrameV3Leaf", (1, ROW)): (110125, 538268, 11450),
    ("FrameV3Leaf", (2, SEG)): (110922, 540966, 11307),
    ("FrameV3Leaf", (3, ROW)): (111683, 542517, 11198),
    ("FrameV3Node", (1,)): (166475, 815851, 16050),
    ("FrameV3Node", (2,)): (167630, 818406, 16011),
}
#: (D, SCHEMA) -> the read's gates (`SpecializedDefinition.gates`): the size a unit holding it is cut to
READS = {(1, ROW): 1659755, (13, SEG): 13658304, (17, ROW): 17668695}


def _hex(x: str) -> bytes:
    return bytes.fromhex(x)


def _trees() -> list[tuple[bytes, str, list[bytes], list[bytes], bytes, list[list[bytes]]]]:
    """Every vector tree: (domain id, schema, values, leaves, root, paths)."""
    return [(_hex(t["domain"]["domain_id"]), t["schema"], [_hex(x) for x in t["digests"]], [_hex(x) for x in t["leaves"]],
             _hex(t["root"]), [[_hex(p) for p in ps] for ps in t["paths"]])
            for k in ("row_trees", "bit_row_trees", "binding64_trees") for t in V[k]]


def _hiding() -> list[dict]:
    """The hiding leaves: two bit rows under ``hm96-sha512/row/v2`` and the 8 KiB row-seg row under ``hm96-sha512/row-seg/v1``,
    each at rank 1 of a two-leaf domain."""
    return [*V["bit_row_hiding_leaves"], *(r["hiding"] for r in V["row_seg_leaves"] if "hiding" in r)]


def _bits(d: bytes) -> list[int]:
    return [(w >> j) & 1 for w in words_be(d) for j in range(64)]


def _digest(bits) -> bytes:
    w = np.asarray(bits).reshape(8, 64)
    return b"".join(sum(int(b) << j for j, b in enumerate(row)).to_bytes(8, "big") for row in w)


def _columns(rows) -> list[np.ndarray]:
    return [np.array([r[k] for r in rows], dtype=np.uint8) for k in range(len(rows[0]))]


def test_the_reference_reproduces_the_vector_trees() -> None:
    trees = _trees()
    assert sorted(len(t[2]) for t in trees) == [1, 1, 2, 3, 3, 3, 5, 5]
    for dom, schema, values, leaves, root, paths in trees:
        for r, (value, leaf, path) in enumerate(zip(values, leaves, paths)):
            assert F.leaf_ref(dom, schema, value, r) == leaf
            assert F.root_ref(dom, schema, value, r, path) == root
            if path:
                assert F.opens(schema, root, dom, value, r, path) == 1
                assert F.opens(schema, root, dom, value, r ^ 1, path) == 0
    for h in _hiding():
        assert F.leaf_ref(_hex(h["domain"]["domain_id"]), h["schema"], _hex(h["hm96_leaf"]), h["rank"]) == _hex(h["leaf"])


def test_the_reference_is_the_library_at_every_uint_length() -> None:
    """Positions and indices whose uint takes one, two and three bytes, against `MerkleTree` and `CommitmentDomain`."""
    rng = random.Random("frame-v3-read")
    d = domain(300, "gates/frame-v3")
    values = {p: rng.randbytes(64) for p in range(300)}
    tree = MerkleTree(d, values, lambda p: ROW)
    for p in (0, 1, 2, 127, 255, 256, 257, 299):
        o = tree.open(p)
        assert len(o.path) == merkle_depth(300) == 9
        assert F.root_ref(d.domain_id, ROW, values[p], p, o.path) == tree.commitment.root
    l, r = rng.randbytes(64), rng.randbytes(64)
    for x in (0, 1, 255, 256, 65535, 65536, (1 << 24) - 1):
        assert F.leaf_ref(d.domain_id, SEG, l, x) == d.leaf(x, x, SEG, l)
        assert F.node_ref(d.domain_id, 7, x, l, r) == d.node(7, x, l, r)


def test_the_read_is_typed_by_its_shape() -> None:
    for D in (1, 13, 17):
        target = F.read(D, ROW)
        assert target is F.read(D, ROW)
        assert [t.leaves for _, t in target.signature()[0]] == [512, 512, 512, D, 512 * D]
        assert target.signature()[1] == BL.BIT
    assert [F.index_bytes(b) for b in (0, 1, 8, 9, 16, 17, 24)] == [1, 1, 1, 2, 2, 3, 3]


def test_malformed_bindings_are_refused() -> None:
    for D in (0, 65, "3"):
        with pytest.raises(ValueError, match="D is an int in 1 ... 64"):
            F.read(D, ROW).signature()
    with pytest.raises(ValueError, match="nonempty str"):
        F.read(3, "").signature()
    with pytest.raises(ValueError, match="one block count per binding"):
        F.read(9, "x" * 27).signature()
    F.read(8, "x" * 27).signature()
    for B in (0, 9):
        with pytest.raises(ValueError, match="B is an int in 1 ... 8"):
            bind(F.FrameV3Node, B=B).signature()
    with pytest.raises(ValueError, match="below 2"):
        F.arguments(bytes(64), bytes(64), bytes(64), 4, [bytes(64)] * 2)
    with pytest.raises(ValueError, match="64 bytes"):
        F.arguments(bytes(64), bytes(32), bytes(64), 0, [bytes(64)])
    with pytest.raises(ValueError, match="D is an int"):
        F.arguments(bytes(64), bytes(64), bytes(64), 0, [])


def _read_cases(dom: bytes, schema: str, values: list[bytes], root: bytes, paths: list[list[bytes]],
                rng: random.Random) -> tuple[list, list[int]]:
    """Every position opened honestly, then each read with one thing wrong: the position (its low bit, its high bit), the root,
    a sibling, the value, the domain id."""
    D = len(paths[0])
    rows, want = [], []
    for pos, (value, path) in enumerate(zip(values, paths)):
        bad = list(path)
        bad[rng.randrange(D)] = rng.randbytes(64)
        cases = [(root, dom, value, pos, path), (root, dom, value, pos ^ 1, path), (rng.randbytes(64), dom, value, pos, path),
                 (root, dom, value, pos, bad), (root, dom, rng.randbytes(64), pos, path), (root, rng.randbytes(64), value, pos, path)]
        if D > 1:
            cases.append((root, dom, value, pos ^ 1 << (D - 1), path))
        for args in cases:
            rows.append(F.arguments(*args))
            want.append(F.opens(schema, *args))
    return rows, want


@pytest.mark.slow
def test_reads_open_the_vector_trees_and_refuse_every_wrong_read() -> None:
    """A few minutes, most of it tracing a leaf per schema and the node once: the only check that the read's wiring (each level's
    position bit, index and level constant) climbs to the roots the library wrote, at D = 1, 2 and 3."""
    rng = random.Random("frame-v3-vectors")
    shapes = set()
    for dom, schema, values, _leaves, root, paths in _trees():
        if not paths[0]:
            continue
        D = len(paths[0])
        shapes.add((D, schema))
        rows, want = _read_cases(dom, schema, values, root, paths, rng)
        honest = len(values)
        assert want.count(1) == honest and len(want) == honest * (6 + (D > 1))
        got = evaluate_bits(F.read(D, schema), *_columns(rows)).reshape(-1).tolist()
        assert got == want, (D, schema)
    assert shapes == {(1, "sha512/row/v1"), (2, "sha512/row/v1"), (3, "sha512/row/v1"), (2, "sha512/row/v2")}


@pytest.mark.slow
def test_reads_open_the_hiding_leaves_at_their_rank() -> None:
    """Each hiding leaf's frame leaf, from its domain id, rank and hm96 leaf; and the read opens it at rank 1 of its two-leaf
    tree and refuses it at rank 0."""
    rng = random.Random("frame-v3-hiding")
    for h in _hiding():
        dom, schema, value, rank = _hex(h["domain"]["domain_id"]), h["schema"], _hex(h["hm96_leaf"]), h["rank"]
        assert h["domain"]["range"] == [0, 2] and rank == 1
        x = np.array([[(rank >> j) & 1 for j in range(8)]], dtype=np.uint8)
        got = evaluate_bits(bind(F.FrameV3Leaf, B=1, SCHEMA=schema), np.array([_bits(dom)]), np.array([_bits(value)]), x)
        assert _digest(got[0]) == _hex(h["leaf"])
        other = F.leaf_ref(dom, schema, rng.randbytes(64), 0)
        root = F.node_ref(dom, 0, 0, other, _hex(h["leaf"]))
        rows = [F.arguments(root, dom, value, rank, [other]), F.arguments(root, dom, value, rank ^ 1, [other])]
        assert evaluate_bits(F.read(1, schema), *_columns(rows)).reshape(-1).tolist() == [1, 0]


@pytest.mark.slow
def test_a_read_at_d_b_opens_the_library_tree_and_refuses_wrong_positions() -> None:
    """About a minute and a half: the read at D_B = 17, whose leaf takes a three-byte uint and whose nodes switch from two-byte
    to one-byte indices at level 8, opens `MerkleTree`'s paths of a 2^16 + 1-leaf tree at its edge positions, and refuses each
    with its low or high position bit flipped. No shallower read reaches either width change."""
    n = (1 << 16) + 1
    d = domain(n, "gates/frame-v3/d17")
    rng = random.Random("frame-v3-d17")
    values = {p: rng.randbytes(64) for p in range(n)}
    tree = MerkleTree(d, values, lambda p: ROW)
    rows, want = [], []
    for p in (0, 255, 256, 65535, 65536):
        o = tree.open(p)
        assert len(o.path) == merkle_depth(n) == 17 and o.value == values[p]
        for q in (p, p ^ 1, p ^ 1 << 16):
            rows.append(F.arguments(tree.commitment.root, d.domain_id, values[p], q, o.path))
            want.append(int(q == p))
    assert evaluate_bits(F.read(17, ROW), *_columns(rows)).reshape(-1).tolist() == want


def _x_bits(x: int, B: int) -> list[int]:
    return [(x >> j) & 1 for j in range(8 * B)]


@pytest.mark.slow
def test_the_parts_agree_with_the_library_at_the_deployed_widths() -> None:
    """A few minutes of tracing: the uint-length selection of D_A = 13's leaf (two bytes, row-seg), D_B = 17's leaf (three
    bytes) and a node's two-byte index, at each length's edges, which the vector trees (one-byte positions) never reach."""
    rng = random.Random("frame-v3-widths")
    d = domain(1 << 17, "gates/frame-v3/widths")
    dom, l, r = d.domain_id, rng.randbytes(64), rng.randbytes(64)
    for B, schema, xs in ((2, SEG, (0, 1, 255, 256, 8191, 65535)), (3, ROW, (0, 255, 256, 65535, 65536, 131071, (1 << 24) - 1))):
        got = evaluate_bits(bind(F.FrameV3Leaf, B=B, SCHEMA=schema), np.array([_bits(dom)] * len(xs)), np.array([_bits(l)] * len(xs)),
                            np.array([_x_bits(x, B) for x in xs], dtype=np.uint8))
        assert [_digest(g) for g in got] == [d.leaf(x, x, schema, l) for x in xs], (B, schema)
    cases = [(0, 0, 0), (0, 255, 1), (0, 256, 0), (3, 4095, 1), (7, 65535, 0), (15, 1, 1)]
    got = evaluate_bits(bind(F.FrameV3Node, B=2), np.array([_bits(dom)] * len(cases)),
                        np.array([_x_bits(k, 1) for k, _, _ in cases], dtype=np.uint8),
                        np.array([_x_bits(i, 2) for _, i, _ in cases], dtype=np.uint8), np.array([[s] for *_, s in cases], dtype=np.uint8),
                        np.array([_bits(l)] * len(cases)), np.array([_bits(r)] * len(cases)))
    assert [_digest(g) for g in got] == [d.node(k, i, *((r, l) if s else (l, r))) for k, i, s in cases]


@pytest.mark.slow
def test_the_parts_and_reads_cost_the_pinned_gates() -> None:
    for (name, statics), counts in PARTS.items():
        assert getattr(F, name).traced(*statics).counts == dict(zip(("and", "xor", "not"), counts)), (name, statics)
    for (D, schema), gates in READS.items():
        assert F.read(D, schema).gates == gates, (D, schema)
