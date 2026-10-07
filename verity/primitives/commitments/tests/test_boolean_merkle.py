"""`verity.primitives.commitments.gates.merkle`: the hidden-index Merkle read agrees with `opens` (``hashlib``) at every position of small binary
and 4-ary trees, refuses a wrong root, sibling, leaf or position, is a function of its shape alone (pinned by gate counts and
program digest), and costs per level the ANDs and C-Flock rows pinned here; `sha512.compress_ref` is SHA-512's compression.
The subtree read agrees with `opens_run` at every run of small trees over hm96-sha512 leaves, refuses a wrong root, sibling,
cell, position or cell order, and costs exactly its cells' leaves, its run's nodes and its levels above the run."""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import pytest

from verity.primitives.circuits import evaluation as E
from verity.primitives.commitments import hm96
from verity.primitives.circuits.evaluation.reference import standalone_call
from verity.primitives.circuits import boolean as BL
from verity.primitives.circuits import codec
from verity.primitives.circuits.defs import bind
from verity.primitives.commitments.gates import merkle as M
from verity.primitives.commitments.gates import sha512 as S
from verity.primitives.circuits.boolean.forms import ONE, Circuit

#: (ARITY, D) -> (And, Xor, Not gates, standalone program digest)
PINS = {
    (2, 1): (103101, 479098, 14170, "98e0dc34f59b9e7ef3145fe5a110225d53cbe2ef280c20750dd666c187f63dbc"),
    (4, 1): (163576, 769803, 19049, "6745e9f7599cb374c60ebdca2cdf075db75a4dafda4de1467a39e519282d683a"),
}
#: ARITY -> (ANDs, C-Flock rows) one level adds: the node's compressions and the slot multiplexers; rows are the ANDs and the
#: copies `sha512.commit` adds (an untraced `forms.Circuit`, no common-subexpression pass)
PER_LEVEL = {2: (102591, 142847), 4: (163066, 226774)}
#: (ARITY, D, J) -> (And, Xor, Not gates, standalone program digest) of the subtree read at its circuit-check bindings
SUBTREE_PINS = {
    (2, 2, 1): (409241, 1909832, 54776, "ee057ded7e8afab6a717f7cad0486ece5c08d6fc0b125381c9f0fc88efa1862c"),
    (4, 1, 0): (265607, 1246645, 32523, "032b690e05d19cb97d93139f16d07e801ae5941ac2e73771ecd6a26f854e18f7"),
}
#: (ANDs, rows) of one cell's leaf from its b || c, of the comparison with the root, and per ARITY of one node inside a run
LEAF, COMPARE = (102031, 142282), (511, 511)
RUN_NODE = {2: (102079, 142335), 4: (159992, 223700)}


def _read(A: int, D: int):
    return bind(M.MerkleRead, ARITY=A, D=D)


def _subtree(A: int, D: int, J: int):
    return bind(M.MerkleSubtreeRead, ARITY=A, D=D, J=J)


def _tree(rng: random.Random, A: int, D: int, leaves: list[bytes] | None = None) -> list[list[bytes]]:
    levels = [leaves or [rng.randbytes(64) for _ in range(A ** D)]]
    while len(levels[-1]) > 1:
        lv = levels[-1]
        levels.append([M.node_ref(lv[i:i + A]) for i in range(0, len(lv), A)])
    return levels


def _path(levels: list[list[bytes]], A: int, pos: int) -> list[list[bytes]]:
    out = []
    for lv in levels[:-1]:
        g = pos - pos % A
        out.append([lv[g + j] for j in range(A) if g + j != pos])
        pos //= A
    return out


def _columns(rows) -> list[np.ndarray]:
    return [np.array([r[k] for r in rows], dtype=np.uint8) for k in range(4)]


def test_the_reference_is_sha512() -> None:
    for m in (b"", b"abc", bytes(range(111))):
        assert S.compress_ref(S.IV, S.words_be(m + S.pad_tail(len(m)))) == S.words_be(hashlib.sha512(m).digest())
    long = os.urandom(300)
    cv, padded = S.IV, long + S.pad_tail(len(long))
    for i in range(0, len(padded), 128):
        cv = S.compress_ref(cv, S.words_be(padded[i:i + 128]))
    assert cv == S.words_be(hashlib.sha512(long).digest())


def test_the_reference_opens_every_leaf_of_a_tree() -> None:
    rng = random.Random(1)
    for A, D in ((2, 3), (4, 2)):
        levels = _tree(rng, A, D)
        for pos, leaf in enumerate(levels[0]):
            path = _path(levels, A, pos)
            assert M.opens(A, levels[-1][0], leaf, pos, path) == 1
            assert M.opens(A, levels[-1][0], leaf, (pos + 1) % A ** D, path) == 0


def _cases(A: int, D: int, rng: random.Random) -> tuple[list, list[int]]:
    """Every position of a random tree opened honestly, then each read with one thing wrong: the root, a sibling, the leaf, the
    position."""
    levels = _tree(rng, A, D)
    root, rows, want = levels[-1][0], [], []
    for pos, leaf in enumerate(levels[0]):
        path = _path(levels, A, pos)
        bad = [list(s) for s in path]
        bad[rng.randrange(D)][rng.randrange(A - 1)] = rng.randbytes(64)
        for args in ((root, leaf, pos, path), (rng.randbytes(64), leaf, pos, path), (root, leaf, pos, bad),
                     (root, rng.randbytes(64), pos, path), (root, leaf, pos ^ 1, path)):
            rows.append(M.arguments(A, *args))
            want.append(M.opens(A, *args))
    return rows, want


@pytest.mark.slow
@pytest.mark.parametrize("shape", [(2, 1), (4, 1), (2, 2)], ids=str)
def test_reads_agree_with_the_reference(shape) -> None:
    A, D = shape
    rows, want = _cases(A, D, random.Random(f"merkle{shape}"))
    assert want.count(1) == A ** D and want.count(0) == 4 * A ** D
    got = np.asarray(E.evaluate_batch(_read(A, D), *_columns(rows))).reshape(-1).tolist()
    assert got == want
    assert list(E.evaluate(_read(A, D), *rows[0])) == [1]


@pytest.mark.slow
@pytest.mark.parametrize("shape", sorted(PINS), ids=str)
def test_the_read_is_a_function_of_its_shape(shape) -> None:
    """Its circuit is pinned by digest: no position, path or leaf enters it but as the values of its arguments."""
    ands, xors, nots, digest = PINS[shape]
    target = _read(*shape)
    assert BL.is_boolean(target) and target is _read(*shape)
    assert M.traced(*shape).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    assert codec.program_digest(codec.encode_program(prog)) == digest
    A, D = shape
    assert [p.leaves for _, p in target.signature()[0]] == [512, 512, (A.bit_length() - 1) * D, 512 * (A - 1) * D]


def _counts(A: int, D: int, J: int | None = None) -> tuple[int, int]:
    C = Circuit()
    ps, _ = M._types(A, D, J)
    els = [C.input(f"e{i}", w) for i, w in enumerate(w for t in ps for w in BL.elements(t))]
    C.output("ok", M.build(C, A, D, *els) if J is None else M.build_subtree(C, A, D, J, *els))
    ands = copies = 0
    for v in C.live():
        if C.kind[v] == "and":
            a, b = C.data[v]
            copies += ONE.key() in (a.key(), b.key())
            ands += ONE.key() not in (a.key(), b.key())
    return ands, ands + copies


@pytest.mark.slow
@pytest.mark.parametrize("A", sorted(PER_LEVEL))
def test_each_level_costs_the_pinned_ands_and_rows(A) -> None:
    one, two = _counts(A, 1), _counts(A, 2)
    assert (two[0] - one[0], two[1] - one[1]) == PER_LEVEL[A]


# --- the subtree read --------------------------------------------------------------------------------------------------------

def _hm96_tree(rng: random.Random, A: int, D: int) -> tuple[list[bytes], list[list[bytes]]]:
    """Random cells' commit strings and the levels of their tree of hm96-sha512 leaves."""
    bcs = [rng.randbytes(128) for _ in range(A ** D)]
    return bcs, _tree(rng, A, D, [M.leaf_ref(bc) for bc in bcs])


def test_a_cells_leaf_is_hm96s_tree_leaf() -> None:
    key = hm96.SHA512.default_key
    assert M.LEAF_PREFIX == hm96.SHA512.leaf_prefix(key)
    for bc in (bytes(128), os.urandom(128)):
        assert M.leaf_ref(bc) == hm96.SHA512.tree_leaf(key, bc)


def test_the_reference_opens_every_run_of_a_tree() -> None:
    rng = random.Random(2)
    for A, D in ((2, 4), (4, 2)):
        bcs, levels = _hm96_tree(rng, A, D)
        root = levels[-1][0]
        for J in range(D):
            R = A ** J
            for p in range(A ** (D - J)):
                run, path = bcs[p * R:(p + 1) * R], _path(levels[J:], A, p)
                assert M.opens_run(A, root, run, p, path) == 1
                assert M.opens_run(A, root, run, (p + 1) % A ** (D - J), path) == 0
                if J == 0:
                    assert M.opens(A, root, M.leaf_ref(run[0]), p, path) == 1


def _run_cases(A: int, D: int, J: int, rng: random.Random) -> tuple[list, list[int]]:
    """Every run of a random tree opened honestly, then each read with one thing wrong: the root, a sibling, one bit of a cell,
    the position, and (a run of two cells or more) its first two cells swapped."""
    bcs, levels = _hm96_tree(rng, A, D)
    root, R, rows, want = levels[-1][0], A ** J, [], []
    for p in range(A ** (D - J)):
        run, path = bcs[p * R:(p + 1) * R], _path(levels[J:], A, p)
        bad = [list(s) for s in path]
        bad[rng.randrange(D - J)][rng.randrange(A - 1)] = rng.randbytes(64)
        flipped, i = list(run), rng.randrange(R)
        flipped[i] = bytes([run[i][0] ^ 1]) + run[i][1:]
        cases = [(root, run, p, path), (rng.randbytes(64), run, p, path), (root, run, p, bad), (root, flipped, p, path),
                 (root, run, p ^ 1, path)]
        if R > 1:
            cases.append((root, [run[1], run[0], *run[2:]], p, path))
        for args in cases:
            rows.append(M.run_arguments(A, *args))
            want.append(M.opens_run(A, *args))
    return rows, want


@pytest.mark.slow
@pytest.mark.parametrize("shape", [(2, 1, 0), (2, 2, 1), (4, 1, 0), (4, 2, 1)], ids=str)
def test_subtree_reads_agree_with_the_reference(shape) -> None:
    A, D, J = shape
    rows, want = _run_cases(A, D, J, random.Random(f"subtree{shape}"))
    n = A ** (D - J)
    assert want.count(1) == n and want.count(0) == (5 if J else 4) * n
    got = np.asarray(E.evaluate_batch(_subtree(*shape), *_columns(rows))).reshape(-1).tolist()
    assert got == want


@pytest.mark.slow
@pytest.mark.parametrize("shape", sorted(SUBTREE_PINS), ids=str)
def test_the_subtree_read_is_a_function_of_its_shape(shape) -> None:
    ands, xors, nots, digest = SUBTREE_PINS[shape]
    target = _subtree(*shape)
    assert BL.is_boolean(target) and target is _subtree(*shape)
    prog, _ = standalone_call(target)
    got = (M.traced_subtree(*shape).counts, codec.program_digest(codec.encode_program(prog)))
    assert got == ({"and": ands, "xor": xors, "not": nots}, digest)
    A, D, J = shape
    n = D - J
    assert [p.leaves for _, p in target.signature()[0]] == [512, 1024 * A ** J, (A.bit_length() - 1) * n, 512 * (A - 1) * n]


def _cost(A: int, D: int, J: int) -> tuple[int, int]:
    """(ANDs, rows) of a read of a run of A^J cells in a depth-D tree: its leaves, its run's nodes, its levels and the comparison."""
    R = A ** J
    parts = [(R, LEAF), ((R - 1) // (A - 1), RUN_NODE[A]), (D - J, PER_LEVEL[A]), (1, COMPARE)]
    return sum(n * c[0] for n, c in parts), sum(n * c[1] for n, c in parts)


@pytest.mark.slow
@pytest.mark.parametrize("shape", [(2, 1, 0), (2, 2, 1), (4, 1, 0), (4, 2, 1)], ids=str)
def test_a_subtree_read_costs_its_leaves_run_nodes_and_levels(shape) -> None:
    assert _counts(*shape) == _cost(*shape)


def test_malformed_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="ARITY is one of"):
        _read(3, 2).signature()
    with pytest.raises(ValueError, match="D is an int >= 1"):
        _read(2, 0).signature()
    with pytest.raises(ValueError, match="siblings a level"):
        M.arguments(4, bytes(64), bytes(64), 0, [[bytes(64)] * 2])
    with pytest.raises(ValueError, match="64 bytes"):
        M.arguments(2, bytes(63), bytes(64), 0, [[bytes(64)]])
    with pytest.raises(ValueError, match="0 <= J < D"):
        _subtree(2, 2, 2).signature()
    with pytest.raises(ValueError, match="power of 2 cells"):
        M.run_arguments(2, bytes(64), [bytes(128)] * 3, 0, [[bytes(64)]])
    with pytest.raises(ValueError, match="128 bytes"):
        M.run_arguments(2, bytes(64), [bytes(127)], 0, [[bytes(64)]])
