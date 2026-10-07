"""`verity.ml.commitments.gates.merkle`: the hidden-index Merkle read agrees with `opens` (``hashlib``) at every position of small binary
and 4-ary trees, refuses a wrong root, sibling, leaf or position, is a function of its shape alone (pinned by gate counts and
program digest), and costs per level the ANDs and C-Flock rows pinned here; `sha512.compress_ref` is SHA-512's compression."""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import bind
from verity.ml.commitments.gates import merkle as M
from verity.ml.commitments.gates import sha512 as S
from verity.ml.circuits.boolean.forms import ONE, Circuit

#: (ARITY, D) -> (And, Xor, Not gates, standalone program digest)
PINS = {
    (2, 1): (103101, 479098, 14170, "292bd91ed222d65237d273ed86f6cf1f5bc7237c282b12abbaf0df48bb84728f8b5804070b4cf078cd2c8e357122860c022298aba2bb87fdaa4f1e3f2a592e7b"),
    (4, 1): (163576, 769803, 19049, "b8c72007efe698f6d379de091be61af8d46829b0641b989b58f762f475e3662d9df9c5e5ebd147b6b39998624c83788e150b30c86bb9520e61d236ed73fc80f5"),
}
#: ARITY -> (ANDs, C-Flock rows) one level adds: the node's compressions and the slot multiplexers; rows are the ANDs and the
#: copies `sha512.commit` adds (an untraced `forms.Circuit`, no common-subexpression pass)
PER_LEVEL = {2: (102591, 142847), 4: (163066, 226774)}


def _read(A: int, D: int):
    return bind(M.MerkleRead, ARITY=A, D=D)


def _tree(rng: random.Random, A: int, D: int) -> list[list[bytes]]:
    levels = [[rng.randbytes(64) for _ in range(A ** D)]]
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


def _counts(A: int, D: int) -> tuple[int, int]:
    C = Circuit()
    ps, _ = M._types(A, D)
    els = [C.input(f"e{i}", w) for i, w in enumerate(w for t in ps for w in BL.elements(t))]
    C.output("ok", M.build(C, A, D, *els))
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


def test_malformed_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="ARITY is one of"):
        _read(3, 2).signature()
    with pytest.raises(ValueError, match="D is an int >= 1"):
        _read(2, 0).signature()
    with pytest.raises(ValueError, match="siblings a level"):
        M.arguments(4, bytes(64), bytes(64), 0, [[bytes(64)] * 2])
    with pytest.raises(ValueError, match="64 bytes"):
        M.arguments(2, bytes(63), bytes(64), 0, [[bytes(64)]])
