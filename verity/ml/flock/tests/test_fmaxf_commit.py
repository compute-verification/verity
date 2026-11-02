"""The `F32Fmaxf_v1` and `F32Fminf_v1` pieces commit their result (`tail_pieces`): each result bit is a column whose row is
``(A_j · z) · z_const = z_j``, ``A_j`` the mux form the piece's own comparator and select compute and ``z_const`` the
statement's pinned constant 1, so ``z_j`` is that form and nothing else. A K-step fold of them (`Int8RowScale`'s amax) reads
32 committed columns a step: it lowers to today's outputs, its rows stay linear in K, and a witness whose committed max is
anything else fails that row, however every other column is chosen."""
from __future__ import annotations

import random
from functools import reduce
from operator import xor

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops.boolean.scalar import fmaxmin
from verity.ml.circuits.ops.scalar import F32Fmaxf, F32Fminf
from verity.ml.definitions.registry import boolean_int8  # noqa: F401  (registers the Boolean F32Rint and F32ToS8Rzi the quantizer lowers through)
from verity.ml.definitions.registry import int8_exact as X
from verity_flock import class_statement as CS
from verity_flock import gf2
from verity_flock import ir_lower as IL

PIECES = {"F32Fmaxf_v1": (F32Fmaxf, True), "F32Fminf_v1": (F32Fminf, False)}
F32_SPECIAL = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFBFFFFF, 1, 0x80000001, 0x7FFFFF,
               0x800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000]
#: NaNs (quiet, signalling, both signs), ±inf, ±0, subnormals, the smallest normal and the largest finite
BF16_SPECIAL = [0x7FC0, 0xFFC0, 0x7F81, 0xFFFF, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x7F7F,
                0xFF7F]


def witness(lay: IL.LoweredUnit, ports, T: int, tamper: tuple[int, int] | None = None) -> tuple[list[int], list[int]]:
    """The witness the rows compute from ``ports`` over T lanes (bit-sliced), with ``tamper = (column, lane)``'s bit flipped as
    it is computed, so every later row reads the tampered value: a prover that sets that column freely and every other one
    honestly from it. Returns the witness and, per row, the lanes whose constraint ``(A·z)(B·z) = z`` fails."""
    mask = (1 << T) - 1
    z = [0] * lay.useful
    for at, width, words in zip(lay.in_port_cols, lay.in_bits, ports):
        z[at:at + width] = IL.slice_words(words, width)
    z[lay.const] = mask

    def ev(r):
        return reduce(xor, (z[k] for k in r), 0)

    for i in range(lay.in_words * IL.WORD, lay.useful):
        if i != lay.const:
            z[i] = ev(lay.rows_a[i]) & ev(lay.rows_b[i])
            if tamper is not None and i == tamper[0]:
                z[i] ^= 1 << tamper[1]
    return z, [((ev(a) & ev(b)) ^ z[i]) & mask for i, (a, b) in enumerate(zip(lay.rows_a, lay.rows_b))]


def outputs(lay: IL.LoweredUnit, z: list[int], T: int) -> list[np.ndarray]:
    return [IL.unslice_words(z[at:at + nb], T) for at, nb in zip(lay.out_port_cols, lay.out_bits)]


def piece_layout(name: str):
    C = gf2.Circuit()
    a, b = C.input("a", 32), C.input("b", 32)
    C.output("o", IL.PIECES[name](C, a, b))
    return C, IL.layout(C, ["a", "b"], ["o"])


def f32_pairs(seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    pairs = [(a, b) for a in F32_SPECIAL for b in F32_SPECIAL]
    pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(256)]
    pairs += [(a, a ^ rng.choice([1, 0x80000000, 1 << rng.randrange(31)])) for a in (rng.getrandbits(32) for _ in range(128))]
    return pairs


@pytest.mark.parametrize("name", sorted(PIECES))
def test_each_result_bit_is_a_commit_row_of_the_pieces_mux(name):
    """Every result bit is one column ``j`` with ``B_j = [const]`` and ``A_j`` a form of earlier columns that holds one of the
    piece's own AND gates (its select): the commit row ties ``z_j`` to what the gates compute."""
    _C, lay = piece_layout(name)
    first_and = lay.in_words * IL.WORD
    ands = set(range(first_and, first_and + lay.counts["ands"]))
    (at,), (nb,) = lay.out_port_cols, lay.out_bits
    for k in range(nb):
        (j,) = lay.rows_a[at + k]
        assert lay.rows_b[at + k] == [lay.const]
        assert j in ands and lay.rows_b[j] == [lay.const], (k, j)
        assert all(c < j for c in lay.rows_a[j]) and ands & set(lay.rows_a[j]) and len(lay.rows_a[j]) >= 2, (k, lay.rows_a[j])


@pytest.mark.parametrize("name", sorted(PIECES))
def test_the_committed_pieces_are_their_primitives(name):
    prim, _ = PIECES[name]
    _C, lay = piece_layout(name)
    pairs = f32_pairs(7)
    T = len(pairs)
    z, bad = witness(lay, [[a for a, _ in pairs], [b for _, b in pairs]], T)
    assert not any(bad)
    assert [int(w) for w in outputs(lay, z, T)[0]] == [prim.evaluate(a, b) for a, b in pairs]


@pytest.mark.parametrize("name", sorted(PIECES))
def test_a_tampered_committed_result_fails_its_commit_row(name):
    """A committed result bit set to the other value, every later row computed from it, fails its commit row in that lane and
    no other row or lane; flipped after the witness is computed (`evaluate_layout`'s ``flip``), the lane fails too."""
    _C, lay = piece_layout(name)
    pairs = f32_pairs(8)[::7][:64]
    T, ports = len(pairs), [[a for a, _ in pairs], [b for _, b in pairs]]
    honest, _ = witness(lay, ports, T)
    (at,), (nb,) = lay.out_port_cols, lay.out_bits
    for k in range(nb):
        (j,) = lay.rows_a[at + k]
        lane = (5 * k) % T
        z, bad = witness(lay, ports, T, tamper=(j, lane))
        assert z[j] == honest[j] ^ (1 << lane)
        assert {i: m for i, m in enumerate(bad) if m} == {j: 1 << lane}, k
        assert IL.evaluate_layout(lay, ports, T, flip=(j, lane))[1] == ((1 << T) - 1) & ~(1 << lane)


def row_quant(K: int, R: int, monkeypatch=None):
    """`Int8RowQuant_v3{K,R}`'s unit on its one row port (x, K u16 words) and its circuit; with ``monkeypatch``, lowered through
    the uncommitted pieces (the circuit before the commit)."""
    if monkeypatch is not None:
        for name, (_, is_max) in PIECES.items():
            monkeypatch.setitem(IL.PIECES, name, lambda C, a, b, m=is_max: fmaxmin(C, a, b, m))
    d = bind(X.Int8RowQuantV3, K=K, R=R)
    ports, _out_words, _rows = CS.unit_ports(d, "row-quant", [("x", ["x"])])
    C = CS.unit_circuit(d, ports, CS._UnitCircuit)
    lay = IL.layout(C, [f"in{j}" for j in range(len(ports.ins))], [f"out{o}" for o in range(len(ports.outs))])
    if monkeypatch is not None:
        monkeypatch.undo()
    return d, C, lay


def bf16_rows(K: int, T: int, seed: int) -> np.ndarray:
    """T rows of K bf16 words: a fifth of them specials, row 0 all NaN, row 1 all zero (the floored scale), the rest random."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 1 << 16, (T, K))
    sp = rng.random((T, K)) < 0.2
    x[sp] = rng.choice(BF16_SPECIAL, int(sp.sum()))
    x[0], x[1] = 0x7FC0, 0
    x[2] = rng.choice(BF16_SPECIAL, K)
    return x


def quant_words(outs: list[np.ndarray], K: int) -> np.ndarray:
    """The unit's u16 output ports as the reference's words: K s8 codes, two a word, low byte first, then s."""
    codes = [(outs[k // 2] >> np.uint64(8 * (k % 2))) & np.uint64(0xFF) for k in range(K)]
    return np.stack([*codes, outs[K // 2] | outs[K // 2 + 1] << np.uint64(16)], axis=1)


def test_the_committed_fold_is_todays_circuit_at_k16(monkeypatch):
    """`Int8RowQuant_v3{K=16,R=1}` on 256 rows with NaN, ±inf, ±0 and subnormal words: the committed unit's outputs are the
    uncommitted unit's and the reference's, and every row of both holds."""
    K, T = 16, 256
    d, _C, lay = row_quant(K, 1)
    _d, _C0, old = row_quant(K, 1, monkeypatch)
    x = bf16_rows(K, T, 16)
    ports = [x[:, j].tolist() for j in range(K)]
    z, bad = witness(lay, ports, T)
    z0, bad0 = witness(old, ports, T)
    assert not any(bad) and not any(bad0)
    got, was = quant_words(outputs(lay, z, T), K), quant_words(outputs(old, z0, T), K)
    want = np.asarray(evaluate_batch(d, x.reshape(T, 1, K).astype(np.uint64), kernel="reference"), dtype=np.uint64)
    assert (got == was).all() and (got == want).all()
    assert lay.counts["nnz_a"] < old.counts["nnz_a"]


def test_the_committed_fold_is_the_reference_across_rows_at_k64():
    """`Int8RowQuant_v3{K=64,R=4}`: 64 fold steps, each committed, the running max carried across x's four rows of 16."""
    K, R, T = 64, 4, 128
    d, _C, lay = row_quant(K, R)
    x = bf16_rows(K, T, 64)
    z, bad = witness(lay, [x[:, j].tolist() for j in range(K)], T)
    assert not any(bad)
    want = np.asarray(evaluate_batch(d, x.reshape(T, R, K // R).astype(np.uint64), kernel="reference"), dtype=np.uint64)
    assert (quant_words(outputs(lay, z, T), K) == want).all()


def test_the_committed_folds_nonzeros_are_linear_in_k():
    """Doubling K at most about doubles the unit's nonzeros (uncommitted, the running max's forms made them grow with K²)."""
    n16, n32 = (row_quant(K, 1)[2].counts for K in (16, 32))
    assert n32["nnz_a"] + n32["nnz_b"] < 2.1 * (n16["nnz_a"] + n16["nnz_b"])


def test_a_tampered_running_max_fails_its_commit_row(monkeypatch):
    """The fold's running max after step 8 of 16 in a row of rising words, its top exponent bit set (the max +inf) in that
    lane and every later column computed from it: the row's scale and codes move, and only that bit's commit row fails."""
    K, T, lane = 16, 64, 9
    calls: list[list] = []
    real = IL.PIECES["F32Fmaxf_v1"]

    def recording(C, a, b):
        out = real(C, a, b)
        calls.append(out)
        return out

    monkeypatch.setitem(IL.PIECES, "F32Fmaxf_v1", recording)
    _d, C, lay = row_quant(K, 1)
    monkeypatch.undo()
    assert len(calls) == K + 1 + K and lay.counts["reads"] == 0       # the fold, the floor, each code's clamp
    live = sorted(v for v in C.live() if C.kind[v] == "and")
    col = {v: lay.in_words * IL.WORD + i for i, v in enumerate(live)}
    (v,) = calls[7][30].s
    j = col[v]
    assert lay.rows_b[j] == [lay.const]
    x = bf16_rows(K, T, 3)
    x[lane] = np.arange(1, K + 1) * 0x10 + 0x3F00
    ports = [x[:, k].tolist() for k in range(K)]
    honest, _ = witness(lay, ports, T)
    assert not honest[j] >> lane & 1
    z, bad = witness(lay, ports, T, tamper=(j, lane))
    assert {i: m for i, m in enumerate(bad) if m} == {j: 1 << lane}
    moved = quant_words(outputs(lay, z, T), K) != quant_words(outputs(lay, honest, T), K)
    assert moved[lane].any() and not np.delete(moved, lane, axis=0).any()
