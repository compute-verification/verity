"""`verity.ml.commitments.gates.embedding`: the embedding's row read at a hidden position returns the token's row of a table
committed one token a row (`row_table.commit` over ``hm96-sha512/row-seg/v1`` rows, as `one_stage.registered` registers a
row-seg value) with ok = 1, and `Embedding_v1`'s 0x7FFF fill with ok = 1 for a token outside [0, V); it refuses (ok = 0)
another token's position, a changed row, salt, root, domain id or sibling; and its parts cost the gates pinned here."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments import row_table as RT
from verity.ml.commitments.gates import embedding as E
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth
from verity.ml.commitments.rowleaf import bit_row

#: (V, H) -> EmbeddingReadSelect_v1's (And, Xor, Not) gates: in by 23 ANDs (V = 128,256 has its low 8 bits clear), then
#: 16 H + 1
SELECT = {(128256, 2048): (32792, 0, 61464), (128256, 4096): (65560, 0, 122904)}
#: (V, H) -> the read's gates (`SpecializedDefinition.gates`): the size a unit holding it is cut to
READS = {(128256, 2048): 32322856, (128256, 4096): 45294886}


def _table(rng: random.Random, V: int, H: int) -> tuple[CommitmentDomain, list[list[int]], list[bytes], RT.RowTable]:
    """A table of V random rows of H bf16 words, one token a row, under a registered value's dense domain (owner -2)."""
    dom = CommitmentDomain(rng.randbytes(64), -2, RangeIndexedDomain(0, V), hash="sha512")
    rows = [[rng.randrange(1 << 16) for _ in range(H)] for _ in range(V)]
    salts = [rng.randbytes(E.SALT_BYTES) for _ in range(V)]
    return dom, rows, salts, RT.commit(dom, [bit_row((r, 16)) for r in rows], salts)


def _path(table: RT.RowTable, rank: int) -> list[bytes]:
    p = table.path(rank)
    return [p[k:k + 64] for k in range(0, len(p), 64)]


def _cases(rng: random.Random, V: int, H: int) -> list[tuple]:
    """Every token read honestly, then each read with one thing wrong (another in-range token, a row bit, the salt, the root,
    the domain id, a sibling), and tokens outside [0, V): V, the largest unsigned, a negative i32 and 2^31."""
    dom, rows, salts, table = _table(rng, V, H)
    D = merkle_depth(V)
    out = []
    for t in range(V):
        args = (t, table.root, dom.domain_id, rows[t], salts[t], _path(table, t))
        bad_row = list(rows[t])
        bad_row[rng.randrange(H)] ^= 1 << rng.randrange(16)
        bad_path = list(args[5])
        bad_path[rng.randrange(D)] = rng.randbytes(64)
        out += [args, ((t + 1) % V, *args[1:]), (t, args[1], args[2], bad_row, *args[4:]),
                (*args[:4], rng.randbytes(E.SALT_BYTES), args[5]), (t, rng.randbytes(64), *args[2:]),
                (t, args[1], rng.randbytes(64), *args[3:]), (*args[:5], bad_path)]
    for tok in (V, (1 << 32) - 1, -2, 1 << 31):
        out.append((tok, table.root, dom.domain_id, rows[0], salts[0], _path(table, 0)))
    return out


def _columns(rows) -> list[np.ndarray]:
    return [np.array([r[k] for r in rows], dtype=np.uint8) for k in range(len(rows[0]))]


def _evaluate(V: int, H: int, cases: list[tuple]) -> list[tuple[list[int], int]]:
    got = evaluate_bits(E.read(V, H), *_columns([E.arguments(V, *c) for c in cases])).reshape(len(cases), -1)
    return [([int(sum(int(b) << j for j, b in enumerate(g[16 * w:16 * w + 16]))) for w in range(H)], int(g[16 * H]))
            for g in got]


def test_the_read_is_typed_by_its_shape() -> None:
    for (V, H), D in (((2, 1), 1), ((5, 2), 3), ((128256, 2048), 17), ((128256, 4096), 17)):
        target = E.read(V, H)
        assert target is E.read(V, H)
        assert merkle_depth(V) == D
        assert [t.leaves for _, t in target.signature()[0]] == [32, 512, 512, 16 * H, 1536, 512 * D]
        assert target.signature()[1].leaves == 16 * H + 1
        assert BL.elements(target.signature()[1]) == [16] * H + [1]


def test_malformed_bindings_are_refused() -> None:
    for V in (0, 1, (1 << 31) + 1, "5", True):
        with pytest.raises(ValueError, match="V is an int in 2"):
            E.read(V, 2).signature()
    for H in (0, -1, "2"):
        with pytest.raises(ValueError, match="H is a positive int"):
            E.read(5, H).signature()
    with pytest.raises(ValueError, match="path of 3 siblings"):
        E.arguments(5, 0, bytes(64), bytes(64), [0, 0], bytes(192), [bytes(64)] * 2)
    with pytest.raises(ValueError, match="salt is 192 bytes"):
        E.arguments(5, 0, bytes(64), bytes(64), [0, 0], bytes(96), [bytes(64)] * 3)
    with pytest.raises(ValueError, match="16 bits"):
        E.arguments(5, 0, bytes(64), bytes(64), [0, 1 << 16], bytes(192), [bytes(64)] * 3)


def test_the_reference_opens_the_registered_row_and_refuses_every_wrong_read() -> None:
    """The reference on a 300-row table (D = 9, so the leaf's and the first node's uint take two bytes): each honest read is
    its row with ok = 1, each wrong one ok = 0, and a token outside [0, V) the fill with ok = 1."""
    rng = random.Random("embedding-read-reference")
    V, H = 300, 3
    cases = _cases(rng, V, H)
    got = [E.reference(V, *c) for c in cases]
    for t in range(V):
        honest, *wrong = got[7 * t:7 * t + 7]
        assert honest == (cases[7 * t][3], 1)
        assert [ok for _, ok in wrong] == [0] * 6
    assert got[7 * V:] == [([E.FILL] * H, 1)] * 4


def test_the_selection_is_embedding_v1s_on_every_token_edge() -> None:
    """The selection alone (a few hundred gates): the row at 0 and V - 1, the fill at V, 2^31 - 1, 2^31 and -1, at the deployed
    V; ok is opens inside the vocabulary and 1 outside it."""
    rng = random.Random("embedding-read-select")
    V, H = 128256, 3
    toks = [0, 1, V - 1, V, V + 1, (1 << 17) - 1, 1 << 17, (1 << 31) - 1, 1 << 31, (1 << 32) - 1]
    rows, want = [], []
    for t in toks:
        for opened in (0, 1):
            row = [rng.randrange(1 << 16) for _ in range(H)]
            rows.append(([(t >> j) & 1 for j in range(32)], [opened], [(w >> j) & 1 for w in row for j in range(16)]))
            inside = t < V
            want.append(([w if inside else E.FILL for w in row], opened if inside else 1))
    got = evaluate_bits(bind(E.EmbeddingReadSelect, V=V, H=H), *_columns(rows)).reshape(len(rows), -1)
    assert [([int(sum(int(b) << j for j, b in enumerate(g[16 * w:16 * w + 16]))) for w in range(H)], int(g[16 * H]))
            for g in got] == want


@pytest.mark.slow
def test_reads_open_a_registered_table_and_refuse_every_wrong_read() -> None:
    """A few minutes, most of it tracing the hm96 leaf and the read's parts: the only evaluation of the wiring (the row's bits
    into its digest, the digest and salt into the hm96 leaf, its leaf and the token's low D bits into the read, the read's bit
    into the selection), at D = 3 over two-word rows."""
    rng = random.Random("embedding-read-gates")
    V, H = 5, 2
    cases = _cases(rng, V, H)
    want = [E.reference(V, *c) for c in cases]
    assert [ok for _, ok in want].count(1) == V + 4
    assert _evaluate(V, H, cases) == want


@pytest.mark.slow
def test_the_read_costs_the_pinned_gates() -> None:
    """Several minutes of tracing the deployed parts (a full row-seg segment, the outer hashes, the hm96 leaf and the read's
    nodes): the sizes a unit holding the read is cut to at Llama-3.2-1B's and Llama-3.1-8B's embedding."""
    for (V, H), counts in SELECT.items():
        assert E.EmbeddingReadSelect.traced(V, H).counts == dict(zip(("and", "xor", "not"), counts)), (V, H)
    for (V, H), gates in READS.items():
        assert E.read(V, H).gates == gates, (V, H)
