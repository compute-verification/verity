"""`verity.ml.boolean.comb`: a hidden unit circuit's lincheck share. The references against the verifier's dense fold, the
builders against the references (bit-sliced), and the traced Definitions through the IR evaluator."""
from __future__ import annotations

import random

import pytest

from verity.evaluation import evaluate
from verity.ml.boolean import comb as K
from verity.ml.boolean.forms import Circuit


def _point(rng: random.Random, SL: int) -> K.Point:
    g = rng.getrandbits
    return K.Point(g(128), [g(128) for _ in range(64)], [g(128) for _ in range(SL - 6)], [g(128) for _ in range(SL - 6)])


def _bits(v: int, n: int) -> list[int]:
    return [(v >> j) & 1 for j in range(n)]


def _elements(out, reps: int) -> list[int]:
    return [sum(out[128 * r + k] << k for k in range(128)) for r in range(reps)]


def _matrix(rng: random.Random, SL: int, rows: int, COL: int):
    ra = [[rng.randrange(1 << SL) for _ in range(rng.randrange(6))] for _ in range(rows)]
    rb = [[rng.randrange(1 << SL) for _ in range(rng.randrange(6))] for _ in range(rows)]
    for r in ra[::3]:
        r.append(COL)
    for r in rb[::5]:
        r.append(COL)
    ra[1] += [7, 7]  # a column read twice cancels
    return ra, rb


def test_entry_value_and_values_agree():
    rng = random.Random(1)
    for SL in (7, 9):
        ps = [_point(rng, SL) for _ in range(2)]
        es = [rng.getrandbits(K.entry_bits(SL)) for _ in range(40)] + [0, 0b00 | 5 << 2]
        want = [0, 0]
        for e in es:
            want = [w ^ K.entry_value(p, e) for w, p in zip(want, ps)]
        assert K.values(ps, es) == want


@pytest.mark.parametrize("blocks", [1, 2])
def test_share_is_the_dense_fold(blocks):
    """Entries (bucketed, padded, packed and read back) plus the dense column give the verifier's fold of the matrices."""
    rng = random.Random(2)
    SL, COL = 8, 255
    ra, rb = _matrix(rng, SL, 220, COL)
    buckets, a, b = K.entries(ra, rb, SL, COL)
    E = K.per_row(SL, blocks)
    sh = K.Shape(SL, -(-max(map(len, buckets)) // E) + 1, COL, blocks)
    rows = K.pack(sh, buckets, a, b)
    assert len(rows) == sh.rows and all(len(r) == sh.words for r in rows)
    got_b, got_a, got_bb = K.unpack(sh, rows)
    assert [[e for e in x if e & 3] for x in got_b] == buckets and (got_a, got_bb) == (a, b)
    for p in (_point(rng, SL), _point(rng, SL)):
        sums = []
        for s in range(64):
            acc = [0]
            for r in rows[s * sh.R:(s + 1) * sh.R]:
                acc = K.chunk([p], sh.E, acc, r[:K.row_words(SL, sh.E)])
            sums.append(acc[0])
        share = K.template_fold(p, sums, K.column([p], COL, a, b)[0], COL)
        assert share == K.fold_dense(p.alpha, 0, p.lw, p.rho, list(reversed(p.rr)), ra, rb, 0)


def test_local_point_is_the_ranges_weight():
    """A unit circuit in two slots of a block of k_log SL + 1 (no Δ): its share at `local_point` is the block's dense fold."""
    rng = random.Random(3)
    SL, COL = 7, 127
    ra, rb = _matrix(rng, SL, 100, COL)
    n = 1 << SL
    block_a = ra + [[] for _ in range(n - len(ra))] + [[c + n for c in r] for r in ra]
    block_b = rb + [[] for _ in range(n - len(rb))] + [[c + n for c in r] for r in rb]
    big = _point(rng, SL + 1)
    ts = list(reversed(big.rr))
    p = K.local_point(big.alpha, big.lw, big.rho, ts, SL, [(0, 2)])
    buckets, a, b = K.entries(ra, rb, SL, COL)
    share = K.template_fold(p, [K.values([p], x)[0] for x in buckets], K.column([p], COL, a, b)[0], COL)
    assert share == K.fold_dense(big.alpha, 0, big.lw, big.rho, ts, block_a, block_b, 0)


def test_pack_refuses_a_bucket_over_its_bound():
    sh = K.Shape(7, 1, None, 1, 2)
    with pytest.raises(ValueError, match="over the bound 2"):
        K.pack(sh, [[1, 1, 1]] + [[]] * 63)
    with pytest.raises(ValueError, match="holds no column bits"):
        K.pack(sh, [[]] * 64, [1] + [0] * 127, [0] * 128)


@pytest.mark.parametrize("SL,reps", [(7, 1), (8, 2), (9, 2)])
def test_entry_builder_is_the_reference(SL, reps):
    """Bit-sliced over 64 random entries at every commit spacing; with no commit the live ANDs are `and_count`."""
    rng = random.Random(4 + SL)
    ps = [_point(rng, SL) for _ in range(reps)]
    W, T = K.entry_bits(SL), 64
    es = [rng.getrandbits(W) for _ in range(T)]
    for every in (1, 2, 1 << 30):
        C = Circuit()
        out = K.build_entry(C, SL, ps, C.input("e", W), every)
        C.output("o", [x for X in out for x in X])
        o, _ = C.evaluate({"e": [sum(((e >> j) & 1) << t for t, e in enumerate(es)) for j in range(W)]}, T)
        for t, e in enumerate(es):
            assert _elements([(x >> t) & 1 for x in o["o"]], reps) == [K.entry_value(p, e) for p in ps]
        if every == 1 << 30:
            assert C.counts()["ands"] == K.and_count(SL, reps)


def test_and_count_formula():
    assert (K.low_bits(13, 2), K.and_count(13, 2)) == (7, 3702)
    assert (K.low_bits(13, 1), K.and_count(13, 1)) == (6, 1975)
    assert K.and_count(16, 2) == 5238


def test_point_static_round_trip():
    rng = random.Random(5)
    ps = (_point(rng, 7), _point(rng, 7))
    assert K.points(K.point_static(ps)) == ps
    with pytest.raises(ValueError, match="all of one slot log"):
        K.point_static([_point(rng, 7), _point(rng, 8)])


@pytest.mark.slow
def test_definitions_evaluate_as_the_references():
    """CombEntry, CombChunk and CombColumn through the IR evaluator, and the chunk and column circuits, at SL = 7, two reps."""
    rng = random.Random(6)
    SL, reps, E = 7, 2, 2
    ps = [_point(rng, SL) for _ in range(reps)]
    W = K.entry_bits(SL)
    e = rng.getrandbits(W)
    assert _elements(evaluate(K.entry_definition(SL, ps), _bits(e, W)), reps) == [K.entry_value(p, e) for p in ps]

    acc = [rng.getrandbits(128) for _ in ps]
    words = [rng.getrandbits(16) for _ in range(K.row_words(SL, E))]
    want = K.chunk(ps, E, acc, words)
    got = evaluate(K.chunk_definition(SL, E, ps), [x for v in acc for x in _bits(v, 128)], [x for w in words for x in _bits(w, 16)])
    assert _elements(got, reps) == want
    C = K.chunk_circuit(SL, E, ps)
    o, _ = C.evaluate({f"in{j}": _bits(w, 16) for j, w in enumerate([x for v in acc for x in K.f128_words(v)] + words)}, 1)
    assert [K.f128_of([sum(o[f"out{8 * r + w}"][j] << j for j in range(16)) for w in range(8)]) for r in range(reps)] == want

    COL = 100
    a, b = [rng.getrandbits(1) for _ in range(1 << SL)], [rng.getrandbits(1) for _ in range(1 << SL)]
    want = K.column(ps, COL, a, b)
    assert _elements(evaluate(K.column_definition(SL, COL, ps), a, b), reps) == want
    C = K.column_circuit(SL, COL, ps)
    o, _ = C.evaluate({f"in{j}": (a + b)[16 * j:16 * j + 16] for j in range(1 << (SL - 3))}, 1)
    assert [K.f128_of([sum(o[f"out{8 * r + w}"][j] << j for j in range(16)) for w in range(8)]) for r in range(reps)] == want


def test_a_changed_entry_changes_the_chunk():
    rng = random.Random(7)
    SL, E = 8, 4
    ps = [_point(rng, SL) for _ in range(2)]
    W = K.entry_bits(SL)
    es = [1 | rng.getrandbits(W - 2) << 2 for _ in range(E)]
    words = K.words_of([(e >> j) & 1 for e in es for j in range(W)], K.row_words(SL, E))
    base = K.chunk(ps, E, [0, 0], words)
    for bit in range(E * W):
        flipped = list(words)
        flipped[bit // 16] ^= 1 << (bit % 16)
        assert K.chunk(ps, E, [0, 0], flipped) != base
