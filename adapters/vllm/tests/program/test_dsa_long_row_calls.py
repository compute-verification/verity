"""The long row's units written as Calls (`dsa_topk_reads.DsaPositionsMerge_v2`, `boolean_dsa_topk_mask.DsaRadixTies_v2` and
the mask units that call it): each has its traced version's signature and is that version and its reference bit for bit; the
merge has the traced one's ANDs count for count and the ties no more from a few dozen keys on; the pieces are the steps they
name; at the served statics a description is a few rows of Calls and the unit fits a Flock block; and the pieces lower by
their builders to the circuit their gates lower to."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import random

import numpy as np
import pytest

from verity.circuits.codec import canonical_json, encode_program
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.program import Program
from verity.definitions.registry import boolean_dsa_topk_mask as BM
from verity.definitions.registry import dsa_topk_mask_cut as X
from verity.definitions.registry import dsa_topk_reads as R

#: the long row's merge as served: the longest row's 8 blocks, TOPK slots
SERVED = dict(M=8, K=2048)
PIECES = (R.DsaPositionsKeep, R.DsaPositionsShift, R.DsaPositionsShiftIn, R.DsaPositionsFirstOffset, R.DsaPositionsOffset)
TIES_PIECES = (BM.DsaCountSum, BM.DsaCountSplit, BM.DsaRadixTie)


def _bits(words, width: int = 32) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint64)
    return ((w[..., None] >> np.arange(width, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8).reshape(len(w), -1)


def _words(bits, width: int = 32) -> list[list[int]]:
    b = np.asarray(bits, dtype=np.uint64)
    b = b.reshape(len(b), -1, width)
    return (b << np.arange(width, dtype=np.uint64)).sum(axis=2).tolist()


def _lists(M: int, K: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """``rows`` instances of the merge's operands: lists with words past their counts, and counts at K's edges, past K and
    past 2^31, then random up to K + 1, so that lists fill K and overflow it."""
    rng = random.Random(M * 100 + K)
    edge = [0, 1, K - 1, K, K + 1, 1 << 31, (1 << 32) - 1]
    lists = np.array([[[rng.getrandbits(32) for _ in range(K)] for _ in range(M)] for _ in range(rows)], dtype=np.uint64)
    counts = np.array([[edge[(r + b) % len(edge)] if r < 2 * len(edge) else rng.randrange(K + 2) for b in range(M)]
                       for r in range(rows)], dtype=np.uint64)
    return lists, counts


@pytest.mark.parametrize("M,K", [(1, 4), (2, 4), (3, 4), (3, 5), (2, 1), (4, 3), (8, 16), (3, 64)])
def test_the_merge_as_calls_is_the_traced_merge(M: int, K: int) -> None:
    """`DsaPositionsMerge_v2` has `_v1`'s signature, its function and `merge`'s (the reference: the merge has no word view),
    and its ANDs, count for count.  A K that is a power of two takes its last stage whole from `DsaPositionsShiftIn`."""
    calls, traced = bind(R.DsaPositionsMergeV2, M=M, K=K), bind(R.DsaPositionsMerge, M=M, K=K)
    assert calls.signature() == traced.signature()
    lists, counts = _lists(M, K, 48)
    args = [*(_bits(lists[:, b]) for b in range(M)), *(_bits(counts[:, b]) for b in range(M))]
    want = [R.merge(li, c, K) for li, c in zip(lists.tolist(), counts.tolist())]
    for unit in (calls, traced):
        assert _words(BS.evaluate_bits(unit, *args)) == want, unit.id
    assert BS.compile(calls).counts["and"] == BS.compile(traced).counts["and"]


@pytest.mark.parametrize("K", [1, 4, 5, 2048])
def test_the_merges_pieces_are_its_steps(K: int) -> None:
    """A kept slot, a slot past a stage of the shift with and without a slot below it, and the offset past the first block
    and past each next one, on random words and on counts and offsets at K's edges."""
    rng = random.Random(K)
    n, RB = 64, K.bit_length()
    a, w = ([rng.getrandbits(32) for _ in range(n)] for _ in range(2))
    s = [rng.getrandbits(1) for _ in range(n)]
    one = lambda d, *args: [r[0] for r in _words(BS.evaluate_bits(bind(d), *args))]  # noqa: E731
    assert one(R.DsaPositionsKeep, _bits(s, 1), _bits(w)) == [x if k else 0 for k, x in zip(s, w)]
    assert one(R.DsaPositionsShift, _bits(s, 1), _bits(a), _bits(w)) == [x if k else y for k, x, y in zip(s, a, w)]
    assert one(R.DsaPositionsShiftIn, _bits(s, 1), _bits(w)) == [0 if k else y for k, y in zip(s, w)]
    edge = [0, 1, K - 1, K, K + 1, 1 << 31, (1 << 32) - 1]
    counts = [edge[r % len(edge)] if r < 2 * len(edge) else rng.randrange(K + 2) for r in range(n)]
    offs = [(0, K, K - 1, 1)[r % 4] if r < 28 else rng.randrange(K + 1) for r in range(n)]
    first = _words(BS.evaluate_bits(bind(R.DsaPositionsFirstOffset, K=K), _bits(counts)), RB)
    assert [r[0] for r in first] == [min(c, K) for c in counts]
    nxt = _words(BS.evaluate_bits(bind(R.DsaPositionsOffset, K=K), _bits(offs, RB), _bits(counts)), RB)
    assert [r[0] for r in nxt] == [min(o + min(c, K), K) for o, c in zip(offs, counts)]


def test_the_merge_as_calls_stays_small_at_the_served_statics() -> None:
    """At the served statics the merge written as Calls is a few rows of Calls a block: its Program's description is under a
    byte for every 64 of its gates (the traced merge's holds every gate; what grows with K here is the keep bits' trie,
    `DsaRowValid_v1{K}`, an AND a slot), and its Flock block, read off the Definition as circuit-check's ``block`` reads
    it, is within the proved range."""
    from verity_flock import circuit as FC

    unit = bind(R.DsaPositionsMergeV2, **SERVED)
    assert len(canonical_json(encode_program(Program(unit)))) * 64 < unit.gates
    b = FC.unit_block(unit)
    assert FC.past_block(b) is None, (unit.id, b["k_log"])


@pytest.mark.parametrize("M,K", [(1, 4), (3, 4), (8, 16)])
def test_the_merge_as_calls_has_the_traced_merges_block(M: int, K: int) -> None:
    """The Flock block of the merge written as Calls is the traced one's: the same rows in and out, the same ANDs."""
    from verity_flock import circuit as FC

    assert FC.unit_block(bind(R.DsaPositionsMergeV2, M=M, K=K)) == FC.unit_block(bind(R.DsaPositionsMerge, M=M, K=K))


def _lowered(unit) -> tuple[list, list[str]]:
    """Flock's flat lowering of ``unit`` (`ir_lower.lower`, as a served unit's stager runs it): each committed bit and the
    output's forms, and the name of each activation it lowered by a builder."""
    from verity.circuits.evaluation.reference import signature_of
    from verity_flock import ir_lower as IL
    from verity_flock.class_statement import _UnitCircuit

    C, built = _UnitCircuit(), []
    activation = IL._lower_activation

    def counted(C, s, *rest):
        built.append(s.spec.id.split("_v")[0])
        return activation(C, s, *rest)

    params, _ = signature_of(unit)
    ins = [C.input(f"in{j}", w) for j, w in enumerate(w for t in params for w in t.leaf_widths())]
    IL._lower_activation = counted
    try:
        out = [b for w in IL.lower(C, unit, ins) for b in w]
    finally:
        IL._lower_activation = activation
    bits = [(k, (tuple(d[0].s), d[0].c, tuple(d[1].s), d[1].c) if k == "and" else d) for k, d in zip(C.kind, C.data)]
    return [bits, [(tuple(x.s), x.c) for x in out]], built


def test_the_merges_builders_lower_to_their_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each piece of the merge carries a builder, so Flock's flat lowering runs each of its activations and keeps no word a
    gate of it.  The circuit is the one its gates lower to, AND for AND: each piece alone, and a merge of 3 lists of 5 slots
    (its builders read their operands from the other Calls' wires, the keep bits' and the XORs' among them)."""
    M, K = 3, 5
    units = [bind(R.DsaPositionsKeep), bind(R.DsaPositionsShift), bind(R.DsaPositionsShiftIn),
             bind(R.DsaPositionsFirstOffset, K=K), bind(R.DsaPositionsOffset, K=K), bind(R.DsaPositionsMergeV2, M=M, K=K)]
    names = {p.name for p in PIECES}
    with_builders = [_lowered(u) for u in units]
    assert [sum(n in names for n in built) for _, built in with_builders] == [1, 1, 1, 1, 1,
                                                                             M * K + (M - 1) * K.bit_length() * K + M - 1]
    for piece in PIECES:
        monkeypatch.delattr(piece, "build")
    for unit, (circuit, _) in zip(units, with_builders, strict=True):
        by_gates, built = _lowered(unit)
        assert not names & set(built) and by_gates == circuit, unit.id


def _ties_rows(N: int, TOPK: int, rows: int) -> tuple[list[np.ndarray], list[list[int]]]:
    """``rows`` instances of the ties' operands and the mask's rule on each: eq sparse, dense and all ones, gt apart from eq
    as the select leaves it and (every other row) any bits, r at 0, 1, TOPK and at random."""
    rng = random.Random(N * 1000 + TOPK)
    RB = TOPK.bit_length()
    eqs, gts, rs, want = [], [], [], []
    for t in range(rows):
        p = (0.1, 0.5, 0.9, 1.0)[t % 4]
        eq = [int(rng.random() < p) for _ in range(N)]
        gt = [0 if q else rng.getrandbits(1) for q in eq] if t % 2 else [rng.getrandbits(1) for _ in range(N)]
        r = (0, 1, TOPK)[t % 3] if t < 9 else rng.randrange(1 << RB)
        eqs.append(eq), gts.append(gt), rs.append([r >> i & 1 for i in range(RB)])
        before = np.cumsum(eq) - np.asarray(eq)
        want.append([g ^ (q & int(c < r)) for g, q, c in zip(gt, eq, before)])
    return [np.array(x, dtype=np.uint8) for x in (eqs, gts, rs)], want


@pytest.mark.parametrize("N,TOPK", [(1, 1), (2, 1), (2, 2), (3, 2), (5, 3), (8, 4), (13, 13), (16, 8), (20, 16), (33, 8),
                                    (100, 16), (257, 64), (1000, 100)])
def test_the_ties_as_calls_are_the_traced_ties(N: int, TOPK: int) -> None:
    """`DsaRadixTies_v2` has `_v1`'s signature and its function, the mask's rule (the reference: the ties have no word view):
    a key above the threshold, or equal to it with fewer than r equal keys before it.  From 33 keys on it has no more ANDs
    than `_v1`; below that the traced one folds an AND or two more at a row length that is a power of two."""
    calls, traced = bind(BM.DsaRadixTiesV2, N=N, TOPK=TOPK), bind(BM.DsaRadixTies, N=N, TOPK=TOPK)
    assert calls.signature() == traced.signature()
    args, want = _ties_rows(N, TOPK, 40)
    for unit in (calls, traced):
        assert np.asarray(BS.evaluate_bits(unit, *args)).reshape(len(want), -1).tolist() == want, unit.id
    if N >= 33:
        assert BS.compile(calls).counts["and"] <= BS.compile(traced).counts["and"]


def test_the_ties_pieces_are_its_steps() -> None:
    """The sum of two counts, a count handed down to a node's two halves, and a key's tie with and without a count before
    it, on random operands at widths the served rows reach."""
    rng = random.Random(5)
    n = 64
    for w in (1, 3, 15):
        a, b = ([rng.randrange(1 << w) for _ in range(n)] for _ in range(2))
        got = _words(BS.evaluate_bits(bind(BM.DsaCountSum, W=w), _bits(a, w), _bits(b, w)), w + 1)
        assert [r[0] for r in got] == [x + y for x, y in zip(a, b)]
    for w, u in ((2, 1), (5, 3), (16, 15), (16, 1)):
        p, x = [rng.randrange(1 << w) for _ in range(n)], [rng.randrange(1 << u) for _ in range(n)]
        got = _words(BS.evaluate_bits(bind(BM.DsaCountSplit, W=w, U=u), _bits(p, w), _bits(x, u)), w)
        assert got == [[c, (c + y) % (1 << w)] for c, y in zip(p, x)]
    for w, rb in ((0, 3), (1, 12), (12, 12), (16, 12), (3, 5)):
        g, q = ([rng.getrandbits(1) for _ in range(n)] for _ in range(2))
        c, r = [rng.randrange(1 << w) for _ in range(n)], [rng.randrange(1 << rb) for _ in range(n)]
        args = [_bits(g, 1), _bits(q, 1), *([_bits(c, w)] if w else []), _bits(r, rb)]
        got = np.asarray(BS.evaluate_bits(bind(BM.DsaRadixTie, W=w, RB=rb), *args)).reshape(-1).tolist()
        assert got == [x ^ (y & int(k < z)) for x, y, k, z in zip(g, q, c, r)]


@pytest.mark.parametrize("N,AT", [(8, 0), (6, 8), (40, 16)])
def test_the_mask_units_as_calls_are_the_traced_units(N: int, AT: int) -> None:
    """`DsaTopKMaskBlock_v2` and `DsaTopKMaskWhole_v2` have their `_v1`'s signature and its mask, on rows of few distinct
    scores (so keys tie at the threshold), n inside the block, at its edges and past it, and r from 0 up."""
    TOPK, rows = 4, 40
    rng = random.Random(N + AT)
    pool = [rng.getrandbits(32) for _ in range(3)]
    scores = np.array([[rng.choice(pool) for _ in range(N)] for _ in range(rows)], dtype=np.uint64)
    n = np.array([(0, AT, AT + 1, AT + N, AT + N + 3)[t % 5] if t < 10 else AT + rng.randrange(N + 1) for t in range(rows)])
    theta = np.array([int(X.keys_of([rng.choice(pool)])[0]) for _ in range(rows)], dtype=np.uint64)
    r = np.array([rng.randrange(TOPK + 2) for _ in range(rows)], dtype=np.uint64)
    block = [_bits(scores.reshape(-1)).reshape(rows, -1), _bits(n), _bits(theta), _bits(r, X.wide(TOPK.bit_length()))]
    calls, traced = (bind(d, N=N, AT=AT, TOPK=TOPK) for d in (X.DsaTopKMaskBlockV2, X.DsaTopKMaskBlock))
    assert calls.signature() == traced.signature()
    got = np.asarray(BS.evaluate_bits(traced, *block))
    assert got.any() and np.array_equal(np.asarray(BS.evaluate_bits(calls, *block)), got)
    if AT == 0:
        calls, traced = (bind(d, N=N, TOPK=TOPK) for d in (X.DsaTopKMaskWholeV2, X.DsaTopKMaskWhole))
        assert calls.signature() == traced.signature()
        got = np.asarray(BS.evaluate_bits(traced, *block[:2]))
        assert got.any() and np.array_equal(np.asarray(BS.evaluate_bits(calls, *block[:2])), got)


@pytest.mark.parametrize("N", [X.BLOCK, X.WHOLE_MAX])
def test_the_ties_as_calls_stay_small_at_the_served_statics(N: int) -> None:
    """At the long row's served statics (a block of the cut, and the longest row masked whole) the ties written as Calls are
    a few batches a level of the count's tree and a piece for each width a count takes: a description under a mebibyte and
    under a byte for every 3 of the gates, where the traced ties' holds every gate."""
    unit = bind(BM.DsaRadixTiesV2, N=N, TOPK=2048)
    size = len(canonical_json(encode_program(Program(unit))))
    assert size < 1 << 20 and size * 3 < unit.gates


def test_the_ties_builders_lower_to_their_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each piece of the ties carries a builder, as the merge's do: the circuit Flock's flat lowering makes of the ties over
    13 keys, and of a mask unit over them, is the one their gates lower to, AND for AND."""
    units = [bind(BM.DsaRadixTiesV2, N=13, TOPK=4), bind(X.DsaTopKMaskBlockV2, N=13, AT=16, TOPK=4)]
    names = {p.name for p in TIES_PIECES}
    with_builders = [_lowered(u) for u in units]
    assert all(names & set(built) for _, built in with_builders)
    for piece in TIES_PIECES:
        monkeypatch.delattr(piece, "build")
    for unit, (circuit, _) in zip(units, with_builders, strict=True):
        by_gates, built = _lowered(unit)
        assert not names & set(built) and by_gates == circuit, unit.id
