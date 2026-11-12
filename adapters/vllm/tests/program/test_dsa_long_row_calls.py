"""The long row's units written as Calls (`dsa_topk_reads.DsaPositionsMerge_v2`, `boolean_dsa_topk_mask.DsaRadixTies_v2`,
`DsaRadixPass_v2` and `DsaRowValid_v2` and the mask units that call them, `dsa_topk_mask_cut.DsaTopKDigitPick_v2`,
`dsa_topk_reads.DsaMaskPositions_v2`): each has its traced version's signature and is that version and its reference bit for
bit; the merge, the pick and the valid bits have the traced one's ANDs count for count, the ties and the positions no more
from a few dozen keys on, and the pass a few more past a chunk of keys; the pieces are the steps they name; at the served
statics a description is a few rows of Calls and the unit fits a Flock block; and the pieces lower by their builders to the
circuit their gates lower to, a count sum's with the rows more that commit its sum."""
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
POSITION_PIECES = (BM.DsaCountSum, BM.DsaCountSplit, BM.DsaButterflyTag, R.DsaBitAppend, R.DsaPositionWord)
SELECT_PIECES = (BM.DsaRadixPassCount, BM.DsaRadixPassLeft, BM.DsaRadixPassKey, BM.DsaRowValidRoot, BM.DsaRowValidStep,
                 BM.DsaRowValidLast, X.DsaRowValidOffset)
PICK_PIECES = (X.DsaTopKCountAdd, X.DsaTopKDigitGe, X.DsaTopKDigitEdge, X.DsaTopKDigitTake, X.DsaTopKDigitBits,
               X.DsaTopKDigitLeft, X.DsaTopKDigitShare)


def _bits(words, width: int = 32) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint64)
    return ((w[..., None] >> np.arange(width, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8).reshape(len(w), -1)


def _words(bits, width: int = 32) -> list[list[int]]:
    b = np.asarray(bits, dtype=np.uint64)
    b = b.reshape(len(b), -1, width)
    return (b << np.arange(width, dtype=np.uint64)).sum(axis=2).tolist()


def _one(d, *args, width: int = 32) -> list[int]:
    """Each instance's first output word of ``d`` on ``args``."""
    return [r[0] for r in _words(BS.evaluate_bits(d, *args), width)]


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
    assert _one(bind(R.DsaPositionsKeep), _bits(s, 1), _bits(w)) == [x if k else 0 for k, x in zip(s, w)]
    assert _one(bind(R.DsaPositionsShift), _bits(s, 1), _bits(a), _bits(w)) == [x if k else y for k, x, y in zip(s, a, w)]
    assert _one(bind(R.DsaPositionsShiftIn), _bits(s, 1), _bits(w)) == [0 if k else y for k, y in zip(s, w)]
    edge = [0, 1, K - 1, K, K + 1, 1 << 31, (1 << 32) - 1]
    counts = [edge[r % len(edge)] if r < 2 * len(edge) else rng.randrange(K + 2) for r in range(n)]
    offs = [(0, K, K - 1, 1)[r % 4] if r < 28 else rng.randrange(K + 1) for r in range(n)]
    first = _words(BS.evaluate_bits(bind(R.DsaPositionsFirstOffset, K=K), _bits(counts)), RB)
    assert [r[0] for r in first] == [min(c, K) for c in counts]
    nxt = _words(BS.evaluate_bits(bind(R.DsaPositionsOffset, K=K), _bits(offs, RB), _bits(counts)), RB)
    assert [r[0] for r in nxt] == [min(o + min(c, K), K) for o, c in zip(offs, counts)]


def test_the_merge_as_calls_stays_small_at_the_served_statics() -> None:
    """At the served statics the merge written as Calls is a few rows of Calls a block: its Program's description is under
    128 KiB and under a byte for every 128 of its gates (the traced merge's holds every gate; what grows with K here is a row
    for each of K's bits, a stage of the shift and a level of the keep bits' trie), and its Flock block, read off the
    Definition as circuit-check's ``block`` reads it, is within the proved range."""
    from verity_flock import circuit as FC

    unit = bind(R.DsaPositionsMergeV2, **SERVED)
    size = len(canonical_json(encode_program(Program(unit))))
    assert size < 1 << 17 and size * 128 < unit.gates
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


def _commits_read(circuit: list) -> tuple[list, int]:
    """``circuit`` (`_lowered`'s) with each commit (`forms.Circuit.commit`'s row, A a form and B the constant one) dropped and
    its bit read as that form wherever a form holds it: the bits that remain renumbered in order, each form's bits sorted;
    and how many commits it had.  The row holds the bit to the form, so the two circuits compute the same."""
    bits, out = circuit
    forms: list[tuple[frozenset, int]] = []
    kept: list = []

    def read(s, c) -> tuple[tuple, int]:
        acc: frozenset = frozenset()
        for v in s:
            acc, c = acc ^ forms[v][0], c ^ forms[v][1]
        return tuple(sorted(acc)), c

    for kind, d in bits:
        if kind == "and" and d[2:] == ((), 1):
            f, c = read(*d[:2])
            forms.append((frozenset(f), c))
            continue
        forms.append((frozenset((len(kept),)), 0))
        kept.append((kind, (*read(*d[:2]), *read(*d[2:])) if kind == "and" else d))
    return [kept, [read(*x) for x in out]], len(bits) - len(kept)


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


def _pass_rows(N: int, TOPK: int, rows: int) -> tuple[list[np.ndarray], np.ndarray]:
    """``rows`` instances of a pass's operands and its rule on each, (eq, gt, r) side by side: eq sparse, dense and all ones,
    gt apart from eq as the select leaves it and (every other row) any bits, r at 0, 1, the count, one past it and TOPK, then
    at random."""
    rng = random.Random(N * 1000 + TOPK)
    RB = TOPK.bit_length()
    args, want = [[], [], [], []], []
    for t in range(rows):
        p = (0.1, 0.5, 0.9, 1.0)[t % 4]
        eq = [int(rng.random() < p) for _ in range(N)]
        gt = [0 if q else rng.getrandbits(1) for q in eq] if t % 2 else [rng.getrandbits(1) for _ in range(N)]
        e = [rng.getrandbits(1) for _ in range(N)]
        a = [q & x for q, x in zip(eq, e)]
        cnt = sum(a)
        r = (0, 1, cnt, cnt + 1, TOPK)[t % 5] % (1 << RB) if t < 10 else rng.randrange(1 << RB)
        take = int(cnt >= r)
        left = r if take else r - cnt
        for x, v in zip(args, (eq, gt, [r >> i & 1 for i in range(RB)], e)):
            x.append(v)
        want.append([*(q & int(x == take) for q, x in zip(eq, e)), *(g ^ (x & (1 - take)) for g, x in zip(gt, a)),
                     *(left >> i & 1 for i in range(RB))])
    return [np.array(x, dtype=np.uint8) for x in args], np.array(want, dtype=np.uint8)


@pytest.mark.parametrize("N,TOPK", [(1, 1), (5, 3), (20, 16), (37, 8), (300, 16), (600, 512), (1000, 100), (2048, 64),
                                    (2560, 2048)])
def test_the_pass_as_calls_is_the_traced_pass(N: int, TOPK: int) -> None:
    """`DsaRadixPass_v2` has `_v1`'s signature and its function, the pass's rule (the reference: the pass has no word view):
    p says the keys level with the threshold whose bit is set number r or more, a level key stays level where its bit is p,
    a key with its bit set where p is 0 goes above (gt XOR it, an OR where gt and eq are apart, as the select leaves them),
    and r drops by the count where p is 0.  Rows of under a chunk, of chunks and a rest, and of a count of chunks that is
    and is not a power of two.  Up to a chunk of keys it has `_v1`'s ANDs (one more at a single key, where the traced one
    folds a constant); past that, the ANDs of summing the chunks' counts, under one for every 16 keys."""
    calls, traced = bind(BM.DsaRadixPassV2, N=N, TOPK=TOPK), bind(BM.DsaRadixPass, N=N, TOPK=TOPK)
    assert calls.signature() == traced.signature()
    args, want = _pass_rows(N, TOPK, 30)
    for unit in (calls, traced):
        assert np.array_equal(_flat(BS.evaluate_bits(unit, *args), len(want)), want), unit.id
    more = BS.compile(calls).counts["and"] - BS.compile(traced).counts["and"]
    assert more == (N == 1) if N <= BM.PASS_CHUNK else 0 < more < N // 16


@pytest.mark.parametrize("N", [1, 2, 3, 5, 8, 13, 20, 37, 300, 2048])
def test_the_valid_bits_as_calls_are_the_traced_bits(N: int) -> None:
    """`DsaRowValid_v2` has `_v1`'s signature and its function, bit j n > j (the reference), and its ANDs, count for count:
    n from 0 to past the row, at the trie's top 2^L and past 2^31."""
    rng = random.Random(N)
    top = 1 << (N - 1).bit_length()
    n = [0, 1, N - 1, N, N + 1, top - 1, top, top + 1, 1 << 31, (1 << 32) - 1, *(rng.randrange(N + 2) for _ in range(30))]
    calls, traced = bind(BM.DsaRowValidV2, N=N), bind(BM.DsaRowValid, N=N)
    assert calls.signature() == traced.signature()
    want = [[int(x > j) for j in range(N)] for x in n]
    for unit in (calls, traced):
        assert np.asarray(BS.evaluate_bits(unit, _bits(n))).reshape(len(n), -1).tolist() == want, unit.id
    assert BS.compile(calls).counts["and"] == BS.compile(traced).counts["and"]


def test_the_selects_pieces_are_their_steps() -> None:
    """A chunk's count, a pass's p and r past it, a key past a pass, the trie's root, a node's two children and its two keys,
    and n past a block's start, on random operands and at their edges."""
    rng = random.Random(11)
    n = 64
    for N in (1, 2, 7, 44, 256):
        a = _masks(N, n, N)
        assert _one(bind(BM.DsaRadixPassCount, N=N), a, width=N.bit_length()) == a.sum(axis=1).tolist()
    for w, rb in ((1, 3), (9, 5), (16, 12), (6, 12)):
        cnt, r = ([rng.choice((0, 1, (1 << b) - 1, rng.randrange(1 << b))) for _ in range(n)] for b in (w, rb))
        got = _one(bind(BM.DsaRadixPassLeft, W=w, RB=rb), _bits(cnt, w), _bits(r, rb), width=1 + rb)
        assert got == [int(c >= x) | (x if c >= x else (x - c) % (1 << rb)) << 1 for c, x in zip(cnt, r)]
    q, g, a, p = ([rng.getrandbits(1) for _ in range(n)] for _ in range(4))
    got = _one(bind(BM.DsaRadixPassKey), *(_bits(x, 1) for x in (q, g, a, p)), width=2)
    assert got == [x ^ y ^ (t & x) | (z ^ y ^ (t & y)) << 1 for x, z, y, t in zip(q, g, a, p)]
    word = [rng.choice((0, 1, (1 << 32) - 1, rng.getrandbits(32), rng.getrandbits(8))) for _ in range(n)]
    for L in (0, 4, 16, 31):
        assert _one(bind(BM.DsaRowValidRoot, L=L), _bits(word), width=2) == [int(x < 1 << L) | int(x >= 1 << L) << 1 for x in word]
    node, b = [rng.randrange(4) for _ in range(n)], [rng.getrandbits(1) for _ in range(n)]
    t = [x & 1 & y for x, y in zip(node, b)]
    got = _words(BS.evaluate_bits(bind(BM.DsaRowValidStep), _bits(node, 2), _bits(b, 1)), 2)
    assert got == [[(x & 1 ^ k) | (x >> 1 ^ k) << 1, k | (x >> 1) << 1] for x, k in zip(node, t)]
    assert _one(bind(BM.DsaRowValidLast), _bits(node, 2), _bits(b, 1), width=2) == [(x >> 1 ^ k) | (x >> 1) << 1 for x, k in zip(node, t)]
    for AT in (1, 16, 16384, (1 << 31) + 5):
        at = [AT - 1, AT, AT + 1, *word[3:]]
        assert _one(bind(X.DsaRowValidOffset, AT=AT), _bits(at), width=32) == [max(x - AT, 0) for x in at]


@pytest.mark.parametrize("N,AT", [(8, 0), (6, 8), (40, 16), (600, 0)])
def test_the_mask_units_as_calls_are_the_traced_units(N: int, AT: int) -> None:
    """`DsaTopKMaskBlock_v2` and `DsaTopKMaskWhole_v2` have their `_v1`'s signature and its mask, on rows of few distinct
    scores (so keys tie at the threshold), n inside the block, at its edges and past it, and r from 0 up; a row of two
    chunks of a pass's count and a rest among them."""
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


@pytest.mark.parametrize("N,AT", [(X.BLOCK, 0), (X.BLOCK, X.BLOCK), (X.WHOLE_MAX, None)])
def test_the_mask_units_as_calls_stay_small_at_the_served_statics(N: int, AT: int | None) -> None:
    """At the long row's served statics a mask unit written as Calls (a block of the cut, its first and one past the row's
    start, and the longest row masked whole, its passes and its valid bits Calls too) has a description under a mebibyte,
    where `_v1`'s holds every gate of its ties, its valid bits and a pass; a pass's is under 128 KiB and the valid bits' under
    16 KiB.  The mask whole is within a Flock block's proved range."""
    from verity_flock import circuit as FC

    def size(d) -> int:
        return len(canonical_json(encode_program(Program(d))))

    if AT is None:
        unit = bind(X.DsaTopKMaskWholeV2, N=N, TOPK=2048)
        assert size(bind(BM.DsaRadixPassV2, N=N, TOPK=2048)) < 1 << 17
        b = FC.unit_block(unit)
        assert FC.past_block(b) is None, (unit.id, b["k_log"])
    else:
        unit = bind(X.DsaTopKMaskBlockV2, N=N, AT=AT, TOPK=2048)
    assert size(unit) < 1 << 20 and size(bind(BM.DsaRowValidV2, N=N)) < 1 << 14


def test_the_selects_builders_lower_to_their_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each piece of the pass and of the valid bits carries a builder, as the merge's do: the circuit Flock's flat lowering
    makes of a pass over two chunks and a rest, of the valid bits of 13 keys, of a block's mask past the row's start and of
    the mask whole is the one their gates lower to, AND for AND."""
    units = [bind(BM.DsaRadixPassV2, N=2 * BM.PASS_CHUNK + 88, TOPK=16), bind(BM.DsaRowValidV2, N=13),
             bind(X.DsaTopKMaskBlockV2, N=13, AT=16, TOPK=4), bind(X.DsaTopKMaskWholeV2, N=13, TOPK=4)]
    names = {p.name for p in SELECT_PIECES}
    with_builders = [_lowered(u) for u in units]
    assert set().union(*(built for _, built in with_builders)) >= names
    for piece in SELECT_PIECES:
        monkeypatch.delattr(piece, "build")
    for unit, (circuit, _) in zip(units, with_builders, strict=True):
        by_gates, built = _lowered(unit)
        assert not names & set(built) and by_gates == circuit, unit.id


def test_the_ties_builders_lower_to_their_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each piece of the ties carries a builder, as the merge's do: the circuit Flock's flat lowering makes of the ties over
    13 keys, of a mask unit over them, and of a pass whose count sums two chunks' and a rest's, is the one their gates
    lower to, AND for AND, and the rows that commit each count sum (`boolean_dsa_topk_mask._committed`): with each
    committed bit read as the form its row holds it to, the two are one circuit."""
    units = [bind(BM.DsaRadixTiesV2, N=13, TOPK=4), bind(X.DsaTopKMaskBlockV2, N=13, AT=16, TOPK=4),
             bind(BM.DsaRadixPassV2, N=2 * BM.PASS_CHUNK + 88, TOPK=16)]
    names = {p.name for p in TIES_PIECES}
    with_builders = [_lowered(u) for u in units]
    assert all(names & set(built) for _, built in with_builders)
    for piece in TIES_PIECES:
        monkeypatch.delattr(piece, "build")
    for unit, (circuit, _) in zip(units, with_builders, strict=True):
        by_gates, built = _lowered(unit)
        read, commits = _commits_read(circuit)
        assert not names & set(built) and commits and (read, 0) == _commits_read(by_gates), unit.id


def _flat(out, rows: int) -> np.ndarray:
    """An evaluation's outputs, a row an instance, its fields side by side."""
    return np.concatenate([np.asarray(f).reshape(rows, -1) for f in (out if isinstance(out, tuple) else (out,))], axis=1)


@pytest.mark.parametrize("N,TOPK,block", [(14, 4, 8), *(pytest.param(*c, marks=pytest.mark.slow)
                                                        for c in ((8, 4, 8), (40, 16, 8), (30, 100, 16)))])
def test_the_pick_as_calls_is_the_traced_pick(N: int, TOPK: int, block: int) -> None:
    """`DsaTopKDigitPick_v2` has `_v1`'s signature, its function and the cut's twin's (`values`, the reference: the pick has
    no word view) at every round, and its ANDs, count for count: on rows of few distinct scores (so the threshold's key
    repeats across blocks), n from none to past the row, each round reading the counts and the pick before it as the twin
    has them, with any bits past those a pick reads; and a TOPK past the row, where no digit reaches r."""
    rng = random.Random(N * 100 + TOPK)
    pool = [rng.getrandbits(32) for _ in range(5)]
    rows = [[rng.choice(pool[:2 + t % 4]) if t % 3 else rng.getrandbits(32) for _ in range(N)] for t in range(24)]
    ns = [(0, 1, TOPK, N, N + 1)[t % 5] if t < 10 else rng.randrange(N + 1) for t in range(len(rows))]
    twin = [X.values(row, n, TOPK, block) for row, n in zip(rows, ns)]
    M, RB = len(X.blocks(N, TOPK, block)), TOPK.bit_length()
    for K in range(X.ROUNDS):
        calls, traced = (bind(d, M=M, K=K, TOPK=TOPK) for d in (X.DsaTopKDigitPickV2, X.DsaTopKDigitPick))
        assert calls.signature() == traced.signature()
        args = [_bits([c for v in twin for c in v[("count", K, b)][0]]).reshape(len(rows), -1) for b in range(M)]
        if K:
            for f, read in enumerate((X.DIGIT * K, RB)):
                past = X.wide(read) - read
                args.append(_bits([v[("pick", K - 1)][f] | rng.getrandbits(past) << read if past else v[("pick", K - 1)][f]
                                   for v in twin], X.wide(read)))
        widths = [X.wide(X.DIGIT * (K + 1)), *[X.wide(RB)] * (M if K == X.ROUNDS - 1 else 1)]
        ends = np.cumsum(widths)
        for unit in (calls, traced):
            got = _flat(BS.evaluate_bits(unit, *args), len(rows))
            fields = [_words(got[:, e - w:e], w) for w, e in zip(widths, ends)]
            assert [tuple(f[i][0] for f in fields) for i in range(len(rows))] == [v[("pick", K)] for v in twin], (unit.id, K)
        assert BS.compile(calls).counts["and"] == BS.compile(traced).counts["and"]


def test_the_picks_pieces_are_its_steps() -> None:
    """A sum of counts, a level's count against r (read, and TOPK), a digit's one-hot bit, a row under it, the digit of the
    one-hot bits, r past a digit and a block's share of the keys at the threshold, on random operands and at their edges."""
    rng = random.Random(9)
    n, TOPK = 64, 100
    RB = TOPK.bit_length()

    def word(bits: int = 32) -> list[int]:
        return [rng.choice((0, 1, (1 << bits) - 1, rng.getrandbits(bits))) for _ in range(n)]

    a, b, r, s = word(), word(), word(RB), word(1)
    assert _one(bind(X.DsaTopKCountAdd), _bits(a), _bits(b)) == [(x + y) % (1 << 32) for x, y in zip(a, b)]
    t = [x if i % 2 else y for i, (x, y) in enumerate(zip(a, r))]
    assert _one(bind(X.DsaTopKDigitGe, TOPK=TOPK, READ=1), _bits(t), _bits(r, RB), width=1) == [int(x >= y) for x, y in zip(t, r)]
    t = [TOPK - 1, TOPK, TOPK + 1, *a[3:]]
    assert _one(bind(X.DsaTopKDigitGe, TOPK=TOPK, READ=0), _bits(t), width=1) == [int(x >= TOPK) for x in t]
    assert _one(bind(X.DsaTopKDigitEdge), _bits(s, 1), _bits(b, 1), width=1) == [x & ~y & 1 for x, y in zip(s, b)]
    assert _one(bind(X.DsaTopKDigitTake, W=32), _bits(s, 1), _bits(a)) == [x if k else 0 for k, x in zip(s, a)]
    assert _one(bind(X.DsaTopKDigitBits), np.eye(X.BINS, dtype=np.uint8), width=X.DIGIT) == list(range(X.BINS))
    above = word(RB)
    assert _one(bind(X.DsaTopKDigitLeft, TOPK=TOPK, READ=1), _bits(above, RB), _bits(r, RB), width=RB) == [
        (x - y) % (1 << RB) for x, y in zip(r, above)]
    assert _one(bind(X.DsaTopKDigitLeft, TOPK=TOPK, READ=0), _bits(above, RB), width=RB) == [(TOPK - y) % (1 << RB) for y in above]
    before = [x if i % 2 else y for i, (x, y) in enumerate(zip(a, word(RB)))]
    assert _one(bind(X.DsaTopKDigitShare, RB=RB), _bits(r, RB), _bits(before), width=RB) == [max(x - y, 0) for x, y in zip(r, before)]


@pytest.mark.parametrize("K", range(X.ROUNDS))
def test_the_pick_as_calls_stays_small_at_the_served_statics(K: int) -> None:
    """At the long row's served statics (8 blocks' counts, TOPK 2,048) a round's pick written as Calls is a batch a block of
    the sum, a Call a digit of T, and a batch a step after it: a description under 256 KiB (the traced pick's holds every
    gate), and a Flock block within the proved range."""
    from verity_flock import circuit as FC

    unit = bind(X.DsaTopKDigitPickV2, M=8, K=K, TOPK=2048)
    assert len(canonical_json(encode_program(Program(unit)))) < 1 << 18
    b = FC.unit_block(unit)
    assert FC.past_block(b) is None, (unit.id, b["k_log"])


@pytest.mark.parametrize("K", [0, pytest.param(1, marks=pytest.mark.slow), 3])
def test_the_pick_as_calls_has_the_traced_picks_block(K: int) -> None:
    """The Flock block of a pick written as Calls is the traced one's: the same rows in and out, the same ANDs."""
    from verity_flock import circuit as FC

    S = dict(M=3, K=K, TOPK=100)
    assert FC.unit_block(bind(X.DsaTopKDigitPickV2, **S)) == FC.unit_block(bind(X.DsaTopKDigitPick, **S))


@pytest.mark.slow
def test_the_picks_builders_lower_to_their_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each piece of the pick carries a builder, as the merge's do: the circuit Flock's flat lowering makes of a last
    round's pick over 2 blocks, which reaches every piece, is the one its gates lower to, AND for AND.  Slow by the lowering
    without builders, an activation at a time over 255 digits; the merge's and the ties' tests are the quick ones of a
    builder, and this one holds the pick's seven."""
    unit = bind(X.DsaTopKDigitPickV2, M=2, K=3, TOPK=4)
    names = {p.name for p in PICK_PIECES}
    circuit, built = _lowered(unit)
    assert names <= set(built)
    for piece in PICK_PIECES:
        monkeypatch.delattr(piece, "build")
    by_gates, built = _lowered(unit)
    assert not names & set(built) and by_gates == circuit



def _masks(N: int, rows: int, seed: int) -> np.ndarray:
    """``rows`` masks of N keys: sparse, half, dense and full."""
    rng = random.Random(seed)
    return np.array([[int(rng.random() < (0.1, 0.5, 0.9, 1.0)[t % 4]) for _ in range(N)] for t in range(rows)], dtype=np.uint8)


@pytest.mark.parametrize("N,K", [(2, 1), (2, 2), (3, 2), (5, 3), (8, 4), (13, 13), (16, 8), (20, 16), (33, 8), (64, 16),
                                 (100, 7), (257, 64), (1000, 100)])
def test_the_tags_and_count_as_calls_are_the_traced_tags_and_count(N: int, K: int) -> None:
    """`DsaButterflyTagsCount_v1` is `DsaButterflyTagBits_v1`'s tags (the reference: each key's mask bit and the members
    before it mod 2^kb) and the mask's count, as `DsaTopKPopCount_v1` has it.  From 13 keys on it has no more ANDs than the
    two traced ones together; below that the traced ones fold an AND or two more."""
    rows, kb = 24, max(1, (K - 1).bit_length())
    mask = _masks(N, rows, N * 100 + K)
    got = _flat(BS.evaluate_bits(bind(BM.DsaButterflyTagsCount, N=N, K=K), mask), rows)
    before = (np.cumsum(mask, axis=1, dtype=np.int64) - mask) % (1 << kb)
    want = np.concatenate([mask[..., None], (before[..., None] >> np.arange(kb)) & 1], axis=2).reshape(rows, -1)
    assert np.array_equal(got[:, :-32], want)
    assert np.array_equal(got[:, :-32], np.asarray(BS.evaluate_bits(bind(BM.DsaButterflyTagBits, N=N, K=K), mask)).reshape(rows, -1))
    assert [r[0] for r in _words(got[:, -32:])] == mask.sum(axis=1).tolist()
    if N >= 13:
        traced = sum(BS.compile(bind(d, **st)).counts["and"] for d, st in ((BM.DsaButterflyTagBits, dict(N=N, K=K)),
                                                                           (X.DsaTopKPopCount, dict(N=N))))
        assert BS.compile(bind(BM.DsaButterflyTagsCount, N=N, K=K)).counts["and"] <= traced


@pytest.mark.parametrize("N,AT,K", [(8, 0, 4), (6, 8, 4), (14, 0, 4), (40, 16, 16), (16, 16, 8), (37, 5, 9), (64, 192, 16)])
def test_the_mask_positions_as_calls_are_the_traced_positions(N: int, AT: int, K: int) -> None:
    """`DsaMaskPositions_v2` has `_v1`'s signature and its function, every slot of it, and the reference's (the unit has no
    word view): the mask's count, and where the mask has at most K members, as a counted mask has, slot s its s-th member's
    position.  Blocks that start at a multiple of their length, off one (a run of one position among them) and past a power
    of two."""
    rows = 24
    mask = _masks(16 * -(-N // 16), rows, N + AT)
    calls, traced = (bind(d, N=N, AT=AT, K=K) for d in (R.DsaMaskPositionsV2, R.DsaMaskPositions))
    assert calls.signature() == traced.signature()
    got = _flat(BS.evaluate_bits(calls, mask), rows)
    assert np.array_equal(got, _flat(BS.evaluate_bits(traced, mask), rows))
    words, few = _words(got), 0
    for i, m in enumerate(mask[:, :N].tolist()):
        members = [AT + j for j, b in enumerate(m) if b]
        assert words[i][K] == len(members)
        if len(members) <= K:
            few += 1
            assert words[i][:len(members)] == members
    assert few >= 3


def test_the_positions_pieces_are_its_steps() -> None:
    """The indices 0 .. 2^L - 1 from the two constant bits, an index's position in its run, a bit after a row, and a key's
    tag of its mask bit and the count before it."""
    rng = random.Random(3)
    zero, one = np.zeros((1, 1), np.uint8), np.ones((1, 1), np.uint8)
    for L in (1, 2, 7):
        got = np.asarray(BS.evaluate_bits(bind(R.DsaIndexBits, L=L), zero, one)).reshape(1 << L, L)
        assert [r[0] for r in _words(got, L)] == list(range(1 << L))
    idx = [rng.randrange(1 << 5) for _ in range(32)]
    for hi in (0, 1, 5, (1 << 26) - 1):
        got = _words(BS.evaluate_bits(bind(R.DsaPositionWord, L=5, HI=hi), _bits(idx, 5)))
        assert [r[0] for r in got] == [hi << 5 | i for i in idx]
    b = [rng.getrandbits(1) for _ in idx]
    assert [r[0] for r in _words(BS.evaluate_bits(bind(R.DsaBitAppend, W=5), _bits(idx, 5), _bits(b, 1)), 6)] == [
        i | x << 5 for i, x in zip(idx, b)]
    assert [r[0] for r in _words(BS.evaluate_bits(bind(BM.DsaButterflyTag, W=5, KB=7), _bits(b, 1), _bits(idx, 5)), 8)] == [
        x | i << 1 for i, x in zip(idx, b)]
    assert [r[0] for r in _words(BS.evaluate_bits(bind(BM.DsaButterflyTag, W=0, KB=7), _bits(b, 1)), 8)] == b


@pytest.mark.parametrize("N,AT", [(X.BLOCK, X.BLOCK), (X.WHOLE_MAX, 0)])
def test_the_mask_positions_as_calls_stay_small_at_the_served_statics(N: int, AT: int) -> None:
    """At the long row's served statics (a block of the cut, and the longest row masked whole) the positions written as
    Calls are the switches' batches, as `_v1`'s are, a few batches a level of the count's tree and a run of positions for
    each bit of the block's length: a description under 2 MiB and under a byte for every 32 of the gates, where `_v1`'s
    holds every gate of its tags and its count and a constant for every bit of a position."""
    unit = bind(R.DsaMaskPositionsV2, N=N, AT=AT, K=2048)
    size = len(canonical_json(encode_program(Program(unit))))
    assert size < 1 << 21 and size * 32 < unit.gates


def test_the_positions_builders_lower_to_their_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each piece of the positions carries a builder: the circuit Flock's flat lowering makes of a block of 13 keys off a
    multiple of its length is the one its gates lower to, AND for AND, and the rows that commit each count sum, each
    committed bit read as the form its row holds it to (`_commits_read`).  `DsaBitAppend_v1` has no gate, so the lowering
    meets no activation of it: its output is its operands' wires."""
    unit = bind(R.DsaMaskPositionsV2, N=13, AT=6, K=4)
    names = {p.name for p in POSITION_PIECES}
    circuit, built = _lowered(unit)
    assert names - set(built) == {"DsaBitAppend"}
    for piece in POSITION_PIECES:
        monkeypatch.delattr(piece, "build")
    by_gates, built = _lowered(unit)
    read, commits = _commits_read(circuit)
    assert not names & set(built) and commits and (read, 0) == _commits_read(by_gates)
