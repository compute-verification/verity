"""The long row's units written as Calls (`dsa_topk_reads.DsaPositionsMerge_v2`): each has its traced version's signature and
is that version and its reference bit for bit, with its ANDs count for count; the pieces are the steps they name; at the
served statics its description is a few rows of Calls and it fits a Flock block, the traced one's; and the pieces lower by
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
from verity.definitions.registry import dsa_topk_reads as R

#: the long row's merge as served: the longest row's 8 blocks, TOPK slots
SERVED = dict(M=8, K=2048)
PIECES = (R.DsaPositionsKeep, R.DsaPositionsShift, R.DsaPositionsShiftIn, R.DsaPositionsFirstOffset, R.DsaPositionsOffset)


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
