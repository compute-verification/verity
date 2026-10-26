"""`verity.ml.circuits.ops.boolean.shared_row`: `SharedRowCheck_v1{G, W}` is 0 exactly when W·j + W <= G and segment j of the
G-word row is y, at every 16-bit j for rows of 1 to 15 words in segments of 1 to 4, with every one-bit change of the segment
caught and an out-of-range j refused even where every segment is y; `SharedRowWindow_v1{FN, G}` is its producer's words then
that check, for the s8 coordinate on two rows, PV's head row, the embedding's coordinate and a cast, and adds exactly the
check's gates; j is a position-kind port and FN's own position ports carry over; the check costs 8,198, 8,203 and 8,204 ANDs
at a 1 KiB row (G = 512) for W = 1, 32 and 64, and 2,056 and 2,062 at G = 128 for W = 1 and 64; and a word producer, a G and W
that don't tile a row a 16-bit j indexes and a producer with a parameter named row or j are refused at `bind`."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import ports as PT
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import moe as BM
from verity.ml.circuits.ops.boolean import rowwise as BR
from verity.ml.circuits.ops.boolean import shared_row as SR
from verity.ml.circuits.ops.boolean import softmax_row as SM
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array

J = np.arange(1 << 16, dtype=np.uint16)


def _check(y, row, j, G: int, W: int) -> np.ndarray:
    return np.asarray(BS.evaluate_words(bind(SR.SharedRowCheck, G=G, W=W), y, row, j)).astype(np.uint16).reshape(len(j))


def _segment_rows(rng, y, j, G: int) -> np.ndarray:
    """Random rows of G words with segment j (where it lies inside the row) set to y."""
    N, W = y.shape
    row = rng.integers(0, 1 << 16, (N, G), dtype=np.uint16)
    inside = np.flatnonzero(j < G // W)
    for w in range(W):
        row[inside, j[inside].astype(np.int64) * W + w] = y[inside, w]
    return row


@pytest.mark.parametrize("G,W", [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (8, 1), (2, 2), (6, 2), (6, 3), (15, 3), (4, 4),
                                 (12, 4)])
def test_the_check_is_zero_exactly_at_its_segment_at_every_j(G: int, W: int) -> None:
    """Every 16-bit j: with segment j set to y, c = 0 for W·j + W <= G and 0xFFFF past the row; with every segment y, the
    same, so an out-of-range j is refused by the range bit, not by a mismatch; with one bit of segment j flipped, c is that
    bit. Random rows with no segment set give `shared_row_check_reference`'s c."""
    rng = np.random.default_rng(1000 * G + W)
    n = G // W
    y = rng.integers(0, 1 << 16, (len(J), W), dtype=np.uint16)
    want = np.where(J < n, 0, 0xFFFF).astype(np.uint16)
    assert np.array_equal(_check(y, _segment_rows(rng, y, J, G), J, G, W), want)
    assert np.array_equal(_check(y, np.tile(y, (1, n)), J, G, W), want)
    inside = J[J < n]
    yy = y[:len(inside)]
    for w in range(W):
        for b in range(16):
            row = _segment_rows(rng, yy, inside, G)
            row[np.arange(len(inside)), inside.astype(np.int64) * W + w] ^= np.uint16(1 << b)
            assert np.array_equal(_check(yy, row, inside, G, W), np.full(len(inside), 1 << b, np.uint16)), (w, b)
    row = rng.integers(0, 1 << 16, (len(J), G), dtype=np.uint16)
    assert np.array_equal(_check(y, row, J, G, W), SR.shared_row_check_reference(y, row, J))


def test_a_change_outside_the_segment_is_not_the_units() -> None:
    """At G = 12, W = 4: every word outside segment j may be anything, so a producer binds only its own segment of the row."""
    rng = np.random.default_rng(7)
    N, G, W = 3 * 64, 12, 4
    j = np.repeat(np.arange(3, dtype=np.uint16), 64)
    y = rng.integers(0, 1 << 16, (N, W), dtype=np.uint16)
    row = _segment_rows(rng, y, j, G)
    for other in range(G):
        mine = (other // W) == j
        flipped = row.copy()
        flipped[:, other] ^= np.uint16(0x8001)
        assert np.array_equal(_check(y, flipped, j, G, W) == 0, ~mine), other


def _inputs(rng, target, N: int) -> list[np.ndarray]:
    """Random words for each of ``target``'s parameters, shaped as `evaluate_words` takes them."""
    out = []
    for _, t in target.signature()[0]:
        widths = BL.elements(t)
        out.append(np.stack([rng.integers(0, 1 << w, N, dtype=np.uint64) for w in widths], axis=1))
    return out


PRODUCERS = [
    pytest.param(bind(BG.F2fpBf16), 5, id="cast"),
    pytest.param(bind(BG.ScaledMmS8CoordinateRows, K=64, R=2, **BG.GEMM_SM90_S8), 128, id="s8-coordinate-rows"),
    pytest.param(bind(BG.PvRow, T=2, K=16, D=3, **BG.GEMM_HOPPER), 6, id="pv-row"),
    pytest.param(bind(BR.EmbeddingCoordinate, V=16), 128, id="embedding"),
]


@pytest.mark.parametrize("FN,G", PRODUCERS)
def test_the_wrapper_is_its_producer_then_the_check(FN, G: int) -> None:
    """On random inputs, rows with segment j set to FN's output for about half the instances and j over 0..G/W + 2: the
    wrapper is `shared_row_check_reference` of FN's words (`evaluate_words` of FN), 0 exactly where segment j holds them
    inside the row."""
    rng = np.random.default_rng(G)
    N = 96
    xs = _inputs(rng, FN, N)
    y = np.asarray(BS.evaluate_words(FN, *xs)).astype(np.uint16).reshape(N, -1)
    W = y.shape[1]
    n = G // W
    j = rng.integers(0, n + 3, N).astype(np.uint16)
    row = _segment_rows(rng, y, j, G)
    other = rng.random(N) < 0.5
    row[other] = rng.integers(0, 1 << 16, row[other].shape, dtype=np.uint16)
    target = bind(SR.SharedRowWindow, FN=FN, G=G)
    got = np.asarray(BS.evaluate_words(target, *xs, row, j)).astype(np.uint16).reshape(N)
    assert np.array_equal(got, SR.shared_row_check_reference(y, row, j))
    holds = np.array([j[i] < n and np.array_equal(row[i, W * int(j[i]):W * (int(j[i]) + 1)], y[i]) for i in range(N)])
    assert np.array_equal(got == 0, holds) and holds.any() and (~holds).any()


@pytest.mark.parametrize("FN,G", PRODUCERS)
def test_the_wrapper_adds_exactly_the_check(FN, G: int) -> None:
    """The wrapper's ANDs are FN's and the check's, and its one output is 16 bits."""
    target = bind(SR.SharedRowWindow, FN=FN, G=G)
    W = len(BL.elements(FN.signature()[1]))
    ands = lambda t: BS.compile(t).counts["and"]  # noqa: E731
    assert ands(target) == ands(FN) + ands(bind(SR.SharedRowCheck, G=G, W=W))
    assert BL.is_boolean(target) and BL.elements(target.signature()[1]) == [16]


def test_the_check_costs_a_select_tree_and_the_range_bit() -> None:
    """(G/W - 1)·16·W tree ANDs, (W - 1)·16 to fold the segment's words, the OR of j's high bits and 16 to set c: at a 1 KiB
    row (G = 512), 8,198 for W = 1, 8,203 for W = 32 and 8,204 for W = 64; at G = 128, 2,056 for W = 1
    (`MoeSelectBf16x128_v2`, a select alone, is 2,072) and 2,062 for PV's two 64-word segments."""
    ands = lambda t: BS.compile(t).counts["and"]  # noqa: E731
    assert ands(bind(SR.SharedRowCheck, G=512, W=1)) == 511 * 16 + 6 + 16 == 8198
    assert ands(bind(SR.SharedRowCheck, G=512, W=32)) == 15 * 16 * 32 + 31 * 16 + 11 + 16 == 8203
    assert ands(bind(SR.SharedRowCheck, G=512, W=64)) == 7 * 16 * 64 + 63 * 16 + 12 + 16 == 8204
    assert ands(bind(SR.SharedRowCheck, G=128, W=1)) == 127 * 16 + 8 + 16 == 2056
    assert ands(bind(SR.SharedRowCheck, G=128, W=64)) == 64 * 16 + 63 * 16 + 14 + 16 == 2062
    assert ands(BM.select_bf16(128)) == 2072


def _window(FN, G: int):
    return bind(SR.SharedRowWindow, FN=FN, G=G)


def test_j_is_a_position_kind_port_and_the_producers_ports_carry_over() -> None:
    """j is a run input of kind position, public as a key count is; a caller passing it a value it didn't declare as a position
    is refused (`ports.E_PORT_KIND`). A producer's own position port (`QkSoftmaxRowPad_v1`'s n) is the wrapper's too."""
    target = _window(bind(BG.F2fpBf16), 4)
    facts = PT.facts(Program(target).root)
    assert facts.node(2).one.kind == "position" and facts.node(0).one.kind != "position"
    params, ret = target.signature()
    undeclared = CompositeDefinition("SharedRowWindowTestUndeclared", 1, (), lambda S: (params, ret),
                                     lambda B, S, *a: B.call(target, *a), register=False)
    with pytest.raises(PT.PortError, match="takes kind=position") as e:
        Program(undeclared)
    assert e.value.code == "E_PORT_KIND"
    padded = _window(bind(SM.QkSoftmaxRowPad, T=2, D=16, **BG.GEMM_HOPPER), 4)
    names = [n for n, _ in padded.signature()[0]]
    assert names == ["q", "k", "n", "row", "j"]
    facts = PT.facts(Program(padded).root)
    assert [facts.node(i).one.kind == "position" for i in range(len(names))] == [False, False, True, False, True]


def test_a_word_producer_an_untiled_row_and_a_reserved_name_are_refused() -> None:
    cast = bind(BG.F2fpBf16)
    with pytest.raises(ValueError, match="Boolean form"):
        bind(SR.SharedRowWindow, FN=bind(G.ScaledMmS8CoordinateRows, K=64, R=2, **G.GEMM_SM90_S8), G=128)
    for g in (0, True, 2.0):
        with pytest.raises(ValueError, match="at least 1"):
            bind(SR.SharedRowWindow, FN=cast, G=g)
    with pytest.raises(ValueError, match="past 65536"):
        bind(SR.SharedRowWindow, FN=cast, G=SR.J_LIMIT + 1)
    pv = bind(BG.PvRow, T=2, K=16, D=3, **BG.GEMM_HOPPER)
    with pytest.raises(ValueError, match="isn't whole segments"):
        bind(SR.SharedRowWindow, FN=pv, G=8)
    assert bind(SR.SharedRowWindow, FN=pv, G=9).signature()[0][-2][1] == Array(9, SR.BF16)
    for name in ("row", "j"):
        clash = CompositeDefinition(f"SharedRowWindowTestClash{name}", 1, (), lambda S, n=name: (((n, Array(16, BL.BIT)),),
                                                                                                  Array(16, BL.BIT)),
                                    lambda B, S, a: a, register=False)
        with pytest.raises(ValueError, match="the wrapper's own"):
            bind(SR.SharedRowWindow, FN=clash, G=2)
    with pytest.raises(ValueError, match="at least 1"):
        bind(SR.SharedRowCheck, G=2, W=0).signature()
    with pytest.raises(ValueError, match="isn't whole segments"):
        bind(SR.SharedRowCheck, G=6, W=4).signature()
