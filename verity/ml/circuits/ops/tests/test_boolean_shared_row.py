"""`verity.ml.circuits.ops.boolean.shared_row`: `SharedRowCheck_v1{W, ROW, OFF, G}` is segment j XOR y bit for bit, every
bit set past G, so 0 exactly when j < G and segment j (words [OFF + W·j, OFF + W·j + W) of the row) is y, at every 16-bit
j, for rows that one family tiles and rows it shares with
another family's segments before and after its own, with every one-bit change of the segment caught, every word outside it
the block's to ignore, and a j >= G refused even where the row goes on holding y; `SharedRowSelect_v1` is segment j, 0 where
j >= G, at every j; `SharedRowWindow_v1{FN, ROW, OFF, G}` is its producer's words then that check, for the writers table's
families on small statics, and adds exactly the check's gates; `SharedRowRead_v1{FN, PORT, ROW, OFF, G}` is FN with PORT
read from its segment, at no gate for G = 1 (the RMS norm's f32 chunk sums from a pooled row, low half first) and the select's
for G > 1, and nests in itself and in the writer; j and PORT_j are position-kind ports and FN's own carry over; the costs at a
1 KiB row are as the module states; and a word FN, segments past the row or past a 16-bit j, a PORT that isn't one of FN's
parameters or isn't whole words, and FN parameters named as the wrappers' own are refused at `bind`."""
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
from verity.ml.circuits.ops.boolean import quant as BQ
from verity.ml.circuits.ops.boolean import rmsnorm as RN
from verity.ml.circuits.ops.boolean import rows as BRW
from verity.ml.circuits.ops.boolean import rowwise as BR
from verity.ml.circuits.ops.boolean import shared_row as SR
from verity.ml.circuits.ops.boolean import softmax_row as SM
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array

J = np.arange(1 << 16, dtype=np.uint16)
U16 = SR.U16


def ands(t) -> int:
    return BS.compile(t).counts["and"]


def _words(target, *args) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *args)).astype(np.uint64).reshape(len(args[0]), -1)


def _check(y, row, j, W: int, OFF: int, G: int) -> np.ndarray:
    return _words(bind(SR.SharedRowCheck, W=W, ROW=row.shape[1], OFF=OFF, G=G), y, row, j).astype(np.uint16)


def _set(row, y, j, W: int, OFF: int, G: int) -> np.ndarray:
    """``row`` with segment j set to y where j < G."""
    row = row.copy()
    inside = np.flatnonzero(j < G)
    for w in range(W):
        row[inside, OFF + j[inside].astype(np.int64) * W + w] = y[inside, w]
    return row


#: (W, ROW, OFF, G): rows one family tiles, and rows with words before OFF and after its last segment
SHAPES = [(1, 1, 0, 1), (1, 2, 0, 2), (1, 3, 0, 3), (1, 5, 1, 3), (1, 8, 0, 8), (1, 10, 4, 1), (2, 4, 0, 2), (2, 7, 1, 3),
          (3, 15, 0, 5), (4, 12, 0, 3), (4, 14, 1, 2)]


@pytest.mark.parametrize("W,ROW,OFF,G", SHAPES)
def test_the_check_is_zero_exactly_at_its_segment_at_every_j(W: int, ROW: int, OFF: int, G: int) -> None:
    """Every 16-bit j: with segment j set to y, d = 0 for j < G and every bit set for j >= G; with y in every W words from
    OFF that fit the row, the same, so a j >= G is refused by the range bit even where the row goes on holding y; with one
    bit of segment j flipped, d is that bit alone. Random rows give `shared_row_check_reference`'s d, segment j XOR y."""
    rng = np.random.default_rng(1000 * ROW + 100 * OFF + 10 * W + G)
    y = rng.integers(0, 1 << 16, (len(J), W), dtype=np.uint16)
    want = np.where(J[:, None] < G, 0, 0xFFFF).astype(np.uint16).repeat(W, axis=1)
    noise = rng.integers(0, 1 << 16, (len(J), ROW), dtype=np.uint16)
    assert np.array_equal(_check(y, _set(noise, y, J, W, OFF, G), J, W, OFF, G), want)
    tiled = noise.copy()
    for s in range((ROW - OFF) // W):
        tiled[:, OFF + W * s:OFF + W * (s + 1)] = y
    assert (ROW - OFF) // W >= G and np.array_equal(_check(y, tiled, J, W, OFF, G), want)
    inside = J[J < G]
    yy = y[:len(inside)]
    for w in range(W):
        for b in range(16):
            row = _set(noise[:len(inside)], yy, inside, W, OFF, G)
            row[np.arange(len(inside)), OFF + inside.astype(np.int64) * W + w] ^= np.uint16(1 << b)
            one = np.zeros((len(inside), W), np.uint16)
            one[:, w] = 1 << b
            assert np.array_equal(_check(yy, row, inside, W, OFF, G), one), (w, b)
    got = _check(y, noise, J, W, OFF, G)
    assert np.array_equal(got, SR.shared_row_check_reference(y, noise, J, G, OFF))
    inside = J < G
    seg = np.stack([noise[inside, OFF + W * J[inside].astype(np.int64) + w] for w in range(W)], 1)
    assert np.array_equal(got[inside], seg ^ y[inside])


def test_a_change_outside_the_segment_is_not_the_blocks() -> None:
    """In a row of 15 words with 3 segments of 4 from word 2 (words 0, 1, 14 another family's): every word outside segment j
    may be anything, so a proof block binds only its own segment of the row."""
    rng = np.random.default_rng(7)
    W, ROW, OFF, G = 4, 15, 2, 3
    j = np.repeat(np.arange(G, dtype=np.uint16), 64)
    y = rng.integers(0, 1 << 16, (len(j), W), dtype=np.uint16)
    row = _set(rng.integers(0, 1 << 16, (len(j), ROW), dtype=np.uint16), y, j, W, OFF, G)
    assert not _check(y, row, j, W, OFF, G).any()
    for other in range(ROW):
        mine = (other >= OFF) & ((other - OFF) // W == j) & (other < OFF + G * W)
        flipped = row.copy()
        flipped[:, other] ^= np.uint16(0x8001)
        assert np.array_equal(_check(y, flipped, j, W, OFF, G).any(axis=1), mine), other


@pytest.mark.parametrize("W,ROW,OFF,G", SHAPES)
def test_the_select_is_its_segment_at_every_j(W: int, ROW: int, OFF: int, G: int) -> None:
    """Every 16-bit j: `SharedRowSelect_v1` is segment j for j < G and 0 for j >= G (`shared_row_read_reference`), and a
    change to any word outside segment j changes nothing."""
    rng = np.random.default_rng(ROW + 31 * G)
    row = rng.integers(0, 1 << 16, (len(J), ROW), dtype=np.uint16)
    target = bind(SR.SharedRowSelect, W=W, ROW=ROW, OFF=OFF, G=G)
    got = _words(target, row, J)
    want = SR.shared_row_read_reference(row, J, W, G, OFF)
    assert np.array_equal(got, want) and not want[J >= G].any()
    inside = J < G
    assert np.array_equal(want[inside], np.stack([row[inside, OFF + W * J[inside].astype(np.int64) + w] for w in range(W)], 1))
    for other in range(ROW):
        flipped = row.copy()
        flipped[:, other] ^= np.uint16(0x4002)
        mine = inside & (other >= OFF) & ((other - OFF) // W == J)
        changed = (_words(target, flipped, J) != got).any(axis=1)
        assert np.array_equal(changed, mine), other


def _inputs(rng, target, N: int) -> list[np.ndarray]:
    """Random words for each of ``target``'s parameters, shaped as `evaluate_words` takes them."""
    out = []
    for _, t in target.signature()[0]:
        widths = BL.elements(t)
        out.append(np.stack([rng.integers(0, 1 << w, N, dtype=np.uint64) for w in widths], axis=1))
    return out


def _as_words(v: np.ndarray, widths: list[int]) -> np.ndarray:
    """A value's elements of ``widths`` bits as 16-bit words, least significant first."""
    bits = np.concatenate([(v[:, [i]] >> np.arange(w, dtype=np.uint64)) & 1 for i, w in enumerate(widths)], axis=1)
    return (bits.reshape(len(v), -1, 16) << np.arange(16, dtype=np.uint64)).sum(axis=2).astype(np.uint16)


#: the writers table's families on small statics, each at a (ROW, OFF, G) of its shape there
PRODUCERS = [
    pytest.param(bind(BG.F2fpBf16), 7, 2, 5, id="cast"),
    pytest.param(bind(BG.ScaledMmS8CoordinateRows, K=64, R=2, **BG.GEMM_SM120_S8), 512, 0, 512, id="s8-coordinate-rows"),
    pytest.param(bind(BG.ScaledMmS8CoordinateRows, K=64, R=2, **BG.GEMM_SM120_S8), 64, 0, 64, id="s8-k-v"),
    pytest.param(bind(BR.EmbeddingCoordinate, V=16), 512, 0, 512, id="embedding"),
    pytest.param(bind(BG.PvRow, T=2, K=16, D=3, **BG.GEMM_HOPPER), 20, 1, 6, id="pv-row"),
    pytest.param(bind(SM.QkSoftmaxRowPad, T=2, D=16, **BG.GEMM_HOPPER), 32, 0, 16, id="softmax-row"),
    pytest.param(bind(BRW.Bf16AddRow, N=4), 16, 0, 4, id="residual"),
    pytest.param(bind(RN.RmsSumSqChunks, C=8, K=4), 32, 16, 4, id="mlp-sums"),
    pytest.param(bind(RN.RmsNormOutSums, C=4, S=6, G=2, N=24, EPS=1e-5), 16, 0, 4, id="norm"),
    pytest.param(bind(BQ.RowAbsMax, N=4, RELU=True), 5, 1, 3, id="amax-part"),
    pytest.param(bind(BQ.RowAbsMax, N=4, RELU=False), 2, 1, 1, id="amax"),
    pytest.param(bind(BRW.Argmax, V=4), 2, 1, 1, id="argmax"),
    pytest.param(bind(BQ.QuantChunkS8, N=4, RELU=False), 4, 0, 1, id="quant-codes-and-scale"),
]


@pytest.mark.parametrize("FN,ROW,OFF,G", PRODUCERS)
def test_the_writer_is_its_producer_then_the_check(FN, ROW: int, OFF: int, G: int) -> None:
    """On random inputs, rows with segment j set to FN's output (its bits as 16-bit words, least significant first) for about
    half the instances and j over 0..G + 2: the writer is `shared_row_check_reference` of FN's words, W words, 0 exactly
    where j < G and segment j holds them."""
    rng = np.random.default_rng(ROW + G)
    N = 96
    xs = _inputs(rng, FN, N)
    y = _as_words(_words(FN, *xs), BL.elements(FN.signature()[1]))
    W = y.shape[1]
    j = rng.integers(0, G + 3, N).astype(np.uint16)
    row = _set(rng.integers(0, 1 << 16, (N, ROW), dtype=np.uint16), y, j, W, OFF, G)
    other = rng.random(N) < 0.5
    row[other] = rng.integers(0, 1 << 16, row[other].shape, dtype=np.uint16)
    target = bind(SR.SharedRowWindow, FN=FN, ROW=ROW, OFF=OFF, G=G)
    got = _words(target, *xs, row, j).astype(np.uint16)
    assert np.array_equal(got, SR.shared_row_check_reference(y, row, j, G, OFF))
    holds = np.array([j[i] < G and np.array_equal(row[i, OFF + W * int(j[i]):OFF + W * (int(j[i]) + 1)], y[i])
                      for i in range(N)])
    assert np.array_equal(~got.any(axis=1), holds) and holds.any() and (~holds).any()
    assert ands(target) == ands(FN) + ands(bind(SR.SharedRowCheck, W=W, ROW=ROW, OFF=OFF, G=G))
    assert BL.is_boolean(target) and BL.elements(target.signature()[1]) == [16] * W


@pytest.mark.parametrize("FN,ROW,OFF,G", PRODUCERS)
def test_the_all_zero_instance_writes_zero(FN, ROW: int, OFF: int, G: int) -> None:
    """C-Flock pads every proof block with the all-zero instance, whose pinned outputs must be zero: there the writer's d is 0
    for every family that writes a shared row. The quant chunk's scale is 1.0 there, and it writes no shared row: its row is
    its own output row."""
    target = bind(SR.SharedRowWindow, FN=FN, ROW=ROW, OFF=OFF, G=G)
    zeros = [np.zeros((1, len(BL.elements(t))), np.uint64) for _, t in target.signature()[0]]
    assert _words(target, *zeros).any() == (FN.id.startswith("QuantChunkS8"))


NORM = bind(RN.RmsNormOutSums, C=4, S=6, G=2, N=24, EPS=1e-5)


@pytest.mark.parametrize("ROW,OFF", [(12, 0), (24, 12), (20, 3)])
def test_the_static_read_is_fn_on_its_words_at_no_gate(ROW: int, OFF: int) -> None:
    """The norm's chunk sums (6 f32s, 12 words) read from words [OFF, OFF + 12) of a pooled row, each f32 two words with its
    low half first: the reader is the norm of the same sums, whatever the rest of the row holds, at the norm's ANDs exactly,
    and has no index port."""
    rng = np.random.default_rng(ROW + OFF)
    N = 64
    x, w, s = _inputs(rng, NORM, N)
    row = rng.integers(0, 1 << 16, (N, ROW), dtype=np.uint16)
    row[:, OFF:OFF + 12] = _as_words(s, [32] * 6)
    target = bind(SR.SharedRowRead, FN=NORM, PORT="s", ROW=ROW, OFF=OFF, G=1)
    assert np.array_equal(_words(target, x, w, row), _words(NORM, x, w, s))
    assert [n for n, _ in target.signature()[0]] == ["x", "w", "s"] and target.signature()[0][2][1] == Array(ROW, U16)
    assert ands(target) == ands(NORM)


def test_the_selected_read_is_fn_on_segment_j_and_adds_the_select() -> None:
    """The residual add reading a from segment a_j of a 16-word row of 4 chunks: FN of that chunk for a_j < 4, of zeros for
    a_j >= 4, at FN's ANDs and the select's."""
    FN = bind(BRW.Bf16AddRow, N=4)
    rng = np.random.default_rng(3)
    N = 128
    a, b = _inputs(rng, FN, N)
    row = rng.integers(0, 1 << 16, (N, 16), dtype=np.uint16)
    j = rng.integers(0, 7, N).astype(np.uint16)
    target = bind(SR.SharedRowRead, FN=FN, PORT="a", ROW=16, OFF=0, G=4)
    assert [n for n, _ in target.signature()[0]] == ["a", "b", "a_j"]
    want = _words(FN, SR.shared_row_read_reference(row, j, 4, 4).astype(np.uint64), b)
    assert np.array_equal(_words(target, row, b, j), want) and (j >= 4).any()
    assert ands(target) == ands(FN) + ands(bind(SR.SharedRowSelect, W=4, ROW=16, OFF=0, G=4))


def test_a_norm_reads_and_writes_shared_rows() -> None:
    """The norm's served shape, small: its sums read at a static offset of the pooled sums row, its x selected from a residual
    row, its output written to segment j of the norm row. The writer of the readers is the check of the norm of the read
    values, and its ANDs are the norm's, x's select and the check's."""
    sums = bind(SR.SharedRowRead, FN=NORM, PORT="s", ROW=24, OFF=12, G=1)
    xs = bind(SR.SharedRowRead, FN=sums, PORT="x", ROW=16, OFF=0, G=4)
    target = bind(SR.SharedRowWindow, FN=xs, ROW=16, OFF=0, G=4)
    assert [n for n, _ in target.signature()[0]] == ["x", "w", "s", "x_j", "row", "j"]
    rng = np.random.default_rng(11)
    N = 64
    _, w, s = _inputs(rng, NORM, N)
    xrow = rng.integers(0, 1 << 16, (N, 16), dtype=np.uint16)
    srow = rng.integers(0, 1 << 16, (N, 24), dtype=np.uint16)
    srow[:, 12:] = _as_words(s, [32] * 6)
    i = rng.integers(0, 16, N).astype(np.uint16)
    xj, j = i % 4, i % 4
    y = _words(NORM, SR.shared_row_read_reference(xrow, xj, 4, 4).astype(np.uint64), w, s).astype(np.uint16)
    nrow = _set(rng.integers(0, 1 << 16, (N, 16), dtype=np.uint16), y, j, 4, 0, 4)
    bad = rng.random(N) < 0.3
    nrow[bad, 4 * j[bad].astype(np.int64)] ^= np.uint16(1)
    got = _words(target, xrow, w, srow, xj, nrow, j).astype(np.uint16)
    assert got.shape == (N, 4) and np.array_equal(got.any(axis=1), bad) and bad.any()
    assert ands(target) == ands(NORM) + ands(bind(SR.SharedRowSelect, W=4, ROW=16, OFF=0, G=4)) + ands(
        bind(SR.SharedRowCheck, W=4, ROW=16, OFF=0, G=4))


def test_the_costs_at_a_1_kib_row() -> None:
    """(G - 1)·16·W tree ANDs, the range bit's and 16·W to set d's bits past G: at a 1 KiB row (ROW = 512) 8,198 for 512
    segments of 1 word, 8,203 for 16 of 32, 8,204 for 8 of 64 and 8,205 for 4 of 128, 4,109 for the MLP sums' 4 of 64 from
    word 256; at a 256 B row, 2,056 for 128 of 1 (`MoeSelectBf16x128_v2`, a select alone, is 2,072) and 2,062 for 2 of 64.
    The select is the tree, 16·W to zero it past G and the range bit's."""
    check = lambda W, ROW, OFF, G: ands(bind(SR.SharedRowCheck, W=W, ROW=ROW, OFF=OFF, G=G))  # noqa: E731
    select = lambda W, ROW, OFF, G: ands(bind(SR.SharedRowSelect, W=W, ROW=ROW, OFF=OFF, G=G))  # noqa: E731
    assert check(1, 512, 0, 512) == 511 * 16 + 6 + 16 == 8198
    assert check(32, 512, 0, 16) == 15 * 16 * 32 + 11 + 16 * 32 == 8203
    assert check(64, 512, 0, 8) == 7 * 16 * 64 + 12 + 16 * 64 == 8204
    assert check(128, 512, 0, 4) == 3 * 16 * 128 + 13 + 16 * 128 == 8205
    assert check(64, 512, 256, 4) == 3 * 16 * 64 + 13 + 16 * 64 == 4109
    assert check(1, 128, 0, 128) == 127 * 16 + 8 + 16 == 2056
    assert check(64, 128, 0, 2) == 64 * 16 + 14 + 16 * 64 == 2062
    assert ands(BM.select_bf16(128)) == 2072
    assert select(128, 512, 0, 4) == 3 * 16 * 128 + 16 * 128 + 13 == 8205
    assert select(64, 512, 0, 8) == 7 * 16 * 64 + 16 * 64 + 12 == 8204
    assert select(32, 512, 0, 16) == 15 * 16 * 32 + 16 * 32 + 11 == 8203


def test_j_and_port_j_are_position_kind_ports_and_fns_ports_carry_over() -> None:
    """j and PORT_j are run inputs of kind position, public as a key count is; a caller passing j a value it didn't declare as
    a position is refused (`ports.E_PORT_KIND`). A producer's own position port (`QkSoftmaxRowPad_v1`'s n) is the wrapper's
    too, and a reader's."""
    target = bind(SR.SharedRowWindow, FN=bind(BG.F2fpBf16), ROW=4, OFF=0, G=4)
    facts = PT.facts(Program(target).root)
    assert facts.node(2).one.kind == "position" and facts.node(0).one.kind != "position"
    params, ret = target.signature()
    undeclared = CompositeDefinition("SharedRowWindowTestUndeclared", 1, (), lambda S: (params, ret),
                                     lambda B, S, *a: B.call(target, *a), register=False)
    with pytest.raises(PT.PortError, match="takes kind=position") as e:
        Program(undeclared)
    assert e.value.code == "E_PORT_KIND"
    qk = bind(SM.QkSoftmaxRowPad, T=2, D=16, **BG.GEMM_HOPPER)
    read = bind(SR.SharedRowRead, FN=qk, PORT="q", ROW=64, OFF=0, G=4)
    padded = bind(SR.SharedRowWindow, FN=read, ROW=32, OFF=0, G=16)
    names = [n for n, _ in padded.signature()[0]]
    assert names == ["q", "k", "n", "q_j", "row", "j"]
    facts = PT.facts(Program(padded).root)
    assert [facts.node(i).one.kind == "position" for i in range(len(names))] == [False, False, True, True, False, True]


_BITS16 = Array(16, BL.BIT)


def _clash(name: str, t=_BITS16) -> CompositeDefinition:
    return CompositeDefinition(f"SharedRowTestClash{name}", 1, (), lambda S: (((name, t), ("a", U16)), U16),
                               lambda B, S, x, a: a, register=False)


def test_a_word_fn_segments_past_the_row_or_j_and_reserved_names_are_refused() -> None:
    cast = bind(BG.F2fpBf16)
    word = bind(G.ScaledMmS8CoordinateRows, K=64, R=2, **G.GEMM_SM120_S8)
    with pytest.raises(ValueError, match="Boolean form"):
        bind(SR.SharedRowWindow, FN=word, ROW=128, OFF=0, G=128)
    with pytest.raises(ValueError, match="Boolean form"):
        bind(SR.SharedRowRead, FN=word, PORT="sx", ROW=2, OFF=0, G=1)
    for static in ("ROW", "G"):
        for v in (0, True, 2.0):
            with pytest.raises(ValueError, match="at least 1"):
                bind(SR.SharedRowWindow, FN=cast, **{"ROW": 4, "OFF": 0, "G": 4, static: v})
    with pytest.raises(ValueError, match="at least 0"):
        bind(SR.SharedRowWindow, FN=cast, ROW=4, OFF=-1, G=4)
    with pytest.raises(ValueError, match="past 65536"):
        bind(SR.SharedRowWindow, FN=cast, ROW=SR.J_LIMIT + 1, OFF=0, G=SR.J_LIMIT + 1)
    pv = bind(BG.PvRow, T=2, K=16, D=3, **BG.GEMM_HOPPER)
    with pytest.raises(ValueError, match="past the row's ROW = 17"):
        bind(SR.SharedRowWindow, FN=pv, ROW=17, OFF=0, G=6)
    with pytest.raises(ValueError, match="past the row's ROW = 18"):
        bind(SR.SharedRowWindow, FN=pv, ROW=18, OFF=1, G=6)
    assert bind(SR.SharedRowWindow, FN=pv, ROW=19, OFF=1, G=6).signature()[0][-2][1] == Array(19, U16)
    with pytest.raises(ValueError, match="at least 1"):
        bind(SR.SharedRowCheck, W=0, ROW=2, OFF=0, G=2).signature()
    with pytest.raises(ValueError, match="past the row's ROW = 12"):
        bind(SR.SharedRowCheck, W=4, ROW=12, OFF=1, G=3).signature()
    with pytest.raises(ValueError, match="past the row's ROW = 12"):
        bind(SR.SharedRowSelect, W=4, ROW=12, OFF=1, G=3).signature()
    for name in ("row", "j"):
        with pytest.raises(ValueError, match="the wrapper's own"):
            bind(SR.SharedRowWindow, FN=_clash(name), ROW=2, OFF=0, G=2)


def test_a_read_of_no_parameter_of_part_words_or_past_the_row_is_refused() -> None:
    with pytest.raises(ValueError, match="is one of FN"):
        bind(SR.SharedRowRead, FN=NORM, PORT="y", ROW=12, OFF=0, G=1)
    with pytest.raises(ValueError, match="is one of FN"):
        bind(SR.SharedRowRead, FN=NORM, PORT=0, ROW=12, OFF=0, G=1)
    with pytest.raises(ValueError, match="not bits in whole 16-bit words"):
        bind(SR.SharedRowRead, FN=_clash("b", Array(8, BL.BIT)), PORT="b", ROW=2, OFF=0, G=1)
    with pytest.raises(ValueError, match="past the row's ROW = 12"):
        bind(SR.SharedRowRead, FN=NORM, PORT="s", ROW=12, OFF=1, G=1)
    with pytest.raises(ValueError, match="past the row's ROW = 23"):
        bind(SR.SharedRowRead, FN=NORM, PORT="s", ROW=23, OFF=0, G=2)
    with pytest.raises(ValueError, match="the reader's own"):
        bind(SR.SharedRowRead, FN=_clash("a_j"), PORT="a", ROW=2, OFF=0, G=2)
