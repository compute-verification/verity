"""FlashMLA's sparse FP8 decode at its V32 instance (`registry.flashmla_dsv32`, `numerics.flashmla_dsv32_twin`).

* the row on vectors worked by hand: an f32 scale is rounded to bf16 (ties to even) before it multiplies, so the row is not
  the E4M3 value times the f32 scale; an invalid row is the zero row;
* a split's end, the weights and the weighing on vectors worked by hand;
* a token on vectors worked by hand: under a zero query every valid row weighs the same, so one valid row comes back whole,
  through one split or through the combine, and no valid row gives +0;
* validity is the index alone, and no byte of an invalid row reaches an output;
* the units are the registered arithmetic: a block is the registered block's first DV columns, the weights and the weighing
  in column blocks the registered combine on a sink of -inf, and a head the registered head at the key's width on a sink of
  -inf, cut to the value's width;
* a head's and a token's bodies hold their cut's calls and nothing else;
* the twin (the arithmetic over numpy words) word for word against the reference on every binding it registers;
* the splits a step's token count gives a token's 32 blocks.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate, self_check
from verity.definitions.numerics import flashmla_twin as T
from verity.definitions.numerics import flashmla_dsv32_twin as TV
from verity.definitions.registry import flashmla as F
from verity.definitions.registry import flashmla_dsv32 as V

U8, U16, U32 = np.uint8, np.uint16, np.uint32
DEFS = {d.id: d for d in V.FLASHMLA_V32_COMPOSITES}
TWIN_TARGETS = [t for family in TV.KERNELS for t in TV._targets(DEFS, family)]
DOT = V.FLASHMLA_GLM53["DOT"]
NINF = 0xFF800000
#: an index that is not -1
ROW = [7]
GONE = [0xFFFFFFFF]
#: small statics: a 48-word key whose first 32 words are the value, 16-row blocks
SMALL = dict(DN=32, DR=16, QT=16, BN=16, DOT=DOT, SCALE=0.25)
HEAD = dict(DK=48, DV=32, BN=16, DOT=DOT, SCALE=0.25)


def _ev(target, *args) -> np.ndarray:
    return np.asarray(evaluate(target, *[np.asarray(a).reshape(-1) for a in args])).reshape(-1)


def _f(*x) -> np.ndarray:
    return np.array(x, dtype=np.float32).view(U32)


def _is_nan16(w) -> np.ndarray:
    w = np.asarray(w, dtype=U16)
    return ((w & U16(0x7F80)) == U16(0x7F80)) & ((w & U16(0x007F)) != 0)


def _token(rng, n: int):
    """A token's rows at `SMALL`: finite bytes, scales near 1 and RoPE words."""
    nope = rng.integers(0, 256, (n, 32)).astype(U8)
    nope[(nope & 0x7F) == 0x7F] ^= 1
    scale = (1.0 + rng.random((n, 2))).astype(np.float32).view(U32)
    rope = ((rng.standard_normal((n, 16)).astype(np.float32)).view(U32) >> U32(16)).astype(U16)
    return nope, scale, rope


# --- the row ---------------------------------------------------------------------------------------------------------------------

def test_a_row_by_hand():
    """E4M3 0x3E is 1.75 (exponent field 7, mantissa 6/8) and 0xBE its negative.  Each scale is first rounded to bf16:

        0x3F808008 = 1 + 2^-8 + 2^-20   over the tie         -> 0x3F81 = 1 + 2^-7
        0x3F808000 = 1 + 2^-8           a tie, 0x3F80 even   -> 0x3F80 = 1
        0x3F818000 = 1 + 2^-7 + 2^-8    a tie, 0x3F81 odd    -> 0x3F82 = 1 + 2^-6

    and the bf16 product is rounded once: 1.75 (1 + 2^-7) = 1 + 97.75 / 128 -> 1 + 98 / 128 = 0x3FE2; 1.75 x 1 = 0x3FE0;
    1.75 (1 + 2^-6) = 1 + 99.5 / 128, a tie -> 1 + 100 / 128 = 0x3FE4.  (The E4M3 value times the f32 scale 0x3F808008,
    rounded once, is 1 + 96.875 / 128 -> 0x3FE1: the other read of the same row.)  The RoPE words pass as stored, and the
    last word out is the row's validity."""
    row = bind(V.FlashMlaV32KvRow, DN=6, DR=2, QT=2)
    nope = np.array([0x3E, 0xBE, 0x3E, 0x3E, 0x3E, 0x00], dtype=U8)
    scale = np.array([0x3F808008, 0x3F808000, 0x3F818000], dtype=U32)
    rope = np.array([0x1234, 0xFFC0], dtype=U16)
    assert _ev(row, nope, scale, rope, ROW).tolist() == [0x3FE2, 0xBFE2, 0x3FE0, 0x3FE0, 0x3FE4, 0x0000, 0x1234, 0xFFC0, 1]
    assert _ev(row, nope, scale, rope, [0]).tolist() == [0x3FE2, 0xBFE2, 0x3FE0, 0x3FE0, 0x3FE4, 0x0000, 0x1234, 0xFFC0, 1]
    assert _ev(row, nope, scale, rope, GONE).tolist() == [0] * 9
    assert TV.kv_rows(nope[None], scale[None], rope[None], [True], 2).reshape(-1).tolist() == [
        0x3FE2, 0xBFE2, 0x3FE0, 0x3FE0, 0x3FE4, 0x0000, 0x1234, 0xFFC0]


def test_a_rows_scales_at_their_ends_by_hand():
    """One scale an element.  FLT_MAX rounds to the bf16 +inf, so 1.75 becomes +inf; FLT_MIN is the bf16 0x0080 and 1.75 of
    it 0x00E0; the f32 subnormals 2^-149 and 2^-134 (a tie between 0 and the bf16 2^-133, 0 even) round to +0, and 0x00018000
    (1.5 x 2^-133, a tie, 0x0001 odd) to 0x0002 = 2^-132, whose 1.75-fold 3.5 x 2^-133 is a tie that rounds to 0x0004.  A NaN
    byte, 0 x inf and -0 x -inf are NaNs."""
    row = bind(V.FlashMlaV32KvRow, DN=8, DR=1, QT=1)
    nope = np.array([0x3E, 0x3E, 0x3E, 0x3E, 0x3E, 0x7F, 0x00, 0x80], dtype=U8)
    scale = np.array([0x7F7FFFFF, 0x00800000, 0x00000001, 0x00008000, 0x00018000, 0x3F800000, 0x7F800000, 0xFF800000], dtype=U32)
    out = _ev(row, nope, scale, [0x8000], ROW)
    assert out[:5].tolist() == [0x7F80, 0x00E0, 0x0000, 0x0000, 0x0004]
    assert _is_nan16(out[5:8]).all()
    assert out[8:].tolist() == [0x8000, 1]
    assert np.array_equal(out[:9], TV.kv_rows(nope[None], scale[None], np.array([[0x8000]], dtype=U16), [True], 1).reshape(-1))


def test_the_rows_statics():
    with pytest.raises(ValueError, match="QT=3 must divide DN=8"):
        _ = bind(V.FlashMlaV32KvRow, DN=8, DR=1, QT=3).body
    with pytest.raises(ValueError, match="first DV=64 of DK=48"):
        _ = bind(V.FlashMlaV32DecodeHead, SPLITS=(0, 1), **{**HEAD, "DV": 64}).body
    with pytest.raises(ValueError, match="first DV=64 of DK=48"):
        _ = bind(V.FlashMlaV32Block, FIRST=True, **{**HEAD, "DV": 64}).body
    with pytest.raises(ValueError, match="SPLITS"):
        _ = bind(V.FlashMlaV32DecodeHead, SPLITS=(0, 2, 2), **HEAD).body
    with pytest.raises(ValueError, match="NS=1 splits"):
        _ = bind(V.FlashMlaV32CombineWeights, NS=1).body
    with pytest.raises(ValueError, match="NS=1 splits"):
        _ = bind(V.FlashMlaV32CombineValues, NS=1, G=4).body
    with pytest.raises(ValueError, match="DV=384 must be a multiple of 256"):
        V.combine_columns(384)
    assert [V.combine_columns(dv) for dv in (1, 32, 256, 512, 768)] == [1, 32, 256, 256, 256]


# --- a split's end and the combine -------------------------------------------------------------------------------------------

def test_a_splits_end_by_hand():
    """The state is -m, the four threads' sums and O.  With sums 0.5, 0.25, 0.25 and 1 the total is 2: a partial is O / 2 and
    its lse log2f(2) + m = 1 + 3 = 4 at m = 3 (-m = -3); the only split's output is O / (2 + exp2f(-inf - 3)) = O / 2 in bf16.
    With every sum 0 (no valid row) O is multiplied by +0, so each word is a zero of O's sign, and the partial's lse is -inf."""
    part, whole = bind(V.FlashMlaV32SplitEnd, DV=4, PARTIAL=True), bind(V.FlashMlaV32SplitEnd, DV=4, PARTIAL=False)
    l, o = _f(0.5, 0.25, 0.25, 1.0), _f(1.0, -3.0, 0.0, 2.5)
    assert _ev(part, _f(-3.0), l, o).tolist() == _f(0.5, -1.5, 0.0, 1.25, 4.0).tolist()
    assert _ev(whole, _f(-3.0), l, o).tolist() == [0x3F00, 0xBFC0, 0x0000, 0x3FA0]
    assert _ev(part, _f(-3.0), _f(0, 0, 0, 0), o).tolist() == [0, 0x80000000, 0, 0, NINF]
    assert _ev(whole, _f(-3.0), _f(0, 0, 0, 0), o).tolist() == [0, 0x8000, 0, 0]


def test_the_combines_weights_by_hand():
    """Two splits of lse 0: the max is 0, each exp2f(0 - 0) = 1, the sum 2 and g = log2f(2) + 0 = 1, so each weighs
    exp2f(0 - 1) = 1/2.  An empty split (lse -inf) beside one of lse 3 weighs exp2f(-inf) = 0 and the other exp2f(3 - 3) = 1.
    With every split empty the max is taken as 0, the sum is 0, g is +inf and every weight exp2f(-inf) = 0."""
    two, three = bind(V.FlashMlaV32CombineWeights, NS=2), bind(V.FlashMlaV32CombineWeights, NS=3)
    assert _ev(two, _f(0.0, 0.0)).tolist() == _f(0.5, 0.5).tolist()
    assert _ev(two, [NINF, _f(3.0)[0]]).tolist() == _f(0.0, 1.0).tolist()
    assert _ev(three, [NINF, NINF, NINF]).tolist() == [0, 0, 0]
    assert _ev(three, _f(5.0, 5.0, 4.0)).tolist() == _f(0.4, 0.4, 0.2).tolist()


def test_the_combines_values_by_hand():
    """From +0, one FFMA a split: 1/2 x 1 + 0 = 1/2 then 1/2 x 2 + 1/2 = 3/2; 1/2 x 3 then 1/2 x -3 + 3/2 = +0."""
    values = bind(V.FlashMlaV32CombineValues, NS=2, G=2)
    assert _ev(values, _f(0.5, 0.5), _f(1.0, 3.0, 2.0, -3.0)).tolist() == [0x3FC0, 0x0000]
    assert _ev(values, _f(0.0, 0.0), _f(1.0, 3.0, 2.0, -3.0)).tolist() == [0, 0]


# --- a token ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("splits", [(0, 2), (0, 1, 2), (0, 1, 2, 3)])
def test_a_zero_query_returns_its_one_valid_row(splits):
    """With q = 0 every score is 0: the block's max is 0, a valid row's weight exp2f(0) = 1 and an invalid row's
    exp2f(-inf) = 0, so the sum is 1, the value sum is the row (+0 + 1 x v, a -0 becoming +0) and 1 / (1 + exp2f(-inf)) = 1.
    Through the combine the row's split has lse log2f(1) + 0 = 0 and the empty ones -inf, so its weight is exp2f(0 - 0) = 1
    and theirs 0.  With no valid row every word is +0."""
    rng = np.random.default_rng(len(splits))
    n = splits[-1] * 16
    dec = bind(V.FlashMlaV32Decode, NH=2, SPLITS=splits, **SMALL)
    nope, scale, rope = _token(rng, n)
    nope[n - 3, :4] = [0x80, 0x00, 0x3E, 0xBE]
    q = np.zeros((2, 48), dtype=U16)
    q[1, ::2] = 0x8000
    for r in (0, 5, n - 3):
        idx = np.full(n, 0xFFFFFFFF, dtype=U32)
        idx[r] = 7
        row = TV.kv_rows(nope[r:r + 1], scale[r:r + 1], rope[r:r + 1], [True], 16)[0, :32]
        want = np.where(row == 0x8000, U16(0), row)
        got = _ev(dec, q, nope, scale, rope, idx).reshape(2, 32)
        assert np.array_equal(got[0], want) and np.array_equal(got[1], want), r
    assert not _ev(dec, q, nope, scale, rope, np.full(n, 0xFFFFFFFF, dtype=U32)).any()


def test_two_equal_rows_average_to_the_row():
    """q = 0 over two valid copies of one row: the weights are 1 and 1, the sum 2, and (v + v) / 2 is v."""
    rng = np.random.default_rng(7)
    dec = bind(V.FlashMlaV32Decode, NH=1, SPLITS=(0, 1), **SMALL)
    nope, scale, rope = _token(rng, 16)
    nope[9], scale[9], rope[9] = nope[2], scale[2], rope[2]
    idx = np.full(16, 0xFFFFFFFF, dtype=U32)
    idx[[2, 9]] = [40, 41]
    row = TV.kv_rows(nope[2:3], scale[2:3], rope[2:3], [True], 16)[0, :32]
    assert np.array_equal(_ev(dec, np.zeros(48, dtype=U16), nope, scale, rope, idx), np.where(row == 0x8000, U16(0), row))


def test_validity_is_the_index_alone():
    """A row counts iff its index is not -1: 0, the largest positive and the most negative index are rows like any other, the
    index's value reaches nothing else, and no byte, scale or RoPE word of an invalid row reaches an output.  Under a zero
    query, where every valid row weighs the same, one row fewer is another output."""
    rng = np.random.default_rng(11)
    dec = bind(V.FlashMlaV32Decode, NH=2, SPLITS=(0, 1, 2), **SMALL)
    nope, scale, rope = _token(rng, 32)
    q = (((rng.standard_normal((2, 48)) / 32).astype(np.float32)).view(U32) >> U32(16)).astype(U16)
    idx = rng.integers(0, 1 << 20, 32).astype(U32)
    idx[[0, 1, 2]] = [0, 0x7FFFFFFF, 0x80000000]
    gone = [4, 17, 18, 31]
    idx[gone] = 0xFFFFFFFF
    assert TV.row_valid(idx).tolist() == [r not in gone for r in range(32)]
    base = _ev(dec, q, nope, scale, rope, idx)
    other = idx.copy()
    other[[0, 1, 2, 9]] = [123, 5, 0xFFFFFFFE, 0]
    assert np.array_equal(_ev(dec, q, nope, scale, rope, other), base)
    poison = [a.copy() for a in (nope, scale, rope)]
    poison[0][gone], poison[1][gone], poison[2][gone] = 0x7F, 0x7FC00000, 0x7FC0
    assert np.array_equal(_ev(dec, q, *poison, idx), base)
    dropped, zero = idx.copy(), np.zeros_like(q)
    dropped[0] = 0xFFFFFFFF
    assert not np.array_equal(_ev(dec, zero, nope, scale, rope, dropped), _ev(dec, zero, nope, scale, rope, idx))


# --- the units against the registered arithmetic -----------------------------------------------------------------------------

@pytest.mark.parametrize("first", [True, False])
def test_a_block_is_the_registered_block_on_the_values_columns(first):
    """`FlashMlaV32Block_v1` is `FlashMlaBlock_v1` at D = DK with O's last DK - DV columns left out: the max, the four sums and
    the first DV columns of O are the registered block's, whatever its other columns of O held before."""
    rng = np.random.default_rng(3 + first)
    mine = bind(V.FlashMlaV32Block, FIRST=first, **HEAD)
    theirs = bind(F.FlashMlaBlock, D=48, BN=16, FIRST=first, DOT=DOT, SCALE=0.25)
    for args in zip(*TV._sample_block(rng, 6, None, FIRST=first, **HEAD)):
        wide = list(args[:2]) if first else [*args[:4], np.concatenate([args[4], T._f32(rng, (16,))])]
        assert np.array_equal(_ev(mine, *args), _ev(theirs, *wide)[:5 + 32])


@pytest.mark.parametrize("ns, g", [(2, 32), (3, 16), (33, 8)])
def test_the_weights_and_the_weighing_are_the_registered_combine_on_no_sink(ns, g):
    """`FlashMlaV32CombineValues_v1` on each block of G columns, over `FlashMlaV32CombineWeights_v1`'s weights, side by side:
    `FlashMlaCombine_v1` at D = 32 on a sink of -inf, word for word."""
    rng = np.random.default_rng(ns)
    weights, values = bind(V.FlashMlaV32CombineWeights, NS=ns), bind(V.FlashMlaV32CombineValues, NS=ns, G=g)
    theirs = bind(F.FlashMlaCombine, NS=ns, D=32)
    for lse, oacc, _ in zip(*T._sample_combine(rng, 8, None, NS=ns, D=32)):
        w = _ev(weights, lse)
        got = np.concatenate([_ev(values, w, oacc[:, c:c + g]) for c in range(0, 32, g)])
        assert np.array_equal(got, _ev(theirs, lse, oacc, [NINF]))


@pytest.mark.parametrize("splits", [(0, 2), (0, 1, 2), (0, 2, 3, 4)])
def test_a_head_is_the_registered_head_on_no_sink_cut_to_the_value(splits):
    """`FlashMlaV32DecodeHead_v1` is `FlashMlaDecodeHead_v1` at D = DK on a sink of -inf, its first DV words: the registered
    split and combine at the key's width compute the value's words and DK - DV more that this instance never forms."""
    rng = np.random.default_rng(13 + len(splits))
    mine = bind(V.FlashMlaV32DecodeHead, SPLITS=splits, **HEAD)
    theirs = bind(F.FlashMlaDecodeHead, SPLITS=splits, D=48, BN=16, DOT=DOT, SCALE=0.25)
    for q, k in zip(*TV._sample_head(rng, 6, None, SPLITS=splits, **HEAD)):
        assert np.array_equal(_ev(mine, q, k), _ev(theirs, q, k, [NINF])[:32])


def test_a_head_past_the_combines_columns_is_weighed_in_blocks_of_them():
    """A head 512 words wide leaves the weighing to two units of `COMBINE_G` columns: the words are the twin's."""
    rng = np.random.default_rng(5)
    wide = dict(DK=528, DV=512, BN=16, DOT=DOT, SCALE=0.0625)
    head = bind(V.FlashMlaV32DecodeHead, SPLITS=(0, 1, 2), **wide)
    assert dict((u.id, n) for u, n in V.head_cut(head))["FlashMlaV32CombineValues_v1{NS=2,G=256}"] == 2
    q, k = (a[0] for a in TV._sample_head(rng, 1, None, SPLITS=(0, 1, 2), **wide))
    assert np.array_equal(_ev(head, q, k), TV._head_fn(q, k, SPLITS=(0, 1, 2), **wide))


# --- the cut -----------------------------------------------------------------------------------------------------------------

def _calls(target) -> dict:
    """The Definitions ``target``'s body calls, each with how many times."""
    out: dict = {}
    for node in target.body.nodes:
        out[node.fn.id] = out.get(node.fn.id, 0) + node.n
    return out


@pytest.mark.parametrize("splits", [(0, 4), (0, 1, 2), (0, 2, 3, 4), (0, 1, 2, 3, 4)])
def test_a_heads_body_is_its_cut(splits):
    """A head's body holds one call for each unit of `head_cut` and no gate of its own, so the units are the head."""
    head = bind(V.FlashMlaV32DecodeHead, SPLITS=splits, **HEAD)
    cut = {u.id: n for u, n in V.head_cut(head)}
    assert _calls(head) == cut
    ns = len(splits) - 1
    blk = "FlashMlaV32Block_v1{DK=48,DV=32,BN=16,FIRST=%s,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.25}"
    want = {blk % True: ns, "FlashMlaV32SplitEnd_v1{DV=32,PARTIAL=%s}" % (ns > 1): ns}
    if splits[-1] > ns:
        want[blk % False] = splits[-1] - ns
    if ns > 1:
        want |= {"FlashMlaV32CombineWeights_v1{NS=%d}" % ns: 1, "FlashMlaV32CombineValues_v1{NS=%d,G=32}" % ns: 1}
    assert cut == want


def test_a_tokens_body_is_its_cut():
    """A token's body is its rows and its heads; its cut is the rows and each head's cut.  At GLM-5.3's statics a step of up to
    three tokens cuts a token into 2048 rows and, for each of 8 heads, 32 one-block splits, their ends, the weights and two
    blocks of 256 columns."""
    dec = bind(V.FlashMlaV32Decode, NH=2, SPLITS=(0, 2, 3), **SMALL)
    assert _calls(dec) == {"FlashMlaV32KvRow_v1{DN=32,DR=16,QT=16}": 48,
                           "FlashMlaV32DecodeHead_v1{SPLITS=(0,2,3),DK=48,DV=32,BN=16,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.25}": 2}
    head = bind(V.FlashMlaV32DecodeHead, SPLITS=(0, 2, 3), **HEAD)
    assert V.decode_cut(dec) == [(bind(V.FlashMlaV32KvRow, DN=32, DR=16, QT=16), 48)] + [(u, 2 * n) for u, n in V.head_cut(head)]
    glm = bind(V.FlashMlaV32Decode, **{**V.FLASHMLA_GLM53, "SPLITS": TV.step_splits(1, topk=V.GLM53_TOPK)})
    blk = "FlashMlaV32Block_v1{DK=576,DV=512,BN=64,FIRST=True,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.0625}"
    assert [(u.id, n) for u, n in V.decode_cut(glm)] == [
        ("FlashMlaV32KvRow_v1{DN=512,DR=64,QT=128}", 2048), (blk, 256), ("FlashMlaV32SplitEnd_v1{DV=512,PARTIAL=True}", 256),
        ("FlashMlaV32CombineWeights_v1{NS=32}", 8), ("FlashMlaV32CombineValues_v1{NS=32,G=256}", 16)]
    one = bind(V.FlashMlaV32Decode, **{**V.FLASHMLA_GLM53, "SPLITS": TV.step_splits(67, topk=V.GLM53_TOPK)})
    assert [(u.id, n) for u, n in V.decode_cut(one)] == [
        ("FlashMlaV32KvRow_v1{DN=512,DR=64,QT=128}", 2048), (blk, 8), (blk.replace("FIRST=True", "FIRST=False"), 248),
        ("FlashMlaV32SplitEnd_v1{DV=512,PARTIAL=False}", 8)]
    with pytest.raises(ValueError, match="head_cut cuts"):
        V.head_cut(dec)
    with pytest.raises(ValueError, match="decode_cut cuts"):
        V.decode_cut(head)


# --- the twin and the schedule -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", TWIN_TARGETS, ids=[t.id for t in TWIN_TARGETS])
def test_the_twin_is_the_reference(target):
    sc = self_check(target, kernel=T.KERNEL, n=24, seed=0)
    assert sc.ok, str(sc)
    assert sc.declined == 0, str(sc)


def test_the_twin_claims_only_the_hopper_step():
    from verity.definitions.numerics import kernel_registry as KR
    from verity.definitions.ops.prims import AmpereBF16TcDot16

    assert KR.covers(bind(V.FlashMlaV32DecodeHead, SPLITS=(0, 1), **HEAD), T.KERNEL) is None
    assert KR.covers(bind(V.FlashMlaV32DecodeHead, SPLITS=(0, 1), **{**HEAD, "DOT": AmpereBF16TcDot16}),
                     T.KERNEL) == f"DOT={AmpereBF16TcDot16.id}"


def test_a_steps_token_count_gives_its_splits():
    """One call holds the step's T tokens as one request, so the planner has max(132 / T, 1) partitions for a token's 32 blocks
    and 5 blocks of fixed overhead: a partition's payload is ceil(37 / parts) + 5 blocks, less the 5 a request costs."""
    first = {1: 32, 4: 16, 7: 11, 11: 8, 14: 7, 17: 6, 19: 5, 23: 4, 27: 4, 34: 3, 45: 2, 67: 1}
    cuts = {t: TV.step_splits(t, topk=V.GLM53_TOPK) for t in range(1, 140)}
    assert {t: len(cuts[t]) - 1 for t in first} == first
    assert sorted(t for t in cuts if t == 1 or cuts[t] != cuts[t - 1]) == sorted(first)
    assert cuts[1] == tuple(range(33)) and cuts[3] == cuts[1]
    assert cuts[14] == (0, 5, 10, 15, 20, 25, 30, 32) and cuts[23] == (0, 8, 16, 24, 32) and cuts[27] == (0, 10, 20, 30, 32)
    assert cuts[66] == (0, 19, 32) and cuts[67] == cuts[139] == (0, 32)
    assert all(c[0] == 0 and c[-1] == 32 and list(c) == sorted(set(c)) for c in cuts.values())


def test_glm53_statics():
    """GLM-5.3's row is 512 E4M3 bytes, four f32 scales and 64 bf16 words (656 bytes); a token reads 2048 rows."""
    S = V.FLASHMLA_GLM53
    assert (S["NH"], S["DN"], S["DR"], S["QT"], S["BN"], S["SCALE"]) == (8, 512, 64, 128, 64, 0.0625)
    assert S["DN"] + 4 * (S["DN"] // S["QT"]) + 2 * S["DR"] == 656
    assert V.GLM53_TOPK == 32 * S["BN"]
    assert F.scale_log2_bits(S["SCALE"]) == T.scale_log2(S["SCALE"])
    dec = bind(V.FlashMlaV32Decode, **{**S, "SPLITS": TV.step_splits(14, topk=V.GLM53_TOPK)})
    assert dec.out_leaves == 8 * 512
