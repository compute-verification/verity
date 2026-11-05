"""`verity.ml.circuits.ops.quant` and `ops.boolean.quant`: the s8 quantizer's proof blocks are torch's per-row s8 quantize of the
row, relu'd first under RELU (here in numpy float32, the same IEEE operations), on random rows and on the edge rows (all zero, all
-0, all negative, ties at .5, NaN with saturating neighbours, ±inf, subnormal); a row wider than one proof block, as RowAbsMax
over the row then a chunk proof block a chunk, is the row proof block; the scale reads any amax word as
``torch.where(amax > 0, amax / 127, 1)``; each Boolean proof block computes its word Definition, at small widths; and `bind`
refuses statics outside the family."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import quant as Q
from verity.ml.circuits.ops.boolean import quant as BQ

H = 128


def _f32(xb: np.ndarray) -> np.ndarray:
    return (xb.astype(np.uint32) << 16).view(np.float32)


def reference(xb: np.ndarray, relu: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``quantize`` of bf16 rows (u16 words), relu'd first under ``relu``: (the s8 codes as u8, the f32 scales' words, the
    amaxes as f32)."""
    x = _f32(xb)
    if relu:
        x = np.where(x < 0, np.float32(0), x)
    with np.errstate(all="ignore"):
        amax = np.abs(x).max(axis=1)
        scale = np.where(amax > 0, amax / np.float32(127), np.float32(1)).astype(np.float32)
        q = np.rint(x / scale[:, None])
    codes = np.where(np.isnan(q), 0, np.clip(q, -128, 127)).astype(np.int8)
    return codes.view(np.uint8), scale.view(np.uint32), amax


def _bf16(f) -> np.ndarray:
    return (np.asarray(f, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def rows(h: int, rng) -> np.ndarray:
    """Random rows at three magnitudes, then the edge rows."""
    out = [_bf16(rng.normal(0, s, (4, h))) for s in (0.05, 1.0, 300.0)]
    edge = np.zeros((6, h), np.float32)
    edge[1] = -0.0
    edge[2] = -np.abs(rng.normal(0, 1, h))
    ties = [127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 125.5, -126.5, 64.5, -63.5, 3.0]
    edge[3, :min(h, len(ties))] = ties[:h]
    edge[4] = rng.normal(0, 1, h)
    edge[4, :min(h, 8)] = [200, np.nan, -300, 129, -129, 128, -128, 1.5][:h]
    edge[5] = rng.normal(0, 1, h)
    edge[5, 1 % h] = np.inf
    edge[5, 2 % h] = -np.inf
    out.append(_bf16(edge))
    sub = (rng.integers(1, 0x80, (1, h)) | rng.integers(0, 2, (1, h)) << 15).astype(np.uint16)
    return np.concatenate(out + [sub])


def _same_amax(got: np.ndarray, want: np.ndarray) -> bool:
    """bf16 amax words against f32 amaxes: equal, or both NaN (which NaN is the row's largest payload's)."""
    g = _f32(got)
    return bool(((g.view(np.uint32) == want.view(np.uint32)) | (np.isnan(g) & np.isnan(want))).all() and not (got >> 15).any())


def _split(out: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A proof block's output leaves as (its codes as u8, its scale's word)."""
    return out[:, :n].astype(np.uint8), out[:, n].astype(np.uint32)


@pytest.mark.parametrize("relu", [True, False])
def test_the_row_unit_is_the_loops_quantize(relu) -> None:
    xb = rows(H, np.random.default_rng(3))
    codes, scale, amax = reference(xb, relu)
    got, s = _split(E.evaluate_batch(bind(Q.QuantRowS8, N=H, RELU=relu), xb), H)
    assert (got == codes).all(), np.argwhere(got != codes)[:5]
    assert (s == scale).all()
    assert _same_amax(E.evaluate_batch(bind(Q.RowAbsMax, N=H, RELU=relu), xb).astype(np.uint16), amax)


def test_the_edge_rows_reach_their_edges() -> None:
    """The edge rows reach what they are for: a zero and a NaN amax's scale 1.0, an infinite scale, ties to even, both
    saturations and a NaN's code 0."""
    codes, scale, _ = reference(rows(H, np.random.default_rng(3)), False)
    zero, neg0, _neg, ties, nan, inf, _sub = range(12, 19)
    assert scale[zero] == scale[neg0] == scale[nan] == 0x3F800000 and scale[inf] == 0x7F800000
    assert codes[ties, :8].view(np.int8).tolist() == [127, 0, 2, 2, 0, -2, -2, 126]
    assert codes[nan, :8].view(np.int8).tolist() == [127, 0, -128, 127, -128, 127, -128, 2]
    assert not codes[inf].any()


@pytest.mark.parametrize("relu", [True, False])
def test_a_row_in_chunks_is_the_row_unit(relu) -> None:
    """A row wider than one proof block: its amax (whole; as the amax of four parts' amaxes without the ReLU; and as the last part's
    values with the other three parts' amaxes, under the row's RELU), then four chunk proof blocks under it, whose codes are the
    row's and whose scales are each the row's."""
    xb = rows(H, np.random.default_rng(4))
    codes, scale, amax = reference(xb, relu)
    c = H // 4
    part = bind(Q.RowAbsMax, N=c, RELU=relu)
    parts = np.stack([E.evaluate_batch(part, xb[:, k * c:(k + 1) * c]) for k in range(4)], axis=1).astype(np.uint16)
    tree = E.evaluate_batch(bind(Q.RowAbsMax, N=4, RELU=False), parts).astype(np.uint16)
    last = E.evaluate_batch(bind(Q.RowAbsMax, N=c + 3, RELU=relu), np.concatenate([xb[:, 3 * c:], parts[:, :3]], axis=1))
    whole = E.evaluate_batch(bind(Q.RowAbsMax, N=H, RELU=relu), xb).astype(np.uint16)
    assert _same_amax(tree, amax) and _same_amax(last.astype(np.uint16), amax) and _same_amax(whole, amax)
    chunk = bind(Q.QuantChunkS8, N=c, RELU=relu)
    for k in range(4):
        got, s = _split(E.evaluate_batch(chunk, xb[:, k * c:(k + 1) * c], whole), c)
        assert (got == codes[:, k * c:(k + 1) * c]).all() and (s == scale).all(), k


def test_the_scale_is_torchs_where_on_every_kind_of_amax_word() -> None:
    """Every sign and exponent of amax with the mantissa's ends, its middle and two random ones (zeros, negatives, subnormals,
    ±inf and NaNs included): RN(amax / 127) when amax > 0, else 1.0."""
    rng = np.random.default_rng(9)
    se = np.arange(512, dtype=np.uint16) << 7
    a = (se[:, None] | np.array([0, 1, 0x40, 0x7F], np.uint16)[None, :]).ravel()
    a = np.concatenate([a, se | rng.integers(0, 0x80, 512).astype(np.uint16), se | rng.integers(0, 0x80, 512).astype(np.uint16)])
    with np.errstate(all="ignore"):
        f = _f32(a)
        want = np.where(f > 0, f / np.float32(127), np.float32(1)).astype(np.float32).view(np.uint32)
    _, s = _split(E.evaluate_batch(bind(Q.QuantChunkS8, N=1, RELU=False), np.zeros((len(a), 1), np.uint16), a), 1)
    assert (s == want).all()


@pytest.mark.parametrize("h", [8, 3])
@pytest.mark.parametrize("relu", [True, False])
def test_the_boolean_units_compute_their_word_definitions(h, relu) -> None:
    """At small widths, since a proof block's circuit is the same circuit at every element (H = 128 traces for ~40 s a proof
    block)."""
    rng = np.random.default_rng(7 + h)
    xb = np.concatenate([rows(h, rng), rng.integers(0, 1 << 16, (64, h)).astype(np.uint16)])
    amax = E.evaluate_batch(bind(Q.RowAbsMax, N=h, RELU=relu), xb).astype(np.uint16)
    amax[-64:] = rng.integers(0, 1 << 16, 64)
    for d, a in ((BQ.QuantRowS8, (xb,)), (BQ.RowAbsMax, (xb,)), (BQ.QuantChunkS8, (xb, amax))):
        target = bind(d, N=h, RELU=relu)
        word = TR.word_of(target)
        assert word.id == target.id.replace("_v2", "_v1")
        got = BS.evaluate_words(target, *a).astype(np.uint64)
        ref = E.evaluate_batch(word, *a).astype(np.uint64)
        assert (got == ref).all(), (target.id, np.argwhere(got != ref)[:5])


_BAD = {"N": (0, "4"), "RELU": (2, "no")}


@pytest.mark.parametrize("d,k,v", [(d, k, v) for d in (Q.QuantRowS8, Q.RowAbsMax, Q.QuantChunkS8, BQ.QuantRowS8, BQ.RowAbsMax,
                                                        BQ.QuantChunkS8) for k in d.statics for v in _BAD[k]],
                         ids=lambda a: getattr(a, "id", repr(a)))
def test_bind_refuses_statics_outside_the_family(d, k, v) -> None:
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(d, **{**{"N": 4, "RELU": False}, k: v})
