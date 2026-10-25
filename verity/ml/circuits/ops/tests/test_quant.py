"""`verity.ml.circuits.ops.quant` and `ops.boolean.quant`: the s8 quantizer's units are the torch loop's ``quantize`` of the row,
relu'd first under RELU (here in numpy float32, the same IEEE operations; torch itself in `integrations/torch_loop`'s tests), on
random rows and on the edge rows (all zero, all -0, all negative, ties at .5, NaN with saturating neighbours, ±inf, subnormal);
a pair unit's codes are its first two coordinates' under any order of the rest; each Boolean unit computes its word Definition,
at H = 128 and at a small odd H; and `bind` refuses statics outside the family."""
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


def reference(xb: np.ndarray, relu: bool) -> tuple[np.ndarray, np.ndarray]:
    """``quantize`` of bf16 rows (u16 words), relu'd first under ``relu``: (the s8 codes as u8, the f32 scales' words)."""
    x = (xb.astype(np.uint32) << 16).view(np.float32)
    if relu:
        x = np.where(x < 0, np.float32(0), x)
    with np.errstate(all="ignore"):
        amax = np.abs(x).max(axis=1)
        scale = np.where(amax > 0, amax / np.float32(127), np.float32(1)).astype(np.float32)
        q = np.rint(x / scale[:, None])
        codes = np.where(np.isnan(q), 0, np.clip(q, -128, 127)).astype(np.int8)
    return codes.view(np.uint8), scale.view(np.uint32)


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


@pytest.mark.parametrize("relu", [True, False])
def test_the_word_units_are_the_loops_quantize(relu) -> None:
    rng = np.random.default_rng(3)
    xb = rows(H, rng)
    codes, scale = reference(xb, relu)
    for j in (0, 1, 31, 63):
        got = E.evaluate_batch(bind(Q.QuantS8Pair, H=H, RELU=relu), np.roll(xb, -2 * j, axis=1)).astype(np.uint8)
        assert (got == codes[:, 2 * j:2 * j + 2]).all(), (j, np.argwhere(got != codes[:, 2 * j:2 * j + 2])[:5])
    for half in (0, 1):
        got = E.evaluate_batch(bind(Q.QuantS8Scale, H=H, HALF=half, RELU=relu), xb).astype(np.uint32)
        assert (got == (scale >> np.uint32(16 * half)) & np.uint32(0xFFFF)).all(), half


def test_the_edge_rows_reach_their_edges() -> None:
    """The edge rows reach what they are for: a zero and a NaN amax's scale 1.0, an infinite scale, ties to even, both
    saturations and a NaN's code 0."""
    codes, scale = reference(rows(H, np.random.default_rng(3)), False)
    zero, neg0, _neg, ties, nan, inf, _sub = range(12, 19)
    assert scale[zero] == scale[neg0] == scale[nan] == 0x3F800000 and scale[inf] == 0x7F800000
    assert codes[ties, :8].view(np.int8).tolist() == [127, 0, 2, 2, 0, -2, -2, 126]
    assert codes[nan, :8].view(np.int8).tolist() == [127, 0, -128, 127, -128, 127, -128, 2]
    assert not codes[inf].any()


@pytest.mark.parametrize("relu", [True, False])
def test_a_pair_unit_reads_the_rest_of_the_row_in_any_order(relu) -> None:
    rng = np.random.default_rng(5)
    xb = rows(H, rng)
    perm = np.concatenate([[0, 1], 2 + rng.permutation(H - 2)])
    d = bind(Q.QuantS8Pair, H=H, RELU=relu)
    assert (E.evaluate_batch(d, xb[:, perm]) == E.evaluate_batch(d, xb)).all()


@pytest.mark.parametrize("h", [H, 3])
@pytest.mark.parametrize("relu", [True, False])
def test_the_boolean_units_compute_their_word_definitions(h, relu) -> None:
    rng = np.random.default_rng(7 + h)
    xb = np.concatenate([rows(h, rng), rng.integers(0, 1 << 16, (64, h)).astype(np.uint16)])
    for target in [bind(BQ.QuantS8Pair, H=h, RELU=relu)] + [bind(BQ.QuantS8Scale, H=h, HALF=k, RELU=relu) for k in (0, 1)]:
        word = TR.word_of(target)
        assert word.id == target.id.replace("_v2{", "_v1{")
        got = BS.evaluate_words(target, xb).astype(np.uint64)
        ref = E.evaluate_batch(word, xb).astype(np.uint64)
        assert (got == ref).all(), (target.id, np.argwhere(got != ref)[:5])


@pytest.mark.parametrize("d", [Q.QuantS8Pair, Q.QuantS8Scale, BQ.QuantS8Pair, BQ.QuantS8Scale], ids=lambda d: d.id)
@pytest.mark.parametrize("bad", [{"H": 1}, {"H": True}, {"RELU": 1}, {"HALF": 2}])
def test_bind_refuses_statics_outside_the_family(d, bad) -> None:
    if "HALF" in bad and "HALF" not in d.statics:
        pytest.skip("no HALF")
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(d, **bad)
