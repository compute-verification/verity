"""`verity.ml.circuits.ops.quant` and `ops.boolean.quant`: the s8 quantizer's units are the torch loop's ``quantize`` of the row,
relu'd first under RELU (here in numpy float32, the same IEEE operations; torch itself in `integrations/torch_loop`'s tests), on
random rows and on the edge rows (all zero, all -0, all negative, ties at .5, NaN with saturating neighbours, ±inf, subnormal);
a row's amax is the max of its parts' amaxes; the scale reads any amax word as ``torch.where(amax > 0, amax / 127, 1)``; each
Boolean unit computes its word Definition, at H = 128 and at a small odd H; and `bind` refuses statics outside the family."""
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


@pytest.mark.parametrize("relu", [True, False])
def test_the_word_units_are_the_loops_quantize(relu) -> None:
    xb = rows(H, np.random.default_rng(3))
    codes, scale, amax = reference(xb, relu)
    got = E.evaluate_batch(bind(Q.RowAbsMax, N=H, RELU=relu), xb).astype(np.uint16)
    assert _same_amax(got, amax)
    for j in (0, 1, 31, 63):
        pair = E.evaluate_batch(bind(Q.QuantS8Pair, RELU=relu), xb[:, 2 * j:2 * j + 2], got).astype(np.uint8)
        assert (pair == codes[:, 2 * j:2 * j + 2]).all(), (j, np.argwhere(pair != codes[:, 2 * j:2 * j + 2])[:5])
    for half in (0, 1):
        s = E.evaluate_batch(bind(Q.QuantS8Scale, HALF=half), got).astype(np.uint32)
        assert (s == (scale >> np.uint32(16 * half)) & np.uint32(0xFFFF)).all(), half


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
def test_a_rows_amax_is_the_max_of_its_parts_amaxes(relu) -> None:
    """The tree a row wider than one unit's reads takes: four parts' amaxes, then the amax of those without the ReLU."""
    xb = rows(H, np.random.default_rng(4))
    part = bind(Q.RowAbsMax, N=H // 4, RELU=relu)
    parts = np.stack([E.evaluate_batch(part, xb[:, k * H // 4:(k + 1) * H // 4]) for k in range(4)], axis=1).astype(np.uint16)
    whole = E.evaluate_batch(bind(Q.RowAbsMax, N=H, RELU=relu), xb).astype(np.uint16)
    assert _same_amax(E.evaluate_batch(bind(Q.RowAbsMax, N=4, RELU=False), parts).astype(np.uint16), _f32(whole))


def test_the_scale_is_torchs_where_on_every_amax_word() -> None:
    """Every bf16 word as amax, negative ones and NaNs included: RN(amax / 127) when amax > 0, else 1.0."""
    a = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    with np.errstate(all="ignore"):
        f = _f32(a)
        want = np.where(f > 0, f / np.float32(127), np.float32(1)).astype(np.float32).view(np.uint32)
    lo, hi = (E.evaluate_batch(bind(Q.QuantS8Scale, HALF=k), a).astype(np.uint32) for k in (0, 1))
    assert ((hi << np.uint32(16)) | lo == want).all()


def _targets(h: int, relu: bool) -> list:
    return [bind(BQ.RowAbsMax, N=h, RELU=relu), bind(BQ.QuantS8Pair, RELU=relu)] + [bind(BQ.QuantS8Scale, HALF=k) for k in (0, 1)]


@pytest.mark.parametrize("h", [H, 3])
@pytest.mark.parametrize("relu", [True, False])
def test_the_boolean_units_compute_their_word_definitions(h, relu) -> None:
    rng = np.random.default_rng(7 + h)
    xb = np.concatenate([rows(h, rng), rng.integers(0, 1 << 16, (64, h)).astype(np.uint16)])
    amax = E.evaluate_batch(bind(Q.RowAbsMax, N=h, RELU=relu), xb).astype(np.uint16)
    amax[-64:] = rng.integers(0, 1 << 16, 64)
    args = {"RowAbsMax": (xb,), "QuantS8Pair": (xb[:, :2], amax), "QuantS8Scale": (amax,)}
    for target in _targets(h, relu):
        word = TR.word_of(target)
        assert word.id == target.id.replace("_v2", "_v1")
        a = args[target.id.split("_")[0]]
        got = BS.evaluate_words(target, *a).astype(np.uint64)
        ref = E.evaluate_batch(word, *a).astype(np.uint64)
        assert (got == ref).all(), (target.id, np.argwhere(got != ref)[:5])


_BAD = {"N": (0, "4"), "RELU": (2, "no"), "HALF": (2, -1)}


@pytest.mark.parametrize("d,k,v", [(d, k, v) for d in (Q.RowAbsMax, Q.QuantS8Pair, Q.QuantS8Scale, BQ.RowAbsMax, BQ.QuantS8Pair,
                                                        BQ.QuantS8Scale) for k in d.statics for v in _BAD[k]],
                         ids=lambda a: getattr(a, "id", repr(a)))
def test_bind_refuses_statics_outside_the_family(d, k, v) -> None:
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(d, **{**{s: {"N": 4, "RELU": False, "HALF": 0}[s] for s in d.statics}, k: v})
