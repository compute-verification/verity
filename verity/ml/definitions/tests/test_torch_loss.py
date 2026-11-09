"""`registry.torch_loss`: each Boolean Definition computes, word for word, the numpy model of its torch CUDA kernel
(`numerics.torch_kernels`; a row's, `numerics.torch_blocks`), warp by warp at V = 12,288 (three loads a thread, the least vocabulary the block kernel runs at), on
rows of normal logits and rows with special words (NaN, ±inf, all -0, the extremes). The model then matches torch's CPU
cross-entropy and its backward to f32 rounding: the CPU kernels sum in another order and with another expf. The CUDA kernels
themselves are `test_torch_training_gpu.py`'s. The target's logit is picked in numpy, since `GatherBf16x{V}_v2` is made on demand and has
its own tests."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.definitions.numerics import torch_blocks as TB
from verity.ml.definitions.numerics import torch_kernels as TK
from verity.ml.definitions.registry import torch_loss as TL

V = 12288
IGNORE = 0xFFFFFF9C
F32_SPECIALS = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
                         0xFFC00001, 0x3F800000, 0xBF800000], np.uint32)


def _f32(a) -> np.ndarray:
    return np.asarray(a, np.float32).view(np.uint32)


def logits(rng, rows: int = 2) -> np.ndarray:
    """Normal rows, then one with a NaN, one with +inf, one with -inf and the bf16 extremes, and one all -0."""
    x = TK.bf16(_f32(rng.normal(0, 3, (rows + 4, V))))
    x[rows, 77] = 0x7FC1
    x[rows + 1, 5000] = 0x7F80
    x[rows + 2, [1, 4096, 8191]] = [0xFF80, 0x7F7F, 0xFF7F]
    x[rows + 3] = 0x8000
    return x


def _same(got, want) -> None:
    got, want = np.asarray(got, np.uint64), np.asarray(want, np.uint64)
    assert got.shape == want.shape and (got == want).all(), np.argwhere(got != want)[:5]


def _ev(d, *args) -> np.ndarray:
    return np.asarray(evaluate_words(d, *args))


def test_the_warp_trees_are_lane_zeros_shuffles() -> None:
    rng = np.random.default_rng(1)
    v = np.concatenate([_f32(rng.normal(0, 3, (16, 32))), rng.choice(F32_SPECIALS, (16, 32)),
                        rng.integers(0, 1 << 32, (8, 32), dtype=np.uint64).astype(np.uint32)])
    _same(_ev(TL.WarpReduceMax, v).reshape(-1), TK.warp_reduce(v, TK.max_combine))
    _same(_ev(TL.WarpReduceSum, v).reshape(-1), TK.warp_reduce(v, TK.add))


def test_each_warps_max_is_its_threads_fmaxf_then_the_tree() -> None:
    x = logits(np.random.default_rng(2))
    wm = TK.log_softmax_maxes(x)
    for w in (0, 13, 31):
        _same(_ev(bind(TL.LogSoftmaxWarpMax, V=V), TK.chunk(x, w)).reshape(-1), wm[:, w])


@pytest.mark.slow
def test_each_warps_sum_is_its_threads_expf_sum_then_the_tree() -> None:
    """Traces 12 expf's a lane, about 20 s."""
    x = logits(np.random.default_rng(3))
    _, _, _, wm, ws = TK.log_softmax_forward(x)
    for w in (0, 31):
        _same(_ev(bind(TL.LogSoftmaxWarpSumExp, V=V), TK.chunk(x, w), wm).reshape(-1), ws[:, w])


def test_the_row_end_is_the_trees_logf_and_the_epilogue() -> None:
    rng = np.random.default_rng(4)
    m = np.concatenate([_f32(rng.normal(0, 3, (8, 32))), rng.choice(F32_SPECIALS, (4, 32))])
    s = np.concatenate([_f32(rng.uniform(0, 400, (8, 32))), rng.choice(F32_SPECIALS, (4, 32))])
    xt = np.concatenate([TK.bf16(_f32(rng.normal(0, 3, 8))), np.array([0x7FFF, 0x7F80, 0x8000, 0xFF7F], np.uint16)])
    mx, ls = TK.warp_reduce(m, TK.max_combine), TK.logf(TK.warp_reduce(s, TK.add))
    lp = TK.add(TK.add(TK.f32(xt), TK.neg(mx)), TK.neg(ls))
    _same(_ev(TL.LogSoftmaxRowEnd, m, s, xt), np.stack([mx, ls, lp], 1))


def _labels(rng, M: int) -> np.ndarray:
    t = rng.integers(0, V, M).astype(np.uint32)
    t[::7] = IGNORE
    t[1] = V
    return t


@pytest.mark.parametrize("M,MEAN", [(16, 0), (16, 1), pytest.param(40, 1, marks=pytest.mark.slow),
                                    pytest.param(736, 1, marks=pytest.mark.slow)])
def test_the_loss_is_the_reduce_kernels_threads_and_tree(M, MEAN) -> None:
    """M = 16 and 40 run 32 threads (one row each, then some two). M = 736 is the least M of 64 threads, the one case of a
    stride tree with a level above 32 and of T chosen by the rounding of log2(M // 16); it traces about a minute."""
    rng = np.random.default_rng(M)
    lp = TK.neg(_f32(rng.uniform(0, 12, (5, M))))
    lp[1, 3], lp[2] = 0x7FC00000, 0x80000000
    t = np.stack([_labels(rng, M) for _ in range(5)])
    t[4] = IGNORE
    want = np.array([TK.nll_loss(lp[r], t[r], V, bool(MEAN)) for r in range(5)], np.uint32)
    _same(_ev(bind(TL.NllLossTorch, M=M, V=V, MEAN=MEAN), lp, t), want)


@pytest.mark.parametrize("MEAN", [0, 1])
def test_the_loss_gradient_is_minus_go_over_the_weight(MEAN) -> None:
    go = np.array([0x3F800000, 0x3A000000, 0, 0x80000000, 0x7FC00000, 0x7F800000], np.uint32)
    tw = np.array([0x457F8000, 0x3F800000, 0, 0x45800000, 0x3F800000, 0x7F800000], np.uint32)
    got = _ev(bind(TL.NllLossTorchBackward, MEAN=MEAN), *((go, tw) if MEAN else (go,))).reshape(-1)
    _same(got, [TK.nll_loss_backward(a, b, bool(MEAN)) for a, b in zip(go, tw)])


def test_the_slots_place_the_label_in_its_warps_chunk() -> None:
    t = np.array([0, 127, 128, 4095, 4096, 4096 + 128 * 5 + 3, V - 1, V, IGNORE, 0xFFFFFFFF], np.uint32)
    g = np.full(len(t), 0xB9800000, np.uint32)
    g[1:4] = [0x80000000, 0x7FC00001, 0]
    got = _ev(bind(TL.LogSoftmaxNllSlots, V=V), t, g)
    pos = np.full((len(t), 32), TL.NO_SLOT, np.uint32)
    for i, x in enumerate(t):
        if x < V:
            w = (int(x) >> 7) & 31
            pos[i, w] = TL.chunk_columns(V, w).index(int(x))
    _same(got[:, :32], pos)
    _same(got[:, 32], np.where(t < V, TK.add(np.zeros(len(t), np.uint32), g), 0))


def test_each_warps_backward_is_the_epilogue_on_its_chunk() -> None:
    rng = np.random.default_rng(5)
    x = logits(rng)
    mx = TK.warp_reduce(TK.log_softmax_maxes(x), TK.max_combine)
    ls = np.full(len(x), 0x41140000, np.uint32)
    ls[-1] = 0x7FFFFFFF
    t = np.array([5, 4096 + 128 * 7 + 2, IGNORE, 77, 1, 4096 * 2 + 128 * 7], np.uint32)
    g = np.array([0xB9800000, 0xB9800000, 0xB9800000, 0x80000000, 0x7FC00000, 0xBF800000], np.uint32)
    want = TK.log_softmax_nll_backward(x, t, mx, ls, g, V)
    sl = _ev(bind(TL.LogSoftmaxNllSlots, V=V), t, g)
    for w in (0, 7):
        got = _ev(bind(TL.LogSoftmaxNllBackwardWarp, V=V), TK.chunk(x, w), sl[:, w], mx, ls, sl[:, 32], g)
        _same(got, TK.chunk(want, w))


# --- the step's blocks ---------------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_a_row_is_its_warps_then_the_row_end() -> None:
    """Six rows (NaN, ±inf, the extremes, all -0), labels in all three chunks' warps, ignored and V. About 40 s."""
    x = logits(np.random.default_rng(7))
    t = np.array([4096 * 2 + 128 * 31 + 127, 0, IGNORE, 4096 + 5, V, 77], np.uint32)
    _same(_ev(bind(TL.LogSoftmaxRow, V=V), x, t), np.stack(TB.MODELS["LogSoftmaxRow_v1"]({"V": V}, x, t), 1))


def test_a_rows_backward_is_its_slots_then_its_warps() -> None:
    rng = np.random.default_rng(8)
    x = logits(rng)
    t = np.array([5, 4096 + 128 * 7 + 2, IGNORE, 77, V - 1, 4096 * 2 + 128 * 7], np.uint32)
    mx = TK.warp_reduce(TK.log_softmax_maxes(x), TK.max_combine)
    ls = np.full(len(x), 0x41140000, np.uint32)
    ls[-1] = 0x7FFFFFFF
    g = np.array([0xB9800000, 0xB9800000, 0xB9800000, 0x80000000, 0x7FC00000, 0xBF800000], np.uint32)
    want = TB.MODELS["LogSoftmaxNllBackwardRow_v1"]({"V": V}, x, t, mx, ls, g)
    _same(_ev(bind(TL.LogSoftmaxNllBackwardRow, V=V), x, t, mx, ls, g), want)


# --- the model against torch's CPU cross-entropy -----------------------------------------------------------------------------

def _torch():
    return pytest.importorskip("torch")


@pytest.mark.slow
def test_the_model_matches_torch_cross_entropy_to_rounding() -> None:
    """16 rows, a label ignored in every seventh: the log-probabilities to 2e-5, the mean loss to a few f32 ulps, and the
    logits' bf16 gradient to one bf16 ulp. About 15 s, most of it the model's expf."""
    torch = _torch()
    rng = np.random.default_rng(6)
    M = 16
    x = TK.bf16(_f32(rng.normal(0, 3, (M, V))))
    t = _labels(rng, M)
    t[1] = 9
    xt = torch.tensor(x.view(np.int16)).view(torch.bfloat16).requires_grad_(True)
    lab = torch.tensor(np.where(t == IGNORE, -100, t.astype(np.int64)))
    loss = torch.nn.functional.cross_entropy(xt.float(), lab, ignore_index=-100)
    loss.backward()
    out, mx, ls, _, _ = TK.log_softmax_forward(x)
    ref = torch.log_softmax(xt.detach().float(), -1).numpy()
    assert np.allclose(out.view(np.float32), ref, rtol=0, atol=2e-5)
    lp = out[np.arange(M), np.minimum(t, V - 1)]
    got, tw = TK.nll_loss(lp, t, V, True)
    assert tw == _f32(np.float32((t < V).sum()))
    assert abs(np.uint32(got).view(np.float32) - loss.item()) <= 4e-7 * abs(loss.item())
    g = TK.nll_loss_backward(0x3F800000, tw, True)
    dx = TK.log_softmax_nll_backward(x, t, mx, ls, g, V)
    dref = xt.grad.contiguous().view(torch.int16).numpy().view(np.uint16)
    assert (np.abs(dx.astype(np.int32) - dref.astype(np.int32)) <= 1).all()


@pytest.mark.parametrize("d,statics", [
    (TL.LogSoftmaxWarpMax, {"V": 8192}), (TL.LogSoftmaxWarpMax, {"V": 12289}), (TL.LogSoftmaxWarpSumExp, {"V": 4096}),
    (TL.LogSoftmaxLaneMax, {"KL": 2}), (TL.LogSoftmaxLaneSumExp, {"KL": 0}), (TL.NllLossTorch, {"M": 15, "V": V, "MEAN": 1}),
    (TL.NllLossTorch, {"M": 16, "V": V, "MEAN": 2}), (TL.NllLossTorchBackward, {"MEAN": -1}),
    (TL.LogSoftmaxNllSlots, {"V": 100}), (TL.LogSoftmaxNllBackwardWarp, {"V": 12288 * 2 + 1}),
    (TL.LogSoftmaxRow, {"V": 4096}), (TL.LogSoftmaxNllBackwardRow, {"V": 12289}),
], ids=lambda a: getattr(a, "name", repr(a)))
def test_bind_refuses_statics_outside_the_kernels_range(d, statics) -> None:
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(d, **statics)
