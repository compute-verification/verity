"""`registry.torch_attention`: each Boolean Definition computes, word for word, the numpy model of its torch CUDA kernel
(`numerics.torch_kernels`) on random rows and on rows of special words (±0, subnormals, ±inf, NaNs, the extremes). The model
then matches torch's CPU autograd: bit for bit for RoPE and repeat_kv's backward, whose CPU kernels round as the CUDA ones
do, and to f32 rounding for the softmax, whose CPU kernels sum in another order and with another expf. The CUDA kernels
themselves are `test_torch_training_gpu.py`'s."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.circuits.types import Array
from verity.ml.definitions.numerics import torch_blocks as TB
from verity.ml.definitions.numerics import torch_kernels as TK
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity.ml.definitions.registry import torch_attention as TA

F32_SPECIALS = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
                         0x7FC00000, 0xFFC00001, 0x7F800001, 0x3F800000, 0xBF800000, 0x42B00000, 0xC2D00000], np.uint32)
BF16_SPECIALS = np.array([0, 0x8000, 1, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81,
                          0x3F80, 0xBF80], np.uint16)


def _f32(a) -> np.ndarray:
    return np.asarray(a, np.float32).view(np.uint32)


def _bf16(a) -> np.ndarray:
    return TK.bf16(_f32(a))


def f32_rows(rng, n: int, scale: float = 3.0) -> np.ndarray:
    """Normal rows, rows of random words, and rows with special words dropped into normal ones."""
    normal = _f32(rng.normal(0, scale, (24, n)))
    words = rng.integers(0, 1 << 32, (8, n), dtype=np.uint64).astype(np.uint32)
    special = _f32(rng.normal(0, scale, (len(F32_SPECIALS), n)))
    special[np.arange(len(F32_SPECIALS)), rng.integers(0, n, len(F32_SPECIALS))] = F32_SPECIALS
    picks = _f32(rng.normal(0, scale, (8, n)))
    at = rng.random((8, n)) < 0.3
    picks[at] = rng.choice(F32_SPECIALS, int(at.sum()))
    return np.concatenate([normal, words, special, picks, np.full((1, n), F32_SPECIALS[0]), np.full((1, n), 0xFF800000)])


def bf16_rows(rng, n: int, rows: int | None = None, scale: float = 2.0) -> np.ndarray:
    normal = _bf16(rng.normal(0, scale, (24, n)))
    words = rng.integers(0, 1 << 16, (8, n)).astype(np.uint16)
    special = _bf16(rng.normal(0, scale, (len(BF16_SPECIALS), n)))
    special[np.arange(len(BF16_SPECIALS)), rng.integers(0, n, len(BF16_SPECIALS))] = BF16_SPECIALS
    out = np.concatenate([normal, words, special])
    return out if rows is None else out[rng.integers(0, len(out), rows)]


def _same(got, want) -> None:
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape and (got == want).all(), np.argwhere(got != want)[:5]


@pytest.mark.slow
@pytest.mark.parametrize("N", [5, 12])
def test_the_forward_softmax_is_the_persistent_kernel(N) -> None:
    x = f32_rows(np.random.default_rng(N), N)
    _same(evaluate_words(bind(TA.SoftmaxTorch, N=N), x), TK.softmax_forward(x))


@pytest.mark.slow
def test_the_forward_softmax_folds_lanes_of_two() -> None:
    """At N = 33 each of the 32 lanes holds two elements, the second past N padded with -inf, so a lane's max and its sum
    fold. This traces about a minute; the backward's test covers the fold's order without expf."""
    x = f32_rows(np.random.default_rng(33), 33)
    _same(evaluate_words(bind(TA.SoftmaxTorch, N=33), x), TK.softmax_forward(x))


@pytest.mark.parametrize("N", [1, pytest.param(6, marks=pytest.mark.slow), pytest.param(40, marks=pytest.mark.slow)])
def test_the_softmax_backward_is_the_persistent_kernel(N) -> None:
    rng = np.random.default_rng(100 + N)
    g, y = f32_rows(rng, N), f32_rows(rng, N, 0.1)
    _same(evaluate_words(bind(TA.SoftmaxTorchBackward, N=N), g, y), TK.softmax_backward(g, y))


@pytest.mark.slow
def test_the_eager_row_masks_from_n_and_writes_both_copies() -> None:
    """Every key count from 0 to past N, so n = 0 masks the whole row and n > N none."""
    N, D = 6, 64
    rng = np.random.default_rng(6)
    s = bf16_rows(rng, N, 72, scale=8.0)
    n = rng.integers(0, N + 2, len(s))
    y, p = TK.eager_softmax(s, n, D)
    got = evaluate_words(bind(TA.EagerSoftmax, N=N, D=D), s, n.astype(np.uint16).reshape(-1, 1))
    _same(got[:, :N], y)
    _same(got[:, N:], p)


@pytest.mark.slow
def test_the_eager_backward_is_the_chain_of_its_kernels() -> None:
    N, D = 6, 64
    rng = np.random.default_rng(7)
    s = bf16_rows(rng, N, 72, scale=8.0)
    y, _ = TK.eager_softmax(s, rng.integers(1, N + 1, len(s)), D)
    y = np.concatenate([y, f32_rows(rng, N, 0.2)[:24]])
    dp = bf16_rows(rng, N, len(y))
    _same(evaluate_words(bind(TA.EagerSoftmaxBackward, N=N, D=D), dp, y), TK.eager_softmax_backward(dp, y, D))


@pytest.mark.parametrize("D", [2, 8])
def test_rope_and_its_backward_are_hfs_bf16_ops(D) -> None:
    rng = np.random.default_rng(D)
    x, cs = bf16_rows(rng, D, 96), bf16_rows(rng, D, 96)
    _same(evaluate_words(bind(TA.RopeHf, D=D), x, cs), TK.rope_hf(x, cs))
    _same(evaluate_words(bind(TA.RopeHfBackward, D=D), x, cs), TK.rope_hf_backward(x, cs))


@pytest.mark.parametrize("NREP", [2, 3, 4, 5, 9])
def test_repeat_kv_backward_is_the_reduce_kernels_fold(NREP) -> None:
    rng = np.random.default_rng(NREP)
    g = np.stack([bf16_rows(rng, 4) for _ in range(NREP)], axis=1)
    _same(evaluate_words(bind(TA.RepeatKvBackward, NREP=NREP, N=4), g), TK.repeat_kv_backward(g))


# --- the step's blocks ---------------------------------------------------------------------------------------------------------

def _probe(sig, body):
    return bind(CompositeDefinition("TransposeProbe", 1, (), lambda S: sig, body, register=False))


@pytest.mark.parametrize("M,F", [(5, 3), (130, 130)])
def test_the_transpose_is_its_tiles(M, F) -> None:
    """One tile, and a tile of each shape (128 and 2 rows, 128 and 2 columns)."""
    d = _probe(((("x", Array(M, Array(F, TA.BF16))),), Array(F, Array(M, TA.BF16))), lambda B, S, x: TA.transpose(B, x, M, F))
    x = np.random.default_rng(M).integers(0, 1 << 16, (2, M, F), dtype=np.uint64).astype(np.uint16)
    _same(evaluate_words(d, x), x.transpose(0, 2, 1))


def test_the_transposes_are_each_matrix_at_its_base() -> None:
    """Three 130 x 3 column blocks of a 130 x 9 matrix, each row 9 words past the last: tiles of 128 and 2 rows."""
    M, F, H = 130, 3, 3
    d = _probe(((("x", Array(M, Array(H * F, TA.BF16))),), Array(H, Array(F, Array(M, TA.BF16)))),
               lambda B, S, x: TA._stack(TA.transposes(B, x, M, F, [h * F for h in range(H)], H * F), range(H)))
    x = np.random.default_rng(9).integers(0, 1 << 16, (2, M, H * F), dtype=np.uint64).astype(np.uint16)
    _same(evaluate_words(d, x).reshape(2, H, F, M), x.reshape(2, M, H, F).transpose(0, 2, 3, 1))


def rope_table(N: int, D: int) -> np.ndarray:
    """HF's bf16 [cos | sin] rows at rope_theta 1e5, cos and sin of each half computed in f32."""
    inv = (1.0 / 1e5 ** (np.arange(0, D, 2, dtype=np.float32) / D)).astype(np.float32)
    a = np.arange(N, dtype=np.float32)[:, None] * inv[None, :]
    return np.concatenate([_bf16(np.cos(a)), _bf16(np.sin(a))], 1)


def _projections(rng, N: int, H: int, D: int) -> np.ndarray:
    """A projection's (N, H·D) bf16 output: normal words with a few specials."""
    x = _bf16(rng.normal(0, 1.5, (N, H * D)))
    at = rng.random(x.shape) < 0.01
    x[at] = rng.choice(BF16_SPECIALS, int(at.sum()))
    return x


def _split(out, *shapes):
    out, parts, a = np.asarray(out).reshape(-1), [], 0
    for s in shapes:
        n = int(np.prod(s))
        parts.append(out[a:a + n].reshape(s))
        a += n
    assert a == out.size
    return parts


@pytest.mark.slow
@pytest.mark.parametrize("NH,KVH", [(4, 2), (2, 2)])
def test_the_attention_blocks_are_their_definitions_models(NH, KVH) -> None:
    """One sequence of N = 16 tokens, D = 16, with two query heads a KV head and with one, forward then backward on the saved
    values: every word of every output, against `torch_blocks.MODELS`, the models the step's walk calls. It catches what no
    test of a single Definition can: the blocks' wiring of the heads, the bmm's rows and operands, the transposes, and the
    softmax rows (about 1 minute and 2 GB a case)."""
    N, D = 16, 16
    rng = np.random.default_rng(NH)
    q, k, v = _projections(rng, N, NH, D), _projections(rng, N, KVH, D), _projections(rng, N, KVH, D)
    cs, n = rope_table(N, D), np.arange(1, N + 1).astype(np.uint16)
    st = dict(N=N, NH=NH, KVH=KVH, D=D)
    args = tuple(a[None] for a in (q, k, v, cs, n))
    want = [w[0] for w in TB.MODELS["EagerAttentionTorch_v1"](st, *args)]
    got = evaluate_words(bind(TA.EagerAttentionTorch, **st), *args)
    for g, w in zip(_split(got, *(w.shape for w in want)), want, strict=True):
        _same(g, w)
    _, qe, ke, y, p = want
    args = tuple(a[None] for a in (_projections(rng, N, NH, D), qe, ke, v, y, p, cs))
    want = [w[0] for w in TB.MODELS["EagerAttentionTorchBackward_v1"](st, *args)]
    got = evaluate_words(bind(TA.EagerAttentionTorchBackward, **st), *args)
    for g, w in zip(_split(got, *(w.shape for w in want)), want, strict=True):
        _same(g, w)


@pytest.mark.parametrize("S,D", CB.TRAIN_BMM_SHAPES)
def test_the_blocks_bmms_are_the_captured_bindings(S, D) -> None:
    """Wherever `cublas_bf16.attention_bmm` binds, each of the blocks' six bmm's is that binding itself."""
    for op in CB.TRAIN_BMM:
        assert TA._bmm(op, 9, S, D) is CB.attention_bmm(op, S, D, B=9)


# --- the models against torch's CPU autograd ---------------------------------------------------------------------------------

def _torch():
    return pytest.importorskip("torch")


def _words(t) -> np.ndarray:
    return t.detach().contiguous().view(_torch().int16).numpy().view(np.uint16)


def _tbf16(a):
    torch = _torch()
    return torch.tensor(np.asarray(a, np.float32)).to(torch.bfloat16)


def test_rope_matches_torch_word_for_word() -> None:
    """HF's `apply_rotary_pos_emb` on q at smollm2-135m's D = 64 and rope_theta = 1e5, forward and backward."""
    torch = _torch()
    rng = np.random.default_rng(11)
    D, T = 64, 128
    inv = 1.0 / (1e5 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    emb = torch.arange(T, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([emb, emb], -1)
    cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
    x = _tbf16(rng.normal(0, 2, (T, D))).requires_grad_(True)
    out = x * cos + torch.cat((-x[..., D // 2:], x[..., :D // 2]), -1) * sin
    g = _tbf16(rng.normal(0, 1, (T, D)))
    out.backward(g)
    cs = np.concatenate([_words(cos)[:, :D // 2], _words(sin)[:, :D // 2]], 1)
    _same(TK.rope_hf(_words(x), cs), _words(out))
    _same(TK.rope_hf_backward(_words(g), cs), _words(x.grad))


def test_repeat_kv_backward_matches_torch_word_for_word() -> None:
    """HF's `repeat_kv` at smollm2-135m's three KV heads of three query heads each."""
    _torch()
    rng = np.random.default_rng(12)
    B, KVH, NREP, T, D = 2, 3, 3, 16, 8
    k = _tbf16(rng.normal(0, 1, (B, KVH, T, D))).requires_grad_(True)
    g = _tbf16(rng.normal(0, 1, (B, KVH * NREP, T, D)))
    k[:, :, None].expand(B, KVH, NREP, T, D).reshape(B, KVH * NREP, T, D).backward(g)
    gw = _words(g).reshape(B, KVH, NREP, T, D).transpose(0, 1, 3, 4, 2).reshape(-1, NREP)
    _same(TK.repeat_kv_backward(gw[:, :, None]).reshape(B, KVH, T, D), _words(k.grad))


def test_the_eager_softmax_matches_torch_to_rounding() -> None:
    """HF's causal eager row at S = 64, D = 64: the f32 softmax to a few f32 ulps, its bf16 copy and the score's gradient to
    one bf16 ulp of the row's largest gradient (torch's CPU backward is ``y * (g - sum)``, the CUDA kernel ``g*y - y*sum``)."""
    torch = _torch()
    rng = np.random.default_rng(13)
    N, D = 64, 64
    s = _tbf16(rng.normal(0, 8, (N, N))).requires_grad_(True)
    keys = torch.arange(N)[None, :] <= torch.arange(N)[:, None]
    mask = torch.where(keys, torch.tensor(0.0, dtype=torch.bfloat16), torch.tensor(torch.finfo(torch.bfloat16).min,
                                                                                  dtype=torch.bfloat16))
    y_t = torch.nn.functional.softmax(s * D ** -0.5 + mask, dim=-1, dtype=torch.float32)
    dp = _tbf16(rng.normal(0, 1, (N, N)))
    y_t.to(torch.bfloat16).backward(dp)
    y, p = TK.eager_softmax(_words(s), np.arange(1, N + 1), D)
    want = y_t.detach().numpy()
    assert np.allclose(y.view(np.float32), want, rtol=4e-7, atol=0)
    assert (np.abs(p.astype(np.int32) - _words(y_t.to(torch.bfloat16)).astype(np.int32)) <= 1).all()
    got = (TK.eager_softmax_backward(_words(dp), want.view(np.uint32), D).astype(np.uint32) << 16).view(np.float32)
    ref = s.grad.float().numpy()
    tol = np.abs(ref).max(axis=1, keepdims=True) * 2.0 ** -7
    assert (np.abs(got - ref) <= tol).all()


@pytest.mark.parametrize("d,statics", [
    (TA.SoftmaxTorch, {"N": 0}), (TA.SoftmaxTorch, {"N": 2049}), (TA.SoftmaxTorchBackward, {"N": 1025}),
    (TA.EagerSoftmax, {"N": 4, "D": 7}), (TA.EagerSoftmaxBackward, {"N": 1025, "D": 64}), (TA.RopeHf, {"D": 3}),
    (TA.RopeHfBackward, {"D": 0}), (TA.RepeatKvBackward, {"NREP": 1, "N": 4}), (TA.RepeatKvBackward, {"NREP": 17, "N": 4}),
    (TA.RepeatKvBackward, {"NREP": 3, "N": True}),
    (TA.EagerAttentionTorch, {"N": 24, "NH": 4, "KVH": 2, "D": 16}), (TA.EagerAttentionTorch, {"N": 16, "NH": 4, "KVH": 3, "D": 16}),
    (TA.EagerAttentionTorch, {"N": 2064, "NH": 1, "KVH": 1, "D": 16}), (TA.EagerAttentionTorch, {"N": 16, "NH": 2, "KVH": 2, "D": 8}),
    (TA.EagerAttentionTorchBackward, {"N": 1040, "NH": 1, "KVH": 1, "D": 16}),
    (TA.EagerAttentionTorchBackward, {"N": 16, "NH": 34, "KVH": 2, "D": 16}),
], ids=lambda a: getattr(a, "name", repr(a)))
def test_bind_refuses_statics_outside_the_kernels_range(d, statics) -> None:
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(d, **statics)
