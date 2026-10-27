"""The training kernels' numpy models (`numerics.torch_kernels`), and the Definitions that trace in a second or two at the step's
sizes, on words an sm_120 GPU computed in a 2-layer smollm2-135m training step: a few rows of each kernel, every word equal.
The other tests hold each Definition to its model on random and special words; these hold the model to the GPU. The fixture
(`fixtures/smollm2-step-sm120-2026-10-27/step.npz`) is cut from run r20261027-031630-ce8e's dump (B 1, S 256, torch
2.14.0+cu130 on an RTX PRO 6000, cc 12.0), whose every row run r20261027-035245-3d66 checks the same way. A loss row's 49,152
log-probabilities and gradient are kept as the sha256 of the GPU's words.

The training run's own shapes (B 8, S 512) take kernels the 2-layer step doesn't: the embedding's sort path (M = 4096 ids),
the mean loss's 256-thread tree, and rows of 512 and 1,024 keys. `fixtures/smollm2-train-shapes-sm120-2026-10-27/shapes.npz`
is cut from run r20261027-040221-0e4d (the same GPU and torch), whose every word run r20261027-040917-a3ba checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.definitions.numerics import torch_kernels as TK
from verity.ml.definitions.registry import torch_embedding as TE, torch_loss as TL

FIXTURES = Path(__file__).parent / "fixtures"
V, D = 49152, 64


def _load(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def step():
    return _load(FIXTURES / "smollm2-step-sm120-2026-10-27" / "step.npz")


@pytest.fixture(scope="module")
def shapes():
    return _load(FIXTURES / "smollm2-train-shapes-sm120-2026-10-27" / "shapes.npz")


def _same(got, want) -> None:
    got, want = np.asarray(got, np.uint64), np.asarray(want, np.uint64)
    assert got.shape == want.shape and (got == want).all(), np.argwhere(got != want)[:5]


def _sha(a, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).astype(dtype).tobytes()).hexdigest()


def test_the_eager_rows_are_the_gpus(step) -> None:
    """Four causal rows of N = 256 keys (a query that sees 1, 38, 101 and 256): the saved f32 softmax, its bf16 copy and the
    score's gradient."""
    y, p = TK.eager_softmax(step["attn_s"], step["attn_n"], D)
    _same(y, step["attn_y"])
    _same(p, step["attn_p"])
    _same(TK.eager_softmax_backward(step["attn_d_p"], step["attn_y"], D), step["attn_d_s"])


def test_rope_and_its_backward_are_the_gpus(step) -> None:
    _same(TK.rope_hf(step["rope_x"], step["rope_cs"]), step["rope_out"])
    _same(TK.rope_hf_backward(step["rope_g"], step["rope_cs"]), step["rope_dx"])


def test_repeat_kv_backward_is_the_gpus(step) -> None:
    _same(TK.repeat_kv_backward(step["rkv_g"]), step["rkv_out"])


def test_the_cross_entropy_is_the_gpus(step) -> None:
    """One row's log-probabilities and logit gradient, and the mean loss over the 256 rows' target log-probabilities (the
    last row's label is HF's -100 pad)."""
    x, t, r = step["loss_x"][None], step["loss_t"], int(step["loss_row"])
    out, mx, ls, _, _ = TK.log_softmax_forward(x)
    assert _sha(out[0], "<u4") == str(step["loss_lsm_sha256"])
    loss, tw = TK.nll_loss(step["loss_lp"], t, V, True)
    assert loss == int(step["loss_loss"])
    assert tw == int(np.float32((t < V).sum()).view(np.uint32))
    g = TK.nll_loss_backward(0x3F800000, tw, True)
    _same(evaluate_words(bind(TL.NllLossTorchBackward, MEAN=1), np.array([0x3F800000], np.uint32),
                         np.array([tw], np.uint32)).reshape(-1), [g])
    dx = TK.log_softmax_nll_backward(x, t[r:r + 1], mx, ls, g, V)
    assert dx[0, t[r]] == step["loss_dx_at_target"]
    assert _sha(dx[0], "<u2") == str(step["loss_dx_sha256"])


@pytest.mark.slow
def test_the_cross_entropy_row_definitions_are_the_gpus(step) -> None:
    """`LogSoftmaxRow_v1{V=49152}` and `LogSoftmaxNllBackwardRow_v1{V=49152}` on the kept row: its target log-probability and
    every word of its logit gradient. Traces 48 expf's a lane, a minute or two."""
    x, t, r = step["loss_x"][None], step["loss_t"], int(step["loss_row"])
    _, mx, ls, _, _ = TK.log_softmax_forward(x)
    _same(evaluate_words(bind(TL.LogSoftmaxRow, V=V), x, t[r:r + 1]), [[mx[0], ls[0], step["loss_lp"][r]]])
    _, tw = TK.nll_loss(step["loss_lp"], t, V, True)
    g = np.array([TK.nll_loss_backward(0x3F800000, tw, True)], np.uint32)
    dx = evaluate_words(bind(TL.LogSoftmaxNllBackwardRow, V=V), x, t[r:r + 1], mx, ls, g)
    assert _sha(np.asarray(dx).reshape(-1), "<u2") == str(step["loss_dx_sha256"])


def test_the_embedding_backward_is_the_gpus(step) -> None:
    """embedding_dense_backward's feature kernel (M = 256 ids) on 13 of the 576 features of every entry the step's tokens
    hold, the model and `EmbeddingBackwardFeature_v1{M=256}`."""
    t, g, rows = step["emb_t"], step["emb_g"], step["emb_rows"]
    vocab = np.unique(t)
    _same(TK.embedding_dense_backward(t, g, vocab), rows)
    n, F = len(vocab), g.shape[1]
    got = evaluate_words(bind(TE.EmbeddingBackwardFeature, M=len(t)), np.repeat(vocab, F), np.broadcast_to(t, (n * F, len(t))),
                         np.tile(g.T, (n, 1)))
    _same(np.asarray(got).reshape(n, F), rows)


def test_the_rows_of_512_and_1024_keys_are_the_gpus(shapes) -> None:
    """Four causal rows each at N = 512 and N = 1,024 keys, the backward's last persistent size: the saved f32 softmax, its
    bf16 copy and the score's gradient."""
    for N in (512, 1024):
        y, p = TK.eager_softmax(shapes[f"n{N}_s"], shapes[f"n{N}_n"], D)
        _same(y, shapes[f"n{N}_y"])
        _same(p, shapes[f"n{N}_p"])
        _same(TK.eager_softmax_backward(shapes[f"n{N}_dp"], shapes[f"n{N}_y"], D), shapes[f"n{N}_d_s"])


def test_the_mean_loss_over_4096_rows_is_the_gpus(shapes) -> None:
    """The mean over 4,096 rows' target log-probabilities, every 97th label HF's -100, in nll_loss's 256-thread tree."""
    t = shapes["loss_t"]
    loss, tw = TK.nll_loss(shapes["loss_lp"], t, V, True)
    assert loss == int(shapes["loss_loss"])
    assert tw == int(np.float32((t < V).sum()).view(np.uint32))


def test_the_sort_path_is_the_gpus(shapes) -> None:
    """embedding_dense_backward's sort path (M = 4096 ids) on features 0..7 of 37 entries: entry 5, which 37 tokens hold (four
    partials of 10), among them the gradients NaN, inf, -inf, -0 and subnormal; entries 11, 10 and 1 tokens hold; and 33
    others."""
    _same(TK.embedding_dense_backward(shapes["emb_t"], shapes["emb_g"], shapes["emb_vocab"]), shapes["emb_rows"])


@pytest.mark.slow
def test_the_sort_path_definition_is_the_gpus(shapes) -> None:
    """`EmbeddingBackwardTorch_v1{M=4096}` on entries 3, 5, 7 and 49,151 (1, 37, 10 and 11 tokens)."""
    t, g, vocab, rows = shapes["emb_t"], shapes["emb_g"], shapes["emb_vocab"], shapes["emb_rows"]
    pick = np.flatnonzero(np.isin(vocab, [3, 5, 7, V - 1]))
    vs, F = vocab[pick], g.shape[1]
    got = evaluate_words(bind(TE.EmbeddingBackwardTorch, M=len(t)), np.repeat(vs, F), np.broadcast_to(t, (len(vs) * F, len(t))),
                         np.tile(g.T, (len(vs), 1)))
    _same(np.asarray(got).reshape(len(vs), F), rows[pick])
