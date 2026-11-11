"""`registry.torch_embedding_f32`: the f32 table's gradient on words computes, word for word, the numpy model of torch's CUDA
embedding backward on f32 words (`numerics.torch_kernels`, written from the kernels as they run: chunks led by their first
token, or sort, partials and scatter), on ids with a run of 25 tokens across chunks, entries no token has and ids outside the
vocabulary, and gradients with NaN, ±inf, ±0, subnormals and the f32 extremes. The model is torch CPU's word for word where the
kernels' orders meet the CPU's. A cc 12.0 capture holds the feature kernel to the f32 model (the Definitions' conformance); no
capture holds the sort kernels."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import bind, conformance_label
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.numerics import torch_kernels as TK
from verity.ml.definitions.registry import torch_embedding as TE
from verity.ml.definitions.registry import torch_embedding_f32 as TF

H = 3
SPECIALS = np.array([0x7FC00001, 0xFFFFFFFF, 0x80000000, 0x00000000, 0x7F800000, 0xFF800000, 0x00000001, 0x807FFFFF,
                     0x7F7FFFFF, 0xFF7FFFFF], np.uint32)


def case(rng, M: int, V: int):
    """M ids below V, 25 of them 2 from M = 70, two outside [0, V) (V and 2^32 - 1), and (M, H) f32 gradients with `SPECIALS`
    at random places."""
    t = rng.integers(0, V, M).astype(np.uint32)
    if M >= 70:
        t[rng.choice(M, 25, replace=False)] = 2
    if M > 2:
        t[[M // 3, M // 2]] = V, 0xFFFFFFFF
    g = np.asarray(rng.normal(0, 2, (M, H)), np.float32).view(np.uint32).reshape(-1)
    n = min(g.size, len(SPECIALS))
    g[rng.choice(g.size, n, replace=False)] = SPECIALS[:n]
    return t, g.reshape(M, H)


@pytest.mark.parametrize("M,V", [(1, 3), (5, 2), (70, 9), (70, 200), pytest.param(TE.FEATURE_MAX + 1, 9, marks=pytest.mark.slow)])
def test_the_gradient_is_the_kernels_model(M, V) -> None:
    """The feature kernel at M = 70 (entry 2's 25 tokens across chunks), a vocabulary far larger than the tokens, one token,
    and the sort path at the least M it runs at, every id's tokens in many partials of 10 (about 10 s)."""
    t, g = case(np.random.default_rng(M + V), M, V)
    got = evaluate_batch(bind(TF.EmbeddingGradF32, M=M, V=V, H=H), t[None], g[None], kernel="reference")
    np.testing.assert_array_equal(np.asarray(got, np.uint32).reshape(V, H), TK.embedding_dense_backward(t, g, range(V)))


@pytest.mark.parametrize("M", [200, 3100])
def test_the_f32_model_is_torchs_where_the_orders_meet(M) -> None:
    """torch's CPU backward adds each entry's tokens into +0 in position order. The feature kernel does too when no chunk of 32
    holds an id twice (here each id is in about five chunks), and the sort path when no id has more than ten tokens (here five),
    whose one partial's +0 + p changes no word but -0, which no sum from +0 is: on such ids, without NaN, the model is torch's
    word for word."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(M)
    if M <= TE.FEATURE_MAX:
        V = 40
        t = np.concatenate([rng.choice(V, min(TE.CHUNK, M - a), replace=False) for a in range(0, M, TE.CHUNK)]).astype(np.uint32)
    else:
        V = M // 5
        t = (rng.permutation(M) % V).astype(np.uint32)
    g = np.asarray(rng.normal(0, 2, (M, H)), np.float32)
    g[1, 0], g[2, 1], g[4, 2] = -0.0, np.float32(1e-45), np.inf
    w = torch.zeros(V, H, requires_grad=True)
    torch.nn.functional.embedding(torch.tensor(t.astype(np.int64)), w).backward(torch.tensor(g))
    np.testing.assert_array_equal(TK.embedding_dense_backward(t, g.view(np.uint32), range(V)), w.grad.numpy().view(np.uint32))


@pytest.mark.parametrize("d,statics", [
    (TF.EmbeddingGradF32, {"M": 0, "V": 8, "H": 4}), (TF.EmbeddingGradF32, {"M": 1 << 16, "V": 1 << 16, "H": 1}),
    (TF.EmbeddingGradF32Sort, {"M": 4, "V": 8, "H": 0}), (TF.EmbeddingGradF32Route, {"M": 4, "V": 0, "H": 2}),
    (TF.EmbeddingGradF32SegmentStep, {"V": 1 << 16, "NP": 20, "FEATURE": 0}),
    (TF.EmbeddingGradF32SegmentStep, {"V": 9, "NP": 4, "FEATURE": 2}),
    (TF.EmbeddingGradF32Run, {"N": 0, "FEATURE": 1}), (TF.EmbeddingGradF32Step, {"FEATURE": True}),
    (TF.EmbeddingGradF32Swap, {"H": 0}), (TF.EmbeddingGradF32Move, {"H": 0}), (TF.EmbeddingGradF32Slot, {"H": True}),
], ids=lambda a: getattr(a, "name", repr(a)))
def test_bind_refuses_statics_outside_the_kernels_range(d, statics) -> None:
    """Every static in its range, and sort keys of at most 31 bits (position_bits(M) + id_bits(V))."""
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(d, **statics)


def test_conformance_is_the_feature_kernels_capture() -> None:
    """Each Definition is exact-model-tested on the cc 12.0 capture of the feature kernel, whose run records it cites, and says
    that no capture holds the sort kernels."""
    ev = EVIDENCE["cc12_embedding_backward_capture"]
    for name in TF.__all__:
        d = getattr(TF, name)
        if not hasattr(d, "conformance"):
            continue
        assert conformance_label(d.conformance) == "exact-model-tested", name
        assert ev["run_record"] in d.conformance and ev["fit"] in d.conformance, name
        assert "sort kernels) only the model" in d.conformance, name
