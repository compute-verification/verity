"""The training rules' embedding gradient at a compute capability (`rules.training.EmbeddingDenseBackward` at `cc`): an f32 table's
gradient as embedding_dense_backward_cuda's kernels compute it, one `torch_embedding_f32.EmbeddingGradF32_v1` call, word for word
against `torch_kernels.embedding_dense_backward`, the model `verity/definitions/tests/test_torch_embedding_f32.py` holds the
Definition to: int64 and int32 ids, ids out of range, the feature kernel's chunks and the sort path; and the refusal of sort keys
past 31 bits."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.program.test_train_step import _bits, _outputs, _program, _same_bits
from verity.circuits.defs import bind
from verity.definitions.numerics import torch_kernels as TK
from verity.definitions.registry import cublas_bf16 as CB
from verity.definitions.registry import torch_embedding_f32 as TEF
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.rules.training import training_rules

aten = torch.ops.aten
GPU = training_rules(cc=CB.TRAIN_CC)
V, H = 9, 3


def _backward(g, ids):
    return (aten.embedding_dense_backward(g, ids, V, -1, False),)


def _case(M: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """(g, ids) for M tokens as (2, M / 2): ids repeating within and across chunks of 32, among them V and -1 and, as int64, 2^33 + 2
    (out of range, which a truncation to 32 bits would put in range), and gradients with signed zeros, a subnormal, an infinity and
    a NaN."""
    gen = torch.Generator().manual_seed(M)
    ids = torch.randint(0, V, (M,), generator=gen, dtype=torch.int64)
    bad = [V, -1] + ([(1 << 33) + 2] if dtype == torch.int64 else [])
    ids[[1, M // 3, M // 2][:len(bad)]] = torch.tensor(bad)
    g = torch.randn(M, H, generator=gen) * 2
    g.reshape(-1)[[0, 5, 9, 14, 20]] = torch.tensor([-0.0, 0.0, 1e-45, float("inf"), float("nan")])
    return g.reshape(2, M // 2, H), ids.to(dtype).reshape(2, M // 2)


def _model(g: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The model's gradient table on the same words, every id outside [0, V) taken as V, which no entry matches."""
    t = ids.reshape(-1).numpy()
    t = np.where((t >= 0) & (t < V), t, V).astype(np.uint32)
    want = TK.embedding_dense_backward(t, np.array(_bits(g), np.uint32).reshape(-1, H), range(V))
    return torch.from_numpy(want.reshape(-1).view(np.float32).copy())


@pytest.mark.parametrize("M, dtype", [(70, torch.int64), (64, torch.int32),
                                      pytest.param(TK.EMBEDDING_FEATURE_MAX + 2, torch.int64, marks=pytest.mark.slow)])
def test_the_gradient_is_the_cuda_kernels(M: int, dtype: torch.dtype) -> None:
    """At `cc` the gradient is one `EmbeddingGradF32_v1` call, whose words are the model's: the feature kernel's chunks of 32
    tokens, and past 3,072 tokens the sort path's partials of 10."""
    g, ids = _case(M, dtype)
    prog = _program(_backward, g, ids, rules=GPU)
    calls = [nd.fn.id for nd in prog.root.body.nodes if nd.fn.id.startswith("EmbeddingGradF32")]
    assert calls == [bind(TEF.EmbeddingGradF32, M=M, V=V, H=H).id], calls
    _same_bits(_outputs(prog, g, ids), _model(g, ids))


def test_sort_keys_past_31_bits_are_refused() -> None:
    """65,536 tokens of a 65,536-word vocabulary make sort keys of 33 bits, which `I32Le_v1` can't order: the rule refuses with the
    Definition's reason, and the dense CPU sum doesn't stand in for it."""
    n = 1 << 16
    xs = (torch.empty(n, 1, device="meta"), torch.empty(n, dtype=torch.int64, device="meta"))
    gm = make_fx(lambda g, i: (aten.embedding_dense_backward(g, i, n, -1, False),), tracing_mode="fake")(*xs)
    u = derive_graph(gm, xs, mode="reference", rules=GPU, name="T").unsupported
    assert isinstance(u, Unsupported) and u.reason == "numerics_unregistered" and "33 bits" in u.hint, u
