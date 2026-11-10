"""The training rules' whole-tensor reductions at a compute capability (`rules.training_reduce.FullReduce`): a vector norm of
bf16 and of f32 values and a sum of f32 values, each bit for bit against `torch_kernels.reduce_all`, the model of the GPU's
`reduce_kernel` that `verity/definitions/tests/test_torch_training_gpu.py` holds to an sm_120 GPU's words for bf16 norms.
Lengths on the kernel's scalar path, its vectorized path and across CTAs, with launches patched to many CTAs a lane; a transposed
operand read in memory order; Muon's normalization against the GPU's arithmetic stated in torch CPU's f32 operations; and the
refusals."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.program.test_train_step import _bits, _data, _f32_of, _outputs, _program, _same_bits
from verity.definitions.numerics import torch_kernels as TK
from verity.definitions.registry import cublas_bf16 as CB
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.rules.training import training_rules
from verity_vllm.program.frontend.rules.training_reduce import FullReduce

GPU = training_rules(cc=CB.TRAIN_CC)


def _word(t: torch.Tensor) -> int:
    return int(t.reshape(-1).view(torch.int32)[0]) & 0xFFFFFFFF


def _reduced(f, x: torch.Tensor) -> int:
    """The Program's one output word (a bf16 word widened, as `_outputs` returns it, shifted back)."""
    w = _word(_outputs(_program(f, x, rules=GPU), x))
    return w >> 16 if f(x)[0].dtype == torch.bfloat16 else w


def _models(xb: torch.Tensor, xf: torch.Tensor) -> tuple[int, int, int]:
    """The model's bf16 norm of xb, f32 norm of xf and f32 sum of xf, each read in memory order."""
    wf = np.array(_bits(xf), np.uint32)
    norm_f32 = int(TK.sqrt(np.array([TK.reduce_all(wf, square=True)], np.uint32))[0])
    return TK.vector_norm(np.array(_bits(xb), np.uint16))[1], norm_f32, TK.reduce_all(wf, square=False)


def _check(L: int) -> None:
    g = torch.Generator().manual_seed(L)
    xb, xf = torch.randn(L, generator=g).bfloat16(), torch.randn(L, generator=g) * 3
    got = (_reduced(lambda a: (a.norm(),), xb), _reduced(lambda a: (a.norm(),), xf), _reduced(lambda a: (a.sum(),), xf))
    assert got == _models(xb, xf), (L, TK.reduce_config(L), [hex(w) for w in got], [hex(w) for w in _models(xb, xf)])


@pytest.mark.parametrize("L", [2, 3, 5, 64, 127, 128, 131, 1000, 4099, 130561])
def test_full_reduce_is_reduce_kernels_order(L: int) -> None:
    """The scalar path (2 to 127 elements), the vectorized path with and without its tail, and 16 CTAs past 130,560."""
    _check(L)


@pytest.mark.parametrize("config, L", [((True, 8, 20), 1000), ((True, 8, 20), 1003), ((True, 4, 9), 150)])
def test_full_reduce_adds_every_ctas_partial(monkeypatch, config, L: int) -> None:
    """The last CTA's lanes adding two and three partials each (as past 512 CTAs), on launches patched into the model and the
    rule alike."""
    monkeypatch.setattr(TK, "reduce_config", lambda n: config)
    _check(L)


def test_a_transposed_operand_is_read_in_memory_order() -> None:
    """``x.T.sum()`` is the sum of x's words in x's own order, which here differs from the order of ``x.T``'s elements."""
    x = torch.randn(48, 40, generator=torch.Generator().manual_seed(20))
    in_memory, in_order = (TK.reduce_all(np.array(_bits(t), np.uint32), square=False) for t in (x, x.T.contiguous()))
    assert in_memory != in_order
    assert _reduced(lambda a: (a.T.sum(),), x) == in_memory
    assert _reduced(lambda a: (a.T.norm(),), x.bfloat16()) == TK.vector_norm(np.array(_bits(x.bfloat16()), np.uint16))[1]


@pytest.mark.parametrize("keepdim", [False, True])
def test_muons_normalization_is_the_gpus(keepdim: bool) -> None:
    """``x = g.bfloat16().T; x / (x.norm() + 1e-7)`` (and the keepdim form over both dims): the norm the model's, its add of 1e-7
    in f32 rounded once to bf16, and the division of each element by it in f32 rounded once, every word."""
    g = _data(96, 40, seed=21)

    def f(a):
        x = a.bfloat16().T
        n = x.norm(dim=(-2, -1), keepdim=True) if keepdim else x.norm()
        return (x / (n + 1e-7),)

    got = _outputs(_program(f, g, rules=GPU), g)
    xb = g.bfloat16()
    n = torch.tensor([TK.vector_norm(np.array(_bits(xb), np.uint16))[1] << 16], dtype=torch.int32).view(torch.float32)
    d = (n + _f32_of(1e-7)).bfloat16().float()
    _same_bits(got, (xb.T.float() / d).bfloat16().float().reshape(-1))


def test_gpu_and_cpu_sums_differ() -> None:
    """Without a compute capability a sum is `Reduce`'s, in torch CPU's order, which gives another word on this data."""
    x = _data(1000, seed=22) * 1e3
    cpu, gpu = (_word(_outputs(_program(lambda a: (a.sum(),), x, rules=r), x)) for r in (None, GPU))
    assert gpu == TK.reduce_all(np.array(_bits(x), np.uint32), square=False) != cpu


def test_full_reduce_definitions() -> None:
    xb = _data(1000, seed=23).bfloat16()
    ids = {nd.fn.id.split("{")[0] for nd in _program(lambda a: (a.norm(), a.float().sum()), xb, rules=GPU).root.body.nodes}
    assert {"Bf16ToF32_v1", "F32Mul_v2", "F32Fma_v2", "F32Add_v2", "PouwF32SqrtRn_v1", "F2fpBf16_v1"} <= ids, ids


def _refusal(f, x, rules) -> Unsupported:
    gm = make_fx(torch.func.functionalize(f, remove="mutations"), tracing_mode="fake")(x)
    u = derive_graph(gm, (x,), mode="reference", rules=rules, name="T").unsupported
    assert isinstance(u, Unsupported) and u.reason == "numerics_unregistered", u
    return u


@pytest.mark.parametrize("f, why", [
    (lambda a: (a[:, :20].norm(),), "not dense in memory"),
    (lambda a: (a.reshape(-1)[1:].norm(),), "at element 1"),
    (lambda a: (a[:1, :1].norm(),), "one element"),
    (lambda a: (torch.linalg.vector_norm(a, 3),), "ord other than 2"),
    (lambda a: (torch.linalg.vector_norm(a, dtype=torch.float32),), "a dtype argument"),
])
def test_full_reduce_refusals(f, why: str) -> None:
    assert why in _refusal(f, _data(8, 40, seed=24).bfloat16(), GPU).hint


def test_a_norm_needs_a_compute_capability() -> None:
    assert "compute capability" in _refusal(lambda a: (a.norm(),), _data(8, 40, seed=25), training_rules()).hint
    with pytest.raises(ValueError, match="RTX PRO 6000"):
        FullReduce(cc=(9, 0))
