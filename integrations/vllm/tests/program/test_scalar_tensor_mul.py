"""R13 X-08 — the generic bf16-row x 0-d-bf16-tensor multiply (`ScalarTensorMul[aten]_v1` -> `Bf16MulScalarTensor_v1{N}`) as a
REGISTERED kind of the pinned `b1-eager-v3` vocabulary.

What is under test: `aten.mul.Tensor(x[.., N] bf16, s bf16 [] | [1])` (either operand order; `s` a buffer, a parameter or a graph
input) is bound to one `Bf16MulScalarTensor_v1{N}` call per row of `x` sharing `s`, and the derived Program evaluates BIT-EXACTLY to
eager PyTorch (ATen bf16 mul = widen, one f32 multiply, RNE narrow; `F2fpBf16` and torch agree on every non-NaN word).  Every other
`aten.mul.Tensor` shape / dtype combination is refused fail-closed (`numerics_unregistered`), never bound to a different numeric.
Tests of the operation — Gemma-2's `embed * model.normalizer` (N=2304, s=48) is one instance, not the recognizer.

Run (pod venv, CPU): `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/program/test_scalar_tensor_mul.py`
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
from verity_vllm.program.kernels.serve3_reference import materialize

PROF = dict(mode="profile", profile="b1-eager-v3")
BF = torch.bfloat16
RULE = "ScalarTensorMul[aten]_v1"


# ---------------------------------------------------------------------------------------------------------
# modules: plain `x * s` graphs
# ---------------------------------------------------------------------------------------------------------


class XTimesBuffer(nn.Module):
    """`x[M, N] * s` with `s` a 0-d bf16 BUFFER (Gemma2Model: `self.normalizer = torch.tensor(hidden_size**0.5, dtype=model dtype)`)."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("normalizer", torch.tensor(value, dtype=BF), persistent=False)

    def forward(self, x):
        return x * self.normalizer


class BufferTimesX(XTimesBuffer):
    def forward(self, x):
        return self.normalizer * x


class XTimesParam1(nn.Module):
    """`x[B, T, N] * s[1]`: a `[1]` parameter — the same kernel, one call per row in row-major order."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.tensor([value], dtype=BF), requires_grad=False)

    def forward(self, x):
        return x * self.s


class XTimesInput(nn.Module):
    """`s` is a graph INPUT: the operation does not depend on where the scalar comes from."""

    def forward(self, x, s):
        return x * s


class XTimesF32Buffer(nn.Module):
    """A 0-d f32 buffer: torch keeps bf16 out but the kernel's scalar path differs (CPU-scalar opmath vs cast) — refused."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("normalizer", torch.tensor(48.0), persistent=False)

    def forward(self, x):
        return x * self.normalizer


class XTimesRow(nn.Module):
    """`x[M, N] * w[N]`: a per-column scale — a DIFFERENT op (no kind), refused."""

    def __init__(self, N: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(N).to(BF), requires_grad=False)

    def forward(self, x):
        return x * self.w


class XTimesPython(nn.Module):
    """`x * 48.0` (a python scalar -> `aten.mul.Tensor(x, 48.0)` with a constant): not a scalar TENSOR, refused."""

    def forward(self, x):
        return x * 48.0


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------


def _bits(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.detach().contiguous().view(torch.int16).numpy().astype(np.uint16).reshape(-1)]


def _evaluate(rep, module: nn.Module, inputs: list[torch.Tensor]) -> np.ndarray:
    tensors = {**dict(module.named_buffers()), **dict(module.named_parameters())}
    words = [_bits(x) for x in inputs]
    wbits: list[int] = []
    for w in rep.provenance["weights"]:
        wbits += _bits(tensors[w["name"]])
    if wbits:
        words.append(wbits)
    t = materialize(rep.program, words)
    return np.array([np.uint32(t[g]) for g in rep.program.circuit.root_outputs()], dtype=np.uint32)


def _derive(m: nn.Module, *inputs: torch.Tensor):
    return derive_report(m, tuple(torch.empty_like(t) for t in inputs), **PROF, rules=vllm_ruleset(), name=type(m).__name__)


def _assert_exact(m: nn.Module, *inputs: torch.Tensor, rows: int, N: int):
    rep = _derive(m, *inputs)
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert RULE in rep.provenance["rules_applied"], rep.provenance["rules_applied"]
    apps = [a for a in rep.provenance["applications"] if a["rule"] == RULE]
    assert len(apps) == 1 and apps[0]["emitted"] == f"Bf16MulScalarTensor_v1{{N={N}}} x{rows}", apps
    want = np.array(_bits(m(*inputs)), dtype=np.uint32)
    got = _evaluate(rep, m, list(inputs))
    assert got.shape == want.shape and np.array_equal(got, want), "derived Program is not bit-identical to eager bf16 mul"
    return rep


# ---------------------------------------------------------------------------------------------------------
# bound cases: exact to eager
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [48.0, 0.7, -3.25, 1.0, 1e-3, 3.0e4])
def test_x_times_buffer_bit_exact(value):
    torch.manual_seed(0)
    _assert_exact(XTimesBuffer(value), (torch.randn(5, 24) * 3).to(BF), rows=5, N=24)


def test_scalar_first_is_the_same_operation():
    torch.manual_seed(1)
    _assert_exact(BufferTimesX(48.0), (torch.randn(3, 16) * 3).to(BF), rows=3, N=16)


def test_three_d_operand_and_one_element_parameter():
    torch.manual_seed(2)
    _assert_exact(XTimesParam1(0.3), (torch.randn(2, 3, 8) * 3).to(BF), rows=6, N=8)


def test_scalar_as_graph_input():
    torch.manual_seed(3)
    _assert_exact(XTimesInput(), (torch.randn(4, 10) * 3).to(BF), torch.tensor(2.5, dtype=BF), rows=4, N=10)


def test_rounding_is_one_rne_on_the_f32_product():
    """f32(x) * f32(s) is exact (8 x 8 significant bits); the single RNE narrow decides the word: products chosen to land at ties and
    just off bf16 grid points."""
    m = XTimesBuffer(1.0 + 2.0 ** -7)                                    # s = 1.0078125 (bf16 exact)
    x = torch.tensor([[1.0, 1.0 + 2.0 ** -7, 255.0, 2.0 ** -126, 3.0, -1.5, 1.0 + 2.0 ** -8 + 2.0 ** -7, 65280.0]]).to(BF)
    _assert_exact(m, x, rows=1, N=8)


def test_gemma_shape_instance():
    """The instance that motivated the kind: `embed[M, 2304] * 48` (sqrt(hidden_size) at bf16)."""
    torch.manual_seed(4)
    _assert_exact(XTimesBuffer(2304 ** 0.5), (torch.randn(2, 2304) * 0.05).to(BF), rows=2, N=2304)


# ---------------------------------------------------------------------------------------------------------
# fail-closed
# ---------------------------------------------------------------------------------------------------------


def _refused(m: nn.Module, *inputs: torch.Tensor) -> str:
    rep = _derive(m, *inputs)
    assert rep.unsupported is not None, "must refuse"
    assert RULE not in rep.provenance.get("rules_applied", ())
    return f"{rep.unsupported.reason}: {rep.unsupported.hint}"


def test_f32_scalar_tensor_refused():
    hint = _refused(XTimesF32Buffer(), torch.randn(2, 8).to(BF))
    assert "ScalarTensorMul[aten]_v1: numerics_unregistered: s has dtype torch.float32" in hint, hint


def test_per_column_scale_refused():
    hint = _refused(XTimesRow(8), torch.randn(2, 8).to(BF))
    assert "ScalarTensorMul[aten]_v1: neither operand is a scalar tensor" in hint, hint


def test_python_scalar_is_not_this_rule():
    """`x * 48.0` is `aten.mul.Tensor(x, 48.0)` with a Python constant: NOT this rule (`not both graph values`).  [R15 dense] Since the
    `bf16_mul_scalar` kind joined `b1-eager-v3` (`Bf16MulScalar_v1{N,C}`: ATen's bf16 x scalar kernel, difftested in quarantine/dense),
    `Bf16ScalarChain[aten]_v1` binds it — a different, registered kind with the constant in its identity; before R15 it was refused."""
    rep = _derive(XTimesPython(), torch.randn(2, 8).to(BF))
    assert RULE not in rep.provenance.get("rules_applied", ()), rep.provenance.get("rules_applied")
    assert rep.unsupported is None, rep.unsupported
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "Bf16ScalarChain[aten]_v1"]
    assert len(apps) == 1 and apps[0]["emitted"] == "Bf16MulScalar_v1{N=8,C=48.0} x2", apps


def test_kind_is_registered_in_the_pinned_vocabulary():
    from verity_vllm.program.frontend.rules.vocab import PROFILES
    from verity_vllm.program.registry import b1
    from verity_vllm.program.registry.quarantine.dense import bf16_mul_scalar_tensor as q
    fn = PROFILES["b1-eager-v3"].kinds["scale_tensor"](N=2304)
    assert fn.id == "Bf16MulScalarTensor_v1{N=2304}", fn.id
    assert q.Bf16MulScalarTensor is b1.Bf16MulScalarTensor, "quarantine must re-export the registered Definition"
    assert "scale_tensor" not in PROFILES["b1-eager"].kinds, "the accepted b1-eager vocabulary is unchanged"
