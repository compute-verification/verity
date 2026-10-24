"""R13 X-04 — the general profile-mode broadcast bias add (`BroadcastBiasAdd[profile]_v1` -> `BiasAdd_v1{N}`).

What is under test: `aten.add.Tensor(x[.., N] bf16, b[N] bf16)` (either operand order; `b` may also be `[1,..,1,N]`) is bound to one
`BiasAdd_v1{N}` call per row of `x`, the derived Program evaluates BIT-EXACTLY to eager PyTorch (bf16 add = one f32 add + RNE, the
registered `P.Bf16Add`), the bias parameter is recorded with role `linear.bias`, and every other broadcast pattern is refused
fail-closed (never silently bound to a different numeric).  These are tests of the operation, not of a Qwen layer: the modules are
ordinary `y + b` graphs with unrelated shapes; the vLLM shape (persistent GEMM output + separate bias add) is one case among them.

Run (pod venv, CPU): `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/program/test_broadcast_bias_add.py`
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.kernels.serve3_reference import materialize

PROF = dict(mode="profile", profile="b1-eager")
BF = torch.bfloat16


# ---------------------------------------------------------------------------------------------------------
# modules: plain `x + b` graphs
# ---------------------------------------------------------------------------------------------------------


class XPlusB(nn.Module):
    """`x[M, N] + b[N]` with `b` a parameter (what the batch-invariant linear emits after the GEMM when bias is not None)."""

    def __init__(self, N: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.randn(N).to(BF), requires_grad=False)

    def forward(self, x):
        return x + self.bias


class BPlusX(XPlusB):
    """Bias first: IEEE addition is commutative, the same kind call."""

    def forward(self, x):
        return self.bias + x


class XPlusB3D(XPlusB):
    """A 3-D operand `[B, T, N] + [N]`: one call per row, rows in row-major order."""


class XPlusB1N(nn.Module):
    """`x[M, N] + b[1, N]`: the identical broadcast written with an explicit leading 1."""

    def __init__(self, N: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.randn(1, N).to(BF), requires_grad=False)

    def forward(self, x):
        return x + self.bias


class XPlusBInput(nn.Module):
    """The bias is a graph INPUT, not a parameter: the operation does not depend on where `b` comes from."""

    def forward(self, x, b):
        return x + b


class XPlusBAlpha(XPlusB):
    def forward(self, x):
        return torch.add(x, self.bias, alpha=2)


class LeadingBroadcast(nn.Module):
    """`x[M, N] + c[M, 1]`: a column broadcast — a different op, must NOT be bound as a bias add."""

    def __init__(self, M: int) -> None:
        super().__init__()
        self.col = nn.Parameter(torch.randn(M, 1).to(BF), requires_grad=False)

    def forward(self, x):
        return x + self.col


class F32Bias(nn.Module):
    """bf16 x + f32 b promotes to f32: a different numeric from BiasAdd_v1 (refused: numerics_unregistered)."""

    def __init__(self, N: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.randn(N), requires_grad=False)

    def forward(self, x):
        return x + self.bias


class AmbiguousRowBias(nn.Module):
    """`x[N, N] + b[N]`: N == M — still a trailing-axis broadcast (torch aligns trailing axes) and must be bound as such."""

    def __init__(self, N: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.randn(N).to(BF), requires_grad=False)

    def forward(self, x):
        return x + self.bias


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------


def _bits(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.detach().contiguous().view(torch.int16).numpy().astype(np.uint16).reshape(-1)]


def _evaluate(rep, module: nn.Module, inputs: list[torch.Tensor]) -> np.ndarray:
    tensors = dict(module.named_parameters())
    words = [_bits(x) for x in inputs]
    wbits: list[int] = []
    for w in rep.provenance["weights"]:
        wbits += _bits(tensors[w["name"]])
    if wbits:
        words.append(wbits)
    t = materialize(rep.program, words)
    return np.array([np.uint32(t[g]) for g in rep.program.circuit.root_outputs()], dtype=np.uint32)


def _derive(m: nn.Module, *inputs: torch.Tensor):
    return derive_report(m, tuple(torch.empty_like(t) for t in inputs), **PROF, name=type(m).__name__)


def _assert_exact(m: nn.Module, *inputs: torch.Tensor, rows: int, N: int):
    rep = _derive(m, *inputs)
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert "BroadcastBiasAdd[profile]_v1" in rep.provenance["rules_applied"], rep.provenance["rules_applied"]
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "BroadcastBiasAdd[profile]_v1"]
    assert len(apps) == 1 and apps[0]["emitted"] == f"BiasAdd_v1{{N={N}}} x{rows}", apps
    want = np.array(_bits(m(*inputs)), dtype=np.uint32)
    got = _evaluate(rep, m, list(inputs))
    assert got.shape == want.shape and np.array_equal(got, want), "derived Program is not bit-identical to eager bf16 add"
    return rep


# ---------------------------------------------------------------------------------------------------------
# bound cases: exact to eager
# ---------------------------------------------------------------------------------------------------------


def test_x_plus_b_is_biasadd_per_row_bit_exact():
    torch.manual_seed(0)
    m, x = XPlusB(24), (torch.randn(5, 24) * 3).to(BF)
    rep = _assert_exact(m, x, rows=5, N=24)
    assert rep.provenance["weight_roles"]["bias"]["role"] == "linear.bias", rep.provenance["weight_roles"]


def test_bias_first_is_the_same_operation():
    torch.manual_seed(1)
    _assert_exact(BPlusX(16), (torch.randn(3, 16) * 3).to(BF), rows=3, N=16)


def test_three_d_operand_rows_in_row_major_order():
    torch.manual_seed(2)
    _assert_exact(XPlusB3D(8), (torch.randn(2, 3, 8) * 3).to(BF), rows=6, N=8)


def test_leading_one_bias_is_the_identical_broadcast():
    torch.manual_seed(3)
    _assert_exact(XPlusB1N(12), (torch.randn(4, 12) * 3).to(BF), rows=4, N=12)


def test_bias_as_graph_input():
    torch.manual_seed(4)
    _assert_exact(XPlusBInput(), (torch.randn(3, 10) * 3).to(BF), torch.randn(10).to(BF), rows=3, N=10)


def test_square_operand_still_trailing_axis_broadcast():
    torch.manual_seed(5)
    _assert_exact(AmbiguousRowBias(6), (torch.randn(6, 6) * 3).to(BF), rows=6, N=6)


def test_rounding_is_one_rne_not_double_rounding():
    """Values chosen so that f32(x)+f32(b) lands off a bf16 grid point: the result must be torch's single RNE rounding."""
    m = XPlusB(4)
    with torch.no_grad():
        m.bias.copy_(torch.tensor([1.0, 1.0, 1.0, 1.0]).to(BF))
    x = torch.tensor([[2.0 ** -8, 3 * 2.0 ** -9, 1.0 + 2.0 ** -7, 255.5]]).to(BF)
    _assert_exact(m, x, rows=1, N=4)


# ---------------------------------------------------------------------------------------------------------
# fail-closed: other broadcasts / numerics are refused, never bound
# ---------------------------------------------------------------------------------------------------------


def _refused(m: nn.Module, *inputs: torch.Tensor) -> str:
    rep = _derive(m, *inputs)
    assert rep.unsupported is not None, "must refuse"
    assert "BroadcastBiasAdd[profile]_v1" not in rep.provenance.get("rules_applied", ())
    return rep.unsupported.hint


def test_alpha_not_one_refused():
    hint = _refused(XPlusBAlpha(8), torch.randn(2, 8).to(BF))
    assert "alpha" in hint


def test_leading_axis_broadcast_refused():
    hint = _refused(LeadingBroadcast(4), torch.randn(4, 8).to(BF))
    assert "not a [.., N] + [N] bias broadcast" in hint


def test_f32_bias_refused_as_unregistered_numerics():
    hint = _refused(F32Bias(8), torch.randn(2, 8).to(BF))
    assert "numerics_unregistered" in hint


def test_same_shape_add_still_residual_add():
    """`x + y` with equal shapes is `ResidualAdd[profile]`, untouched by the new rule."""

    class Res(nn.Module):
        def forward(self, x, y):
            return x + y

    rep = _derive(Res(), torch.randn(3, 8).to(BF), torch.randn(3, 8).to(BF))
    assert rep.unsupported is None and "ResidualAdd[profile]_v1" in rep.provenance["rules_applied"]
    assert "BroadcastBiasAdd[profile]_v1" not in rep.provenance["rules_applied"]
