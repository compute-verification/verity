"""GPT-NeoX's `nn.LayerNorm` (weight + bias) exported under the pinned `b1-eager-v3` profile binds `aten.layer_norm` to
`LayerNormAten_v1{N,EPS}` per row (`vllm_bindings.layer_norm`), the Definition the eager Match fold resolves the same op to
(`engine/profiles/gen_ln_patterns.py`); the Program's rows evaluate to the Definition's own words; forms the vectorized kernel does
not run are refused by name.  The kernel equality is the difftest's (`registry/layer_norm_difftest.py`, on the GPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, vllm_ruleset
from verity_vllm.program.registry.layer_norm import LayerNormAten
from verity_vllm.program.kernels.serve3_reference import materialize

PROF = dict(mode="profile", profile="b1-eager-v3")
BF = torch.bfloat16


class Layer(nn.Module):
    def __init__(self, H: int, eps: float = 1e-5, bias: bool = True, dtype=BF) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(H, eps=eps, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.input_layernorm(x)


@pytest.fixture(autouse=True)
def _observed():
    clear_observations()
    yield
    clear_observations()


def _derive(m: nn.Module, x: torch.Tensor):
    return derive_report(m, (torch.empty_like(x),), **PROF, rules=vllm_ruleset(), name=type(m).__name__)


def _bits16(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.detach().contiguous().cpu().view(torch.int16).numpy().astype(np.uint16).reshape(-1)]


def _emitted(rep) -> list[str]:
    return [a["emitted"] for a in rep.provenance["applications"] if a.get("emitted")]


def test_layer_norm_binds_layer_norm_aten_per_row_and_evaluates_to_it():
    torch.manual_seed(0)
    H, M = 32, 3
    m = Layer(H)
    with torch.no_grad():
        m.input_layernorm.weight.copy_((1 + torch.randn(H) * 0.1).to(BF))
        m.input_layernorm.bias.copy_((torch.randn(H) * 0.1).to(BF))
    x = (torch.randn(M, H) * 3).to(BF)
    rep = _derive(m, x)
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert _emitted(rep) == [f"LayerNormAten_v1{{N={H},EPS=1e-05}} x{M}"], _emitted(rep)
    roles = rep.provenance["weight_roles"]
    assert roles["input_layernorm.weight"]["role"] == "layer_norm.weight" and roles["input_layernorm.bias"]["role"] == "layer_norm.bias"
    params = dict(m.named_parameters())
    words = [_bits16(x)] + [sum((_bits16(params[w["name"]]) for w in rep.provenance["weights"]), [])]
    t = materialize(rep.program, words)
    got = [int(np.uint32(t[g])) for g in rep.program.circuit.root_outputs()]
    fn = bind(LayerNormAten, N=H, EPS=1e-5)
    w, b = _bits16(m.input_layernorm.weight), _bits16(m.input_layernorm.bias)
    want = [int(v) for r in range(M) for v in evaluate(fn, _bits16(x[r]), w, b)]
    assert got == want


def test_the_modules_eps_is_the_kinds():
    rep = _derive(Layer(16, eps=1e-6), torch.randn(2, 16).to(BF))
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert _emitted(rep) == ["LayerNormAten_v1{N=16,EPS=1e-06} x2"], _emitted(rep)


@pytest.mark.parametrize("layer,x,why", [
    (Layer(16, bias=False), torch.randn(2, 16).to(BF), "with both weight and bias"),
    (Layer(30), torch.randn(2, 30).to(BF), "not a multiple of 4"),
    (Layer(16, dtype=torch.float32), torch.randn(2, 16), "numerics_unregistered"),
])
def test_forms_the_vectorized_bf16_kernel_does_not_run_are_refused(layer, x, why):
    rep = _derive(layer, x)
    u = rep.unsupported.to_json() if rep.unsupported is not None else None
    assert u is not None and "layer_norm" in u["node"], u
    refusals = f"{u['hint']} {u['detail']}"
    assert "LayerNormAten[aten]" in refusals and why in refusals, u
