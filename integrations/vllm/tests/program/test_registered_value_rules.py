"""The rules under `registered_values` for the remaining chains: the logits softcap's scalars and the generic add-scalar kinds, the
quarantine and reference layer norms and the reference RMSNorm ask for their value-input twins, whose values are fields of the
registered `values` parameter; each Program computes what the constants derivation's computes."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from verity.ml.circuits.constants import program_data
from verity.ml.circuits.program import Program
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, vllm_ruleset
from verity.ml.definitions.registry import dense as DN
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.numerics.serve3_reference import materialize

BF = torch.bfloat16


@pytest.fixture(autouse=True)
def _observed():
    clear_observations()
    yield
    clear_observations()


def _f32(x: float) -> int:
    return int(P.bits_of(np.float32(x)))


def _bits(t: torch.Tensor) -> list[int]:
    if t.dtype == BF:
        return [int(v) for v in t.detach().contiguous().view(torch.int16).numpy().astype(np.uint16).reshape(-1)]
    return [int(v) for v in t.detach().contiguous().view(torch.int32).numpy().astype(np.uint32).reshape(-1)]


def _pair(m: nn.Module, x: torch.Tensor, **kw):
    const = derive_report(m, (torch.empty_like(x),), name=type(m).__name__, **kw)
    clear_observations()
    reg = derive_report(m, (torch.empty_like(x),), name=type(m).__name__, registered_values=True, **kw)
    assert const.unsupported is None and reg.unsupported is None, (const.unsupported, reg.unsupported)
    return const, reg


def _fields(rep) -> list[tuple[str, int]]:
    return [(f["name"], int(f["bits"], 16)) for f in rep.provenance["supplied"]["registration"]["fields"]]


def _ids(rep) -> set[str]:
    return {n.fn.base_id for n in rep.flat_definition.body.nodes}


def _outputs(rep, words) -> list[int]:
    t = materialize(rep.program, words)
    return [int(np.uint32(t[g])) for g in rep.program.circuit.root_outputs()]


def _weights(m: nn.Module, rep) -> list[int]:
    params = dict(m.named_parameters())
    return sum((_bits(params[w["name"]]) for w in rep.provenance["weights"]), [])


def _agree(m, x, const, reg):
    values = [b for _n, b in _fields(reg)]
    w = _weights(m, const)
    held = {cid for h in program_data(Program(reg.flat_definition).root).values() for _g, cid in h}
    assert not {f"Const32[0x{b:08x}]_v1" for b in values} & held
    tail = [w] if w else []
    assert _outputs(reg, [_bits(x), values, *tail]) == _outputs(const, [_bits(x), *tail])


class LogitsSoftcap(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 30.0) * 30.0


class AddScalars(nn.Module):
    def forward(self, x):
        h = x.float()
        return (h + 1e-6).to(BF), (h + 2.0).to(BF), (h + 1e-6).to(BF)


def test_logits_softcap_scalars_are_registered_values():
    x = (torch.randn(2, 16) * 40).to(BF)
    const, reg = _pair(LogitsSoftcap(), x, mode="profile", profile="b1-eager-v3", rules=vllm_ruleset())
    assert _fields(reg) == [("div_scalar", DN.reciprocal_bits(30.0)), ("mul_scalar", _f32(30.0))]
    assert {"Bf16DivScalar_v3", "Bf16MulScalar_v3", "Bf16Tanh_v1"} <= _ids(reg)
    assert not {"Bf16DivScalar_v1", "Bf16MulScalar_v1"} & _ids(reg)
    _agree(LogitsSoftcap(), x, const, reg)


@pytest.mark.parametrize("dtype,kind", [(BF, "bf16"), (torch.float32, "f32")])
def test_distinct_add_scalars_are_distinct_values(dtype, kind):
    """Three adds of two scalars: `x.float()` of a bf16 row is `AddScalarBf16`'s operand, an f32 row `AddScalarF32`'s."""
    x = torch.randn(2, 16).to(dtype)
    const, reg = _pair(AddScalars(), x, mode="profile", profile="b1-eager-v3", rules=vllm_ruleset())
    assert _fields(reg) == [(f"add_scalar_{kind}", _f32(1e-6)), (f"add_scalar_{kind}_1", _f32(2.0))]
    assert f"AddScalar{kind.capitalize()}_v3" in _ids(reg) and f"AddScalar{kind.capitalize()}_v1" not in _ids(reg)
    _agree(AddScalars(), x, const, reg)


class LayerNorm(nn.Module):
    def __init__(self, H: int, dtype=BF) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(H, eps=1e-6, dtype=dtype)
        with torch.no_grad():
            self.norm.weight.copy_((1 + torch.randn(H) * 0.1).to(dtype))
            self.norm.bias.copy_((torch.randn(H) * 0.1).to(dtype))

    def forward(self, x):
        return self.norm(x)


def test_layer_norm_aten_eps_is_a_registered_value():
    torch.manual_seed(1)
    m, x = LayerNorm(32), (torch.randn(3, 32) * 3).to(BF)
    const, reg = _pair(m, x, mode="profile", profile="b1-eager-v3", rules=vllm_ruleset())
    assert _fields(reg) == [("eps", _f32(1e-6))]
    assert "LayerNormAten_v2" in _ids(reg) and "LayerNormAten_v1" not in _ids(reg)
    _agree(m, x, const, reg)


class RefNorms(nn.Module):
    def __init__(self, H: int, dtype) -> None:
        super().__init__()
        self.ln = LayerNorm(H, dtype).norm
        self.w = nn.Parameter((1 + torch.randn(H) * 0.1).to(dtype))

    def forward(self, x):
        return F.rms_norm(self.ln(x), (x.shape[-1],), self.w, 1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, BF])
def test_reference_norms_eps_is_a_registered_value(dtype):
    torch.manual_seed(2)
    m, x = RefNorms(16, dtype), torch.randn(2, 16).to(dtype)
    const, reg = _pair(m, x, mode="reference")
    assert _fields(reg) == [("eps", _f32(1e-6))]
    assert {"LayerNormRef_v2", "RMSNormRef_v2"} <= _ids(reg) and not {"LayerNormRef_v1", "RMSNormRef_v1"} & _ids(reg)
    _agree(m, x, const, reg)


H100 = dict(compute_capability=(9, 0), num_sms=132)
PRO6000 = dict(compute_capability=(12, 0), num_sms=188)


@pytest.mark.parametrize("target,cap,const_base,value_base", [
    ({}, None, "Attention_v3", "Attention_v10"),
    ({}, 50.0, "AttentionSoftcap_v1", "AttentionSoftcap_v4"),
    (dict(H100, flash_attn_version=2), None, "Attention_v2", "Attention_v12"),
    (dict(H100, flash_attn_version=3), None, "Attention_v4", "Attention_v13"),
    (PRO6000, None, "Attention_v5", "Attention_v11"),
    (PRO6000, 50.0, "AttentionSoftcap_v2", "AttentionSoftcap_v5"),
])
def test_every_attention_chain_has_its_value_input_twin(target, cap, const_base, value_base):
    """`targets.attention_spec(..., scale_input=True)` is the constant chain's twin with its statics, less `CAP`; the builder's kind for
    the target names the same Definition."""
    from verity_vllm.program.frontend.rules.vocab import PROFILES
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.frontend import targets as TG

    tp = TargetProfile(**target)
    T, NH, KVH, D = 33, 8, 4, 64
    const = TG.attention_spec(tp, T, NH, KVH, D, 1, row=0, own_rows=1, cap=cap)
    value = TG.attention_spec(tp, T, NH, KVH, D, 1, row=0, own_rows=1, cap=cap, scale_input=True)
    assert (const.base_id, value.base_id) == (const_base, value_base)
    assert value.bindings == {k: v for k, v in const.bindings.items() if k != "CAP"}
    kinds, g = PROFILES["b1-eager-v3"].kinds, value.bindings
    if TG.uses_accepted_kinds(tp):
        got = kinds["attention_softcap_value" if cap else "attention_value"](T=T, NH=NH, KVH=KVH, D=D)
    else:
        mf = {"FA2_MASKED_FROM": g["MASKED_FROM"]} if TG.fa2_check_inf(tp) else {"MASKED_FROM": g["MASKED_FROM"]} if "MASKED_FROM" in g else {}
        role = ("attention_softcap_dot_value" if cap else "attention_dot_value" if TG.fa2_check_inf(tp)
                else "attention_dot_fa3_value" if TG.fa3_check_inf(tp) else "attention_dot_plain_value")
        got = kinds[role](T=T, NH=NH, KVH=KVH, D=D, BN=g["BN"], DOT=g["DOT"], INV=g["INV"], **mf)
    assert got.id == value.id
