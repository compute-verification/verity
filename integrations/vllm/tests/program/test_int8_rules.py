"""The exact-int8 linear and attention exported under the pinned `b1-eager-v3` profile (`vllm_bindings.int8`): `verity_int8::row_quant`
binds to `Int8RowQuant_v1{K}` and `verity_int8::scaled_mm` to `ScaledMmS8_v1{K,N,DOT=BlackwellS8ImmaDot32_v1}` per token row, and an int8 attention layer's call
to `Int8KvQuant_v1{S,D}` per KV head then `Int8LutAttentionRow_v1{D,pos+1,SCALE}` per (row, head), the Definitions the Match fold
resolves the same launches to (`observe/fold/patterns/int8.py`); the Programs' words are the torch ops' bit for bit; forms the ops
do not compute are refused by name."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity_vllm.engine import int8_ops
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, vllm_ruleset
from verity.ml.definitions.numerics.serve3_reference import materialize

PROF = dict(mode="profile", profile="b1-eager-v3")
BF = torch.bfloat16


class Linear(nn.Module):
    def __init__(self, w: torch.Tensor, sw: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(w, requires_grad=False)
        self.weight_scale = nn.Parameter(sw, requires_grad=False)

    def forward(self, x):
        return int8_ops.linear(x, self.weight, self.weight_scale)


@pytest.fixture(autouse=True)
def _observed():
    clear_observations()
    yield
    clear_observations()


def _layer(N: int, K: int, sw_shape=None, wdtype=torch.int8) -> Linear:
    g = torch.Generator().manual_seed(1)
    w = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8).to(wdtype)
    sw = (torch.rand(N, 1, generator=g) * 1e-2 + 1e-4).to(torch.float32)
    return Linear(w, sw.reshape(sw_shape or (N, 1)))


def _derive(m: nn.Module, x: torch.Tensor):
    return derive_report(m, (torch.empty_like(x),), **PROF, rules=vllm_ruleset(), name=type(m).__name__)


def _bits(t: torch.Tensor, view, np_u) -> list[int]:
    return [int(v) for v in t.detach().contiguous().view(view).numpy().astype(np_u).reshape(-1)]


def _emitted(rep) -> list[str]:
    return [a["emitted"] for a in rep.provenance["applications"] if a.get("emitted")]


@pytest.mark.parametrize("sw_shape", [None, "flat"])
def test_int8_linear_binds_int8_linear_per_row_and_evaluates_to_the_op(sw_shape):
    torch.manual_seed(0)
    N, K, M = 3, 64, 2
    m = _layer(N, K, (N,) if sw_shape else None)
    x = (torch.randn(M, K) * 2).to(BF)
    rep = _derive(m, x)
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert _emitted(rep) == [f"Int8RowQuant_v1{{K={K}}} x{M}", f"ScaledMmS8_v1{{K={K},N={N},DOT=BlackwellS8ImmaDot32_v1}} x{M}"], _emitted(rep)
    roles = rep.provenance["weight_roles"]
    assert roles["weight"]["role"] == "int8_w" and roles["weight_scale"]["role"] == "int8_ws"
    params = dict(m.named_parameters())
    enc = {torch.int8: (torch.int8, np.uint8), torch.float32: (torch.int32, np.uint32)}
    wwords = sum((_bits(params[w["name"]], *enc[params[w["name"]].dtype]) for w in rep.provenance["weights"]), [])
    t = materialize(rep.program, [_bits(x, torch.int16, np.uint16), wwords])
    got = [int(np.uint32(t[g])) for g in rep.program.circuit.root_outputs()]
    xq, s_a, y = m(x)
    want = _bits(xq, torch.int8, np.uint8) + _bits(s_a, torch.int32, np.uint32) + _bits(y, torch.int16, np.uint16)
    assert got == want


@pytest.mark.parametrize("m,why", [
    (_layer(3, 64, wdtype=torch.float32), "numerics_unregistered"),
    (_layer(3, 48), "K a multiple of 32"),
])
def test_forms_outside_the_definition_refuse_by_name(m, why):
    rep = _derive(m, torch.randn(2, m.weight.shape[1]).to(BF))
    assert rep.unsupported is not None and why in str(rep.unsupported), rep.unsupported


LAYER = "model.layers.0.self_attn.attn"
UAWO = ("unified_attention_with_output(Tensor query, Tensor key, Tensor value, Tensor(a!) output, str layer_name, "
        "Tensor? output_scale=None, Tensor? output_block_scale=None, Tensor? kv_cache_dummy_dep=None) -> ()")


@pytest.fixture
def vllm_attention():
    """vLLM's attention op schema, for the export only (vLLM is not a test dependency); removed after the test."""
    if hasattr(torch.ops, "vllm") and hasattr(torch.ops.vllm, "unified_attention_with_output"):
        yield
        return
    with torch.library._scoped_library("vllm", "FRAGMENT") as lib:
        lib.define(UAWO)
        lib.impl("unified_attention_with_output", lambda *a, **k: None, "CompositeExplicitAutograd")
        torch.library.register_fake("vllm::unified_attention_with_output", lambda *a, **k: None, lib=lib)
        yield


class Attn(nn.Module):
    def forward(self, q, k, v):
        out = torch.empty_like(q)
        torch.ops.vllm.unified_attention_with_output(q, k, v, out, LAYER)
        return out


def _obs(ks: torch.Tensor, scale: float, **over) -> dict:
    from verity_vllm.program.frontend.rules.vllm_bindings.observations import ATTN_PINS
    o = dict(ATTN_PINS, impl="verity_vllm.engine.int8_attention_vllm.Int8AttentionImpl", backend="CUSTOM", scale=scale,
             sliding_window=None, logits_soft_cap=None, int8_k_scales=[int(b) & 0xFFFFFFFF for b in ks.view(torch.int32).tolist()])
    return o | over


def test_int8_attention_binds_kv_quant_and_rows_and_evaluates_to_the_backend(vllm_attention):
    from verity_vllm.engine import int8_attention as IA
    from verity_vllm.program.frontend.rules.vllm_bindings import observe_attention_impls
    g = torch.Generator().manual_seed(3)
    T, H, G, D = 3, 4, 2, 32
    scale = D ** -0.5
    q, k, v = (torch.randn(T, n, D, generator=g).to(BF) for n in (H, G, G))
    ks = (k.float().abs().amax(dim=(0, 2)) / 127).float()
    observe_attention_impls({LAYER: _obs(ks, scale)})
    rep = derive_report(Attn(), (q, k, v), **PROF, rules=vllm_ruleset(), name="Attn")
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert _emitted(rep) == [f"Int8KvQuant_v1{{S={T},D={D}}} x{G} + Int8LutAttentionRow_v1{{S=1..{T}}} x{T * H} + Const32 x{G + T * H}"], _emitted(rep)
    t = materialize(rep.program, [_bits(a, torch.int16, np.uint16) for a in (q, k, v)])
    got = [int(np.uint32(t[o])) for o in rep.program.circuit.root_outputs()]
    assert got == _bits(IA.window_attention(q, k, v, ks, scale), torch.int16, np.uint16)


@pytest.mark.parametrize("h200,kind", [
    (False, "Attention_v3{T=3,NH=2,KVH=1,D=32,BN=128}"),
    (True, "Attention_v4{T=3,NH=2,KVH=1,D=32,BN=128,DOT=HopperBF16WgmmaDot16_v1,INV=Fa3InvSum_v1,MASKED_FROM=0}"),
])
def test_a_flash_attention_layer_through_the_same_op_binds_the_attention_chain(vllm_attention, h200, kind):
    """A bf16 layer's call goes to the attention rule, which shares `extend_history` with the int8 one: the accepted target's chain,
    and on the H200 FA3's, whose tiles read each row's index on the prefill."""
    from tests.program.test_sigmoid_moe_fp8_derive import H200
    from verity_vllm.program.frontend.rules.vllm_bindings import observe_attention_impls, observe_target_profile
    from verity_vllm.program.frontend.rules.vllm_bindings.observations import ATTN_PINS
    T, H, G, D = 3, 2, 1, 32
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(T, n, D, generator=g).to(BF) for n in (H, G, G))
    observe_attention_impls({LAYER: dict(ATTN_PINS, impl="vllm.v1.attention.backends.flash_attn.FlashAttentionImpl", backend="FLASH_ATTN",
                                         scale=D ** -0.5, sliding_window=None, logits_soft_cap=None,
                                         **({"vllm_flash_attn_version": 3} if h200 else {}))})
    if h200:
        observe_target_profile(H200)
    rep = derive_report(Attn(), (q, k, v), **PROF, rules=vllm_ruleset(), name="Attn")
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert _emitted(rep) == [f"{kind} x{T} (T=1..{T})"], _emitted(rep)


@pytest.mark.parametrize("over,why", [
    ({"impl": "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl"}, "is not Int8AttentionImpl"),
    ({"logits_soft_cap": 30.0}, "soft cap"),
    ({"sliding_window": [7, 0]}, "sliding window"),
    ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
    ({"int8_k_scales": []}, "K scales were not read back"),
])
def test_int8_attention_refuses_layers_it_does_not_model(vllm_attention, over, why):
    from verity_vllm.program.frontend.rules.vllm_bindings import observe_attention_impls
    T, H, G, D = 2, 2, 1, 32
    q, k, v = torch.randn(T, H, D).to(BF), torch.randn(T, G, D).to(BF), torch.randn(T, G, D).to(BF)
    observe_attention_impls({LAYER: _obs(torch.ones(G), D ** -0.5, **over)})
    rep = derive_report(Attn(), (q, k, v), **PROF, rules=vllm_ruleset(), name="Attn")
    assert rep.unsupported is not None and why in str(rep.unsupported), rep.unsupported
