"""The Build's DeepSeek sparse attention (GLM-5.3 on the H200), on a synthetic layer that makes `DeepseekV32Attention.forward`'s calls:

* `export_compat.dsa_leaves` exports `fused_norm_rope`, `fused_q` and `_sparse_indexer_and_attn` as the `vllm_iface` leaves (the
  meta export has no KV cache, so the kernels' slot-mapping asserts never run) and unwinds; what it does not model refuses at export;
  `export_compat.moe_gate_leaf` does the same for the MoE gate's cuteDSL `ll_bf16_gemm`, which checks for CUDA operands;
* `rules/vllm_dsa.py` derives an indexer layer and an IndexShare layer over a prefill and a decode step: per token the norms and
  RoPEs, `DsaIndexerKRow_v1` and `DsaIndexerQ_v1`, per row `DsaMqaLogits_v3{N}` and `DsaTopKShortRow_v1`, and either the explicitly
  pinned `SparseMla_v1{T=n}` or default `SparseMlaSplit_v1{T=n,NS}` over the layer's first n history rows;
* each precondition refuses by name.

The tiny fixture substitutes the old norm because `DsaRMSNorm_v1` is pinned to the real 128-thread widths (512/2048); a direct
vocabulary assertion below checks its binding.  Its MLA absorbs use the registered default cuBLAS `BatchedGemm_v1`.
FA3's sparse geometry is registered at rope 16, latent 16."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.export_compat import dsa_leaves
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules import vllm_dsa
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3, PROFILES, Vocabulary
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels.serve3_reference import materialize
from verity_vllm.program.registry import b1, cublas_bf16, dsa, dsa_norm, hopper, sparse_mla, sparse_mla_split
from verity_vllm.program.registry.boolean_rope import RoPEHeadGptJV2
from verity_vllm.program.registry.sparse_mla_split import launch_num_splits

H200 = TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False, moe_backend="triton")
NH, P, R, L, V, NQ = 2, 16, 16, 16, 16, 16   # heads, nope, rope, latent, value, q_lora
H, D, ROPE = 8, 32, 16                       # the indexer's heads, head dim, rope
TOPK, MAXPOS, EPS, BN = 4, 8, 1e-6, 16
SCALE = (P + R) ** -0.5
STEPS = ((0, 2), (2, 1))                     # (first position, tokens): a prefill of two, one decode step
NAMES = ("layers.0.self_attn.attn", "layers.1.self_attn.attn")
DOT = hopper.HopperBF16WgmmaDot16
IMPL = "vllm.v1.attention.backends.mla.flashattn_mla_sparse.FlashAttnMLASparseImpl"


STANDIN = Vocabulary("dsa-standin", "profile", "dsa-standin",
                     kinds={**PROFILE_B1_EAGER_V3.kinds, "dsa_rmsnorm": PROFILE_B1_EAGER_V3.kinds["rmsnorm"],
                            "batched_gemm": lambda B, K, N, DOT: bind(cublas_bf16.BatchedGemmWord, B=B, K=K, N=N, DOT=DOT)},
                     notes=dict(PROFILE_B1_EAGER_V3.notes))


def _unpatched(*a, **k):
    raise AssertionError("slot_mapping is not None")


SITE = types.ModuleType("dsa_site_standin")
SITE.fused_norm_rope = SITE.fused_q = _unpatched


def _bf16(rng, *shape, sd=1.0) -> torch.Tensor:
    return torch.from_numpy(rng.normal(0.0, sd, shape).astype(np.float32)).to(torch.bfloat16)


def _cos_sin(width: int) -> torch.Tensor:
    f = 1.0 / (10000.0 ** (np.arange(0, width, 2) / width))
    a = np.outer(np.arange(MAXPOS), f)
    return torch.from_numpy(np.concatenate([np.cos(a), np.sin(a)], axis=1).astype(np.float32)).to(torch.bfloat16)


class _Norm(nn.Module):
    def __init__(self, n: int, rng, bias: bool = False) -> None:
        super().__init__()
        dt = torch.float32 if bias else torch.bfloat16
        self.weight = nn.Parameter(torch.from_numpy(rng.normal(1.0, 0.1, n).astype(np.float32)).to(dt), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.from_numpy(rng.normal(0.0, 0.1, n).astype(np.float32)), requires_grad=False)
        self.variance_epsilon = self.eps = EPS


class _Rope(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("cos_sin_cache", _cos_sin(width))


class _Indexer(nn.Module):
    def __init__(self, rng) -> None:
        super().__init__()
        self.k_norm = _Norm(D, rng, bias=True)
        self.softmax_scale, self.n_head, self.head_dim = D ** -0.5, H, D


class _Layer(nn.Module):
    """`DeepseekV32Attention.forward` (vLLM d9105ea80 deepseek_v32/attention.py:297-408) from its projections' outputs on, which are
    the inputs: q_c, kv_c, k_pe, q (q_b_proj's), and on an indexer layer index K, the index weights and the index query (wq_b's)."""

    def __init__(self, name: str, indexer: bool, seed: int, topk: int = TOPK) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.layer_name, self.num_heads, self.kv_lora_rank, self.v_head_dim, self.qk_nope_head_dim = name, NH, L, V, P
        self.scale, self.kv_cache_dtype, self._index_rope_interleave = SCALE, "auto", True
        self.use_pcp, self._fp8_query, self._fp8_kv_needs_view = False, False, False
        self.impl = types.SimpleNamespace(dcp_world_size=1)
        self.fused_qkv_a_proj = nn.Identity()
        self.q_a_layernorm, self.kv_a_layernorm = _Norm(NQ, rng), _Norm(L, rng)
        self.rotary_emb = _Rope(R)
        self.indexer = _Indexer(rng) if indexer else None
        self.indexer_rope_emb = _Rope(ROPE) if indexer else None
        self.W_UK_T = nn.Parameter(_bf16(rng, NH, P, L, sd=0.25), requires_grad=False)
        self.W_UV = nn.Parameter(_bf16(rng, NH, L, V, sd=0.25), requires_grad=False)
        self.topk_indices_buffer = torch.empty(1, topk, dtype=torch.int32)
        self._q_scale = torch.ones(1)
        self.n_inputs = 7 if indexer else 4

    def _sparse_indexer_and_attn(self, *a):
        raise AssertionError("the attention kernels read a KV cache the meta export does not have")

    def forward(self, positions, q_c, kv_c, k_pe, q, index_k=None, index_weights=None, index_q=None):
        output = torch.empty((q_c.shape[0], NH * V), dtype=q_c.dtype, device=q_c.device)
        if self.indexer is not None:
            has_indexer, k_w, k_b, k_eps = True, self.indexer.k_norm.weight, self.indexer.k_norm.bias, self.indexer.k_norm.eps
            ics, ss, hs = self.indexer_rope_emb.cos_sin_cache, self.indexer.softmax_scale, self.indexer.n_head ** -0.5
        else:
            has_indexer, k_w, k_b, k_eps, ics, ss, hs = False, None, None, 1e-6, None, 0.0, 0.0
        kv_c_out = torch.empty_like(kv_c)
        k_pe_out = torch.empty_like(k_pe)
        q_c = SITE.fused_norm_rope(positions, q_c, self.q_a_layernorm.weight, self.q_a_layernorm.variance_epsilon, kv_c,
                                   self.kv_a_layernorm.weight, self.kv_a_layernorm.variance_epsilon, k_pe, self.rotary_emb.cos_sin_cache,
                                   index_k, k_w, k_b, k_eps, ics, self.topk_indices_buffer, slot_mapping=None, indexer_k_cache=None,
                                   mla_kv_cache=None, mla_kv_cache_dtype=self.kv_cache_dtype, mla_k_scale=None, has_indexer=has_indexer,
                                   index_rope_interleave=self._index_rope_interleave, kv_c_out=kv_c_out, k_pe_out=k_pe_out, index_k_out=None)
        q = q.view(-1, NH, P + R)
        q_nope, q_pe = q.split([P, R], dim=-1)
        ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)
        index_q = index_q.view(-1, H, D) if has_indexer else None
        iq, iw, mqa_q = SITE.fused_q(positions, q_pe, self.rotary_emb.cos_sin_cache, index_q, ics, ql_nope, self._q_scale, index_weights,
                                     ss, hs, has_indexer=has_indexer, index_rope_interleave=self._index_rope_interleave,
                                     quantize_mqa=self._fp8_query)
        self._sparse_indexer_and_attn(positions, q_c, q_nope, q_pe, iq, None, iw, kv_c_out, k_pe_out, ql_nope, mqa_q, output)
        return output, q_c


class _Model(nn.Module):
    """The layers over the steps, each step's positions an `arange` (as `vllm_meta`'s wrappers make them)."""

    def __init__(self, layers, steps=STEPS) -> None:
        super().__init__()
        self.layers, self.steps = nn.ModuleList(layers), steps

    def forward(self, *xs):
        outs, k = [], 0
        for p0, T in self.steps:
            positions = torch.arange(p0, p0 + T, dtype=torch.int64)
            for layer in self.layers:
                outs += list(layer(positions, *xs[k:k + layer.n_inputs]))
                k += layer.n_inputs
        return tuple(outs)


def _model(order=(True, False), topk: int = TOPK, steps=STEPS) -> _Model:
    return _Model([_Layer(NAMES[i], indexer, seed=10 + i, topk=topk) for i, indexer in enumerate(order)], steps)


def _inputs(model: _Model, seed: int = 1) -> tuple:
    rng = np.random.default_rng(seed)
    xs = []
    for _, T in model.steps:
        for layer in model.layers:
            xs += [_bf16(rng, T, NQ), _bf16(rng, T, L), _bf16(rng, T, R), _bf16(rng, T, NH * (P + R))]
            if layer.indexer is not None:
                xs += [_bf16(rng, T, D), _bf16(rng, T, H), _bf16(rng, T, H * D)]
    return tuple(xs)


def _observation(**over) -> dict:
    return {"kind": "mla", "impl": IMPL, "backend": "FLASH_ATTN_MLA_SPARSE", "kv_cache_dtype": "auto", "layer_kv_cache_dtype": "auto",
            "num_heads": NH, "scale": SCALE, "kv_lora_rank": L, "qk_rope_head_dim": R, "dcp_world_size": 1, "use_pcp": False,
            "use_sparse": True, "prefill_backend": None, "num_splits": 1, "q_heads_per_rank": NH, "sparse_mla_force_mqa": False, **over}


@pytest.fixture
def served(monkeypatch):
    """The H200 serve as the Build observes it: the target, FA3's sparse MLA at one split, each layer's MLA implementation."""
    saved = dict(VB._OBSERVED)
    monkeypatch.setitem(PROFILES, STANDIN.name, STANDIN)
    monkeypatch.setitem(vllm_dsa.FA3_SPARSE_GEOMETRY, (R, L), {"BN": BN, "DOT": DOT, "INV": hopper.Fa3InvSum})
    VB.observe_target_profile(H200)
    VB.observe_attention_impls({n: _observation() for n in NAMES})
    yield
    VB._OBSERVED.clear()
    VB._OBSERVED.update(saved)


def _derive(model: _Model, inputs, profile: str = STANDIN.name):
    with dsa_leaves(SITE, model):
        return derive_report(model, inputs, mode="profile", profile=profile, name="dsa", rules=vllm_ruleset())


def _hint(rep) -> str:
    assert rep.program is None and rep.unsupported is not None
    return rep.unsupported.hint


def _ids(rep) -> list[str]:
    return [n.fn.id for n in rep.program.fn.body.nodes]


def _words(t: torch.Tensor) -> list[int]:
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().reshape(-1).astype(np.uint16).tolist()
    return t.contiguous().view(torch.int32).numpy().reshape(-1).astype(np.uint32).tolist()


def _run(rep, model: _Model, inputs) -> list[int]:
    held = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    weights = [v for w in rep.provenance["weights"] for v in _words(held[w["name"]])]
    t = materialize(rep.program, [_words(x) for x in inputs] + [weights])
    return [int(t[g]) for g in rep.program.circuit.root_outputs()]


def _reference(model: _Model, inputs) -> list[int]:
    """The module's outputs as the Definitions compose them: per token the norms, the RoPEs and the absorptions, and per row
    `SparseMla_v1{T=n}` over the layer's rows 0..n-1 ([kv_c normed | k_pe roped])."""
    qn, kn = bind(b1.RMSNormTriton, N=NQ, EPS=EPS), bind(b1.RMSNormTriton, N=L, EPS=EPS)
    rope1 = ropeh = bind(RoPEHeadGptJV2, D=R)
    up = bind(cublas_bf16.BatchedGemmWord, B=NH, K=P, N=L, DOT=DOT)
    down = bind(cublas_bf16.BatchedGemmWord, B=NH, K=L, N=V, DOT=DOT)
    hist = {n: [] for n in NAMES}
    out, k = [], 0
    for p0, T in model.steps:
        for layer in model.layers:
            q_c, kv_c, k_pe, q = inputs[k:k + 4]
            k += layer.n_inputs
            cs, qv = layer.rotary_emb.cos_sin_cache, q.view(T, NH, P + R)
            wuk, wuv = _words(layer.W_UK_T.permute(0, 2, 1)), _words(layer.W_UV.permute(0, 2, 1))
            rows, qcs = [], []
            for i in range(T):
                p = p0 + i
                qcs += list(evaluate(qn, _words(q_c[i]), _words(layer.q_a_layernorm.weight)))
                kv = list(evaluate(kn, _words(kv_c[i]), _words(layer.kv_a_layernorm.weight)))
                hist[layer.layer_name].append(kv + list(evaluate(rope1, _words(k_pe[i]), _words(cs[p]))))
                n = len(hist[layer.layer_name])
                ql = list(evaluate(up, _words(qv[i, :, :P]), wuk))
                qr = [w for head in qv[i, :, P:] for w in evaluate(ropeh, _words(head), _words(cs[p]))]
                fn = bind(sparse_mla.SparseMla, T=n, NH=NH, DR=R, DV=L, BN=BN, DOT=DOT, INV=hopper.Fa3InvSum, SCALE=SCALE)
                a = list(evaluate(fn, ql, qr, [w for r in hist[layer.layer_name] for w in r]))
                rows += list(evaluate(down, a, wuv))
            out += rows + qcs
    return [int(w) for w in out]


# -- the export shim ----------------------------------------------------------------------------------------------------------


def test_the_shim_exports_the_kernels_as_leaves_and_unwinds():
    model = _model()
    with dsa_leaves(SITE, model):
        assert SITE.fused_norm_rope is not _unpatched and "_sparse_indexer_and_attn" in vars(model.layers[0])
        ep = torch.export.export(model, _inputs(model), strict=False)
    assert SITE.fused_norm_rope is _unpatched and SITE.fused_q is _unpatched
    assert all("_sparse_indexer_and_attn" not in vars(m) for m in model.layers)
    ops = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
    per_step = len(STEPS) * len(NAMES)
    for op, n in (("vllm_iface.dsa_norm_rope.default", per_step), ("vllm_iface.dsa_q.default", per_step),
                  ("vllm_iface.dsa_indexer_attn.default", per_step), ("aten.bmm.default", 2 * per_step), ("aten.copy_.default", per_step)):
        assert ops.count(op) == n, (op, ops.count(op))


@pytest.mark.parametrize("attr,needle", [("_fp8_query", "fused_q with quantize_mqa"), ("use_pcp", "under context parallelism"),
                                         ("kv_cache_dtype", "MLA KV cache dtype 'fp8'")])
def test_what_the_shim_does_not_model_refuses_at_export(attr, needle, served):
    model = _model()
    setattr(model.layers[0], attr, "fp8" if attr == "kv_cache_dtype" else True)
    rep = _derive(model, _inputs(model))
    assert rep.unsupported.reason == "export" and needle in _hint(rep), _hint(rep)


def test_the_meta_instantiation_puts_an_explicit_cuda_buffer_on_meta():
    """DeepSeek-V3.2's `topk_indices_buffer` is `torch.empty(..., device=current_platform.device_type)` in `DeepseekV2Model.__init__`."""
    from verity_vllm.program.frontend.vllm_meta import _CudaFactoriesOnMeta
    with torch.device("meta"), _CudaFactoriesOnMeta():
        made = [torch.empty(4, TOPK, dtype=torch.int32, device="cuda"), torch.zeros(2, device=torch.device("cuda", 0)), torch.ones(3)]
    assert [t.device.type for t in made] == ["meta"] * 3
    assert torch.empty(1, device="cpu").device.type == "cpu"


def _cuda_only(hidden_states, router_weight, output_dtype=torch.float32):
    raise ValueError("hidden_states and router_weight must have device_type=cuda")


GATE_SITE = types.ModuleType("ll_bf16_site_standin")
GATE_SITE.ll_bf16_gemm = _cuda_only


class _Gate(nn.Module):
    """`GateLinear.forward`'s tier 1: `ll_bf16_gemm` read from its module at the call."""

    def __init__(self, rng, out_dtype=torch.float32, allow: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(_bf16(rng, 6, 8), requires_grad=False)
        self.out_dtype, self.allow_ll_bf16_gemm = out_dtype, allow

    def forward(self, x):
        return GATE_SITE.ll_bf16_gemm(x, self.weight, self.out_dtype)


class _CublasGate(nn.Module):
    """`GateLinear.forward`'s tier 4: default cuBLAS `torch.mm(x, W.T, out_dtype=float32)` above 16 tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(256, 6144, device="meta", dtype=torch.bfloat16), requires_grad=False)

    def forward(self, x):
        return torch.mm(x, self.weight.T, out_dtype=torch.float32)


@pytest.mark.parametrize("M", [1, 5, 16])
def test_the_gate_shim_exports_the_cutedsl_gemm_as_a_leaf_and_unwinds(M):
    from verity_vllm.program.frontend.export_compat import moe_gate_leaf
    rng = np.random.default_rng(3)
    gate = _Gate(rng)
    with moe_gate_leaf(gate, site=GATE_SITE):
        ep = torch.export.export(gate, (_bf16(rng, M, 8),), strict=False)
    assert GATE_SITE.ll_bf16_gemm is _cuda_only
    (leaf,) = [n for n in ep.graph.nodes if n.op == "call_function"]
    assert str(leaf.target) == "vllm_iface.ll_bf16_gemm.default"
    assert tuple(leaf.meta["val"].shape) == (M, 6) and leaf.meta["val"].dtype == torch.float32
    gate = _Gate(rng, torch.bfloat16)
    with moe_gate_leaf(gate, site=GATE_SITE), pytest.raises(NotImplementedError, match="ll_bf16_gemm with output dtype torch.bfloat16"):
        torch.export.export(gate, (_bf16(rng, M, 8),), strict=False)


def test_the_large_gate_binds_the_captured_default_cublas_splitk_route(served):
    model = _CublasGate()
    rep = _derive(model, (torch.empty(256, 6144, device="meta", dtype=torch.bfloat16),))
    assert rep.program is not None, rep.unsupported
    want = PROFILE_B1_EAGER_V3.kind("cublas_splitk_gemm_f32out", K=6144, N=256, SPLIT_K=3, DOT=DOT).id
    assert _ids(rep) == [want] * 256
    (app,) = [a for a in rep.provenance["applications"] if a["rule"] == "CublasRouter[vllm-default]_v1"]
    assert app["preconditions_checked"] == {
        "M": 256, "K": 6144, "N": 256, "kernel": "splitk", "SPLIT_K": 3, "DOT": DOT.name
    }
    assert rep.provenance["weight_roles"]["weight"] == {
        "rule": "CublasRouter[vllm-default]_v1", "role": "moe.gate.weight", "K": 6144, "N": 256
    }


def test_the_large_gate_refuses_an_uncaptured_default_cublas_route(served):
    rep = _derive(_CublasGate(), (torch.empty(18, 6144, device="meta", dtype=torch.bfloat16),))
    assert "no captured vLLM-default cuBLAS router route for M=18, K=6144, N=256" in _hint(rep)


def test_the_observed_split_count_is_the_pin_on_the_impls_module(monkeypatch):
    """`vllm_meta.attention_impls` reads an MLA layer's FA3 split count from the `flash_attn_varlen_func` its impl's module calls:
    the one-split pin's marker, else None (vLLM's heuristic)."""
    from verity_vllm.program.frontend import vllm_meta as vm
    attention = types.ModuleType("vllm.model_executor.layers.attention")
    attention.Attention, attention.MLAAttention = type("Attention", (nn.Module,), {}), type("MLAAttention", (nn.Module,), {})
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    backend = types.ModuleType("sparse_backend_standin")
    exec("class Impl:\n    dcp_world_size = 1\n", backend.__dict__)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    layer = attention.MLAAttention()
    layer.impl, layer.layer_name = backend.Impl(), NAMES[0]
    layer._vllm_config = types.SimpleNamespace(model_config=types.SimpleNamespace(get_num_attention_heads=lambda parallel: 64 // parallel.tp),
                                               parallel_config=types.SimpleNamespace(tp=8),
                                               attention_config=types.SimpleNamespace(sparse_mla_force_mqa=False))
    model = nn.Sequential(layer)
    backend.flash_attn_varlen_func = lambda *a, **k: None
    assert vm.attention_impls(model)[NAMES[0]]["num_splits"] is None
    backend.flash_attn_varlen_func.verity_sparse_mla_splits = 1
    obs = vm.attention_impls(model)[NAMES[0]]
    assert obs["num_splits"] == 1 and obs["q_heads_per_rank"] == 8 and obs["sparse_mla_force_mqa"] is False


def test_the_shims_leave_a_model_without_their_layers_unchanged(monkeypatch):
    """A gate that does not take tier 1 and a model without a sparse attention layer patch nothing (and import no vLLM site); the
    DSA kernels are patched in the module the layer's `forward` is defined in."""
    from verity_vllm.program.frontend.export_compat import dsa_ops_leaf, moe_gate_leaf
    rng = np.random.default_rng(3)
    with moe_gate_leaf(_Gate(rng, allow=False)), dsa_ops_leaf(_Gate(rng)):
        assert GATE_SITE.ll_bf16_gemm is _cuda_only
    site = types.ModuleType("dsa_forward_site")
    site.fused_norm_rope = site.fused_q = _unpatched
    site._Layer = _Layer
    exec("class Attn(_Layer):\n    def forward(self, *a):\n        return fused_norm_rope(*a)\n", site.__dict__)
    monkeypatch.setitem(sys.modules, site.__name__, site)
    layer = site.Attn(NAMES[0], indexer=True, seed=0)
    with dsa_ops_leaf(layer):
        assert site.fused_norm_rope is not _unpatched and site.fused_q is not _unpatched
    assert site.fused_norm_rope is _unpatched and site.fused_q is _unpatched


# -- the rules ----------------------------------------------------------------------------------------------------------------


def test_an_indexer_and_an_indexshare_layer_derive_over_prefill_and_decode(served):
    model = _model()
    xs = _inputs(model)
    rep = _derive(model, xs)
    assert rep.program is not None, rep.unsupported
    ids = _ids(rep)
    norm_q = bind(b1.RMSNormTriton, N=NQ, EPS=EPS).id
    norm_kv = bind(b1.RMSNormTriton, N=L, EPS=EPS).id
    kr = qr = bind(RoPEHeadGptJV2, D=R).id
    up = bind(cublas_bf16.BatchedGemmWord, B=NH, K=P, N=L, DOT=DOT).id
    down = bind(cublas_bf16.BatchedGemmWord, B=NH, K=L, N=V, DOT=DOT).id
    krow, iq = bind(dsa.DsaIndexerKRow, D=D, ROPE=ROPE, EPS=EPS).id, bind(dsa.DsaIndexerQ, H=H, D=D, ROPE=ROPE).id
    sel = bind(dsa.DsaTopKShortRow, TOPK=TOPK).id

    def attn(n):
        return bind(sparse_mla.SparseMla, T=n, NH=NH, DR=R, DV=L, BN=BN, DOT=DOT, INV=hopper.Fa3InvSum, SCALE=SCALE).id

    want = []
    for p0, T in STEPS:
        pos = range(p0, p0 + T)
        for indexer in (True, False):
            want += [norm_q] * T + [norm_kv] * T + [kr] * T + ([krow] * T if indexer else []) + [up] * T + [qr] * (T * NH)
            want += [iq] * T if indexer else []
            for p in pos if indexer else ():
                want += [bind(dsa.DsaMqaLogitsV3, N=p + 1, H=H, D=D).id, f"Const32[0x{p + 1:08x}]_v1", sel]
            want += [attn(p + 1) for p in pos] + [down] * T
    assert ids == want
    assert _run(rep, model, xs) == _reference(model, xs)
    assert rep.dead.total > 0   # the indexer's logits and top-k: with n <= TOPK the selection is rows 0..n-1 whatever they are
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "DsaIndexerAttn[vllm-dsa]_v1"]
    assert [(a["preconditions_checked"]["layer_name"], a["preconditions_checked"]["top_k_from"]) for a in apps] == [
        (NAMES[0], NAMES[0]), (NAMES[1], NAMES[0])] * len(STEPS)
    roles = rep.provenance["weight_roles"]
    assert roles["layers.0.W_UK_T"]["role"] == "mla.absorb" and roles["layers.0.indexer.k_norm.bias"]["role"] == "dsa.indexer.k_norm.bias"


class _LargeMlaAbsorb(nn.Module):
    """The real q-nope spelling at enough leaves that its first transpose has no compact generic view."""

    def __init__(self, escape_bmm: bool = False) -> None:
        super().__init__()
        rng = np.random.default_rng(32)
        self.weight = nn.Parameter(_bf16(rng, 16, 32, 8), requires_grad=False)
        self.escape_bmm = escape_bmm

    def forward(self, q):
        x, _q_pe = q.split([32, 1], dim=-1)
        y = torch.bmm(x.transpose(0, 1), self.weight)
        z = y.transpose(0, 1)
        return (z, y) if self.escape_bmm else z


def _large_absorb(escape_bmm: bool = False):
    model = _LargeMlaAbsorb(escape_bmm)
    q = torch.empty(256, 16, 33, dtype=torch.bfloat16)
    return derive_report(model, (q,), mode="profile", profile=STANDIN.name, name="large-mla-absorb", rules=vllm_ruleset())


def test_large_mla_absorb_is_bound_before_its_composite_transpose(served):
    rep = _large_absorb()
    assert rep.program is not None, rep.unsupported
    (app,) = [a for a in rep.provenance["applications"] if a["rule"] == "MlaAbsorbTranspose[vllm-dsa]_v1"]
    assert app["preconditions_checked"]["chain"] == ["transpose", "bmm", "transpose_1"]
    assert app["preconditions_checked"]["M"] == 256
    assert app["emitted"].startswith("BatchedGemm_v1") and app["emitted"].endswith("x256")
    assert rep.provenance["weight_roles"]["weight"]["role"] == "mla.absorb"


def test_large_mla_absorb_refuses_when_the_bmm_intermediate_escapes(served):
    rep = _large_absorb(escape_bmm=True)
    assert rep.program is None and rep.unsupported.reason == "unsupported_view"
    assert "permute (1, 0, 2) of 131072 leaves over a composite reference sequence has no O(1) form" in rep.unsupported.hint


def test_default_sparse_mla_uses_each_tokens_launched_split_count(served):
    VB.observe_attention_impls({n: _observation(num_splits=None) for n in NAMES})
    model = _model()
    rep = _derive(model, _inputs(model))
    assert rep.program is not None, rep.unsupported
    got = [spec for spec in _ids(rep) if spec.startswith("SparseMlaSplit_v1")]
    keys = [1, 2, 1, 2, 3, 3]
    splits = launch_num_splits(keys, H200.num_sms, TOPK, 0)
    want = [
        bind(sparse_mla_split.SparseMlaSplit, T=n, NS=ns, NH=NH, DR=R, DV=L, BN=BN,
             DOT=DOT, INV=hopper.Fa3InvSum, SCALE=SCALE).id
        for n, ns in zip(keys, splits)
    ]
    assert got == want


def test_a_history_opens_at_the_first_position_a_layer_sees(served):
    """The step wrapper's Program starts at `pos0` (`EngineStep`), and a layer's history opens there, as the dense attention rule's
    does: row i of a call reads P0 + i + 1 keys, while RoPE reads the position."""
    model = _model(steps=((5, 2), (7, 1)))
    xs = _inputs(model)
    rep = _derive(model, xs)
    assert rep.program is not None, rep.unsupported
    ids = _ids(rep)
    logits = [i for i in ids if i.startswith("DsaMqaLogits")]
    assert logits == [bind(dsa.DsaMqaLogitsV3, N=n, H=H, D=D).id for n in (1, 2, 3)]
    assert _run(rep, model, xs) == _reference(model, xs)


def test_a_call_that_does_not_continue_the_history_refuses(served):
    model = _model(steps=((0, 2), (3, 1)))
    hint = _hint(_derive(model, _inputs(model)))
    assert f"positions 3..3 do not continue {NAMES[0]!r}'s history of 2 rows (next position 2)" in hint, hint


def test_the_served_vocabulary_binds_the_norm_and_gptj_rope_roles():
    assert PROFILE_B1_EAGER_V3.kind("dsa_rmsnorm", N=NQ, EPS=EPS).id == bind(dsa_norm.DsaRMSNorm, N=NQ, EPS=EPS).id
    assert PROFILE_B1_EAGER_V3.kind("dsa_kv_rope", D=R).id == bind(RoPEHeadGptJV2, D=R).id
    assert PROFILE_B1_EAGER_V3.kind("dsa_q_rope", D=R).id == bind(RoPEHeadGptJV2, D=R).id


@pytest.mark.parametrize("case,needle", [
    ("topk", "'layers.0.self_attn.attn': position 2 has 3 keys > TOPK 2"),
    ("shared-first", "IndexShare layer 'layers.0.self_attn.attn' reads a top-k no indexer layer selected for positions 0..1"),
    ("prefill-backend", "a call of 2 query rows is a prefill (decode threshold 1 at 2 query heads per rank), which its dense-MHA "
                        "prefill backend 'FlashAttnMLAPrefill' serves"),
    ("heads", "no observed query heads per rank for 'layers.0.self_attn.attn', the decode threshold its prefill backend"),
    ("impl", "runs 'FlashMLASparseImpl'; SparseMla_v1 states FLASH_ATTN_MLA_SPARSE"),
    ("target", "SparseMla_v1 is FA3 at compute capability 9.0; the target is"),
])
def test_each_attention_precondition_refuses_by_name(case, needle, served, monkeypatch):
    model = _model(order=(False, True) if case == "shared-first" else (True, False), topk=2 if case == "topk" else TOPK)
    monkeypatch.setitem(vllm_dsa.DECODE_THRESHOLD, NH, 1)
    over = {"prefill-backend": {"prefill_backend": "FlashAttnMLAPrefill"},
            "heads": {"prefill_backend": "FlashAttnMLAPrefill", "q_heads_per_rank": None}, "impl": {"impl": "FlashMLASparseImpl"}}.get(case)
    if over:
        VB.observe_attention_impls({n: _observation(**over) for n in NAMES})
    if case == "target":
        VB.observe_target_profile(TargetProfile(compute_capability=(8, 0), num_sms=108, batch_invariant=False))
    hint = _hint(_derive(model, _inputs(model)))
    assert needle in hint and (case in ("topk", "shared-first") or "DsaIndexerAttn[vllm-dsa]_v1" in hint), hint


@pytest.mark.parametrize("threshold,force_mqa", [(None, False), (1, True)])
def test_a_layer_with_a_prefill_backend_derives_where_every_call_takes_the_mqa_path(threshold, force_mqa, served, monkeypatch):
    """The H200 serve: GLM-5.3's layers have FA3's dense-MHA prefill backend, and its 256-token prompt is at most the decode threshold
    (256 at 8 query heads per rank), so the builder counts it a decode and the layer serves it by sparse MQA, as a step; and under
    `sparse_mla_force_mqa` a longer call does too.  The Program is the one a layer without a prefill backend derives."""
    model = _model()
    xs = _inputs(model)
    plain = _ids(_derive(model, xs))
    if threshold is not None:
        monkeypatch.setitem(vllm_dsa.DECODE_THRESHOLD, NH, threshold)
    VB.observe_attention_impls({n: _observation(prefill_backend="FlashAttnMLAPrefill", sparse_mla_force_mqa=force_mqa) for n in NAMES})
    rep = _derive(model, xs)
    assert rep.program is not None, rep.unsupported
    assert _ids(rep) == plain and _run(rep, model, xs) == _reference(model, xs)
    assert vllm_dsa.DECODE_THRESHOLD_DEFAULT == 256 and vllm_dsa.DECODE_THRESHOLD.get(8, vllm_dsa.DECODE_THRESHOLD_DEFAULT) == 256
