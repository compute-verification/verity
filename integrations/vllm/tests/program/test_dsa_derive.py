"""The Build's DeepSeek sparse attention (GLM-5.3 on the H200), on a synthetic layer that makes `DeepseekV32Attention.forward`'s calls:

* `export_compat.dsa_leaves` exports `fused_norm_rope`, `fused_q` and `_sparse_indexer_and_attn` as the `vllm_iface` leaves (the
  meta export has no KV cache, so the kernels' slot-mapping asserts never run) and unwinds; what it does not model refuses at export;
* `rules/vllm_dsa.py` derives an indexer layer and an IndexShare layer over a prefill and a decode step: per token the norms and
  RoPEs, `DsaIndexerKRow_v1` and `DsaIndexerQ_v1`, per row `DsaMqaLogits_v3{N}` and `DsaTopKShortRow_v1`, and `SparseMla_v1{T=n}` over
  the layer's first n history rows, equal to those Definitions composed by hand;
* each precondition refuses by name.

The roles with no Definition on this branch (`dsa_rmsnorm`, `dsa_kv_rope`, `dsa_q_rope`, `batched_gemm`) are stand-ins here
(`RMSNormTriton_v1`, `RoPE_v1`, a batch of `Gemm_v2`), and FA3's sparse geometry is registered at rope 16, latent 16."""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.types import Array
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.export_compat import dsa_leaves
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules import vllm_dsa
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3, PROFILES, Vocabulary
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels.serve3_reference import materialize
from verity_vllm.program.registry import b1, dsa, hopper, sparse_mla
from verity_vllm.program.registry.prims import BF16

H200 = TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False, moe_backend="triton", fp8_block_gemm="cutlass")
NH, P, R, L, V, NQ = 2, 16, 16, 16, 16, 16   # heads, nope, rope, latent, value, q_lora
H, D, ROPE = 8, 32, 16                       # the indexer's heads, head dim, rope
TOPK, MAXPOS, EPS, BN = 4, 8, 1e-6, 16
SCALE = (P + R) ** -0.5
STEPS = ((0, 2), (2, 1))                     # (first position, tokens): a prefill of two, one decode step
NAMES = ("layers.0.self_attn.attn", "layers.1.self_attn.attn")
DOT = hopper.HopperBF16WgmmaDot16
IMPL = "vllm.v1.attention.backends.mla.flashattn_mla_sparse.FlashAttnMLASparseImpl"


@composite("BatchedGemmStandIn", 1, ["B", "K", "N", "DOT"],
           lambda S: ((("x", Array(S.B, Array(S.K, BF16))), ("w", Array(S.B, Array(S.N, Array(S.K, BF16))))), Array(S.B, Array(S.N, BF16))))
def BatchedGemmStandIn(B, S, x, w):
    return B.batch(bind(b1.GemmV2, K=S.K, N=S.N, DOT=S.DOT), x, w, axes=(0, 0))


STANDIN = Vocabulary("dsa-standin", "profile", "dsa-standin",
                     kinds={**PROFILE_B1_EAGER_V3.kinds, "dsa_rmsnorm": PROFILE_B1_EAGER_V3.kinds["rmsnorm"],
                            "dsa_kv_rope": PROFILE_B1_EAGER_V3.kinds["rope"], "dsa_q_rope": PROFILE_B1_EAGER_V3.kinds["rope"],
                            "batched_gemm": lambda B, K, N, DOT: bind(BatchedGemmStandIn, B=B, K=K, N=N, DOT=DOT)},
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

    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *xs):
        outs, k = [], 0
        for p0, T in STEPS:
            positions = torch.arange(p0, p0 + T, dtype=torch.int64)
            for layer in self.layers:
                outs += list(layer(positions, *xs[k:k + layer.n_inputs]))
                k += layer.n_inputs
        return tuple(outs)


def _model(order=(True, False), topk: int = TOPK) -> _Model:
    return _Model([_Layer(NAMES[i], indexer, seed=10 + i, topk=topk) for i, indexer in enumerate(order)])


def _inputs(model: _Model, seed: int = 1) -> tuple:
    rng = np.random.default_rng(seed)
    xs = []
    for _, T in STEPS:
        for layer in model.layers:
            xs += [_bf16(rng, T, NQ), _bf16(rng, T, L), _bf16(rng, T, R), _bf16(rng, T, NH * (P + R))]
            if layer.indexer is not None:
                xs += [_bf16(rng, T, D), _bf16(rng, T, H), _bf16(rng, T, H * D)]
    return tuple(xs)


def _observation(**over) -> dict:
    return {"kind": "mla", "impl": IMPL, "backend": "FLASH_ATTN_MLA_SPARSE", "kv_cache_dtype": "auto", "layer_kv_cache_dtype": "auto",
            "num_heads": NH, "scale": SCALE, "kv_lora_rank": L, "qk_rope_head_dim": R, "dcp_world_size": 1, "use_pcp": False,
            "use_sparse": True, "prefill_backend": None, **over}


@pytest.fixture
def served(monkeypatch):
    """The H200 serve as the Build observes it: the target, FA3's sparse MLA at one split, each layer's MLA implementation."""
    saved = dict(VB._OBSERVED)
    monkeypatch.setitem(PROFILES, STANDIN.name, STANDIN)
    monkeypatch.setitem(vllm_dsa.FA3_SPARSE_GEOMETRY, (R, L), {"BN": BN, "DOT": DOT, "INV": hopper.Fa3InvSum})
    monkeypatch.setenv("VERITY_SPARSE_MLA_NUM_SPLITS", "1")
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
    rope1, ropeh = bind(b1.RoPE, NHEADS=1, D=R), bind(b1.RoPE, NHEADS=NH, D=R)
    up, down = bind(BatchedGemmStandIn, B=NH, K=P, N=L, DOT=DOT), bind(BatchedGemmStandIn, B=NH, K=L, N=V, DOT=DOT)
    hist = {n: [] for n in NAMES}
    out, k = [], 0
    for p0, T in STEPS:
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
                ql = list(evaluate(up, _words(qv[i, :, :P]), wuk))
                qr = list(evaluate(ropeh, _words(qv[i, :, P:]), _words(cs[p])))
                fn = bind(sparse_mla.SparseMla, T=p + 1, NH=NH, DR=R, DV=L, BN=BN, DOT=DOT, INV=hopper.Fa3InvSum, SCALE=SCALE)
                a = list(evaluate(fn, ql, qr, [w for r in hist[layer.layer_name][:p + 1] for w in r]))
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


# -- the rules ----------------------------------------------------------------------------------------------------------------


def test_an_indexer_and_an_indexshare_layer_derive_over_prefill_and_decode(served):
    model = _model()
    xs = _inputs(model)
    rep = _derive(model, xs)
    assert rep.program is not None, rep.unsupported
    ids = _ids(rep)
    norm_q, norm_kv = f"RMSNormTriton_v1{{N={NQ},EPS={EPS}}}", f"RMSNormTriton_v1{{N={L},EPS={EPS}}}"
    kr, qr = bind(b1.RoPE, NHEADS=1, D=R).id, bind(b1.RoPE, NHEADS=NH, D=R).id
    up = bind(BatchedGemmStandIn, B=NH, K=P, N=L, DOT=DOT).id
    down = bind(BatchedGemmStandIn, B=NH, K=L, N=V, DOT=DOT).id
    krow, iq = bind(dsa.DsaIndexerKRow, D=D, ROPE=ROPE, EPS=EPS).id, bind(dsa.DsaIndexerQ, H=H, D=D, ROPE=ROPE).id
    sel = bind(dsa.DsaTopKShortRow, TOPK=TOPK).id

    def attn(n):
        return bind(sparse_mla.SparseMla, T=n, NH=NH, DR=R, DV=L, BN=BN, DOT=DOT, INV=hopper.Fa3InvSum, SCALE=SCALE).id

    want = []
    for p0, T in STEPS:
        pos = range(p0, p0 + T)
        for indexer in (True, False):
            want += [norm_q] * T + [norm_kv] * T + [kr] * T + ([krow] * T if indexer else []) + [up] * T + [qr] * T
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


def test_the_served_vocabulary_names_the_roles_it_lacks(served):
    model = _model()
    hint = _hint(_derive(model, _inputs(model), profile="b1-eager-v3"))
    assert "DsaNormRope[vllm-dsa]_v1" in hint and "the vocabulary has no dsa_rmsnorm" in hint and "no dsa_kv_rope" in hint, hint


@pytest.mark.parametrize("case,needle", [
    ("splits", "the split-combine has no Definition; SparseMla_v1 is num_splits=1"),
    ("topk", "'layers.0.self_attn.attn': position 2 has 3 keys > TOPK 2"),
    ("shared-first", "IndexShare layer 'layers.0.self_attn.attn' reads a top-k no indexer layer selected for positions 0..1"),
    ("prefill-backend", "has the dense-MHA prefill backend 'FlashAttnMLAPrefill'"),
    ("impl", "runs 'FlashMLASparseImpl'; SparseMla_v1 states FLASH_ATTN_MLA_SPARSE"),
    ("target", "SparseMla_v1 is FA3 on sm_90; the target is"),
])
def test_each_attention_precondition_refuses_by_name(case, needle, served, monkeypatch):
    model = _model(order=(False, True) if case == "shared-first" else (True, False), topk=2 if case == "topk" else TOPK)
    if case == "splits":
        monkeypatch.delenv("VERITY_SPARSE_MLA_NUM_SPLITS")
    if case in ("prefill-backend", "impl"):
        over = {"prefill_backend": "FlashAttnMLAPrefill"} if case == "prefill-backend" else {"impl": "FlashMLASparseImpl"}
        VB.observe_attention_impls({n: _observation(**over) for n in NAMES})
    if case == "target":
        VB.observe_target_profile(TargetProfile(compute_capability=(8, 0), num_sms=108, batch_invariant=False))
    hint = _hint(_derive(model, _inputs(model)))
    assert needle in hint and (case in ("topk", "shared-first") or "DsaIndexerAttn[vllm-dsa]_v1" in hint), hint
