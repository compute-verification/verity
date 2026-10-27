"""The Build of DeepSeek-V4-Pro on the H200, on synthetic modules that make the calls vLLM's DeepSeek-V4 makes:

* `export_compat.cuda_streams_inert`: the aux streams and events DeepSeek-V4 makes in `__init__` (no driver on a GPU-free Build
  host) are inert, and a forward that overlaps on them exports its functions in the order the engine launches them;
* `loader_parity.post_load_tensors_as_the_loader_does`: a tensor the model-level post-load hook makes (`hc_attn_fn_broadcast`) is a
  Parameter under its served name;
* `export_compat.registered_op_bodies_as_ops`: the decoder layer calls the Python bodies of vLLM's mHC ops, which export as the ops;
* `export_compat.mhc_pre_broadcast_leaf`: layer 0's broadcast pre, which vLLM does not register, exports as the leaf op;
* `dsv4_leaves.dsv4_attention_leaves`: the sparse MLA's cache insert, compressor state and attention export as the leaf ops;
* `export_compat.deep_gemm_fp8_einsum_leaf`: the o-projection's DeepGEMM einsum exports as the leaf op;
* `marlin_moe_leaf.mxfp4_marlin_moe_leaf`: the MXFP4 Marlin MoE exports as its steps, its two expert GEMMs as the leaf op;
* `rules/vllm_mhc.py`: the four mHC ops derive per token at the dispatch's own split counts, word for word the numpy twin's
  (`kernels.mhc_twin`) on the FMA path and the TF32 path, under a declared launch batch at its token count, and what the
  Definitions do not take refuses by name;
* `CublasRouterRule`: a direct `torch.mm(x, W.T, out_dtype=float32)` binds the captured f32-output route;
* `FusedQKvRmsNormRule`: the launch of `_fused_q_kv_rmsnorm_kernel` derives per token word for word its PTX's (`test_dsa_norm`), and
  another kernel body, a block of one element a thread or a stride outside the PTX's specialization refuses;
* `rules/vllm_dsv4.py`: the insert and the compressor state derive per token word for word the numpy twin's
  (`kernels.dsv4_qnorm_insert_twin`, FlashMLA's `fadd`), and statics the Definitions do not take or positions the tables do not
  hold refuse by name; the sparse attention over a request's prefill and decodes, at ratios 4, 128 and 1, derives word for word
  the twins' composition over its histories (the compressors, FlashMLA's gather and prefill, its decode at the planner's splits,
  under a declared launch batch among the launch's decodes), and a geometry, head padding, selection, history, target or launch
  the Definitions do not take refuses by name; the o-projection
  input's inverse RoPE and quantisation derives per token and head word for word the reference (`test_dsv4_indexer.head`) in
  the wrapper's [T, G, ...] layout, and another scale layout, clamp, rotary split or target refuses by name;
* `DeepGemmFp8EinsumRule`: the o-projection's `fp8_einsum` over that quant's words and scales derives per token word for word the
  grouped DeepGEMM twin (`kernels.deepgemm_twins`), and another expression or recipe refuses by name;
* `SiluAndMulWithClampRule`: the shared expert's clamped SwiGLU derives per row word for word the twin the capture holds to the
  served kernel (`dsv4_moe_twin.clamp_twin`), and SwiGLU-OAI's alpha and beta or a limit that is no bf16 value refuses by name;
* `TopkSoftplusSqrtHashRule`: a hash layer's router derives per token word for word the capture's twin over the table's ids
  (`dsv4_moe_twin.hash_twin`), and the kernel's scored path or another normalisation refuses by name;
* `Dsv4TopkRule`: a scored layer's `_dsv4_topk_kernel` launch derives per token word for word the capture's twin
  (`dsv4_moe_twin.router_twin`), and another kernel body or specialization refuses by name;
* `rules/vllm_marlin_moe.py`: the Marlin MoE's align derives as `moe_align_block_size` with each expert's pairs in key order, and
  its two expert GEMMs per pair word for word the twin (`kernels.marlin_moe_rows.launch`) at the config the capture gives, and
  a batch, block or shape the capture does not give, or a launch that also routes other requests' tokens, refuses by name;
* `loader_parity.fp8_block_weights_as_the_loader_does`: UE8M0 block scales are f32 and an `is_bmm` layer is grouped, as the loader
  leaves them."""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import torch
import torch.fx.traceback as fxt
import torch.nn as nn
from torch._library.infer_schema import infer_schema

from verity_vllm.program.frontend import marlin_moe_leaf as mml
from verity_vllm.program.frontend import targets as TG
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.dsv4_leaves import dsv4_attention_leaves
from verity_vllm.program.frontend.export_compat import (InertEvent, InertStream, cuda_streams_inert, deep_gemm_fp8_einsum_leaf,
                                                        mhc_pre_broadcast_leaf, registered_op_bodies_as_ops)
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules import vllm_dsv4, vllm_marlin_moe, vllm_mhc
from verity_vllm.program.frontend.rules.vllm_bindings import observations as OB
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
from verity_vllm.program.frontend.rules.vllm_bindings.pins import DSV4_TOPK_PINS
from verity_vllm.program.frontend.rules.triton_iface import triton_launch
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.frontend.export_ops import CollectiveBus, TPStub
from verity_vllm.program.frontend.loader_parity import fp8_block_weights_as_the_loader_does, post_load_tensors_as_the_loader_does
from verity_vllm.program.frontend.vllm_meta import ScheduledWithPeers, with_peers
from verity.ml.definitions.numerics import deepgemm_twins as DT
from verity.ml.definitions.numerics import dsv4_compressor_twin as DC
from verity.ml.definitions.numerics import dsv4_qnorm_insert_twin as TQ
from verity.ml.definitions.numerics import flashmla_prefill_twin as FP
from verity.ml.definitions.numerics import flashmla_twin as FT
from verity.ml.definitions.numerics import marlin_moe_rows as MR
from verity.ml.definitions.numerics import mhc_twin as W
from verity.ml.definitions.numerics.dsv4_moe_twin import clamp_twin, hash_twin, router_twin
from verity.ml.definitions.numerics.serve3_reference import materialize
from verity.ml.definitions.registry import bf16_gemm_check as BC
from verity.ml.definitions.registry import hopper
from tests.program.test_dsa_norm import ptx_block_model
from tests.program.test_dsv4_indexer import head as inv_rope_head


class _Overlapped(nn.Module):
    """`multi_stream_utils.execute_in_parallel` with one aux stream: the aux function is launched first, on its stream."""

    def __init__(self) -> None:
        super().__init__()
        self.stream, self.start, self.done = torch.cuda.Stream(), torch.cuda.Event(), torch.cuda.Event()

    def forward(self, x):
        self.start.record()
        with torch.cuda.stream(self.stream):
            self.start.wait()
            aux = x + 1
            self.done.record()
        default = x * 2
        self.done.wait()
        return default, aux


def test_streams_and_events_are_inert_for_construction_and_export_and_unwind():
    stream, event, select = torch.cuda.Stream, torch.cuda.Event, torch.cuda.stream
    with cuda_streams_inert():
        m = _Overlapped()
        assert isinstance(m.stream, InertStream) and isinstance(m.start, InertEvent)
        ep = torch.export.export(m, (torch.empty(4, device="meta"),), strict=False)
    assert (torch.cuda.Stream, torch.cuda.Event, torch.cuda.stream) == (stream, event, select)
    assert [str(n.target) for n in ep.graph.nodes if n.op == "call_function"] == ["aten.add.Tensor", "aten.mul.Tensor"]


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fn = nn.Parameter(torch.empty(24, 4 * 8, device="meta"), requires_grad=False)
        self.fn_broadcast = None


class _Model(nn.Module):
    """`DeepseekV4ForCausalLM.process_weights_after_loading` -> `finalize_mhc_broadcast_weights`: the first layer's `fn` summed over
    its 4 streams, kept as a plain attribute; a buffer of the same kind made in `__init__` stays as it is."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])
        self.topk_buffer = torch.empty(2, 3, dtype=torch.int32, device="meta")

    def process_weights_after_loading(self) -> None:
        layer = self.layers[0]
        layer.fn_broadcast = layer.fn.detach().view(-1, 4, 8).sum(dim=1)


def test_the_post_load_hook_makes_its_tensors_parameters_under_their_served_names():
    model = _Model()
    buffer = model.topk_buffer
    fact = post_load_tensors_as_the_loader_does(model)
    assert isinstance(model.layers[0].fn_broadcast, nn.Parameter) and model.layers[1].fn_broadcast is None
    assert "layers.0.fn_broadcast" in dict(model.named_parameters()) and model.topk_buffer is buffer
    assert fact["parameters"] == {"layers.0.fn_broadcast": {"shape": [24, 8], "dtype": "float32"}}
    assert post_load_tensors_as_the_loader_does(nn.Linear(2, 2)) == {"applies": False}


def mixed(x: torch.Tensor, scale: float, n_splits: int = 1) -> torch.Tensor:
    raise RuntimeError("launches a kernel")


def accumulate(out: torch.Tensor, x: torch.Tensor) -> None:
    raise RuntimeError("launches a kernel")


def unregistered(x: torch.Tensor) -> torch.Tensor:
    return x + 1


_LIB = torch.library.Library("verity_dsv4_test", "FRAGMENT")
for _f, _written in ((mixed, ()), (accumulate, ("out",))):
    # `direct_register_custom_op`'s registration
    _LIB.define(_f.__name__ + infer_schema(_f, mutates_args=_written))
    _LIB.impl(_f.__name__, _f, "CompositeExplicitAutograd")
_LIB._register_fake("mixed", lambda x, scale, n_splits=1: torch.empty_like(x))
_LIB._register_fake("accumulate", lambda out, x: None)


class _Decoder(nn.Module):
    def forward(self, x):
        out = torch.zeros_like(x)
        accumulate(out, mixed(x, 0.5, n_splits=1))
        return unregistered(out)


def test_a_registered_ops_python_body_called_by_the_model_exports_as_the_op():
    model, ns, module = _Decoder(), torch.ops.verity_dsv4_test, sys.modules[__name__]
    with registered_op_bodies_as_ops(model, ns=ns):
        assert (module.mixed, module.accumulate, module.unregistered) == (ns.mixed, ns.accumulate, unregistered)
        ep = torch.export.export(model, (torch.empty(4, device="meta"),), strict=False)
    assert (module.mixed, module.accumulate) == (mixed, accumulate)
    targets = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
    assert "verity_dsv4_test.mixed.default" in targets and "aten.add.Tensor" in targets
    assert any("accumulate" in t for t in targets)


def mhc_pre_broadcast(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                      n_splits=1, norm_weight=None, norm_eps=1e-6, fn_broadcast=None):
    raise RuntimeError("sizes DeepGEMM's split-K from torch.cuda.get_device_properties")


class _FirstLayer(nn.Module):
    """`DeepseekV4DecoderLayer.forward`'s first-layer branch, HC = 4 streams of H = 8."""

    def __init__(self) -> None:
        super().__init__()
        self.fn = nn.Parameter(torch.empty(24, 4 * 8, device="meta"), requires_grad=False)
        self.fn_broadcast = nn.Parameter(torch.empty(24, 8, device="meta"), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(3, device="meta"), requires_grad=False)
        self.base = nn.Parameter(torch.empty(24, device="meta"), requires_grad=False)
        self.norm = nn.Parameter(torch.empty(8, dtype=torch.bfloat16, device="meta"), requires_grad=False)

    def forward(self, x):
        return mhc_pre_broadcast(x, self.fn, self.scale, self.base, 1e-6, 1e-6, 1e-6, 2.0, 20, norm_weight=self.norm, norm_eps=1e-6,
                                 fn_broadcast=self.fn_broadcast)


def test_layer_zeros_broadcast_pre_exports_as_the_leaf_op_of_its_inferred_schema():
    model = _FirstLayer()
    with mhc_pre_broadcast_leaf(model, original=mhc_pre_broadcast):
        ep = torch.export.export(model, (torch.empty(3, 8, dtype=torch.bfloat16, device="meta"),), strict=False)
    assert sys.modules[__name__].mhc_pre_broadcast is mhc_pre_broadcast
    (node,) = [n for n in ep.graph.nodes if n.op == "call_function" and "mhc" in str(n.target)]
    assert str(node.target) == "vllm_iface.mhc_pre_broadcast_tilelang.default"
    outs = [tuple(v.meta["val"].shape) for v in ep.graph.nodes if v.op == "call_function" and v.args and v.args[0] is node]
    assert outs == [(3, 4, 8), (3, 4, 1), (3, 4, 4), (3, 8)]


def _launches(*args, **kwargs):
    raise RuntimeError("launches a kernel")


def _meta(*shape, dtype=torch.bfloat16):
    return nn.Parameter(torch.empty(*shape, dtype=dtype, device="meta"), requires_grad=False)


def _call_targets(ep) -> list[str]:
    return [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cos_sin_cache = _meta(32, 8, dtype=torch.float32)


class _Compressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ape, self.norm, self.rms_norm_eps = _meta(4, 32, dtype=torch.float32), nn.Module(), 1e-6
        self.norm.weight = _meta(16)

    forward = _launches


class _Indexer(nn.Module):
    def __init__(self, max_model_len: int) -> None:
        super().__init__()
        self.max_model_len, self.topk_tokens, self.prefix = max_model_len, 1024, "layers.0.attn.indexer"

    forward = _launches


class _SparseMla(nn.Module):
    """`DeepseekV4MLAAttention._prepare_and_attn` for a ratio-4 layer of 2 live heads padded to 8, head 16 of which 8 rotary."""

    def __init__(self, kv_cache_dtype: str = "fp8_ds_mla", indexer_rows: int = 72) -> None:
        super().__init__()
        self.prefix, self.kv_cache_dtype, self.swa_cache_layer = "layers.0.attn", kv_cache_dtype, nn.Module()
        self.n_local_heads, self.padded_heads, self.compress_ratio, self.window_size = 2, 8, 4, 128
        self.eps, self.scale = 1e-6, 0.25
        self.rotary_emb, self.compressor, self.indexer = _Rotary(), _Compressor(), _Indexer(indexer_rows)
        self.attn_sink = _meta(2, dtype=torch.float32)

    _fused_qnorm_rope_kv_insert = forward_mqa = _launches

    def forward(self, q, kv, kv_score, positions):
        out = q.new_zeros((q.shape[0], self.padded_heads, q.shape[-1]))
        self.indexer(q, positions)
        self.compressor(kv_score, positions, self.rotary_emb)
        self.forward_mqa(self._fused_qnorm_rope_kv_insert(q, kv, positions, None), kv, positions, out)
        return out


def _sparse_mla_inputs(T: int = 3) -> tuple:
    return (torch.empty(T, 2, 16, dtype=torch.bfloat16, device="meta"), torch.empty(T, 16, dtype=torch.bfloat16, device="meta"),
            torch.empty(T, 64, dtype=torch.float32, device="meta"), torch.empty(T, dtype=torch.int64, device="meta"))


def test_the_sparse_mla_exports_as_its_insert_compressor_state_and_attention_leaves():
    model = _SparseMla()
    with dsv4_attention_leaves(model):
        ep = torch.export.export(model, _sparse_mla_inputs(), strict=False)
    assert {m.__func__ for m in (model._fused_qnorm_rope_kv_insert, model.forward_mqa, model.compressor.forward,
                                 model.indexer.forward)} == {_launches}
    targets = _call_targets(ep)
    leaves = [t for t in targets if t.startswith("vllm_iface.")]
    assert leaves == ["vllm_iface.dsv4_compressor_state.default", "vllm_iface.dsv4_qnorm_rope_kv_insert.default",
                      "vllm_iface.dsv4_sparse_attn.default"]
    (insert,) = [n for n in ep.graph.nodes if str(n.target) == leaves[1]]
    rows = [(tuple(v.meta["val"].shape), v.meta["val"].dtype) for v in ep.graph.nodes if v.args and v.args[0] is insert]
    assert rows == [((3, 8, 16), torch.bfloat16), ((3, 8), torch.float8_e4m3fn), ((3, 1), torch.uint8), ((3, 8), torch.bfloat16)]
    (attn,) = [n for n in ep.graph.nodes if str(n.target) == leaves[2]]
    assert tuple(attn.meta["val"].shape) == (3, 2, 16) and attn.args[10:15] == ("layers.0.attn", 2, 4, 128, 1024)
    assert "aten.copy_.default" in targets


@pytest.mark.parametrize("model, refusal", [(_SparseMla(kv_cache_dtype="auto"), "only fp8_ds_mla"),
                                            (_SparseMla(indexer_rows=4096), "long path")])
def test_a_sparse_mla_the_leaves_do_not_model_refuses_at_export(model, refusal):
    with dsv4_attention_leaves(model), pytest.raises(NotImplementedError, match=refusal):
        torch.export.export(model, _sparse_mla_inputs(), strict=False)


def test_deep_gemms_einsum_is_the_leaf_op_in_every_vllm_module_bound_to_it(monkeypatch):
    deep_gemm, user = types.ModuleType("vllm.utils.deep_gemm"), types.ModuleType("vllm.o_proj_standin")
    deep_gemm.fp8_einsum = user.fp8_einsum = _launches
    for module in (deep_gemm, user):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    class _OProj(nn.Module):
        def forward(self, a, a_scale, b, b_scale, c=None):
            d = a.new_empty((a.shape[0], b.shape[0], b.shape[1]), dtype=torch.bfloat16)
            user.fp8_einsum("bhr,hdr->bhd", (a, a_scale), (b, b_scale), d, c, recipe=(1, 128, 128))
            return d

    e4m3 = torch.float8_e4m3fn
    args = (torch.empty(3, 2, 256, dtype=e4m3, device="meta"), torch.empty(3, 2, 2, device="meta"),
            torch.empty(2, 128, 256, dtype=e4m3, device="meta"), torch.empty(2, 1, 2, device="meta"))
    with deep_gemm_fp8_einsum_leaf(_OProj()):
        ep = torch.export.export(_OProj(), args, strict=False)
        with pytest.raises(NotImplementedError, match="accumulator"):
            torch.export.export(_OProj(), args + (torch.empty(3, 2, 128, device="meta"),), strict=False)
    assert user.fp8_einsum is _launches and deep_gemm.fp8_einsum is _launches
    (node,) = [n for n in ep.graph.nodes if str(n.target) == "vllm_iface.deep_gemm_fp8_einsum.default"]
    assert node.args[0] == "bhr,hdr->bhd" and node.args[-1] == [1, 128, 128]


def test_marlins_block_size_is_the_first_at_which_rows_per_expert_per_block_fall_below_nine_tenths():
    assert [mml.marlin_block_size_m(M, 6, 384, -1) for M in (1, 256, 2048, 16384)] == [8, 8, 48, 64]
    assert mml.marlin_block_size_m(2048, 6, 48, 384) == 48


class _Mxfp4Method:
    mxfp4_backend, moe_kernel, moe, experts_cls = "marlin", None, "moe_config", "MarlinExperts"

    def get_fused_moe_quant_config(self, layer):
        return ("quant_config", layer.w13_weight.shape)


class _MarlinMoe(nn.Module):
    """`FusedMoE` on `Mxfp4MoEMethod`'s MARLIN backend, E = 4 experts of hidden K = 64 and intermediate N = 32, top-2."""

    def __init__(self, site, bias=None) -> None:
        super().__init__()
        self.site, self.bias, self.quant_method = site, bias, _Mxfp4Method()
        self.w13_weight, self.w2_weight = _meta(4, 64, 32, dtype=torch.uint8), _meta(4, 64, 16, dtype=torch.uint8)
        self.w13_scale, self.w2_scale = _meta(4, 64, 2, dtype=torch.uint8), _meta(4, 64, 1, dtype=torch.uint8)

    def _expert_routing_tables(self):
        return "routing_tables"

    def forward(self, x, topk_weights, topk_ids):
        def activation(name, out, up, topk_ids, expert_map):
            out.copy_(up[:, :out.shape[1]] * up[:, out.shape[1]:])

        def moe_sum(rows, out, topk_ids, expert_map):
            out.copy_(rows.sum(dim=1))

        return self.site.fused_marlin_moe(x, self.w13_weight, self.w2_weight, self.bias, None, self.w13_scale, self.w2_scale,
                                          topk_weights, topk_ids, 7, global_num_experts=4, activation="swiglu",
                                          activation_func=activation, moe_sum=moe_sum, expert_map=None)


class _Routed(nn.Module):
    """A greedy wrapper over `_MarlinMoe`: its one input is the prompt's hidden states, routed to the top 2 of the first 4 features."""

    def __init__(self, moe) -> None:
        super().__init__()
        self.moe = moe

    def forward(self, x):
        weights, ids = torch.topk(x[:, :4].float(), 2)
        return self.moe(x, weights, ids.to(torch.int32))


@pytest.fixture
def marlin_site(monkeypatch):
    """vLLM's Marlin MoE modules, and rank 0 of a TP2 group with its bus in probe mode."""
    mxfp4, oracle = types.ModuleType(mml._MXFP4), types.ModuleType(mml._ORACLE)
    site, scalar_type = types.ModuleType(mml._MARLIN_MOE), types.ModuleType("vllm.scalar_type")
    parallel_state = types.ModuleType(mml._PARALLEL_STATE)
    mxfp4.Mxfp4MoEMethod = _Mxfp4Method
    oracle.Mxfp4MoeBackend = types.SimpleNamespace(MARLIN="marlin")
    oracle.make_mxfp4_moe_kernel = lambda **kwargs: kwargs
    scalar_type.scalar_types = types.SimpleNamespace(float4_e2m1f=types.SimpleNamespace(id=7))
    site.fused_marlin_moe = _launches
    parallel_state._TP = TPStub(0, 2, CollectiveBus(rank=0, world=2))
    parallel_state.get_tp_group = lambda: parallel_state._TP
    for module in (mxfp4, oracle, site, scalar_type, parallel_state):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return site


def _prompt(M: int = 3) -> tuple:
    return (torch.empty(M, 64, dtype=torch.bfloat16, device="meta"),)


def test_the_mxfp4_marlin_moe_exports_as_its_steps_with_its_expert_gemms_as_leaf_ops(marlin_site):
    bus = sys.modules[mml._PARALLEL_STATE]._TP.bus
    model = _Routed(_MarlinMoe(marlin_site))
    with mml.mxfp4_marlin_moe_leaf(model):
        kernel = model.moe.quant_method.moe_kernel
        torch.export.export(model, _prompt(), strict=False)
        assert bus.schedule_record == [{"k": 0, "shape": [3, 2], "pairs": 6, "offset": 0}]
        wrapped, inputs = with_peers(model, _prompt(), bus)
        ep = torch.export.export(wrapped, inputs, strict=False)
        with pytest.raises(ValueError, match="greedy wrapper's form only"):
            with_peers(model, _prompt() + (torch.empty(1),), bus)
    assert model.moe.quant_method.moe_kernel is None and marlin_site.fused_marlin_moe is _launches
    assert kernel["moe_quant_config"] == ("quant_config", (4, 64, 32)) and kernel["routing_tables"] == "routing_tables"
    assert isinstance(wrapped, ScheduledWithPeers) and [tuple(t.shape) for t in inputs] == [(3, 64), (6,)]
    prompt, schedule = [n for n in ep.graph.nodes if n.op == "placeholder" and n.name in ep.graph_signature.user_inputs]
    (align,) = [n for n in ep.graph.nodes if str(n.target) == "vllm_iface.marlin_moe_align.default"]
    ids, key, E, block_m = align.args
    assert key.args == (key.args[0], [3, 2]) and key.args[0].args == (schedule, 0, 0, 6) and (E, block_m) == (4, 8)
    blk, par = sorted(align.users, key=lambda n: n.args[1])
    gemms = [n for n in ep.graph.nodes if str(n.target) == "vllm_iface.marlin_moe_mxfp4_gemm.default"]
    assert [n.args[3:6] for n in gemms] == [(ids, blk, par)] * 2
    assert [(tuple(n.meta["val"].shape), n.args[-3:]) for n in gemms] == [((6, 64), (8, 2, False)), ((6, 64), (8, 1, True))]


def test_a_marlin_moe_without_a_tp_bus_refuses_at_export(marlin_site, monkeypatch):
    monkeypatch.setattr(sys.modules[mml._PARALLEL_STATE], "_TP", types.SimpleNamespace())
    model = _Routed(_MarlinMoe(marlin_site))
    with mml.mxfp4_marlin_moe_leaf(model), pytest.raises(NotImplementedError, match="no TP bus"):
        torch.export.export(model, _prompt(), strict=False)


def test_a_marlin_moe_beyond_the_leafs_operands_refuses_at_export(marlin_site):
    model = _Routed(_MarlinMoe(marlin_site, bias=_meta(4, 64)))
    with mml.mxfp4_marlin_moe_leaf(model), pytest.raises(NotImplementedError, match=r"given \['bias1'\]"):
        torch.export.export(model, _prompt(), strict=False)


H200 = TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False, moe_backend="triton")
HC, HID, REPEAT, EPS = 2, 64, 2, 1e-6
OUT = HC * (2 + HC)
GEOMETRY = {"TH": 2, "TN": 2, "TH_HEAD": 2}


@pytest.fixture
def served(monkeypatch):
    """The H200 as the Build observes it, and a pinned mHC thread geometry at the small hidden size."""
    saved = dict(VB._OBSERVED)
    monkeypatch.setitem(vllm_mhc.MHC_GEOMETRY, HID, GEOMETRY)
    VB.observe_target_profile(H200)
    yield
    VB._OBSERVED.clear()
    VB._OBSERVED.update(saved)


def _f32(rng, *shape, sd: float = 1.0) -> np.ndarray:
    return (rng.standard_normal(shape) * sd).astype(np.float32).view(np.uint32)


def _tensor(words: np.ndarray) -> torch.Tensor:
    """The torch tensor of bf16 (uint16) or f32 (uint32) words."""
    if words.dtype == np.uint16:
        return torch.from_numpy(words.view(np.int16).copy()).view(torch.bfloat16)
    return torch.from_numpy(words.view(np.float32).copy())


def _words(t: torch.Tensor) -> list[int]:
    width = {torch.bfloat16: (torch.int16, np.uint16), torch.float32: (torch.int32, np.uint32), torch.int32: (torch.int32, np.uint32),
             torch.float8_e4m3fn: (torch.uint8, np.uint8), torch.float8_e8m0fnu: (torch.uint8, np.uint8),
             torch.uint8: (torch.uint8, np.uint8)}[t.dtype]
    return t.contiguous().view(width[0]).numpy().reshape(-1).astype(width[1]).tolist()


def _run(rep, model: nn.Module, inputs) -> list[int]:
    held = dict(model.named_parameters())
    weights = [v for w in rep.provenance["weights"] for v in _words(held[w["name"]])]
    t = materialize(rep.program, [_words(x) for x in inputs] + ([weights] if rep.provenance["weights"] else []))
    return [int(t[g]) for g in rep.program.circuit.root_outputs()]


class _Launched(nn.Module):
    """``model`` traced as request step 0 of the declared launch batch ``batch`` (`observations.launch_region`)."""

    def __init__(self, model: nn.Module, batch: list) -> None:
        super().__init__()
        self.model = model
        VB.observe_launch_context(TG.parse_launch_context({"batch": batch}))

    def forward(self, *xs):
        with OB.launch_region(0, "tokens"):
            return self.model(*xs)


def _derive_launched(derive, model: nn.Module, inputs):
    with fxt.preserve_node_meta():
        return derive(model, inputs)


class _Hyper(nn.Module):
    """`DeepseekV4Model`'s mHC calls on HC = 2 streams of H = 64: layer 0's broadcast pre, the next sublayer's fused post + pre
    (the sublayer itself the identity), the last post and the head.  The calls pass split counts the dispatch does not run."""

    def __init__(self, rng, norm_weight: bool = True, norm_eps: float = EPS) -> None:
        super().__init__()
        p = lambda words: nn.Parameter(_tensor(words), requires_grad=False)
        self.fn0, self.fn0_broadcast = p(_f32(rng, OUT, HC * HID, sd=0.1)), p(_f32(rng, OUT, HID, sd=0.1))
        self.fn1 = p(_f32(rng, OUT, HC * HID, sd=0.1))
        self.scale0, self.base0, self.scale1, self.base1 = (p(_f32(rng, n, sd=0.5)) for n in (3, OUT, 3, OUT))
        self.norm0, self.norm1 = (p(W.bf16(_f32(rng, HID))) for _ in range(2))
        self.head_fn, self.head_scale, self.head_base = p(_f32(rng, HC, HC * HID, sd=0.1)), p(_f32(rng, 1, sd=0.5)), p(_f32(rng, HC, sd=0.5))
        self.norm_weight, self.norm_eps = norm_weight, norm_eps

    def forward(self, x):
        norm0 = self.norm0 if self.norm_weight else None
        streams, post, comb, h = iface.mhc_pre_broadcast_tilelang(x, self.fn0, self.scale0, self.base0, EPS, EPS, EPS, 2.0, REPEAT, 3,
                                                                  norm0, self.norm_eps, self.fn0_broadcast)
        streams, post, comb, h = iface.mhc_fused_post_pre_tilelang(h, streams, post, comb, self.fn1, self.scale1, self.base1, EPS, EPS,
                                                                   EPS, 2.0, REPEAT, 3, 7, self.norm1, EPS)
        streams = iface.mhc_post_tilelang(h, streams, post, comb)
        return iface.hc_head_fused_kernel_tilelang(streams, self.head_fn, self.head_scale, self.head_base, EPS, EPS)


def _twin(model: _Hyper, x: np.ndarray, NS: int, NS0: int, M: int | None = None) -> list[int]:
    """The model's output as the twin composes the kernels at T tokens in a launch of M (T without one): the FMA path at most 16
    tokens, the TF32 path above."""
    u = lambda t: np.array(_words(t), np.uint16 if t.dtype == torch.bfloat16 else np.uint32).reshape(t.shape)
    m = {k: u(v) for k, v in model.named_parameters()}
    TN, TH, TH_HEAD = GEOMETRY["TN"], GEOMETRY["TH"], GEOMETRY["TH_HEAD"]
    h, post, comb = W.pre_broadcast(x, m["fn0_broadcast"], m["scale0"], m["base0"], m["norm0"], NS0, TN, REPEAT, EPS, EPS)
    streams = np.broadcast_to(x[:, None, :], (len(x), HC, HID))
    args = (m["fn1"], m["scale1"], m["base1"], m["norm1"])
    if (M or len(x)) <= vllm_mhc.SMALL_FMA_TOKENS:
        streams, h, post, comb = W.fused_post_pre(h, streams, post, comb, *args, NS, TH, TN, REPEAT, EPS, EPS)
    else:
        streams, h, post, comb = W.post_pre_tf32(h, streams, post, comb, *args, NS, TN, REPEAT, EPS, EPS)
    streams = W.post(h, streams, post, comb)
    y = W.hc_head(streams, m["head_fn"], m["head_scale"], m["head_base"], TH_HEAD, EPS, EPS)
    return [int(v) for v in np.asarray(y).reshape(-1)]


def _derive_mhc(model: nn.Module, inputs):
    return derive_report(model, inputs, mode="profile", profile=PROFILE_B1_EAGER_V3.name, name="mhc", rules=vllm_ruleset())


@pytest.mark.parametrize("T, NS, NS0", [(2, 8, 1), (17, 1, 1)])
def test_the_mhc_ops_derive_per_token_at_the_dispatchs_splits_word_for_word_the_twins(T, NS, NS0, served):
    rng = np.random.default_rng(T)
    model, x = _Hyper(rng), W.bf16(_f32(rng, T, HID, sd=2.0))
    rep = _derive_mhc(model, (_tensor(x),))
    assert rep.program is not None, rep.unsupported
    apps = {a["rule"]: a["preconditions_checked"] for a in rep.provenance["applications"]}
    fused = "mhc_fused_post_pre" if T <= vllm_mhc.SMALL_FMA_TOKENS else "mhc_post_pre_tf32"
    assert apps["MhcPreBroadcast[vllm-dsv4]_v1"]["NS"] == NS0 and apps["MhcFusedPostPre[vllm-dsv4]_v1"]["NS"] == NS
    assert apps["HcHead[vllm-dsv4]_v1"]["TH"] == GEOMETRY["TH_HEAD"]
    want = [PROFILE_B1_EAGER_V3.kind(r, **{k: v for k, v in s.items() if k != "T"}).id for r, s in (
        ("mhc_pre_broadcast", apps["MhcPreBroadcast[vllm-dsv4]_v1"]), (fused, apps["MhcFusedPostPre[vllm-dsv4]_v1"]))]
    ids = {n.fn.id for n in rep.program.fn.body.nodes}
    assert set(want) <= ids and len(ids) == 5
    roles = rep.provenance["weight_roles"]
    assert (roles["fn0_broadcast"]["role"], roles["fn1"]["role"], roles["head_fn"]["role"]) == ("mhc.fn_broadcast", "mhc.fn", "hc_head.fn")
    assert _run(rep, model, (_tensor(x),)) == _twin(model, x, NS, NS0)


@pytest.mark.parametrize("batch, path, NS, NS0", [([[1, 40]], "mhc_fused_post_pre", 8, 1), ([[1, 40], [16, 0]], "mhc_post_pre_tf32", 1, 1)])
def test_the_mhc_ops_take_their_path_and_splits_at_the_launchs_tokens(batch, path, NS, NS0, served):
    """A decode token alone runs the FMA path; beside a 16-row prefill its launch has 17 tokens and runs the TF32 path."""
    rng = np.random.default_rng(1)
    model, x = _Hyper(rng), W.bf16(_f32(rng, 1, HID, sd=2.0))
    launched = _Launched(model, [batch])
    rep = _derive_launched(_derive_mhc, launched, (_tensor(x),))
    assert rep.program is not None, rep.unsupported
    apps = {a["rule"]: a["preconditions_checked"] for a in rep.provenance["applications"]}
    assert apps["MhcPreBroadcast[vllm-dsv4]_v1"]["NS"] == NS0 and apps["MhcFusedPostPre[vllm-dsv4]_v1"]["NS"] == NS
    fused = {k: v for k, v in apps["MhcFusedPostPre[vllm-dsv4]_v1"].items() if k != "T"}
    assert PROFILE_B1_EAGER_V3.kind(path, **fused).id in {n.fn.id for n in rep.program.fn.body.nodes}
    assert _run(rep, launched, (_tensor(x),)) == _twin(model, x, NS, NS0, M=sum(n for n, _ in batch))


@pytest.mark.parametrize("over, refusal", [
    ({"geometry": None}, "no pinned mHC thread geometry at hidden size 64 (pinned: [7168])"),
    ({"norm_weight": False}, "the mHC pre without its fused RMS norm"),
    ({"norm_eps": 1e-5}, "the Definitions take one EPS"),
    ({"target": TargetProfile(compute_capability=(12, 0), num_sms=188, batch_invariant=False, moe_backend="triton")},
     "the TF32 pre is DeepGEMM's cc 9.0 hc-prenorm GEMM"),
])
def test_what_the_mhc_definitions_do_not_take_refuses_by_name(over, refusal, served, monkeypatch):
    rng = np.random.default_rng(0)
    if "geometry" in over:
        monkeypatch.delitem(vllm_mhc.MHC_GEOMETRY, HID)
    if "target" in over:
        VB.observe_target_profile(over["target"])
    model = _Hyper(rng, **{k: v for k, v in over.items() if k in ("norm_weight", "norm_eps")})
    rep = _derive_mhc(model, (_tensor(W.bf16(_f32(rng, 2, HID))),))
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


class _F32Out(nn.Module):
    """DeepSeek-V4's compressor GEMM: a direct `torch.mm(x, W.T, out_dtype=float32)`, W [N, 7168] bf16."""

    def __init__(self, N: int = 1024) -> None:
        super().__init__()
        self.weight = _meta(N, 7168)

    def forward(self, x):
        return torch.mm(x, self.weight.T, out_dtype=torch.float32)


def test_a_direct_f32_output_mm_binds_the_captured_f32out_route(served):
    rep = _derive_mhc(_F32Out(), (torch.empty(4, 7168, dtype=torch.bfloat16, device="meta"),))
    assert rep.program is not None, rep.unsupported
    want = PROFILE_B1_EAGER_V3.kind("cublas_splitk_gemm_f32out", K=7168, N=1024, SPLIT_K=5, DOT=hopper.HopperBF16WgmmaDot16).id
    assert [n.fn.id for n in rep.program.fn.body.nodes] == [want] * 4
    assert rep.provenance["weight_roles"]["weight"]["role"] == "linear.weight"
    M = BC.CUBLAS_ROUTES["m_max"] + 1
    rep = _derive_mhc(_F32Out(), (torch.empty(M, 7168, dtype=torch.bfloat16, device="meta"),))
    assert f"no captured vLLM-default cuBLAS f32out route for M={M}, K=7168, N=1024" in rep.unsupported.hint
    rep = _derive_mhc(_F32Out(2048), (torch.empty(256, 7168, dtype=torch.bfloat16, device="meta"),))
    assert rep.program is not None, rep.unsupported
    want = PROFILE_B1_EAGER_V3.kind("cublas_gemm_f32out", K=7168, N=2048, DOT=hopper.HopperBF16WgmmaDot16).id
    assert {n.fn.id for n in rep.program.fn.body.nodes} == {want}


class _QKvNorm(nn.Module):
    """DeepSeek-V4's attention norms: `fused_q_kv_rmsnorm` of the q and kv column slices of one projection's output, `pad` columns
    after them, launching `_fused_q_kv_rmsnorm_kernel` as the export captures it."""

    def __init__(self, rng, Q: int, KV: int, pad: int = 0, src: str = VB.QKV_RMS_PINS["src_sha256"]) -> None:
        super().__init__()
        self.Q, self.KV, self.pad, self.src = Q, KV, pad, src
        self.q_weight, self.kv_weight = (nn.Parameter(_tensor(W.bf16(_f32(rng, n))), requires_grad=False) for n in (Q, KV))

    def forward(self, qkv):
        q, kv = qkv[:, :self.Q], qkv[:, self.Q:self.Q + self.KV]
        q_out, kv_out = (torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in (q, kv))
        block = 1 << (max(self.Q, self.KV) - 1).bit_length()
        scalars = {"q_in_stride": qkv.stride(0), "q_out_stride": self.Q, "kv_in_stride": qkv.stride(0), "kv_out_stride": self.KV,
                   "eps": EPS, "Q_SIZE": self.Q, "KV_SIZE": self.KV, "BLOCK_SIZE": block, "launch_pdl": True}
        meta = {"kernel": VB.QKV_RMS_PINS["kernel"], "src_sha256": self.src, "grid": [qkv.shape[0], 2], "scalars": scalars,
                "geometry": {"num_warps": 8 if block >= 2048 else 4},
                "tensor_args": ["q_ptr", "q_out_ptr", "q_weight_ptr", "kv_ptr", "kv_out_ptr", "kv_weight_ptr"]}
        triton_launch(meta["kernel"], [q, q_out, self.q_weight, kv, kv_out, self.kv_weight], json.dumps(meta))
        return q_out, kv_out


@pytest.mark.parametrize("T, Q, KV", [(2, 256, 64), (1, 1536, 512)])
def test_the_fused_q_kv_rmsnorm_derives_per_token_word_for_word_its_ptxs(T, Q, KV):
    rng = np.random.default_rng(Q)
    model, x = _QKvNorm(rng, Q, KV), W.bf16(_f32(rng, T, Q + KV, sd=2.0))
    rep = _derive_mhc(model, (_tensor(x),))
    assert rep.program is not None, rep.unsupported
    block = 1 << (Q - 1).bit_length()
    (app,) = [a["preconditions_checked"] for a in rep.provenance["applications"] if a["rule"] == "TritonQKvRmsNorm[vllm-default]_v1"]
    assert (app["BLOCK"], app["WARPS"], app["rows"]) == (block, 8 if block >= 2048 else 4, T)
    roles = rep.provenance["weight_roles"]
    assert (roles["q_weight"]["role"], roles["kv_weight"]["role"]) == ("rmsnorm.q.weight", "rmsnorm.kv.weight")
    w = {n: _words(p) for n, p in model.named_parameters()}
    want = [v for cols, wn in ((slice(0, Q), "q_weight"), (slice(Q, Q + KV), "kv_weight")) for row in x[:, cols]
            for v in ptx_block_model(row.tolist(), w[wn], EPS, block, app["WARPS"])]
    assert _run(rep, model, (_tensor(x),)) == want


@pytest.mark.parametrize("over, refusal", [({"src": "00" * 32}, "differs from the pin"),
                                           ({"Q": 128}, "at least two elements a thread"),
                                           ({"pad": 8}, "another Triton specialization than the PTX")])
def test_a_fused_q_kv_rmsnorm_launch_beyond_the_ptx_refuses(over, refusal):
    rng = np.random.default_rng(1)
    model = _QKvNorm(rng, **{"Q": 256, "KV": 64, **over})
    rep = _derive_mhc(model, (_tensor(W.bf16(_f32(rng, 2, model.Q + model.KV + model.pad))),))
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


class _InsertState(nn.Module):
    """The sparse MLA's compressor state and insert leaves at positions p0 .. p0 + T - 1: the state width 8 at ratio 4; 2 live
    heads padded to NHP, head 64 of which 16 rotary, quant block QB."""

    def __init__(self, rng, NHP: int = 4, QB: int = 16, p0: int = 5) -> None:
        super().__init__()
        self.NHP, self.QB, self.p0 = NHP, QB, p0
        self.ape = nn.Parameter(_tensor(_f32(rng, 4, 8, sd=0.1)), requires_grad=False)
        self.cos_sin_cache = nn.Parameter(_tensor(TQ._cs(rng, (32, 16))), requires_grad=False)

    def forward(self, kv_score, q, kv):
        positions = torch.arange(self.p0, self.p0 + q.shape[0], dtype=torch.int64)
        return (*iface.dsv4_compressor_state(kv_score, positions, self.ape),
                *iface.dsv4_qnorm_rope_kv_insert(q, kv, positions, self.cos_sin_cache, self.NHP, EPS, self.QB))


def _insert_state_inputs(rng, T: int) -> tuple:
    return _f32(rng, T, 16), TQ._bf16(rng, (T, 2, 64)), TQ._bf16(rng, (T, 64))


@pytest.mark.parametrize("T", [1, 3])
def test_the_insert_and_compressor_state_derive_per_token_word_for_word_the_twins(T):
    rng = np.random.default_rng(T)
    model, xs = _InsertState(rng), _insert_state_inputs(rng, T)
    rep = _derive_mhc(model, tuple(map(_tensor, xs)))
    assert rep.program is not None, rep.unsupported
    apps = {a["rule"]: a["preconditions_checked"] for a in rep.provenance["applications"]}
    assert apps["Dsv4QNormRopeKvInsert[vllm-dsv4]_v1"]["positions"] == apps["Dsv4CompressorState[vllm-dsv4]_v1"]["positions"] == [5, 4 + T]
    roles = rep.provenance["weight_roles"]
    assert (roles["ape"]["role"], roles["cos_sin_cache"]["role"]) == ("dsv4.compressor.ape", "rope.cos_sin_cache")
    kv_score, q, kv = xs
    pos = np.arange(5, 5 + T)
    ape, cs = (np.array(_words(p), np.uint32).reshape(p.shape) for p in (model.ape, model.cos_sin_cache))
    outs = (kv_score[:, :8], FT.fadd(kv_score[:, 8:], ape[pos % 4]), *TQ.insert(q, kv, cs[pos], NHP=4, DR=16, QB=16, EPS=EPS))
    assert _run(rep, model, tuple(map(_tensor, xs))) == [int(v) for o in outs for v in np.asarray(o).reshape(-1)]


@pytest.mark.parametrize("over, refusal", [({"NHP": 1}, "the live heads NH=2 must be in [1, NHP=1]"),
                                           ({"QB": 20}, "QB=20 must divide DN=48"),
                                           ({"p0": 30}, "a position outside cos_sin (32 rows)"),
                                           ({"p0": -1}, "a negative position")])
def test_an_insert_or_state_beyond_the_definitions_refuses_by_name(over, refusal):
    rng = np.random.default_rng(0)
    rep = _derive_mhc(_InsertState(rng, **over), tuple(map(_tensor, _insert_state_inputs(rng, 3))))
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


SPARSE_D, SPARSE_DR, SPARSE_BN, SPARSE_SCALE = 64, 16, 16, 0.125


class _SparseAttn(nn.Module):
    """A DeepSeek-V4 attention layer's leaves over one request, a call of each length in ``calls`` from position p0, ``skip``
    positions between calls: 2 live heads padded to NHP, head 64 of which 16 rotary, quant block 16; the compressor's state 2D wide
    at ratio 4, D at 128, none at 1."""

    def __init__(self, rng, ratio: int, window: int, calls: tuple[int, ...], topk: int = 1024, NHP: int = 64, p0: int = 0,
                 skip: int = 0) -> None:
        super().__init__()
        p = lambda words: nn.Parameter(_tensor(words), requires_grad=False)
        self.ratio, self.window, self.calls, self.topk, self.NHP, self.p0, self.skip = ratio, window, calls, topk, NHP, p0, skip
        self.cos_sin_cache, self.attn_sink = p(TQ._cs(rng, (p0 + sum(calls) + skip * len(calls), SPARSE_DR))), p(_f32(rng, NHP))
        if ratio > 1:
            self.ape = p(_f32(rng, ratio, SPARSE_D * (2 if ratio == 4 else 1), sd=0.1))
            self.norm_weight = p(W.bf16(_f32(rng, SPARSE_D)))

    def forward(self, *xs):
        k, outs, p0 = 3 if self.ratio > 1 else 2, [], self.p0
        for call in zip(*(xs[i::k] for i in range(k))):
            q, kv = call[-2:]
            positions = torch.arange(p0, p0 + q.shape[0], dtype=torch.int64)
            p0 += q.shape[0] + self.skip
            with OB.launch_region(len(outs), "tokens"):
                state = iface.dsv4_compressor_state(call[0], positions, self.ape) if self.ratio > 1 else (None, None)
                qp, nope, scale, rope = iface.dsv4_qnorm_rope_kv_insert(q, kv, positions, self.cos_sin_cache, self.NHP, EPS, SPARSE_DR)
                outs.append(iface.dsv4_sparse_attn(qp, nope, scale, rope, positions, self.cos_sin_cache, self.attn_sink, *state,
                                                   self.norm_weight if self.ratio > 1 else None, "layers.0.attn", 2, self.ratio,
                                                   self.window, self.topk, SPARSE_SCALE, EPS))
        return tuple(outs)


def _sparse_inputs(rng, model: _SparseAttn) -> list[np.ndarray]:
    state = [] if model.ratio == 1 else [lambda T: _f32(rng, T, 2 * model.ape.shape[1])]
    return [x(T) for T in model.calls for x in state + [lambda T: W.bf16(_f32(rng, T, 2, SPARSE_D)),
                                                         lambda T: W.bf16(_f32(rng, T, SPARSE_D))]]


def _sparse_twin(model: _SparseAttn, xs: list[np.ndarray], launch: dict | None = None) -> list[int]:
    """The layer's outputs as the twins compose the kernels over its window, state and compressed histories from position P; the
    decode at position p planned among the (window, compressed) lengths and at the index `launch[p]` gives, else alone."""
    u = lambda t: np.array(_words(t), np.uint16 if t.dtype == torch.bfloat16 else np.uint32).reshape(t.shape)
    m = {k: u(v) for k, v in model.named_parameters()}
    ratio, window, cs, sink = model.ratio, model.window, m["cos_sin_cache"], m["attn_sink"][:2]
    D, DN, QB, BN, k = SPARSE_D, SPARSE_D - SPARSE_DR, SPARSE_DR, SPARSE_BN, 3 if model.ratio > 1 else 2
    stack = lambda rows: [np.stack(c) for c in zip(*rows)]
    words = lambda v: np.array(v, np.int64).astype(np.uint32)
    win, state, comp, outs, P = [], [], [], [], model.p0
    p0 = P
    for call in zip(*(xs[i::k] for i in range(k))):
        q, kv = call[-2:]
        pos = np.arange(p0, p0 + len(q))
        p0 += len(q)
        qp, nope, sc, rope = TQ.insert(q, kv, cs[pos], NHP=model.NHP, DR=SPARSE_DR, QB=QB, EPS=EPS)
        win += list(zip(nope, sc[:, :DN // QB], rope))
        if ratio > 1:
            w = m["ape"].shape[1]
            state += list(zip(call[0][:, :w], FT.fadd(call[0][:, w:], m["ape"][pos % ratio])))
            for p in pos[(pos + 1) % ratio == 0]:
                taken = [(r, s - P) for r, s in enumerate(range(p - (8 if ratio == 4 else 128) + 1, p + 1)) if s >= P]
                kvr, scr = (np.stack([state[s][j][r // ratio * D:(r // ratio + 1) * D] for r, s in taken]) for j in (0, 1))
                args = (kvr, scr, m["norm_weight"], cs[p - ratio + 1])
                row = DC.compress_c4(*args, QB, EPS) if ratio == 4 else DC.compress_c128(*args, 8, 16, QB, EPS)
                n, ro, e = (a.reshape(-1) for a in row)
                comp.append((n, e[:DN // QB], ro))
        for i, p in enumerate(pos):
            s0, nc = max(p - window + 1, P), (p + 1) // ratio - P // ratio if ratio > 1 else 0
            w0 = s0 - P
            if len(q) == 1:
                swa = p - s0 + 1
                NW, N = FT.request_blocks(swa, nc, BN)
                rows = win[w0:w0 + swa] + win[w0:w0 + 1] * (NW - swa) + comp[:nc] + win[w0:w0 + 1] * (N - NW - nc)
                idx = words([0] * swa + [-1] * (NW - swa) + [0] * nc + [-1] * (N - NW - nc))
                lens, req = (launch or {}).get(p, ([(swa, nc)], 0))
                metas, ns = FT.sched_meta([w for w, _ in lens], [c for _, c in lens] if ratio > 1 else None, topk=window,
                                          extra_topk=int(ratio > 1), parts=FT.num_sm_parts(model.NHP), block_size_n=BN)
                outs.append(FT.decode(qp[i, :2], *stack(rows), idx, [swa] + ([nc] if N > NW else []), sink,
                                      SPLITS=FT.request_splits(metas, ns, req, N // BN), NW=NW, QT=QB, BN=BN, SCALE=SPARSE_SCALE))
            else:
                rows = comp[:nc] + win[w0:p - P + 1]
                n, N = len(rows), -(-len(rows) // (2 * BN)) * 2 * BN
                kvw = np.zeros((N, D), np.uint16)
                kvw[:n] = FP.gather_rows(*stack(rows), QB)
                outs.append(FP.prefill(qp[i, :2], kvw, words([0] * n + [-1] * (N - n)), words(n), words(1), sink, BN=BN,
                                       SCALE=SPARSE_SCALE))
    return [int(v) for o in outs for v in np.asarray(o).reshape(-1)]


@pytest.fixture
def flashmla_small(served, monkeypatch):
    """FlashMLA's sparse kernels at head 64 of which 16 rotary: scale groups and blocks of 16 rows."""
    monkeypatch.setitem(vllm_dsv4.FLASHMLA_GEOMETRY, (SPARSE_D, SPARSE_DR), {"QT": SPARSE_DR, "BN": SPARSE_BN, "DOT": FT.HopperBF16WgmmaDot16})


@pytest.mark.parametrize("ratio, window, calls, p0, paths", [
    pytest.param(4, 28, (30, 1, 1), 0, [("prefill", list(range(7))), ("decode", []), ("decode", [7])], marks=pytest.mark.slow),
    pytest.param(128, 8, (128, 1), 0, [("prefill", [0]), ("decode", [])], marks=pytest.mark.slow),
    (1, 8, (12, 1), 0, [("prefill", []), ("decode", [])]),
    pytest.param(4, 28, (26, 1, 1), 5, [("prefill", list(range(1, 7))), ("decode", [7]), ("decode", [])], marks=pytest.mark.slow),
    (4, 28, (1,), 8, [("decode", [])]),
    (128, 8, (1,), 8, [("decode", [])]),
])
def test_the_sparse_attention_derives_over_its_histories_word_for_word_the_twins(ratio, window, calls, p0, paths, flashmla_small):
    """A history opened past position 0 (the step wrapper's `pos0`; the last two cases are its one-token call) reads no row before it."""
    rng = np.random.default_rng(ratio)
    model = _SparseAttn(rng, ratio, window, calls, p0=p0)
    xs = _sparse_inputs(rng, model)
    rep = _derive_mhc(model, tuple(map(_tensor, xs)))
    assert rep.program is not None, rep.unsupported
    apps = [a["preconditions_checked"] for a in rep.provenance["applications"] if a["rule"] == "Dsv4SparseAttn[vllm-dsv4]_v1"]
    assert [(a["path"], a["compressed_rows"]) for a in apps] == paths
    assert {a["history_from"] for a in apps} == {p0}
    if (ratio, p0) == (4, 0):
        assert len(apps[2]["SPLITS"]) > 2, "the last decode combines its splits"
    roles = {n: r["role"] for n, r in rep.provenance["weight_roles"].items()}
    assert roles["attn_sink"] == "dsv4.attn_sink" and roles.get("norm_weight", "dsv4.compressor.norm.weight") == "dsv4.compressor.norm.weight"
    assert _run(rep, model, tuple(map(_tensor, xs))) == _sparse_twin(model, xs)


@pytest.mark.parametrize("over, refusal", [
    ({"geometry": None}, "no FlashMLA sparse geometry at head 64, rope 16"),
    ({"NHP": 4}, "64 or 128 padded heads, at least one live (2 of 4)"),
    ({"topk": 4}, "position 29 has 7 compressed rows, more than the indexer's top-4"),
    ({"p0": 4, "skip": 1, "calls": (26, 1)}, "positions 31..31 are not the 1 after 'layers.0.attn''s history of 26 rows from position 4"),
    ({"target": TargetProfile(compute_capability=(12, 0), num_sms=188, batch_invariant=False, moe_backend="triton")},
     "the FlashMLA Definitions are its cc 9.0 kernels"),
])
def test_a_sparse_attention_beyond_the_definitions_refuses_by_name(over, refusal, flashmla_small, monkeypatch):
    rng = np.random.default_rng(0)
    if "geometry" in over:
        monkeypatch.delitem(vllm_dsv4.FLASHMLA_GEOMETRY, (SPARSE_D, SPARSE_DR))
    if "target" in over:
        VB.observe_target_profile(over["target"])
    model = _SparseAttn(rng, 4, 28, over.get("calls", (30,)), **{k: v for k, v in over.items() if k in ("NHP", "topk", "p0", "skip")})
    rep = _derive_mhc(model, tuple(map(_tensor, _sparse_inputs(rng, model))))
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


def _decode_splits(lens: list[tuple[int, int]], req: int, window: int) -> tuple[int, ...]:
    metas, ns = FT.sched_meta([w for w, _ in lens], [c for _, c in lens], topk=window, extra_topk=1, parts=FT.num_sm_parts(64),
                              block_size_n=SPARSE_BN)
    return FT.request_splits(metas, ns, req, sum(-(-n // SPARSE_BN) for n in lens[req]))


@pytest.mark.slow
def test_a_decode_takes_its_splits_among_the_launchs_decodes(flashmla_small):
    """FlashMLA plans a step's decodes together: the request's decode at position 30, second of 31 decodes beside a prefill, takes
    the splits of its index among their lengths, which differ from its splits alone and from those of the first decode."""
    rng = np.random.default_rng(4)
    model = _SparseAttn(rng, 4, 28, (30, 1))
    xs = _sparse_inputs(rng, model)
    decodes = [1500, 30] + [1500] * 29
    VB.observe_launch_context(TG.parse_launch_context({"batch": [[[30, 0]], [[1, q] for q in decodes] + [[12, 0]]]}))
    rep = _derive_launched(_derive_mhc, model, tuple(map(_tensor, xs)))
    assert rep.program is not None, rep.unsupported
    lens = [(min(q + 1, 28), (q + 1) // 4) for q in decodes]
    (app,) = [a["preconditions_checked"] for a in rep.provenance["applications"]
              if a["rule"] == "Dsv4SparseAttn[vllm-dsv4]_v1" and a["preconditions_checked"]["path"] == "decode"]
    assert tuple(app["SPLITS"]) == _decode_splits(lens, 1, 28) == (0, 1, 3)
    assert _decode_splits(lens[1:2], 0, 28) == (0, 1, 2, 3) and _decode_splits([lens[1], lens[0], *lens[2:]], 0, 28) == (0, 3)
    assert _run(rep, model, tuple(map(_tensor, xs))) == _sparse_twin(model, xs, {30: (lens, 1)})


@pytest.mark.parametrize("p0, decodes, refusal", [
    (0, [30, 30], "the declared launch batch holds 2 decodes at position 30, not one"),
    (0, [12], "the declared launch batch holds 0 decodes at position 30, not one"),
    (5, [30, 9], "the decode at position 30 reads (26, 6) (window, compressed) rows, not the (28, 7) of a sequence from position 0"),
])
def test_a_decode_the_launch_does_not_name_refuses(p0, decodes, refusal, flashmla_small):
    rng = np.random.default_rng(4)
    model = _SparseAttn(rng, 4, 28, (30 - p0, 1), p0=p0)
    VB.observe_launch_context(TG.parse_launch_context({"batch": [[[30 - p0, p0]], [[1, q] for q in decodes]]}))
    rep = _derive_launched(_derive_mhc, model, tuple(map(_tensor, _sparse_inputs(rng, model))))
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


class _OProjInput(nn.Module):
    """`deep_gemm_fp8_o_proj`'s `fused_inv_rope_fp8_quant` on the live heads at positions p0 ..: 2 groups of 2 heads, each head 2
    quantisation groups of 32 with a rotary suffix of 16, the returns transposed to [T, G, ...] as the wrapper returns them."""

    def __init__(self, rng, tma: bool = False, fp8_max: float = 448.0, rope_start: int = 16, p0: int = 3) -> None:
        super().__init__()
        self.tma, self.fp8_max, self.rope_start, self.p0 = tma, fp8_max, rope_start, p0
        self.cos_sin_cache = nn.Parameter(_tensor(TQ._cs(rng, (16, 16))), requires_grad=False)

    def forward(self, o):
        T = o.shape[0]
        positions = torch.arange(self.p0, self.p0 + T, dtype=torch.int64)
        fp8, scale = iface.fused_inv_rope_fp8_quant_kernel(o, positions, self.cos_sin_cache, 2, 32, 2, self.rope_start, 8, self.tma,
                                                           self.fp8_max, -(-T // 4) * 4, T, 2, 2 * 64, 2 * 2)
        return fp8.transpose(0, 1), scale.transpose(0, 1)


def test_the_o_projections_inverse_rope_quant_derives_per_token_and_head_word_for_word_the_reference(served):
    rng = np.random.default_rng(5)
    model, o = _OProjInput(rng), W.bf16(_f32(rng, 3, 4, 64, sd=4.0))
    rep = _derive_mhc(model, (_tensor(o),))
    assert rep.program is not None, rep.unsupported
    (app,) = [a["preconditions_checked"] for a in rep.provenance["applications"] if a["rule"] == "Dsv4InvRopeQuant[vllm-dsv4]_v1"]
    assert (app["G"], app["HPG"], app["CHUNKS"], app["QG"], app["ROPE"]) == (2, 2, 2, 32, 16)
    assert rep.provenance["weight_roles"]["cos_sin_cache"]["role"] == "rope.cos_sin_cache"
    cs = np.array(_words(model.cos_sin_cache), np.uint32).reshape(16, 16)
    heads = [[inv_rope_head(o[t, h].tolist(), cs[3 + t].tolist(), 32) for h in range(4)] for t in range(3)]
    assert _run(rep, model, (_tensor(o),)) == [v for k in (slice(0, 64), slice(64, 66)) for row in heads for hd in row for v in hd[k]]


@pytest.mark.parametrize("over, refusal", [
    ({"tma": True}, "with f32 scales; the target is cc 9.0, tma_aligned_scales True"),
    ({"fp8_max": 240.0}, "fp8_max 240.0 (the Definition clamps to 448)"),
    ({"rope_start": 8}, "the launch is not 3 tokens of 2 heads a group, each 2 groups of 32 with a rotary suffix of 16"),
    ({"p0": 14}, "a position outside cos_sin (16 rows)"),
    ({"target": TargetProfile(compute_capability=(12, 0), num_sms=188, batch_invariant=False, moe_backend="triton")},
     "the Definition is the cc 9.0 PTX with f32 scales; the target is cc 12.0"),
])
def test_an_inverse_rope_quant_launch_beyond_the_definition_refuses_by_name(over, refusal, served):
    rng = np.random.default_rng(0)
    if "target" in over:
        VB.observe_target_profile(over["target"])
    model = _OProjInput(rng, **{k: v for k, v in over.items() if k != "target"})
    rep = _derive_mhc(model, (_tensor(W.bf16(_f32(rng, 3, 4, 64))),))
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


class _OProj(nn.Module):
    """`deep_gemm_fp8_o_proj` up to `wo_b`: the inverse-RoPE quant of 2 groups of 2 heads of 128 (one quantisation group each, a
    rotary suffix of 16), then `fp8_einsum("bhr,hdr->bhd")` into `z` [T, 2, N] with wo_a's grouped weight [2, N, 256] and block
    scales [2, 1, 2] as the loader leaves them."""

    N = 64

    def __init__(self, rng, expr: str = "bhr,hdr->bhd", recipe: tuple = (1, 128, 128)) -> None:
        super().__init__()
        self.expr, self.recipe = expr, list(recipe)
        p = lambda t: nn.Parameter(t, requires_grad=False)
        self.cos_sin_cache = p(_tensor(TQ._cs(rng, (16, 16), p=0.0)))
        self.weight = p(torch.from_numpy(rng.standard_normal((2, self.N, 256)).astype(np.float32)).to(torch.float8_e4m3fn))
        self.weight_scale_inv = p(torch.from_numpy((np.abs(rng.standard_normal((2, 1, 2))) * 0.01).astype(np.float32)))

    def forward(self, o):
        T = o.shape[0]
        positions = torch.arange(3, 3 + T, dtype=torch.int64)
        fp8, scale = iface.fused_inv_rope_fp8_quant_kernel(o, positions, self.cos_sin_cache, 2, 128, 1, 112, 8, False, 448.0,
                                                           -(-T // 4) * 4, T, 2, 2 * 128, 2)
        z = torch.empty(T, 2, self.N, dtype=torch.bfloat16)
        iface.deep_gemm_fp8_einsum(self.expr, fp8.transpose(0, 1), scale.transpose(0, 1), self.weight, self.weight_scale_inv, z,
                                   self.recipe)
        return z


def test_the_o_projections_einsum_derives_per_token_word_for_word_the_grouped_deepgemm_twin(served):
    rng = np.random.default_rng(7)
    model, o = _OProj(rng), W.bf16(_f32(rng, 3, 4, 128, sd=4.0))
    rep = _derive_mhc(model, (_tensor(o),))
    assert rep.program is not None, rep.unsupported
    (app,) = [a for a in rep.provenance["applications"] if a["rule"] == "DeepGemmFp8Einsum[vllm]_v1"]
    assert app["emitted"] == "ScaledMmFp8BlockGrouped_v1{H=2,K=256,N=64,G=128} x3"
    roles = rep.provenance["weight_roles"]
    assert (roles["weight"]["role"], roles["weight_scale_inv"]["role"]) == ("fp8_einsum_w", "fp8_einsum_ws")
    cs = np.array(_words(model.cos_sin_cache), np.uint32).reshape(16, 16)
    w = np.array(_words(model.weight), np.uint8).reshape(2, _OProj.N, 256)
    ws = np.array(_words(model.weight_scale_inv), np.uint32).reshape(2, 1, 2)
    want = []
    for t in range(3):
        heads = [inv_rope_head(o[t, h].tolist(), cs[3 + t].tolist(), 128) for h in range(4)]
        xq = np.array([heads[2 * g][:128] + heads[2 * g + 1][:128] for g in range(2)], np.uint8)
        sx = np.array([[heads[2 * g][128], heads[2 * g + 1][128]] for g in range(2)], np.uint32)
        y, nonfinite = DT.scaled_mm_fp8_block_grouped_row(xq, sx, w, ws)
        assert not nonfinite.any()
        want += [int(v) for v in y.reshape(-1)]
    assert _run(rep, model, (_tensor(o),)) == want


@pytest.mark.parametrize("over, refusal", [
    ({"expr": "bhr,hdr->bdh"}, "fp8_einsum('bhr,hdr->bdh', recipe=[1, 128, 128]); the Definition is 'bhr,hdr->bhd'"),
    ({"recipe": (1, 1, 128)}, "fp8_einsum('bhr,hdr->bhd', recipe=[1, 1, 128])"),
])
def test_an_fp8_einsum_beyond_the_definition_refuses_by_name(over, refusal, served):
    rng = np.random.default_rng(0)
    rep = _derive_mhc(_OProj(rng, **over), (_tensor(W.bf16(_f32(rng, 3, 4, 128))),))
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported



class _ClampedSwiGlu(nn.Module):
    """`SiluAndMulWithClamp.forward_cuda`: `out` [T, I] allocated, then `_C.silu_and_mul_with_clamp(out, x, limit, alpha, beta)`."""

    def __init__(self, limit: float = 10.0, alpha: float = 1.0, beta: float = 0.0) -> None:
        super().__init__()
        self.limit, self.alpha, self.beta = limit, alpha, beta

    def forward(self, x):
        out = torch.empty(*x.shape[:-1], x.shape[-1] // 2, dtype=x.dtype)
        iface.silu_and_mul_with_clamp(out, x, self.limit, self.alpha, self.beta)
        return out


def test_the_clamped_swiglu_derives_per_row_word_for_word_the_twin(served):
    rng = np.random.default_rng(5)
    gu = W.bf16(_f32(rng, 3, 2 * 24, sd=8.0))
    gu[0, 0], gu[1, 30], gu[2, 5] = 0x7FC0, 0xFF80, 0x4120
    rep = _derive_mhc(_ClampedSwiGlu(), (_tensor(gu),))
    assert rep.program is not None, rep.unsupported
    (app,) = [a for a in rep.provenance["applications"] if a["rule"] == "SiluAndMulWithClamp[vllm]_v1"]
    assert app["emitted"] == "SiluMulClamp_v1{I=24,LIMIT=10.0} x3"
    assert _run(rep, _ClampedSwiGlu(), (_tensor(gu),)) == clamp_twin(gu, 10.0).reshape(-1).tolist()


@pytest.mark.parametrize("over, refusal", [
    ({"alpha": 1.702, "beta": 1.0}, "alpha 1.702, beta 1.0; SiluMulClamp_v1 is the clamp at alpha 1.0, beta 0.0"),
    ({"limit": 10.1}, "LIMIT must be a positive finite bf16 value, got 10.1"),
])
def test_a_clamped_swiglu_beyond_the_definition_refuses_by_name(over, refusal, served):
    rep = _derive_mhc(_ClampedSwiGlu(**over), (torch.empty(3, 48, dtype=torch.bfloat16, device="meta"),))
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


class _HashRouter(nn.Module):
    """`fused_topk_bias` on a hash layer: `ops.topk_hash_softplus_sqrt` over the gate's f32 logits, the tokens and `gate.tid2eid`."""

    def __init__(self, table: np.ndarray, scored: bool = False, renormalize: bool = True) -> None:
        super().__init__()
        self.tid2eid = nn.Parameter(torch.from_numpy(table.astype(np.int32)), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(16), requires_grad=False)
        self.scored, self.renormalize = scored, renormalize

    def forward(self, logits, input_ids):
        M = logits.shape[0]
        w, ids, tei = torch.empty(M, 6), torch.empty(M, 6, dtype=torch.int32), torch.empty(M, 6, dtype=torch.int32)
        table, bias = (None, self.bias) if self.scored else (self.tid2eid, None)
        iface.topk_softplus_sqrt(w, ids, tei, logits, self.renormalize, 2.5, bias, input_ids, table, None)
        return w, ids


def _hash_inputs(rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.stack([rng.permutation(16)[:6] for _ in range(10)])
    return table, _f32(rng, 3, 16, sd=4.0), np.array([7, 2, 9], np.int32)


def test_a_hash_layers_router_derives_per_token_word_for_word_the_twin_over_the_tables_ids(served):
    table, x, toks = _hash_inputs(np.random.default_rng(3))
    model, inputs = _HashRouter(table), (_tensor(x), torch.from_numpy(toks))
    rep = _derive_mhc(model, inputs)
    assert rep.program is not None, rep.unsupported
    (app,) = [a for a in rep.provenance["applications"] if a["rule"] == "TopkSoftplusSqrtHash[vllm-moe]_v1"]
    assert app["emitted"] == "(GatherI32x10_v1 x6 + MoeRouterHash_v1{E=16,SCALE=2.5}) x3"
    assert rep.provenance["weight_roles"]["tid2eid"]["role"] == "router.tid2eid"
    ids = table[toks]
    assert _run(rep, model, inputs) == hash_twin(x, ids, 2.5).reshape(-1).tolist() + ids.reshape(-1).tolist()


@pytest.mark.parametrize("over, refusal", [
    ({"scored": True}, "topk_softplus_sqrt without a hash table (the CUDA scored router)"),
    ({"renormalize": False}, "renormalize=False (MoeRouterHash_v1 renormalises)"),
])
def test_a_router_beyond_the_hash_definition_refuses_by_name(over, refusal, served):
    table, x, toks = _hash_inputs(np.random.default_rng(3))
    rep = _derive_mhc(_HashRouter(table, **over), (_tensor(x), torch.from_numpy(toks)))
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


class _ScoredRouter(nn.Module):
    """`dsv4_topk` on a scored layer: `_dsv4_topk_kernel[(M,)]` over the gate's f32 logits and `e_score_correction_bias`, launched
    as the export captures it."""

    def __init__(self, bias: np.ndarray, src: str = DSV4_TOPK_PINS["src_sha256"], has_vl: bool = False) -> None:
        super().__init__()
        self.e_score_correction_bias = nn.Parameter(_tensor(bias), requires_grad=False)
        self.src, self.has_vl = src, has_vl

    def forward(self, logits, input_ids):
        (M, E), w, ids = logits.shape, torch.empty(logits.shape[0], 6), torch.empty(logits.shape[0], 6, dtype=torch.int32)
        scalars = {"routed_scaling_factor": 2.5, "bias_vl_ptr": None, "image_sentinel_lo": 0, "NUM_EXPERTS": E,
                   "BLOCK_N": 1 << (E - 1).bit_length(), "HAS_VL": self.has_vl, "launch_pdl": True}
        meta = {"kernel": DSV4_TOPK_PINS["kernel"], "src_sha256": self.src, "grid": [M], "scalars": scalars, "geometry": {"num_warps": 1},
                "tensor_args": ["gating_output_ptr", "correction_bias_ptr", "topk_weights_ptr", "topk_ids_ptr", "input_ids_ptr"]}
        triton_launch(meta["kernel"], [logits, self.e_score_correction_bias, w, ids, input_ids], json.dumps(meta))
        return w, ids


def _scored_inputs(rng, E: int = 384) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two rows of logits under one bias: a NaN logit, and two experts tied for the largest current."""
    x, b = _f32(rng, 2, E, sd=4.0), _f32(rng, E, sd=0.1)
    x[0, 5], x[1, 7], x[1, 9], b[7], b[9] = 0x7FC00000, *np.array([30.0, 30.0, 0.0, 0.0], np.float32).view(np.uint32)
    return x, b, np.array([7, 2], np.int32)


def test_a_scored_layers_router_derives_per_token_word_for_word_the_twin(served):
    x, b, toks = _scored_inputs(np.random.default_rng(4))
    model, inputs = _ScoredRouter(b), (_tensor(x), torch.from_numpy(toks))
    rep = _derive_mhc(model, inputs)
    assert rep.program is not None, rep.unsupported
    (app,) = [a for a in rep.provenance["applications"] if a["rule"] == "Dsv4Topk[vllm-moe]_v1"]
    assert app["emitted"] == "MoeRouterSqrtSoftplus_v1{E=384,SCALE=2.5} x2"
    assert rep.provenance["weight_roles"]["e_score_correction_bias"]["role"] == "router.e_score_correction_bias"
    w, ids = router_twin(x, b, 2.5)
    assert ids[1, 0] == 7
    assert _run(rep, model, inputs) == w.reshape(-1).tolist() + ids.reshape(-1).astype(np.uint32).tolist()


@pytest.mark.parametrize("over, E, refusal", [
    ({"src": "00" * 32}, 384, "differs from the profile pin"),
    ({}, 256, "constexpr NUM_EXPERTS=256 differs from the profile pin 384"),
    ({"has_vl": True}, 384, "constexpr HAS_VL=True differs from the profile pin False"),
])
def test_a_scored_router_launch_beyond_the_ptx_refuses_by_name(over, E, refusal, served):
    x, b, toks = _scored_inputs(np.random.default_rng(4), E)
    rep = _derive_mhc(_ScoredRouter(b, **over), (_tensor(x), torch.from_numpy(toks)))
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


#: E = 3 experts of hidden 128 and intermediate 128 (w13 N 256), top 2, at M = 3 tokens: the configs a capture would pin
MARLIN_E, MARLIN_H, MARLIN_I, MARLIN_TOPK, MARLIN_M = 3, 128, 128, 2, 3
MARLIN_SMALL = {(MARLIN_E, MARLIN_TOPK, MARLIN_H, 2 * MARLIN_I, MARLIN_TOPK): {MARLIN_M: (64, 128, 1)},
                (MARLIN_E, MARLIN_TOPK, MARLIN_I, MARLIN_H, 1): {MARLIN_M: (64, 128, 2)}}


class _MarlinLaunches(nn.Module):
    """`fused_marlin_moe`'s align and two expert GEMMs as `mxfp4_marlin_moe_leaf` exports them, the w2 launch over its own input
    rows: the layer's codes packed two a byte as the checkpoint stores them, and its scale bytes 118..127 in u8 Parameters as
    `Mxfp4MoEMethod` holds them."""

    def __init__(self, rng, block_m: int = 8) -> None:
        super().__init__()
        E, H, I = MARLIN_E, MARLIN_H, MARLIN_I
        self.codes13, self.codes2 = rng.integers(0, 16, (E, 2 * I, H)), rng.integers(0, 16, (E, H, I))
        self.bytes13, self.bytes2 = rng.integers(118, 128, (E, 2 * I, H // 32)), rng.integers(118, 128, (E, H, I // 32))
        p = lambda a: nn.Parameter(torch.from_numpy(a.astype(np.uint8)), requires_grad=False)
        self.w13, self.w2 = p(MR.pack_e2m1(self.codes13)), p(MR.pack_e2m1(self.codes2))
        self.s13, self.s2 = p(self.bytes13), p(self.bytes2)
        self.block_m = block_m

    def forward(self, x, ids, key, weights, act):
        blk, par = iface.marlin_moe_align(ids, key, MARLIN_E, self.block_m)
        up = iface.marlin_moe_mxfp4_gemm(x, self.w13, self.s13, ids, blk, par, weights, self.block_m, MARLIN_TOPK, False)
        down = iface.marlin_moe_mxfp4_gemm(act, self.w2, self.s2, ids, blk, par, weights, self.block_m, 1, True)
        return blk, par, up, down


def _align(ids: np.ndarray, key: np.ndarray, E: int, BM: int) -> tuple[np.ndarray, int]:
    """`moe_align_block_size` with each expert's pairs sorted by (key, pair): each pair's sorted position / BM, and the padded count
    / BM."""
    blk, start = np.zeros(len(ids), np.int64), 0
    for e in range(E):
        for r, p in enumerate(sorted(np.flatnonzero(ids == e), key=lambda q: (key[q], q))):
            blk[p] = (start + r) // BM
        start += -(-int((ids == e).sum()) // BM) * BM
    return blk, start // BM


def _marlin_inputs(rng) -> tuple:
    P = MARLIN_M * MARLIN_TOPK
    ids = np.stack([rng.permutation(MARLIN_E)[:MARLIN_TOPK] for _ in range(MARLIN_M)]).astype(np.int32)
    key = rng.permutation(P).reshape(MARLIN_M, MARLIN_TOPK).astype(np.int32)
    return (W.bf16(_f32(rng, MARLIN_M, MARLIN_H)), ids, key, _f32(rng, MARLIN_M, MARLIN_TOPK), W.bf16(_f32(rng, P, MARLIN_I)))


def _derive_marlin(model, xs, monkeypatch, served=MARLIN_SMALL):
    monkeypatch.setattr(vllm_marlin_moe, "MARLIN_SERVED", served)
    to = lambda a: torch.from_numpy(a) if a.dtype == np.int32 else _tensor(a)
    inputs = tuple(to(a) for a in xs)
    return derive_report(model, inputs, mode="profile", profile=PROFILE_B1_EAGER_V3.name, name="marlin", rules=vllm_ruleset()), inputs


@pytest.mark.slow
def test_the_marlin_moes_align_and_expert_gemms_derive_per_pair_word_for_word_the_twin(served, monkeypatch):
    rng = np.random.default_rng(11)
    model, xs = _MarlinLaunches(rng), _marlin_inputs(rng)
    rep, inputs = _derive_marlin(model, xs, monkeypatch)
    assert rep.program is not None, rep.unsupported
    apps = {a["rule"]: a for a in rep.provenance["applications"] if "vllm-marlin-moe" in a["rule"]}
    assert apps["MarlinMoeAlign[vllm-marlin-moe]_v2"]["emitted"] == "MoeAlignRanks_v1{E=3,P=6,BM=8} + MoeAlignBlocks_v1{P=6,BM=8}"
    assert apps["MarlinMoeAlign[vllm-marlin-moe]_v2"]["schedule"] == "marlin_moe_align"
    gemms = [a for a in rep.provenance["applications"] if a["rule"] == "MarlinMoeGemm[vllm-marlin-moe]_v1"]
    assert [(a["preconditions_checked"]["TK"], a["preconditions_checked"]["TN"], a["preconditions_checked"]["GRID"]) for a in gemms] == [
        (64, 128, 132), (64, 128, 264)]
    assert rep.provenance["weight_roles"]["w13"]["role"] == "moe.expert_weight_e2m1x2"
    x, ids, key, g, act = xs
    blk, par = _align(ids.reshape(-1), key.reshape(-1), MARLIN_E, 8)
    e, pair = ids.reshape(-1), np.arange(MARLIN_M * MARLIN_TOPK)
    up = MR.launch(x[pair // MARLIN_TOPK], blk, par, e, model.codes13, model.bytes13, np.arange(2 * MARLIN_I), TK=64, TN=128, GRID=132)
    down = MR.launch(act, blk, par, e, model.codes2, model.bytes2, np.arange(MARLIN_H), TK=64, TN=128, GRID=264, g=g.reshape(-1))
    assert _run(rep, model, inputs) == blk.tolist() + [par] + up.reshape(-1).tolist() + down.reshape(-1).tolist()


@pytest.mark.parametrize("block_m, sweep, refusal", [
    (16, MARLIN_SMALL, "moe_block_size 16 is not fused_marlin_moe's at 3 tokens over 3 experts (8)"),
    (8, {}, "no Marlin capture of a launch at E=3, TOPK=2, K=128, N=256, top_k=2 (MARLIN_SERVED)"),
    (8, {k: {460: c[MARLIN_M]} for k, c in MARLIN_SMALL.items()}, "the capture's sweep does not determine the Marlin config at 3 tokens"),
])
def test_a_marlin_launch_the_capture_does_not_give_refuses_by_name(block_m, sweep, refusal, served, monkeypatch):
    rng = np.random.default_rng(11)
    rep, _ = _derive_marlin(_MarlinLaunches(rng, block_m=block_m), _marlin_inputs(rng), monkeypatch, sweep)
    assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


@pytest.mark.parametrize("batch, refusal", [([[MARLIN_M, 0]], None),
                                            ([[MARLIN_M, 0], [1, 9]], "the launch routes 4 tokens, 1 of them other requests'")])
def test_a_marlin_launch_that_routes_other_requests_tokens_refuses(batch, refusal, served, monkeypatch):
    """Alone in its launch the request's MoE derives as before; beside another request's token the align sorts that token's pairs
    too, which no input of this Program holds."""
    rng = np.random.default_rng(11)
    model = _Launched(_MarlinLaunches(rng), [batch])
    rep, _ = _derive_launched(lambda m, xs: _derive_marlin(m, xs, monkeypatch), model, _marlin_inputs(rng))
    if refusal is None:
        assert rep.program is not None, rep.unsupported
    else:
        assert rep.program is None and refusal in rep.unsupported.hint, rep.unsupported


class _Fp8Linear(nn.Module):
    def __init__(self, scale_dtype, g: int = 0) -> None:
        super().__init__()
        self.weight = _meta(2 * 128, 256, dtype=torch.float8_e4m3fn)
        self.weight_scale_inv = _meta(2, 2, dtype=scale_dtype)
        if g:
            self.is_bmm, self.bmm_batch_size = True, g


def test_the_block_fp8_linears_have_the_scales_and_shapes_the_loaders_post_process_leaves():
    model = nn.ModuleDict({"q": _Fp8Linear(torch.float8_e8m0fnu), "wo_a": _Fp8Linear(torch.float8_e8m0fnu, g=2),
                           "f32": _Fp8Linear(torch.float32)})
    f32 = model["f32"].weight_scale_inv
    fact = fp8_block_weights_as_the_loader_does(model)
    assert (fact["upcast"], fact["bmm"]) == (["q", "wo_a"], ["wo_a"]) and "deepgemm_post_process_fp8_weight_block" in fact["rule"]
    shapes = {n: (tuple(p.shape), p.dtype) for n, p in model.named_parameters()}
    assert shapes["q.weight_scale_inv"] == ((2, 2), torch.float32) and shapes["q.weight"] == ((256, 256), torch.float8_e4m3fn)
    assert shapes["wo_a.weight"] == ((2, 128, 256), torch.float8_e4m3fn) and shapes["wo_a.weight_scale_inv"] == ((2, 1, 2), torch.float32)
    assert model["f32"].weight_scale_inv is f32
    assert fp8_block_weights_as_the_loader_does(nn.Linear(2, 2)) == {"upcast": [], "bmm": []}
