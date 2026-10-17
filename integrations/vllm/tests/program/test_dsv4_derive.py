"""The Build of DeepSeek-V4-Pro on the H200, on synthetic modules that make the calls vLLM's DeepSeek-V4 makes:

* `export_compat.cuda_streams_inert`: the aux streams and events DeepSeek-V4 makes in `__init__` (no driver on a GPU-free Build
  host) are inert, and a forward that overlaps on them exports its functions in the order the engine launches them;
* `vllm_meta.post_load_tensors_as_the_loader_does`: a tensor the model-level post-load hook makes (`hc_attn_fn_broadcast`) is a
  Parameter under its served name;
* `export_compat.registered_op_bodies_as_ops`: the decoder layer calls the Python bodies of vLLM's mHC ops, which export as the ops;
* `export_compat.mhc_pre_broadcast_leaf`: layer 0's broadcast pre, which vLLM does not register, exports as the leaf op;
* `dsv4_leaves.dsv4_attention_leaves`: the sparse MLA's cache insert, compressor state and attention export as the leaf ops;
* `export_compat.deep_gemm_fp8_einsum_leaf`: the o-projection's DeepGEMM einsum exports as the leaf op;
* `marlin_moe_leaf.mxfp4_marlin_moe_leaf`: the MXFP4 Marlin MoE exports as its steps, its two expert GEMMs as the leaf op;
* `rules/vllm_mhc.py`: the four mHC ops derive per token at the dispatch's own split counts, word for word the numpy twin's
  (`kernels.mhc_twin`) on the FMA path and the TF32 path, and what the Definitions do not take refuses by name;
* `CublasRouterRule`: a direct `torch.mm(x, W.T, out_dtype=float32)` binds the captured f32-output route;
* `FusedQKvRmsNormRule`: the launch of `_fused_q_kv_rmsnorm_kernel` derives per token word for word its PTX's (`test_dsa_norm`), and
  another kernel body, a block of one element a thread or a stride outside the PTX's specialization refuses;
* `rules/vllm_dsv4.py`: the insert and the compressor state derive per token word for word the numpy twin's
  (`kernels.dsv4_qnorm_insert_twin`, FlashMLA's `fadd`), and statics the Definitions do not take or positions the tables do not
  hold refuse by name; the sparse attention over a request's prefill and decodes, at ratios 4, 128 and 1, derives word for word
  the twins' composition over its histories (the compressors, FlashMLA's gather and prefill, its decode at the planner's splits),
  and a geometry, head padding, selection, history or target the Definitions do not take refuses by name; the o-projection
  input's inverse RoPE and quantisation derives per token and head word for word the reference (`test_dsv4_indexer.head`) in
  the wrapper's [T, G, ...] layout, and another scale layout, clamp, rotary split or target refuses by name;
* `DeepGemmFp8EinsumRule`: the o-projection's `fp8_einsum` over that quant's words and scales derives per token word for word the
  grouped DeepGEMM twin (`kernels.deepgemm_twins`), and another expression or recipe refuses by name;
* `SiluAndMulWithClampRule`: the shared expert's clamped SwiGLU derives per row word for word the twin the capture holds to the
  served kernel (`dsv4_moe_capture_gpu.clamp_twin`), and SwiGLU-OAI's alpha and beta or a limit that is no bf16 value refuses by name;
* `TopkSoftplusSqrtHashRule`: a hash layer's router derives per token word for word the capture's twin over the table's ids
  (`dsv4_moe_capture_gpu.hash_twin`), and the kernel's scored path or another normalisation refuses by name;
* `vllm_meta.fp8_block_weights_as_the_loader_does`: UE8M0 block scales are f32 and an `is_bmm` layer is grouped, as the loader
  leaves them."""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch._library.infer_schema import infer_schema

from verity_vllm.program.frontend import marlin_moe_leaf as mml
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.dsv4_leaves import dsv4_attention_leaves
from verity_vllm.program.frontend.export_compat import (InertEvent, InertStream, cuda_streams_inert, deep_gemm_fp8_einsum_leaf,
                                                        mhc_pre_broadcast_leaf, registered_op_bodies_as_ops)
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules import vllm_dsv4, vllm_mhc
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
from verity_vllm.program.frontend.rules.triton_iface import triton_launch
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.frontend.export_ops import CollectiveBus, TPStub
from verity_vllm.program.frontend.vllm_meta import (ScheduledWithPeers, fp8_block_weights_as_the_loader_does,
                                                    post_load_tensors_as_the_loader_does, with_peers)
from verity_vllm.program.kernels import deepgemm_twins as DT
from verity_vllm.program.kernels import dsv4_compressor_twin as DC
from verity_vllm.program.kernels import dsv4_qnorm_insert_twin as TQ
from verity_vllm.program.kernels import flashmla_prefill_twin as FP
from verity_vllm.program.kernels import flashmla_twin as FT
from verity_vllm.program.kernels import mhc_twin as W
from verity_vllm.program.kernels.serve3_reference import materialize
from verity_vllm.program.registry import hopper
from tests.program.test_dsa_norm import ptx_block_model
from tests.program.test_dsv4_indexer import head as inv_rope_head
from tests.properties.dsv4_moe_capture_gpu import clamp_twin, hash_twin


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
        self.w13_scale, self.w2_scale = _meta(4, 64, 2, dtype=torch.float8_e8m0fnu), _meta(4, 64, 1, dtype=torch.float8_e8m0fnu)

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
             torch.float8_e4m3fn: (torch.uint8, np.uint8)}[t.dtype]
    return t.contiguous().view(width[0]).numpy().reshape(-1).astype(width[1]).tolist()


def _run(rep, model: nn.Module, inputs) -> list[int]:
    held = dict(model.named_parameters())
    weights = [v for w in rep.provenance["weights"] for v in _words(held[w["name"]])]
    t = materialize(rep.program, [_words(x) for x in inputs] + ([weights] if rep.provenance["weights"] else []))
    return [int(t[g]) for g in rep.program.circuit.root_outputs()]


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


def _twin(model: _Hyper, x: np.ndarray, NS: int, NS0: int) -> list[int]:
    """The model's output as the twin composes the kernels at T tokens: the FMA path at most 16 tokens, the TF32 path above."""
    u = lambda t: np.array(_words(t), np.uint16 if t.dtype == torch.bfloat16 else np.uint32).reshape(t.shape)
    m = {k: u(v) for k, v in model.named_parameters()}
    TN, TH, TH_HEAD = GEOMETRY["TN"], GEOMETRY["TH"], GEOMETRY["TH_HEAD"]
    h, post, comb = W.pre_broadcast(x, m["fn0_broadcast"], m["scale0"], m["base0"], m["norm0"], NS0, TN, REPEAT, EPS, EPS)
    streams = np.broadcast_to(x[:, None, :], (len(x), HC, HID))
    args = (m["fn1"], m["scale1"], m["base1"], m["norm1"])
    if len(x) <= vllm_mhc.SMALL_FMA_TOKENS:
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
    """DeepSeek-V4's compressor GEMM: a direct `torch.mm(x, W.T, out_dtype=float32)`, W [1024, 7168] bf16."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = _meta(1024, 7168)

    def forward(self, x):
        return torch.mm(x, self.weight.T, out_dtype=torch.float32)


def test_a_direct_f32_output_mm_binds_the_captured_f32out_route(served):
    rep = _derive_mhc(_F32Out(), (torch.empty(4, 7168, dtype=torch.bfloat16, device="meta"),))
    assert rep.program is not None, rep.unsupported
    want = PROFILE_B1_EAGER_V3.kind("cublas_splitk_gemm_f32out", K=7168, N=1024, SPLIT_K=5, DOT=hopper.HopperBF16WgmmaDot16).id
    assert [n.fn.id for n in rep.program.fn.body.nodes] == [want] * 4
    assert rep.provenance["weight_roles"]["weight"]["role"] == "linear.weight"
    rep = _derive_mhc(_F32Out(), (torch.empty(8, 7168, dtype=torch.bfloat16, device="meta"),))
    assert "no captured vLLM-default cuBLAS f32out route for M=8, K=7168, N=1024" in rep.unsupported.hint


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
    """A DeepSeek-V4 attention layer's leaves over one request, a call of each length in ``calls`` from position p0: 2 live heads
    padded to NHP, head 64 of which 16 rotary, quant block 16; the compressor's state 2D wide at ratio 4, D at 128, none at 1."""

    def __init__(self, rng, ratio: int, window: int, calls: tuple[int, ...], topk: int = 1024, NHP: int = 64, p0: int = 0) -> None:
        super().__init__()
        p = lambda words: nn.Parameter(_tensor(words), requires_grad=False)
        self.ratio, self.window, self.calls, self.topk, self.NHP, self.p0 = ratio, window, calls, topk, NHP, p0
        self.cos_sin_cache, self.attn_sink = p(TQ._cs(rng, (p0 + sum(calls), SPARSE_DR))), p(_f32(rng, NHP))
        if ratio > 1:
            self.ape = p(_f32(rng, ratio, SPARSE_D * (2 if ratio == 4 else 1), sd=0.1))
            self.norm_weight = p(W.bf16(_f32(rng, SPARSE_D)))

    def forward(self, *xs):
        k, outs, p0 = 3 if self.ratio > 1 else 2, [], self.p0
        for call in zip(*(xs[i::k] for i in range(k))):
            q, kv = call[-2:]
            positions = torch.arange(p0, p0 + q.shape[0], dtype=torch.int64)
            p0 += q.shape[0]
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


def _sparse_twin(model: _SparseAttn, xs: list[np.ndarray]) -> list[int]:
    """The layer's outputs as the twins compose the kernels over its window, state and compressed histories."""
    u = lambda t: np.array(_words(t), np.uint16 if t.dtype == torch.bfloat16 else np.uint32).reshape(t.shape)
    m = {k: u(v) for k, v in model.named_parameters()}
    ratio, window, cs, sink = model.ratio, model.window, m["cos_sin_cache"], m["attn_sink"][:2]
    D, DN, QB, BN, k = SPARSE_D, SPARSE_D - SPARSE_DR, SPARSE_DR, SPARSE_BN, 3 if model.ratio > 1 else 2
    stack = lambda rows: [np.stack(c) for c in zip(*rows)]
    words = lambda v: np.array(v, np.int64).astype(np.uint32)
    win, state, comp, outs, p0 = [], [], [], [], 0
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
                taken = [(r, s) for r, s in enumerate(range(p - (8 if ratio == 4 else 128) + 1, p + 1)) if s >= 0]
                kvr, scr = (np.stack([state[s][j][r // ratio * D:(r // ratio + 1) * D] for r, s in taken]) for j in (0, 1))
                args = (kvr, scr, m["norm_weight"], cs[p - ratio + 1])
                row = DC.compress_c4(*args, QB, EPS) if ratio == 4 else DC.compress_c128(*args, 8, 16, QB, EPS)
                n, ro, e = (a.reshape(-1) for a in row)
                comp.append((n, e[:DN // QB], ro))
        for i, p in enumerate(pos):
            s0, nc = max(p - window + 1, 0), (p + 1) // ratio if ratio > 1 else 0
            if len(q) == 1:
                swa = p - s0 + 1
                NW, N = FT.request_blocks(swa, nc, BN)
                rows = win[s0:p + 1] + win[s0:s0 + 1] * (NW - swa) + comp[:nc] + win[s0:s0 + 1] * (N - NW - nc)
                idx = words([0] * swa + [-1] * (NW - swa) + [0] * nc + [-1] * (N - NW - nc))
                metas, ns = FT.sched_meta([swa], [nc] if ratio > 1 else None, topk=window, extra_topk=int(ratio > 1),
                                          parts=FT.num_sm_parts(model.NHP), block_size_n=BN)
                outs.append(FT.decode(qp[i, :2], *stack(rows), idx, [swa] + ([nc] if N > NW else []), sink,
                                      SPLITS=FT.request_splits(metas, ns, 0, N // BN), NW=NW, QT=QB, BN=BN, SCALE=SPARSE_SCALE))
            else:
                rows = comp[:nc] + win[s0:p + 1]
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


@pytest.mark.parametrize("ratio, window, calls, paths", [
    (4, 28, (30, 1, 1), [("prefill", list(range(7))), ("decode", []), ("decode", [7])]),
    (128, 8, (128, 1), [("prefill", [0]), ("decode", [])]),
    (1, 8, (12, 1), [("prefill", []), ("decode", [])]),
])
def test_the_sparse_attention_derives_over_its_histories_word_for_word_the_twins(ratio, window, calls, paths, flashmla_small):
    rng = np.random.default_rng(ratio)
    model = _SparseAttn(rng, ratio, window, calls)
    xs = _sparse_inputs(rng, model)
    rep = _derive_mhc(model, tuple(map(_tensor, xs)))
    assert rep.program is not None, rep.unsupported
    apps = [a["preconditions_checked"] for a in rep.provenance["applications"] if a["rule"] == "Dsv4SparseAttn[vllm-dsv4]_v1"]
    assert [(a["path"], a["compressed_rows"]) for a in apps] == paths
    if ratio == 4:
        assert len(apps[2]["SPLITS"]) > 2, "the last decode combines its splits"
    roles = {n: r["role"] for n, r in rep.provenance["weight_roles"].items()}
    assert roles["attn_sink"] == "dsv4.attn_sink" and roles.get("norm_weight", "dsv4.compressor.norm.weight") == "dsv4.compressor.norm.weight"
    assert _run(rep, model, tuple(map(_tensor, xs))) == _sparse_twin(model, xs)


@pytest.mark.parametrize("over, refusal", [
    ({"geometry": None}, "no FlashMLA sparse geometry at head 64, rope 16"),
    ({"NHP": 4}, "64 or 128 padded heads, at least one live (2 of 4)"),
    ({"topk": 4}, "position 29 has 7 compressed rows, more than the indexer's top-4"),
    ({"p0": 5}, "positions 5..34 are not the 30 after 'layers.0.attn''s history of 0 rows from position 0"),
    ({"target": TargetProfile(compute_capability=(12, 0), num_sms=188, batch_invariant=False, moe_backend="triton")},
     "the FlashMLA Definitions are its cc 9.0 kernels"),
])
def test_a_sparse_attention_beyond_the_definitions_refuses_by_name(over, refusal, flashmla_small, monkeypatch):
    rng = np.random.default_rng(0)
    if "geometry" in over:
        monkeypatch.delitem(vllm_dsv4.FLASHMLA_GEOMETRY, (SPARSE_D, SPARSE_DR))
    if "target" in over:
        VB.observe_target_profile(over["target"])
    model = _SparseAttn(rng, 4, 28, (30,), **{k: v for k, v in over.items() if k in ("NHP", "topk", "p0")})
    rep = _derive_mhc(model, tuple(map(_tensor, _sparse_inputs(rng, model))))
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
