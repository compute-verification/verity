"""The Build of DeepSeek-V4-Pro on the H200, on synthetic modules that make the calls vLLM's DeepSeek-V4 makes:

* `export_compat.cuda_streams_inert`: the aux streams and events DeepSeek-V4 makes in `__init__` (no driver on a GPU-free Build
  host) are inert, and a forward that overlaps on them exports its functions in the order the engine launches them;
* `vllm_meta.post_load_tensors_as_the_loader_does`: a tensor the model-level post-load hook makes (`hc_attn_fn_broadcast`) is a
  Parameter under its served name;
* `export_compat.registered_op_bodies_as_ops`: the decoder layer calls the Python bodies of vLLM's mHC ops, which export as the ops;
* `export_compat.mhc_pre_broadcast_leaf`: layer 0's broadcast pre, which vLLM does not register, exports as the leaf op;
* `dsv4_leaves.dsv4_attention_leaves`: the sparse MLA's cache insert, compressor state and attention export as the leaf ops;
* `export_compat.deep_gemm_fp8_einsum_leaf`: the o-projection's DeepGEMM einsum exports as the leaf op;
* `marlin_moe_leaf.mxfp4_marlin_moe_leaf`: the MXFP4 Marlin MoE exports as its steps, its two expert GEMMs as the leaf op."""
from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn
from torch._library.infer_schema import infer_schema

from verity_vllm.program.frontend import marlin_moe_leaf as mml
from verity_vllm.program.frontend.dsv4_leaves import dsv4_attention_leaves
from verity_vllm.program.frontend.export_compat import (InertEvent, InertStream, cuda_streams_inert, deep_gemm_fp8_einsum_leaf,
                                                        mhc_pre_broadcast_leaf, registered_op_bodies_as_ops)
from verity_vllm.program.frontend.vllm_meta import post_load_tensors_as_the_loader_does


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


@pytest.fixture
def marlin_site(monkeypatch):
    mxfp4, oracle = types.ModuleType(mml._MXFP4), types.ModuleType(mml._ORACLE)
    site, scalar_type = types.ModuleType(mml._MARLIN_MOE), types.ModuleType("vllm.scalar_type")
    mxfp4.Mxfp4MoEMethod = _Mxfp4Method
    oracle.Mxfp4MoeBackend = types.SimpleNamespace(MARLIN="marlin")
    oracle.make_mxfp4_moe_kernel = lambda **kwargs: kwargs
    scalar_type.scalar_types = types.SimpleNamespace(float4_e2m1f=types.SimpleNamespace(id=7))
    site.fused_marlin_moe = _launches
    site.moe_align_block_size = lambda topk_ids, block_m, E, expert_map, ignore_invalid_experts: (
        topk_ids.new_empty((topk_ids.numel() + E * (block_m - 1),)), topk_ids.new_empty((E,)), topk_ids.new_empty((1,)))
    for module in (mxfp4, oracle, site, scalar_type):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return site


def _moe_inputs(M: int = 3) -> tuple:
    return (torch.empty(M, 64, dtype=torch.bfloat16, device="meta"), torch.empty(M, 2, device="meta"),
            torch.empty(M, 2, dtype=torch.int32, device="meta"))


def test_the_mxfp4_marlin_moe_exports_as_its_steps_with_its_expert_gemms_as_leaf_ops(marlin_site):
    model = _MarlinMoe(marlin_site)
    with mml.mxfp4_marlin_moe_leaf(model):
        kernel = model.quant_method.moe_kernel
        ep = torch.export.export(model, _moe_inputs(), strict=False)
    assert model.quant_method.moe_kernel is None and marlin_site.fused_marlin_moe is _launches
    assert kernel["moe_quant_config"] == ("quant_config", (4, 64, 32)) and kernel["routing_tables"] == "routing_tables"
    gemms = [n for n in ep.graph.nodes if str(n.target) == "vllm_iface.marlin_moe_mxfp4_gemm.default"]
    assert [(tuple(n.meta["val"].shape), n.args[-3:]) for n in gemms] == [((6, 64), (8, 2, False)), ((6, 64), (8, 1, True))]


def test_a_marlin_moe_beyond_the_leafs_operands_refuses_at_export(marlin_site):
    model = _MarlinMoe(marlin_site, bias=_meta(4, 64))
    with mml.mxfp4_marlin_moe_leaf(model), pytest.raises(NotImplementedError, match=r"given \['bias1'\]"):
        torch.export.export(model, _moe_inputs(), strict=False)
