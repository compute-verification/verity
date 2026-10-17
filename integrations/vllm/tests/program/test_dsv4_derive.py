"""The Build of DeepSeek-V4-Pro on the H200, on synthetic modules that make the calls vLLM's DeepSeek-V4 makes:

* `export_compat.cuda_streams_inert`: the aux streams and events DeepSeek-V4 makes in `__init__` (no driver on a GPU-free Build
  host) are inert, and a forward that overlaps on them exports its functions in the order the engine launches them;
* `vllm_meta.post_load_tensors_as_the_loader_does`: a tensor the model-level post-load hook makes (`hc_attn_fn_broadcast`) is a
  Parameter under its served name;
* `export_compat.registered_op_bodies_as_ops`: the decoder layer calls the Python bodies of vLLM's mHC ops, which export as the ops;
* `export_compat.mhc_pre_broadcast_leaf`: layer 0's broadcast pre, which vLLM does not register, exports as the leaf op."""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
from torch._library.infer_schema import infer_schema

from verity_vllm.program.frontend.export_compat import (InertEvent, InertStream, cuda_streams_inert, mhc_pre_broadcast_leaf,
                                                        registered_op_bodies_as_ops)
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
