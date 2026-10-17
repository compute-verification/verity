"""The Build of DeepSeek-V4-Pro on the H200, on synthetic modules that make the calls vLLM's DeepSeek-V4 makes:

* `export_compat.cuda_streams_inert`: the aux streams and events DeepSeek-V4 makes in `__init__` (no driver on a GPU-free Build
  host) are inert, and a forward that overlaps on them exports its functions in the order the engine launches them;
* `vllm_meta.post_load_tensors_as_the_loader_does`: a tensor the model-level post-load hook makes (`hc_attn_fn_broadcast`) is a
  Parameter under its served name."""
from __future__ import annotations

import torch
import torch.nn as nn

from verity_vllm.program.frontend.export_compat import InertEvent, InertStream, cuda_streams_inert
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
