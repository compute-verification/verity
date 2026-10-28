from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import green  # noqa: E402


class Fn:
    """A ctypes function as `partition` uses one: called, and given argtypes."""

    def __init__(self, body) -> None:
        self.body = body

    def __call__(self, *args) -> int:
        return self.body(*args)


class FakeLib:
    """topup.cu's sq_green and sq_green_stream: 8 of 188 SMs the filler's, streams numbered by context."""

    def __init__(self) -> None:
        self.made: list[int] = []
        self.sq_green, self.sq_green_stream, self.sq_smid = Fn(self._green), Fn(self._stream), Fn(lambda *a: 0)

    def _green(self, fill, stream, prio, got) -> int:
        stream._obj.value = 0x5000
        got[0], got[1], got[2], got[3] = fill, 188 - fill, 8, 8
        return 0

    def _stream(self, which, prio, out) -> int:
        self.made.append(which)
        out._obj.value = 0x1000 * (which + 1) + len(self.made)
        return 0


def _torch():
    """torch.cuda as `partition` touches it: streams by pointer, the current stream, properties and graphs."""

    class Stream:
        def __init__(self, ptr: int) -> None:
            self.cuda_stream = ptr

    class graph:
        def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode="global") -> None:
            self.cuda_graph, self.pool, self.capture_stream = cuda_graph, pool, stream

    cuda = SimpleNamespace(ExternalStream=Stream, graph=graph, current=None,
                           get_device_properties=lambda *a: SimpleNamespace(multi_processor_count=188, name="sm_120"))
    cuda.set_stream = lambda s: setattr(cuda, "current", s)
    return SimpleNamespace(cuda=cuda)


def test_partition_makes_the_steps_stream_current_and_captures_every_graph_on_the_steps_context():
    """vLLM captures its decode graphs on a stream it makes itself, on the device's context: the capture moves to a
    stream of the step's context (sq_green_stream(0, ...)), whatever stream the caller names."""
    lib, torch = FakeLib(), _torch()
    rec, step, new_stream = green.partition(SimpleNamespace(so=lib), torch, 8)
    assert rec == {"fill_sms": 8, "step_sms": 180, "min_partition": 8, "alignment": 8, "device_sms": 188}
    assert torch.cuda.current is step and step.cuda_stream == 0x5000
    props = torch.cuda.get_device_properties(0)
    assert props.multi_processor_count == 180 and props.name == "sm_120"
    vllms = torch.cuda.ExternalStream(0x9999)
    captured = torch.cuda.graph("g", pool="p", stream=vllms)
    assert lib.made == [0] and captured.capture_stream.cuda_stream == 0x1001 and captured.pool == "p"
    assert new_stream(1).cuda_stream // 0x1000 == 2 and lib.made == [0, 1]
