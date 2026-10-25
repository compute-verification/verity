"""The SM partition (`topup.cu`'s `sq_green`, green contexts): the filler's F SMs and the step's rest, made before the
engine starts, for `serve.py --partition` and the bandwidth arms' `topup_sms` (`vllm_encoded/bench_arm.py`).

The step's stream is torch's current stream from then on, so vLLM launches its eager kernels there; every CUDA graph is
captured on a stream of the step's context, whatever stream its caller names (vLLM captures on one it makes itself, on
the device's context), and replays on the step's; and the device's SM count reads as the step's
(`multi_processor_count`, which sizes the stack's and vLLM's grids). `spread_checks` checks the confinement. This module
imports nothing of its directory's, so a caller that has the encoded stack's `bench` on its path can import it.
"""
from __future__ import annotations

import ctypes


class StepProps:
    """The device's properties with the step's SM count: what a kernel launched on the step's green context can use."""

    def __init__(self, props, sms: int) -> None:
        self._props, self.multi_processor_count = props, sms

    def __getattr__(self, name):
        return getattr(self._props, name)


def partition(tu, torch, fill_sms: int):
    """The green contexts (`sq_green` through `tu`, a `bench.Topup`), the step's stream made torch's current one, and
    the device's SM count read as the step's; returns the record, the step's stream and the stack's side-stream
    factory (`new_stream(0)` on the step's context, `new_stream(1)` on the filler's)."""
    v, i = ctypes.c_void_p, ctypes.c_int
    tu.so.sq_green.argtypes = [i, ctypes.POINTER(v), i, ctypes.POINTER(i)]
    tu.so.sq_green_stream.argtypes = [i, i, ctypes.POINTER(v)]
    tu.so.sq_smid.argtypes = [v, i, ctypes.c_ulonglong, v]
    s, got = v(), (i * 4)()
    if rc := tu.so.sq_green(fill_sms, ctypes.byref(s), 0, got):
        raise RuntimeError(f"sq_green({fill_sms}): {rc}")
    rec = {"fill_sms": got[0], "step_sms": got[1], "min_partition": got[2], "alignment": got[3],
           "device_sms": torch.cuda.get_device_properties(0).multi_processor_count}
    step = torch.cuda.ExternalStream(s.value)
    torch.cuda.set_stream(step)
    gdp = torch.cuda.get_device_properties
    torch.cuda.get_device_properties = lambda *x, **kw: StepProps(gdp(*x, **kw), got[1])

    def new_stream(which: int = 0):
        p = v()
        if rc := tu.so.sq_green_stream(which, 0, ctypes.byref(p)):
            raise RuntimeError(f"sq_green_stream: {rc}")
        return torch.cuda.ExternalStream(p.value)

    on = new_stream(0)

    class graph(torch.cuda.graph):
        def __init__(self, cuda_graph, pool=None, stream=None, **kwargs):
            super().__init__(cuda_graph, pool=pool, stream=on, **kwargs)

    torch.cuda.graph = graph
    return rec, step, new_stream


def spread_checks(tu, torch, step, new_stream, rec: dict) -> dict:
    """The SMs that 4 CTAs an SM ran on: on the step's stream, directly and from a graph captured there; on a side
    stream of the stack's, directly and from a graph captured there and replayed on the step's (vLLM's graphs, as
    `partition` captures them); and on a stream of the filler's context. The step's must be within its count.
    ``device_graph`` is a graph captured on a stream of the device's context and replayed on the step's (vLLM's
    capture without `partition`'s): recorded, not required."""
    n = 4 * rec["device_sms"]
    tu.so.sq_smid.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_void_p]

    def sms(s, captured=None) -> int:
        out = torch.full((n,), -1, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        if captured is not None:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(captured):
                g.capture_begin()
                tu.so.sq_smid(out.data_ptr(), n, 20000, captured.cuda_stream)
                g.capture_end()
            with torch.cuda.stream(s):
                g.replay()
        elif tu.so.sq_smid(out.data_ptr(), n, 20000, s.cuda_stream):
            raise RuntimeError("sq_smid")
        torch.cuda.synchronize()
        return int(torch.unique(out).numel())

    side = new_stream(0)
    got = {"step": sms(step), "step_graph": sms(step, step), "side": sms(side), "side_graph": sms(step, side),
           "fill": sms(new_stream(1)), "device_graph": sms(step, torch.cuda.Stream())}
    got["confined"] = (max(got["step"], got["step_graph"], got["side"], got["side_graph"]) <= rec["step_sms"]
                       and got["fill"] <= rec["fill_sms"])
    return got
