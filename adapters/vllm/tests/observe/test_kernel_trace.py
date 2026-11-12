"""`verity_vllm.observe.kernel_trace`: the device kernels of a Kineto Chrome trace in launch order, their summary by name, and
the trace file's two forms."""
from __future__ import annotations

import gzip
import json
from typing import Any

from verity_vllm.observe import kernel_trace as KT


def _event(name: str, ts: int, dur: float = 1.0, cat: str = "kernel", ph: str = "X", **more: Any) -> dict[str, Any]:
    return {"name": name, "ts": ts, "dur": dur, "cat": cat, "ph": ph, **more}


def test_the_kernels_of_a_trace_are_its_device_events_in_launch_order():
    events = [_event("aten::mm", 0, cat="cpu_op"), _event("gemm", 5), _event("add", 5), _event("Memcpy HtoD", 1, cat="gpu_memcpy"),
              _event("Memset", 9, cat="gpu_memset"), _event("gemm", 7, ph="i")]
    assert [(e["name"], e["ts"]) for e in KT.kernels_of_trace_events(events)] == [("Memcpy HtoD", 1), ("add", 5), ("gemm", 5), ("Memset", 9)]


def test_the_summary_counts_each_kernel_and_keeps_its_first_launch():
    kernels = [_event("gemm", 0, 1.2, args={"grid": [4, 1, 1], "block": [128, 1, 1]}),
               _event("Memcpy HtoD", 1, 0.5, cat="gpu_memcpy"),
               _event("gemm", 2, 2.5, args={"grid": [8, 1, 1], "block": [64, 1, 1]}),
               _event("Memset", 3, 0.3, cat="gpu_memset", args=None)]
    out = KT.summarize_kernels(kernels)
    assert list(out) == ["gemm", "Memcpy HtoD", "Memset"]
    assert out["gemm"] == {"count": 2, "us": 3.7, "grid": [4, 1, 1], "block": [128, 1, 1], "first_index": 0, "cat": "kernel"}
    # an event the profiler gave no `args`, or a null one, has no launch geometry to keep
    assert out["Memcpy HtoD"] == {"count": 1, "us": 0.5, "grid": None, "block": None, "first_index": 1, "cat": "gpu_memcpy"}
    assert out["Memset"] == {"count": 1, "us": 0.3, "grid": None, "block": None, "first_index": 3, "cat": "gpu_memset"}
    empties: tuple[object, ...] = (0, "", [], {})
    for empty in empties:                                          # and so has one whose `args` is empty, whatever its type
        assert KT.summarize_kernels([_event("k", 0, args=empty)])["k"] == {"count": 1, "us": 1.0, "grid": None, "block": None, "first_index": 0, "cat": "kernel"}


def test_a_trace_file_is_an_object_with_its_events_or_the_bare_list(tmp_path):
    events = [_event("gemm", 0)]
    plain, packed = tmp_path / "trace.json", tmp_path / "trace.json.gz"
    plain.write_text(json.dumps(events))
    with gzip.open(packed, "wt") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)
    assert KT.load_trace(plain) == events == KT.load_trace(packed)
