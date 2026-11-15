"""`verity_vllm.pipeline.tp.rank_worker`: a rank's kernel census of the stretch it profiled
(`TP2CaptureWorkerExtension.tp2_profiler_stop`).  The rank waits for its device, ends its profiler session, reads the trace the
session exports and writes `census.rank<r>.json`: the device kernels by first launch with their counts and time, and the 200
CPU ops called most.  It says which kernels are a collective's, and keeps the trace, gzipped, only when asked and only on the
ranks named.  CPU-only: the session is a stand-in that exports the events a test gives it, and the test sets the rank and
`torch.cuda.synchronize`."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from verity_vllm.pipeline.tp import rank_worker as RW
from verity_vllm.records.schema_ids import CENSUS_RANK_SCHEMA

ALL_REDUCE = "ncclDevKernel_AllReduce_Sum_f32_RING_LL"
#: a trace out of launch order: two launches of one kernel, a collective's kernel, a memcpy, and what is no device kernel
EVENTS: list[dict[str, Any]] = [
    {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 20, "dur": 5.25, "args": {"grid": [4, 1, 1], "block": [32, 1, 1]}},
    {"ph": "X", "cat": "kernel", "name": ALL_REDUCE, "ts": 10, "dur": 2.0},
    {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 30, "dur": 1.0},
    {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 40, "dur": 5.25, "args": {"grid": [8, 1, 1], "block": [64, 1, 1]}},
    {"ph": "X", "cat": "cpu_op", "name": "aten::add", "ts": 1},
    {"ph": "X", "cat": "cpu_op", "name": "aten::linear", "ts": 2, "dur": 9.0},
    {"ph": "X", "cat": "cpu_op", "name": "aten::linear", "ts": 3},
    {"ph": "i", "cat": "cpu_op", "name": "an instant, no span", "ts": 4},
    {"ph": "X", "cat": "python_function", "name": "forward", "ts": 5, "dur": 50.0},
]


class _Session:
    """A profiler session that is open: it ends once and exports ``events`` as a Chrome trace."""

    def __init__(self, events: list[dict[str, Any]], order: list[str]) -> None:
        self.events, self.order = events, order

    def __exit__(self, *exc: Any) -> None:
        assert exc == (None, None, None)
        self.order.append("session ended")

    def export_chrome_trace(self, path: str) -> None:
        self.order.append("trace exported")
        Path(path).write_text(json.dumps({"traceEvents": self.events}))


class _Rank(RW.TP2CaptureWorkerExtension):
    _tp2_prof: Any


def _stop(monkeypatch, out: Path, events: list[dict[str, Any]] = EVENTS, rank: int = 1, **kw: Any) -> tuple[dict[str, Any], list[str]]:
    """(what rank ``rank``'s `tp2_profiler_stop` returns over a session that recorded ``events``, the order of what it did)."""
    order: list[str] = []
    monkeypatch.setattr(RW, "_tp_facts", lambda worker: {"tp": {"rank": rank, "world_size": 2}})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: order.append("device waited for"))
    w = _Rank()
    w._tp2_prof = _Session(events, order)
    return w.tp2_profiler_stop(out_dir=str(out), **kw), order


def test_a_ranks_census_is_its_device_kernels_by_first_launch_and_its_cpu_ops_by_calls(monkeypatch, tmp_path):
    out = tmp_path / "not" / "there" / "yet"
    res, order = _stop(monkeypatch, out)
    assert order == ["device waited for", "session ended", "trace exported"]
    doc = json.loads((out / "census.rank1.json").read_text())
    assert (doc["schema"], doc["rank"], doc["pid"], doc["n_kernels"], doc["device_us"]) == (CENSUS_RANK_SCHEMA, 1, os.getpid(), 4, 13.5)
    assert list(doc["kernels"]) == [ALL_REDUCE, "gemm", "Memcpy HtoD"]
    assert doc["kernels"]["gemm"] == {"count": 2, "us": 10.5, "grid": [4, 1, 1], "block": [32, 1, 1], "first_index": 1, "cat": "kernel"}
    assert doc["kernels"]["Memcpy HtoD"]["cat"] == "gpu_memcpy" and doc["kernels"][ALL_REDUCE]["grid"] is None
    assert list(doc["cpu_ops"].items()) == [("aten::linear", 2), ("aten::add", 1)]
    assert "one torch.profiler.profile" in doc["method"] and sorted(doc) == ["cpu_ops", "device_us", "kernels", "method", "n_kernels", "pid",
                                                                             "rank", "schema"]
    assert res == {"rank": 1, "n_kernels": 4, "distinct": 3, "device_us": 13.5, "collective_kernels": {ALL_REDUCE: 1}}
    assert sorted(p.name for p in out.iterdir()) == ["census.rank1.json"]       # the trace is read and gone


@pytest.mark.parametrize("name", ["ncclDevKernel_SendRecv", "custom_AllReduce", "all_reduce_fused", "AllGather_ring", "cross_device_reduce_2stage",
                                  "reduce_kernel_sum"])
def test_a_kernel_whose_name_holds_a_collectives_word_is_counted_as_a_collectives(monkeypatch, tmp_path, name):
    events = [{"ph": "X", "cat": "kernel", "name": n, "ts": i} for i, n in enumerate([name, "gemm", name])]
    res, _ = _stop(monkeypatch, tmp_path, events)
    assert res["collective_kernels"] == {name: 2} and res["distinct"] == 2


@pytest.mark.parametrize("kw, kept", [({}, False), ({"keep_trace": True}, True), ({"keep_trace": True, "keep_ranks": [1, 3]}, True),
                                      ({"keep_trace": True, "keep_ranks": [0]}, False), ({"keep_trace": True, "keep_ranks": []}, False),
                                      ({"keep_ranks": [1]}, False)],
                         ids=["not asked", "asked", "asked, this rank named", "asked, another rank named", "asked, no rank named",
                              "this rank named, not asked"])
def test_the_trace_is_kept_gzipped_only_when_asked_and_only_on_the_ranks_named(monkeypatch, tmp_path, kw, kept):
    _stop(monkeypatch, tmp_path, **kw)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["census.rank1.json", *(["trace.rank1.json.gz"] if kept else [])]
    if kept:
        with gzip.open(tmp_path / "trace.rank1.json.gz", "rt") as f:
            assert json.load(f) == {"traceEvents": EVENTS}


def test_the_census_keeps_the_200_cpu_ops_called_most(monkeypatch, tmp_path):
    events = [{"ph": "X", "cat": "cpu_op", "name": f"op{i}", "ts": i} for i in range(201)]
    events += [{"ph": "X", "cat": "cpu_op", "name": f"op{i}", "ts": 300 + i} for i in range(201) if i != 7]
    res, _ = _stop(monkeypatch, tmp_path, events, rank=0)
    ops = json.loads((tmp_path / "census.rank0.json").read_text())["cpu_ops"]
    assert len(ops) == 200 and "op7" not in ops and set(ops.values()) == {2}
    assert res == {"rank": 0, "n_kernels": 0, "distinct": 0, "device_us": 0, "collective_kernels": {}}
