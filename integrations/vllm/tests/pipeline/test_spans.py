"""`Spans.throughput`: per-stage wall time, bytes, leaves and rates, labelled with the committer's scheme (torch-free)."""
from __future__ import annotations

from verity_vllm.commit import step_rows
from verity_vllm.pipeline.spans import COMPONENTS, Spans


def test_throughput_sums_each_stage_and_names_the_scheme():
    sp = Spans(step_rows.NAME)
    sp.add("hashing", 0.5, step=0, nbytes=1024, leaves=4)
    sp.add("hashing", 1.5, step=1, nbytes=3072, leaves=12)
    sp.add("root_finalisation", 0.25, step=0)
    sp.add("padding", 0.0, step=2, nbytes=256, leaves=1)
    tp = sp.throughput()
    assert tp["scheme"] == step_rows.NAME == "rows-sha512"
    assert list(tp["stages"]) == ["hashing", "root_finalisation", "padding"]
    assert tp["stages"]["hashing"] == {"wall_s": 2.0, "bytes": 4096, "leaves": 16, "bytes_per_s": 2048.0, "leaves_per_s": 8.0}
    assert tp["stages"]["root_finalisation"] == {"wall_s": 0.25, "bytes": 0, "leaves": 0, "bytes_per_s": 0.0, "leaves_per_s": 0.0}
    assert tp["stages"]["padding"]["bytes_per_s"] is None and tp["stages"]["padding"]["leaves_per_s"] is None


def test_snapshot_keeps_its_keys_and_adds_leaves_and_throughput():
    sp = Spans()
    sp.add("hashing", 1.0, step=3, nbytes=64, leaves=2)
    snap = sp.snapshot()
    assert set(snap) == {"total_s", "calls", "bytes", "leaves", "per_step", "throughput"}
    assert list(snap["total_s"]) == list(COMPONENTS) and snap["calls"]["hashing"] == 1
    assert snap["bytes"] == {"hashing": 64} and snap["leaves"] == {"hashing": 2}
    assert snap["per_step"] == {3: {"hashing": 1.0}}
    assert snap["throughput"] == {"scheme": None, "stages": {"hashing": {"wall_s": 1.0, "bytes": 64, "leaves": 2, "bytes_per_s": 64.0,
                                                                         "leaves_per_s": 2.0}}}
