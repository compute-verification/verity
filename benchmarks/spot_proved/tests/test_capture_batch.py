"""`capture.py` commits a step's batch, its request ids in batch order and each one's scheduled tokens, under the rank root,
and `stage.Capture` takes those ids as committed: a ``meta.json`` whose ids or counts were changed after the commit, or
whose rank root doesn't cover them, is refused on load."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1]


def _module(monkeypatch, name: str, file: str):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def finished(tmp_path, monkeypatch):
    """A one-rank step `capture.py`'s ``_finish`` wrote, its batch two requests in the order ``b``, ``a``, and `stage.py`."""
    capture = _module(monkeypatch, "spot_capture_batch", "capture.py")
    S = capture._State({"capture": str(tmp_path), "steps": [0], "mla_tokens": 0, "open": {}, "keep": "all", "min_free_gib": 0})
    S.step = 0
    S.sched = types.SimpleNamespace(num_scheduled_tokens={"a": 2, "b": 1})
    S.input_batch = types.SimpleNamespace(req_ids=["b", "a"])
    S.events = [{"seq": 0, "op": "aten::mm", "parent": None, "layer": 0, "in": {}, "out": {}}]
    monkeypatch.setattr(capture, "S", S)

    def commit(runner, d, rank_root):
        c = {"root": hashlib.sha512(f"verity/spot/step-root\0{rank_root}".encode()).hexdigest(), "coins": "00" * 64,
             "rank_roots": [rank_root], "layers": {"dense": [0]}, "opened": {}}
        (d / "commit.json").write_text(json.dumps(c))
        return c
    monkeypatch.setattr(capture, "_commit_and_open", commit)
    capture._finish(None)
    return capture, _module(monkeypatch, "spot_stage_batch", "stage.py"), tmp_path


def test_the_recorded_batch_is_under_the_rank_root_and_stage_reads_its_request_ids(finished) -> None:
    capture, stage, d = finished
    meta = json.loads((d / "rank0" / "step0" / "meta.json").read_text())
    assert (meta["req_ids"], meta["scheduled"]) == (["b", "a"], {"a": 2, "b": 1})
    assert meta["batch_sha512"] == stage.batch_sha512(["b", "a"], {"a": 2, "b": 1}) == capture._batch_sha512(["b", "a"], {"a": 2, "b": 1})
    assert stage.Capture(d, 0).req_ids == {0: ["b", "a"]}


def test_request_ids_or_counts_changed_after_the_commit_are_refused(finished) -> None:
    _, stage, d = finished
    p = d / "rank0" / "step0" / "meta.json"
    meta = json.loads(p.read_text())
    for change in ({"req_ids": ["a", "b"]}, {"scheduled": {"a": 1, "b": 2}}):
        p.write_text(json.dumps({**meta, **change}))
        with pytest.raises(SystemExit, match="req_ids or scheduled counts are not the batch it committed"):
            stage.Capture(d, 0)
    p.write_text(json.dumps({**meta, "req_ids": ["a", "b"], "batch_sha512": stage.batch_sha512(["a", "b"], meta["scheduled"])}))
    with pytest.raises(SystemExit, match="is not SHA-512 of its files"):
        stage.Capture(d, 0)
