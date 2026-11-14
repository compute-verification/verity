"""Each message the adapter writes or reads through its declaration (`verity_vllm.messages`) keeps the bytes it had
before, and its reader reads what it read before."""

from __future__ import annotations

import json

import pytest

from verity.channel import Refused
from verity_vllm import messages as M
from verity_vllm.pipeline import advice as A


def test_every_declared_message_is_a_type():
    assert {"RunTotals", "Run", "Request", "ServingConfiguration", "CommittedRows", "PoUSStore"} <= M.TYPES.types


def test_a_type_no_protocol_declares_is_refused():
    with pytest.raises(Refused, match="unknown-type"):
        M.body("Summary", b"{}")


def test_run_totals_bytes_unchanged(tmp_path):
    got = {"final": True, "served": 3, "stream_errors": {"n": 1, "first": ["engine: é"]}, "engine_steps": 7,
           "commit": None, "remote_session": None}
    (tmp_path / "before").mkdir()
    (tmp_path / "after").mkdir()
    (tmp_path / "before" / "summary.json").write_text(json.dumps(got, indent=1))
    M.write(tmp_path / "after" / "summary.json", "RunTotals", json.dumps(got, indent=1).encode())
    assert (tmp_path / "after" / "summary.json").read_bytes() == (tmp_path / "before" / "summary.json").read_bytes()
    assert A.stopped(tmp_path / "after") is True
    (tmp_path / "after" / "summary.json").write_text(json.dumps({**got, "final": False}))
    assert A.stopped(tmp_path / "after") is False
    assert A.stopped(tmp_path / "none") is False


def test_serving_configuration_bytes_unchanged(tmp_path):
    record = {"engine": "vllm", "revision": "0.0", "flags": ["--seal"], "phi": {"layers": 2}, "path": tmp_path,
              "clocks": {"perf_counter": 1.5, "time": 2.0}}
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps(record, indent=1, default=str))
    M.write(after, "ServingConfiguration", json.dumps(record, indent=1, default=str).encode())
    assert after.read_bytes() == before.read_bytes()
    assert json.loads(M.read(after, "ServingConfiguration")) == json.loads(before.read_text())


def test_committed_rows_bytes_unchanged(tmp_path):
    doc = {"window": 2, "run_root": "ab" * 64, "statics": {"width": 4}, "ids": [[1, 2], [3]],
           "session": [[None, [[0, 1]]]], "salts": [bytes(range(16)).hex()]}
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps(doc))
    M.write(after, "CommittedRows", json.dumps(doc).encode())
    assert after.read_bytes() == before.read_bytes()
