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
