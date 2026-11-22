"""`ledger_fetch.py`: each look appends the gate's new ledger lines to the serving node's copy, which the hold follows; a
shorter fetch appends nothing, and a copy that isn't a prefix of the gate's ledger is refused. The fetch is handed in."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
_spec = importlib.util.spec_from_file_location("ledger_fetch", ROOT / "adapters/vllm/tools/served_run/ledger_fetch.py")
LF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LF)


def test_each_look_appends_the_gates_new_lines_and_a_shorter_fetch_appends_nothing(tmp_path):
    out, looks, waits = tmp_path / "windows.jsonl", iter([["e0"], ["e0", "e1"], ["e0"], ["e0", "e1", "e2"]]), []
    LF.follow(lambda: next(looks), out, lambda: waits.append(out.read_text()), rounds=4)
    assert out.read_text() == "e0\ne1\ne2\n"
    assert waits == ["e0\n", "e0\ne1\n", "e0\ne1\n"]


def test_a_copy_that_is_not_a_prefix_of_the_gates_ledger_is_refused(tmp_path):
    out = tmp_path / "windows.jsonl"
    out.write_text("written-here\n")
    with pytest.raises(LF.Refused, match="not a prefix"):
        LF.follow(lambda: ["e0", "e1"], out, lambda: None, rounds=1)
    assert out.read_text() == "written-here\n"
