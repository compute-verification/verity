"""run_config.load_stages tolerates a stage record an interrupted save left unreadable.

A run that dies mid-fold (ENOSPC) can leave a 0-byte match/stages.json, and the MATCH_PHASE=cpu resume over its preserved capture
must not fail on json.loads(''). The unreadable record is set aside under a stamped name and the resume starts with an
empty stage list -- every stage of the CPU half re-runs and rewrites the record.
"""
from __future__ import annotations

import json

from verity_vllm.pipeline import match as RC
from verity_vllm.check import gates as G


def test_empty_stages_json_is_set_aside_and_the_resume_starts_empty(tmp_path, capsys):
    p = tmp_path / G.ARTIFACTS["stages"]
    p.write_text("")
    assert RC.load_stages(tmp_path) == []
    assert not p.exists()
    aside = [q for q in tmp_path.iterdir() if q.name.startswith(p.name + ".unreadable-")]
    assert len(aside) == 1 and aside[0].read_text() == ""
    assert "unreadable" in capsys.readouterr().err


def test_a_readable_record_and_a_missing_one_are_unchanged(tmp_path):
    assert RC.load_stages(tmp_path) == []
    RC.save_stages(tmp_path, [{"stage": "fold", "status": "ok"}])
    assert [r["stage"] for r in RC.load_stages(tmp_path)] == ["fold"]
    assert json.loads((tmp_path / G.ARTIFACTS["stages"]).read_text())["schema"] == "verity-gen/stages/v1"
