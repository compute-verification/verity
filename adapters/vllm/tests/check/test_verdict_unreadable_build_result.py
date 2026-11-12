"""verdict.from_record: a build wrapper's result.json that is on the row but cannot be read is named on the verdict, never skipped
silently; the other wrappers' correspondence digests still count, and the decision is unchanged."""
from __future__ import annotations

import json

from tests.check.test_verdict import _base, _row_dir
from verity_vllm.check import verdict as V


def _wrappers(d, request_text: str, step: str) -> None:
    (d / "build_step").mkdir()
    (d / "build_step" / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "p" * 64, "correspondence_digest": step}}))
    (d / "build_request").mkdir()
    (d / "build_request" / "result.json").write_text(request_text)


def test_an_unreadable_build_result_is_named_and_the_other_wrappers_still_count(tmp_path):
    d, _ = _row_dir(tmp_path, _base(), commit_pass=True, rc=0)
    step = "e6" * 32
    _wrappers(d, '{"ok": true, "program": ', step)
    v = V.from_record(d, rc=0, stage_ok="PASS")
    assert v.outcome == V.PASS
    assert v.correspondence_digest == step and v.correspondence_digest_source == "result.json:step" and v.correspondence_digests == {"step": step}
    (note,) = v.notes
    assert note.startswith("build record unreadable: build_request/result.json (JSONDecodeError(") and "other wrappers' records only" in note


def test_readable_build_results_add_no_note(tmp_path):
    d, _ = _row_dir(tmp_path, _base(), commit_pass=True, rc=0)
    step, req = "e6" * 32, "fc" * 32
    _wrappers(d, json.dumps({"ok": True, "program": {"digest": "p" * 64, "correspondence_digest": req}}), step)
    v = V.from_record(d, rc=0, stage_ok="PASS")
    assert v.outcome == V.PASS and not v.notes and v.correspondence_digests == {"request": req, "step": step}
