"""[r19-int-18, F-r19-int-18] row_pod.sh's Match summary accepts an explicit MATCH_PHASE=cpu resume over the preserved capture.

The record-freshness rule (F-global-rec) requires the capture log to be newer than the stage start, so that a re-run whose GPU capture
failed cannot be verified on the previous attempt's record.  A MATCH_PHASE=cpu resume (F-r19-int-14) re-runs the CPU half over a
capture that is older by construction: #57's resume (r20260922-144523-7ea6) reported "record predates this match stage" on a whole,
37-minute-old record.  Under the flag a complete record (capture + control tokens.json beside the log) is accepted and labelled
`resumed`; when the stage record was set aside (F-r19-int-17) the GPU stages have no rc -- accepted on the completion markers,
labelled stage_record_absent.  Without the flag the mtime rule stands.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
ROW_POD = TREE / "verity_capture/sweep/row_pod.sh"


def _summary_program() -> str:
    src = ROW_POD.read_text()
    m = re.search(r'ok=\$\(\$PY312 - "\$D" "\$rcm" "\$rcc" "\$\{GM_RC:-97\}" "\$t0" "\$rcs" <<\'EOF\'\n(.*?)\nEOF\n', src, re.DOTALL)
    assert m, "match summary heredoc not found"
    return m.group(1)


def _row(tmp_path: Path, *, old: bool, stage_record: bool) -> Path:
    d = tmp_path / "row"
    (d / "match" / "capture").mkdir(parents=True)
    (d / "match" / "control").mkdir(parents=True)
    toks = {"requests": [{"request_id": "r0", "prompt_token_ids": [1, 2], "generated_token_ids": [3]}]}
    for p in ("capture", "control"):
        (d / "match" / p / "tokens.json").write_text(json.dumps(toks))
    (d / "match" / "capture" / "log.jsonl.gz").write_bytes(b"x")
    (d / "match" / "program.json").write_text("{}")
    (d / "match" / "instances.jsonl").write_text("")
    if old:
        t = time.time() - 3600
        os.utime(d / "match" / "capture" / "log.jsonl.gz", (t, t))
    if stage_record:
        (d / "match" / "stages.json").write_text(json.dumps({"stages": [{"stage": "capture", "returncode": 0, "status": "ok"},
                                                                        {"stage": "control", "returncode": 0, "status": "ok"},
                                                                        {"stage": "fold", "returncode": 0, "status": "ok"}]}))
    (d / "match_decomp.json").write_text(json.dumps({"verdict": "PASS", "reasons": [], "requests": [{"steps": 3}], "attribution": {}}))
    (d / "global_match.json").write_text(json.dumps({"verdict": "PASS", "comparator": "cb_a.global_match/v1", "checks": {}, "failed_checks": []}))
    return d


def _run(d: Path, phase: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "MATCH_PHASE"}
    if phase:
        env["MATCH_PHASE"] = phase
    t0 = int(time.time()) - 60   # the stage started a minute ago
    r = subprocess.run(["python3", "-", str(d), "0", "0", "0", str(t0), "-1"], input=_summary_program(), capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 0, r.stderr
    return json.load(open(d / "match_summary.json"))


def test_without_the_flag_an_old_record_is_stale(tmp_path):
    s = _run(_row(tmp_path, old=True, stage_record=True), None)
    assert s["record"]["fresh"] is False and s["record"]["resumed"] is False
    assert "predates" in s["record"]["reason"] and s["pass"] is False


def test_a_fresh_record_passes_unchanged(tmp_path):
    s = _run(_row(tmp_path, old=False, stage_record=True), None)
    assert s["record"]["fresh"] is True and s["record"]["reason"] is None


def test_the_cpu_resume_accepts_a_complete_old_record_and_labels_it(tmp_path):
    s = _run(_row(tmp_path, old=True, stage_record=True), "cpu")
    assert s["record"]["fresh"] is True and s["record"]["resumed"] is True and s["record"]["resumed_record_complete"] is True
    assert s["record"]["stage_record_absent"] is False and s["record"]["capture_rc"] == 0


def test_the_cpu_resume_without_a_stage_record_is_accepted_on_the_completion_markers(tmp_path):
    s = _run(_row(tmp_path, old=True, stage_record=False), "cpu")
    assert s["record"]["fresh"] is True and s["record"]["stage_record_absent"] is True and s["record"]["capture_rc"] is None


def test_the_cpu_resume_refuses_an_incomplete_record(tmp_path):
    d = _row(tmp_path, old=True, stage_record=False)
    (d / "match" / "control" / "tokens.json").unlink()
    s = _run(d, "cpu")
    assert s["record"]["fresh"] is False and s["record"]["resumed_record_complete"] is False


def test_match_phase_is_exported_for_the_summary():
    src = ROW_POD.read_text()
    assert re.search(r"^\s*export MATCH_PHASE=\$\{MATCH_PHASE:-all\}", src, re.MULTILINE)
    assert subprocess.run(["bash", "-n", str(ROW_POD)], capture_output=True, check=False).returncode == 0
