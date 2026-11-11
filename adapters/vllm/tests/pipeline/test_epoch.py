"""`verity-vllm epoch`: the word check's inputs are its arguments, the job's end comes from the run's own timeout, the digests read a row dir."""
from __future__ import annotations

import json
from pathlib import Path

from verity_vllm.pipeline import epoch
from verity_vllm.query.required import QUERY_OF_RECORD

ROW = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager"


def _row(tmp_path: Path, *, lines: int = 1) -> Path:
    d = tmp_path / ROW
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"manifest_digest": "m0", "query": {"query_id": QUERY_OF_RECORD, "partition": {"object": "p", "digest": "pd"}}}))
    (d / "manifest.log").write_text("".join(f"Q_word_v1{{X=16,W=32,R=no-recompute}}: 10 calls -> 20 units over 30 gates\n" for _ in range(lines)))
    (d / "build_summary.json").write_text(json.dumps({"request": {"program_digest": "p0"}, "step": {"program_digest": "s0"}}))
    return d


def test_word_check_passes_on_the_record_and_names_each_problem(tmp_path: Path) -> None:
    d = _row(tmp_path)
    assert epoch.word_check(ROW, d, 1, "", "", "")["ok"]
    assert epoch.word_check(ROW, d, 1, "16/32", "G=1", "G=1")["ok"]
    rep = epoch.word_check(ROW, d, 2, "8/16", "G=2", "G=1")
    assert not rep["ok"] and len(rep["problems"]) == 3, rep["problems"]
    assert any("fewer than the 2 components" in p for p in rep["problems"])
    assert any("VERITY_WORD_CHECK=8/16" in p for p in rep["problems"])
    assert any("VERITY_QWORD_MAX_GATES=G=2" in p for p in rep["problems"])


def test_job_end_is_the_run_timeout_less_the_reserve_or_an_earlier_deadline(tmp_path: Path) -> None:
    run, custody = tmp_path / "run", tmp_path / "custody"
    (run / "control").mkdir(parents=True)
    custody.mkdir()
    (run / "status.json").write_text(json.dumps({"transitions": [{"state": "running", "started_utc": "2026-09-29T00:00:00.000Z", "timeout_s": 3600}]}))
    (custody / "cred.json").write_text(json.dumps({"expires_utc": "2026-09-29T02:00:00Z"}))
    start = 1790640000
    assert epoch.job(run, custody, 600, None) == (start + 3600, start + 3000, "2026-09-29T02:00:00Z")
    assert epoch.job(run, custody, 600, "2026-09-29T00:10Z")[1] == start + 600
    (run / "control" / "timeout_s").write_text(json.dumps({"timeout_s": 7200}))
    assert epoch.job(run, custody, 600, None)[:2] == (start + 7200, start + 6600)
    assert epoch.job(tmp_path / "none", tmp_path / "none", 600, None) == (None, None, None)


def test_digests_and_row_state_read_the_row_dir(tmp_path: Path) -> None:
    d = _row(tmp_path)
    r = epoch.row_digests(d)
    assert r["programs"]["0"]["step"] == "s0" and r["programs"]["0"]["requests"] == {"request": "p0"} and r["partition_digests"] == ["pd"]
    fams, outcome, mv = epoch.row_state(d, str(tmp_path / "header.json"))
    assert (fams, outcome, mv) == ([], None, None) and json.loads((tmp_path / "header.json").read_text())["manifest_digest"] == "m0"


def test_a_tp_rows_digests_are_its_build_summarys_ranks(tmp_path: Path) -> None:
    """`row_records.tp_build_summary` keeps every rank's Builds under `ranks`; no rank has a summary of its own."""
    d = tmp_path / "row"
    d.mkdir()
    rank = lambda r: {"build_step": {"program_digest": f"s{r}"}, "build_request_LP8_T4": {"program_digest": f"q{r}"}, "workload": {"workload_digest": f"w{r}"}}  # noqa: E731
    (d / "build_summary.json").write_text(json.dumps({"tp": {"world": 2}, "ranks": {"rank1": rank(1), "rank0": rank(0)}, "step": {"program_digest": "s0"}}))
    progs = epoch.row_digests(d)["programs"]
    assert progs == {str(r): {"step": f"s{r}", "workload": f"w{r}", "requests": {"build_request_LP8_T4": f"q{r}"}} for r in (0, 1)}
    assert list(progs) == ["0", "1"]
    meta = epoch.store_meta(d, kind="build", role="R", rownum=1, run_id="r", epoch_sha="x", gpu="", pod="", epoch="e")
    assert meta["program_digests"] == {"rank0/step": "s0", "rank0/build_request_LP8_T4": "q0", "rank1/step": "s1", "rank1/build_request_LP8_T4": "q1"}
    assert meta["workload_digests"] == {"rank0": "w0", "rank1": "w1"}
