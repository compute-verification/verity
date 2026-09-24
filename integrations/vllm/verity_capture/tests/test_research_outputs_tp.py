"""research_outputs on a tp_stage.sh evidence dir: rank Builds under build/rank<r>/, the `tp-ranks` digest of record, TP Match record."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity_capture.sweep import research_outputs as ro

ROW = "llama32-1b__bf16__l40s__tp4__b8__i256__o32__mixed__greedy__bi-eager"
WDS = ["d" * 64, "a" * 64, "c" * 64, "b" * 64]


def _w(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc) if not isinstance(doc, str) else doc)


def _tp_dir(tmp_path: Path, world: int = 4) -> Path:
    d = tmp_path / ROW
    ranks = {}
    for r in range(world):
        rd = d / "build" / f"rank{r}"
        _w(rd / "build_step" / "result.json", {"ok": True})
        _w(rd / "build_request_LP9_T2" / "result.json", {"ok": True})
        _w(rd / "build_request_LP9_T2" / "instances.json.gz", "planes")
        _w(rd / "build_request_LP9_T2.log", "derive log")
        _w(rd / "build_workload" / "workload_program.json", {"workload_digest": WDS[r]})
        ranks[f"rank{r}"] = {"build_step": {"ok": True, "program_digest": f"{r}" * 64}, "workload": {"ok": True, "workload_digest": WDS[r]}}
    _w(d / "build_summary.json", {"tp": {"world": world}, "ranks": ranks, "step": {"program_digest": "0" * 64}})
    _w(d / "manifest.json", {"program_digest": "x"})
    _w(d / "match" / "tp2_match.json", {"pass": True})
    _w(d / "match_summary.json", {"pass": True})
    _w(d / "stages.txt", "build PASS 2026-09-23T00:00:00Z rc=0\nmatch PASS 2026-09-23T00:00:01Z rc=0/0/0/-1\n")
    return d


def test_tp_digest_of_record_is_the_rank_fold(tmp_path):
    d = _tp_dir(tmp_path)
    dig = ro.program_digest(d, ROW)
    assert dig["tp"] == 4
    assert dig["rank_workload_digests"] == WDS
    assert dig["step_digest"] == "0" * 64
    assert dig["program_digest"] == hashlib.sha256(("tp-ranks/" + ",".join(sorted(WDS))).encode()).hexdigest()


def test_tp_digest_absent_when_a_rank_has_no_workload_program(tmp_path):
    d = _tp_dir(tmp_path)
    bs = json.loads((d / "build_summary.json").read_text())
    bs["ranks"]["rank2"]["workload"] = {"ok": False, "workload_digest": None}
    _w(d / "build_summary.json", bs)
    assert ro.program_digest(d, ROW)["program_digest"] is None


def test_tp_build_and_match_outputs_take_the_rank_files(tmp_path):
    d = _tp_dir(tmp_path)
    run = tmp_path / "run"
    doc = ro.build_outputs(d, ROW, "build", run, {})
    build = next(e for e in doc["outputs"] if e["name"] == "build")
    staged = {p.relative_to(run / "outputs" / "build").as_posix() for p in (run / "outputs" / "build").rglob("*") if p.is_file()}
    for r in range(4):
        assert f"build/rank{r}/build_step/result.json" in staged
        assert f"build/rank{r}/build_request_LP9_T2/result.json" in staged
        assert f"build/rank{r}/build_workload/workload_program.json" in staged
        assert f"build/rank{r}/build_request_LP9_T2/instances.json.gz" not in staged
    assert {"manifest.json", "build_summary.json"} <= staged
    assert build["meta"]["program_digest"] == ro.program_digest(d, ROW)["program_digest"]
    assert build["meta"]["required_manifest"] is True
    logs = next(e for e in doc["outputs"] if e["name"] == "logs")
    assert logs["meta"]["files"] == 4

    doc = ro.build_outputs(d, ROW, "match", tmp_path / "run_m", {"build": "art:" + "f" * 64})
    match = next(e for e in doc["outputs"] if e["name"] == "match")
    assert match["meta"]["tp_match_ok"] is True
    assert match["refs"] == {"build": "art:" + "f" * 64}
    assert (tmp_path / "run_m" / "outputs" / "match" / "match" / "tp2_match.json").is_file()


def _commit_line(d: Path, word: str) -> None:
    with open(d / "stages.txt", "a") as f:
        f.write(f"commit {word} 2026-09-23T00:00:02Z rc=0\n")


def test_tp_commit_reads_tp_commit_summary(tmp_path):
    d = _tp_dir(tmp_path)
    _w(d / "commit" / "summary.json", {"pass": True, "tokens_equal": True, "pairs": 3})
    _commit_line(d, "PASS")
    doc = ro.build_outputs(d, ROW, "commit", tmp_path / "run_c", {"build": "art:" + "f" * 64, "match": "art:" + "e" * 64})
    v = next(e for e in doc["outputs"] if e["name"] == "verdict")
    assert v["meta"]["decision_record"] == "commit/summary.json"
    assert (v["meta"]["tp_pass"], v["meta"]["tokens_equal"], v["meta"]["pairs"]) == (True, True, 3)
    assert "decision_record_missing" not in v["meta"]


def test_commit_pass_without_its_decision_document_is_refused(tmp_path):
    d = _tp_dir(tmp_path)
    _commit_line(d, "PASS")
    try:
        ro.build_outputs(d, ROW, "commit", tmp_path / "run_c", {})
    except ValueError as e:
        assert "commit/summary.json" in str(e)
    else:
        raise AssertionError("a PASS Commit without commit/summary.json published")
    t1 = tmp_path / "t1"
    _w(t1 / "stages.txt", "commit PASS 2026-09-23T00:00:02Z rc=0\n")
    _w(t1 / "commit" / "verdict.json", {"verdict": "PASS"})
    try:
        ro.build_outputs(t1, "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager", "commit", tmp_path / "run_t1", {})
    except ValueError as e:
        assert "verdict.json" in str(e)
    else:
        raise AssertionError("a PASS TP1 Commit without verdict.json published")


def test_failed_commit_names_the_missing_decision_document(tmp_path):
    d = _tp_dir(tmp_path)
    _commit_line(d, "FAIL")
    _w(d / "commit.log", "log")
    doc = ro.build_outputs(d, ROW, "commit", tmp_path / "run_c", {})
    assert all(e["name"] != "verdict" or e["meta"]["decision_record_missing"] == "commit/summary.json" for e in doc["outputs"])


def test_tp1_commit_reads_the_verdict_of_record(tmp_path):
    d = tmp_path / "row"
    _w(d / "stages.txt", "commit PASS 2026-09-23T00:00:02Z rc=0\n")
    _w(d / "verdict.json", {"outcome": "PASS", "runs": {"instrumented": 3, "control": 3, "failed": 0}})
    doc = ro.build_outputs(d, "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager", "commit", tmp_path / "run", {})
    v = next(e for e in doc["outputs"] if e["name"] == "verdict")
    assert (v["meta"]["decision_record"], v["meta"]["outcome"], v["meta"]["n_runs"], v["meta"]["runs_failed"]) == ("verdict.json", "PASS", 6, 0)


def test_tp1_layout_unchanged(tmp_path):
    d = tmp_path / "row"
    _w(d / "build_summary.json", {"request": {"program_digest": "r" * 64}, "step": {"program_digest": "s" * 64}})
    dig = ro.program_digest(d, "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager")
    assert dig == {"program_digest": "r" * 64, "request_digest": "r" * 64, "step_digest": "s" * 64, "workload_digest": None}
