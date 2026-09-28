"""job.json v0.2 over v0.1, the result.json contract, and `research run` honoring both."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research import record, result
from research import runs as R
from research.cli import EX_DATAERR, main
from research.telemetry import run as tele_run

FIXTURES = Path(__file__).parent / "fixtures"
PY = sys.executable


def _run(tmp_path, *extra, code="pass", stage="t", require=False, expect=()):
    rid = R.new_run_id()
    args = ["run", "--runs-dir", str(tmp_path), "--id", rid, "--stage", stage, "--no-sampler"]
    if require:
        args.append("--require-result")
    for e in expect:
        args += ["--expect", e]
    args += [*extra, "--", PY, "-c", code]
    rc = main(args)
    return rc, R.summarize(tmp_path / rid)


# ---- records ---------------------------------------------------------------------------------------------------------------------------

def test_v01_record_reads_with_unknown_not_defaults(tmp_path):
    shutil.copy(FIXTURES / "job_v0.1.json", tmp_path / "job.json")
    job = record.read_job(tmp_path)
    assert job is not None and job["schema"] == "telemetry/job/v0.1"
    for k in record.V2_SECTIONS:
        assert job[k] is None, f"{k} must be unknown (None) on an old record, not defaulted"
    assert job["ownership"]["job"] == "oom_own"                      # the v0.1 content is untouched
    s = R.summarize(tmp_path)
    assert s["readable"] and s["tool"] is None and s["fingerprint_expected"] is None
    assert s["cost"]["usd"] is None and s["cost"]["reason"]
    assert s["run_id"] == "oom_own"                                    # v0.1: the job name stands in for the run id


def test_a_running_run_whose_check_waits_for_a_slot_is_waiting(tmp_path, capsys):
    """af2d (10 Oct) waited 95 min in node 1's check line and was reported running since its launch, and so cancelled as a
    check 107 min in when it had checked for 12: its slot record (slot.py's) makes it `waiting` until it holds a slot."""
    shutil.copy(FIXTURES / "job_v0.1.json", tmp_path / "job.json")
    (tmp_path / "status.json").write_text(json.dumps({"state": "running"}))
    assert R.summarize(tmp_path)["state"] == "running" and R.summarize(tmp_path)["slot"] is None
    wait = {"state": "waiting", "since": "2026-10-10T16:56:27Z", "priority": True, "why": "every check slot is taken (a: busy; b: busy)"}
    (tmp_path / R.SLOT_FILE).write_text(json.dumps(wait))
    s = R.summarize(tmp_path)
    assert s["state"] == "waiting" and s["state"] in R.ACTIVE and s["slot"] == wait
    assert main(["inspect", str(tmp_path)]) == 0
    assert "slot      waiting in the check line since 2026-10-10T16:56:27Z, not started: every check slot is taken" in capsys.readouterr().out
    (tmp_path / R.SLOT_FILE).write_text(json.dumps({**wait, "state": "held", "slot": "a", "cores": "32-63", "took": "2026-10-10T18:31:22Z"}))
    assert R.summarize(tmp_path)["state"] == "running"
    assert main(["inspect", str(tmp_path)]) == 0
    assert "slot      check slot a (cores 32-63) taken 2026-10-10T18:31:22Z, after waiting in line since 2026-10-10T16:56:27Z" \
        in capsys.readouterr().out
    (tmp_path / R.SLOT_FILE).write_text(json.dumps(wait))
    (tmp_path / "status.json").write_text(json.dumps({"state": "cancelled"}))
    assert R.summarize(tmp_path)["state"] == "cancelled"                # an ended run is what it ended as


def test_extend_job_adds_sections_and_changes_nothing_else():
    v1 = json.loads((FIXTURES / "job_v0.1.json").read_text())
    v2 = record.extend_job(dict(v1), run_id="r1", runs_root=Path("/tmp/x"), expected_fingerprint={"elf_sha256": "abc"},
                           cost=record.cost_section(provider="runpod", pod_id="p1", rate_usd_per_hour=2.18, rate_observed_utc="t0", shared=False))
    assert v2["schema"] == record.JOB_SCHEMA
    for k in v1:
        if k != "schema":
            assert v2[k] == v1[k], f"v0.1 section {k} was altered"
    assert v2["run"]["id"] == "r1" and v2["workload_fingerprint"]["expected"] == {"elf_sha256": "abc"}
    assert v2["cost"]["rate_usd_per_hour"] == 2.18 and v2["tool"]["research_version"]
    with pytest.raises(ValueError):
        record.extend_job(v2, run_id="r2", runs_root=Path("/tmp"), expected_fingerprint=None)   # only v0.1 in


def test_unknown_schema_is_not_read(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps({"schema": "telemetry/job/v9", "ownership": {"job": "x"}}))
    assert record.read_job(tmp_path) is None
    s = R.summarize(tmp_path)
    assert s["readable"] is False and s["job"] == "x"                  # best effort fields, flagged


def test_cost_estimate_is_occupancy_only_and_unknown_without_rate():
    job = {"cost": record.cost_section(provider="runpod", pod_id="p", rate_usd_per_hour=3.6, rate_observed_utc="t", shared=True)}
    est = record.cost_estimate(job, {"wall_s": 1800})
    assert est["usd"] == 1.8 and est["shared"] is True and "occupancy" in est["basis"]
    assert record.cost_estimate(job, {})["usd"] is None
    assert record.cost_estimate({"cost": record.cost_section()}, {"wall_s": 10})["usd"] is None


# ---- result.json -----------------------------------------------------------------------------------------------------------------------

def _ok_result(run_id, **over):
    doc = result.template(run_id)
    doc["workload_fingerprint"] = {"elf_sha256": "deadbeef", "mode": "core"}
    doc["validation"] = {"status": "passed", "checker": "verify", "detail": "674/0", "evidence": ["verify.log"]}
    doc["measurements"] = [{"name": "prove_wall", "value": 12.5, "unit": "s"}]
    doc.update(over)
    return doc


def test_result_absent_vs_invalid_vs_wrong_run(tmp_path):
    chk = result.check(tmp_path, "r1")
    assert not chk.present and not chk.valid
    (tmp_path / "result.json").write_text("{not json")
    chk = result.check(tmp_path, "r1")
    assert chk.present and not chk.valid and "not valid JSON" in chk.errors[0]
    (tmp_path / "result.json").write_text(json.dumps(_ok_result("r-other")))
    chk = result.check(tmp_path, "r1")
    assert not chk.valid and any("wrong run" in e for e in chk.errors)
    (tmp_path / "result.json").write_text(json.dumps(_ok_result("r1")))
    chk = result.check(tmp_path, "r1")
    assert chk.valid and chk.validation_status == "passed" and chk.fingerprint["elf_sha256"] == "deadbeef"
    assert chk.artifacts_missing == []


def test_result_rejects_escaping_paths_and_bad_units(tmp_path):
    bad = _ok_result("r1", artifacts=["../secrets"], measurement_files=["/etc/passwd"])
    chk = result.check_doc(tmp_path, bad, "r1")
    assert not chk.valid and sum("stay inside" in e for e in chk.errors) == 2
    bad = _ok_result("r1", measurements=[{"name": "x", "value": 1}])
    assert not result.check_doc(tmp_path, bad, "r1").valid
    bad = _ok_result("r1", validation={"status": "ok"})
    assert not result.check_doc(tmp_path, bad, "r1").valid


def test_result_missing_artifacts_are_reported_not_fatal(tmp_path):
    doc = _ok_result("r1", artifacts=["proofs/", "results.jsonl"])
    (tmp_path / "results.jsonl").write_text("")
    chk = result.check_doc(tmp_path, doc, "r1")
    assert chk.valid and chk.artifacts_missing == ["proofs/"]


def test_result_size_bound(tmp_path):
    doc = _ok_result("r1"); doc["blob"] = "x" * (result.MAX_BYTES + 10)
    (tmp_path / "result.json").write_text(json.dumps(doc))
    chk = result.check(tmp_path, "r1")
    assert not chk.valid and "bytes" in chk.errors[0]


def test_write_is_atomic_and_refuses_invalid(tmp_path):
    with pytest.raises(ValueError):
        result.write(tmp_path, {"schema": "nope", "run_id": "r1"})
    assert not (tmp_path / "result.json").exists() and not list(tmp_path.glob("result.json.tmp.*"))
    p = result.write(tmp_path, _ok_result("r1"))
    assert p.name == "result.json" and result.check(tmp_path, "r1").valid


# ---- research run honoring both --------------------------------------------------------------------------------------------------------

WRITES_RESULT = """
import json, os
d = os.environ["RESEARCH_RUN_DIR"]; rid = os.environ["RESEARCH_RUN_ID"]
open(os.path.join(d, "verify.log"), "w").write("674/0\\n")
json.dump({"schema": "research/result/v0.1", "run_id": rid, "workload_fingerprint": {"elf_sha256": "observed-1", "mode": "core"},
           "validation": {"status": "passed", "checker": "verify", "detail": "674/0", "evidence": ["verify.log"]},
           "measurements": [{"name": "prove_wall", "value": 1.25, "unit": "s"}], "artifacts": ["verify.log"]},
          open(os.path.join(d, "result.json.tmp"), "w"))
os.replace(os.path.join(d, "result.json.tmp"), os.path.join(d, "result.json"))
"""


def test_run_writes_v02_record_and_reads_result(tmp_path):
    rc, s = _run(tmp_path, code=WRITES_RESULT, expect=("elf_sha256=expected-1",))
    assert rc == 0 and s["state"] == "done"
    job = json.loads((Path(s["dir"]) / "job.json").read_text())
    assert job["schema"] == record.JOB_SCHEMA and job["run"]["id"] == s["id"] == job["ownership"]["job"]
    assert job["workload_fingerprint"]["expected"] == {"elf_sha256": "expected-1"}
    assert s["fingerprint_observed"] == {"elf_sha256": "observed-1", "mode": "core"}, "expected and observed are kept apart"
    assert s["result_valid"] and s["validation"] == "passed" and s["measurements"][0]["unit"] == "s"
    assert s["cost"]["usd"] is None                                    # local run: no billed provider, unknown not zero


def test_run_harness_record_matches_prewritten_record(tmp_path):
    """The pre-written job must carry exactly what the harness would have written (same keys, same command handling)."""
    rc, s = _run(tmp_path, code="pass")
    job = json.loads((Path(s["dir"]) / "job.json").read_text())
    v1_keys = set(json.loads((FIXTURES / "job_v0.1.json").read_text()))
    assert v1_keys <= set(job) and job["execution"]["command"][0] == PY and job["telemetry"]["instruments"] == []


def test_require_result_turns_zero_exit_without_result_into_65(tmp_path):
    rc, s = _run(tmp_path, code="pass", require=True)
    assert rc == EX_DATAERR and s["rc"] == 0 and s["state"] == "done", "status.json keeps the true exit; research refuses the contract"
    rc, s = _run(tmp_path, code=WRITES_RESULT, require=True)
    assert rc == 0
    rc, s = _run(tmp_path, code="import sys; sys.exit(4)", require=True)
    assert rc == 4, "a failing workload keeps its own exit code; --require-result only speaks when the exit was 0"


def test_wrong_run_result_is_flagged(tmp_path):
    code = WRITES_RESULT.replace('"run_id": rid', '"run_id": "r-other"')
    rc, s = _run(tmp_path, code=code, require=True)
    assert rc == EX_DATAERR and s["result_present"] and not s["result_valid"] and any("wrong run" in e for e in s["result_errors"])


def test_arguments_with_spaces_survive(tmp_path):
    rc, s = _run(tmp_path, "--input", f"spaced={tmp_path}/a file.txt", code="pass", stage="sp")
    job = json.loads((Path(s["dir"]) / "job.json").read_text())
    ref = job["computation"]["inputs"][0]
    assert rc == 0 and ref["name"] == "spaced" and ref["path"].endswith("/a file.txt") and ref["sha256_reason"] == "absent at launch"
    rid = R.new_run_id()
    rc = main(["run", "--runs-dir", str(tmp_path), "--id", rid, "--stage", "sp2", "--no-sampler", "--", PY, "-c", "import sys; print(sys.argv[1])", "hello world  two"])
    s = R.summarize(tmp_path / rid)
    assert rc == 0 and (Path(s["dir"]) / "stdout.log").read_text() == "hello world  two\n"
    assert s["command"][-1] == "hello world  two"


#: The workload's child: it takes the write end of the FIFO argv[1], then writes its pid to argv[2] and blocks until signalled.
TREE_CHILD = ("import os, signal, sys; os.open(sys.argv[1], os.O_WRONLY); open(sys.argv[2] + '.tmp', 'w').write(str(os.getpid())); "
              "os.replace(sys.argv[2] + '.tmp', sys.argv[2]); signal.pause()")
TREE = f"import signal, subprocess, sys; subprocess.Popen([sys.executable, '-c', {TREE_CHILD!r}, *sys.argv[1:]]); signal.pause()"


def test_timeout_kills_the_workload_tree_and_spares_bystanders(tmp_path, monkeypatch):
    """The stage clock stands still until the workload's child has written its pid, then jumps past --timeout, so the deadline
    passes with the whole tree up however slow the machine is.  The child holds the FIFO's only write end: EOF on the read end
    is its exit, whenever that comes.  The bystander blocks on a pipe the test holds and exits 0 unless something signalled it."""
    fifo, ready = tmp_path / "tree", tmp_path / "child.pid"
    os.mkfifo(fifo)
    tree = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)              # a reader first, so the child's open for writing does not block
    monkeypatch.setattr(tele_run, "clock", lambda: 100.0 if ready.exists() else 0.0)
    bystander = subprocess.Popen([PY, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        rid = R.new_run_id()
        rc = main(["run", "--runs-dir", str(tmp_path), "--id", rid, "--stage", "to", "--no-sampler", "--timeout", "1", "--",
                   PY, "-c", TREE, str(fifo), str(ready)])
        s = R.summarize(tmp_path / rid)
        assert rc in (143, 137) and s["state"] == "cancelled" and s["class"] == "STAGE_TIMEOUT"
        events = [json.loads(l) for l in (Path(s["dir"]) / "events.jsonl").read_text().splitlines()]
        intent = next(e for e in events if e.get("kind") == "cancel_intent")
        assert intent["actor"] == "timeout", "kill evidence preserved"
        assert int(ready.read_text()) in [m["pid"] for m in intent["target"]["members"]], "the signal went to the workload's group"
        os.set_blocking(tree, True)
        assert os.read(tree, 1) == b"", "the workload's child exited"
        bystander.communicate()
        assert bystander.returncode == 0, "an unrelated process must not be touched"
    finally:
        os.close(tree)
        bystander.kill()


# ---- compare ---------------------------------------------------------------------------------------------------------------------------

def test_compare_is_descriptive_and_labels_ratios(tmp_path):
    from research import compare as C
    fast = WRITES_RESULT.replace('"value": 1.25', '"value": 0.5').replace('"observed-1"', '"observed-2"')
    _, b = _run(tmp_path, code=WRITES_RESULT, stage="bench", expect=("elf_sha256=expected-1",))
    _, c = _run(tmp_path, code=fast, stage="bench", expect=("elf_sha256=expected-1",))
    rep = C.compare(Path(b["dir"]), Path(c["dir"]))
    assert rep["differences"]["fingerprint_expected"] == []
    fo = rep["differences"]["fingerprint_observed"]
    assert fo and fo[0]["field"] == "elf_sha256" and fo[0]["baseline"] == "observed-1" and fo[0]["candidate"] == "observed-2"
    m = rep["measurements"][0]
    assert m["name"] == "prove_wall" and m["ratio_candidate_over_baseline"] == 0.4 and m["unit"] == "s"
    assert any("workload contract" in cv for cv in rep["caveats"]) and any("not a confidence interval" in cv for cv in rep["caveats"])
    text = C.render(rep)
    assert "candidate/baseline" in text and "x0.4000" in text and "elf_sha256" in text
    assert main(["compare", b["id"], c["id"], "--dirs", "--runs-dir", str(tmp_path)]) == 0


def test_compare_flags_missing_validation_and_units(tmp_path):
    from research import compare as C
    noval = WRITES_RESULT.replace('"status": "passed"', '"status": "not_run"').replace('"unit": "s"', '"unit": "ms"')
    _, b = _run(tmp_path, code=WRITES_RESULT, stage="bench")
    _, c = _run(tmp_path, code=noval, stage="bench")
    rep = C.compare(Path(b["dir"]), Path(c["dir"]))
    assert rep["measurements"][0]["unit_mismatch"] and "ratio_candidate_over_baseline" not in rep["measurements"][0]
    assert any("did not report validation passed" in cv for cv in rep["caveats"])
