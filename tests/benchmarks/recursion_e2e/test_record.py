"""The recursion bench's recorder (`record`): `run.py run` with its job wrapped, and the job's numbers written into the
run's result through the harness kit."""
from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest
import rec_profile as RP
import rec_table as RT
import record as REC
from test_recursion_e2e import job, timev
from verity_bench import kit


@pytest.fixture(autouse=True)
def _stamp(monkeypatch):
    monkeypatch.setattr(kit, "host_stamp", lambda: {"card": "none", "gpus": [], "driver": None, "cuda": None,
                                                    "node": "cpu-node-0", "commit": "c" * 40, "why": "no nvidia-smi"})
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261114-000000-0000")
    monkeypatch.delenv("OUT", raising=False)       # a research run's OUT, which `_ran` sets where a test needs it


def test_run_launches_run_py_s_run_with_the_job_wrapped(capsys) -> None:
    """`record.py run` is `run.py run`, its job's command `JOB` before job.sh's arguments, so the run's command records."""
    assert REC.main(["run", "--dry-run", "--direct", "--on", "cpu-node-2"]) == 0
    cmd = capsys.readouterr().out.split()
    tail = cmd[cmd.index("--") + 1:]
    assert tail[3:6] == list(REC.JOB) and "K=4096" in tail and "STEPS=" + ",".join(
        s for s in RP.ORDER if s != "forge-after") in tail
    assert REC.main(["run", "--dry-run", "--here"]) == 0
    assert capsys.readouterr().out.split()[:3] == list(REC.JOB)


class _Job:
    """job.sh as `job` runs it: writes `result` under the run's out/ (none when None) and exits `rc`."""

    def __init__(self, out: Path, result: dict | None, rc: int):
        self.out, self.result, self.rc = out, result, rc
        self.argv: list[str] = []
        self.signals: list[int] = []

    def __call__(self, argv):
        self.argv = argv
        return self

    def send_signal(self, sig):
        self.signals.append(sig)

    def wait(self):
        if self.result is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / REC.RESULT).write_text(json.dumps(self.result))
        return self.rc


def _ran(tmp_path: Path, monkeypatch, result: dict | None, rc: int = 0, out: bool = False,
         args: tuple[str, ...] = ("K=4096", "N=2"), files: dict[str, bytes] | None = None) -> tuple[int, dict, _Job]:
    """`record.py job` in a research run whose job.sh is `_Job`; `out`: OUT set to the run's out/, as research run sets it,
    else unset (the run's out/ by default); `files`: what the job leaves in out/ beside its result."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    if out:
        monkeypatch.setenv("OUT", str(run_dir / "out"))
    fake = _Job(run_dir / "out", result, rc)
    for rel, data in (files or {}).items():
        (run_dir / "out" / rel).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "out" / rel).write_bytes(data)
    monkeypatch.setattr(REC.subprocess, "Popen", fake)
    monkeypatch.setattr(REC.signal, "signal", lambda sig, handler: handler(sig, None))
    got = REC.main(["job", *args])
    return got, json.loads((run_dir / kit.RESULT).read_text()), fake


def test_the_job_records_its_numbers_partial_until_the_run_is_whole(tmp_path: Path, monkeypatch) -> None:
    """In a research run, `record.py job` runs job.sh with its arguments, forwards a SIGTERM to it, and then writes
    `METRICS` from job.sh's result: the headline's seconds of one recursive run and of its Lean step, partial while a
    step one recursive run needs didn't run and measured once every one did, the validation the table's verdict, and
    job.sh's exit status as its own. While a step is missing, the run's seconds have no value, and why names the steps:
    the headline's sum would be only the steps that ran."""
    doc = RP.read(job(tmp_path / "a"))
    h = doc["headline"]
    rc, got, fake = _ran(tmp_path / "partial", monkeypatch, doc, rc=3)
    assert rc == 3 and fake.argv[1:] == [str(REC.HERE / "job.sh"), "K=4096", "N=2"] and fake.signals == [signal.SIGTERM]
    assert not h["complete"] and [m["name"] for m in got["measurements"]] == list(REC.METRICS)
    run_s, lean_s = got["measurements"]
    missing = [x for x in RP.E2E_STEPS if not doc["steps"].get(x)]
    assert missing and h["pipeline_sum_s"] > 0
    assert (run_s["value"], run_s["unit"], run_s["mark"]) == (None, "s", "partial")
    assert run_s["why"] == f"the run is missing {', '.join(missing)}: its pipeline_sum_s would sum only the steps that ran"
    assert (lean_s["value"], lean_s["mark"]) == (h["pipeline"]["Lean step_s"], "partial") and lean_s["value"] > 0
    assert run_s["shape"] == "K = 4096, N = 2" and got["measurement_files"] == ["out/recursion_e2e.json"]
    assert got["validation"] == {"status": "failed", "checker": "rec_profile.headline", "evidence": ["out/recursion_e2e.json"],
                                 "detail": RT.verdict(h, "?")}
    assert got["validation"]["detail"].startswith("**not a whole recursive run**")

    o = job(tmp_path / "b")
    vcap = o / "steps" / "vcap" / "r1"
    (vcap / "out").mkdir(parents=True)
    (vcap / "rc").write_text("0\n")
    (vcap / "time-step.txt").write_text(timev("0:05.00"))
    whole = RP.read(o)
    rc, got, _ = _ran(tmp_path / "whole", monkeypatch, whole, out=True)
    assert rc == 0 and [m["mark"] for m in got["measurements"]] == ["measured", "measured"]
    assert got["measurements"][0]["value"] == whole["headline"]["pipeline_sum_s"] > 0
    assert got["measurement_files"] == ["out/recursion_e2e.json"]
    assert got["validation"]["status"] == "failed" and "not accepted" in got["validation"]["detail"]


def test_a_job_that_wrote_no_result_records_why(tmp_path: Path, monkeypatch) -> None:
    rc, got, _ = _ran(tmp_path, monkeypatch, None, rc=143)
    assert rc == 143 and all(m["value"] is None and m["why"] == "job.sh wrote no result (exit 143)"
                             for m in got["measurements"])
    assert got["validation"]["status"] == "failed" and got["measurement_files"] == []


def test_outside_a_research_run_the_job_records_nothing(tmp_path: Path, monkeypatch) -> None:
    fake = _Job(tmp_path / "out", RP.read(job(tmp_path / "a")), 0)
    monkeypatch.setattr(REC.subprocess, "Popen", fake)
    monkeypatch.setattr(REC.signal, "signal", lambda sig, handler: None)
    monkeypatch.chdir(tmp_path)
    assert REC.main(["job"]) == 0 and not list(tmp_path.rglob(kit.RESULT))


def test_a_window_run_names_its_member_and_its_out_holds_no_file_of_the_windows_private_set(tmp_path: Path, monkeypatch) -> None:
    """With WINDOW the job is stage.sh, and its measurements' member is the served workload's and N from the staging's
    `subset`. Every file under out/ that holds a private file of the window (a copy under any name, a member's rows,
    the instance file, the prover's record) is deleted before the run's records are taken, and the validation fails
    naming each; the window's public files and the job's own stay."""
    window = tmp_path / "serving-rows"
    window.mkdir()
    rows, salt, public = b"rows and salts" * 64, b"index salt" * 8, b"b || c" * 32
    (window / "m0-private.bin").write_bytes(rows)
    (window / "index-salt.bin").write_bytes(salt)
    (window / "m0-public.bin").write_bytes(public)
    (window / "window.json").write_text("{}")
    doc = RP.read(job(tmp_path / "a"))
    doc["staged"] = {"subcircuit": "rope-head/d64/neox-bf16", "n": 16, "m": 25, "k_log": 22,
                     "subset": {"member": 0, "n": 880, "units": list(range(16))}}
    args = (f"WINDOW={window}", "MEMBER=0", "UNITS=0-15", "SUBCIRCUIT=rope-head/d64/neox-bf16", "SERVES=llama-3.2-1b-h200")
    leaked = {"steps/inner/r1/out/copy.tar": rows, "steps/a/m0-private.bin": b"subset rows", "steps/a/inst-16.bin": b"x",
              "steps/a/private.json": b"{}", "steps/b/salt": salt}
    kept = {"steps/a/pub-16.bin": public, "steps/a/public.json": b"{}"}
    rc, got, fake = _ran(tmp_path / "leak", monkeypatch, doc, args=args, files={**leaked, **kept})
    out = tmp_path / "leak" / "run" / "out"
    assert rc == 1 and fake.argv[1] == str(REC.HERE / "stage.sh")
    assert sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()) == sorted([*kept, REC.RESULT])
    assert got["validation"]["status"] == "failed" and got["validation"]["checker"] == "record.private_in"
    assert all(f"out/{rel}" in got["validation"]["detail"] for rel in leaked)
    assert {m["member"] for m in got["measurements"]} == {"llama-3.2-1b-h200-n16"}

    rc, got, _ = _ran(tmp_path / "clean", monkeypatch, doc, args=args, files=kept)
    assert rc == 0 and got["validation"]["checker"] == "rec_profile.headline"
    assert {m["member"] for m in got["measurements"]} == {"llama-3.2-1b-h200-n16"}
    assert REC.member({**doc, "staged": {**doc["staged"], "subset": None}}, "llama-3.2-1b-h200") is None
