"""verity_vllm/ops/epoch_row.sh: every stop of a row -- a stage deadline, STOP_AFTER, a signal, the custody guard -- ends inside the job
the same way: the Build and records stored, `research data preserved` on them, then END.

CPU only: the script runs from this tree with stand-ins for the stage runner (`$PY -m verity_vllm.pipeline.cli row run ...`), the
bootstrap (`EPOCH_BOOTSTRAP`) and the store CLI (`RESEARCH_CLI`), in a fake `research run` job (run dir, status.json, custody dir)."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

VLLM = Path(__file__).resolve().parents[2]
SCRIPT = VLLM / "verity_vllm" / "ops" / "epoch_row.sh"
BASH = shutil.which("bash") or "/bin/bash"
ROW = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager"
ART = "art:" + "ab" * 32

STAGE = """#!/bin/bash
# stand-in for $PY: `-m verity_vllm.pipeline.cli row run ROW ROLE REPO REV --stages S` is a stage whose behaviour FAKE_<S> names
# (ok | sleep: FAKE_SLEEP seconds); anything else is the real interpreter
if [ "$1 $2 $3 $4" = "-m verity_vllm.pipeline.cli row run" ]; then
  s=${10}; d=$SWEEP_DIR/$5; mkdir -p "$d"; echo "stage $s" >> "$CALLS"
  if [ "$s" = build ]; then
    echo '{"request": {"program_digest": "p0"}}' > "$d/build_summary.json"
    echo '{"manifest_digest": "m0", "query": {"required_families": ["weights"]}}' > "$d/manifest.json"
  fi
  echo "$s PASS rc=0" >> "$d/stages.txt"
  case "$(eval echo "\\${FAKE_${s^^}:-ok}")" in sleep) exec sleep "${FAKE_SLEEP:-60}";; esac
  exit 0
fi
exec "$REAL_PY" "$@"
"""
CLI = """#!/bin/bash
# stand-in for `python3 -m research`: logs every call; `data put` prints an artifact id
echo "$*" >> "$CALLS"
[ "$1 $2" = "data put" ] && echo '{"artifact": "%s"}'
exit 0
""" % ART


def _utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(tmp_path: Path, *, timeout_s: float = 7200, key_after_run_s: float = 7200) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    root = tmp_path / "research"
    run = root / "runs" / "r1"
    custody = root / "requests" / "r1" / "custody"
    run.mkdir(parents=True)
    custody.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"transitions": [{"state": "running", "started_utc": _utc(now), "timeout_s": timeout_s}]}))
    (custody / "store.toml").write_text("[remote]\n")
    (custody / "cred.json").write_text(json.dumps({"AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": "s", "AWS_SESSION_TOKEN": "t",
                                                   "expires_utc": _utc(now + timedelta(seconds=timeout_s + key_after_run_s))}))
    b = tmp_path / "bin"
    b.mkdir()
    for name, text in (("py", STAGE), ("research", CLI), ("bootstrap", "#!/bin/bash\nexit 0\n")):
        (b / name).write_text(text)
        (b / name).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DEADLINE_", "STOP_AFTER", "JOB_DEADLINE", "VERITY_"))}
    env.update({"RESEARCH_RUN_DIR": str(run), "RESEARCH_RUN_ID": "r1", "RESEARCH_SOURCE_SHA": "e0", "EPOCH_SHA": "e0", "ROW": ROW,
                "ROWNUM": "101", "CLASS": "GREEN", "PY": str(b / "py"), "REAL_PY": sys.executable, "RESEARCH_CLI": str(b / "research"),
                "EPOCH_BOOTSTRAP": str(b / "bootstrap"), "SWEEP_DIR": str(tmp_path / "sweep"), "RESEARCH_STORE": str(tmp_path / "store"),
                "CALLS": str(tmp_path / "calls.txt"), "FAKE_SLEEP": "60"})
    return env


def _run(env: dict[str, str], timeout: float = 90) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=timeout)


def _progress(env: dict[str, str]) -> str:
    return (Path(env["RESEARCH_RUN_DIR"]) / "evidence" / "progress.txt").read_text()


def _calls(env: dict[str, str]) -> list[str]:
    p = Path(env["CALLS"])
    return p.read_text().splitlines() if p.exists() else []


def _stored_then_preserved(env: dict[str, str]) -> None:
    calls = _calls(env)
    puts = [i for i, c in enumerate(calls) if c.startswith("data put")]
    pres = [i for i, c in enumerate(calls) if c.startswith("data preserved")]
    assert len(puts) == 2 and len(pres) == 1 and pres[0] > max(puts), calls
    assert calls[pres[0]] == f"data preserved {ART} {ART}"
    prog = _progress(env)
    assert "store rc=0" in prog and "PRESERVED-CHECK" in (Path(env["RESEARCH_RUN_DIR"]) / "evidence" / "store.log").read_text()
    assert prog.rstrip().endswith("END"), prog


def test_script_parses() -> None:
    for name in ("epoch_row.sh", "epoch_store.sh", "epoch_failfast.sh"):
        r = subprocess.run([BASH, "-n", str(SCRIPT.parent / name)], capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stderr)


def test_a_stage_deadline_stops_the_stage_stores_preserves_and_exits(tmp_path: Path) -> None:
    env = _job(tmp_path)
    env.update({"FAKE_BUILD": "sleep", "DEADLINE_BUILD": _utc(datetime.now(timezone.utc) + timedelta(seconds=4))})
    t0 = time.monotonic()
    r = _run(env)
    assert r.returncode == 124, (r.stdout, r.stderr)
    assert time.monotonic() - t0 < 45
    prog = _progress(env)
    assert "STOP build: deadline" in prog and "stage match" not in _calls(env)
    _stored_then_preserved(env)


def test_the_run_timeout_less_the_store_reserve_is_the_job_deadline(tmp_path: Path) -> None:
    env = _job(tmp_path, timeout_s=8)
    env.update({"FAKE_BUILD": "sleep", "STORE_RESERVE_S": "4"})
    r = _run(env)
    assert r.returncode == 124, (r.stdout, r.stderr)
    assert "STOP build: deadline" in _progress(env)
    _stored_then_preserved(env)


def test_stop_after_build_skips_the_match(tmp_path: Path) -> None:
    env = _job(tmp_path)
    env["STOP_AFTER"] = "build"
    r = _run(env)
    assert r.returncode == 23, (r.stdout, r.stderr)
    assert "STOP after build (STOP_AFTER=build)" in _progress(env)
    assert [c for c in _calls(env) if c.startswith("stage")] == ["stage build"]
    _stored_then_preserved(env)


def test_a_signal_mid_stage_stores_preserves_and_exits(tmp_path: Path) -> None:
    env = _job(tmp_path)
    env["FAKE_BUILD"] = "sleep"
    p = subprocess.Popen([BASH, str(SCRIPT)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 30
    while "stage build" not in _calls(env):
        assert time.monotonic() < deadline and p.poll() is None, p.communicate()
        time.sleep(0.2)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=60)
    assert p.returncode == 143
    assert "STOP signal TERM" in _progress(env)
    _stored_then_preserved(env)


def test_a_custody_key_shorter_than_the_timeout_plus_an_hour_refuses_before_any_stage(tmp_path: Path) -> None:
    env = _job(tmp_path, key_after_run_s=1800)
    r = _run(env)
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert "STOP guard: the custody key expires" in _progress(env)
    assert not [c for c in _calls(env) if c.startswith(("stage", "data"))]
