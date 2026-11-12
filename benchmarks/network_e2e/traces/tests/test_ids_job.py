"""`ids.sh` with a stand-in interpreter that records its command line, says it listens and then waits for the job's
SIGTERM: what the job starts the id front with, that the command it is given runs only once the front listens, and that
the front is stopped after it."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
FRONT = ["-m", "verity_vllm.pipeline.cli", "ids-front"]
STUB = """\
#!{python}
import json, os, signal, sys

with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if os.environ.get("STUB_FRONT") == "dies":
    sys.exit(7)
print(json.dumps({{"listening": ["127.0.0.1", 18746]}}), flush=True)
signal.pause()
"""


def job(tmp_path: Path, *command: str, **env: str) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
    stub = tmp_path / "python"
    stub.write_text(STUB.format(python=sys.executable))
    stub.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    p = subprocess.run(["bash", str(HERE / "ids.sh"), *command], cwd=ROOT, capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PY": str(stub), "STUB_LOG": str(log), **env})
    return p, [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


def test_the_job_runs_its_command_once_the_front_listens_then_stops_the_front(tmp_path):
    run, seen = tmp_path / "run", tmp_path / "seen"
    p, calls = job(tmp_path, "sh", "-c", 'cp "$STUB_LOG" "$1"; exit 3', "sh", str(seen),
                   UPSTREAM="127.0.0.1:18745", RESEARCH_RUN_DIR=str(run))
    assert p.returncode == 3, p.stdout + p.stderr
    assert calls == [[*FRONT, "--upstream", "127.0.0.1:18745", "--listen", "127.0.0.1:18746", "--id-payload", "4087",
                      "--log", f"{run}/out/front.jsonl"]]
    assert [json.loads(x) for x in seen.read_text().splitlines()] == calls and (run / "out").is_dir()
    assert 'ids: the front {"listening": ["127.0.0.1", 18746]} before 127.0.0.1:18745' in p.stdout


def test_the_job_fails_without_running_its_command_when_the_front_never_listens(tmp_path):
    seen = tmp_path / "seen"
    p, calls = job(tmp_path, "sh", "-c", 'cp "$STUB_LOG" "$1"', "sh", str(seen), UPSTREAM="h:1", STUB_FRONT="dies")
    assert p.returncode == 1 and "the front ended before it listened" in p.stdout
    assert len(calls) == 1 and not seen.exists()


def test_without_a_command_the_job_is_the_front_for_the_cells_it_is_given(tmp_path):
    p, calls = job(tmp_path, UPSTREAM="node:18745", LISTEN="0.0.0.0:9", CELL_BYTES="512", LOG=str(tmp_path / "f.jsonl"),
                   STUB_FRONT="dies")
    assert p.returncode == 7
    assert calls == [[*FRONT, "--upstream", "node:18745", "--listen", "0.0.0.0:9", "--id-payload", "503", "--log",
                      str(tmp_path / "f.jsonl")]]
    p, calls = job(tmp_path, STUB_FRONT="dies")
    assert p.returncode != 0 and "UPSTREAM: HOST:PORT of the server" in p.stderr and len(calls) == 1
