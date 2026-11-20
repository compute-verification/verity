"""No request value reaches a job's $OUT, which custody uploads: the records endpoint step names no request, and a
runner that publishes a network attester run stops when the gate's publisher fails rather than going on."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = (Path(__file__).resolve().parents[3] / "benchmarks" / "network_traces")
LOG = 'log() { echo "$*" >> "$OUT/job.log"; }\n'


def block(script: str, first: str, last: str) -> str:
    """``script``'s lines from the one starting with ``first`` through the next one starting with ``last``."""
    lines = (HERE / script).read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(first))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith(last))
    return "\n".join(lines[i:j + 1]) + "\n"


def run(code: str, **env: Path | str) -> subprocess.CompletedProcess:
    return subprocess.run(["env", *(f"{k}={v}" for k, v in env.items()), "bash", "-c", LOG + code],
                          capture_output=True, text=True)


def test_the_records_endpoint_step_keeps_its_verdict_and_counts_of_windows_alone():
    """The endpoint step (chain_record.py's, from the private report) names no request and counts no window's
    requests: the attested I/O's violations by their count, the advice's binding by whether it is bound and rebuilt."""
    sys.path.insert(0, str(HERE))
    import chain_record as cr

    bits = {"bound": False, "rebuilt": False, "bound_why": "2 certified requests are in none (s0-needle, s1-needle)",
            "commit_windows": [{"window": 0, "requests": 7, "bound": True, "rebuilt": True},
                               {"window": 1, "requests": 5, "bound": False, "why": "s1-needle"}]}
    for io in ({"accepted": True}, {"accepted": False, "violations": [[3, "a request of s0-needle"], [9, "x"]]}):
        step = cr.network_attester_endpoint_step({"warden_run": "r", "io": io, "chain": {"bits": bits}})
        assert step["verdict"] == "refused" and "needle" not in json.dumps(step) and '"requests"' not in json.dumps(step)
        assert step["advice"] == {"bound": False, "rebuilt": False}
        assert step["io"] == {"accepted": io["accepted"], "violations": len(io.get("violations") or [])}


def test_each_runner_fails_by_name_when_the_publisher_fails(tmp_path):
    """A runner that publishes a network attester run stops, by name, when the publisher exits nonzero, rather than
    going on with an empty outcome.json."""
    (py := tmp_path / "py").write_text("#!/bin/sh\nexit 3\n")
    py.chmod(0o755)
    for script, first in (("capture_loopback.sh", 'mkdir -p "$OUT/run"'),):
        out = tmp_path / script / "OUT"
        out.mkdir(parents=True)
        got = run(block(script, first, first.split("mkdir")[0] + '"$PY" -m') + "echo went on\n",
                  W=tmp_path / script / "W", RUN=tmp_path / script / "run", OUT=out, PY=py)
        assert got.returncode == 6 and "went on" not in got.stdout, (script, got.returncode, got.stdout)
        assert "the publisher (auditor.py publish) exited 3" in (out / "job.log").read_text(), script
