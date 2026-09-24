"""`research pods health`: the reference merge (the best pod per metric) and the ssh wrapper (the pod-side command, the DEGRADED
exit code, `--record` writing the worktree's pod_health_ref.json)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from research.pods import health, part


def _rep(sku: str = "rtx-4090", enc: float = 2.0, mm: float = 150.0, verdict: str = "OK") -> dict:
    return {"sku": sku, "bench": {"encode_ms": enc, "matmul_tflops": mm}, "verdict": verdict}


def test_the_reference_keeps_the_best_pod_per_metric() -> None:
    refs: dict = {}
    assert health.merge(refs, _rep(), "vy-a", "t1") == ["encode_ms", "matmul_tflops"]
    assert health.merge(refs, _rep(enc=2.5, mm=140.0), "vy-b", "t2") == []
    assert health.merge(refs, _rep(enc=1.8, mm=140.0), "vy-c", "t3") == ["encode_ms"]
    assert refs["skus"]["rtx-4090"] == {"encode_ms": {"value": 1.8, "pod": "vy-c", "utc": "t3"},
                                        "matmul_tflops": {"value": 150.0, "pod": "vy-a", "utc": "t1"}}
    assert health.merge(refs, {"sku": "h100-sxm", "bench": None}, "vy-d", "t4") == []


def test_health_runs_pod_health_over_ssh_and_record_writes_the_worktree_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                                  capsys: pytest.CaptureFixture) -> None:
    real, calls, answer = subprocess.run, [], {"rc": 0, "rep": _rep()}

    def run(argv, **kw):
        if argv[0] != "ssh":
            return real(argv, **kw)
        calls.append(argv)
        return subprocess.CompletedProcess(argv, answer["rc"], json.dumps(answer["rep"]) if kw.get("capture_output") else None, "")

    monkeypatch.setattr(health.connect, "ssh_base", lambda target: ["ssh", "root@pod"])
    monkeypatch.setattr(subprocess, "run", run)
    assert part.main(["health", "vy-x", "--seconds", "5"]) == 0
    script = calls[-1][2]
    assert calls[-1][:2] == ["ssh", "root@pod"] and script.startswith(". /workspace/env.sh 2>/dev/null; cd /workspace/src && ")
    assert script.endswith("-m backends.direct.ligero.pod_health --seconds 5.0")
    answer["rc"] = 3
    assert part.main(["health", "vy-x", "--no-bench"]) == 3 and calls[-1][2].endswith("pod_health --no-bench")

    real(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / health.REF).parent.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    answer["rc"] = 0
    assert part.main(["health", "vy-x", "--record"]) == 0 and calls[-1][2].endswith("pod_health --json")
    doc = json.loads((tmp_path / health.REF).read_text())
    assert doc["schema"] == "pod-health-ref/v1" and doc["skus"]["rtx-4090"]["encode_ms"]["value"] == 2.0
    assert doc["skus"]["rtx-4090"]["matmul_tflops"]["pod"] == "vy-x"
    assert "rtx-4090 reference: set encode_ms, matmul_tflops" in capsys.readouterr().out
    answer["rep"] = _rep(enc=9.0, verdict="DEGRADED")
    assert part.main(["health", "vy-y", "--record"]) == 3
    assert json.loads((tmp_path / health.REF).read_text())["skus"]["rtx-4090"]["encode_ms"]["pod"] == "vy-x"
    assert "kept (it is better)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        part.main(["health", "vy-x", "--record", "--no-bench"])
