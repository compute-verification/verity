"""`gate_lease.sh`: the prover's lease with a deadline asks nothing once LEASE_BY has passed, runs run.sh once inside a
granted lease with what is left until LEASE_BY as the agent's wait, and never reruns it, whatever it exits with. The lease
is a stub gpu-lease (GPU_LEASE) that grants at once; run.sh a stub that exits RC."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

LEASE = Path(__file__).resolve().parents[1] / "gate_lease.sh"


def run(tmp_path: Path, lease_by: str, rc: int = 0) -> tuple[subprocess.CompletedProcess, list[str]]:
    tree, out = tmp_path / "tree", tmp_path / "run"
    (tree / "benchmarks/remote_session").mkdir(parents=True)
    (tree / "benchmarks/remote_session/run.sh").write_text(f"exit {rc}\n")
    stub = tmp_path / "gpu-lease"
    stub.write_text(f'#!/bin/bash\necho "$GPU_LEASE_AGENT_WAIT_S $GPU_LEASE_UUID_TABLE $*" >> {tmp_path}/calls\n'
                    'while [ "$1" != -- ]; do shift; done\nshift\nexec "$@"\n')
    stub.chmod(0o755)
    env = {**os.environ, "RESEARCH_RUN_DIR": str(out), "LEASE_BY": lease_by, "GPU_LEASE": str(stub),
           "GPU_LEASE_UUID_TABLE": "GPU-a,GPU-b"}
    p = subprocess.run(["bash", str(LEASE), "8", "--max-min", "100", "--mem-gb", "256"], cwd=tree, env=env,
                       capture_output=True, text=True)
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return p, calls


def test_past_lease_by_nothing_is_asked(tmp_path):
    p, calls = run(tmp_path, "2020-01-01T00:00Z")
    assert p.returncode == 75 and "no 8 GPUs leased by 2020-01-01T00:00:00Z: nothing ran" in p.stdout
    assert calls == [] and not (tmp_path / "run/out/leased").exists()


@pytest.mark.parametrize("rc", [0, 75, 3])
def test_a_granted_lease_runs_run_sh_once(tmp_path, rc):
    p, calls = run(tmp_path, "2100-01-01T00:00Z", rc)
    assert p.returncode == rc, p.stdout + p.stderr
    assert len(calls) == 1
    wait, table, args = calls[0].split(" ", 2)
    assert int(wait) > 0 and table == "GPU-a,GPU-b" and args.startswith("8 --max-min 100 --mem-gb 256 -- bash -c")
    assert (tmp_path / "run/out/leased").exists()
