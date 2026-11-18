"""A pod job fails on a failed selftest: `pod/selftest.sh` on a stub prover that prints each kind of verdict, and every
selftest the pod scripts start goes through it, with the script ending in `selftest_end`."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

POD = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proofs" / "flock" / "prover" / "pod"
PASS = 'NEG\t{"case":"honest","pass":true}\nSELFTEST\t{"gpu":true,"all_pass":true}\n'
FAIL = ('NEG\t{"case":"honest","pass":true}\nNEG\t{"case":"gpu_proofs_match_cpu","expect":"x","pass":false}\n'
        'SELFTEST\t{"gpu":true,"all_pass":false}\n')


def job(tmp_path: Path, stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    stub = tmp_path / "prover"
    stub.write_text(f"#!/usr/bin/env bash\nprintf '%s' '{stdout}'\nexit {rc}\n")
    stub.chmod(0o755)
    script = (f". {POD / 'selftest.sh'}\n"
              f"selftest_run rope/gpu {tmp_path}/out.txt {tmp_path}/out.err {stub} selftest --gpu\n"
              "selftest_end\necho END\n")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_a_passing_selftest_lets_the_job_go_on(tmp_path: Path):
    r = job(tmp_path, PASS)
    assert r.returncode == 0 and r.stdout.rstrip().endswith("END"), r.stdout + r.stderr


@pytest.mark.parametrize("stdout,rc,why", [
    (FAIL, 0, "rope/gpu (not all passed: gpu_proofs_match_cpu)"),
    (PASS, 3, "rope/gpu (exit 3)"),
    ('NEG\t{"case":"honest","pass":true}\n', 0, "rope/gpu (no SELFTEST line)"),
])
def test_a_failed_selftest_fails_the_job_naming_what_failed(tmp_path: Path, stdout: str, rc: int, why: str):
    r = job(tmp_path, stdout, rc)
    assert r.returncode == 1 and f"SELFTEST FAILED: {why}" in r.stdout and "END" not in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("name", ["20-gpu-link.sh", "60-circuit.sh"])
def test_each_selftest_a_pod_script_starts_is_judged(name: str):
    text = (POD / name).read_text().replace("\\\n", " ")
    bare = [line for line in text.splitlines()
            if re.search(r"\$B selftest\b", line) and "selftest_run" not in line and not line.lstrip().startswith("#")]
    assert bare == [], f"{name}: start each selftest as `selftest_run LABEL OUT ERR $B selftest ...` (pod/selftest.sh)"
    assert "pod/selftest.sh" in text and text.rstrip().endswith("selftest_end"), (
        f"{name}: source pod/selftest.sh and end with `selftest_end`, so a failed selftest fails the job")
