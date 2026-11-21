"""A pod job fails on a failed selftest: `pod/selftest.sh` on a stub prover that prints each kind of verdict, and every
selftest any pod script starts (found by a scan of `pod/`) goes through it, with the script calling `selftest_end`."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

POD = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proof" / "flock" / "prover" / "pod"
PASS = 'NEG\t{"case":"honest","pass":true}\nSELFTEST\t{"gpu":true,"all_pass":true}\n'
FAIL = ('NEG\t{"case":"honest","pass":true}\nNEG\t{"case":"gpu_proofs_match_cpu","expect":"x","pass":false}\n'
        'SELFTEST\t{"gpu":true,"all_pass":false}\n')


def job(tmp_path: Path, stdout: str, rc: int = 0, *args: str) -> subprocess.CompletedProcess:
    stub = tmp_path / "prover"
    stub.write_text(f"#!/usr/bin/env bash\nprintf '%s' '{stdout}'\nexit {rc}\n")
    stub.chmod(0o755)
    script = (f". {POD / 'selftest.sh'}\n"
              f"selftest_run rope/gpu {tmp_path}/out.txt {tmp_path}/out.err {stub} selftest --gpu {' '.join(args)}\n"
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


@pytest.mark.parametrize("stdout,rc,ok", [
    ('NEG\t{"case":"prover_is_deterministic","pass":true}\n', 0, True),
    ('NEG\t{"case":"prover_is_deterministic","pass":false}\n', 0, False),
    ("", 0, False),
])
def test_an_only_case_is_judged_by_its_one_neg_line(tmp_path: Path, stdout: str, rc: int, ok: bool):
    r = job(tmp_path, stdout, rc, "--only", "prover_is_deterministic")
    assert (r.returncode == 0) == ok and ("SELFTEST FAILED: rope/gpu (case" in r.stdout) != ok, r.stdout + r.stderr


#: a prover's `selftest` subcommand (`$B selftest --vus 8`, `.../flock-circuit-selftest selftest $Z ...`), not a flag or a mode
STARTS = re.compile(r"(?<![\w/.$=+-])selftest(?:-\w+)?\s+(?:--|\$)")


def starts(path: Path) -> list[str]:
    lines = path.read_text().replace("\\\n", " ").splitlines()
    return [line for line in lines if STARTS.search(line) and not line.lstrip().startswith("#")]


SCRIPTS = sorted(p.name for p in POD.glob("*.sh") if starts(p))


def test_the_scan_finds_the_selftest_scripts():
    assert {"20-gpu-link.sh", "60-circuit.sh", "86-zk-cpu-steps.sh"} <= set(SCRIPTS), SCRIPTS


@pytest.mark.parametrize("name", SCRIPTS)
def test_each_selftest_a_pod_script_starts_is_judged(name: str):
    path = POD / name
    bare = [line for line in starts(path) if "selftest_run" not in line]
    assert bare == [], f"{name}: start each selftest as `selftest_run LABEL OUT ERR CMD...` (pod/selftest.sh)"
    text = path.read_text()
    assert "pod/selftest.sh" in text and re.search(r"^selftest_end$", text, re.M), (
        f"{name}: source pod/selftest.sh and call `selftest_end` before it exits, so a failed selftest fails the job")
