"""`run.sh`: a placement value's ``${NAME}`` is filled from the environment the demo is launched in, so the window
run's gate reaches the challenger host at the address that environment holds; with NAME unset nothing starts and the
refusal names it, unless a FLAG sets that value; a run whose placement names no such variable reads none."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
#: the challenger host's address as a launch's environment holds it, here a documentation address
CHALLENGER_HOST = "192.0.2.21"


def run(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    """`run.sh ARGS` in a copy of the demo's tree whose workload is a stand-in (the real one is a fetched fixture), in
    this environment less the address's variable, with ``env`` over it: the completed run."""
    tree = tmp_path / "tree"
    if not tree.exists():
        (tree / "demos" / "vllm").mkdir(parents=True)
        for name in ("run.sh", "placement.toml"):
            shutil.copy(HERE / name, tree / "demos" / "vllm" / name)
        row = tomllib.loads((HERE / "placement.toml").read_text())["row"]
        workload = json.dumps({"model": "some/model", "case": "served", "engine": {"revision": "ab12"}}).encode()
        (tree / "adapters" / "vllm" / "workloads").mkdir(parents=True)
        (tree / "adapters" / "vllm" / "workloads" / f"{row}.json").write_bytes(workload)
        (tree / "adapters" / "vllm" / "workloads" / "SHA256SUMS").write_text(
            f"{hashlib.sha256(workload).hexdigest()}  {row}.json\n")
    held = {k: v for k, v in os.environ.items() if k != "VY_CHALLENGER_1_HOST"}
    return subprocess.run(["bash", str(tree / "demos" / "vllm" / "run.sh"), *args], capture_output=True, text=True,
                          check=False, env={**held, "HOME": str(tmp_path), **env})


def envs(got: subprocess.CompletedProcess) -> list[str]:
    """The ``--env`` values of the `research run` command a dry run printed, in order."""
    words = shlex.split(got.stdout)
    return [b for a, b in zip(words, words[1:], strict=False) if a == "--env"]


def test_the_window_runs_gate_is_at_the_address_the_launchers_environment_holds(tmp_path):
    got = run(tmp_path, "window", "--dry-run", VY_CHALLENGER_1_HOST=CHALLENGER_HOST)
    assert got.returncode == 0, got.stderr
    assert f"H1_GATE=ssh -i ~/.ssh/vllm_gate_ed25519 -o BatchMode=yes research@{CHALLENGER_HOST}" in envs(got)
    assert "${" not in got.stdout


@pytest.mark.parametrize("env", [{}, {"VY_CHALLENGER_1_HOST": ""}])
def test_without_the_address_the_window_run_starts_nothing_and_names_the_variable(tmp_path, env):
    got = run(tmp_path, "window", "--dry-run", **env)
    assert (got.returncode, got.stdout) == (2, "")
    assert got.stderr.strip() == ("run.sh: VY_CHALLENGER_1_HOST is unset: placement.toml's H1_GATE names it, so set it "
                                  "where the demo is launched, or give --env H1_GATE=VALUE")


def test_a_value_the_flags_set_reads_no_variable_and_the_flags_value_is_the_runs(tmp_path):
    got = run(tmp_path, "window", "--dry-run", "--env", "H1_GATE=", "--env", "WC_TIMEOUT=45m")
    assert got.returncode == 0, got.stderr
    assert [e for e in envs(got) if e.startswith("H1_GATE=")] == ["H1_GATE="] and "WC_TIMEOUT=45m" in envs(got)


@pytest.mark.parametrize("which", ["commit", "serve"])
def test_a_run_whose_placement_names_no_variable_reads_none(tmp_path, which):
    got = run(tmp_path, which, "--dry-run")
    assert got.returncode == 0 and "--on vy-dc-h200 " in got.stdout, got.stderr
