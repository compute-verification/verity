"""`run.sh`: a placement value's ``${NAME}`` is filled from the environment the run is launched in, so the window
run's gate reaches the challenger host at the address that environment holds; with NAME unset nothing starts and the
refusal names it, unless a FLAG sets that value; a run whose placement names no such variable reads none. The window
run takes the gate's coins by default, and development coins only with --development, which the run's environment says."""
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

HERE = Path(__file__).resolve().parents[5] / "adapters" / "vllm" / "tools" / "workload"
#: where the workload's scripts sit in the tree
WORKLOAD_RUN = Path("adapters/vllm/tools/workload")
#: the challenger host's address as a launch's environment holds it, here a documentation address
CHALLENGER_HOST = "192.0.2.21"


def run(tmp_path: Path, *args: str, key: bool = True, **env: str) -> subprocess.CompletedProcess:
    """`run.sh ARGS` in a copy of the tree whose workload is a stand-in (the real one is a fetched fixture) and, with
    ``key``, a stand-in challenger key record at its root, in this environment less the address's variable, with
    ``env`` over it: the completed run."""
    tree = tmp_path / "tree"
    if not tree.exists():
        (tree / WORKLOAD_RUN).mkdir(parents=True)
        for name in ("run.sh", "placement.toml"):
            shutil.copy(HERE / name, tree / WORKLOAD_RUN / name)
        row = tomllib.loads((HERE / "placement.toml").read_text())["row"]
        workload = json.dumps({"model": "some/model", "case": "served", "engine": {"revision": "ab12"}}).encode()
        (tree / "adapters" / "vllm" / "workloads").mkdir(parents=True)
        (tree / "adapters" / "vllm" / "workloads" / f"{row}.json").write_bytes(workload)
        (tree / "adapters" / "vllm" / "workloads" / "SHA256SUMS").write_text(
            f"{hashlib.sha256(workload).hexdigest()}  {row}.json\n")
    (tree / "challenger-key.json").unlink(missing_ok=True)
    if key:
        (tree / "challenger-key.json").write_text('{"schema": "challenger-key/v1"}\n')
    held = {k: v for k, v in os.environ.items() if k != "VY_CHALLENGER_1_HOST"}
    return subprocess.run(["bash", str(tree / WORKLOAD_RUN / "run.sh"), *args], capture_output=True, text=True, check=False,
                          env={**held, "HOME": str(tmp_path), **env})


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
    assert (got.returncode, got.stdout) == (1, "")
    assert got.stderr.strip() == ("run.sh: VY_CHALLENGER_1_HOST is unset: placement.toml's H1_GATE names it, so set it "
                                  "where the run is launched, or give --env H1_GATE=VALUE")


def test_a_value_the_flags_set_reads_no_variable_and_the_flags_value_is_the_runs(tmp_path):
    got = run(tmp_path, "window", "--dry-run", "--development", "--env", "H1_GATE=", "--env", "WC_TIMEOUT=45m")
    assert got.returncode == 0, got.stderr
    assert [e for e in envs(got) if e.startswith("H1_GATE=")] == ["H1_GATE="] and "WC_TIMEOUT=45m" in envs(got)


@pytest.mark.parametrize("which", ["commit", "serve"])
def test_a_run_whose_placement_names_no_variable_reads_none(tmp_path, which):
    got = run(tmp_path, which, "--dry-run")
    assert got.returncode == 0 and "--on vy-dc-h200 " in got.stdout, got.stderr


def test_the_window_run_takes_the_gates_coins_by_default(tmp_path):
    got = run(tmp_path, "window", "--dry-run", VY_CHALLENGER_1_HOST=CHALLENGER_HOST)
    assert got.returncode == 0, got.stderr
    assert {f"PP_CHALLENGER=gate:research@{CHALLENGER_HOST}", "PP_PEER=vy-dc-h200",
            "PP_GATE_KEY=~/.ssh/vllm_gate_ed25519"} <= set(envs(got))
    assert not [e for e in envs(got) if e.startswith("DEVELOPMENT=")]
    words = shlex.split(got.stdout)
    assert ("--send", "challenger-key.json") in zip(words, words[1:], strict=False)
    after = words[words.index("--") + 1:]
    assert after[-2:] == ["bash", "benchmarks/one_stage/served_verdicts.sh"]
    wrapped = subprocess.run([*after[:-2], "bash", "-c", 'printf %s "$PP_CHALLENGER_KEY"'], capture_output=True, text=True,
                             check=True, env={"PATH": os.environ["PATH"], "RESEARCH_RUN_DIR": "/runs/r1"})
    assert wrapped.stdout == "/runs/r1/inputs/challenger-key.json"


def test_the_gates_coins_need_the_key_record_at_the_trees_root(tmp_path):
    got = run(tmp_path, "window", "--dry-run", key=False, VY_CHALLENGER_1_HOST=CHALLENGER_HOST)
    assert (got.returncode, got.stdout) == (1, "")
    assert "sends challenger-key.json, which is not at the tree's root" in got.stderr and "--development" in got.stderr


def test_development_coins_are_chosen_and_the_run_says_so(tmp_path):
    got = run(tmp_path, "window", "--development", "--dry-run", key=False, VY_CHALLENGER_1_HOST=CHALLENGER_HOST)
    assert got.returncode == 0, got.stderr
    assert {"PP_CHALLENGER=local", "DEVELOPMENT=1"} <= set(envs(got))
    assert not [e for e in envs(got) if e.startswith(("PP_PEER=", "PP_GATE_KEY="))] and "--send" not in got.stdout
    assert shlex.split(got.stdout)[-3:] == ["--", "bash", "benchmarks/one_stage/served_verdicts.sh"]


def test_development_on_a_run_without_coins_is_refused(tmp_path):
    got = run(tmp_path, "commit", "--development", "--dry-run")
    assert (got.returncode, got.stdout) == (1, "")
    assert got.stderr.strip() == "run.sh: the commit run takes no coins, so --development chooses nothing: drop it"
