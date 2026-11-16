"""`run.sh`: a placement value's ``${NAME}`` is filled from the environment the run is launched in or a FLAG's
``--env``, so the window run's gate reaches the challenger host at the address that environment holds and its card,
cores and end are the launch's; with NAME unset nothing starts and the refusal names it, unless a FLAG sets that value;
a run whose placement names no such variable reads none. The window run takes the gate's coins by default, with main's
first-parent line sent beside it, and development coins only with --development, which the run's environment says. The
compose run sends the window's hand-off."""
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

HERE = Path(__file__).resolve().parents[5] / "adapters" / "vllm" / "tools" / "served_run"
#: where the served run's scripts sit in the tree
SERVED_RUN = Path("adapters/vllm/tools/served_run")
#: the challenger host's address as a launch's environment holds it, here a documentation address
CHALLENGER_HOST = "192.0.2.21"
#: the window run's parameters, as a launch gives them
ROW_OF_CARDS = {"CARD": "5", "CORES": "0-11", "END": "2026-11-17T03:00Z"}
PLACEMENT = tomllib.loads((HERE / "placement.toml").read_text())
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]


def run(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    """`run.sh ARGS` in a copy of the tree whose workload is a stand-in (the real one is a fetched fixture), a git
    repository whose origin's main is its one commit, in this environment less the variables placement.toml names,
    with ``env`` over it: the completed run."""
    tree = tmp_path / "tree"
    if not tree.exists():
        (tree / SERVED_RUN).mkdir(parents=True)
        for name in ("run.sh", "placement.toml"):
            shutil.copy(HERE / name, tree / SERVED_RUN / name)
        workload = json.dumps({"model": "some/model", "case": "served", "engine": {"revision": "ab12"}}).encode()
        (tree / "adapters" / "vllm" / "workloads").mkdir(parents=True)
        (tree / "adapters" / "vllm" / "workloads" / f"{PLACEMENT['row']}.json").write_bytes(workload)
        (tree / "adapters" / "vllm" / "workloads" / "SHA256SUMS").write_text(
            f"{hashlib.sha256(workload).hexdigest()}  {PLACEMENT['row']}.json\n")
        for cmd in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "tree"],
                    ["clone", "-q", "--bare", ".", str(tmp_path / "origin.git")], ["remote", "add", "origin", str(tmp_path / "origin.git")]):
            subprocess.run([*GIT, *cmd], cwd=tree, check=True, capture_output=True)
    held = {k: v for k, v in os.environ.items() if k not in {"VY_CHALLENGER_1_HOST", "HANDOFF", *ROW_OF_CARDS}}
    return subprocess.run(["bash", str(tree / SERVED_RUN / "run.sh"), *args], capture_output=True, text=True, check=False,
                          env={**held, "HOME": str(tmp_path), **env})


def envs(got: subprocess.CompletedProcess) -> list[str]:
    """The ``--env`` values of the `research run` command a dry run printed, in order."""
    words = shlex.split(got.stdout)
    return [b for a, b in zip(words, words[1:], strict=False) if a == "--env"]


def sends(got: subprocess.CompletedProcess) -> list[str]:
    """The ``--send`` paths of the `research run` command a dry run printed, in order."""
    words = shlex.split(got.stdout)
    return [b for a, b in zip(words, words[1:], strict=False) if a == "--send"]


def test_the_window_run_serves_on_node_a_on_the_launchs_card_and_cores(tmp_path):
    got = run(tmp_path, "window", "--dry-run", VY_CHALLENGER_1_HOST=CHALLENGER_HOST, **ROW_OF_CARDS)
    assert got.returncode == 0, got.stderr
    words = shlex.split(got.stdout)
    assert words[words.index("--on") + 1] == "vy-spot-glm53"
    assert {"CARD=5", "CORES=0-11", "END=2026-11-17T03:00Z", "H1_GATE=", "COMPOSITE=", "V3C_CLASS=",
            f"HIDDEN_SO={PLACEMENT['node']['vy-spot-glm53']['HIDDEN_SO']}"} <= set(envs(got))
    assert words[words.index("--") + 1:] == ["taskset", "-c", "0-11", "bash", "adapters/vllm/tools/served_run/window.sh"]
    assert f"adapters/vllm/workloads/{PLACEMENT['row']}.json" in sends(got)
    assert "${" not in got.stdout


def test_the_window_run_takes_the_gates_coins_by_default_with_mains_line(tmp_path):
    got = run(tmp_path, "window", "--dry-run", VY_CHALLENGER_1_HOST=CHALLENGER_HOST, **ROW_OF_CARDS)
    assert got.returncode == 0, got.stderr
    assert {f"PP_CHALLENGER=gate:research@{CHALLENGER_HOST}", "PP_PEER=vy-spot-glm53", "PP_GATE_KEY=~/.ssh/vllm_lane_ed25519",
            "PP_CHALLENGER_KEY=tests/benchmarks/pouw/fixtures/challenger-key.json"} <= set(envs(got))
    assert not [e for e in envs(got) if e.startswith("DEVELOPMENT=")]
    main = [Path(s) for s in sends(got) if s.endswith("/main.txt")]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path / "tree", capture_output=True, text=True, check=True)
    assert len(main) == 1 and main[0].read_text() == head.stdout


@pytest.mark.parametrize("missing", ["VY_CHALLENGER_1_HOST", *ROW_OF_CARDS])
def test_without_an_address_or_a_parameter_the_window_run_starts_nothing_and_names_it(tmp_path, missing):
    given = {"VY_CHALLENGER_1_HOST": CHALLENGER_HOST, **ROW_OF_CARDS}
    del given[missing]
    got = run(tmp_path, "window", "--dry-run", **given)
    assert (got.returncode, got.stdout) == (1, "")
    assert got.stderr.startswith(f"run.sh: {missing} is unset: placement.toml's ") and f"--env {missing}=VALUE" in got.stderr


def test_a_value_the_flags_set_reads_no_variable_and_the_flags_value_is_the_runs(tmp_path):
    got = run(tmp_path, "window", "--dry-run", "--env", "PP_CHALLENGER=gate:research@192.0.2.9", "--env", "WC_TIMEOUT=45m",
              **ROW_OF_CARDS)
    assert got.returncode == 0, got.stderr
    assert [e for e in envs(got) if e.startswith("PP_CHALLENGER=")] == ["PP_CHALLENGER=gate:research@192.0.2.9"]
    assert "WC_TIMEOUT=45m" in envs(got)


def test_a_parameter_a_flag_gives_fills_the_wrap(tmp_path):
    got = run(tmp_path, "window", "--dry-run", "--development", "--env", "CORES=12-23", CARD="6", END="2026-11-17T03:00Z")
    assert got.returncode == 0, got.stderr
    words = shlex.split(got.stdout)
    assert words[words.index("--") + 1:words.index("--") + 4] == ["taskset", "-c", "12-23"]


def test_development_coins_are_chosen_and_the_run_says_so(tmp_path):
    got = run(tmp_path, "window", "--development", "--dry-run", **ROW_OF_CARDS)
    assert got.returncode == 0, got.stderr
    assert {"PP_CHALLENGER=local", "DEVELOPMENT=1"} <= set(envs(got))
    assert not [e for e in envs(got) if e.startswith(("PP_PEER=", "PP_GATE_KEY=", "PP_CHALLENGER_KEY="))]
    assert not [s for s in sends(got) if s.endswith("/main.txt")]


def test_the_compose_run_sends_the_windows_hand_off_to_the_compose_node(tmp_path):
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    (handoff / "compose.tar").write_bytes(b"tar")
    (handoff / "compose.tar.sha256").write_text("sums\n")
    got = run(tmp_path, "compose", "--dry-run", HANDOFF=str(handoff))
    assert got.returncode == 0, got.stderr
    words = shlex.split(got.stdout)
    assert words[words.index("--on") + 1] == "vy-nebius-cpu-14" == PLACEMENT["run"]["window"]["env"]["COMPOSE_NODE"]
    assert sends(got) == [str(handoff / "compose.tar"), str(handoff / "compose.tar.sha256")]
    assert words[words.index("--") + 1:] == ["bash", "adapters/vllm/tools/served_run/compose.sh"] and not envs(got)


def test_the_compose_run_without_its_hand_off_starts_nothing(tmp_path):
    got = run(tmp_path, "compose", "--dry-run")
    assert (got.returncode, got.stdout) == (1, "") and got.stderr.startswith("run.sh: HANDOFF is unset")
    got = run(tmp_path, "compose", "--dry-run", HANDOFF=str(tmp_path / "nothing"))
    assert (got.returncode, got.stdout) == (1, "") and "compose.tar, which is not a file" in got.stderr


def test_a_run_whose_placement_names_no_variable_reads_none(tmp_path):
    got = run(tmp_path, "commit", "--dry-run")
    assert got.returncode == 0 and "--on vy-dc-h200 " in got.stdout, got.stderr
    assert "ROW=" + PLACEMENT["row"] in envs(got) and not [e for e in envs(got) if e.startswith("HIDDEN_SO=")]


def test_development_on_a_run_without_coins_is_refused(tmp_path):
    got = run(tmp_path, "commit", "--development", "--dry-run")
    assert (got.returncode, got.stdout) == (1, "")
    assert got.stderr.strip() == "run.sh: the commit run takes no coins, so --development chooses nothing: drop it"


@pytest.mark.parametrize("script", ["window.sh", "compose.sh"])
def test_the_runs_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(HERE / script)], capture_output=True, check=False).returncode == 0
