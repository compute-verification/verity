import subprocess
import sys
from pathlib import Path

import pytest

from verity_bench import jobs
from verity_bench.cli import main
from verity_bench.vllm import result as result_format

ROOT = Path(__file__).resolve().parents[3]
NAME = "llama32-1b-1gpu-pouw-protocol-served-h100"
SHIPPED = "c" * 40


def _no_git(monkeypatch):
    """A shipped tree, as a Slurm job of VCF's cluster runs it: git can name no commit, and research run's --tree-sha
    exported the shipped one."""

    for name in ("run", "check_output"):
        real = getattr(subprocess, name)

        def refuse(argv, *args, _real=real, _name=name, **kwargs):
            if list(argv)[:1] != ["git"]:
                return _real(argv, *args, **kwargs)
            if _name == "check_output":
                raise subprocess.CalledProcessError(128, argv)
            return subprocess.CompletedProcess(argv, 128, "", "fatal: not a git repository")

        monkeypatch.setattr(subprocess, name, refuse)
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", SHIPPED)
    monkeypatch.setattr(jobs, "lease_host", lambda: False)


@pytest.mark.parametrize("card", ["NVIDIA H100 80GB HBM3", "NVIDIA H200"])
def test_run_runs_the_one_point_in_this_run_on_the_card_it_was_given_with_no_git_lease_or_slot(
        tmp_path, monkeypatch, capsys, card):
    """A machine not ours (a Slurm job of VCF's cluster): no git that knows the commit, so no catalog lookup, which `run`
    says by name, and the point runs on the cards the job was given."""

    _no_git(monkeypatch)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("VERITY_BENCH_VENV", raising=False)
    monkeypatch.delenv("VERITY_BENCH_SETUP", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.chdir(ROOT)
    commands = []
    dispatch = jobs.dispatch
    monkeypatch.setattr(jobs, "dispatch", lambda *a, **k: dispatch(
        *a, **k, execute=lambda command, cwd: commands.append((list(command), cwd)) or 0))
    assert main(["run", NAME]) == 0
    [(command, cwd)] = commands
    assert "no catalog lookup here, so no passing run is reused: git doesn't know commit" in capsys.readouterr().err
    item = jobs.make_plan(NAME, root=ROOT, lookup=lambda identity, commit: None).points[0]
    assert cwd == ROOT and item.point.id == "b32-p512-mixed"
    assert command == ["env", "VERITY_BENCH_SETUP=benchmarks/harness/setup/pearl_c_sm90.sh",
                       "bash", "benchmarks/harness/run_point.sh", "--spec", f"benchmarks/harness/specs/{NAME}.toml",
                       "--point", item.point.id, "--identity", item.identity]
    assert not {"research", "gpu-lease", "taskset", "python3"} & set(command)

    monkeypatch.setenv("VERITY_BENCH_SETUP", "benchmarks/harness/setup/pearl_c_sm90.sh")
    monkeypatch.setattr(result_format.kit, "host_stamp", lambda: {"card": card})
    monkeypatch.setattr("verity_bench.cli.run_benchmark", lambda *args, **kwargs: sys.exit(f"timed at {kwargs['git_commit']}"))
    with pytest.raises(SystemExit) as stop:
        main(["_vllm-point", *command[command.index("--spec"):]])
    if card == "NVIDIA H100 80GB HBM3":
        assert stop.value.code == f"timed at {SHIPPED}"
    else:
        assert stop.value.code == 2 and f"the card is a {card}, not the h100 of spec {NAME!r}" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--here"], ["--direct", "vy-h100-1"], ["--card", "0"], ["--cores", "0-15"],
                                  ["--end", "2026-11-18T07:13Z"]])
def test_run_takes_no_flag_of_the_launches_it_replaced(monkeypatch, capsys, argv):
    _no_git(monkeypatch)
    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit):
        main(["run", NAME, *argv])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_run_refuses_a_spec_with_more_than_one_point_unless_one_is_named(tmp_path, monkeypatch, capsys):
    _no_git(monkeypatch)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.chdir(ROOT)
    commands = []
    dispatch = jobs.dispatch
    monkeypatch.setattr(jobs, "dispatch", lambda *a, **k: dispatch(
        *a, **k, execute=lambda command, cwd: commands.append(command) or 0))
    name = "llama31-8b-fp8-1gpu-commit-served-width-h200"
    with pytest.raises(SystemExit):
        main(["run", name])
    assert ("has 3 points (b1-p512-mixed, b8-p512-mixed, b32-p512-mixed) and a run runs one: name it with --point"
            in capsys.readouterr().err) and commands == []
    assert main(["run", name, "--point", "b8-p512-mixed"]) == 0
    [command] = commands
    assert command[command.index("--point") + 1] == "b8-p512-mixed"


def test_a_tree_git_cant_name_refuses_naming_the_variable_unless_research_run_set_it(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCH_SOURCE_SHA", raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    with pytest.raises(ValueError, match="RESEARCH_SOURCE_SHA is unset, so nothing says which commit this tree is"):
        jobs._commit(tmp_path)
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", SHIPPED)
    assert jobs._commit(tmp_path) == SHIPPED
