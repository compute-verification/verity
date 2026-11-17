"""`verity`'s dispatch: an unknown command is named on its own line, then the usage, and every script a command starts is
in the tree."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import verity_cli

#: each command that starts a script of the check, and the script
STARTS = {("check",): "check.py", ("check", "move", "BASE", "HEAD"): "move_check.py", ("ci", "ls"): "live.py",
          ("tasks", "list", "--json"): "tasks.py"}


def test_every_script_a_command_starts_is_in_the_tree(monkeypatch):
    """The check's scripts are started by path, so one that moves leaves its command failing with python's "can't open
    file" and nothing else red: the path each command hands `_run` is a file of the tree."""
    started: list[list[str]] = []
    monkeypatch.setattr(verity_cli, "_run", lambda argv: started.append(argv) or 0)
    for command in STARTS:
        assert verity_cli.main(list(command)) == 0
    scripts = [Path(argv[1]) for argv in started]
    assert [s.name for s in scripts] == list(STARTS.values())
    gone = [f"verity {' '.join(c)}: {s.relative_to(verity_cli.ROOT)}" for c, s in zip(STARTS, scripts) if not s.is_file()]
    assert not gone, f"point `verity_cli.CHECK` (or the command's line in `main`) at where the script is now: {gone}"


def test_an_unknown_command_is_named_then_the_usage_follows():
    with pytest.raises(SystemExit) as e:
        verity_cli.main(["nosuch"])
    first, rest = str(e.value.code).split("\n", 1)
    assert first == "verity: no command 'nosuch'"
    assert rest == verity_cli.__doc__


@pytest.mark.parametrize("name", ["remote", "onsite_full", "onsite_network"])
def test_a_demo_runs_its_file_with_an_on_for_each_of_its_machines(monkeypatch, name):
    from verity_cli import run

    started: list[list[str]] = []
    monkeypatch.setattr(run, "run_main", lambda argv, demo: started.append((argv, demo)) or 0)
    assert verity_cli.main(["demo", name, "--periods", "2"]) == 0
    config = verity_cli.ROOT / "demos" / f"{name}.toml"
    machines = dict.fromkeys(p["machine"] for p in tomllib.loads(config.read_text())["party"])
    assert len(machines) > 1
    assert started == [([str(config), *(x for m in machines for x in ("--on", m)), "--periods", "2"], True)]


def test_a_demo_whose_args_name_a_machine_runs_only_that_one(monkeypatch):
    from verity_cli import run

    started: list[list[str]] = []
    monkeypatch.setattr(run, "run_main", lambda argv, demo: started.append((argv, demo)) or 0)
    assert verity_cli.main(["demo", "remote", "--on", "prover-node"]) == 0
    assert started == [([str(verity_cli.ROOT / "demos" / "remote.toml"), "--on", "prover-node"], True)]
