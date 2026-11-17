"""`verity`'s dispatch: an unknown command is named on its own line, then the usage, and every script a command starts is
in the tree."""
from __future__ import annotations

import sys
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


def test_verity_calibrate_runs_a_calibrations_runner_and_lists_them_given_none(tmp_path, monkeypatch):
    monkeypatch.setattr(verity_cli, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as e:
        verity_cli.main(["calibrate"])
    assert str(e.value.code).endswith("NAME is one of none")
    (tmp_path / "calibration" / "tc_probe").mkdir(parents=True)
    (tmp_path / "calibration" / "tc_probe" / "run.py").write_text("")
    started: list[list[str]] = []
    monkeypatch.setattr(verity_cli, "_run", lambda argv: started.append(argv) or 0)
    assert verity_cli.main(["calibrate", "tc-probe", "--card", "0"]) == 0
    assert started == [[sys.executable, str(tmp_path / "calibration" / "tc_probe" / "run.py"), "--card", "0"]]


def test_verity_demo_is_no_command():
    with pytest.raises(SystemExit) as e:
        verity_cli.main(["demo", "remote"])
    assert str(e.value.code).split("\n", 1)[0] == "verity: no command 'demo'"
