"""`research run` refuses a workload whose shell script pipes without `pipefail`: such a script exits with the pipe's last
command's status (`tee`'s), so a run whose suite failed ended `done`, rc 0. Refused on the launching side before anything is
allocated; `--allow-pipe` passes it, and the machine side (`--claimed`) never judges it."""
from __future__ import annotations

import pytest

from research import cli, pipefail


@pytest.fixture
def launched(monkeypatch):
    """Each launch `research run` went on to (a placed run, or `--on auto`'s placement), in place of making it."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_run_placed", lambda ns, harness, command, clones: seen.append(command) or 0)
    monkeypatch.setattr(cli, "_auto_placement", lambda ns, harness: seen.append(["auto"]) or "placed")
    return seen


def run(tmp_path, *argv: str) -> int:
    return cli.main(["run", "--runs-dir", str(tmp_path), "--on", "vy-nowhere", "--project", "p", *argv])


@pytest.mark.parametrize("command", [
    ["bash", "-lc", 'uv run tools/verity/check/suites.py --quick 2>&1 | tee "$OUT/suite.log"'],
    ["nice", "-n", "19", "bash", "-lc", "a | tee f"],
    ["env", "K=V", "/bin/sh", "-c", "a | tee f"],
    ["timeout", "60", "uv", "run", "zsh", "-ec", "a |& tee f"],
    ["bash", "-c", "set -e; x=1\na 2>&1 | tail -25"],
    ["bash", "-c", "a | b; set -o pipefail"],
], ids=["tee", "nice", "env", "pipe-and", "tail", "set-after"])
def test_a_pipe_without_pipefail_is_refused_before_anything_is_launched(tmp_path, capsys, launched, command):
    assert run(tmp_path, "--", *command) == 2
    err = capsys.readouterr().err
    assert "research run: refused: the workload's" in err and "pipes without `pipefail`" in err
    assert "`set -o pipefail;`" in err and "| tee FILE" in err and "--allow-pipe" in err, "the message names each cure"
    assert launched == [] and list(tmp_path.iterdir()) == [], "nothing launched, no run dir taken"
    assert cli.main(["run", "--runs-dir", str(tmp_path), "--on", "auto", "--", *command]) == 2 and launched == []


@pytest.mark.parametrize("command", [
    ["bash", "-lc", "a || b"],
    ["bash", "-lc", "echo 'a|b'; tr '|' , < f"],
    ["bash", "-lc", "grep a\\|b f"],
    ["bash", "-lc", "set -o pipefail; a | b"],
    ["bash", "-lc", "set -euo pipefail; a | b"],
    ["bash", "-c", "# the run's log\nset -eo pipefail\na | b"],
    ["bash", "-o", "pipefail", "-c", "a | b"],
    ["bash", "-c", 'case "$1" in a|b) x;; esac'],
    ["bash", "-c", "python3 - <<'EOF'\nprint(1 | 2)\nEOF"],
    ["bash", "x.sh", "a | b"],
    ["python3", "x.py", "|", "tee", "f"],
], ids=["or", "quoted", "escaped", "set-o", "set-euo", "comment-then-set", "shell-o", "case", "heredoc", "file", "no-shell"])
def test_a_command_that_pipes_with_pipefail_or_not_at_all_is_launched(tmp_path, launched, command):
    assert pipefail.refusal(command) is None
    assert run(tmp_path, "--", *command) == 0 and launched == [command]


def test_allow_pipe_launches_a_pipe_whose_own_exit_code_is_meant(tmp_path, launched):
    command = ["bash", "-lc", "a 2>&1 | tee f"]
    assert pipefail.refusal(command) is not None
    assert run(tmp_path, "--allow-pipe", "--", *command) == 0 and launched == [command]


def test_run_help_names_the_flag_without_a_command(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--help"])
    assert e.value.code == 0 and "--allow-pipe" in capsys.readouterr().out


def test_the_machine_side_does_not_judge_again(tmp_path, launched):
    """The launcher judged the command; on the machine it may come wrapped (slot.py, gpu-lease) and is run as sent."""
    command = ["bash", "-lc", "a | tee f"]
    assert cli.main(["run", "--runs-dir", str(tmp_path), "--id", "r20261114-000000-abcd", "--claimed", "--", *command]) == 0
    assert launched == [command]


def test_a_script_shlex_cannot_parse_is_refused_only_for_a_bare_pipe():
    assert pipefail.unsafe_pipe("echo 'a | b")
    assert not pipefail.unsafe_pipe("echo 'a || b")
    assert not pipefail.unsafe_pipe('set -uo pipefail\necho "$([ "$c" = 0 ] && echo "the record\'s ok")" | cut -c1-9')
