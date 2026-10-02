"""`verity-vllm`: one entry point dispatching to every command's ``main(Options)``.

* an unknown command or no command is a usage error (exit 2) and ``--help`` lists every command;
* the options come from flags, then the environment variable an option names, then its default;
* a command run as the process sees ``sys.argv`` as ``python -m <module>`` gave it (its run records store it); a call
  with an explicit argv leaves ``sys.argv`` alone;
* a ``UsageError`` from ``main`` exits 2;
* a command needs no core: ``row stage build`` with ``VERITY_DENSE_THREADS`` set runs where ``verity`` can't be imported, as in
  a Build's top-level process.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from dataclasses import dataclass

import pytest

from verity_vllm.config import UsageError, option
from verity_vllm.pipeline import cli


@dataclass(frozen=True, kw_only=True)
class ProbeOptions:
    """probe command"""

    n: int = option("--n", env="PROBE_N", type=int, default=0, help="a number")


@pytest.fixture
def probe(monkeypatch):
    mod = types.ModuleType("probe_command")
    mod.__file__ = "/tree/probe_command.py"
    mod.Options = ProbeOptions
    seen = []

    def main(o):
        seen.append((o.n, list(sys.argv)))
        if o.n < 0:
            raise UsageError("n must not be negative")
        return 0

    mod.main = main
    monkeypatch.setitem(sys.modules, "probe_command", mod)
    monkeypatch.setitem(cli.COMMANDS, "probe", ("probe_command", "a probe"))
    return seen


def test_no_command_and_an_unknown_command_are_usage_errors(capsys):
    assert cli.main([]) == 2
    assert cli.main(["no-such-command"]) == 2
    assert "unknown command 'no-such-command'" in capsys.readouterr().err


def test_help_lists_every_command(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in cli.COMMANDS)


def test_flags_then_environment_then_default(probe, monkeypatch):
    assert cli.parse("probe", [], {}).n == 0
    assert cli.parse("probe", [], {"PROBE_N": "5"}).n == 5
    assert cli.parse("probe", ["--n", "7"], {"PROBE_N": "5"}).n == 7


def test_a_process_command_sees_its_own_argv(probe, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["/tree/cli.py", "probe", "--n", "3"])
    assert cli.main() == 0
    assert probe[-1] == (3, ["/tree/probe_command.py", "--n", "3"])


def test_an_explicit_argv_leaves_sys_argv_alone(probe, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["/tree/pytest"])
    assert cli.main(["probe", "--n", "4"]) == 0
    assert probe[-1] == (4, ["/tree/pytest"])


def test_a_usage_error_from_main_exits_2(probe):
    with pytest.raises(SystemExit) as e:
        cli.main(["probe", "--n", "-1"])
    assert e.value.code == 2


#: a Build's top-level process: the template puts the integration and the runner on its path, and the row adds the core only for the
#: stage processes it starts
_WITHOUT_CORE = """
import sys

class NoCore:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "verity":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

sys.meta_path.insert(0, NoCore())
from verity_vllm.pipeline import cli, row
row.main = lambda o: print(type(o).__name__)
cli.main(["row", "stage", "build", "cov-gm128-r2", "GEMMA2_9B", "google/gemma-2-9b-it", "main"])
print(sorted(m for m in sys.modules if m.split(".")[0] == "verity"))
"""


def test_a_dense_threads_cap_needs_no_core():
    r = subprocess.run([sys.executable, "-c", _WITHOUT_CORE], env={**os.environ, "VERITY_DENSE_THREADS": "16"},
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\n")[:2] == ["Stage", "[]"]
