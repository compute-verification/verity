"""What a shipped research package must do before a machine runs it (pods/nebius/tool_rules.py): a package whose stall dumps
print command lines (before #7656) is refused by the launcher before anything ships.  The machine is a directory and ssh is
`sh -c`, as in test_run_cwd_clone.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from research import remote
from research.cli import main
from research.pods.nebius import tool_rules

PY = sys.executable
OLD_PYSPY = 'def dump(procs, *, run):\n    return run(["py-spy", "dump"]).stdout.decode()\n'


def _package(root: Path, pyspy: str | None) -> Path:
    pkg = root / "tools" / "research" / "src" / "research"
    (pkg / "telemetry").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    if pyspy is not None:
        (pkg / "telemetry" / "pyspy.py").write_text(pyspy)
    return pkg


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def test_a_package_whose_stall_dumps_print_command_lines_is_refused_and_this_one_is_not(tmp_path):
    why = tool_rules.refusal(_package(tmp_path / "old", OLD_PYSPY))
    assert why is not None and "predates #7656 (telemetry/pyspy.py has no ARGV_LINE)" in why and "command line" in why
    assert "Merge origin/main into the branch" in why
    assert tool_rules.refusal(_package(tmp_path / "older", None)) is None, "a package without stall dumps prints no command line"
    assert tool_rules.refusal(Path(remote.__file__).parent) is None


def test_the_launcher_refuses_a_source_tree_whose_recorder_predates_the_fix_before_shipping(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    _package(repo, OLD_PYSPY)
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "a branch behind #7656")
    root = tmp_path / "machine"
    (tmp_path / "machines.toml").write_text(f'[machines.m]\nprovider = "runpod"\npod_id = "p1"\nproject = "p"\nroot = "{root}"\npython = "{PY}"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "h", "port": 22, "pod_name": "m", "cost_per_hr": 0.1,
                                                         "rate_observed_utc": "t0", "gpu": None})
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **k: ["sh", "-c"])
    rc = main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p", "--source", str(repo), "--no-custody-r2",
               "--no-sampler", "--", PY, "-c", "print('no')"])
    err = capsys.readouterr().err
    assert rc == 3 and "predates #7656" in err and "Merge origin/main into the branch" in err
    assert not (root / "tool").exists() and not (root / "src").exists(), "refused before anything shipped"
