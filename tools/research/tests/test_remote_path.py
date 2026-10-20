"""The PATH a machine's runner, and so its workload, gets: the run's own shipped `research` first (INF-102), and the research user's
`~/.elan/bin` last, where vy-nebius-2 keeps `lake` (proofs' Lean job exited 127, `lake: command not found`, r20261019-072345-9757)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from research import remote

TOOL = str(Path(remote.__file__).resolve().parents[1])


def test_the_runners_path_has_the_runs_research_first_and_elan_last(tmp_path, monkeypatch):
    home, runs = tmp_path / "home", tmp_path / "root" / "runs"
    (home / ".elan" / "bin").mkdir(parents=True)
    runs.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = remote._runner_env({"tool": TOOL, "env": {}}, runs, None, {})
    path = env["PATH"].split(os.pathsep)
    bin_ = runs.parent / "cache" / "bin" / Path(TOOL).name
    assert path == [str(bin_), "/usr/bin", "/bin", str(home / ".elan" / "bin")]
    r = subprocess.run(["research", "cancel", "--help"], env={"PATH": env["PATH"]}, capture_output=True, text=True)
    assert r.returncode == 0 and "usage: research cancel" in r.stdout, r.stderr
    assert remote._runner_env({"tool": TOOL, "env": {}}, runs, None, {})["PATH"] == env["PATH"], "the shim is written once, then reused"

    monkeypatch.setenv("PATH", f"/usr/bin:{home / '.elan' / 'bin'}")
    (home / ".elan" / "bin").rmdir()
    assert remote._runner_env({"env": {}}, runs, None, {})["PATH"] == f"/usr/bin:{home / '.elan' / 'bin'}", "no tool, no new entry"
