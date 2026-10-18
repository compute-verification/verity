"""`research agents quiet` (INF-55 h): a worker whose worktree has had no commit, no change and no live process for hours is QUIET."""
import os
import subprocess
import sys
import time

from research import agents

GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(cwd, *a, when=None):
    env = {**os.environ, **GIT_ENV, **({"GIT_AUTHOR_DATE": f"@{when}", "GIT_COMMITTER_DATE": f"@{when}"} if when else {})}
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, env=env)


def test_a_worktree_without_commit_change_or_process_for_hours_is_quiet(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agents, "git", lambda repo, *a, timeout=60: subprocess.run(
        ["git", "-C", str(repo), *a], capture_output=True, text=True, env={**os.environ, **GIT_ENV}).stdout)
    now = time.time()
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "a").write_text("a\n")
    git(repo, "add", "a")
    git(repo, "commit", "-q", "-m", "a", when=int(now - 5 * 3600))
    for name in ("stalled", "editing", "running"):
        git(repo, "worktree", "add", "-q", "-b", name, str(tmp_path / name))
    os.utime(repo / "a", (now - 5 * 3600, now - 5 * 3600))
    for name in ("stalled", "running"):
        os.utime(tmp_path / name / "a", (now - 5 * 3600, now - 5 * 3600))
    (tmp_path / "editing" / "a").write_text("edited\n")                        # a change minutes old
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], cwd=tmp_path / "running", stdin=subprocess.PIPE)
    try:
        rc = agents.main(["quiet", "--repo", str(repo), "--hours", "2"])
    finally:
        proc.kill()
        proc.wait()
    out = capsys.readouterr().out.splitlines()
    assert rc == 1
    verdict = {line.split()[1]: line.split()[0] for line in out}
    assert verdict == {str(repo): "QUIET", str(tmp_path / "stalled"): "QUIET", str(tmp_path / "editing"): "active",
                       str(tmp_path / "running"): "active"}
    assert any("stalled (stalled): commit 5.0 h ago; no live process" in line for line in out)
    assert any(str(tmp_path / "running") in line and "live: python" in line for line in out)
