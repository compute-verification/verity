"""`research merge`: a commit merges only with a passing `check` attempt of that exact commit in the store, recorded by a real
`research run --tool` in a throwaway repository."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from research.cli import main as research_main
from research.merge import main as merge_main


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def _repos(tmp_path: Path) -> tuple[Path, Path, str]:
    """(the main checkout, a worktree of branch `feat`, feat's tip)."""
    main = tmp_path / "main"
    (main / "mytools").mkdir(parents=True)
    (main / "mytools" / "decl_check.py").write_text("from research.store.tool import Tool\nT = Tool(name='check', version='1')\n")
    (main / "a.txt").write_text("a\n")
    (main / ".gitignore").write_text("__pycache__/\n")
    _git(main, "init", "-q", "-b", "main")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "init")
    feat = tmp_path / "feat"
    _git(main, "worktree", "add", "-q", "-b", "feat", str(feat))
    (feat / "a.txt").write_text("a\nb\n")
    _git(feat, "commit", "-q", "-am", "feat")
    return main, feat, _git(feat, "rev-parse", "HEAD")


def _record(tmp_path: Path, checkout: Path, monkeypatch, command: str) -> int:
    monkeypatch.chdir(checkout)
    return research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool", "mytools.decl_check:T",
                          "--cwd", "source", "--", command])


def test_merge_needs_a_passing_check_of_the_exact_commit(tmp_path, monkeypatch, capsys):
    main, feat, tip = _repos(tmp_path)
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 1 and "no passing `check` attempt" in capsys.readouterr().err
    assert _record(tmp_path, feat, monkeypatch, "false") == 1                     # a failed check is an attempt, never a pass
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 1 and "failed rc=1" in capsys.readouterr().err
    (feat / "a.txt").write_text("dirty\n")
    assert _record(tmp_path, feat, monkeypatch, "true") == 0                      # passed, but on an uncommitted tree
    _git(feat, "checkout", "--", "a.txt")
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 1 and "dirty" in capsys.readouterr().err
    assert _record(tmp_path, feat, monkeypatch, "true") == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 0
    msg = _git(main, "log", "-1", "--format=%B")
    assert msg.startswith(f"Merge feat ({tip[:8]})") and "\nCheck: r" in msg
    assert _git(main, "rev-parse", "HEAD^{tree}") == _git(main, "rev-parse", f"{tip}^{{tree}}")
    assert _git(main, "rev-parse", "HEAD^2") == tip


def test_an_attempt_recorded_on_another_machine_counts(tmp_path, monkeypatch, capsys):
    """The run publishes to its store and the shared remote; the merger's store is empty until the gate refreshes its attempts."""
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    main, feat, tip = _repos(tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "pod-store"))
    assert _record(tmp_path, feat, monkeypatch, "true") == 0
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "merger-store"))
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 0 and f"({tip[:12]}) may be merged into main" in capsys.readouterr().out


def test_merge_refuses_a_commit_the_target_has_moved_past(tmp_path, monkeypatch, capsys):
    main, feat, tip = _repos(tmp_path)
    assert _record(tmp_path, feat, monkeypatch, "true") == 0
    (main / "c.txt").write_text("c\n")
    _git(main, "add", "c.txt")
    _git(main, "commit", "-q", "-m", "main moves")
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 1 and "main has moved past" in capsys.readouterr().err
    _git(main, "checkout", "-q", "-b", "other")
    assert merge_main(["feat", "--dry-run"]) == 1 and "not main" in capsys.readouterr().err


def _branch(main: Path, name: str, path: str) -> None:
    _git(main, "checkout", "-q", "-b", name, "main")
    (main / path).write_text(name + "\n")
    _git(main, "add", path)
    _git(main, "commit", "-q", "-m", name)
    _git(main, "checkout", "-q", "main")


def test_a_train_lands_what_it_checked_and_bisects_to_the_breaking_ref(tmp_path, monkeypatch, capsys):
    """A B C, where B adds the file the check refuses: the candidate fails, the prefixes are checked (A passes, A+B fails), B is
    named and main fast-forwards to the checked A prefix; then A C (without B) lands whole on one check."""
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))
    main, _feat, _tip = _repos(tmp_path)
    for name, path in (("A", "a.txt"), ("B", "BAD"), ("C", "c.txt")):
        _branch(main, name, path)
    monkeypatch.chdir(main)
    before = _git(main, "rev-parse", "HEAD")
    cmd = ["--tool", "mytools.decl_check:T", "--command", "sh -c 'test ! -e BAD'"]
    assert merge_main(["--train", "A", "B", "C", *cmd]) == 1
    out = capsys.readouterr().out
    assert "train A B C: B breaks it; main fast-forwarded to" in out and out.count("recorded as") == 3
    assert _git(main, "rev-parse", "HEAD^1") == before and (main / "a.txt").exists() and not (main / "BAD").exists()
    assert "Train:" in _git(main, "log", "-1", "--format=%B")
    assert merge_main(["--train", "C", *cmd]) == 0
    out = capsys.readouterr().out
    assert out.count("recorded as") == 1 and "`check` passed; main fast-forwarded" in out and (main / "c.txt").exists()
    assert not _git(main, "worktree", "list").count("research-train")
