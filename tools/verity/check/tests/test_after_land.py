"""`after_land.py`: the full check that trails main starts once per newest tip, and only a landing on a full check gets the
regression's rebuilds."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("after_land_under_test", ROOT / "tools" / "verity" / "check" / "after_land.py")
AL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AL)


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def _land(repo: Path, msg: str) -> str:
    _git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    _git(repo, "push", "-q", "origin", "main")
    return _git(repo, "rev-parse", "HEAD")


def _world(tmp_path: Path) -> tuple[Path, Path]:
    """(a lander's checkout, a second lander's), both of one bare origin."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    a, b = tmp_path / "a", tmp_path / "b"
    _git(tmp_path, "clone", "-q", str(origin), str(a))
    _land(a, "init")
    _git(tmp_path, "clone", "-q", str(origin), str(b))
    return a, b


def _mark(repo: Path) -> str:
    out = _git(repo, "ls-remote", "origin", AL.TRAILING_REF)
    return out.split()[0] if out else ""


def test_one_full_check_starts_per_newest_tip_whichever_lander_gets_there(tmp_path):
    """Two landers' `after_land` on one tip start one check; a landing behind the started one starts none, since that check
    covers it; the next tip starts its own."""
    a, b = _world(tmp_path)
    first = _land(a, "landing 1")
    assert AL.claim(first, cwd=a) is None and _mark(a) == first
    _git(b, "fetch", "-q", "origin")
    assert "started already" in AL.claim(first, cwd=b)
    second = _land(a, "landing 2")
    third = _land(a, "landing 3")
    assert AL.claim(third, cwd=a) is None and _mark(a) == third
    assert "covers " + second[:12] in AL.claim(second, cwd=a)
    assert _mark(a) == third


def test_the_trailing_check_is_the_full_recorded_check_on_node_1_from_a_worktree_of_the_tip(tmp_path):
    a, _b = _world(tmp_path)
    (a / "f.txt").write_text("landed\n")
    _git(a, "add", "f.txt")
    tip = _land(a, "landing")
    seen = []

    def run(cmd, cwd):
        seen.append((cmd, Path(cwd), _git(Path(cwd), "rev-parse", "HEAD"), (Path(cwd) / "f.txt").read_text()))
        return subprocess.CompletedProcess(cmd, 0)

    assert AL.trailing_check(tip, run, cwd=a) == 0
    (cmd, tree, head, text), = seen
    assert cmd == [sys.executable, "tools/verity/check/check.py", "--record", "--on", "vy-nebius-1", "--agreement", "--base", tip]
    assert head == tip and text == "landed\n" and tree != a
    assert not tree.exists() and str(tree) not in _git(a, "worktree", "list")


def test_a_landing_on_a_targeted_check_gets_the_trailing_check_and_no_rebuilds(tmp_path, capsys):
    """`post_train.py` labels the full check a landing's `Check:` trailer names; a `--fast` landing's `Checked:` names none."""
    a, _b = _world(tmp_path)
    ran = []
    run = lambda cmd, cwd: ran.append(cmd) or subprocess.CompletedProcess(cmd, 0)  # noqa: E731
    fast = _land(a, "Merge x (abc)\n\nChecked: r20261007-230000-aaaa\nSkipped-check: the full `check`")
    assert AL.main([fast], run, cwd=a) == 0
    assert [c[1] for c in ran] == ["tools/verity/check/check.py"] and "no regression rebuilds" in capsys.readouterr().out
    ran.clear()
    full = _land(a, "Merge y (def)\n\nCheck: r20261007-230000-bbbb")
    assert AL.main([full], run, cwd=a) == 0
    assert [c[1] for c in ran] == ["tools/verity/check/check.py", str(a / "tools" / "verity" / "check" / "post_train.py")]
    assert ran[1][2] == full
    ran.clear()
    assert AL.main([full], run, cwd=a) == 0                                    # its check was started: only the rebuilds
    assert [c[1] for c in ran] == [str(a / "tools" / "verity" / "check" / "post_train.py")]
    assert "started already" in capsys.readouterr().out
