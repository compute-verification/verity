"""`research merge REF --fast [RUN...]`: a landing on targeted checks of the exact commit (or, for docs and removals, none), with
every refusal of the full gate but the full check's, and the push past the guard that takes only a full check."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from test_merge_gate import _RESULT, CHECKED, DECL, _declared_by_the_throwaway_repo, _git, _record, _repos

import research.merge as M
from research.cli import main as research_main
from research.merge import main as merge_main

#: a targeted check of the throwaway target: its check.py with a part, which the full gate refuses
PART = (*CHECKED, "tests")


def _run(tmp_path: Path, checkout: Path, monkeypatch, *command: str, status: str = "passed", tool: str = "mytools.decl_check:T"
         ) -> tuple[int, str]:
    """(rc, run id) of ``command`` recorded on ``checkout``; the target's check.py writes a result whose step is ``status``."""
    runs = tmp_path / "runs"
    before = set(runs.iterdir()) if runs.exists() else set()
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("FAKE_CHECK", _RESULT % status)
    rc = research_main(["run", "--runs-dir", str(runs), "--project", "t", "--no-sampler", "--tool", tool, "--cwd", "source", "--",
                        *command])
    new, = set(runs.iterdir()) - before
    return rc, new.name


def _commit(repo: Path, files: dict[str, str | None], msg: str) -> str:
    for rel, text in files.items():
        p = repo / rel
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def test_targeted_names_the_check_parts_an_argv_ran_behind_a_launcher_or_a_slot_and_nothing_else():
    py = sys.executable
    assert M.targeted([py, "tools/verity/check/check.py", "tests"]) == "tools/verity/check/check.py"
    assert M.targeted(["uv", "run", "--locked", "--extra", "torch-cpu", "python", "tools/verity/check/suites.py", "--quick"]) == (
        "tools/verity/check/suites.py")
    assert M.targeted(["uv", "run", "circuit-check", "--all"]) == "circuit-check"
    assert M.targeted([*M.SLOT_WRAP, "--by", "ci", "--", "python3", "tools/verity/lean/check.py"]) == "tools/verity/lean/check.py"
    for argv in ([py, "evil.py", "tools/verity/check/check.py"],              # another script first
                 ["bash", "-c", "true; python tools/verity/check/check.py"],
                 ["env", "X=1", "python", "tools/verity/check/check.py"],
                 [py, "tools/verity/check/check.py", "--help"],
                 [py, "tools/verity/check/post_train.py"], [], None, ["uv", "--extra"]):
        assert M.targeted(argv) is None, argv


def test_fast_lands_on_a_passing_targeted_run_of_the_exact_commit_with_its_trailers(tmp_path, monkeypatch, capsys):
    main, feat, tip = _repos(tmp_path)
    rc, rid = _run(tmp_path, feat, monkeypatch, *PART)
    assert rc == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 3                                  # a part is no full check
    assert merge_main(["feat", "--fast", rid, "--dry-run"]) == 0
    assert f"({tip[:12]}) may be merged into main on a targeted check: {rid}" in capsys.readouterr().out
    assert merge_main(["feat", "--fast", rid]) == 0
    msg = _git(main, "log", "-1", "--format=%B")
    assert f"\nChecked: {rid}\n" in msg and "\nSkipped-check: the full `check`, which trails main" in msg and "\nCheck: " not in msg
    assert _git(main, "rev-parse", "HEAD^{tree}") == _git(main, "rev-parse", f"{tip}^{{tree}}")


def test_fast_refuses_what_isnt_a_passing_targeted_run_of_the_exact_commit(tmp_path, monkeypatch, capsys):
    main, feat, _tip = _repos(tmp_path)
    _, other = _run(tmp_path, feat, monkeypatch, sys.executable, "-c", "pass")
    _, failed = _run(tmp_path, feat, monkeypatch, *PART, status="failed")
    assert _run(tmp_path, feat, monkeypatch, "false")[0] == 1
    _, ok = _run(tmp_path, feat, monkeypatch, *PART)
    monkeypatch.chdir(main)
    for rid, why in ((other, "none of a targeted check's"), (failed, "agree didn't pass")):
        assert merge_main(["feat", "--fast", rid, "--dry-run"]) == 3
        assert why in capsys.readouterr().err, rid
    _commit(feat, {"b.txt": "b\n"}, "feat moves on")
    assert merge_main(["feat", "--fast", ok, "--dry-run"]) == 3                   # feat's tip is no longer the one it ran on
    assert "its record says commit" in capsys.readouterr().err
    assert merge_main(["feat", "--fast", "r20261007-000000-dead", "--dry-run"]) == 3
    assert "can't be read" in capsys.readouterr().err


def test_fast_keeps_the_full_gates_refusals_and_full_only_paths(tmp_path, monkeypatch, capsys):
    main, feat, _tip = _repos(tmp_path)
    _commit(main, {"c.txt": "c\n"}, "main moves")
    _, rid = _run(tmp_path, feat, monkeypatch, *PART)
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--fast", rid, "--dry-run"]) == 3 and "main has moved past" in capsys.readouterr().err
    _git(feat, "merge", "-q", "main")
    for path, rule in (("tools/move/layout_map.toml", "tools/move/"), ("tools/verity/check/x.py", "tools/verity/check/"),
                       ("tools/research/src/research/merge.py", "tools/research/src/research/merge.py")):
        _commit(feat, {path: "x\n"}, f"touch {path}")
        _, rid = _run(tmp_path, feat, monkeypatch, *PART)
        monkeypatch.chdir(main)
        assert merge_main(["feat", "--fast", rid, "--dry-run"]) == 3
        err = capsys.readouterr().err
        assert f"changes {rule}:" in err and "lands on a full `check`" in err
        _commit(feat, {path: None}, f"untouch {path}")


def test_fast_with_no_run_lands_docs_and_removals_with_a_reason_and_nothing_else(tmp_path, monkeypatch, capsys):
    main, _feat, _tip = _repos(tmp_path)
    _commit(main, {"t.py": "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n", "gone.py": "x = 1\n"}, "tests")
    _git(main, "checkout", "-q", "-b", "rm")
    _commit(main, {"t.py": "def test_a():\n    pass\n", "gone.py": None, "README.md": "new words\n"}, "delete a flaky test")
    _git(main, "checkout", "-q", "-b", "adds")
    _commit(main, {"t.py": "def test_a():\n    assert 1\n"}, "an edit")
    _git(main, "checkout", "-q", "main")
    monkeypatch.chdir(main)
    with pytest.raises(SystemExit):
        merge_main(["rm", "--why", "a test deletion"])                             # --why goes with --fast
    assert merge_main(["rm", "--fast", "--dry-run"]) == 3 and "needs --why" in capsys.readouterr().err
    assert merge_main(["adds", "--fast", "--why", "trust me", "--dry-run"]) == 3
    assert "adds lines to t.py" in capsys.readouterr().err
    assert merge_main(["rm", "--fast", "--why", "deletes test_b, which flakes"]) == 0
    msg = _git(main, "log", "-1", "--format=%B")
    assert "\nSkipped-check: " in msg and "\nWhy: deletes test_b, which flakes" in msg and "Checked:" not in msg


def test_fast_needs_the_step_the_declaration_requires_of_the_change_passed_in_a_cited_run(tmp_path, monkeypatch, capsys):
    """C-Flock's case: `lean-agreement` must have passed for a change under its paths, as at the full gate."""
    _declared_by_the_throwaway_repo(monkeypatch)
    main, feat, _tip = _repos(tmp_path)
    _commit(main, {"tools/verity/check/tool.py": DECL.format(", merge_requires={'agree': ['flock/']}")}, "rule")
    _git(feat, "merge", "-q", "main")
    _commit(feat, {"flock/x.txt": "x\n"}, "flock change")
    _, skipped = _run(tmp_path, feat, monkeypatch, *PART, status="skipped", tool="check")
    _, passed = _run(tmp_path, feat, monkeypatch, *PART, tool="check")
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--fast", skipped, "--dry-run"]) == 3
    assert "changes flock/, where agree must have passed in a cited run" in capsys.readouterr().err
    assert merge_main(["feat", "--fast", skipped, passed, "--dry-run"]) == 0


def test_fast_pushes_past_the_guard_that_takes_only_a_full_check(tmp_path, monkeypatch, capsys):
    from research.store.local import LocalStore

    main, feat, _tip = _repos(tmp_path)
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    _git(main, "remote", "add", "origin", str(origin))
    _git(main, "push", "-q", "origin", "main")
    base = _git(main, "rev-parse", "HEAD")
    hook = M.install_push_guard(main)
    _, rid = _run(tmp_path, feat, monkeypatch, *PART)
    monkeypatch.setattr(M, "caught_up", lambda *_a, **_k: "tip")
    monkeypatch.setattr(M, "_sweep_after", lambda *_a, **_k: None)
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--fast", rid, "--push"]) == 0, capsys.readouterr().err
    merged = _git(main, "rev-parse", "HEAD")
    assert _git(main, "ls-remote", "origin", "refs/heads/main").split()[0] == merged
    assert M.unchecked_pushes(main, [f"refs/heads/main {merged} refs/heads/main {base}"], LocalStore()) == [merged]
    assert hook is not None and subprocess.run([str(hook)], input=f"refs/heads/main {merged} refs/heads/main {base}\n", text=True,
                                               cwd=main, capture_output=True).returncode == 3
