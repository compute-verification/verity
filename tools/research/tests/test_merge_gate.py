"""`research merge`: a commit merges only with a passing `check` attempt of that exact commit in the store, recorded by a real
`research run --tool` in a throwaway repository."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from research.cli import main as research_main
from research.merge import main as merge_main


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


_RESULT = ("import json, os; d = os.environ['RESEARCH_RUN_DIR']; json.dump({'schema': 'research/result/v0.1', 'kind': 'check/v1', "
           "'run_id': os.environ['RESEARCH_RUN_ID'], 'workload_fingerprint': {}, 'validation': {'status': 'passed', 'checker': 't'}, "
           "'measurements': [], 'steps': [{'step': 'agree', 'status': '%s'}]}, open(d + '/result.json', 'w'))")
#: the throwaway target's tools/check/check.py: it passes as check does, with a passing `check/v1` result, or runs the Python in
#: $FAKE_CHECK instead (`_checked`)
CHECK_PY = f"import os\nexec(os.environ.get('FAKE_CHECK') or {_RESULT % 'passed'!r})\n"
#: the command the throwaway target declares for `check`: the gate admits only a run of it, with a passing result of a kind the
#: declaration produces
CHECKED = (sys.executable, "tools/check/check.py")
#: the throwaway target's declaration of `check`, which the gate reads `command`, `produces` and `merge_requires` from
DECL = ("from research.store.tool import Tool\nCHECK = Tool(name='check', version='1', "
        f"command={list(CHECKED)!r}, produces=['check/v1']{{}})\n")
#: what a throwaway target holds of its check: the declaration and the script it declares
CHECK_FILES = {"tools/check/tool.py": DECL.format(""), "tools/check/check.py": CHECK_PY}


def _repos(tmp_path: Path) -> tuple[Path, Path, str]:
    """(the main checkout, a worktree of branch `feat`, feat's tip)."""
    main = tmp_path / "main"
    (main / "mytools").mkdir(parents=True)
    (main / "mytools" / "decl_check.py").write_text("from research.store.tool import Tool\nT = Tool(name='check', version='1')\n")
    (main / "tools" / "check").mkdir(parents=True)
    for rel, text in CHECK_FILES.items():
        (main / rel).write_text(text)
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


def _declared_by_the_throwaway_repo(monkeypatch) -> None:
    """`--tool check` imports `tools.check.tool` whenever `tools` is importable in this process (the suite's own checkout, from
    sys.path or an earlier test's import) and reads the throwaway repo's file only when that fails: make it fail."""
    for name in ("tools", "tools.check", "tools.check.tool"):
        monkeypatch.setitem(sys.modules, name, None)


def _record(tmp_path: Path, checkout: Path, monkeypatch, *command: str) -> int:
    monkeypatch.chdir(checkout)
    return research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool", "mytools.decl_check:T",
                          "--cwd", "source", "--", *command])


def _checked(tmp_path: Path, checkout: Path, monkeypatch, program: str | None = None) -> int:
    """Record the target's declared check (`CHECKED`) on ``checkout``: a passing `check/v1` result, or, given ``program``, the
    Python it runs instead (`CHECK_PY`)."""
    if program is None:
        monkeypatch.delenv("FAKE_CHECK", raising=False)
    else:
        monkeypatch.setenv("FAKE_CHECK", program)
    return _record(tmp_path, checkout, monkeypatch, *CHECKED)


def test_merge_needs_a_passing_check_of_the_exact_commit(tmp_path, monkeypatch, capsys):
    main, feat, tip = _repos(tmp_path)
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 3 and "no passing `check` attempt" in capsys.readouterr().err
    assert _record(tmp_path, feat, monkeypatch, "false") == 1                     # a failed check is an attempt, never a pass
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 3 and "failed rc=1" in capsys.readouterr().err
    (feat / "a.txt").write_text("dirty\n")
    assert _checked(tmp_path, feat, monkeypatch) == 0                              # passed, but on an uncommitted tree
    _git(feat, "checkout", "--", "a.txt")
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 3 and "dirty" in capsys.readouterr().err
    assert _checked(tmp_path, feat, monkeypatch, "pass") == 0                      # passed, with no result saying what ran
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 3 and "refused: it published no result" in capsys.readouterr().err
    assert _checked(tmp_path, feat, monkeypatch) == 0
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
    assert _record(tmp_path, feat, monkeypatch, *CHECKED) == 0
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "merger-store"))
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 0 and f"({tip[:12]}) may be merged into main" in capsys.readouterr().out


def test_merge_refuses_a_commit_the_target_has_moved_past(tmp_path, monkeypatch, capsys):
    main, feat, tip = _repos(tmp_path)
    assert _record(tmp_path, feat, monkeypatch, *CHECKED) == 0
    (main / "c.txt").write_text("c\n")
    _git(main, "add", "c.txt")
    _git(main, "commit", "-q", "-m", "main moves")
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 3 and "main has moved past" in capsys.readouterr().err
    _git(main, "checkout", "-q", "-b", "other")
    assert merge_main(["feat", "--dry-run"]) == 3 and "not main" in capsys.readouterr().err


def test_a_train_checked_on_the_one_ahead_lands_after_it_without_a_recheck(tmp_path, monkeypatch, capsys):
    """B was checked on A's train; A lands as a merge commit B lacks, with A's tree. Merging B then gives B's tree, the checked
    one, so B's attempt counts; once main gains a change B lacks, it no longer does."""
    main, _feat, _tip = _repos(tmp_path)
    _branch(main, "A", "x.txt")
    _git(main, "checkout", "-q", "-b", "B", "A")
    (main / "y.txt").write_text("y\n")
    _git(main, "add", "y.txt")
    _git(main, "commit", "-q", "-m", "B on A")
    b = _git(main, "rev-parse", "HEAD")
    _git(main, "checkout", "-q", "main")
    _git(main, "worktree", "add", "-q", str(tmp_path / "b"), b)
    assert _record(tmp_path, tmp_path / "b", monkeypatch, *CHECKED) == 0
    monkeypatch.chdir(main)
    _git(main, "merge", "-q", "--no-ff", "-m", "A lands", "A")
    assert merge_main(["B"]) == 0, capsys.readouterr().err
    assert _git(main, "rev-parse", "HEAD^{tree}") == _git(main, "rev-parse", f"{b}^{{tree}}") and _git(main, "rev-parse", "HEAD^2") == b
    _git(main, "reset", "-q", "--hard", "HEAD^1")
    (main / "c.txt").write_text("c\n")
    _git(main, "add", "c.txt")
    _git(main, "commit", "-q", "-m", "main moves on its own")
    assert merge_main(["B"]) == 3 and "main has moved past" in capsys.readouterr().err


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
    monkeypatch.setenv("FAKE_CHECK", "import os, sys; os.path.exists('BAD') and sys.exit(1); " + _RESULT % "passed")
    cmd = ["--tool", "mytools.decl_check:T", "--command", shlex.join(CHECKED)]
    assert merge_main(["--train", "A", "B", "C", *cmd]) == 1
    out = capsys.readouterr().out
    assert "train A B C: B breaks it; main fast-forwarded to" in out and out.count("recorded as") == 3
    assert _git(main, "rev-parse", "HEAD^1") == before and (main / "a.txt").exists() and not (main / "BAD").exists()
    assert "Train:" in _git(main, "log", "-1", "--format=%B")
    assert merge_main(["--train", "C", *cmd]) == 0
    out = capsys.readouterr().out
    assert out.count("recorded as") == 1 and "`check` passed; main fast-forwarded" in out and (main / "c.txt").exists()
    assert not _git(main, "worktree", "list").count("research-train")


def _remote(tmp_path: Path, main: Path) -> Path:
    """A bare `origin` holding main and A, B, BAD (each adding its own file), with A also as pull request 7's head."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(main, "remote", "add", "origin", str(origin))
    for name, path in (("A", "a2.txt"), ("B", "b.txt"), ("BAD", "BAD")):
        _branch(main, name, path)
    _git(main, "push", "-q", "origin", "main", "A", "B", "BAD", "A:refs/pull/7/head")
    return origin


def test_prepare_pushes_the_pinned_train_and_never_touches_the_checkout(tmp_path, monkeypatch, capsys):
    """The lander's train build (old-circuits-and-proofs, 2 Oct: a hand-built train fell back to the main checkout and put an
    unchecked merge on main): run from a checkout on another branch with changes, the train is built in a scratch worktree from
    origin's main, pushed as its branch, and the checkout is as it was; a second train stacks on the first's branch."""
    main, _feat, _tip = _repos(tmp_path)
    origin = _remote(tmp_path, main)
    a, b = _git(main, "rev-parse", "A"), _git(main, "rev-parse", "B")
    _git(main, "checkout", "-q", "-b", "lander-work", "BAD")
    (main / "a.txt").write_text("uncommitted\n")
    monkeypatch.chdir(main)
    lint = ["--lint", "sh -c 'test ! -e BAD'"]
    assert merge_main(["--train", "--prepare", "train-prep/1", f"7@{a[:9]}", *lint]) == 0, capsys.readouterr().err
    tip = _git(origin, "rev-parse", "train-prep/1")
    out = capsys.readouterr().out
    assert f"pushed origin/train-prep/1 = {tip[:12]} (tree {_git(origin, 'rev-parse', 'train-prep/1^{tree}')[:12]})" in out
    assert f"`tools/check/train.sh {tip[:12]} MACHINE`" in out, "the branch is only on origin; train.sh resolves the sha here"
    assert _git(origin, "rev-parse", "train-prep/1^1") == _git(origin, "rev-parse", "main") and _git(origin, "rev-parse", "train-prep/1^2") == a
    assert _git(origin, "log", "-1", "--format=%s", "train-prep/1") == f"Merge #7 ({a[:8]})"
    assert _git(main, "symbolic-ref", "--short", "HEAD") == "lander-work" and (main / "a.txt").read_text() == "uncommitted\n"
    assert _git(main, "rev-parse", "HEAD") == _git(main, "rev-parse", "BAD") and "research-train" not in _git(main, "worktree", "list")
    assert merge_main(["--train", "--prepare", "train-prep/2", "--onto", "train-prep/1", f"B@{b[:9]}", *lint]) == 0
    assert _git(origin, "rev-parse", "train-prep/2^1") == tip and _git(origin, "rev-parse", "train-prep/2^2") == b


def test_prepares_default_lint_runs_under_uv_when_this_interpreter_lacks_pytest(monkeypatch):
    """A system python3 has no pytest, so the preflight's lints failed under it (old-circuits-and-proofs, 2 Oct)."""
    import importlib.util
    import shutil

    from research import merge

    real = importlib.util.find_spec
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/uv" if name == "uv" else None)
    assert merge.preflight_argv() == [sys.executable, *merge.PREFLIGHT]
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "pytest" else real(name, *a))
    assert merge.preflight_argv() == ["/opt/uv", "run", "--locked", "python", *merge.PREFLIGHT]
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert merge.preflight_argv() == [sys.executable, *merge.PREFLIGHT], "without uv, this interpreter is all there is"


def test_prepare_refuses_a_moved_head_a_conflict_and_a_failing_lint_and_pushes_nothing(tmp_path, monkeypatch, capsys):
    main, _feat, _tip = _repos(tmp_path)
    origin = _remote(tmp_path, main)
    a = _git(main, "rev-parse", "A")
    _git(main, "checkout", "-q", "A")
    (main / "a2.txt").write_text("A, reworked\n")
    _git(main, "commit", "-q", "-am", "A moves")
    _git(main, "push", "-q", "origin", "HEAD:refs/pull/7/head")
    _git(main, "checkout", "-q", "main")
    monkeypatch.chdir(main)
    lint = ["--lint", "sh -c 'test ! -e BAD'"]
    assert merge_main(["--train", "--prepare", "train-prep/x", f"#7@{a[:9]}", *lint]) == 3
    assert "#7 has moved: its head on origin is" in capsys.readouterr().err
    assert merge_main(["--train", "--prepare", "train-prep/x", f"BAD@{_git(main, 'rev-parse', 'BAD')}", *lint]) == 3
    assert "failed on the train" in capsys.readouterr().err
    _git(main, "checkout", "-q", "-b", "C", "main")
    (main / "b.txt").write_text("C's b\n")
    _git(main, "add", "b.txt")
    _git(main, "commit", "-q", "-m", "C")
    _git(main, "checkout", "-q", "main")
    assert merge_main(["--train", "--prepare", "train-prep/x", "B", "C", *lint]) == 3
    assert "C does not merge onto the train so far" in capsys.readouterr().err
    assert _git(origin, "for-each-ref", "refs/heads/train-prep") == "" and "research-train" not in _git(main, "worktree", "list")


def test_prepare_refuses_a_train_the_gate_would_refuse_for_a_stacked_base_it_does_not_pin(tmp_path, monkeypatch, capsys):
    """The lander's 15:41Z refusal (5 Oct): `--prepare` pushed a train holding #1204, stacked on #1203's open head, which only the
    landing gate then refused.  Pinning the base below it builds the train."""
    main, _feat, _tip = _repos(tmp_path)
    origin, heads = _origin_with(tmp_path, main, {"base.txt": "base\n"}, {8: {"e.txt": "8\n"}})
    _git(main, "checkout", "-q", "-b", "pr9", "pr8")
    (main / "n.txt").write_text("9\n")
    _git(main, "add", "n.txt")
    _git(main, "commit", "-q", "-m", "9 on 8")
    heads[9] = _git(main, "rev-parse", "HEAD")
    _git(main, "checkout", "-q", "main")
    _git(main, "push", "-q", "origin", "pr9:refs/pull/9/head")
    monkeypatch.chdir(main)
    assert merge_main(["--train", "--prepare", "train-prep/s", f"9@{heads[9][:9]}", "--lint", "true"]) == 3
    err = capsys.readouterr().err
    assert f"#9 at {heads[9][:12]} is stacked on #8 at {heads[8][:12]}, which is neither on main nor pinned below" in err
    assert "the merge gate would refuse the train" in err and _git(origin, "for-each-ref", "refs/heads/train-prep") == ""
    pins = [f"8@{heads[8][:9]}", f"9@{heads[9][:9]}"]
    assert merge_main(["--train", "--prepare", "train-prep/s", *pins, "--lint", "true"]) == 0, capsys.readouterr().err
    assert _git(origin, "rev-parse", "train-prep/s^2") == heads[9]


def test_a_train_that_would_revert_a_landed_change_is_refused_unless_its_path_is_allowed(tmp_path, monkeypatch, capsys):
    """ci's #1142 train (`25157eb17`) undid #1012: #1142's merge of main dropped #1012's lines, and merging #1142 then removed
    them from main without a word. A line the train loses must be one a ref removes in a commit of its own."""
    from research.merge import reverted

    main, _feat, _tip = _repos(tmp_path)
    origin = _remote(tmp_path, main)
    (main / "w.txt").write_text("old\nx1\nx2\nx3\nkeep\n")
    _git(main, "add", "w.txt")
    _git(main, "commit", "-q", "-m", "w")
    _git(main, "checkout", "-q", "-b", "P", "main")
    (main / "p.txt").write_text("P\n")
    _git(main, "add", "p.txt")
    _git(main, "commit", "-q", "-m", "P's own change")
    _git(main, "checkout", "-q", "-b", "Q", "main")
    (main / "w.txt").write_text("x1\nx2\nx3\nkeep\n")
    _git(main, "commit", "-q", "-am", "Q removes old itself")
    _git(main, "checkout", "-q", "main")
    (main / "w.txt").write_text("old\nx1\nx2\nx3\nkeep\nlanded\n")
    _git(main, "commit", "-q", "-am", "a landed change")
    _git(main, "checkout", "-q", "P")
    _git(main, "merge", "-q", "-s", "ours", "-m", "merge main, dropping what it gained", "main")
    _git(main, "checkout", "-q", "main")
    _git(main, "push", "-q", "origin", "main", "P", "Q")
    p, q, base = _git(main, "rev-parse", "P"), _git(main, "rev-parse", "Q"), _git(main, "rev-parse", "main")

    def onto_base(sha: str) -> str:
        return _git(main, "commit-tree", _git(main, "merge-tree", "--write-tree", base, sha), "-p", base, "-p", sha, "-m", "t")

    assert reverted(main, base, onto_base(q), [("Q", q)]) == []
    lost = reverted(main, base, onto_base(p), [("P", p)])
    assert len(lost) == 1 and lost[0].startswith("w.txt loses 1 line(s)") and "'landed'" in lost[0]
    monkeypatch.chdir(main)
    lint = ["--lint", "true"]
    assert merge_main(["--train", "--prepare", "train-prep/r", f"P@{p[:9]}", *lint]) == 3
    err = capsys.readouterr().err
    assert "would revert what landed: w.txt loses 1 line(s)" in err and "--allow-removal PATH" in err
    assert _git(origin, "for-each-ref", "refs/heads/train-prep") == ""
    assert merge_main(["--train", "--prepare", "train-prep/r", f"P@{p[:9]}", "--allow-removal", "w.txt", *lint]) == 0
    assert merge_main(["--train", "--prepare", "train-prep/q", f"Q@{q[:9]}", *lint]) == 0, capsys.readouterr().err


def test_keep_both_imports_takes_ours_then_the_lines_theirs_adds():
    from research.merge import keep_both_imports

    text = "-- head\n<<<<<<< HEAD\nimport A\nimport B\n\n=======\nimport B\nfrom x import y\n\n>>>>>>> abc1234 (C)\nrest\n"
    assert keep_both_imports(text) == "-- head\nimport A\nimport B\n\nfrom x import y\nrest\n"
    assert keep_both_imports(text.replace("from x import y", "def f := 1")) is None
    assert keep_both_imports("no conflict\n") is None and keep_both_imports("<<<<<<< HEAD\nimport A\n=======\n") is None


def _files_branch(main: Path, name: str, files: dict[str, str]) -> str:
    _git(main, "checkout", "-q", "-b", name, "main")
    for path, text in files.items():
        (main / path).write_text(text)
    _git(main, "add", *files)
    _git(main, "commit", "-q", "-m", name)
    _git(main, "checkout", "-q", "main")
    return _git(main, "rev-parse", name)


def _origin_with(tmp_path: Path, main: Path, files: dict[str, str], prs: dict[int, dict[str, str]]) -> tuple[Path, dict[int, str]]:
    """A bare `origin` holding main with ``files`` added, and pull request N's head adding ``prs[N]`` on that main."""
    for path, text in files.items():
        (main / path).write_text(text)
    _git(main, "add", *files)
    _git(main, "commit", "-q", "-m", "base")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(main, "remote", "add", "origin", str(origin))
    heads = {n: _files_branch(main, f"pr{n}", f) for n, f in prs.items()}
    _git(main, "push", "-q", "origin", "main", *[f"pr{n}:refs/pull/{n}/head" for n in prs])
    return origin, heads


def test_a_conflict_only_on_import_lines_keeps_both_sides_and_any_other_line_refuses(tmp_path, monkeypatch, capsys):
    """Stacked PRs each adding a module to one root import list (old-circuits-and-proofs resolved #850, #857, #860, #861, #863
    and #866 on FlockSoundness.lean by hand, 2 Oct): the train keeps ours, then the lines theirs adds, and says so in the
    merge's `Imports:` trailer; a conflict holding any other line still refuses, and nothing is pushed."""
    main, _feat, _tip = _repos(tmp_path)
    tail = "\ntheorem t : True := trivial\n"
    origin, heads = _origin_with(tmp_path, main, {"Root.lean": "import A\n" + tail}, {
        8: {"Root.lean": "import A\nimport B\n" + tail, "B.lean": "def b := 1\n"},
        9: {"Root.lean": "import A\nimport C\n" + tail},
        10: {"Root.lean": "import A\n-- D's note\nimport D\n" + tail}})
    monkeypatch.chdir(main)
    pins = [f"{n}@{heads[n][:9]}" for n in (8, 9, 10)]
    assert merge_main(["--train", "--prepare", "train-prep/i", *pins[:2], "--lint", "true"]) == 0, capsys.readouterr().err
    assert _git(origin, "show", "train-prep/i:Root.lean") + "\n" == "import A\nimport B\nimport C\n" + tail
    assert _git(origin, "log", "-1", "--format=%B", "train-prep/i").endswith("Imports: Root.lean")
    assert "Imports:" not in _git(origin, "log", "-1", "--format=%B", "train-prep/i^1"), "#8 merged cleanly"
    capsys.readouterr()
    assert merge_main(["--train", "--prepare", "train-prep/j", *pins, "--lint", "true"]) == 3
    assert "#10 does not merge onto the train so far (a conflict in Root.lean holds more than import lines)" in capsys.readouterr().err
    assert _git(origin, "for-each-ref", "refs/heads/train-prep/j") == "" and "research-train" not in _git(main, "worktree", "list")


def test_a_json_file_git_merges_into_a_repeated_key_refuses_the_train(tmp_path, monkeypatch, capsys):
    """Two PRs adding a guarantee of the same name in one `lean-audit.json`, a few lines apart: git merges the lines cleanly into
    an object with the key twice, which `json.loads` would read as the last one silently."""
    main, _feat, _tip = _repos(tmp_path)
    audit = '{\n  "guarantees": {\n    "a": 1,\n    "m": 5,\n    "z": 9\n  }\n}\n'
    origin, heads = _origin_with(tmp_path, main, {"lean-audit.json": audit}, {
        11: {"lean-audit.json": audit.replace('"a": 1,\n', '"a": 1,\n    "b": 2,\n')},
        12: {"lean-audit.json": audit.replace('"m": 5,\n', '"m": 5,\n    "b": 3,\n')}})
    monkeypatch.chdir(main)
    assert merge_main(["--train", "--prepare", "train-prep/k", "11@" + heads[11][:9], "12@" + heads[12][:9], "--lint", "true"]) == 3
    assert "merging #12 onto the train leaves lean-audit.json unparseable as JSON" in capsys.readouterr().err
    assert _git(origin, "for-each-ref", "refs/heads/train-prep/k") == ""


def _store_remote(tmp_path: Path, monkeypatch) -> None:
    """A store remote (a directory): the gate reads grants only after pulling the target's labels from one."""
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))


def test_a_change_under_a_rules_paths_goes_into_a_train_only_with_its_grant_at_its_head(tmp_path, monkeypatch, capsys):
    """The queue's rule for a path refuses a ref that changes it before its train is built (`--prepare`) or checked (`--train`),
    until the `grant` label is on the pull request's pinned head; a change elsewhere needs none."""
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    audit = {"roots": ["P"], "guarantees": {"P.t": {"type_hash": "1"}}, "reads": {}}
    _origin, heads = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "red-team"\npaths = ["lean-audit.json", "other/"]\n',
        "lean-audit.json": json.dumps(audit, indent=1)}, {
        20: {"lean-audit.json": json.dumps({**audit, "guarantees": {"P.t": {"type_hash": "2"}}}, indent=1)},
        21: {"x.txt": "x\n"}})
    _git(main, "push", "-q", "origin", "pr20:refs/heads/side")
    monkeypatch.chdir(main)
    pins, lint = [f"20@{heads[20][:9]}", f"21@{heads[21][:9]}"], ["--lint", "true"]
    assert merge_main(["--train", "--prepare", "train-prep/g", *pins, *lint]) == 3
    err = capsys.readouterr().err
    assert f"#20 needs the red-team grant at {heads[20][:12]} (tools/check/queue.toml: from its rule, which names no `by`)" in err
    assert f"`research data label pr:20@{heads[20]} grant red-team --by REVIEWER`" in err and "#21" not in err
    assert merge_main(["--train", "--prepare", "train-prep/g", f"side@{heads[20][:9]}", *lint]) == 3
    assert "side needs the red-team grant (tools/check/queue.toml), which only a pull request's pinned head can hold" \
        in capsys.readouterr().err
    assert merge_main(["--train", "pr20", "--tool", "mytools.decl_check:T", "--command", shlex.join(CHECKED)]) == 3
    out, err = capsys.readouterr()
    assert "pr20 needs the red-team grant" in err and "nothing was checked" in err and "recorded as" not in out
    LocalStore().add_label(f"pr:20@{heads[20]}", "grant", "red-team", "red-team")
    assert merge_main(["--train", "--prepare", "train-prep/g", *pins, *lint]) == 0, capsys.readouterr().err


def _pr_train(tmp_path: Path, monkeypatch, rules: str | None = None, grant: str | None = None) -> tuple[Path, Path, str, str]:
    """A store remote; main, with ``rules`` as its queue rules; origin, with A as pull request 7's head; and origin's train-prep/t
    merging #7 at that head, fetched here as a branch, with a passing check of its tip: (main, origin, A, the tip).  With
    ``grant``, a role the rules ask of A, `--prepare` is refused until `pr:7@A` holds it."""
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    if rules:
        (main / "tools" / "check" / "queue.toml").write_text(rules)
        _git(main, "add", "-A")
        _git(main, "commit", "-q", "-m", "queue rules")
    origin = _remote(tmp_path, main)
    a = _git(main, "rev-parse", "A")
    monkeypatch.chdir(main)
    if grant:
        assert merge_main(["--train", "--prepare", "train-prep/t", f"7@{a[:9]}", "--lint", "true"]) == 3
        LocalStore().add_label(f"pr:7@{a}", "grant", grant, grant)
    assert merge_main(["--train", "--prepare", "train-prep/t", f"7@{a[:9]}", "--lint", "true"]) == 0
    _git(main, "fetch", "-q", "origin", "train-prep/t:train-prep/t")
    tip = _git(main, "rev-parse", "train-prep/t")
    _git(main, "worktree", "add", "-q", str(tmp_path / "tip"), tip)
    assert _record(tmp_path, tmp_path / "tip", monkeypatch, *CHECKED) == 0
    monkeypatch.chdir(main)
    return main, origin, a, tip


def test_a_tip_lands_only_while_each_pr_it_merges_is_at_its_pinned_head_with_its_grants(tmp_path, monkeypatch, capsys):
    """A tip merging `Merge #1126 (83c25a27b)` would have landed after #1126 moved to a head that fixed two red-team findings
    against that one (the lander, 4 Oct): pins and grants were read at `--prepare` only.  The gate reads them again at landing:
    #7's head on origin must still be the merged one, and `pr:7@{head}` must hold every grant the queue's rules say its change
    needs; a check of the tip doesn't stand in for either.  No ready mark is asked: the train's own check is the gate (Daniel,
    6 Oct)."""
    main, _origin, a, _tip = _pr_train(tmp_path, monkeypatch, '[[grant]]\nrole = "red-team"\npaths = ["a2.txt"]\n', "red-team")
    capsys.readouterr()
    rc, err = merge_main(["train-prep/t", "--dry-run"]), capsys.readouterr().err
    assert rc == 0 and "ready" not in err, err
    _git(main, "checkout", "-q", "-b", "A-again", "A")                     # a new head that adds nothing the tip lacks: lands as A
    _git(main, "commit", "-q", "--allow-empty", "-m", "merge main (nothing new)")
    merged = _git(main, "rev-parse", "HEAD")
    _git(main, "checkout", "-q", "main")
    _git(main, "push", "-q", "-f", "origin", f"{merged}:refs/pull/7/head")
    assert merge_main(["train-prep/t", "--dry-run"]) == 0, capsys.readouterr().err
    assert (f"#7's head moved from {a[:12]} to {merged[:12]}, which adds nothing to" in capsys.readouterr().out)
    _git(main, "push", "-q", "-f", "origin", f"{a}:refs/pull/7/head")
    before = _git(main, "rev-parse", "main")
    _git(main, "checkout", "-q", "A")
    (main / "a2.txt").write_text("A, after the review\n")
    _git(main, "commit", "-q", "-am", "A fixes the findings")
    moved = _git(main, "rev-parse", "HEAD")
    _git(main, "push", "-q", "origin", "HEAD:refs/pull/7/head")
    _git(main, "checkout", "-q", "main")
    assert merge_main(["train-prep/t", "--dry-run"]) == 3
    assert f"#7 was merged at {a[:12]}, but its head on origin is now {moved[:12]}: rebuild the train" in capsys.readouterr().err
    assert merge_main(["train-prep/t"]) == 3 and _git(main, "rev-parse", "HEAD") == before


def test_a_pr_stacked_on_another_in_the_tip_needs_grants_only_for_what_it_adds(tmp_path, monkeypatch, capsys):
    """Lean's #1141 and #1144 stack on #1150, whose C-Flock lock change has red-team's grant: judged from main, each would need
    that grant too.  The gate judges each pull request by what it adds onto the train below its merge, so a base that is neither
    on main nor pinned in the tip at its head is refused, by name, whatever the stacked one's grants (root's condition, 5 Oct),
    and a base pinned without its grant is refused for itself."""
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    origin, heads = _origin_with(tmp_path, main, {"tools/check/queue.toml": '[[grant]]\nrole = "red-team"\npaths = ["f.txt"]\n'},
                                 {30: {"f.txt": "the lock\n"}})
    _git(main, "checkout", "-q", "pr30")
    (main / "g.txt").write_text("on top\n")
    _git(main, "add", "g.txt")
    _git(main, "commit", "-q", "-m", "stacked")
    heads[31] = _git(main, "rev-parse", "HEAD")
    _git(main, "push", "-q", "origin", "HEAD:refs/pull/31/head")
    _git(main, "checkout", "-q", "main")
    monkeypatch.chdir(main)
    store = LocalStore()
    pins = ["--train", "--prepare", "train-prep/s", f"30@{heads[30][:9]}", f"31@{heads[31][:9]}", "--lint", "true"]
    capsys.readouterr()
    assert merge_main(pins) == 3
    err = capsys.readouterr().err
    assert f"#30 needs the red-team grant at {heads[30][:12]}" in err and "#31" not in err
    store.add_label(f"pr:30@{heads[30]}", "grant", "red-team", "red-team")
    assert merge_main(pins) == 0, capsys.readouterr().err
    _git(main, "fetch", "-q", "origin", "train-prep/s:train-prep/s")
    _git(main, "worktree", "add", "-q", str(tmp_path / "tip"), _git(main, "rev-parse", "train-prep/s"))
    assert _record(tmp_path, tmp_path / "tip", monkeypatch, *CHECKED) == 0
    monkeypatch.chdir(main)
    assert merge_main(["train-prep/s", "--dry-run"]) == 0, capsys.readouterr().err

    store.add_label(f"pr:31@{heads[31]}", "grant", "red-team", "red-team")
    # #31 alone: its base #30 is neither on main nor pinned in the tip, refused by name even with a grant at #31's head; `--prepare`
    # refuses to build it, so the tip is merged by hand
    assert merge_main(["--train", "--prepare", "train-prep/u", f"31@{heads[31][:9]}", "--lint", "true"]) == 3
    _git(main, "worktree", "add", "-q", "--detach", str(tmp_path / "tip-u"), "main")
    _git(tmp_path / "tip-u", "merge", "-q", "--no-ff", "-m", f"Merge #31 ({heads[31][:8]})", heads[31])
    assert _record(tmp_path, tmp_path / "tip-u", monkeypatch, *CHECKED) == 0
    monkeypatch.chdir(main)
    capsys.readouterr()
    assert merge_main([_git(tmp_path / "tip-u", "rev-parse", "HEAD"), "--dry-run"]) == 3
    assert f"#31 at {heads[31][:12]} is stacked on #30 at {heads[30][:12]}" in capsys.readouterr().err


def test_a_tip_whose_pr_heads_or_labels_cant_be_read_is_refused(tmp_path, monkeypatch, capsys):
    """The gate fails closed: with origin unreadable it can't tell the pull requests' heads, and with no store remote its labels may be stale
    (`StoreGrants` raises STALE), so it refuses either way, saying why.  `--train` refuses before recording anything."""
    from research.store.local import LocalStore

    main, _origin, a, _tip = _pr_train(tmp_path, monkeypatch, '[[grant]]\nrole = "red-team"\npaths = ["a2.txt"]\n', "red-team")
    assert merge_main(["train-prep/t", "--dry-run"]) == 0, capsys.readouterr().err
    url = _git(main, "remote", "get-url", "origin")
    _git(main, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    assert merge_main(["train-prep/t", "--dry-run"]) == 3
    assert "origin's pull request heads can't be read (" in capsys.readouterr().err
    _git(main, "remote", "set-url", "origin", url)
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent.toml"))
    assert merge_main(["train-prep/t", "--dry-run"]) == 3
    assert "the pull requests" in (err := capsys.readouterr().err) and "can't be judged: STALE" in err
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))
    assert merge_main(["--train", f"7@{a[:9]}", "--tool", "mytools.decl_check:T", "--command", shlex.join(CHECKED)]) == 3
    out, err = capsys.readouterr()
    assert "refused: STALE: the labels of pr:7@" in err and "recorded as" not in out


def _main_moves(main: Path, origin: Path) -> None:
    (main / "c.txt").write_text("main moves\n")
    _git(main, "add", "c.txt")
    _git(main, "commit", "-q", "-m", "main moves")
    _git(main, "push", "-q", "origin", "main")


def _main_merged_into(main: Path, branch: str) -> str:
    _git(main, "checkout", "-q", "-b", f"{branch}-merged", branch)
    _git(main, "merge", "-q", "--no-ff", "--no-edit", "main")
    _git(main, "checkout", "-q", "main")
    return _git(main, "rev-parse", f"{branch}-merged")


def test_a_grant_carries_into_a_train_across_a_restack_or_a_merge_of_main_and_not_to_a_restated_pin(
        tmp_path, monkeypatch, capsys):
    """Daniel's ruling (2 Oct), at the train's gate: the grant on the head that was reviewed counts at a head that only restacks it
    or merges `main` into it, and the train says which head it carried from; a head whose pin says something else needs a new one."""
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    audit = {"roots": ["P"], "pins": {"P.t": {"type_hash": "1"}}, "reads": {}}
    pinned = json.dumps({**audit, "pins": {"P.t": {"type_hash": "2"}}}, indent=1)
    origin, heads = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "red-team"\npaths = ["lean-audit.json", "other/"]\n',
        "lean-audit.json": json.dumps(audit, indent=1)}, {20: {"lean-audit.json": pinned}})
    old = heads[20]
    LocalStore().add_label(f"pr:20@{old}", "grant", "red-team", "reviewer")
    _main_moves(main, origin)
    restacked = _files_branch(main, "pr20-restacked", {"lean-audit.json": pinned})
    merged = _main_merged_into(main, "pr20")
    restated = _files_branch(main, "pr20-restated", {"lean-audit.json": json.dumps({**audit, "pins": {"P.t": {"type_hash": "3"}}})})
    monkeypatch.chdir(main)
    for head, how in ((restacked, "restack"), (merged, "merge of main")):
        _git(main, "push", "-q", "-f", "origin", f"{head}:refs/pull/20/head")
        assert merge_main(["--train", "--prepare", f"train-prep/{how[0]}", f"20@{head[:9]}", "--lint", "true"]) == 0, capsys.readouterr()
        assert (f"research merge: grant red-team carried from pr:20@{old} to {head[:12]} "
                f"({how}; lean-audit.json change identical)") in capsys.readouterr().out
    _git(main, "push", "-q", "-f", "origin", f"{restated}:refs/pull/20/head")
    assert merge_main(["--train", "--prepare", "train-prep/x", f"20@{restated[:9]}", "--lint", "true"]) == 3
    out, err = capsys.readouterr()
    assert f"#20 needs the red-team grant at {restated[:12]}" in err and "nothing was built" in err
    assert (f"research merge: grant red-team on pr:20@{old} doesn't carry to {restated[:12]}: its lean-audit.json patch "
            f"isn't byte-identical (lean-audit.json)") in out
    assert _git(origin, "for-each-ref", "--format=%(refname)", "refs/heads/train-prep") == \
        "refs/heads/train-prep/m\nrefs/heads/train-prep/r"


def test_a_grant_carries_at_landing_as_at_prepare(tmp_path, monkeypatch, capsys):
    """Proofs' restacked PRs (#1325 #1326 #1331 #1307 #1313, 6 Oct) were built into trains on grants carried from their reviewed
    heads, but the landing gate read grants at the pinned head alone and would have refused the checked tip.  It carries them
    under the same rule (`research.queue.carry`, Daniel's Q5), saying from which head."""
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    audit = {"roots": ["P"], "pins": {"P.t": {"type_hash": "1"}}, "reads": {}}
    pinned = json.dumps({**audit, "pins": {"P.t": {"type_hash": "2"}}}, indent=1)
    origin, heads = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "red-team"\npaths = ["lean-audit.json", "other/"]\n',
        "lean-audit.json": json.dumps(audit, indent=1)}, {20: {"lean-audit.json": pinned}})
    old = heads[20]
    LocalStore().add_label(f"pr:20@{old}", "grant", "red-team", "reviewer")
    _main_moves(main, origin)
    restacked = _files_branch(main, "pr20-restacked", {"lean-audit.json": pinned})
    monkeypatch.chdir(main)
    _git(main, "push", "-q", "-f", "origin", f"{restacked}:refs/pull/20/head")
    assert merge_main(["--train", "--prepare", "train-prep/r", f"20@{restacked[:9]}", "--lint", "true"]) == 0, capsys.readouterr()
    _git(main, "fetch", "-q", "origin", "train-prep/r:train-prep/r")
    _git(main, "worktree", "add", "-q", str(tmp_path / "tip"), _git(main, "rev-parse", "train-prep/r"))
    assert _record(tmp_path, tmp_path / "tip", monkeypatch, *CHECKED) == 0
    monkeypatch.chdir(main)
    capsys.readouterr()
    assert merge_main(["train-prep/r", "--dry-run"]) == 0, capsys.readouterr().err
    assert f"research merge: grant red-team carried from pr:20@{old} to {restacked[:12]} (restack" in capsys.readouterr().out


def test_a_grant_on_a_stacked_pr_carries_when_only_the_pr_below_it_moved(tmp_path, monkeypatch, capsys):
    """Proofs' 12 stacked heads (7 Oct): judged against `main`, a stacked PR's change holds its parent's, so any new commit on the
    parent voided the child's grant though the child's own patch was byte-identical.  The gate takes each head's change against
    the PR below it in the train, as it does for the grants a PR needs, at `--prepare` and at landing."""
    from research.queue import carry
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    audit = {"roots": ["P"], "pins": {"P.t": {"type_hash": "1"}}, "reads": {}}
    pinned = json.dumps({**audit, "pins": {"P.t": {"type_hash": "2"}}}, indent=1)
    origin, heads = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "red-team"\npaths = ["lean-audit.json"]\n',
        "lean-audit.json": json.dumps(audit, indent=1)}, {30: {"p.txt": "parent\n"}})

    def commit_on(start: str, branch: str, files: dict[str, str]) -> str:
        _git(main, "checkout", "-q", "-b", branch, start)
        for path, text in files.items():
            (main / path).write_text(text)
        _git(main, "add", *files)
        _git(main, "commit", "-q", "-m", branch)
        _git(main, "checkout", "-q", "main")
        return _git(main, "rev-parse", branch)

    old = commit_on(heads[30], "pr31", {"lean-audit.json": pinned})
    LocalStore().add_label(f"pr:31@{old}", "grant", "red-team", "reviewer")
    parent = commit_on(heads[30], "pr30-fixed", {"p.txt": "parent, fixed\n"})
    new = commit_on(parent, "pr31-restacked", {"lean-audit.json": pinned})
    _git(main, "push", "-q", "-f", "origin", f"{parent}:refs/pull/30/head", f"{new}:refs/pull/31/head")
    assert carry(main, "main", old, new) == (False, "its change against main differs (p.txt)"), "from main, the parent's commit counts"
    monkeypatch.chdir(main)
    capsys.readouterr()
    pins = [f"30@{parent[:9]}", f"31@{new[:9]}"]
    assert merge_main(["--train", "--prepare", "train-prep/st", *pins, "--lint", "true"]) == 0, capsys.readouterr().err
    said = f"research merge: grant red-team carried from pr:31@{old} to {new[:12]} (restack; lean-audit.json change identical)"
    assert said in capsys.readouterr().out
    _git(main, "fetch", "-q", "origin", "train-prep/st:train-prep/st")
    _git(main, "worktree", "add", "-q", str(tmp_path / "tip"), _git(main, "rev-parse", "train-prep/st"))
    assert _record(tmp_path, tmp_path / "tip", monkeypatch, *CHECKED) == 0
    monkeypatch.chdir(main)
    capsys.readouterr()
    assert merge_main(["train-prep/st", "--dry-run"]) == 0, capsys.readouterr().err
    assert said in capsys.readouterr().out


def test_a_head_with_two_merge_bases_carries_no_grant_and_needs_every_one(tmp_path, monkeypatch, capsys):
    """Red team of #955: a head merging #30's commit (which `main` merged too) into the reviewed head while keeping the reviewed
    tree drops #30's pins.  Its merge bases with `main` are both the reviewed head's base and #30's commit; `git merge-base`
    printed the first (the newer), against which the head changes what the reviewed one did, so the grant carried and the train
    reverted #30's pins.  `git merge` diffs against all of them: the grant doesn't carry, and the rules, reading the tree the
    merge writes, see the pins it reverts."""
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    audit = {"roots": ["P"], "pins": {"P.t": {"type_hash": "1"}}, "reads": {}}
    (main / "other").mkdir()
    days = iter(range(1, 10))

    def next_day() -> None:
        monkeypatch.setenv("GIT_COMMITTER_DATE", f"2026-01-0{next(days)}T00:00:00 +0000")

    next_day()
    origin, _ = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "red-team"\npaths = ["lean-audit.json", "other/"]\n',
        "lean-audit.json": json.dumps(audit, indent=1),
        "other/lean-audit.json": json.dumps({**audit, "pins": {"O.t": {"type_hash": "old"}}}, indent=1)}, {})
    next_day()
    theirs = _files_branch(main, "pr30", {"other/lean-audit.json": json.dumps({**audit, "pins": {"O.t": {"type_hash": "new"}}})})
    next_day()
    _main_moves(main, origin)
    next_day()
    old = _files_branch(main, "pr20", {"lean-audit.json": json.dumps({**audit, "pins": {"P.t": {"type_hash": "2"}}}, indent=1)})
    LocalStore().add_label(f"pr:20@{old}", "grant", "red-team", "reviewer")
    next_day()
    _git(main, "merge", "-q", "--no-ff", "--no-edit", "pr30")
    _git(main, "push", "-q", "origin", "main")
    next_day()
    new = _git(main, "commit-tree", f"{old}^{{tree}}", "-p", old, "-p", theirs, "-m", "Merge branch pr30")
    _git(main, "push", "-q", "origin", f"{new}:refs/pull/20/head")
    monkeypatch.chdir(main)
    assert merge_main(["--train", "--prepare", "train-prep/c", f"20@{new[:9]}", "--lint", "true"]) == 3
    out, err = capsys.readouterr()
    tip = _git(main, "rev-parse", "main")
    listed = _git(main, "merge-base", "--all", tip, new).split()
    assert sorted(listed) == sorted([_git(main, "rev-parse", "main^1"), theirs])
    assert _git(main, "merge-base", tip, new) == listed[0] == _git(main, "rev-parse", "main^1"), "the base the grant would carry over"
    bases = ", ".join(b[:12] for b in listed)
    merged = _git(main, "merge-tree", "--write-tree", tip, new)
    assert json.loads(_git(main, "show", f"{merged}:other/lean-audit.json"))["pins"] == {"O.t": {"type_hash": "old"}}
    assert (f"research merge: grant red-team on pr:20@{old} doesn't carry to {new[:12]}: {new[:12]} has 2 merge bases "
            f"with {tip[:12]} ({bases}), so no one diff says what merging it changes; merging {tip[:12]} into it again gives it "
            f"one") in out
    assert f"#20 needs the red-team grant at {new[:12]}" in err and "nothing was built" in err
    assert _git(origin, "for-each-ref", "refs/heads/train-prep") == ""


def test_a_train_the_store_cant_answer_for_is_refused_as_stale_not_as_the_rules(tmp_path, monkeypatch, capsys):
    from research.store import labels_sync

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    audit = {"roots": ["P"], "pins": {"P.t": {"type_hash": "1"}}, "reads": {}}
    _origin, heads = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "red-team"\npaths = ["lean-audit.json", "other/"]\n',
        "lean-audit.json": json.dumps(audit, indent=1)}, {20: {"lean-audit.json": json.dumps({**audit, "pins": {"P.t": {"type_hash": "2"}}})}})

    def unreachable(store, prefix):
        raise OSError("injected")

    monkeypatch.setattr(labels_sync, "pull_prefix", unreachable)
    monkeypatch.chdir(main)
    assert merge_main(["--train", "--prepare", "train-prep/s", f"20@{heads[20][:9]}", "--lint", "true"]) == 3
    err = capsys.readouterr().err
    assert "refused: STALE: the labels of pr:20@* weren't pulled from the store's remote (OSError: injected)" in err
    assert "the queue's rules" not in err


def test_merge_refuses_an_attempt_whose_result_is_the_lean_fast_paths_whatever_tool_recorded_it(tmp_path, monkeypatch, capsys):
    """The coordinator's ruling: no merge accepts the Lean fast path's result (`lean_changed.py --records`, no kernel replay).
    An attempt of `check` that passed on the exact commit is refused when its result is of a kind in `REFUSED_KINDS`, or can't
    be read; the push guard reads the same `passing`. A `check/v1` result of the same commit merges."""
    from research.merge import REFUSED_KINDS, unchecked_pushes
    from research.store import ids
    from research.store.local import LocalStore

    main, feat, tip = _repos(tmp_path)
    base = _git(main, "rev-parse", "HEAD")

    def run(kind: str) -> int:
        return _checked(tmp_path, feat, monkeypatch, (_RESULT % "passed").replace("check/v1", kind))

    assert "lean-fast/v1" in REFUSED_KINDS and run("lean-fast/v1") == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 3
    assert "refused: its result is a lean-fast/v1, which no merge accepts" in capsys.readouterr().err
    assert unchecked_pushes(main, [f"refs/heads/main {tip} refs/heads/main {base}"], LocalStore()) == [tip]
    assert run("check/v1") == 0
    store = LocalStore()
    results = [store.get_attempt(r[0])["outputs"]["result"] for r in store.index.sql("SELECT id FROM attempts WHERE tool = 'check'")]
    art = next(a for a in results if store.get_manifest(a).kind == "check/v1")
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 0
    store.manifest_path(ids.hex_of(art)).unlink()
    assert merge_main(["feat", "--dry-run"]) == 3
    assert "refused: its result can't be read (NotFound" in capsys.readouterr().err


def test_merge_admits_only_a_result_of_a_kind_the_targets_declaration_produces(tmp_path, monkeypatch, capsys):
    """Refusing the kinds it knew to refuse let through a `check` attempt with no result (`research run --tool check -- python3
    tools/lean/audit.py --no-replay PKG` writes none) and a result of any other kind (red team, #966).  The gate admits only a
    kind the target's own declaration produces (`check/v1` for check), and nothing when the target declares none."""
    from research.merge import produced

    main, feat, _tip = _repos(tmp_path)
    assert _checked(tmp_path, feat, monkeypatch, "pass") == 0
    assert _checked(tmp_path, feat, monkeypatch, (_RESULT % "passed").replace("check/v1", "bench-result/v1")) == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 3
    err = capsys.readouterr().err
    assert "refused: it published no result, and the target's declaration of the tool produces check/v1" in err
    assert "refused: its result is a bench-result/v1, and the target's declaration of the tool produces check/v1" in err
    (main / "tools" / "check" / "tool.py").write_text(DECL.replace(", produces=['check/v1']", "").format(""))
    _git(main, "commit", "-q", "-am", "check produces nothing")
    _git(feat, "merge", "-q", "main")
    assert _checked(tmp_path, feat, monkeypatch) == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 3
    assert "refused: its result is a check/v1, and the target's declaration of the tool produces nothing" in capsys.readouterr().err
    assert produced(tmp_path, "check") == frozenset(), "no declaration at all"


#: a "check" that runs no check: one line of shell writing what check.py writes, with the verdict STATUS
_FORGED = ("printf '%s' '{\"schema\": \"research/result/v0.1\", \"kind\": \"check/v1\", \"run_id\": \"'\"$RESEARCH_RUN_ID\"'\", "
           "\"validation\": {\"status\": \"STATUS\", \"checker\": \"sh\"}, \"steps\": []}' > \"$RESEARCH_RUN_DIR/result.json\"")


def test_merge_admits_only_a_run_of_the_declared_command_whose_own_result_passed_once(tmp_path, monkeypatch, capsys):
    """The canary's control was a check that ran nothing (red team, #990): `research run --tool check -- sh -c '<write a check/v1
    result.json>'` passed the gate, with its verdict failed too, since the gate read neither the attempt's argv nor the result's
    verdict.  An attempt passes only as a run of the command the target declares (`Tool.command`) whose own result passed, and
    lists no step twice (`step_status` reads the first, the train gate the last)."""
    main, feat, _tip = _repos(tmp_path)
    for status in ("passed", "failed"):
        assert _record(tmp_path, feat, monkeypatch, "sh", "-c", _FORGED.replace("STATUS", status)) == 0
    ok = _RESULT % "passed"
    assert _checked(tmp_path, feat, monkeypatch, ok.replace("{'status': 'passed'", "{'status': 'failed'")) == 0
    twice = "[{'step': 'agree', 'status': 'skipped'}, {'step': 'agree', 'status': 'passed'}]"
    assert _checked(tmp_path, feat, monkeypatch, ok.replace("[{'step': 'agree', 'status': 'passed'}]", twice)) == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 3
    err = capsys.readouterr().err
    assert err.count(f"isn't the target's declared `{shlex.join(CHECKED)}`") == 2
    assert "refused: its result's verdict is failed, not passed" in err and "refused: its result lists step agree more than once" in err
    assert _checked(tmp_path, feat, monkeypatch) == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 0


def test_the_layout_moves_check_at_the_moved_paths_is_the_declared_command_for_the_moves_head_alone(tmp_path, monkeypatch, capsys):
    """Main declares `tools/check/check.py` until the layout move lands, and a check of the moved tree can only run
    `tools/verity/check/check.py`, so no check of the move could pass (the move's check dc20, 7 Oct). `MOVED_CHECK` reads the moved
    paths as the declared ones for the move's head and nothing else."""
    import research.merge as M

    main, feat, tip = _repos(tmp_path)
    (feat / "tools" / "verity" / "check").mkdir(parents=True)
    (feat / "tools" / "verity" / "check" / "check.py").write_text(CHECK_PY)
    _git(feat, "add", "-A")
    _git(feat, "commit", "-q", "-m", "the move")
    tip = _git(feat, "rev-parse", "HEAD")
    moved = (sys.executable, "tools/verity/check/check.py")
    monkeypatch.delenv("FAKE_CHECK", raising=False)
    assert _record(tmp_path, feat, monkeypatch, *moved) == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 3
    assert f"isn't the target's declared `{shlex.join(CHECKED)}`" in capsys.readouterr().err
    monkeypatch.setattr(M, "MOVED_CHECK", {"f" * 40: {"tools/verity/check/check.py": "tools/check/check.py"}})
    assert merge_main(["feat", "--dry-run"]) == 3, "another commit's entry admits nothing"
    capsys.readouterr()
    monkeypatch.setattr(M, "MOVED_CHECK", {tip: {"tools/verity/check/check.py": "tools/check/check.py"}})
    assert merge_main(["feat", "--dry-run"]) == 0
    monkeypatch.undo()
    head, declared = next(iter(M.MOVED_CHECK)), ("uv", "run", "--locked", "--extra", "torch-cpu", "python", "tools/check/check.py")
    ran = ["python3", "tools/verity/check/slot.py", "--priority", "--by", "x", "--", "uv", "run", "--locked", "--extra", "torch-cpu",
           "python", "tools/verity/check/check.py", "--keep-going", "--verdicts-in", "v.tgz"]
    assert M.unlike(M.as_declared(ran, head), declared) is None
    assert "isn't the target's declared" in M.unlike(M.as_declared(ran, "f" * 40), declared)


def test_a_run_of_the_declared_command_is_it_in_a_check_slot_or_not_with_check_pys_own_flags_at_most():
    """What `check.py --record [--on]`, `train.sh`, the jobs dispatch and `research merge --train` record: the declared command,
    behind `slot.py` when it was sent to a machine, followed by check.py's own flags.  Anything else is another command."""
    from research.merge import unlike

    cmd = ("uv", "run", "--locked", "python", "tools/check/check.py")
    slot = ["python3", "tools/check/slot.py"]
    for argv in ([*cmd], [*cmd, "--keep-going"], [*cmd, "--no-cache", "--cores", "8"], [*cmd, "--keep-going", "--verdicts-in", "v.tgz"],
                 [*cmd, "--verdicts-in", "v.tgz", "--cores", "4"], [*slot, "--", *cmd], [*slot, "--by", "x", "--", *cmd],
                 [*slot, "--priority", "--by", "gates", "--", *cmd, "--keep-going"]):
        assert unlike(argv, cmd) is None, argv
    other, extra = "isn't the target's declared", "which check.py doesn't take"
    for argv, why in (([], other), (None, "its record has no argv"), ([*cmd[:-1], "check.py"], other), (["env", "X=1", *cmd], other),
                      (["bash", "-c", f"{shlex.join(cmd)} || true"], other), ([*slot, *cmd], other), ([*slot, "--priority", *cmd], other),
                      ([*cmd, "--preflight"], extra), ([*cmd, "--record"], extra), ([*cmd, "&&", "true"], extra),
                      ([*cmd, "--cores"], extra), ([*cmd, "--cores", "x"], extra), ([*cmd, "--cores", "2", "--cores", "2"], extra),
                      ([*cmd, "--verdicts-in"], extra), ([*cmd, "--verdicts-in", "a", "--verdicts-in", "b"], extra)):
        assert why in (unlike(argv, cmd) or ""), argv
    assert unlike([*cmd], ()) == "the target's declaration of the tool names no command to hold it to"


def test_a_refused_kind_or_another_runs_result_passes_no_gate_even_where_the_declaration_produces_its_kind(tmp_path):
    """`REFUSED_KINDS` (the Lean fast path's lean-fast/v1, an audit without the kernel replay) holds whatever a declaration says
    (red team, #990): a passing run of the declared command whose result is one is refused for its kind alone.  A result that
    names another run (which `research run` doesn't publish) is refused too."""
    from research.merge import refused
    from research.store.local import LocalStore

    store = LocalStore(tmp_path / "store", config_path=tmp_path / "none.toml")
    commit, kinds = "c" * 40, frozenset({"check/v1", "lean-fast/v1"})

    def attempt(rid: str, kind: str, of: str) -> str:
        meta = {"schema": "research/result/v0.1", "kind": kind, "run_id": of, "validation": {"status": "passed"}, "steps": []}
        store.put_attempt({
            "schema": "research/attempt/v0.1", "id": rid, "tool": {"name": "check", "version": None}, "derivation": "drv:" + "0" * 64,
            "reuse_key": None, "argv": list(CHECKED), "params": {}, "key_params": [], "inputs": {}, "conditions": {},
            "key_conditions": [], "source": {"commit": commit, "dirty": False}, "closure": None, "execution": {}, "cost": None,
            "status": {"state": "done", "rc": 0}, "outputs": {"result": store.put_artifact(kind, meta)}, "telemetry": None})
        return rid

    fast = attempt("r20261003-000000-abcd", "lean-fast/v1", "r20261003-000000-abcd")
    assert refused(store, fast, kinds, "check", commit, CHECKED) == "its result is a lean-fast/v1, which no merge accepts"
    assert refused(store, fast, kinds, "check", commit) == "the target's declaration of the tool names no command to hold it to"
    other = attempt("r20261003-000001-abcd", "check/v1", "r-other")
    assert refused(store, other, kinds, "check", commit, CHECKED) == "its result names run r-other, not this one"
    assert refused(store, attempt("r20261003-000002-abcd", "check/v1", "r20261003-000002-abcd"), kinds, "check", commit, CHECKED) is None


def test_a_train_whose_command_isnt_the_declared_one_is_refused_before_anything_is_checked(tmp_path, monkeypatch, capsys):
    """`research merge --train --command "bash -c '... check.py || true'"` (red team, #990): the gate refuses every attempt of it,
    so the train's check and its bisection would run for nothing.  Check.py's own flags are still the declared command."""
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))
    monkeypatch.delenv("FAKE_CHECK", raising=False)
    main, _feat, _tip = _repos(tmp_path)
    _branch(main, "A", "x.txt")
    monkeypatch.chdir(main)
    tool = ["--tool", "mytools.decl_check:T"]
    assert merge_main(["--train", "A", *tool, "--command", shlex.join(["bash", "-c", f"{shlex.join(CHECKED)} || true"])]) == 3
    out, err = capsys.readouterr()
    assert "the train's check would pass no gate: its command `bash -c" in err and "nothing was checked" in err
    assert "recorded as" not in out
    assert merge_main(["--train", "A", *tool, "--command", shlex.join([*CHECKED, "--keep-going"])]) == 0, capsys.readouterr().err
    assert capsys.readouterr().out.count("recorded as") == 1 and (main / "x.txt").exists()


def test_merge_refuses_an_attempt_whose_record_cant_be_read_or_disagrees_with_the_index(tmp_path, monkeypatch, capsys):
    """The index answers `passing`'s query, but the attempt's record is what says what ran: with it gone from the store (and no
    remote to read it from), unreadable, or saying something other than the index row did, the gate can't tell what passed."""
    from research.store.local import LocalStore

    main, feat, _tip = _repos(tmp_path)
    assert _record(tmp_path, feat, monkeypatch, *CHECKED) == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 0
    capsys.readouterr()
    store = LocalStore()
    (rid,) = [r[0] for r in store.index.sql("SELECT id FROM attempts WHERE tool = 'check'")]
    path = store.attempt_path(rid)
    doc = json.loads(path.read_text())
    path.unlink()
    assert merge_main(["feat", "--dry-run"]) == 3
    assert f"{rid} done rc=0 refused: its attempt can't be read" in capsys.readouterr().err
    path.write_text("{not json")
    assert merge_main(["feat", "--dry-run"]) == 3
    assert f"{rid} done rc=0 refused: its attempt can't be read (JSONDecodeError" in capsys.readouterr().err
    path.write_text(json.dumps({**doc, "status": {**doc["status"], "rc": 1}}))
    assert merge_main(["feat", "--dry-run"]) == 3
    assert f"{rid} done rc=0 refused: its record says rc 1, not the index's passing check" in capsys.readouterr().err


def test_the_gate_reads_its_own_checkout_whatever_git_environment_it_inherits(tmp_path, monkeypatch, capsys):
    """Run from a hook or under another checkout's scratch index, git reads GIT_DIR, GIT_WORK_TREE and GIT_INDEX_FILE before the
    cwd: the gate would judge another repository's refs and status.  Its git calls drop them (`research.gates.git_env`)."""
    main, feat, tip = _repos(tmp_path)
    assert _record(tmp_path, feat, monkeypatch, *CHECKED) == 0
    other = tmp_path / "other"
    _git(tmp_path, "init", "-q", "-b", "main", str(other))
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "scratch-index"))
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 0, capsys.readouterr().err
    assert f"({tip[:12]}) may be merged into main" in capsys.readouterr().out


def test_a_path_rule_requires_its_step_for_a_change_under_the_path(tmp_path, monkeypatch, capsys):
    """The target's declaration says `agree` must pass for a change under flock/: an attempt that skipped it admits a change
    elsewhere but not one under flock/; one where it passed admits both."""
    _declared_by_the_throwaway_repo(monkeypatch)
    main, feat, _tip = _repos(tmp_path)
    (main / "tools" / "check" / "tool.py").write_text(DECL.format(", merge_requires={'agree': ['flock/']}"))
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "rule")
    _git(feat, "merge", "-q", "main")
    def run(status: str) -> int:
        monkeypatch.chdir(feat)
        monkeypatch.setenv("FAKE_CHECK", _RESULT % status)
        return research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool", "check",
                              "--cwd", "source", "--", *CHECKED])
    assert run("skipped") == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 0                                  # a.txt only: the rule does not apply
    (feat / "flock").mkdir()
    (feat / "flock" / "x.txt").write_text("x\n")
    _git(feat, "add", "-A")
    _git(feat, "commit", "-q", "-m", "flock change")
    assert run("skipped") == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 3
    assert "changes flock/, where `check` must have run agree" in capsys.readouterr().err
    assert run("passed") == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 0


def test_a_file_renamed_out_of_a_path_rule_requires_its_step(tmp_path):
    """git reports a rename by its destination alone unless asked not to, which would leave out the step the file's old path
    requires."""
    from research.merge import required_steps

    main, _feat, _tip = _repos(tmp_path)
    (main / "tools" / "check").mkdir(parents=True, exist_ok=True)
    (main / "tools" / "check" / "tool.py").write_text("from research.store.tool import Tool\n"
                                                      "CHECK = Tool(name='check', version='1', merge_requires={'agree': ['flock/']})\n")
    (main / "flock").mkdir()
    (main / "flock" / "x.txt").write_text("x\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "rule")
    base = _git(main, "rev-parse", "HEAD")
    _git(main, "mv", "flock/x.txt", "x.txt")
    _git(main, "commit", "-q", "-m", "move")
    assert required_steps(main, base, _git(main, "rev-parse", "HEAD"), "check") == {"agree": ["flock/"]}


def test_a_stacked_train_meets_the_steps_the_declaration_main_gained_from_the_train_ahead(tmp_path, monkeypatch, capsys):
    """A adds the rule that `agree` must pass for a change under flock/; C, checked on A, changes flock/. Once A lands, C merges
    tree-identically, and the steps are read from main's declaration (now A's) against the diff from main's tip (C's flock/
    change): an attempt of C that skipped `agree` is refused, one where it passed lands."""
    _declared_by_the_throwaway_repo(monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    _git(main, "checkout", "-q", "-b", "A")
    (main / "tools" / "check" / "tool.py").write_text(DECL.format(", merge_requires={'agree': ['flock/']}"))
    _git(main, "commit", "-q", "-am", "A: the rule")
    _git(main, "checkout", "-q", "-b", "C")
    (main / "flock").mkdir()
    (main / "flock" / "x.txt").write_text("x\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "C: a flock change, on A")
    c = _git(main, "rev-parse", "HEAD")
    _git(main, "checkout", "-q", "main")
    _git(main, "worktree", "add", "-q", str(tmp_path / "c"), c)

    def run(status: str) -> int:
        monkeypatch.chdir(tmp_path / "c")
        monkeypatch.setenv("FAKE_CHECK", _RESULT % status)
        return research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool", "check",
                              "--cwd", "source", "--", *CHECKED])

    assert run("skipped") == 0
    monkeypatch.chdir(main)
    assert merge_main(["C", "--dry-run"]) == 0                  # main's own declaration has no rule yet
    _git(main, "merge", "-q", "--no-ff", "-m", "A lands", "A")
    assert subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", c], cwd=main).returncode == 1    # main moved past C's base
    assert merge_main(["C", "--dry-run"]) == 3
    assert "changes flock/, where `check` must have run agree" in capsys.readouterr().err
    assert run("passed") == 0
    monkeypatch.chdir(main)
    assert merge_main(["C"]) == 0, capsys.readouterr().err
    assert _git(main, "rev-parse", "HEAD^{tree}") == _git(main, "rev-parse", f"{c}^{{tree}}") and _git(main, "rev-parse", "HEAD^2") == c


def test_a_path_rule_leaves_out_what_its_patterns_name():
    from research.merge import rule_touches

    rule = ["flock/", "!README.md", "!docs/*.md"]
    assert rule_touches(rule, ["flock/README.md", "flock/sub/README.md", "docs/x.md"]) == []
    assert rule_touches(rule, ["flock/README.md", "flock/a.rs"]) == ["flock/"]
    assert rule_touches(rule, ["flockish/a.rs", "other/README.md"]) == []
    assert rule_touches(["flock/"], ["flock/README.md"]) == ["flock/"]            # no pattern, no exception


def test_the_push_guard_refuses_a_main_commit_no_check_passed_and_allows_a_checked_merge(tmp_path, monkeypatch, capsys):
    """A push to main is the step a stray local merge reaches the remote through (old-circuits-and-proofs, 2 Oct): the hook
    `research merge` installs refuses every added first-parent commit whose tree no passing `check` attempt has."""
    import io

    from research.merge import ZERO, install_push_guard, unchecked_pushes
    from research.store.local import LocalStore

    main, feat, tip = _repos(tmp_path)
    base = _git(main, "rev-parse", "HEAD")
    assert _record(tmp_path, feat, monkeypatch, *CHECKED) == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 0 and "installed the push guard" in capsys.readouterr().err
    hook = Path(_git(main, "rev-parse", "--git-path", "hooks/pre-push"))
    assert (main / hook if not hook.is_absolute() else hook).exists() and install_push_guard(main) is None   # never clobbers
    merged = _git(main, "rev-parse", "HEAD")
    store = LocalStore()
    line = f"refs/heads/main {merged} refs/heads/main {base}"
    assert unchecked_pushes(main, [line], store) == []                            # its tree is feat's, which passed
    (main / "stray.txt").write_text("x\n")
    _git(main, "add", "stray.txt")
    _git(main, "commit", "-q", "-m", "a stray local commit")
    stray = _git(main, "rev-parse", "HEAD")
    assert unchecked_pushes(main, [f"refs/heads/main {stray} refs/heads/main {base}"], store) == [stray]
    assert unchecked_pushes(main, [f"refs/heads/other {stray} refs/heads/other {ZERO}"], store) == []   # only main is guarded
    monkeypatch.setattr("sys.stdin", io.StringIO(f"refs/heads/main {stray} refs/heads/main {base}\n"))
    assert merge_main(["--guard-push"]) == 3 and "refused: the push" in capsys.readouterr().err


def test_the_push_guard_allows_a_train_fast_forward_and_its_hook_ignores_other_branches(tmp_path, monkeypatch):
    """ci's case: a `--train` fast-forward lands the checked tip over prep merges nobody checked one by one; and infra's: an
    older worktree's push to a lane branch must pass without the guard starting at all."""
    from research.merge import ZERO, install_push_guard, unchecked_pushes
    from research.store.local import LocalStore

    main, feat, _ = _repos(tmp_path)
    base = _git(main, "rev-parse", "HEAD")
    for name in ("prep-a", "prep-b"):                                             # the candidate: base + two prep commits
        (feat / f"{name}.txt").write_text(name + "\n")
        _git(feat, "add", "-A")
        _git(feat, "commit", "-q", "-m", name)
    tip = _git(feat, "rev-parse", "HEAD")
    assert _record(tmp_path, feat, monkeypatch, *CHECKED) == 0                    # only the tip was checked
    assert unchecked_pushes(main, [f"refs/heads/main {tip} refs/heads/main {base}"], LocalStore()) == []
    hook = install_push_guard(main)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    lane = subprocess.run([str(hook)], input=f"refs/heads/lane {tip} refs/heads/lane {ZERO}\n", text=True, cwd=main, env=env,
                          capture_output=True)
    assert lane.returncode == 0 and lane.stderr == ""
    (main / "stray.txt").write_text("x\n")
    _git(main, "add", "stray.txt")
    _git(main, "commit", "-q", "-m", "stray")
    stray = _git(main, "rev-parse", "HEAD")
    pushed = subprocess.run([str(hook)], input=f"refs/heads/main {stray} refs/heads/main {base}\n", text=True, cwd=main,
                            env=env, capture_output=True)
    assert pushed.returncode == 3 and "refused: the push" in pushed.stderr


def test_a_flock_file_whose_name_git_quotes_still_requires_the_agreement(tmp_path):
    """`git diff --name-only` quotes a name with bytes outside ASCII (`"backends/flock/\\303\\251.py"`), which no prefix of a
    `merge_requires` rule matched, so a train changing such a file required no `lean-agreement`."""
    from research.merge import required_steps

    repo = tmp_path / "repo"
    (repo / "mytools").mkdir(parents=True)
    (repo / "mytools" / "agree.py").write_text("from research.store.tool import Tool\nT = Tool(name='check', version='1', "
                                               "merge_requires={'lean-agreement': ['backends/flock/', '!README.md']})\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "backends" / "flock").mkdir(parents=True)
    (repo / "backends" / "flock" / "é.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "é")
    assert required_steps(repo, base, _git(repo, "rev-parse", "HEAD"), "mytools.agree:T") == {"lean-agreement": ["backends/flock/"]}


def test_the_canary_passes_only_while_every_gate_refuses_the_bad_attempts_and_admits_the_control(tmp_path, monkeypatch, capsys):
    """The monitors' daily `research merge --canary`: under the target's own declaration, each bad attempt (no result, another
    kind or a refused one, a record that can't be read or disagrees with the index, another command, a failed verdict, a step
    listed twice) must be refused by the gate, the push guard and, where it judges the case, the train gate, and a run of the
    declared command with a passing result admitted by all three.  A gate that stops reading the record (main before this, as
    `refused` answering None), the command (`unlike`) or the train's result (its `gate`) fails it; so does a declaration that
    produces nothing.  It reads and writes nothing of the evidence store."""
    import research.merge as M
    from research.jobs import trains
    from research.store.local import LocalStore

    main, _feat, _tip = _repos(tmp_path)
    monkeypatch.chdir(main)
    assert merge_main(["--canary", "--json"]) == 0
    out, _ = capsys.readouterr()
    report = json.loads(out)
    assert report["schema"] == "research/merge-canary/v1" and report["produces"] == ["check/v1"] and report["ok"]
    assert report["command"] == list(CHECKED)
    R, A = "refused", "admitted"
    assert {c["case"]: (c["gate"].split(":")[0], c["push_guard"], c["train_gate"] and c["train_gate"].split(":")[0])
            for c in report["cases"]} == {
        "no-result": (R, R, R), "wrong-kind": (R, R, R), "refused-kind": (R, R, R), "unreadable-record": (R, R, R),
        "disagreeing-record": (R, R, R), "wrong-command": (R, R, None), "failed-verdict": (R, R, None), "repeated-step": (R, R, R),
        "control": (A, A, A)}
    why = {c["case"]: (c["gate"], c["train_gate"]) for c in report["cases"]}
    assert "its attempt can't be read" in why["unreadable-record"][0]
    assert "its record says rc 1" in why["disagreeing-record"][0] and "is done rc=1" in why["disagreeing-record"][1]
    assert "refused: its result is a lean-fast/v1, which no merge accepts" in why["refused-kind"][0]
    assert "isn't the target's declared" in why["wrong-command"][0]
    assert "refused: its result's verdict is failed, not passed" in why["failed-verdict"][0]
    assert all("result lists step suites more than once" in w for w in why["repeated-step"]), "the two gates read it the same"
    assert not LocalStore().index.sql("SELECT id FROM attempts")
    assert _git(main, "status", "--porcelain") == "" and "pre-push" not in os.listdir(main / ".git" / "hooks")

    with monkeypatch.context() as m:
        m.setattr(M, "refused", lambda *a, **k: None)
        assert merge_main(["--canary"]) == 1
        out, err = capsys.readouterr()
        assert out.count("the gate admitted it, the push guard admitted it") == 9 and "canary FAILED" in err
    with monkeypatch.context() as m:
        m.setattr(M, "unlike", lambda *a: None)
        assert merge_main(["--canary"]) == 1
        assert ("wrong-command (a passing result of `bash -c '<the declared command> || true'`): the gate admitted it, the push "
                "guard admitted it, the train gate doesn't judge it") in capsys.readouterr().out
    with monkeypatch.context() as m:
        m.setattr(trains.TrainService, "gate", lambda self, tr: None)
        assert merge_main(["--canary"]) == 1
        assert capsys.readouterr().out.count("the train gate admitted it") == 7
    with monkeypatch.context() as m:
        m.setattr(M, "produced", lambda *a: frozenset())
        assert merge_main(["--canary"]) == 1
        assert ("control (a run of the declared command whose passing result is of a kind the declaration produces): the gate "
                "refused it") in capsys.readouterr().out

    monkeypatch.chdir(tmp_path)
    assert merge_main(["--canary"]) == 3 and "no tools/check/tool.py, its declaration of `check`" in capsys.readouterr().err
    for argv in (["--canary", "feat"], ["--canary", "--sweep"], ["--json", "feat"]):
        try:
            merge_main(argv)
        except SystemExit as e:
            assert e.code == 2, argv
        else:
            raise AssertionError(f"{argv} is a usage error")
