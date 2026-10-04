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
    """Two PRs pinning a record of the same name in one `lean-audit.json`, a few lines apart: git merges the lines cleanly into
    an object with the key twice, which `json.loads` would read as the last one silently."""
    main, _feat, _tip = _repos(tmp_path)
    audit = '{\n  "pins": {\n    "a": 1,\n    "m": 5,\n    "z": 9\n  }\n}\n'
    origin, heads = _origin_with(tmp_path, main, {"lean-audit.json": audit}, {
        11: {"lean-audit.json": audit.replace('"a": 1,\n', '"a": 1,\n    "b": 2,\n')},
        12: {"lean-audit.json": audit.replace('"m": 5,\n', '"m": 5,\n    "b": 3,\n')}})
    monkeypatch.chdir(main)
    assert merge_main(["--train", "--prepare", "train-prep/k", "11@" + heads[11][:9], "12@" + heads[12][:9], "--lint", "true"]) == 3
    assert "merging #12 onto the train leaves lean-audit.json unparseable as JSON" in capsys.readouterr().err
    assert _git(origin, "for-each-ref", "refs/heads/train-prep/k") == ""


def _store_remote(tmp_path: Path, monkeypatch) -> None:
    """A store remote (a directory): the gate reads grants and reviews only after pulling the target's labels from one."""
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))


def test_a_pin_change_goes_into_a_train_only_with_the_statement_reviewer_grant_at_its_head(tmp_path, monkeypatch, capsys):
    """old-circuits-and-proofs held #828 and #858-#866 out of trains by hand on 2 Oct: each changed a pinned record with no
    statement reviewer named.  The queue's rule for it refuses such a ref before its train is built (`--prepare`) or checked
    (`--train`), until the `grant` label is on the pull request's pinned head; a change to another section needs none."""
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    audit = {"roots": ["P"], "pins": {"P.t": {"type_hash": "1"}}, "reads": {}}
    _origin, heads = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "statement-reviewer"\nlean_audit = ["pins", "reads"]\n',
        "lean-audit.json": json.dumps(audit, indent=1)}, {
        20: {"lean-audit.json": json.dumps({**audit, "pins": {"P.t": {"type_hash": "2"}}}, indent=1)},
        21: {"lean-audit.json": json.dumps({**audit, "roots": ["P", "Q"]}, indent=1)}})
    _git(main, "push", "-q", "origin", "pr20:refs/heads/side")
    monkeypatch.chdir(main)
    pins, lint = [f"20@{heads[20][:9]}", f"21@{heads[21][:9]}"], ["--lint", "true"]
    assert merge_main(["--train", "--prepare", "train-prep/g", *pins, *lint]) == 3
    err = capsys.readouterr().err
    assert f"#20 needs the statement-reviewer grant at {heads[20][:12]} (tools/check/queue.toml: a named reviewer read" in err
    assert f"`research data label pr:20@{heads[20]} grant statement-reviewer --by REVIEWER`" in err and "#21" not in err
    assert merge_main(["--train", "--prepare", "train-prep/g", f"side@{heads[20][:9]}", *lint]) == 3
    assert "side needs the statement-reviewer grant (tools/check/queue.toml), which only a pull request's pinned head can hold" \
        in capsys.readouterr().err
    assert merge_main(["--train", "pr20", "--tool", "mytools.decl_check:T", "--command", shlex.join(CHECKED)]) == 3
    out, err = capsys.readouterr()
    assert "pr20 needs the statement-reviewer grant" in err and "nothing was checked" in err and "recorded as" not in out
    LocalStore().add_label(f"pr:20@{heads[20]}", "grant", "statement-reviewer", "red-team")
    assert merge_main(["--train", "--prepare", "train-prep/g", *pins, *lint]) == 0, capsys.readouterr().err


def test_a_move_audit_py_prints_as_the_same_statements_needs_no_grant_and_a_changed_statement_still_does(tmp_path, monkeypatch, capsys):
    """Daniel's ruling (#736, AGENTS.md "Lean"): a guarantee that only moved needs no statement reviewer when `audit.py --update
    --moved` prints it as the same statement under its old names, though the record's hashes change with the names.  The train
    takes that verdict from the run's audit.json, which a `lean_review` label on the pinned head names: a move it printed whole as
    the same statements goes in without the grant; a moved record whose statement also changed refuses, and so does the move's
    report on a head whose record changed after it ran."""
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    base = {"roots": ["P"], "pins": {"P.t": {"type_hash": "1"}}, "reads": {"P.Defs": {"hash": "d"}}}
    moved = {**base, "pins": {"Q.t": {"type_hash": "1q"}}, "reads": {"Q.Defs": {"hash": "dq"}}}
    restated = {**moved, "pins": {"Q.t": {"type_hash": "2q"}}}
    (main / "lean").mkdir()
    _origin, heads = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "statement-reviewer"\nlean_audit = ["pins", "reads"]\n',
        "lean/lean-audit.json": json.dumps(base, indent=1)}, {
        22: {"lean/lean-audit.json": json.dumps(moved, indent=1)},
        23: {"lean/lean-audit.json": json.dumps(restated, indent=1)}})
    store = LocalStore()

    def review(n: int, after: dict, changes: bool) -> None:
        f = tmp_path / f"audit-{n}-{changes}.json"
        f.write_text(json.dumps({"passed": True, "failed": [], "packages": [{
            "package": "/w/verity/lean", "review": [], "statement_changes": changes,
            "before": {k: base[k] for k in ("pins", "reads")}, "pins": after["pins"], "reads": after["reads"]}]}))
        store.add_label(f"pr:{n}@{heads[n]}", "lean_review", store.put_artifact("lean-review/v1", {}, file=f), "lean")

    monkeypatch.chdir(main)
    lint = ["--lint", "true"]
    review(22, moved, False)
    assert merge_main(["--train", "--prepare", "train-prep/m", f"22@{heads[22][:9]}", *lint]) == 0, capsys.readouterr().err
    review(23, restated, True)                       # it printed `pin Q.t: changed (moved from P.t)` for a reviewer
    assert merge_main(["--train", "--prepare", "train-prep/n", f"23@{heads[23][:9]}", *lint]) == 3
    err = capsys.readouterr().err
    assert f"#23 needs the statement-reviewer grant at {heads[23][:12]}" in err
    assert f"`research data label pr:23@{heads[23]} lean_review ART --by LANE`" in err
    review(23, moved, False)                         # the move's own report: it never saw the record #23's head holds
    assert merge_main(["--train", "--prepare", "train-prep/n", f"23@{heads[23][:9]}", *lint]) == 3
    assert "#23 needs the statement-reviewer grant" in capsys.readouterr().err


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


def test_a_statement_review_carries_into_a_train_across_a_restack_or_a_merge_of_main_and_not_to_a_restated_pin(
        tmp_path, monkeypatch, capsys):
    """Daniel's ruling (2 Oct), at the train's gate: the grant on the head that was reviewed counts at a head that only restacks it
    or merges `main` into it, and the train says which head it carried from; a head whose pin says something else needs a new one."""
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    audit = {"roots": ["P"], "pins": {"P.t": {"type_hash": "1"}}, "reads": {}}
    pinned = json.dumps({**audit, "pins": {"P.t": {"type_hash": "2"}}}, indent=1)
    origin, heads = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "statement-reviewer"\nlean_audit = ["pins", "reads"]\n',
        "lean-audit.json": json.dumps(audit, indent=1)}, {20: {"lean-audit.json": pinned}})
    old = heads[20]
    LocalStore().add_label(f"pr:20@{old}", "grant", "statement-reviewer", "reviewer")
    _main_moves(main, origin)
    restacked = _files_branch(main, "pr20-restacked", {"lean-audit.json": pinned})
    merged = _main_merged_into(main, "pr20")
    restated = _files_branch(main, "pr20-restated", {"lean-audit.json": json.dumps({**audit, "pins": {"P.t": {"type_hash": "3"}}})})
    monkeypatch.chdir(main)
    for head, how in ((restacked, "restack"), (merged, "merge of main")):
        _git(main, "push", "-q", "-f", "origin", f"{head}:refs/pull/20/head")
        assert merge_main(["--train", "--prepare", f"train-prep/{how[0]}", f"20@{head[:9]}", "--lint", "true"]) == 0, capsys.readouterr()
        assert (f"research merge: grant statement-reviewer carried from pr:20@{old} to {head[:12]} "
                f"({how}; lean-audit.json change identical)") in capsys.readouterr().out
    _git(main, "push", "-q", "-f", "origin", f"{restated}:refs/pull/20/head")
    assert merge_main(["--train", "--prepare", "train-prep/x", f"20@{restated[:9]}", "--lint", "true"]) == 3
    out, err = capsys.readouterr()
    assert f"#20 needs the statement-reviewer grant at {restated[:12]}" in err and "nothing was built" in err
    assert (f"research merge: grant statement-reviewer on pr:20@{old} doesn't carry to {restated[:12]}: its lean-audit.json change "
            f"isn't byte-identical (lean-audit.json)") in out
    assert _git(origin, "for-each-ref", "--format=%(refname)", "refs/heads/train-prep") == \
        "refs/heads/train-prep/m\nrefs/heads/train-prep/r"


def test_a_lean_review_carries_under_the_same_rule(tmp_path, monkeypatch, capsys):
    """A `lean_review` on the head `audit.py --update --moved` ran on counts at a restack of it; at a head that also changes another
    file it doesn't, though the report would still cover its lean-audit.json change: the restack rule binds before `reviewed`."""
    from research.store.local import LocalStore

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    base = {"roots": ["P"], "pins": {"P.t": {"type_hash": "1"}}, "reads": {"P.Defs": {"hash": "d"}}}
    moved = {**base, "pins": {"Q.t": {"type_hash": "1q"}}, "reads": {"Q.Defs": {"hash": "dq"}}}
    (main / "lean").mkdir()
    origin, heads = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "statement-reviewer"\nlean_audit = ["pins", "reads"]\n',
        "lean/lean-audit.json": json.dumps(base, indent=1)}, {22: {"lean/lean-audit.json": json.dumps(moved, indent=1)}})
    old, store = heads[22], LocalStore()
    report = tmp_path / "audit.json"
    report.write_text(json.dumps({"passed": True, "failed": [], "packages": [{
        "package": "/w/verity/lean", "review": [], "statement_changes": False,
        "before": {k: base[k] for k in ("pins", "reads")}, "pins": moved["pins"], "reads": moved["reads"]}]}))
    art = store.put_artifact("lean-review/v1", {}, file=report)
    store.add_label(f"pr:22@{old}", "lean_review", art, "lean")
    _main_moves(main, origin)
    restacked = _files_branch(main, "pr22-restacked", {"lean/lean-audit.json": json.dumps(moved, indent=1)})
    more = _files_branch(main, "pr22-more", {"lean/lean-audit.json": json.dumps(moved, indent=1), "x.txt": "x\n"})
    monkeypatch.chdir(main)
    _git(main, "push", "-q", "-f", "origin", f"{restacked}:refs/pull/22/head")
    assert merge_main(["--train", "--prepare", "train-prep/r", f"22@{restacked[:9]}", "--lint", "true"]) == 0, capsys.readouterr()
    assert (f"research merge: lean_review {art} carried from pr:22@{old} to {restacked[:12]} (restack; lean-audit.json change "
            f"identical)") in capsys.readouterr().out
    _git(main, "push", "-q", "-f", "origin", f"{more}:refs/pull/22/head")
    assert merge_main(["--train", "--prepare", "train-prep/x", f"22@{more[:9]}", "--lint", "true"]) == 3
    out, err = capsys.readouterr()
    assert f"#22 needs the statement-reviewer grant at {more[:12]}" in err
    main_tip = _git(main, "rev-parse", "main")
    assert (f"research merge: lean_review {art} on pr:22@{old} doesn't carry to {more[:12]}: its change against {main_tip[:12]} "
            f"differs (x.txt)") in out


def test_a_head_with_two_merge_bases_carries_no_grant_and_needs_every_one(tmp_path, monkeypatch, capsys):
    """Red team of #955: a head merging #30's commit (which `main` merged too) into the reviewed head while keeping the reviewed
    tree drops #30's pins.  Its merge bases with `main` are both the reviewed head's base and #30's commit; `git merge-base`
    printed the first (the newer), against which the head changes what the reviewed one did, so the grant carried and the train
    reverted #30's pins.  `git merge` diffs against all of them: neither the grant nor the rules may read one."""
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
        "tools/check/queue.toml": '[[grant]]\nrole = "statement-reviewer"\nlean_audit = ["pins", "reads"]\n',
        "lean-audit.json": json.dumps(audit, indent=1),
        "other/lean-audit.json": json.dumps({**audit, "pins": {"O.t": {"type_hash": "old"}}}, indent=1)}, {})
    next_day()
    theirs = _files_branch(main, "pr30", {"other/lean-audit.json": json.dumps({**audit, "pins": {"O.t": {"type_hash": "new"}}})})
    next_day()
    _main_moves(main, origin)
    next_day()
    old = _files_branch(main, "pr20", {"lean-audit.json": json.dumps({**audit, "pins": {"P.t": {"type_hash": "2"}}}, indent=1)})
    LocalStore().add_label(f"pr:20@{old}", "grant", "statement-reviewer", "reviewer")
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
    assert (f"research merge: {new[:12]} has 2 merge bases with {tip[:12]} ({bases}), so no one diff says what merging it changes: "
            f"it needs every grant the rules name until {tip[:12]} is merged into it again") in out
    assert (f"research merge: grant statement-reviewer on pr:20@{old} doesn't carry to {new[:12]}: {new[:12]} has 2 merge bases "
            f"with {tip[:12]} ({bases}), so no one diff says what merging it changes; merging {tip[:12]} into it again gives it "
            f"one") in out
    assert f"#20 needs the statement-reviewer grant at {new[:12]}" in err and "nothing was built" in err
    assert _git(origin, "for-each-ref", "refs/heads/train-prep") == ""


def test_a_train_the_store_cant_answer_for_is_refused_as_stale_not_as_the_rules(tmp_path, monkeypatch, capsys):
    from research.store import labels_sync

    _store_remote(tmp_path, monkeypatch)
    main, _feat, _tip = _repos(tmp_path)
    audit = {"roots": ["P"], "pins": {"P.t": {"type_hash": "1"}}, "reads": {}}
    _origin, heads = _origin_with(tmp_path, main, {
        "tools/check/queue.toml": '[[grant]]\nrole = "statement-reviewer"\nlean_audit = ["pins", "reads"]\n',
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
