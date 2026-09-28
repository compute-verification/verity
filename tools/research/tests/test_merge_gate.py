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
#: a workload that passes as `check` does, with a `check/v1` result: the gate admits no attempt without a result of a kind the
#: target's declaration produces
CHECKED = (sys.executable, "-c", _RESULT % "passed")
#: the throwaway target's declaration of `check`, which the gate reads `produces` and `merge_requires` from
DECL = "from research.store.tool import Tool\nCHECK = Tool(name='check', version='1', produces=['check/v1']{})\n"


def _repos(tmp_path: Path) -> tuple[Path, Path, str]:
    """(the main checkout, a worktree of branch `feat`, feat's tip)."""
    main = tmp_path / "main"
    (main / "mytools").mkdir(parents=True)
    (main / "mytools" / "decl_check.py").write_text("from research.store.tool import Tool\nT = Tool(name='check', version='1')\n")
    (main / "tools" / "check").mkdir(parents=True)
    (main / "tools" / "check" / "tool.py").write_text(DECL.format(""))
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


def test_merge_needs_a_passing_check_of_the_exact_commit(tmp_path, monkeypatch, capsys):
    main, feat, tip = _repos(tmp_path)
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 1 and "no passing `check` attempt" in capsys.readouterr().err
    assert _record(tmp_path, feat, monkeypatch, "false") == 1                     # a failed check is an attempt, never a pass
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 1 and "failed rc=1" in capsys.readouterr().err
    (feat / "a.txt").write_text("dirty\n")
    assert _record(tmp_path, feat, monkeypatch, *CHECKED) == 0                    # passed, but on an uncommitted tree
    _git(feat, "checkout", "--", "a.txt")
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 1 and "dirty" in capsys.readouterr().err
    assert _record(tmp_path, feat, monkeypatch, "true") == 0                      # passed, with no result saying what ran
    monkeypatch.chdir(main)
    assert merge_main(["feat"]) == 1 and "refused: it published no result" in capsys.readouterr().err
    assert _record(tmp_path, feat, monkeypatch, *CHECKED) == 0
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
    assert merge_main(["feat"]) == 1 and "main has moved past" in capsys.readouterr().err
    _git(main, "checkout", "-q", "-b", "other")
    assert merge_main(["feat", "--dry-run"]) == 1 and "not main" in capsys.readouterr().err


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
    assert merge_main(["B"]) == 1 and "main has moved past" in capsys.readouterr().err


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
    check = shlex.join([sys.executable, "-c", "import os, sys; os.path.exists('BAD') and sys.exit(1); " + _RESULT % "passed"])
    cmd = ["--tool", "mytools.decl_check:T", "--command", check]
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
    assert merge_main(["--train", "--prepare", "train-prep/x", f"#7@{a[:9]}", *lint]) == 1
    assert "#7 has moved: its head on origin is" in capsys.readouterr().err
    assert merge_main(["--train", "--prepare", "train-prep/x", f"BAD@{_git(main, 'rev-parse', 'BAD')}", *lint]) == 1
    assert "failed on the train" in capsys.readouterr().err
    _git(main, "checkout", "-q", "-b", "C", "main")
    (main / "b.txt").write_text("C's b\n")
    _git(main, "add", "b.txt")
    _git(main, "commit", "-q", "-m", "C")
    _git(main, "checkout", "-q", "main")
    assert merge_main(["--train", "--prepare", "train-prep/x", "B", "C", *lint]) == 1
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
    assert merge_main(["--train", "--prepare", "train-prep/j", *pins, "--lint", "true"]) == 1
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
    assert merge_main(["--train", "--prepare", "train-prep/k", "11@" + heads[11][:9], "12@" + heads[12][:9], "--lint", "true"]) == 1
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
    assert merge_main(["--train", "--prepare", "train-prep/g", *pins, *lint]) == 1
    err = capsys.readouterr().err
    assert f"#20 needs the statement-reviewer grant at {heads[20][:12]} (tools/check/queue.toml: a named reviewer read" in err
    assert f"`research data label pr:20@{heads[20]} grant statement-reviewer --by REVIEWER`" in err and "#21" not in err
    assert merge_main(["--train", "--prepare", "train-prep/g", f"side@{heads[20][:9]}", *lint]) == 1
    assert "side needs the statement-reviewer grant (tools/check/queue.toml), which only a pull request's pinned head can hold" \
        in capsys.readouterr().err
    assert merge_main(["--train", "pr20", "--tool", "mytools.decl_check:T", "--command", "true"]) == 1
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
    assert merge_main(["--train", "--prepare", "train-prep/n", f"23@{heads[23][:9]}", *lint]) == 1
    err = capsys.readouterr().err
    assert f"#23 needs the statement-reviewer grant at {heads[23][:12]}" in err
    assert f"`research data label pr:23@{heads[23]} lean_review ART --by LANE`" in err
    review(23, moved, False)                         # the move's own report: it never saw the record #23's head holds
    assert merge_main(["--train", "--prepare", "train-prep/n", f"23@{heads[23][:9]}", *lint]) == 1
    assert "#23 needs the statement-reviewer grant" in capsys.readouterr().err


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
        monkeypatch.chdir(feat)
        return research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool",
                              "mytools.decl_check:T", "--cwd", "source", "--", sys.executable, "-c",
                              (_RESULT % "passed").replace("check/v1", kind)])

    assert "lean-fast/v1" in REFUSED_KINDS and run("lean-fast/v1") == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 1
    assert "refused: its result is a lean-fast/v1, which no merge accepts" in capsys.readouterr().err
    assert unchecked_pushes(main, [f"refs/heads/main {tip} refs/heads/main {base}"], LocalStore()) == [tip]
    assert run("check/v1") == 0
    store = LocalStore()
    results = [store.get_attempt(r[0])["outputs"]["result"] for r in store.index.sql("SELECT id FROM attempts WHERE tool = 'check'")]
    art = next(a for a in results if store.get_manifest(a).kind == "check/v1")
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 0
    store.manifest_path(ids.hex_of(art)).unlink()
    assert merge_main(["feat", "--dry-run"]) == 1
    assert "refused: its result can't be read (NotFound" in capsys.readouterr().err


def test_merge_admits_only_a_result_of_a_kind_the_targets_declaration_produces(tmp_path, monkeypatch, capsys):
    """Refusing the kinds it knew to refuse let through a `check` attempt with no result (`research run --tool check -- python3
    tools/lean/audit.py --no-replay PKG` writes none) and a result of any other kind (red team, #966).  The gate admits only a
    kind the target's own declaration produces (`check/v1` for check), and nothing when the target declares none."""
    from research.merge import produced

    main, feat, _tip = _repos(tmp_path)
    assert _record(tmp_path, feat, monkeypatch, "true") == 0
    assert _record(tmp_path, feat, monkeypatch, sys.executable, "-c", (_RESULT % "passed").replace("check/v1", "bench-result/v1")) == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "refused: it published no result, and the target's declaration of the tool produces check/v1" in err
    assert "refused: its result is a bench-result/v1, and the target's declaration of the tool produces check/v1" in err
    (main / "tools" / "check" / "tool.py").write_text(DECL.replace(", produces=['check/v1']", "").format(""))
    _git(main, "commit", "-q", "-am", "check produces nothing")
    _git(feat, "merge", "-q", "main")
    assert _record(tmp_path, feat, monkeypatch, *CHECKED) == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 1
    assert "refused: its result is a check/v1, and the target's declaration of the tool produces nothing" in capsys.readouterr().err
    assert produced(tmp_path, "check") == frozenset(), "no declaration at all"


def test_merge_refuses_an_attempt_the_index_lists_whose_document_is_gone(tmp_path, monkeypatch, capsys):
    """The index answers `passing`'s query, but the attempt's document holds its result: with the document gone from the store
    and no remote to read it from, `get_attempt` returns None, and the gate can't tell what the result was (red team, #966)."""
    from research.store.local import LocalStore

    main, feat, _tip = _repos(tmp_path)
    monkeypatch.chdir(feat)
    assert research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool",
                          "mytools.decl_check:T", "--cwd", "source", "--", sys.executable, "-c",
                          (_RESULT % "passed").replace("check/v1", "lean-fast/v1")]) == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 1 and "which no merge accepts" in capsys.readouterr().err
    store = LocalStore()
    (rid,) = [r[0] for r in store.index.sql("SELECT id FROM attempts WHERE tool = 'check'")]
    store.attempt_path(rid).unlink()
    assert merge_main(["feat", "--dry-run"]) == 1
    assert f"{rid} done rc=0 refused: its attempt can't be read" in capsys.readouterr().err


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
        return research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool", "check",
                              "--cwd", "source", "--", sys.executable, "-c", _RESULT % status])
    assert run("skipped") == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 0                                  # a.txt only: the rule does not apply
    (feat / "flock").mkdir()
    (feat / "flock" / "x.txt").write_text("x\n")
    _git(feat, "add", "-A")
    _git(feat, "commit", "-q", "-m", "flock change")
    assert run("skipped") == 0
    monkeypatch.chdir(main)
    assert merge_main(["feat", "--dry-run"]) == 1
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
        return research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--project", "t", "--no-sampler", "--tool", "check",
                              "--cwd", "source", "--", sys.executable, "-c", _RESULT % status])

    assert run("skipped") == 0
    monkeypatch.chdir(main)
    assert merge_main(["C", "--dry-run"]) == 0                  # main's own declaration has no rule yet
    _git(main, "merge", "-q", "--no-ff", "-m", "A lands", "A")
    assert subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", c], cwd=main).returncode == 1    # main moved past C's base
    assert merge_main(["C", "--dry-run"]) == 1
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
    assert merge_main(["--guard-push"]) == 1 and "refused push" in capsys.readouterr().err


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
    assert pushed.returncode == 1 and "refused push" in pushed.stderr
