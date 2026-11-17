"""What `merge` reads of a recorded check: an attempt passes only as a passing `check` attempt of that exact commit in the store,
recorded by a real `research run --tool` in a throwaway repository (`passing`, `refused`); the steps a change requires
(`required_steps`); and the lock records a check regenerated, which vouch for the commit that lands them
(`regenerated_parent`)."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from research.cli import main as research_main


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
#: the command the throwaway target declares for `check`: `passing` admits only a run of it, with a passing result of a kind the
#: declaration produces
CHECKED = (sys.executable, "tools/check/check.py")
#: the throwaway target's declaration of `check`, which `merge` reads `command`, `produces` and `merge_requires` from
DECL = ("from research.store.tool import Tool\nCHECK = Tool(name='check', version='1', "
        f"command={list(CHECKED)!r}, produces=['check/v1']{{}})\n")
#: what a throwaway target holds of its check: the declaration and the script it declares
CHECK_FILES = {"tools/check/tool.py": DECL.format(""), "tools/check/check.py": CHECK_PY}


def _repos(tmp_path: Path) -> tuple[Path, Path, str]:
    """(the main checkout, a worktree of branch `feat`, feat's tip)."""
    main = tmp_path / "main"
    (main / "mytools").mkdir(parents=True)
    (main / "mytools" / "decl_check.py").write_text("from research.store.tool import Tool\nT = Tool(name='check', version='1')\n")
    (main / "tools" / "check").mkdir(parents=True, exist_ok=True)
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


def _passing(main: Path, commit: str) -> tuple[list[str], dict[str, str]]:
    """(the attempts `passing` admits for ``commit`` under ``main``'s declaration of `check`, {each other ended clean attempt of
    it with rc 0: why `refused` refuses it})."""
    from research.merge import declared_command, passing, produced, refused
    from research.store.local import LocalStore

    store, kinds, command = LocalStore(), produced(main, "check"), declared_command(main, "check")
    rows = store.index.sql("SELECT id FROM attempts WHERE tool = 'check' AND source_commit = ? AND source_dirty = 0 "
                           "AND state = 'done' AND rc = 0", (commit,))
    why = {r[0]: refused(store, r[0], kinds, "check", commit, command, main) for r in rows}
    return passing(store, commit, "check", kinds=kinds, command=command, repo=main), {k: v for k, v in why.items() if v}


def test_a_check_passes_only_as_a_passing_attempt_of_the_exact_clean_commit(tmp_path, monkeypatch):
    main, feat, tip = _repos(tmp_path)
    assert _passing(main, tip) == ([], {})
    assert _record(tmp_path, feat, monkeypatch, "false") == 1                     # a failed check is an attempt, never a pass
    (feat / "a.txt").write_text("dirty\n")
    assert _checked(tmp_path, feat, monkeypatch) == 0                              # passed, but on an uncommitted tree
    _git(feat, "checkout", "--", "a.txt")
    assert _passing(main, tip) == ([], {})
    assert _checked(tmp_path, feat, monkeypatch, "pass") == 0                      # passed, with no result saying what ran
    ids, why = _passing(main, tip)
    assert ids == [] and [w.split(",")[0] for w in why.values()] == ["it published no result"]
    assert _checked(tmp_path, feat, monkeypatch) == 0
    ids, why = _passing(main, tip)
    assert len(ids) == 1 and ids[0] not in why


def test_an_attempt_whose_result_is_the_lean_fast_paths_never_passes_whatever_tool_recorded_it(tmp_path, monkeypatch):
    """No landing accepts the Lean fast path's result (`lean_changed.py --records`, no kernel replay).
    An attempt of `check` that passed on the exact commit is refused when its result is of a kind in `REFUSED_KINDS`, or can't
    be read. A `check/v1` result of the same commit passes."""
    from research.merge import REFUSED_KINDS
    from research.store import ids
    from research.store.local import LocalStore

    main, feat, tip = _repos(tmp_path)

    def run(kind: str) -> int:
        return _checked(tmp_path, feat, monkeypatch, (_RESULT % "passed").replace("check/v1", kind))

    assert "lean-fast/v1" in REFUSED_KINDS and run("lean-fast/v1") == 0
    assert _passing(main, tip)[0] == [] and "its result is a lean-fast/v1, which no merge accepts" in _passing(main, tip)[1].values()
    assert run("check/v1") == 0
    store = LocalStore()
    results = [store.get_attempt(r[0])["outputs"]["result"] for r in store.index.sql("SELECT id FROM attempts WHERE tool = 'check'")]
    art = next(a for a in results if store.get_manifest(a).kind == "check/v1")
    assert len(_passing(main, tip)[0]) == 1
    store.manifest_path(ids.hex_of(art)).unlink()
    passed, why = _passing(main, tip)
    assert passed == [] and any(w.startswith("its result can't be read (NotFound") for w in why.values())


def test_an_attempt_passes_only_with_a_result_of_a_kind_the_targets_declaration_produces(tmp_path, monkeypatch):
    """Refusing the kinds it knew to refuse let through a `check` attempt with no result (`research run --tool check -- python3
    tools/verity/lean/check.py --no-replay PKG` writes none) and a result of any other kind.  An attempt passes only with a
    kind the target's own declaration produces (`check/v1` for check), and none when the target declares none."""
    from research.merge import produced

    main, feat, tip = _repos(tmp_path)
    assert _checked(tmp_path, feat, monkeypatch, "pass") == 0
    assert _checked(tmp_path, feat, monkeypatch, (_RESULT % "passed").replace("check/v1", "bench-result/v1")) == 0
    ids, why = _passing(main, tip)
    assert ids == [] and sorted(why.values()) == [
        "it published no result, and the target's declaration of the tool produces check/v1",
        "its result is a bench-result/v1, and the target's declaration of the tool produces check/v1"]
    (main / "tools" / "check" / "tool.py").write_text(DECL.replace(", produces=['check/v1']", "").format(""))
    _git(main, "commit", "-q", "-am", "check produces nothing")
    _git(feat, "merge", "-q", "main")
    assert _checked(tmp_path, feat, monkeypatch) == 0
    ids, why = _passing(main, _git(feat, "rev-parse", "HEAD"))
    assert ids == [] and list(why.values()) == ["its result is a check/v1, and the target's declaration of the tool produces nothing"]
    assert produced(tmp_path, "check") == frozenset(), "no declaration at all"


#: a "check" that runs no check: one line of shell writing what check.py writes, with the verdict STATUS
_FORGED = ("printf '%s' '{\"schema\": \"research/result/v0.1\", \"kind\": \"check/v1\", \"run_id\": \"'\"$RESEARCH_RUN_ID\"'\", "
           "\"validation\": {\"status\": \"STATUS\", \"checker\": \"sh\"}, \"steps\": []}' > \"$RESEARCH_RUN_DIR/result.json\"")


def test_an_attempt_passes_only_as_a_run_of_the_declared_command_whose_own_result_passed_once(tmp_path, monkeypatch):
    """A check that runs nothing: `research run --tool check -- sh -c '<write a check/v1
    result.json>'` would pass, with its verdict failed too, if neither the attempt's argv nor the result's verdict were read.
    An attempt passes only as a run of the command the target declares (`Tool.command`) whose own result passed, and lists
    no step twice (`step_status` reads the first)."""
    main, feat, tip = _repos(tmp_path)
    for status in ("passed", "failed"):
        assert _record(tmp_path, feat, monkeypatch, "sh", "-c", _FORGED.replace("STATUS", status)) == 0
    ok = _RESULT % "passed"
    assert _checked(tmp_path, feat, monkeypatch, ok.replace("{'status': 'passed'", "{'status': 'failed'")) == 0
    twice = "[{'step': 'agree', 'status': 'skipped'}, {'step': 'agree', 'status': 'passed'}]"
    assert _checked(tmp_path, feat, monkeypatch, ok.replace("[{'step': 'agree', 'status': 'passed'}]", twice)) == 0
    ids, why = _passing(main, tip)
    assert ids == [] and len(why) == 4
    assert sum(f"isn't the target's declared `{shlex.join(CHECKED)}`" in w for w in why.values()) == 2
    assert {"its result's verdict is failed, not passed", "its result lists step agree more than once"} <= set(why.values())
    assert _checked(tmp_path, feat, monkeypatch) == 0
    assert len(_passing(main, tip)[0]) == 1


def test_a_run_of_the_declared_command_is_it_in_a_check_slot_or_not_with_check_pys_own_flags_at_most():
    """What `check.py --record [--on]`, `train.sh` and the jobs dispatch record: the declared command,
    behind `slot.py` when it was sent to a machine, followed by check.py's own flags.  Anything else is another command."""
    from research.merge import unlike

    cmd = ("uv", "run", "--locked", "python", "tools/check/check.py")
    slot = ["python3", "tools/check/slot.py"]
    for argv in ([*cmd], [*cmd, "--keep-going"], [*cmd, "--no-cache", "--cores", "8"], [*cmd, "--keep-going", "--verdicts-in", "v.tgz"],
                 [*cmd, "--keep-going", "--no-cache", "--circuit-cache", "--verdicts-in", "v.tgz"],
                 [*cmd, "--verdicts-in", "v.tgz", "--cores", "4"], [*slot, "--", *cmd], [*slot, "--by", "x", "--", *cmd],
                 [*slot, "--priority", "--by", "gates", "--", *cmd, "--keep-going"], [*slot, "--twin", "p@vy-nebius-1", "--", *cmd],
                 [*slot, "--priority", "--by", "gates", "--twin", "p", "--", *cmd, "--keep-going"],
                 [*slot, "--priority", "--rank", "39", "--by", "ci", "--", *cmd, "--keep-going", "--verdicts-in", "v.tgz"],
                 [*slot, "--by", "ci", "--wait-max", "60", "--priority", "--run-max", "9", "--", *cmd]):
        assert unlike(argv, cmd) is None, argv
    for argv in ([*slot, "--by", "a", "--by", "b", "--", *cmd], [*slot, "--priority", "--priority", "--", *cmd],
                 [*slot, "--cores", "8", "--", *cmd], [*slot, "--rank", "--", *cmd],
                 [*slot, "--budget-gb", "64", "--", *cmd]):         # how many of its steps CMD starts at once: another run
        assert "isn't the target's declared" in (unlike(argv, cmd) or ""), argv
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
    a passing run of the declared command whose result is one is refused for its kind alone.  A result that
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


def test_an_attempt_whose_record_cant_be_read_or_disagrees_with_the_index_never_passes(tmp_path, monkeypatch):
    """The index answers `passing`'s query, but the attempt's record is what says what ran: with it gone from the store (and no
    remote to read it from), unreadable, or saying something other than the index row did, nothing can tell what passed."""
    from research.store.local import LocalStore

    main, feat, tip = _repos(tmp_path)
    assert _record(tmp_path, feat, monkeypatch, *CHECKED) == 0
    (rid,) = _passing(main, tip)[0]
    path = LocalStore().attempt_path(rid)
    doc = json.loads(path.read_text())
    path.unlink()
    ids, why = _passing(main, tip)
    assert ids == [] and why[rid].startswith("its attempt can't be read")
    path.write_text("{not json")
    assert _passing(main, tip)[1][rid].startswith("its attempt can't be read (JSONDecodeError")
    path.write_text(json.dumps({**doc, "status": {**doc["status"], "rc": 1}}))
    assert _passing(main, tip) == ([], {rid: "its record says rc 1, not the index's passing check"})


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


def test_a_path_rule_leaves_out_what_its_patterns_name():
    from research.merge import rule_touches

    rule = ["flock/", "!README.md", "!docs/*.md"]
    assert rule_touches(rule, ["flock/README.md", "flock/sub/README.md", "docs/x.md"]) == []
    assert rule_touches(rule, ["flock/README.md", "flock/a.rs"]) == ["flock/"]
    assert rule_touches(rule, ["flockish/a.rs", "other/README.md"]) == []
    assert rule_touches(["flock/"], ["flock/README.md"]) == ["flock/"]            # no pattern, no exception


def test_a_flock_file_whose_name_git_quotes_still_requires_the_agreement(tmp_path):
    """`git diff --name-only` quotes a name with bytes outside ASCII (`"verity/protocols/proofs/flock/prover/\\303\\251.py"`), which no prefix of a
    `merge_requires` rule matched, so a candidate changing such a file required no `lean-agreement`."""
    from research.merge import required_steps

    repo = tmp_path / "repo"
    (repo / "mytools").mkdir(parents=True)
    (repo / "mytools" / "agree.py").write_text("from research.store.tool import Tool\nT = Tool(name='check', version='1', "
                                               "merge_requires={'lean-agreement': ['verity/protocols/proofs/flock/prover/', '!README.md']})\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "verity" / "protocols" / "proofs" / "flock" / "prover").mkdir(parents=True)
    (repo / "verity" / "protocols" / "proofs" / "flock" / "prover" / "é.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "é")
    assert required_steps(repo, base, _git(repo, "rev-parse", "HEAD"), "mytools.agree:T") == {"lean-agreement": ["verity/protocols/proofs/flock/prover/"]}


def test_the_bases_declaration_requires_the_agreement_when_the_checkout_holds_an_older_one(tmp_path):
    """The loop's checkout sat days behind `main`, its declaration naming Flock's verifier at the path before a move, so a
    candidate changing the verifier at its new path required no `lean-agreement` and its check skipped the step by name. Read
    at the candidate's base (``at``), the declaration names the path the candidate changed."""
    from research.merge import required_steps

    repo = tmp_path / "repo"
    (repo / "mytools").mkdir(parents=True)
    decl = ("from research.store.tool import Tool\nT = Tool(name='check', version='1', "
            "merge_requires={{'lean-agreement': ['{path}']}})\n")
    (repo / "mytools" / "agree.py").write_text(decl.format(path="verity/ml/flock/"))
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "before the move")
    before = _git(repo, "rev-parse", "HEAD")
    (repo / "mytools" / "agree.py").write_text(decl.format(path="verity/protocols/proofs/flock/verifier/"))
    _git(repo, "commit", "-q", "-am", "the move")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "verity" / "protocols" / "proofs" / "flock" / "verifier").mkdir(parents=True)
    (repo / "verity" / "protocols" / "proofs" / "flock" / "verifier" / "Hello.lean").write_text("-- a verifier change\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "a verifier change")
    tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", before)
    assert required_steps(repo, base, tip, "mytools.agree:T") == {}
    assert required_steps(repo, base, tip, "mytools.agree:T", at=base) == {
        "lean-agreement": ["verity/protocols/proofs/flock/verifier/"]}


def _deferred_lock(feat: Path, tip: str, files: dict[str, str], checked: str, made_as: dict[str, str] | None = None) -> str:
    """Drive's lock commit on ``tip``, naming its lock run ``checked``; its author and committer ``made_as``'s (as git's
    environment names them) when given."""
    _git(feat, "reset", "-q", "--hard", tip)
    for rel, text in files.items():
        (feat / rel).parent.mkdir(parents=True, exist_ok=True)
        (feat / rel).write_text(text)
        _git(feat, "add", rel)
    _git(feat, "commit", "-q", "-m", f"Lock records regenerated for #1\n\nChecked: {checked} (the lock run of `checked` {tip[:12]}, "
                                     f"passed by regenerating them)\nRegenerated: {', '.join(files)}")
    if made_as is None:
        return _git(feat, "rev-parse", "HEAD")
    return subprocess.run(["git", "commit-tree", "HEAD^{tree}", "-p", tip, "-m", _git(feat, "log", "-1", "--format=%B")], cwd=feat,
                          check=True, capture_output=True, text=True, env={**os.environ, **made_as}).stdout.strip()


def test_a_deferred_landings_lock_commit_is_vouched_for_only_by_the_records_its_running_check_regenerated(tmp_path):
    """The queue commits a lock run's records as soon as it passed, so `regenerated_parent` reads them from its run dir on its
    node, hashing the bytes it fetches; a lock commit whose records differ, that names another run, that leaves out a package
    the run regenerated, or whose run ran on another commit is refused, and without the queue's verdict (`deferred`) the run, a check of
    lean-audit alone with no passing attempt of the whole check, vouches for nothing."""
    from research.merge import regenerated_parent
    from research.store.local import LocalStore
    from test_landing import _live_run

    main, feat, tip = _repos(tmp_path)
    base, store, lock = _git(main, "rev-parse", "HEAD"), LocalStore(), "pkg/lean-audit.json"
    _, fetch, _ = _live_run(tmp_path, "r20261020-000001-0001", tip, {"pkg": '{"records": 2}\n'})
    _, both, _ = _live_run(tmp_path, "r20261020-000002-0002", tip, {"pkg": '{"records": 2}\n', "pkg2": '{"records": 5}\n'})
    _, elsewhere, _ = _live_run(tmp_path, "r20261020-000003-0003", base, {"pkg": '{"records": 2}\n'})

    def vouched(commit: str, rid: str, f, deferred: bool = True) -> str | None:
        return regenerated_parent(feat, commit, store, tool="check", kinds=frozenset({"check/v1"}), command=CHECKED,
                                  deferred=rid if deferred else None, fetch=f)

    ok = _deferred_lock(feat, tip, {lock: '{"records": 2}\n'}, "r20261020-000001-0001")
    assert vouched(ok, "r20261020-000001-0001", fetch) == "r20261020-000001-0001"
    assert vouched(ok, "r20261020-000001-0001", fetch, deferred=False) is None
    assert vouched(_deferred_lock(feat, tip, {lock: '{"records": 1}\n'}, "r20261020-000001-0001"), "r20261020-000001-0001", fetch) is None    # forged bytes
    assert vouched(_deferred_lock(feat, tip, {lock: '{"records": 2}\n'}, "r20261020-000009-0009"), "r20261020-000001-0001", fetch) is None    # another check
    assert vouched(_deferred_lock(feat, tip, {lock: '{"records": 2}\n'}, "r20261020-000002-0002"), "r20261020-000002-0002", both) is None     # pkg2's left out
    assert vouched(_deferred_lock(feat, tip, {lock: '{"records": 2}\n', "pkg2/lean-audit.json": '{"records": 5}\n'}, "r20261020-000002-0002"),
                   "r20261020-000002-0002", both) == "r20261020-000002-0002"
    assert vouched(_deferred_lock(feat, tip, {lock: '{"records": 2}\n'}, "r20261020-000003-0003"), "r20261020-000003-0003", elsewhere) is None  # it checked main


def test_a_lock_commit_is_vouched_for_by_its_records_whoever_made_it(tmp_path):
    """The queue makes a lock commit as its own identity (`queue.IDENTITY`) or, where its repository's rules name one, as that
    author (`Rules.author`): `regenerated_parent` reads the commit's trailers, its parent and its records' bytes, so it vouches for
    either against the run, and neither author vouches for records the run didn't regenerate."""
    from research.merge import regenerated_parent
    from research.queue import IDENTITY, Rules
    from research.store.local import LocalStore
    from test_landing import _live_run
    from test_queue import AUTHORED, NAMED, OWN, _who

    main, feat, tip = _repos(tmp_path)
    store, lock, rid = LocalStore(), "pkg/lean-audit.json", "r20261020-000001-0001"
    _, fetch, _ = _live_run(tmp_path, rid, tip, {"pkg": '{"records": 2}\n'})
    for made_as, who in ((IDENTITY, OWN), (Rules.parse(AUTHORED).identity, NAMED)):
        ok, forged = (_deferred_lock(feat, tip, {lock: text}, rid, made_as) for text in ('{"records": 2}\n', '{"records": 1}\n'))
        assert _who(feat, ok) == _who(feat, forged) == who
        got = [regenerated_parent(feat, c, store, tool="check", kinds=frozenset({"check/v1"}), command=CHECKED, deferred=rid,
                                  fetch=fetch) for c in (ok, forged)]
        assert got == [rid, None], who


def test_a_lean_report_naming_a_repaired_file_outside_repaired_is_unreadable():
    import pytest

    from research.merge import Unreadable, _regenerated_in

    with pytest.raises(Unreadable, match="no lock run repairs"):
        _regenerated_in("r1", {"regenerated": {}, "repaired": {"AGENTS.md": "/x/lean-audit/repaired/AGENTS.md"}})
    assert _regenerated_in("r1", {"repaired": {"tests/test_lean_packages.py": "/r/lean-audit/repaired/t.py"}}) == {
        "tests/test_lean_packages.py": "lean-audit/repaired/t.py"}


def test_a_lock_commit_carries_the_files_its_lock_run_repaired_and_no_other_file(tmp_path):
    """The lock run's `--bookkeeping warn` repairs files beside the locks (`REPAIRED`, its report's `repaired`): a lock commit
    holding the repaired copy byte for byte is vouched for; one whose copy differs, one that leaves it out, and one changing
    any other file is refused."""
    import json

    from research.merge import REPAIRED, regenerated_parent
    from research.store.local import LocalStore
    from test_landing import _live_run

    main, feat, tip = _repos(tmp_path)
    store, lock, packages, rid = LocalStore(), "pkg/lean-audit.json", "tests/test_lean_packages.py", "r20261020-000005-0005"
    assert packages in REPAIRED
    d, fetch, _ = _live_run(tmp_path, rid, tip, {"pkg": '{"records": 2}\n'})
    (d / "lean-audit" / "repaired" / "tests").mkdir(parents=True)
    (d / "lean-audit" / "repaired" / packages).write_text('UNWITNESSED = {"pkg": 3}\n')
    audit = json.loads((d / "lean-audit" / "audit.json").read_text())
    audit["repaired"] = {packages: f"/workspace/research/runs/{rid}/lean-audit/repaired/{packages}"}
    (d / "lean-audit" / "audit.json").write_text(json.dumps(audit))

    def vouched(files: dict[str, str]) -> str | None:
        return regenerated_parent(feat, _deferred_lock(feat, tip, files, rid), store, tool="check", kinds=frozenset({"check/v1"}),
                                  command=CHECKED, deferred=rid, fetch=fetch)

    assert vouched({lock: '{"records": 2}\n', packages: 'UNWITNESSED = {"pkg": 3}\n'}) == rid
    assert vouched({lock: '{"records": 2}\n', packages: 'UNWITNESSED = {"pkg": 9}\n'}) is None
    assert vouched({lock: '{"records": 2}\n'}) is None
    assert vouched({lock: '{"records": 2}\n', packages: 'UNWITNESSED = {"pkg": 3}\n', "a.txt": "x\n"}) is None


def test_a_report_naming_a_regenerated_record_the_stores_files_lack_is_unreadable_not_none(tmp_path, monkeypatch):
    """`regenerated_records` returning {} both for a check that regenerated nothing and for one whose records it couldn't
    read would land the candidate without them; a record the report names and the run's files lack raises."""
    import pytest
    from research.merge import Unreadable, regenerated_records
    from research.store.local import LocalStore

    main, feat, tip = _repos(tmp_path)
    assert _checked(tmp_path, feat, monkeypatch, _REGENERATING.replace("open(d + '/lean-audit/audit-pkg/lean-audit.json', 'w')"
                                                                       ".write('{\"records\": 2}\\n'); ", "")) == 0
    store = LocalStore()
    (rid,), = store.index.sql("SELECT id FROM attempts WHERE source_commit = ?", (tip,))
    with pytest.raises(Unreadable, match="pkg's lock"):
        regenerated_records(store, rid)
    assert regenerated_records(store, "r20261020-000004-0004") is None


#: a passing check whose Lean check regenerated pkg's lock (`check.py --regenerate`): its report and the record, in its run files
_REGENERATING = ("import json, os; d = os.environ['RESEARCH_RUN_DIR']; os.makedirs(d + '/lean-audit/audit-pkg'); "
                 "open(d + '/lean-audit/audit-pkg/lean-audit.json', 'w').write('{\"records\": 2}\\n'); "
                 "json.dump({'passed': True, 'regenerated': {'pkg': d + '/lean-audit/audit-pkg/lean-audit.json'}}, "
                 "open(d + '/lean-audit/audit.json', 'w')); "
                 "json.dump({'schema': 'research/result/v0.1', 'kind': 'check/v1', 'run_id': os.environ['RESEARCH_RUN_ID'], "
                 "'workload_fingerprint': {}, 'validation': {'status': 'passed', 'checker': 't'}, 'measurements': [], "
                 "'steps': [{'step': 'agree', 'status': 'passed'}], 'artifacts': ['lean-audit/']}, open(d + '/result.json', 'w'))")


def _relock(feat: Path, text: str, trailer: str | None) -> str:
    (feat / "pkg").mkdir(exist_ok=True)
    (feat / "pkg" / "lean-audit.json").write_text(text)
    _git(feat, "add", "pkg/lean-audit.json")
    _git(feat, "commit", "-q", "-m", "Lock records regenerated" + (f"\n\nRegenerated: {trailer}" if trailer else ""))
    return _git(feat, "rev-parse", "HEAD")


def test_a_commit_of_the_locks_a_passing_check_regenerated_is_vouched_for_by_that_check(tmp_path, monkeypatch):
    """A check's Lean step passes by regenerating records, and the records are committed on top of what it checked. That
    commit's tree is the one the check audited, byte for byte, so the check vouches for it (`regenerated_parent` reading the
    passing attempts of its parent); other bytes or no `Regenerated:` trailer don't."""
    from research.merge import declared_command, produced, regenerated_parent
    from research.store.local import LocalStore

    main, feat, tip = _repos(tmp_path)
    assert _checked(tmp_path, feat, monkeypatch, _REGENERATING) == 0
    (rid,) = _passing(main, tip)[0]

    def vouched(commit: str) -> str | None:
        return regenerated_parent(main, commit, LocalStore(), tool="check", kinds=produced(main, "check"),
                                  command=declared_command(main, "check"))

    assert vouched(_relock(feat, '{"records": 1}\n', "pkg/lean-audit.json")) is None      # not what the check wrote
    _git(feat, "reset", "-q", "--hard", tip)
    assert vouched(_relock(feat, '{"records": 2}\n', None)) is None                       # the bytes, but no trailer
    _git(feat, "reset", "-q", "--hard", tip)
    assert vouched(_relock(feat, '{"records": 2}\n', "pkg/lean-audit.json")) == rid


