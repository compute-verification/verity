"""`check.py --plan` (`plan.py`): each part's key at a commit and the recorded passes under it, found by the tree's own key code,
and `research merge --verdicts`'s judgment of that plan (`plan_refusal`)."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
#: the command the fake store's attempts ran, held to itself (`research.merge.invocation`)
COMMAND = ("uv", "run", "python", "tools/verity/check/check.py")


@pytest.fixture
def P(monkeypatch):
    """plan.py, and every module it loads from a planned tree's check dropped afterwards."""
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("plan_under_test", HERE / "plan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    for name in set(sys.modules) - before:
        del sys.modules[name]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()


def _commit(root: Path, msg: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)
    return _git(root, "rev-parse", "HEAD")


def _tree(root: Path) -> str:
    """A checkout with this directory's check and a workspace of suites `one` and `two` and the repository's own (`repository`, at the
    root, keying every file but Markdown); no verifier, so check would skip its Lean group.  Its first commit."""
    shutil.copytree(HERE, root / "tools" / "verity" / "check",
                    ignore=shutil.ignore_patterns("tests", "__pycache__", "pyproject.toml", "*.md"))
    (root / ".agents" / "skills" / "using-slack").mkdir(parents=True)
    shutil.copy(REPO / ".agents" / "skills" / "using-slack" / "registry.json", root / ".agents" / "skills" / "using-slack")
    (root / ".gitignore").write_text("__pycache__/\n")
    (root / "pyproject.toml").write_text('[project]\nname = "ws"\nversion = "0"\n\n[tool.uv.workspace]\nmembers = ["one", "two"]\n\n'
                                         '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (root / "tests").mkdir()
    (root / "tests" / "test_ws.py").write_text("def test_ws():\n    pass\n")
    for name in ("one", "two"):
        (root / name / "tests").mkdir(parents=True)
        (root / name / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0"\n\n'
                                                    '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
        (root / name / "tests" / "test_it.py").write_text(f"def test_{name}():\n    pass\n")
    _git(root, "init", "-q")
    return _commit(root, "a")


def _checked(P, root: Path, tmp_path: Path, rid: str) -> Path:
    """Run the tree's suites as check's pytest step runs them (`suites.py` under the steps' environment inside a recorded run),
    then export what they cached as that run's verdict pack."""
    C = P._load("planned_check", root / "tools" / "verity" / "check" / "check.py")
    cache, check = tmp_path / f"tests-{rid}", tmp_path / f"check-{rid}"
    env = {**P.step_env(C), "VERITY_TEST_CACHE": str(cache), "RESEARCH_STORE": str(tmp_path / "store"),
           "RESEARCH_STORE_CONFIG": str(tmp_path / "absent.toml"), "RESEARCH_FIXTURES_URL": "", "CHECK_SLOT_MEM_GB": "1000"}
    p = subprocess.run([sys.executable, str(root / "tools" / "verity" / "check" / "suites.py"), "--jobs", "1", "--run", rid,
                        "--logs", str(tmp_path / f"logs-{rid}")], cwd=root, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    pack = tmp_path / f"{rid}.tar.gz"
    subprocess.run([sys.executable, str(root / "tools" / "verity" / "check" / "verdicts.py"), "export", str(pack)],
                   env={**os.environ, "VERITY_TEST_CACHE": str(cache), "CHECK_CACHE": str(check)}, check=True, capture_output=True)
    return pack


def _plan(P, root: Path, tmp_path: Path, packs: dict[str, Path]) -> dict:
    """The plan of ``root``'s HEAD from ``packs`` alone, as `check.py --plan --packs-only` makes it (circuit-check left out: the
    tree has none)."""
    V = P._load("plan_verdicts", HERE / "verdicts.py")
    d = Path(tmp_path / f"unpacked-{len(list(tmp_path.glob('unpacked-*')))}")
    sources = [P.unpack(V, pack, d / rid, rid) for rid, pack in packs.items()]
    parts = P.plan_tree(root, sources, False, circuit=lambda C, s: [])
    return P.document(parts, _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}"), None, sources, [])


class _Store:
    """The evidence store as `plan_refusal` reads it: each attempt a `check` that ended, ran `COMMAND`, and published ``packs``'s
    pack."""

    def __init__(self, packs: dict[str, Path]):
        import hashlib

        self.sha = {rid: hashlib.sha256(p.read_bytes()).hexdigest() for rid, p in packs.items()}

    def get_attempt(self, rid):
        return {"tool": {"name": "check"}, "status": {"state": "done"}, "argv": list(COMMAND), "outputs": {"run_files": f"art:{rid}"}}

    def listing(self, art):
        return [{"path": "verdicts.tar.gz", "sha256": self.sha[art.removeprefix("art:")]}]


def _judged(doc: dict, packs: dict[str, Path]) -> tuple[list[str], str | None]:
    from research.merge import plan_refusal

    return plan_refusal(doc, _Store(packs), doc["commit"], doc["tree"], "check", COMMAND, {})


@pytest.mark.slow
def test_a_change_no_key_reads_plans_as_the_checked_commit_and_one_a_suite_reads_must_run_until_its_pass_is_recorded(P, tmp_path):
    """The trust property of `research merge --verdicts`: the passes a check of A recorded (its verdict pack) count for B, which
    differs from A only in a Markdown file no suite's key reads, part for part at A's keys; and for C, which changes a file of
    suite `one`, they count for every suite but `one`, which must run at its new key, so the gate refuses C until a check
    records a pass under it.  The repository's own suite, whose key reads every file, has a new key too, and passes as `check`
    would reuse it: by its traced pass (`PerTest.whole`), every file its run read being as it was.  What no plan passes is the parts without a key
    (flock-rust-tests here without its toolchain, and so flock-circuit-build, which only its pass implies) and the Lean group
    this tree has no verifier for (`lean`, skipped, in place of lean-build and every Lean part): the gate refuses on those
    alone, whatever passes exist.  The other build steps (`IMPLIED`) are implied by the suites, so they must run with one.  Runs the suites twice for real (about 5 s)."""
    root = tmp_path / "tree"
    a = _tree(root)
    packs = {"r-a": _checked(P, root, tmp_path, "r-a")}
    plan_a = _plan(P, root, tmp_path, packs)
    by_a = {p["part"]: p for p in plan_a["parts"]}
    suites = sorted(n for n in by_a if n.startswith("suite:"))
    assert suites == ["suite:one", "suite:repository", "suite:two"]
    assert all(by_a[n]["state"] == "passed" and by_a[n]["found"][0]["attempt"] == "r-a" for n in suites)
    assert by_a["suite:one"]["found"][0]["paths"] == [f"tests/{by_a['suite:one']['key']}.json"], "what `check.Reused` packs"
    lints = by_a["preflight-lints"]
    assert lints["state"] == "passed" and lints["found"] == by_a["suite:repository"]["found"], "no check stored its own pass here"
    assert lints["key"] and lints["key"] != by_a["suite:repository"]["key"]
    for build in ("preflight-lock", "preflight-fixtures"):
        assert by_a[build]["state"] == "implied" and by_a[build]["implied_by"] == suites
    assert by_a["flock-circuit-build"]["state"] == "must-run" and by_a["flock-circuit-build"]["implied_by"] == ["flock-rust-tests"]
    unpassed = {n for n, p in by_a.items() if p.get("required", True) and p["state"] not in P.OK}
    assert unpassed == {"flock-circuit-build", "flock-rust-tests", "lean"}
    assert by_a["lean"]["state"] == "skipped" and by_a["flock-rust-tests"]["state"] == "unkeyed"
    assert not plan_a["fully_passed"] and P.text(plan_a)[-1].startswith(
        "plan: not fully passed: 6 of 9 required parts passed (2 implied by those that read what they build)")
    cited, why = _judged(plan_a, packs)
    assert cited == [] and why.startswith(f"{len(unpassed)} of ") and "suite:" not in why and "lean skipped" in why

    (root / "one" / "NOTES.md").write_text("prose no suite reads\n")
    _commit(root, "b")
    plan_b = _plan(P, root, tmp_path, packs)
    assert plan_b["commit"] != a
    assert ([(p["part"], p.get("key") if p["part"] != "preflight-lints" else None, p["state"]) for p in plan_b["parts"]]
            == [(p["part"], p.get("key") if p["part"] != "preflight-lints" else None, p["state"]) for p in plan_a["parts"]])
    assert plan_b["parts"][1]["key"] != lints["key"], "the lints read every file, Markdown too"
    assert _judged(plan_b, packs)[1] == why

    (root / "one" / "tests" / "test_it.py").write_text("def test_one():\n    assert True\n")
    _commit(root, "c")
    by_c = {p["part"]: p for p in _plan(P, root, tmp_path, packs)["parts"]}
    assert by_c["suite:one"]["state"] == "must-run" and by_c["suite:one"]["key"] != by_a["suite:one"]["key"]
    assert (by_c["suite:two"]["state"], by_c["suite:two"]["key"]) == ("passed", by_a["suite:two"]["key"])
    ws = by_c["suite:repository"]
    assert ws["key"] != by_a["suite:repository"]["key"] and ws["state"] == "passed" and ws["found"][0]["traced"]
    whole, reads = ws["found"][0]["paths"]
    assert whole.startswith("tests/per-test/repository/(suite)--") and reads.startswith("tests/per-test/repository/traced/")
    _, why_c = _judged(_plan(P, root, tmp_path, packs), packs)
    assert "suite:one must-run" in why_c and "suite:two" not in why_c and "suite:repository" not in why_c
    assert "preflight-lock must-run (it makes the current uv.lock every suite's key resolves: suite:one not passed)" in why_c

    packs["r-c"] = _checked(P, root, tmp_path, "r-c")
    plan_c = _plan(P, root, tmp_path, packs)
    by_c = {p["part"]: p for p in plan_c["parts"]}
    assert [f["attempt"] for f in by_c["suite:one"]["found"]] == ["r-c"]
    assert by_c["suite:two"]["found"][0]["attempt"] == "r-a"
    _, why_c = _judged(plan_c, packs)
    assert "suite:" not in why_c and why_c.startswith(f"{len(unpassed)} of ")


def test_a_job_off_the_merge_gate_is_planned_and_required_of_no_check(P, tmp_path):
    """circuit-check and ratchet-gate are off the merge gate (ci.toml's `gate = false`): their parts are listed, none required, so
    `research merge --verdicts` admits a commit whose targets no check passed."""
    root = tmp_path / "tree"
    _tree(root)
    parts = P.plan_tree(root, [], False, circuit=lambda C, s: [P.Part("circuit-check:t", "circuit-check", "v1/k", "must-run")])
    doc = P.document(parts, _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}"), None, [], [])
    by = {p["part"]: p for p in doc["parts"]}
    assert by["circuit-check:t"]["state"] == "must-run" and by["circuit-check:t"]["required"] is False
    assert by["ratchet-gate"]["required"] is False
    off = ("circuit-check:t", "ratchet-gate", "lean-changed", "lean-agreement")
    assert all(p.get("required", True) for n, p in by.items() if n not in off)
    _cited, why = _judged(doc, {})
    assert why and "circuit-check" not in why and "ratchet-gate" not in why


def test_lean_agreement_is_required_for_a_change_its_merge_requires_covers_in_the_commits_own_ci_toml(P, tmp_path):
    root = tmp_path / "repo"
    (root / "tools" / "verity" / "check").mkdir(parents=True)
    (root / "tools" / "verity" / "check" / "ci.toml").write_text(
        '[[job]]\nname = "lean-agreement"\nmerge_requires = ["verity/ml/flock/", "!README.md"]\n')
    _git(root, "init", "-q")
    base = _commit(root, "rule")
    (root / "verity" / "ml" / "flock").mkdir(parents=True)
    (root / "verity" / "ml" / "flock" / "README.md").write_text("x\n")
    prose = _commit(root, "prose")
    assert not P.required_agreement(prose, base, root)
    (root / "verity" / "ml" / "flock" / "x.py").write_text("x = 1\n")
    code = _commit(root, "code")
    assert P.required_agreement(code, base, root) and P.required_agreement(code, prose, root)
    assert not P.required_agreement(code, None, root)


def test_the_plan_takes_both_packs_of_each_check_of_the_commit_and_of_each_run_given(P, tmp_path):
    """The reused pack (`check.Reused`) runs to MBs and a GB unpacked, so it is taken only from a check of the commit planned
    and each --from, beside the newest checks' own packs; a check gathered already isn't fetched again."""
    import argparse

    fetched = []
    have = {("r-new", "verdicts.tar.gz"), ("r-mine", "verdicts.tar.gz"), ("r-mine", P.REUSED), ("r-from", P.REUSED),
            ("r-new", P.REUSED)}

    class Host:
        def gathered_packs(self, d, store):
            return [("r-new", d / "new.tar.gz")]

        def attempt_pack(self, store, rid, d, name="verdicts.tar.gz"):
            fetched.append((rid, name))
            return d / rid / name if (rid, name) in have else None

    class Store:
        index = None

        def __init__(self):
            self.index = self

        def sql(self, query, params):
            assert params[0] == "c0ffee" and "source_commit" in query
            return [("r-mine", "check")]

    import research.store.local as local

    old, local.LocalStore = local.LocalStore, Store
    try:
        notes = []
        a = argparse.Namespace(no_gather=False, from_=["r-from", "r-none"], verdicts_in=[])
        got = P.gather(Host(), tmp_path, a, notes, "c0ffee")
    finally:
        local.LocalStore = old
    assert [(rid, p.name) for rid, p in got] == [("r-new", "new.tar.gz"), ("r-from", P.REUSED), ("r-mine", "verdicts.tar.gz"),
                                                  ("r-mine", P.REUSED)]
    assert ("r-new", P.REUSED) not in fetched and ("r-new", "verdicts.tar.gz") not in fetched
    assert notes == ["--from r-none: that attempt published no verdict pack"]


def _fake_check(tmp_path: Path, keys: dict):
    """A planned tree's check as `keyed_step_parts` and `circuit_parts` read it: its steps' environment, `STEP_KEYS`, its cache."""
    import types

    return types.SimpleNamespace(env=lambda: {"PATH": "/usr/bin"}, STORE_REQUIRED_ENV="S", REGRESSION_ENV="R", CUSTODY_ENV="C",
                                 STEP_KEYS={n: (lambda e, k=k: k) for n, k in keys.items()}, ROOT=tmp_path,
                                 cache_dir=lambda: tmp_path / "check")


def test_a_keyed_steps_pass_names_its_entry_under_its_existing_key(P, tmp_path):
    """lean-agreement's and flock-rust-tests' passes (`agreement_key`, `rust_tests_key`): what `check.Reused` packs when a
    check reused one, at the key the step itself looks up."""
    C = _fake_check(tmp_path, {"lean-agreement": "a1", "flock-rust-tests": "r1"})
    src = P.Source("own", tmp_path / "check", [tmp_path / "tests"])
    (tmp_path / "check" / "lean-agreement").mkdir(parents=True)
    (tmp_path / "check" / "lean-agreement" / "a1.json").write_text("{}")
    agree, rust = P.keyed_step_parts(C, [src], ["lean-agreement", "flock-rust-tests"])
    assert agree.state == "passed" and agree.found == [{"source": "own", "paths": ["check/lean-agreement/a1.json"], "key": "a1"}]
    assert rust.state == "must-run" and rust.key == "r1"


def test_a_circuit_targets_pass_names_its_entry_and_the_task_list_that_keys_it(P, tmp_path, monkeypatch):
    """A target's key comes only from the task list's entry (`CIRCUIT_PROGRAM`), so its pass is that target's entry and the
    task list's, each under its existing key, from whichever source held them."""
    import json
    import subprocess as sp

    (tmp_path / "tools" / "verity" / "circuit_check").mkdir(parents=True)
    C = _fake_check(tmp_path, {})
    own, pack = P.Source("own", tmp_path / "own", [tmp_path / "own-tests"]), P.Source("pack:r1", tmp_path / "p", [], "r1", "s")
    got = {"repo": str(tmp_path), "untraced": [], "list": {"source": "pack:r1", "entry": "L", "key": "v1/ll"},
           "targets": [{"kind": "def", "target": "a", "key": "v1/aa", "found": [["own", "E"]]},
                       {"kind": "def", "target": "b", "key": "v1/bb", "found": []}]}
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 0, "CIRCUIT-PLAN " + json.dumps(got), ""))
    a, b = P.circuit_parts(C, [own, pack])
    assert (a.part, a.key, a.state, b.state) == ("circuit-check:a", "v1/aa", "passed", "must-run")
    assert a.found == [{"source": "own", "paths": ["check/circuit-check/v1/aa/E.json", "check/circuit-check/v1/ll/L.json"],
                        "entry": "E"}]


def test_a_plan_on_a_machine_ships_the_commit_with_this_plan_and_the_packs_and_reads_back_what_it_wrote(P, tmp_path, monkeypatch):
    """`--on MACHINE` (`on_machine`): a `research run --wait` of a worktree of the commit, sent this plan.py, verdicts.py and
    each gathered pack under a name of its own, runs the inner plan in that tree's environment on those packs; the plan it
    writes comes back, and the worktree is gone.  The run's command goes through a real shell here, a fake `uv` its end; not a
    login shell, whose profile would change the PATH the steps' keys read from the one a check's steps see."""
    import json
    import subprocess as sp

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f").write_text("1")
    first = _commit(repo, "one")
    (repo / "f").write_text("2")
    _commit(repo, "two")
    monkeypatch.setattr(P, "ROOT", repo)
    for d, name in (("a", "verdicts.tar.gz"), ("b", "verdicts-reused.tar.xz")):
        (tmp_path / d).mkdir()
        (tmp_path / d / name).write_bytes(name.encode())
    packs = [("r1", tmp_path / "a" / "verdicts.tar.gz"), (None, tmp_path / "b" / "verdicts-reused.tar.xz")]
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "uv").write_text('#!/bin/sh\nwhile [ "$1" != python ]; do shift; done; shift\nexec python3 -c \'import json, os, sys; '
                             'json.dump(sys.argv[1:], open(os.environ["RESEARCH_RUN_DIR"] + "/out/plan.json", "w"))\' "$@"\n')
    (bin_ / "uv").chmod(0o755)
    run_dir, seen = tmp_path / "run", {}

    def run(argv, **k):
        if argv[3] == "status":
            return sp.CompletedProcess(argv, 0, json.dumps({"runs": [{"dir": str(run_dir)}]}), "")
        src = Path(argv[argv.index("--source") + 1])
        seen.update(head=_git(src, "rev-parse", "HEAD"), on=argv[argv.index("--on") + 1], wait="--wait" in argv,
                    pool=argv[argv.index("--pool") + 1], shell=argv[-3:-1])
        (run_dir / "inputs").mkdir(parents=True)
        for i, x in enumerate(argv):
            if x == "--send":
                shutil.copy(argv[i + 1], run_dir / "inputs")
        seen["sent"] = sorted(p.name for p in (run_dir / "inputs").iterdir())
        sp.run(["bash", "-c", argv[-1]], cwd=src, check=True,
               env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "RESEARCH_RUN_DIR": str(run_dir)})
        return sp.CompletedProcess(argv, 1, "research: run r20261019-000000-abcd launched on m (None) pid=1\n", "")

    out, notes = tmp_path / "plan.json", []
    assert P.on_machine("auto", first, tmp_path, packs, [], out, notes, run=run) == 2
    assert P.on_machine("m", first, tmp_path, packs, ["--packs-only", "--base", "B"], out, notes, run=run) == 1
    assert seen == {"head": first, "on": "m", "wait": True, "pool": "research", "shell": ["bash", "-c"],
                    "sent": ["pack-0.tar.gz", "pack-1.tar.xz", "plan.py", "verdicts.py"]}
    i = f"{run_dir}/inputs"
    assert json.loads(out.read_text()) == [f"{i}/plan.py", "--tree", ".", "--out", f"{run_dir}/out/plan.json", "--packs-only",
                                           "--base", "B", f"--pack=r1={i}/pack-0.tar.gz", f"--pack=={i}/pack-1.tar.xz"]
    assert notes == ["planned on m in run r20261019-000000-abcd, on the 2 packs gathered where it was asked"]
    assert len(_git(repo, "worktree", "list").splitlines()) == 1
