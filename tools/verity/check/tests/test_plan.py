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
    (`KEYLESS`, flock-rust-tests here without its toolchain) and the Lean group this tree has no verifier for (`lean`,
    skipped, in place of lean-build and every Lean part): the gate refuses on those alone, whatever passes exist.  Runs the suites twice for real (about 5 s)."""
    root = tmp_path / "tree"
    a = _tree(root)
    packs = {"r-a": _checked(P, root, tmp_path, "r-a")}
    plan_a = _plan(P, root, tmp_path, packs)
    by_a = {p["part"]: p for p in plan_a["parts"]}
    suites = sorted(n for n in by_a if n.startswith("suite:"))
    assert suites == ["suite:one", "suite:repository", "suite:two"]
    assert all(by_a[n]["state"] == "passed" and by_a[n]["found"][0]["attempt"] == "r-a" for n in suites)
    assert by_a["preflight-lints"]["state"] == "passed" and by_a["preflight-lints"]["key"] == by_a["suite:repository"]["key"]
    unpassed = {n for n, p in by_a.items() if p.get("required", True) and p["state"] != "passed"}
    assert unpassed == {*P.KEYLESS, "flock-rust-tests", "lean"} - {"lean-build", "lean-unit-cut"}
    assert by_a["lean"]["state"] == "skipped" and all(by_a[n]["state"] == "unkeyed" for n in unpassed - {"lean"})
    assert not plan_a["fully_passed"] and P.text(plan_a)[-1].startswith("plan: not fully passed: 4 of 9 required parts passed")
    cited, why = _judged(plan_a, packs)
    assert cited == [] and why.startswith(f"{len(unpassed)} of ") and "suite:" not in why and "lean skipped" in why

    (root / "one" / "NOTES.md").write_text("prose no suite reads\n")
    _commit(root, "b")
    plan_b = _plan(P, root, tmp_path, packs)
    assert plan_b["commit"] != a
    assert ([(p["part"], p.get("key"), p["state"]) for p in plan_b["parts"]]
            == [(p["part"], p.get("key"), p["state"]) for p in plan_a["parts"]])
    assert _judged(plan_b, packs)[1] == why

    (root / "one" / "tests" / "test_it.py").write_text("def test_one():\n    assert True\n")
    _commit(root, "c")
    by_c = {p["part"]: p for p in _plan(P, root, tmp_path, packs)["parts"]}
    assert by_c["suite:one"]["state"] == "must-run" and by_c["suite:one"]["key"] != by_a["suite:one"]["key"]
    assert (by_c["suite:two"]["state"], by_c["suite:two"]["key"]) == ("passed", by_a["suite:two"]["key"])
    ws = by_c["suite:repository"]
    assert ws["key"] != by_a["suite:repository"]["key"] and ws["state"] == "passed" and ws["found"][0]["traced"]
    _, why_c = _judged(_plan(P, root, tmp_path, packs), packs)
    assert "suite:one must-run" in why_c and "suite:two" not in why_c and "suite:repository" not in why_c

    packs["r-c"] = _checked(P, root, tmp_path, "r-c")
    plan_c = _plan(P, root, tmp_path, packs)
    by_c = {p["part"]: p for p in plan_c["parts"]}
    assert [f["attempt"] for f in by_c["suite:one"]["found"]] == ["r-c"]
    assert by_c["suite:two"]["found"][0]["attempt"] == "r-a"
    _, why_c = _judged(plan_c, packs)
    assert "suite:" not in why_c and why_c.startswith(f"{len(unpassed)} of ")


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
