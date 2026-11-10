"""`tools/verity/lean/warm.py`: the Lean audit of a ready mark's merge audits the packages the change reaches and no other,
in a Lean audit slot only when a pass is missing, leaves a pack of the Lean verdicts it stored, and says why it failed in its
last line. The audit itself is a stand-in here: what it is asked to run is the subject."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W = _load(HERE.parent / "warm.py", "lean_warm_under_test")
LA, V = W.tools(ROOT)

BASE = "b" * 40
#: a tree of three audited packages: `b` requires `a` by path, `c` stands alone and names `b` as its prover
FILES = {
    "tools/verity/lean/check.py": "", "tools/verity/lean/lean_audit.py": "", "tools/verity/lean/merge.py": "",
    "tools/verity/lean/README.md": "", "tools/verity/lean/tests/test_x.py": "", "tools/verity/check/common.py": "",
    LA.CONTRACT: "", "docs/note.md": "",
    "pkg/a/lakefile.toml": "", "pkg/a/A.lean": "theorem a : True := trivial\n", "pkg/a/lean-audit.json": "{}",
    "pkg/a/lake-manifest.json": json.dumps({"packages": [{"name": "dep", "type": "git", "rev": "abc"}]}),
    "pkg/b/lakefile.toml": "", "pkg/b/B.lean": "", "pkg/b/lean-audit.json": "{}",
    "pkg/b/lake-manifest.json": json.dumps({"packages": [{"name": "a", "type": "path", "dir": "../a"}]}),
    "pkg/c/lakefile.toml": "", "pkg/c/C.lean": "", "pkg/c/lean-audit.json": json.dumps({"proved_in": ["../b"]}),
    "pkg/c/lake-manifest.json": json.dumps({"packages": []}),
}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """`FILES` on disk as a checkout `warm` reads with this suite's `lean_audit` and `verdicts`, the caches under ``tmp_path``."""
    root = tmp_path / "repo"
    for f, text in FILES.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    monkeypatch.setattr(LA, "ROOT", root)
    monkeypatch.setattr(LA.COMMON, "tracked_files", lambda r=None: sorted(FILES))
    monkeypatch.setattr(W, "tools", lambda t: (LA, V))
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "cache" / "check"))
    monkeypatch.setenv("VERITY_TEST_CACHE", str(tmp_path / "cache" / "tests"))
    monkeypatch.delenv("CHECK_LEAN_DEPS_URLS", raising=False)
    return root


def _diff(monkeypatch, paths: list[str]) -> None:
    monkeypatch.setattr(W, "changed", lambda t, base: sorted(paths))


class Audit:
    """A stand-in for the process `audit_merge` starts: it notes the command, stores what a real audit would under the caches
    (``stores``: a path under the check cache and its text) and leaves ``report`` as `audit.json` in `--out`."""

    def __init__(self, rc: int = 0, stores: dict[str, str] | None = None, report: dict | None = None):
        self.rc, self.stores, self.report, self.calls = rc, stores or {}, report, []

    def __call__(self, argv, cwd=None):
        self.calls.append((list(argv), cwd))
        cache = Path(LA.COMMON.cache_dir())
        for rel, text in self.stores.items():
            (cache / rel).parent.mkdir(parents=True, exist_ok=True)
            (cache / rel).write_text(text)
        if self.report is not None:
            out = Path(argv[argv.index("--out") + 1])
            (out / "audit.json").write_text(json.dumps(self.report))
        return types.SimpleNamespace(returncode=self.rc)


def _members(pack: Path) -> list[str]:
    with tarfile.open(pack) as tar:
        return sorted(tar.getnames())


def test_the_names_warm_reads_are_the_auditors_and_the_verdict_packs_own():
    """`warm` loads `lean_audit.py` and `verdicts.py` by path from the checkout it audits, so a rename there breaks it only at a
    mark: this pins each name it reads, and that the auditor it loads audits the checkout it was loaded from."""
    assert LA.ROOT == ROOT
    for name in ("packages", "inputs", "reusable", "key", "FLAGS"):
        assert hasattr(LA, name), name
    assert callable(LA.COMMON.tracked_files) and callable(LA.COMMON.cache_dir)
    for name in ("snapshot", "changed", "export"):
        assert callable(getattr(V, name)), name
    assert V.SETTINGS.CHECK_LEAN_DEPS_URLS is None or isinstance(V.SETTINGS.CHECK_LEAN_DEPS_URLS, str)
    assert LA.tool_files([f"{LA.LEAN_TOOL}warm.py", f"{LA.LEAN_TOOL}check.py"])[-1] == f"{LA.LEAN_TOOL}check.py", \
        "no audit reads warm.py, so an edit to it moves no package's key (`lean_audit.NOT_READ`)"


@pytest.mark.parametrize("diff, want", [
    (["pkg/a/A.lean"], ["pkg/a", "pkg/b", "pkg/c"]),            # b requires a by path, and c reads the build of its prover b
    (["pkg/b/B.lean"], ["pkg/b", "pkg/c"]),
    (["pkg/c/C.lean"], ["pkg/c"]),
    (["pkg/c/lean-audit.json"], ["pkg/c"]),                     # a lock alone
    (["pkg/a/Gone.lean"], ["pkg/a", "pkg/b", "pkg/c"]),         # removed: no longer among the tree's files
    (["tools/verity/lean/check.py"], ["pkg/a", "pkg/b", "pkg/c"]),  # the auditor is in every package's key
    (["tools/verity/check/common.py"], ["pkg/a", "pkg/b", "pkg/c"]),
    (["tools/verity/lean/merge.py", "tools/verity/lean/README.md", "tools/verity/lean/tests/test_x.py", "docs/note.md"], []),
    ([], []),
])
def test_a_change_reaches_each_package_whose_audit_can_read_a_changed_path(tree, diff, want):
    assert W.reached(LA, sorted(FILES), diff) == want


def test_every_package_whose_key_a_change_moves_is_reached(tree):
    """`reached` must name at least the packages whose pass the change invalidates, or a candidate audits one the mark never
    built: for each file of the tree, the packages whose key moves when its content does are among those reached."""
    files = sorted(FILES)
    before = {p: LA.key(p, files, LA.FLAGS) for p in LA.packages(files)}
    moves = {}
    for f in files:
        old = (tree / f).read_text()
        (tree / f).write_text(old + "\n")                      # other bytes, and still JSON where it was
        moves[f] = {p for p in before if LA.key(p, files, LA.FLAGS) != before[p]}
        (tree / f).write_text(old)
        assert moves[f] <= set(W.reached(LA, files, [f])), f
    assert moves["pkg/a/A.lean"] >= {"pkg/a", "pkg/b"} and moves["pkg/b/B.lean"] == {"pkg/b", "pkg/c"}
    assert moves["tools/verity/lean/check.py"] == set(before) and not moves["docs/note.md"]


def test_the_audit_runs_for_the_reached_packages_alone_in_an_audit_slot_and_packs_the_lean_verdicts_it_stored(tree, tmp_path, monkeypatch):
    cache = Path(LA.COMMON.cache_dir())
    (cache / "lean-audit").mkdir(parents=True)
    (cache / "lean-audit" / "old.json").write_text("{}")                     # stored before the audit: not this audit's
    _diff(monkeypatch, ["pkg/b/B.lean"])
    audit = Audit(stores={"lean-audit/new.json": "{}", "lean-audit/unread/new.json": "{}", "lean-replay/m.json": "{}",
                          "circuit-check/t.json": "{}",                      # another process of the node, meanwhile
                          "lean-artifacts/keys/k.json": "{}"},               # never a verdict (`verdicts.SKIP`)
                  report={"passed": True, "failed": [], "packages": []})
    out, said = tmp_path / "run" / "out" / "lean-audit", []
    assert W.audit_merge(tree, BASE, out, run=audit, log=said.append) is None
    [(argv, cwd)] = audit.calls
    assert cwd == tree
    assert argv == [sys.executable, str(tree / "tools/verity/check/lean_slot.py"), "--pool", "audit", "--",
                    sys.executable, str(tree / "tools/verity/lean/lean_audit.py"), "--out", str(out),
                    "--cache", str(cache / "lean-audit"), "--deps", str(cache / "lean-deps"), "--bundle",
                    "--package", "pkg/b", "--package", "pkg/c"]
    assert _members(out / W.PACK) == ["check/lean-audit/new.json", "check/lean-audit/unread/new.json", "check/lean-replay/m.json"]
    assert V.refusal(out / W.PACK) is None, "a check imports the pack as it imports a recorded check's"
    assert "pkg/b, pkg/c" in said[0] and "3 Lean verdict(s)" in said[-1]


def test_an_audit_whose_passes_are_all_kept_here_waits_for_no_slot(tree, tmp_path, monkeypatch):
    """`lean_audit.py` then only reports each pass as reused, in seconds: in a slot it would wait behind an hour-long audit."""
    cache = Path(LA.COMMON.cache_dir()) / "lean-audit"
    cache.mkdir(parents=True)
    _diff(monkeypatch, ["pkg/c/C.lean"])
    (cache / f"{LA.key('pkg/c', sorted(FILES), LA.FLAGS)}.json").write_text("{}")
    audit = Audit(report={"passed": True, "failed": [], "packages": []})
    assert W.audit_merge(tree, BASE, tmp_path / "out", run=audit, log=lambda s: None) is None
    [(argv, _)] = audit.calls
    assert argv[:2] == [sys.executable, str(tree / "tools/verity/lean/lean_audit.py")] and argv[-2:] == ["--package", "pkg/c"]
    assert _members(tmp_path / "out" / W.PACK) == []


def test_the_audit_takes_the_bundles_its_caller_names(tree, tmp_path, monkeypatch):
    urls = tmp_path / "lean-deps-urls.json"
    monkeypatch.setenv("CHECK_LEAN_DEPS_URLS", str(urls))
    _diff(monkeypatch, ["pkg/c/C.lean"])
    audit = Audit()
    W.audit_merge(tree, BASE, tmp_path / "out", run=audit, log=lambda s: None)
    argv = audit.calls[0][0]
    assert argv[argv.index("--urls") + 1] == str(urls)


def test_a_change_that_reaches_no_package_runs_nothing(tree, tmp_path, monkeypatch):
    _diff(monkeypatch, ["docs/note.md", "tools/verity/lean/merge.py"])
    audit, said, out = Audit(), [], tmp_path / "out"
    assert W.audit_merge(tree, BASE, out, run=audit, log=said.append) is None
    assert audit.calls == [] and not out.exists()
    assert said == [f"lean warm: the change from {BASE[:12]} reaches no audited Lean package: nothing to audit"]


def test_a_failed_audit_names_each_failed_package_and_its_first_failure_and_still_packs_what_passed(tree, tmp_path, monkeypatch):
    _diff(monkeypatch, ["pkg/a/A.lean"])
    report = {"passed": False, "failed": ["pkg/b", "pkg/c"],
              "packages": [{"package": "pkg/a", "failures": []},
                           {"package": "pkg/b", "failures": ["B.lean:3:0: error:\n  unsolved goals", "B.lean:9:0: error"]},
                           {"package": "pkg/c", "failures": []}]}
    audit = Audit(rc=1, stores={"lean-audit/a.json": "{}"}, report=report)
    out = tmp_path / "out"
    assert W.audit_merge(tree, BASE, out, run=audit, log=lambda s: None) == \
        "the Lean audit on the merge failed in 2 package(s): pkg/b: B.lean:3:0: error: unsolved goals (and 1 more); pkg/c"
    assert _members(out / W.PACK) == ["check/lean-audit/a.json"], "pkg/a passed beside the two that failed"


@pytest.mark.parametrize("report, how", [
    (None, " (FileNotFoundError)"), ({"passed": False}, " (KeyError)"), ([], " (TypeError)"),
    ({"failed": ["pkg/c"], "packages": "garbled"}, " (TypeError)"), ({"failed": [], "packages": []}, ""),
])
def test_an_audit_that_died_without_a_usable_report_is_refused_by_its_exit(tree, tmp_path, monkeypatch, report, how):
    _diff(monkeypatch, ["pkg/c/C.lean"])
    why = W.audit_merge(tree, BASE, tmp_path / "out", run=Audit(rc=-9, report=report), log=lambda s: None)
    assert why == f"the Lean audit of 1 package(s) on the merge exited -9 and left no report of what failed{how}: pkg/c"


def test_what_cant_be_read_is_a_refusal_and_never_a_pass(tree, tmp_path, monkeypatch):
    audit = Audit()
    monkeypatch.setattr(W, "changed", lambda t, base: (_ for _ in ()).throw(ValueError("`git diff bbbb` exited 128: bad object")))
    assert W.audit_merge(tree, BASE, tmp_path / "out", run=audit, log=lambda s: None) == "`git diff bbbb` exited 128: bad object"
    monkeypatch.setattr(LA.COMMON, "tracked_files", lambda r=None: None)
    assert "is no git checkout" in W.audit_merge(tree, BASE, tmp_path / "out", run=audit, log=lambda s: None)
    assert audit.calls == []


def test_the_last_line_says_why_within_what_a_caller_quotes(tree, tmp_path, monkeypatch, capsys):
    """The quick tier quotes the last 300 characters of the audit's last line as the mark's refusal: a longer line would reach
    the author without its start, which names what failed."""
    monkeypatch.setattr(W, "ROOT", tree)
    outcomes = iter(["the Lean audit on the merge failed in 1 package(s): pkg/a: " + "x" * 400, None])
    seen = []
    monkeypatch.setattr(W, "audit_merge", lambda t, base, out, log=print: seen.append((t, base, out)) or next(outcomes))
    assert W.main(["--base", BASE, "--out", str(tmp_path / "o")]) == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert len(last) == W.LINE and last.startswith("lean warm: the Lean audit on the merge failed in 1 package(s): pkg/a: x")
    assert last.endswith("...")
    assert W.main(["--base", BASE, "--out", str(tmp_path / "o")]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "lean warm: passed"
    assert seen == [(tree, BASE, tmp_path / "o")] * 2


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
                       cwd=repo, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def test_the_change_from_the_base_is_every_path_whose_content_differs_a_removed_one_too(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    for name in ("kept.lean", "edited.lean", "removed.lean", "moved.lean"):
        (repo / name).write_text(f"-- {name}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "edited.lean").write_text("-- edited\n")
    (repo / "added file.lean").write_text("-- added\n")
    _git(repo, "rm", "-q", "removed.lean")
    _git(repo, "mv", "moved.lean", "moved-to.lean")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")
    assert W.changed(repo, base) == ["added file.lean", "edited.lean", "moved-to.lean", "moved.lean", "removed.lean"]
    assert W.changed(repo, _git(repo, "rev-parse", "HEAD")) == []
    with pytest.raises(ValueError, match="exited 128"):
        W.changed(repo, "0" * 39 + "1")
