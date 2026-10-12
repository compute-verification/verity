"""`tools/verity/lean/lean_changed.py`: what it builds (each package whose audit misses, what it requires first), the tree it keeps
built, dependencies as read-only hard links to the audit's, and a failure only on a Lean error in the tree's own files or on
a guarantee the audit fails on that build."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("check_lean_changed", ROOT / "tools" / "verity" / "lean" / "lean_changed.py")
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)

#: Lake 4.34's output for a module that no longer compiles and for one of a package it requires by path (the trace lines cut)
LAKE_ERRORS = """\
✖ [8/10] Building Aa.Two (202ms)
error: Aa/Two.lean:2:39: Unknown identifier `one_eq_missing`
error: Aa/Two.lean:2:30: unsolved goals
⊢ one = 42
✖ [6/10] Building Pp (170ms)
error: Pp.lean:2:20: Type mismatch
Some required targets logged failures:
- Aa.Two
- Pp
error: build failed
"""


def _tree(tmp: Path, monkeypatch) -> tuple[Path, list[str]]:
    """A checkout with package `a` (a git dependency, and `p` by path), `p`, and `c`, which needs nothing."""
    root = tmp / "repo"
    files = {
        "a/lakefile.toml": "", "a/lean-toolchain": "v1\n", "a/lean-audit.json": "{}", "a/Aa.lean": "import Aa.One\n",
        "a/Aa/One.lean": "def one := 1\n",
        "a/lake-manifest.json": json.dumps({"packages": [{"name": "dep", "type": "git", "rev": "abc"},
                                                         {"name": "p", "type": "path", "dir": "../p"}]}),
        "p/lakefile.toml": "", "p/lean-toolchain": "v1\n", "p/lean-audit.json": "{}", "p/Pp.lean": "def p := 1\n",
        "c/lakefile.toml": "", "c/lean-toolchain": "v1\n", "c/lean-audit.json": "{}", "c/C.lean": "def c := 1\n",
    }
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    for m in (LC, LC.LA):
        monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda r=None: sorted(files))
    monkeypatch.setattr(LC.LA.COMMON, "cores", lambda: 2)
    return root, sorted(files)


def test_it_builds_each_package_whose_audit_misses_those_it_requires_first(tmp_path, monkeypatch):
    _root, files = _tree(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    assert LC.missed(files, cache) == ["c", "p", "a"]
    cache.mkdir()
    (cache / f"{LC.LA.key('c', files, LC.LA.FLAGS)}.json").write_text("{}")
    assert LC.missed(files, cache) == ["p", "a"]
    assert LC.closure("a") == ["a", "p"]


def test_a_spec_is_built_after_the_packages_proving_its_guarantees_and_its_tree_holds_them(tmp_path, monkeypatch):
    """`s`'s guarantees are proved in `sp` (`proved_in`), which requires `s`: `records` reads `sp`'s build for them."""
    root, files = _tree(tmp_path, monkeypatch)
    more = {"s/lakefile.toml": "", "s/lean-audit.json": json.dumps({"proved_in": ["../sp"]}), "s/S.lean": "",
            "sp/lakefile.toml": "", "sp/lean-audit.json": "{}", "sp/Sp.lean": "",
            "sp/lake-manifest.json": json.dumps({"packages": [{"name": "s", "type": "path", "dir": "../s"}]})}
    for f, text in more.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    files = sorted([*files, *more])
    assert LC.missed(files, tmp_path / "cache") == ["c", "p", "a", "sp", "s"]
    assert LC.closure("s") == ["s", "sp"] and LC.closure("sp") == ["sp", "s"]


def test_the_tree_holds_the_tracked_files_and_keeps_each_lake(tmp_path, monkeypatch):
    root, files = _tree(tmp_path, monkeypatch)
    tree = tmp_path / "tree"
    assert LC.sync(["a", "p"], files, tree) == 10
    assert (tree / "a" / "Aa" / "One.lean").read_text() == "def one := 1\n" and not (tree / "c").exists()
    (tree / "a" / ".lake" / "build").mkdir(parents=True)
    (tree / "a" / ".lake" / "build" / "One.olean").write_text("built")
    (tree / "a" / "Gone.lean").write_text("untracked")
    (root / "a" / "Aa" / "One.lean").write_text("def one := 2\n")
    assert LC.sync(["a", "p"], files, tree) == 1, "only what changed is written, so Lake's traces of the rest still hold"
    assert (tree / "a" / "Aa" / "One.lean").read_text() == "def one := 2\n" and not (tree / "a" / "Gone.lean").exists()
    assert (tree / "a" / ".lake" / "build" / "One.olean").read_text() == "built"
    assert LC.sync(["a", "p"], files, tree) == 0


def test_dependencies_are_hard_links_to_the_audits_taken_once_per_manifest_and_toolchain(tmp_path, monkeypatch):
    _root, files = _tree(tmp_path, monkeypatch)
    tree, deps, links = tmp_path / "tree", tmp_path / "deps", {}
    LC.sync(["a", "p"], files, tree)
    assert not LC.link("a", tree, deps, links, log=lambda s: None) and links == {}, "nothing kept here: the audit builds it"
    k = LC.LA.deps_key(tree / "a")
    (deps / k / "dep" / ".lake" / "build").mkdir(parents=True)
    (deps / k / "dep" / ".lake" / "build" / "Dep.olean").write_text("olean")
    assert LC.link("a", tree, deps, links, log=lambda s: None) and links == {"a": k}
    linked = tree / "a" / ".lake" / "packages" / "dep" / ".lake" / "build" / "Dep.olean"
    assert linked.stat().st_ino == (deps / k / "dep" / ".lake" / "build" / "Dep.olean").stat().st_ino
    os.replace(deps / k, tmp_path / "held-by-an-audit")
    assert LC.link("a", tree, deps, links, log=lambda s: None) and linked.read_text() == "olean", "the links outlive a move"
    assert LC.link("p", tree, deps, links, log=lambda s: None), "a package that fetches nothing needs no links"


def test_only_a_lean_error_in_the_trees_own_files_fails_it(tmp_path, monkeypatch):
    _root, files = _tree(tmp_path, monkeypatch)
    text = LAKE_ERRORS.replace("Aa/Two.lean", "Aa/One.lean")
    assert LC.errors(text, "a", files) == ["error: Aa/One.lean:2:39: Unknown identifier `one_eq_missing`",
                                           "error: Aa/One.lean:2:30: unsolved goals", "error: Pp.lean:2:20: Type mismatch"]
    assert LC.errors(text, "c", files) == []
    for other in ("error: Mathlib/Order/Basic.lean:3:1: unknown constant", "error: Lean exited with code 137",
                  "error: ././././Aa.lean: read-only file system", "error: build failed"):
        assert LC.errors(other, "a", files) == [], other


FAKE_LAKE = r"""#!/usr/bin/env bash
# `lake build` as Lake reports it: a Lean error on a line `BROKEN`, a failure of its own on `INFRA`, else a build; and
# `lake env lean --run Facts.lean CONFIG` writing the facts and canonical forms in the file $FACTS, failing without one
set -u
echo "$*" >> "$LOG.args"
if [ "$1" = env ]; then
  [ -f "${FACTS:-}" ] || { echo "error: unknown module prefix 'C' (a root that was not built?)"; exit 1; }
  PYTHON -c 'import json, sys
config, facts = (json.load(open(p)) for p in sys.argv[1:])
json.dump(facts["facts"], open(config["out"], "w"))
open(config["canon"], "w").write("".join(json.dumps(r) + "\n" for r in facts["canon"]))' "${@: -1}" "$FACTS"
  exit
fi
mkdir -p .lake/build && echo "$PWD" >> "$LOG"
if grep -l INFRA -r --include='*.lean' . >/dev/null; then echo "error: could not resolve dependency"; exit 1; fi
f=$(grep -l BROKEN -r --include='*.lean' --exclude-dir=.lake . | head -1)
[ -z "$f" ] && { echo "Build completed successfully"; exit 0; }
echo "✖ [1/1] Building x"; echo "error: ${f#./}:1:0: unknown identifier 'gone'"; echo "error: build failed"; exit 1
"""


def _fake_sandbox(tmp: Path, monkeypatch) -> tuple[Path, Path]:
    """`lake` as `FAKE_LAKE`, in a sandbox that only runs the command (the real one: `test_the_sandbox_keeps_...`)."""
    lake, log = tmp / "bin" / "lake", tmp / "lake.log"
    lake.parent.mkdir()
    lake.write_text(FAKE_LAKE.replace("PYTHON", sys.executable))
    lake.chmod(0o755)
    (tmp / "sandbox.sh").write_text('while [ "$1" != -- ]; do shift; done; shift; exec "$@"\n')
    monkeypatch.setattr(LC, "SANDBOX", tmp / "sandbox.sh")
    monkeypatch.setattr(LC, "UNSHARE", ["env"])
    monkeypatch.setattr(LC, "READ_ONLY", 'while [ "$1" != -- ]; do shift; done; shift; exec "$@"')
    monkeypatch.setattr(LC, "sandboxed", lambda: True)
    monkeypatch.setenv("PATH", f"{lake.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("LOG", str(log))
    return log, lake


def test_a_module_that_does_not_compile_fails_it_and_anything_else_skips_the_package(tmp_path, monkeypatch, capsys):
    root, files = _tree(tmp_path, monkeypatch)
    log, _ = _fake_sandbox(tmp_path, monkeypatch)
    cache, deps, tree = tmp_path / "cache", tmp_path / "deps", tmp_path / "tree"
    k = LC.LA.deps_key(root / "a")
    (deps / k / "dep").mkdir(parents=True)
    assert LC.run(cache, deps, tree) == 0
    assert [Path(p).name for p in log.read_text().split()] == ["c", "p", "a"]
    assert json.loads((tree / LC.LINKS).read_text()) == {"a": k}
    assert "lean-changed: PASS: built 3 of the 3 package(s) whose audit misses" in capsys.readouterr().out

    (root / "p" / "Pp.lean").write_text("def p := gone -- BROKEN\n")
    log.unlink()
    assert LC.run(cache, deps, tree) == 1
    out = capsys.readouterr().out
    assert "lean-changed: p: FAILED" in out and "error: Pp.lean:1:0: unknown identifier 'gone'" in out
    assert [Path(p).name for p in log.read_text().split()] == ["c", "p"], "the first failure ends it"
    assert "lean-changed: FAIL (p): built 1 of the 3 package(s)" in out

    (root / "p" / "Pp.lean").write_text("def p := 1\n")
    (root / "a" / "Aa.lean").write_text("-- INFRA\n")
    assert LC.run(cache, deps, tree) == 0
    out = capsys.readouterr().out
    assert "lean-changed: a: skipped: `lake build` exited 1" in out
    assert "lean-changed: PASS: built 2 of the 3 package(s)" in out, "a skipped package is not counted as built"
    assert json.loads((tree / LC.LINKS).read_text()) == {} and not (tree / "a" / ".lake" / "packages").exists()


def test_it_skips_where_it_cannot_build_and_when_another_check_holds_the_tree(tmp_path, monkeypatch, capsys):
    import fcntl

    _tree(tmp_path, monkeypatch)
    _fake_sandbox(tmp_path, monkeypatch)
    cache, deps, tree = tmp_path / "cache", tmp_path / "deps", tmp_path / "tree"
    tree.mkdir()
    with open(tree / ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert LC.run(cache, deps, tree) == 0
    assert "skipped: another check on this machine holds the tree" in capsys.readouterr().out
    monkeypatch.setattr(LC, "sandboxed", lambda: False)
    assert LC.run(cache, deps, tree) == 0
    assert "skipped: no build sandbox here" in capsys.readouterr().out


def _facts(definition: str) -> dict:
    """What `Facts.lean` writes for package `c`, whose one guarantee `C.c_eq` reads the definition `c` (canonical form
    ``definition``)."""
    pin = {"kind": "theorem", "axioms": [], "hypotheses": [], "signature": "theorem C.c_eq : c = 1",
           "reads": [{"name": "c", "module": "C", "kind": "defn", "lines": [1, 1],
                      "hash": hashlib.sha256(definition.encode()).hexdigest()[:16]}]}
    return {"facts": {"offenders": [], "declared_axioms": [], "escapes": [], "modules": ["C"], "imports": {"C": []},
                      "pins": {"C.c_eq": pin}, "oleans": [], "axioms": [], "lean": "v1", "declarations": 1},
            "canon": [{"id": "pin:C.c_eq", "canon": "c = 1"}, {"id": "read:c", "canon": definition}]}


def _recorded(facts: dict) -> dict:
    """The policy `check.py --update` writes for package `c` from ``facts`` (`facts_of`'s hashes of the canonical forms)."""
    sha = {r["id"]: hashlib.sha256(r["canon"].encode()).hexdigest() for r in facts["canon"]}
    f = json.loads(json.dumps(facts["facts"]))
    for name, p in f["pins"].items():
        p["type_sha256"] = sha[f"pin:{name}"]
        for r in p["reads"]:
            r["sha256"] = sha[f"read:{r['name']}"]
    return LC.AUDIT.judge({"roots": ["C"], "guarantees": {"C.c_eq": {"owner": "@lean"}}}, f, {}, update=True)[1]


def test_a_guarantee_s_record_a_merge_left_stale_fails_it_and_an_unreadable_build_skips_it(tmp_path, monkeypatch, capsys):
    root, files = _tree(tmp_path, monkeypatch)
    log, _ = _fake_sandbox(tmp_path, monkeypatch)
    cache, deps, tree, facts = tmp_path / "cache", tmp_path / "deps", tmp_path / "tree", tmp_path / "facts.json"
    monkeypatch.setenv("FACTS", str(facts))
    facts.write_text(json.dumps(_facts("def c := 1")))
    (root / "c" / "lean-audit.json").write_text(json.dumps(_recorded(_facts("def c := 1"))))
    assert LC.run(cache, deps, tree) == 0
    out = capsys.readouterr().out
    assert "lean-changed: c: its guarantees match their records" in out
    assert "lean-changed: PASS: built 2 of the 3 package(s) whose audit misses; compared the guarantees of 1" in out
    calls = Path(f"{log}.args").read_text().splitlines()
    assert calls[:2] == ["build", "build +C"], "a package with guarantees has its roots built, as the audit builds them"
    assert calls[2].split()[:4] == ["env", "lean", "--run", str(tree / "c" / ".lake" / "lean-changed-facts" / "Facts.lean")]
    assert (tree / "c" / ".lake" / "lean-changed-facts" / "Facts.lean").read_bytes() == (
        LC.AUDIT.TOOL / "Facts.lean").read_bytes(), "the audit's own `Facts.lean`, copied where the sandbox shows it"
    assert not (tree / "p" / ".lake" / "lean-changed-facts").exists(), "a package without guarantees is only built"

    facts.write_text(json.dumps(_facts("def c := 2")))
    assert LC.run(cache, deps, tree) == 1, "the definition the guarantee reads changed and its record did not"
    out = capsys.readouterr().out
    assert "lean-changed: c: FAILED" in out and "guarantees: definitions in C that C.c_eq read changed (changed: c)" in out
    assert "lean-changed: FAIL (c)" in out

    facts.unlink()
    assert LC.run(cache, deps, tree) == 0
    out = capsys.readouterr().out
    assert "lean-changed: c: its guarantees are not compared here: `Facts.lean` failed (exit 1" in out
    assert "lean-changed: PASS: built 2 of the 3 package(s) whose audit misses; compared the guarantees of 0" in out


def test_only_the_guarantees_failures_fail_it(tmp_path, monkeypatch):
    """The audit's other failures (an orphan file, an unlisted escape, a sorry outside the guarantees) are the audit's to
    report."""
    _tree(tmp_path, monkeypatch)
    _, lake = _fake_sandbox(tmp_path, monkeypatch)
    tree, facts = tmp_path / "tree", tmp_path / "facts.json"
    monkeypatch.setenv("FACTS", str(facts))
    doc = _facts("def c := 1")
    policy = _recorded(doc)
    doc["facts"]["escapes"] = [{"name": "C.io", "why": ["opaque"], "module": "C"}]
    doc["facts"]["offenders"] = [{"name": "C.other", "axioms": ["sorryAx"]}]
    facts.write_text(json.dumps(doc))
    (tree / "c").mkdir(parents=True)
    (tree / "c" / "lean-audit.json").write_text(json.dumps(policy))
    (tree / "c" / "Orphan.lean").write_text("def o := 1\n")
    assert LC.records("c", tree, str(lake)) == []
    (tree / "c" / "lean-audit.json").write_text(json.dumps(policy | {"guarantees": {**policy["guarantees"], "C.gone": {"owner": "@lean"}}}))
    assert LC.records("c", tree, str(lake)) == ["guarantees: C.gone is missing"]


def test_the_facts_are_read_in_the_sandbox(tmp_path, monkeypatch):
    """`Facts.lean` runs as `lake env` in the package, whose `.lake/packages` are hard links to the dependencies the audit
    keeps: it reads them there, and cannot write them. It runs from a copy in the package's `.lake`, since the sandbox's
    private /tmp hides a checkout there."""
    if not LC.sandboxed():
        pytest.skip("no unprivileged user namespaces here (`unshare`); lean_changed skips its builds where there are none")
    _tree(tmp_path, monkeypatch)
    tree = tmp_path / "tree"
    pkg = tree / "c"
    store = tmp_path / "deps" / "k"
    (store / "dep").mkdir(parents=True)
    (store / "dep" / "Dep.olean").write_text("kept")
    (pkg / ".lake" / "packages" / "dep").mkdir(parents=True)
    os.link(store / "dep" / "Dep.olean", pkg / ".lake" / "packages" / "dep" / "Dep.olean")
    doc = _facts("def c := 1")
    (pkg / "lean-audit.json").write_text(json.dumps(_recorded(doc)))
    (tmp_path / "facts.json").write_text(json.dumps(doc))
    lake = tmp_path / "lake"
    lake.write_text(f"#!/usr/bin/env bash\necho tampered > .lake/packages/dep/Dep.olean 2>/dev/null && exit 3\n"
                    f'[ -r "$4" ] && [[ "$4" == {pkg / ".lake"}/* ]] || exit 4\n'
                    f"{sys.executable} -c 'import json, sys\nconfig, facts = (json.load(open(p)) for p in sys.argv[1:])\n"
                    f"json.dump(facts[\"facts\"], open(config[\"out\"], \"w\"))\nopen(config[\"canon\"], \"w\").write("
                    f"\"\".join(json.dumps(r) + \"\\n\" for r in facts[\"canon\"]))' \"${{@: -1}}\" {tmp_path / 'facts.json'}\n")
    lake.chmod(0o755)
    assert LC.records("c", tree, str(lake)) == []
    assert (store / "dep" / "Dep.olean").read_text() == "kept"
    assert (pkg / ".lake" / "lean-changed-facts" / "facts.json").is_file()


def test_the_sandbox_keeps_the_dependencies_read_only_and_the_build_writable(tmp_path):
    """The audit's own sandbox (`tools/verity/lean/sandbox.sh`): what this build may write is its `.lake` but not the hard links in
    `.lake/packages`, which share their files with the dependencies the audit keeps."""
    if not LC.sandboxed():
        pytest.skip("no unprivileged user namespaces here (`unshare`); lean_changed skips its builds where there are none")
    pkg = tmp_path / "tree" / "a"
    store = tmp_path / "deps" / "k"
    (store / "dep").mkdir(parents=True)
    (store / "dep" / "Dep.olean").write_text("kept")
    (pkg / ".lake" / "packages" / "dep").mkdir(parents=True)
    os.link(store / "dep" / "Dep.olean", pkg / ".lake" / "packages" / "dep" / "Dep.olean")
    lake = tmp_path / "lake"
    lake.write_text("#!/usr/bin/env bash\nmkdir -p .lake/build && echo built > .lake/build/A.olean\n"
                    "echo tampered > .lake/packages/dep/Dep.olean 2>/dev/null && echo WROTE-DEPS\nexit 0\n")
    lake.chmod(0o755)
    rc, text = LC.build("a", tmp_path / "tree", str(lake))
    assert rc == 0 and "WROTE-DEPS" not in text, text
    assert (pkg / ".lake" / "build" / "A.olean").read_text() == "built\n"
    assert (store / "dep" / "Dep.olean").read_text() == "kept"
