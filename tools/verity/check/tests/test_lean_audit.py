"""`tools/verity/lean/lean_audit.py`: audit verdicts reused only while every input is unchanged, the hermetic scratch tree as the read
guard, and dependencies kept warm across runs."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("check_lean_audit", ROOT / "tools" / "verity" / "lean" / "lean_audit.py")
LA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LA)
spec = importlib.util.spec_from_file_location("check_for_test_lean_audit", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)

#: a stand-in for tools/verity/lean/check.py --build PKG... [--replay-cache D --artifacts D] --out O, run in the scratch tree: each named package
#: fails when a file its policy names under "reads" is missing (an input the key would not cover) or a package it requires by
#: path is not in the tree, fetches its git dependencies into .lake/packages, and notes what ran
FAKE_AUDIT = r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
root, out = Path.cwd(), Path(a[a.index("--out") + 1])
named = [Path(x) for i, x in enumerate(a) if not x.startswith("--") and not (i and a[i - 1] in ("--out", "--repo", "--replay-cache", "--artifacts"))]
out.mkdir(parents=True, exist_ok=True)
(out / "repo.txt").write_text(a[a.index("--repo") + 1] if "--repo" in a else "")
reps, failed = [], []
for pkg in named:
    pol = pkg / "lean-audit.json"
    missing = [r for r in json.loads(pol.read_text()).get("reads", []) if not (root / r).exists()]
    missing += [d["dir"] for d in json.loads((pkg / "lake-manifest.json").read_text())["packages"]
                if d["type"] == "path" and not (pkg / d["dir"] / "lean-audit.json").exists()]
    deps = pkg / ".lake" / "packages" / "dep"
    warm = (deps / "built").exists()
    if json.loads((pkg / "lake-manifest.json").read_text())["packages"]:
        deps.mkdir(parents=True, exist_ok=True)
        (deps / "built").write_text("x")
    with open(out / "ran.txt", "a") as f:
        f.write(f"{pkg.relative_to(root)} {'warm' if warm else 'cold'}\n")
    lib = pkg / ".lake" / "build" / "lib" / "lean"
    with open(out / "builds.txt", "a") as f:
        f.write(f"{pkg.relative_to(root)} {'warm' if lib.is_dir() else 'cold'}\n")
    lib.mkdir(parents=True, exist_ok=True)
    for src in pkg.glob("*.lean"):
        (lib / (src.stem + ".olean")).write_text("built")
    reps.append({"package": str(pkg), "failures": [f"reads {m}" for m in missing]})
    if missing:
        failed.append(str(pkg))
(out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "packages": reps}))
sys.exit(1 if failed else 0)
'''


def _tree(tmp: Path, monkeypatch) -> Path:
    root = tmp / "repo"
    files = {
        "tools/verity/lean/check.py": FAKE_AUDIT, "tools/verity/lean/README.md": "", "tools/verity/lean/tests/fixtures/control.lean": "",
        "tools/verity/lean/lean_audit.py": "", "tools/verity/lean/lean-deps.json": "{}", "tools/verity/check/common.py": "", "tools/verity/check/check.py": "",
        "pkg/a/lakefile.toml": "", "pkg/a/lean-toolchain": "v1\n", "pkg/a/A.lean": "theorem a : True := trivial\n",
        "pkg/a/lake-manifest.json": json.dumps({"packages": [{"name": "dep", "type": "git", "rev": "abc"}]}),
        "pkg/a/lean-audit.json": "{}",
        "pkg/b/lakefile.toml": "", "pkg/b/lean-toolchain": "v1\n", "pkg/b/B.lean": "",
        "pkg/b/lake-manifest.json": json.dumps({"packages": [{"name": "a", "type": "path", "dir": "../a"}]}),
        "pkg/b/lean-audit.json": json.dumps({"reads": ["docs/outside.txt"]}),
        "docs/outside.txt": "read by b's audit, but no input of it",
    }
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    monkeypatch.setattr(LA, "ROOT", root)
    monkeypatch.setattr(LA.COMMON, "tracked_files", lambda r=None: sorted(files))
    monkeypatch.setattr(LA.COMMON, "cores", lambda: 2)
    return root


def _run(tmp: Path, n: int) -> tuple[int, list[str], dict]:
    out = tmp / f"out{n}"
    rc = LA.run(out, tmp / "cache" / "lean-audit", tmp / "cache" / "lean-deps", log=lambda s: None)
    ran = (out / "ran.txt").read_text().split("\n")[:-1] if (out / "ran.txt").exists() else []
    return rc, ran, json.loads((out / "audit.json").read_text())


def test_verdicts_are_reused_only_while_every_input_is_unchanged(tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)
    # b reads a file outside its inputs: in the scratch tree it is not there, so b fails and nothing is stored for it
    rc, ran, doc = _run(tmp_path, 1)
    assert rc == 1 and sorted(ran) == ["pkg/a cold", "pkg/b cold"] and doc["failed"] == ["pkg/b"] and doc["cached"] == {}
    (root / "pkg/b/lean-audit.json").write_text("{}")
    rc, ran, doc = _run(tmp_path, 2)                   # a reused; only b audited, with a, its path dependency, in its tree
    assert rc == 0 and ran == ["pkg/b cold"] and list(doc["cached"]) == ["pkg/a"]
    rc, ran, doc = _run(tmp_path, 3)                   # nothing changed: nothing runs
    assert rc == 0 and ran == [] and sorted(doc["cached"]) == ["pkg/a", "pkg/b"] and doc["passed"]
    (root / "pkg/a/A.lean").write_text("theorem a : True := by trivial\n")
    rc, ran, doc = _run(tmp_path, 4)                   # a changed: a, and b which requires it by path, both run again
    assert rc == 0 and sorted(ran) == ["pkg/a warm", "pkg/b cold"] and doc["cached"] == {}
    (root / "tools/verity/lean/check.py").write_text(FAKE_AUDIT + "\n# the auditor changed\n")
    assert sorted(_run(tmp_path, 5)[1]) == ["pkg/a warm", "pkg/b cold"]     # the auditor is an input of every package


def _move(root: Path, monkeypatch, old: str, new: str) -> None:
    """`git mv old new`: the directory, and the tracked files' paths."""
    files = LA.COMMON.tracked_files()
    (root / new).parent.mkdir(parents=True, exist_ok=True)
    os.rename(root / old, root / new)
    moved = sorted(new + f[len(old):] if f.startswith(old + "/") else f for f in files)
    monkeypatch.setattr(LA.COMMON, "tracked_files", lambda r=None: moved)


def _requires_a_at(root: Path, b: str, path: str) -> None:
    (root / b / "lakefile.toml").write_text(f'name = "b"\n\n[[require]]\nname = "a"\npath = "{path}"\n')
    (root / b / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": "a", "type": "path", "dir": path}]}))


def test_a_moved_package_reuses_its_pass(tmp_path, monkeypatch):
    """The key holds each file's path relative to its package, and a package required by path as its content, so a move with
    the same content is no input change. Keyed by path, the migration's every move audited the moved packages cold
    (afacde975's check, 60.5 min, lean-audit 59.5 of them)."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    _requires_a_at(root, "pkg/b", "../a")
    assert _run(tmp_path, 1)[0] == 0
    _move(root, monkeypatch, "pkg", "lib/lean")                     # both, side by side as before
    rc, ran, doc = _run(tmp_path, 2)
    assert rc == 0 and ran == [] and sorted(doc["cached"]) == ["lib/lean/a", "lib/lean/b"]
    assert sorted(r["package"] for r in doc["packages"]) == ["lib/lean/a", "lib/lean/b"]
    _move(root, monkeypatch, "lib/lean/a", "core/lean/a")           # a alone, and b's lakefile and manifest follow it
    _requires_a_at(root, "lib/lean/b", "../../../core/lean/a")
    rc, ran, doc = _run(tmp_path, 3)
    assert rc == 0 and ran == [] and sorted(doc["cached"]) == ["core/lean/a", "lib/lean/b"]
    rc, lines = LA.explain("lib/lean/b", tmp_path / "cache" / "lean-audit")
    assert rc == 0 and "PASS reused" in lines[0]
    (root / "core/lean/a/A.lean").write_text("theorem a : True := by trivial\n")
    rc, ran, doc = _run(tmp_path, 4)                                # then one module changes: a and b, which requires it
    assert rc == 0 and sorted(ran) == ["core/lean/a warm", "lib/lean/b cold"] and doc["cached"] == {}


def test_one_changed_byte_in_any_input_misses(tmp_path, monkeypatch):
    """Every file of the package, of a package it requires by path, and of the auditor moves the key; so does pointing the
    dependent at another directory (a package of other content there)."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/a/Sub").mkdir()
    (root / "pkg/a/Sub/C.lean").write_text("def c := 1\n")
    files = sorted([*LA.COMMON.tracked_files(), "pkg/a/Sub/C.lean"])
    monkeypatch.setattr(LA.COMMON, "tracked_files", lambda r=None: files)
    _requires_a_at(root, "pkg/b", "../a")
    keys = lambda: {p: LA.key(p, files, LA.FLAGS) for p in ("pkg/a", "pkg/b")}
    was = keys()
    for f in LA.inputs("pkg/b", files):
        before = (root / f).read_bytes()
        (root / f).write_bytes(before + b" ")                       # one more byte, still a manifest or a lakefile
        now = keys()
        assert now["pkg/b"] != was["pkg/b"], f
        assert (now["pkg/a"] != was["pkg/a"]) == (f.startswith("pkg/a/") or not f.startswith("pkg/")), f
        (root / f).write_bytes(before)
    assert keys() == was
    for f in ("lakefile.toml", "lean-toolchain", "lake-manifest.json", "lean-audit.json", "A.lean"):
        (root / "pkg/c" / f).parent.mkdir(parents=True, exist_ok=True)
        (root / "pkg/c" / f).write_bytes((root / "pkg/a" / f).read_bytes())
    files += [f"pkg/c/{f}" for f in ("lakefile.toml", "lean-toolchain", "lake-manifest.json", "lean-audit.json", "A.lean")]
    _requires_a_at(root, "pkg/b", "../c")                           # a copy of a without Sub/C.lean
    assert keys()["pkg/b"] != was["pkg/b"]
    (root / "pkg/c/Sub").mkdir()
    (root / "pkg/c/Sub/C.lean").write_text("def c := 1\n")
    files.append("pkg/c/Sub/C.lean")
    assert keys()["pkg/b"] == was["pkg/b"]                          # the same content there: the same key
    shutil.copytree(root / "pkg/c", root / "pkg/d")
    files += [f"pkg/d/{f[6:]}" for f in files if f.startswith("pkg/c/")]
    single = lambda d: (root / "pkg/b/lakefile.toml").write_text(f"name = \"b\"\n\n[[require]]\nname = \"a\"\npath = '../{d}'\n")
    single("c")
    k = keys()["pkg/b"]
    _requires_a_at(root, "pkg/b", "../d")
    single("d")
    assert keys()["pkg/b"] != k                                     # a path spelled as the key can't name by content stays a path


def test_a_package_that_may_read_beyond_its_directory_is_keyed_by_its_path(tmp_path, monkeypatch):
    """A relative path out of the package, other than to a package it requires, finds other files once the package moves, and
    a pass reused by content alone would not see it. Lake's and the audit's own files left with a `..`, or a module with an
    `include_str`, keep the package's path in its key."""
    root = _tree(tmp_path, monkeypatch)
    _requires_a_at(root, "pkg/b", "../a")
    files = LA.COMMON.tracked_files()
    assert LA.WHERE not in LA.content("pkg/b", files)                # the paths to a are content, not paths
    for f, text in (("lakefile.toml", 'name = "a"\nsrcDir = "../shared"\n'), ("A.lean", 'def s := include_str "../x.txt"\n')):
        was = (root / "pkg/a" / f).read_text()
        (root / "pkg/a" / f).write_text(text)
        k = LA.key("pkg/a", LA.COMMON.tracked_files(), LA.FLAGS)
        assert LA.content("pkg/a", LA.COMMON.tracked_files())[LA.WHERE] == "pkg/a"
        _move(root, monkeypatch, "pkg/a", "other/a")
        assert LA.key("other/a", LA.COMMON.tracked_files(), LA.FLAGS) != k
        _move(root, monkeypatch, "other/a", "pkg/a")
        (root / "pkg/a" / f).write_text(was)


def test_the_controls_and_the_docs_are_no_input(tmp_path, monkeypatch):
    """b201 (r20261004-150518-b201) audited all 8 packages again, 69 min, partly on edits to `tools/verity/lean/README.md` and
    `tools/verity/lean/tests/`, which no audit reads, and so is the pin file `lean-deps.json` (preflight's, read before the
    audit). They stay out of the scratch tree too, so an audit that read one would fail."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    assert _run(tmp_path, 1)[0] == 0
    (root / "tools/verity/lean/README.md").write_text("reworded\n")
    (root / "tools/verity/lean/lean-deps.json").write_text('{"deps": {"k": {}}}')
    (root / "tools/verity/lean/tests/fixtures/control.lean").write_text("theorem c : True := trivial\n")
    rc, ran, doc = _run(tmp_path, 2)
    assert rc == 0 and ran == [] and sorted(doc["cached"]) == ["pkg/a", "pkg/b"]
    (root / "pkg/b/lean-audit.json").write_text(json.dumps({"reads": ["tools/verity/lean/tests/fixtures/control.lean"]}))
    rc, _ran, doc = _run(tmp_path, 3)
    assert rc == 1 and doc["failed"] == ["pkg/b"]


def test_a_passing_audit_keeps_its_build(tmp_path, monkeypatch):
    """b201 rebuilt all 561 of `soundness`'s modules (2,508 s) because the auditor changed, though no `.olean` depends on it."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    store = tmp_path / "cache" / "lean-builds"
    builds = lambda n: sorted((tmp_path / f"out{n}" / "builds.txt").read_text().split("\n")[:-1])
    assert _run(tmp_path, 1)[0] == 0 and builds(1) == ["pkg/a cold", "pkg/b cold"]
    ka, kb = LA.build_key("pkg/a", root), LA.build_key("pkg/b", root)
    assert (store / ka / "lib/lean/A.olean").is_file() and (store / kb / "lib/lean/B.olean").is_file()
    (store / ka / "lib/lean/Gone.olean").write_text("its source was deleted")
    (root / "tools/verity/lean/check.py").write_text(FAKE_AUDIT + "\n# the auditor changed\n")
    assert _run(tmp_path, 2)[0] == 0 and builds(2) == ["pkg/a warm", "pkg/b warm"]
    assert (store / ka / "lib/lean/A.olean").is_file() and not (store / ka / "lib/lean/Gone.olean").exists()
    (root / "pkg/b/B.lean").write_text("-- b changed\n")
    logged = []
    rc = LA.run(tmp_path / "out3", tmp_path / "cache" / "lean-audit", tmp_path / "cache" / "lean-deps", log=logged.append)
    assert rc == 0 and builds(3) == ["pkg/b warm"]                    # a reused; its build copied in as b's path dependency
    assert any(ln.startswith("lean-audit: pkg/a: build warm") and "a copy" in ln for ln in logged) and (store / ka).is_dir()
    (root / "pkg/a/lean-audit.json").write_text(json.dumps({"reads": ["nowhere.txt"]}))
    assert _run(tmp_path, 4)[0] == 1 and not (store / ka).exists()     # a failed: its build is dropped, not kept
    (root / "pkg/a/lean-toolchain").write_text("v2\n")
    assert LA.build_key("pkg/a", root) != ka


def _built(pkg: Path, mod: str, outputs: dict[str, str]) -> None:
    """A module of ``pkg``'s build as Lake leaves it: each output beside its `<file>.hash`, and the trace naming them."""
    build = pkg / ".lake" / "build"
    for ext, h in outputs.items():
        f = build / ("ir" if ext == "c" else "lib/lean") / f"{mod}.{ext}"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"{mod}.{ext} {h}")
        (f.parent / f"{f.name}.hash").write_text(h)
    trace = build / "lib/lean" / f"{mod}.trace"
    trace.write_text(json.dumps({"depHash": hashlib.sha256(mod.encode()).hexdigest()[:16], "outputs": {
        "o": [f"{outputs['olean']}.olean"], "m": False, "i": f"{outputs['ilean']}.ilean", "c": f"{outputs['c']}.c"}}))


def test_the_artifact_cache_takes_a_module_only_with_every_output_it_names(tmp_path):
    """Lake restores a module from the cache by its input hash; an entry naming an output the cache lacks is a miss. A module
    whose build holds an output under another hash than its trace names (an edit after the build) stays out."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "lake-manifest.json").write_text(json.dumps({"name": "p", "packages": []}))
    _built(pkg, "P/A", {"olean": "a" * 16, "ilean": "b" * 16, "c": "c" * 16})
    _built(pkg, "P/B", {"olean": "d" * 16, "ilean": "e" * 16, "c": "f" * 16})
    (pkg / ".lake/build/lib/lean/P/B.olean.hash").write_text("0" * 16)
    arts = LA.LakeArtifacts(tmp_path / "store")
    assert arts.put({pkg}) == 1 and arts.put({pkg}) == 0
    store = tmp_path / "store"
    assert sorted(f.name for f in (store / "artifacts").iterdir()) == ["a" * 16 + ".olean", "b" * 16 + ".ilean", "c" * 16 + ".c"]
    entry, = (store / "outputs" / "p").iterdir()
    assert entry.name == hashlib.sha256(b"P/A").hexdigest()[:16] + ".json"
    assert json.loads(entry.read_text())["o"] == ["a" * 16 + ".olean"]
    assert (store / "artifacts" / ("a" * 16 + ".olean")).read_text() == "P/A.olean " + "a" * 16


def test_the_artifact_cache_drops_its_oldest_past_the_cap_counting_only_what_it_alone_holds(tmp_path):
    store = tmp_path / "store"
    (store / "artifacts").mkdir(parents=True)
    (store / "outputs" / "p").mkdir(parents=True)
    kept = tmp_path / "kept.olean"
    kept.write_bytes(b"k" * 100)
    os.link(kept, store / "artifacts" / "shared.olean")              # a kept build holds it too: no disk of the cache's own
    for i in range(4):
        f, e = store / "artifacts" / f"{i}.olean", store / "outputs" / "p" / f"{i}.json"
        f.write_bytes(b"x" * 10)
        e.write_text("{}")
        os.utime(f, (i + 10, i + 10))
        os.utime(e, (i + 10, i + 10))
    os.utime(store / "artifacts" / "shared.olean", (1, 1))
    assert LA.LakeArtifacts(store, cap=40).prune() == 0
    assert LA.LakeArtifacts(store, cap=30).prune() == 3               # 40 > 30: the shared one goes first, then 0 and 1
    assert sorted(f.name for f in (store / "artifacts").iterdir()) == ["2.olean", "3.olean"] and kept.is_file()
    assert sorted(f.name for f in (store / "outputs" / "p").iterdir()) == ["2.json", "3.json"]


def _bundle_of(tmp: Path, pkg: Path) -> tuple[Path, str]:
    arts = LA.LakeArtifacts(tmp / "node-a")
    arts.put({pkg})
    dest = tmp / "lean-artifacts.tar.gz"
    assert arts.bundle(dest) == len(arts.added) > 0
    return dest, hashlib.sha256(dest.read_bytes()).hexdigest()


def test_a_bundle_carries_the_module_versions_one_node_added_to_another_and_nothing_else(tmp_path):
    """Kept builds and the artifact cache are per node, and the trains spread over three machines (infra, 12 Oct: share the
    cache, don't pin). A node takes a recorded check's bundle once, checked against its sha256, and never replaces a file."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "lake-manifest.json").write_text(json.dumps({"name": "p", "packages": []}))
    _built(pkg, "P/A", {"olean": "a" * 16, "ilean": "b" * 16, "c": "c" * 16})
    _built(pkg, "P/B", {"olean": "d" * 16, "ilean": "b" * 16, "c": "f" * 16})
    dest, h = _bundle_of(tmp_path, pkg)
    b = LA.LakeArtifacts(tmp_path / "node-b")
    (tmp_path / "node-b" / "artifacts").mkdir(parents=True)
    (tmp_path / "node-b" / "artifacts" / ("a" * 16 + ".olean")).write_text("node b's own")
    logged = []
    assert b.take({LA.SHARED + h: dest.as_uri(), LA.SHARED + "0" * 64: dest.as_uri(), "k": "https://deps"}, logged.append) == 2
    assert any("is not the one named" in ln for ln in logged)
    got = tmp_path / "node-b"
    assert sorted(f.relative_to(got).as_posix() for f in got.rglob("*") if f.is_file() and f.parent.name != "taken"
                  and f.name != "lock") == sorted(f"artifacts/{n}" for n in
                  ["a" * 16 + ".olean", "b" * 16 + ".ilean", "c" * 16 + ".c", "d" * 16 + ".olean", "f" * 16 + ".c"]) + sorted(
                  f"outputs/p/{hashlib.sha256(m.encode()).hexdigest()[:16]}.json" for m in ("P/A", "P/B"))
    assert (got / "artifacts" / ("a" * 16 + ".olean")).read_text() == "node b's own"
    assert not os.stat(got / "artifacts" / ("d" * 16 + ".olean")).st_mode & 0o222
    assert b.take({LA.SHARED + h: dest.as_uri()}, logged.append) == 0 and logged[-1].startswith("lean-audit: 2 module")
    junk = tmp_path / "junk.tar.gz"
    junk.write_bytes(b"not a tar")
    j = hashlib.sha256(junk.read_bytes()).hexdigest()
    seen = len(logged)
    assert b.take({LA.SHARED + j: junk.as_uri()}, logged.append) == 0 and "could not be unpacked" in logged[seen]
    assert b.take({LA.SHARED + j: junk.as_uri(), LA.SHARED + "0" * 64: dest.as_uri()}, logged.append) == 0
    assert len(logged) == seen + 2                         # bytes that failed once are never fetched again


def test_a_bundle_member_outside_lake_s_layout_or_an_entry_without_its_outputs_is_left_out(tmp_path):
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as t:
        for name, data in [("../escape", b"x"), ("artifacts/../../x.olean", b"x"), ("artifacts/" + "e" * 16 + ".olean", b"ok"),
                           ("outputs/p/" + "1" * 16 + ".json", json.dumps({"o": ["e" * 16 + ".olean"]}).encode()),
                           ("outputs/p/" + "2" * 16 + ".json", json.dumps({"o": ["9" * 16 + ".olean"]}).encode()),
                           ("outputs/p/" + "3" * 16 + ".json", b"not json"),
                           ("outputs/../" + "4" * 16 + ".json", json.dumps({"o": ["e" * 16 + ".olean"]}).encode()),
                           ("outputs/./" + "5" * 16 + ".json", json.dumps({"o": ["e" * 16 + ".olean"]}).encode())]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    h = hashlib.sha256(bad.read_bytes()).hexdigest()
    store = tmp_path / "node"
    assert LA.LakeArtifacts(store).take({LA.SHARED + h: bad.as_uri()}, lambda s: None) == 1
    assert sorted(f.relative_to(store).as_posix() for f in store.rglob("*") if f.is_file() and f.name != "lock"
                  and f.parent.name != "taken") == ["artifacts/" + "e" * 16 + ".olean", "outputs/p/" + "1" * 16 + ".json"]
    assert not (tmp_path / "escape").exists() and not (tmp_path / "x.olean").exists()


def test_the_launcher_sends_the_newest_recorded_checks_bundles(tmp_path):
    runs = {"r3": [("lean-audit/lean-artifacts.tar.gz", "c" * 64, True), ("verdicts.tar.gz", "v" * 64, True)],
            "r2": [("lean-audit/lean-artifacts.tar.gz", "b" * 64, False)],          # not on the remote: no node could fetch it
            "r1": [("lean-audit/lean-artifacts.tar.gz", "a" * 64, True), ("lean-audit/lean-artifacts-2.tar.gz", "d" * 64, True),
                   ("lean-audit/x/lean-artifacts.tar.gz", "e" * 64, True)]}
    store = types.SimpleNamespace(
        remote=None, index=types.SimpleNamespace(sql=lambda q, a: [(r,) for r in runs]),
        get_attempt=lambda r: {"outputs": {"run_files": f"art:{r}"}},
        listing=lambda art: [{"path": p, "sha256": h, "remote": on} for p, h, on in runs[art[4:]]])
    assert LA.shared_bundles(store) == ["c" * 64, "a" * 64, "d" * 64]
    assert LA.shared_bundles(store, limit=1) == ["c" * 64]


def test_a_package_whose_sources_sit_beside_it_keeps_its_build(tmp_path):
    """`Security/Proofs` has `srcDir = ".."`. Pruning looked for each kept output's source inside the package, found none,
    and deleted the whole kept build on every warm take, so each check's audit rebuilt `Security/Proofs` from scratch."""
    pkg = tmp_path / "Security" / "Proofs"
    (pkg / "Flock").mkdir(parents=True)
    (pkg / "lakefile.toml").write_text('name = "proofs"\nsrcDir = ".."\n\n[[lean_lib]]\nname = "Proofs"\n')
    (pkg / "Flock" / "X.lean").write_text("theorem x : True := trivial\n")
    (tmp_path / "Security" / "Flock").mkdir()
    (tmp_path / "Security" / "Flock" / "Y.lean").write_text("theorem y : True := trivial\n")
    lib, ir = pkg / ".lake/build/lib/lean", pkg / ".lake/build/ir"
    kept = [lib / "Proofs/Flock/X.olean", lib / "Proofs/Flock/X.olean.hash", lib / "Proofs/Flock/X.ilean",
            lib / "Proofs/Flock/X.trace", ir / "Proofs/Flock/X.c", ir / "Proofs/Flock/X.c.hash", lib / "Flock/Y.olean"]
    stale = [lib / "Proofs/Flock/Gone.olean", ir / "Proofs/Flock/Gone.c", lib / "Proofs/Proofs/Flock/X.olean"]
    for f in kept + stale:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("built")
    assert LA.prune(pkg) == len(stale)
    assert all(f.is_file() for f in kept) and not any(f.exists() for f in stale)


def test_a_moved_package_takes_the_build_it_left(tmp_path, monkeypatch):
    """A kept build is keyed by the package's toolchain, manifest and lakefile, not its path or the paths to what it requires:
    after a move, its build comes in warm, and Lake reuses it where the package is now (`.lake/build` survives a move)."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    _requires_a_at(root, "pkg/b", "../a")
    builds = lambda n: sorted((tmp_path / f"out{n}" / "builds.txt").read_text().split("\n")[:-1])
    assert _run(tmp_path, 1)[0] == 0
    kb = LA.build_key("pkg/b", root)
    _move(root, monkeypatch, "pkg/a", "core/a")
    _requires_a_at(root, "pkg/b", "../../core/a")
    assert LA.build_key("pkg/b", root) == kb
    (root / "core/a/A.lean").write_text("theorem a : True := by trivial\n")      # so that both are audited again
    assert _run(tmp_path, 2)[0] == 0 and builds(2) == ["core/a warm", "pkg/b warm"]
    (root / "pkg/b/lakefile.toml").write_text('name = "b2"\n\n[[require]]\nname = "a"\npath = "../../core/a"\n')
    assert LA.build_key("pkg/b", root) != kb


def test_a_cache_reached_through_a_symlink_keeps_every_report(tmp_path, monkeypatch):
    """check.py names each package by its resolved path. With the cache behind a symlink (node 1's `~/.cache/verity-check`),
    the scratch tree's unresolved path matched no report, so a package that passed was failed as unreported
    (r20261001-060430-71e3)."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    (root / "tools/verity/lean/check.py").write_text(FAKE_AUDIT.replace('"package": str(pkg)', '"package": str(pkg.resolve())'))
    (tmp_path / "real-cache").mkdir()
    (tmp_path / "cache-link").symlink_to(tmp_path / "real-cache")
    out = tmp_path / "out"
    rc = LA.run(out, tmp_path / "cache-link" / "lean-audit", tmp_path / "cache-link" / "lean-deps", log=lambda s: None)
    doc = json.loads((out / "audit.json").read_text())
    assert rc == 0 and doc["failed"] == [] and sorted(r["package"] for r in doc["packages"]) == ["pkg/a", "pkg/b"]


def test_a_packages_test_generates_its_data_from_the_checkout_not_the_scratch_tree(tmp_path, monkeypatch):
    """A test's `generate` (`runs`) reads fixtures that are no package's input (PoUW's reads `verity/ml/circuits/silicon/captures/vectors`), so the
    audit, run from the scratch tree, is told where the checkout is."""
    root = _tree(tmp_path, monkeypatch)
    _run(tmp_path, 1)
    assert (tmp_path / "out1" / "repo.txt").read_text() == str(root)


#: runs lean_audit.py's main on the tree `_tree` wrote, as check.py's step does: argv is the module, the root, the tracked files
#: (JSON) and main's arguments
DRIVER = r'''
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("lean_audit_driven", sys.argv[1])
LA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LA)
LA.ROOT, files = Path(sys.argv[2]), json.loads(sys.argv[3])
LA.COMMON.tracked_files = lambda r=None: files
LA.COMMON.cores = lambda: 2
sys.exit(LA.main(sys.argv[4:]))
'''


def test_a_stopped_audit_removes_its_scratch_tree(tmp_path, monkeypatch):
    """A manual stop (`research cancel`) sends SIGTERM to the check's whole process group. The audit died of it without
    running its finally, and left its scratch tree, about 60 GB, in `~/.cache/verity-check`: two of them filled node 2's root
    disk until a check refused to start (3 Oct, r20261003-055703-86e3 and r20261003-030801-94eb)."""
    root = _tree(tmp_path, monkeypatch)
    (root / "tools/verity/lean/check.py").write_text('import sys\nprint("auditing", flush=True)\nsys.stdin.read()\n')
    cache = tmp_path / "cache"
    p = subprocess.Popen([sys.executable, "-c", DRIVER, str(ROOT / "tools/verity/lean/lean_audit.py"), str(root),
                          json.dumps(LA.COMMON.tracked_files()), "--out", str(tmp_path / "out"),
                          "--cache", str(cache / "lean-audit"), "--deps", str(cache / "lean-deps")],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, text=True)
    while "auditing" not in (line := p.stdout.readline()):
        assert line, p.communicate()[1]
    assert len(list(cache.glob("lean-audit-scratch-*"))) == 1
    os.killpg(p.pid, signal.SIGTERM)
    p.communicate()
    assert p.returncode != 0 and list(cache.glob("lean-audit-scratch-*")) == []


def test_a_check_py_edit_keeps_every_verdict_and_a_pack_carries_them_to_another_machine(tmp_path, monkeypatch):
    """`check.py` only launches the audit step, so an edit to it alone keeps every package's key: a pack exported on one machine
    and imported on another through `check.py --verdicts-in` answers for both packages, and nothing is audited again. An edit
    to `common.py`, whose helpers the step runs, misses both."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    files = LA.COMMON.tracked_files()
    keys = {p: LA.key(p, files, LA.FLAGS) for p in LA.packages(files)}
    monkeypatch.setenv("VERITY_TEST_CACHE", str(tmp_path / "tests-cache"))
    one, two = tmp_path / "machine-1", tmp_path / "machine-2"
    monkeypatch.setenv("CHECK_CACHE", str(one))
    assert LA.run(tmp_path / "out1", one / "lean-audit", one / "lean-deps", log=lambda s: None) == 0
    pack = tmp_path / "verdicts.tar.gz"
    assert CHECK.VERDICTS.export(pack) == 2

    (root / "tools/verity/check/check.py").write_text("# check.py changed, and nothing the audit runs\n")
    assert {p: LA.key(p, files, LA.FLAGS) for p in keys} == keys
    monkeypatch.setenv("CHECK_CACHE", str(two))
    assert CHECK.verdicts_in(pack, None)["added"] == 2
    out = tmp_path / "out2"
    assert LA.run(out, two / "lean-audit", two / "lean-deps", log=lambda s: None) == 0
    doc = json.loads((out / "audit.json").read_text())
    assert not (out / "ran.txt").exists() and sorted(doc["cached"]) == ["pkg/a", "pkg/b"] and doc["passed"]

    (root / "tools/verity/check/common.py").write_text("# the helpers the audit step runs changed\n")
    assert not set(keys.values()) & {LA.key(p, files, LA.FLAGS) for p in keys}


def test_explain_names_the_inputs_a_miss_changed(tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    cache = tmp_path / "cache" / "lean-audit"
    assert _run(tmp_path, 1)[0] == 0
    assert LA.explain("pkg/b", cache) == (0, [f"pkg/b: PASS reused ({LA.key('pkg/b', LA.COMMON.tracked_files(), LA.FLAGS)[:12]}): "
                                              "every file its audit reads is unchanged"])
    (root / "tools/verity/check/check.py").write_text("# check.py changed\n")
    assert "PASS reused" in LA.explain("b", cache)[1][0]                          # a unique part of the path names it too
    (root / "pkg/a/A.lean").write_text("theorem a : True := by trivial\n")
    rc, lines = LA.explain("pkg/b", cache)
    assert rc == 0 and lines[0].startswith("pkg/b: missed (") and "differs in 2 input(s):" in lines[0]
    assert lines[1:] == ["  changed: pkg/a/A.lean",                               # b requires a by path
                         "  changed: pkg/b/lake-manifest.json (it names each package it requires by path by that package's content)"]
    assert LA.explain("pkg", cache)[0] == 2                                       # two packages match
    _move(root, monkeypatch, "pkg/b", "elsewhere/b")
    (root / "elsewhere/b/lake-manifest.json").write_text(json.dumps({"packages": [{"name": "a", "type": "path", "dir": "../../pkg/a"}]}))
    (root / "elsewhere/b/B.lean").write_text("-- b changed too\n")
    rc, lines = LA.explain("elsewhere/b", cache)                                  # its passes were audited at pkg/b
    assert rc == 0 and "audited at pkg/b) differs in 3 input(s):" in lines[0] and "  changed: elsewhere/b/B.lean" in lines


def test_every_file_the_step_loads_is_in_its_key():
    """What `lean_audit.py` loads by path is in `inputs`, and `common.py` imports only the standard library, so a key covers
    everything the step runs. Loading `check.py` again would need it back in the key."""
    import ast

    files = LA.COMMON.tracked_files()
    got = LA.inputs("Security/Proofs", files)
    loaded = re.findall(r'_module\(ROOT / "tools" / "verity" / "check" / "([a-z_]+\.py)"', (ROOT / "tools/verity/lean/lean_audit.py").read_text())
    assert loaded == ["common.py"] and all(f"tools/verity/check/{m}" in got for m in loaded)
    assert "tools/verity/check/check.py" not in got
    for node in ast.walk(ast.parse((ROOT / "tools/verity/check/common.py").read_text())):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
        assert all(n.split(".")[0] in sys.stdlib_module_names or n == "__future__" for n in names), names


def test_the_dependencies_are_kept_by_manifest_and_toolchain(tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)
    k = LA.deps_key(root / "pkg/a")
    assert k and LA.deps_key(root / "pkg/b") is None                         # b fetches nothing (a path dependency only)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    _run(tmp_path, 1)
    assert (tmp_path / "cache" / "lean-deps" / k / "dep" / "built").exists()   # moved back after the run
    (root / "pkg/a/lean-audit.json").write_text(json.dumps({"reads": ["nowhere.txt"]}))     # a fails its audit this time
    rc, ran, _doc = _run(tmp_path, 2)
    assert rc == 1 and "pkg/a warm" in ran and not (tmp_path / "cache" / "lean-deps" / k).exists()   # dropped, not kept
    (root / "pkg/a/lean-toolchain").write_text("v2\n")
    assert LA.deps_key(root / "pkg/a") != k


def test_a_path_entry_changes_no_dependencies_key(tmp_path, monkeypatch):
    """`.lake/packages` holds only fetched packages, so requiring another package of the repository by path needs no new
    bundle (#1499's trailing check was refused for want of one), while a changed fetched entry needs one; node 1's daily run
    keeps its copies under the same key."""
    root = _tree(tmp_path, monkeypatch)
    m, k = root / "pkg/a/lake-manifest.json", LA.deps_key(root / "pkg/a")
    doc = json.loads(m.read_text())
    m.write_text(json.dumps({**doc, "packages": [*doc["packages"], {"name": "b", "type": "path", "dir": "../b"}]}, indent=1))
    assert LA.deps_key(root / "pkg/a") == k
    spec = importlib.util.spec_from_file_location("lean_run_daily", ROOT / "tools/verity/lean/run_daily.py")
    daily = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daily)
    assert daily.deps_key(root / "pkg/a") == k
    m.write_text(json.dumps({**doc, "packages": [{**doc["packages"][0], "rev": "abd"}]}))
    assert LA.deps_key(root / "pkg/a") != k


def test_packages_of_one_manifest_and_toolchain_both_build_warm(tmp_path, monkeypatch):
    """POUS and PoUW pin one Mathlib on one toolchain. A run that audits both moves the kept tree into one and a copy of it
    into the other, which would otherwise clone its dependencies from GitHub."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    tracked, twin = LA.COMMON.tracked_files(), ["lakefile.toml", "lean-toolchain", "lake-manifest.json", "lean-audit.json"]
    (root / "pkg/c").mkdir()
    for f in twin:
        (root / "pkg/c" / f).write_bytes((root / "pkg/a" / f).read_bytes())
    monkeypatch.setattr(LA.COMMON, "tracked_files", lambda r=None: sorted([*tracked, *(f"pkg/c/{f}" for f in twin)]))
    assert LA.deps_key(root / "pkg/c") == LA.deps_key(root / "pkg/a")
    rc, ran, _doc = _run(tmp_path, 1)
    assert rc == 0 and sorted(ran) == ["pkg/a cold", "pkg/b cold", "pkg/c cold"]
    (root / "tools/verity/lean/check.py").write_text(FAKE_AUDIT + "\n# the auditor changed: every package runs again\n")
    rc, ran, _doc = _run(tmp_path, 2)
    assert rc == 0 and sorted(ran) == ["pkg/a warm", "pkg/b cold", "pkg/c warm"]
    assert (tmp_path / "cache" / "lean-deps" / LA.deps_key(root / "pkg/a") / "dep" / "built").exists()


def test_kept_builds_past_the_cap_go_least_recently_used_first_and_a_held_one_stays(tmp_path):
    """Every tree whose manifest or lakefile differs keeps a build of its own, and nothing removed them: infra found 71, 19 GB,
    on a 200 GB disk (12 Oct). A build given back is the newest whatever its directory's mtime was; one whose lock a run
    holds stays."""
    import fcntl

    store = tmp_path / "lean-builds"
    for i in range(6):
        (store / f"k{i}").mkdir(parents=True)
        os.utime(store / f"k{i}", (1000 + i, 1000 + i))
    work = tmp_path / "work"
    (work / "pkg" / ".lake" / "build").mkdir(parents=True)
    os.utime(work / "pkg" / ".lake" / "build", (1, 1))                 # renamed in, it would keep this mtime
    builds = LA.WarmBuilds(store, kept=3)
    with open(store / "k0.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        builds.give_back(work, {work / "pkg"})
    left = sorted(p.name for p in store.iterdir() if p.is_dir())
    assert left == sorted(["k0", "k4", "k5", LA.build_key("pkg", work)])
    assert builds.evict() == 1 and "k0" not in {p.name for p in store.iterdir()}


def test_inputs_are_the_package_its_path_dependencies_and_the_auditor(tmp_path, monkeypatch):
    _tree(tmp_path, monkeypatch)
    files = LA.COMMON.tracked_files()
    assert LA.packages(files) == ["pkg/a", "pkg/b"]
    assert LA.path_deps("pkg/b") == ["pkg/a"]
    got = LA.inputs("pkg/b", files)
    assert "pkg/a/A.lean" in got and "pkg/b/B.lean" in got and "tools/verity/lean/check.py" in got and "docs/outside.txt" not in got
    nested = [*files, "pkg/a/inner/lakefile.toml", "pkg/a/inner/X.lean"]
    assert "pkg/a/inner/X.lean" not in LA.own_files("pkg/a", nested)


def test_a_spec_reads_the_packages_proving_its_guarantees_and_misses_with_them(tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/a/lean-audit.json").write_text(json.dumps({"proved_in": ["../b", "../b/"]}))
    files = LA.COMMON.tracked_files()
    assert LA.provers("pkg/a") == ["pkg/b"] and LA.provers("pkg/b") == []
    assert "pkg/b/B.lean" in LA.inputs("pkg/a", files) and LA.inputs("pkg/a", files) == LA.inputs("pkg/b", files)
    (root / "pkg/a/lean-audit.json").write_text("{not json")
    assert LA.provers("pkg/a") == []


def test_the_real_packages_and_their_inputs():
    files = LA.COMMON.tracked_files()
    pkgs = LA.packages(files)
    assert "Security/Proofs" in pkgs and not [p for p in pkgs if p.startswith("tools/verity/lean")]
    got = LA.inputs("Security/Proofs", files)
    assert "Security/lean-audit.json" in got and "verity/core/service/flock/FlockVerify.lean" in got
    assert "tools/verity/lean/check.py" in got and not [f for f in got if "/.lake/" in f]
    assert "tools/verity/lean/Replay.lean" in got and "tools/verity/lean/README.md" not in got
    assert not [f for f in got if f.startswith("tools/verity/lean/tests/")]


def test_a_failed_dependency_fetch_is_no_verdict_and_is_retried():
    clone = ("build: exit 1\ninfo: plausible: cloning https://github.com/leanprover-community/plausible\nfatal: could not read "
             "Username for 'https://github.com': No such device or address")
    assert LA.transient({"failures": [clone]})
    assert not LA.transient({"failures": [clone, "axioms: sorryAx in Foo.bar"]})          # a real failure is never retried away
    assert not LA.transient({"failures": ["build: exit 1\nerror: unknown identifier 'x'"]}) and not LA.transient({"failures": []})


def test_a_fetch_failure_keeps_what_was_fetched(tmp_path, monkeypatch):
    """A package that failed only on fetching a dependency compiled nothing: its partial fetch comes back, for the retry."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    k = LA.deps_key(root / "pkg/a")
    fake = FAKE_AUDIT.replace('reps.append({"package": str(pkg), "failures": [f"reads {m}" for m in missing]})',
                              'reps.append({"package": str(pkg), "failures": ["build: exit 1\\nfatal: could not read Username for '
                              '\'https://github.com\': No such device"] if pkg.name == "a" else []})\n    missing = [pkg] if pkg.name == "a" else missing')
    assert fake != FAKE_AUDIT
    (root / "tools/verity/lean/check.py").write_text(fake)
    monkeypatch.setattr(LA, "RETRIES", ())
    rc, ran, _doc = _run(tmp_path, 1)
    assert rc == 1 and (tmp_path / "cache" / "lean-deps" / k / "dep" / "built").exists()


def _packages_dir(d: Path) -> Path:
    """A `.lake/packages` with Mathlib (its build from Mathlib's cache), a package its manifest pins, and one built here."""
    for f, text in {"mathlib/lake-manifest.json": json.dumps({"packages": [{"name": "batteries"}]}),
                    "mathlib/Mathlib.lean": "", "mathlib/.lake/build/lib/lean/Mathlib.olean": "cdn",
                    "mathlib/.git/HEAD": "abc\n", "batteries/.lake/build/lib/lean/Batteries.olean": "cdn",
                    "Arklib/ArkLib.lean": "", "Arklib/.lake/build/lib/lean/ArkLib.olean": "built here"}.items():
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text(text)
    return d


def test_a_bundle_leaves_out_what_mathlibs_cache_supplies_and_restores_only_when_pinned(tmp_path):
    src = _packages_dir(tmp_path / "deps" / "k1")
    assert LA.cache_covered(src) == {"mathlib", "batteries"}
    pin = LA.bundle(src, tmp_path / "k1.tar.gz")
    with tarfile.open(tmp_path / "k1.tar.gz") as t:
        names = set(t.getnames())
    assert "Arklib/.lake/build/lib/lean/ArkLib.olean" in names and "mathlib/.git/HEAD" in names
    assert not [n for n in names if n.startswith(("mathlib/.lake/build", "batteries/.lake/build"))]
    assert pin == {"sha256": hashlib.sha256((tmp_path / "k1.tar.gz").read_bytes()).hexdigest(),
                   "bytes": (tmp_path / "k1.tar.gz").stat().st_size}
    store, url, logs = tmp_path / "store", (tmp_path / "k1.tar.gz").as_uri(), []
    store.mkdir()
    assert not LA.restore(store, "k1", url, {**pin, "sha256": "0" * 64}, "pkg", logs.append)       # not the pinned bytes
    assert not (store / "k1").exists() and "not the pinned bundle" in logs[-1]
    assert not LA.restore(store, "k1", (tmp_path / "gone.tar.gz").as_uri(), pin, "pkg", logs.append)
    assert "could not be restored" in logs[-1] and "gone" not in logs[-1]                           # the URL is never logged
    assert LA.restore(store, "k1", url, pin, "pkg", logs.append)
    assert (store / "k1" / "Arklib/.lake/build/lib/lean/ArkLib.olean").read_text() == "built here"
    assert sorted(p.name for p in store.iterdir()) == ["k1"]                                        # no part file left


def test_a_cold_machine_takes_the_pinned_bundle_from_the_store(tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    k = LA.deps_key(root / "pkg/a")
    (tmp_path / "made" / "dep").mkdir(parents=True)
    (tmp_path / "made" / "dep" / "built").write_text("x")
    pin = LA.bundle(tmp_path / "made", tmp_path / f"{k}.tar.gz")
    (root / LA.PIN).write_text(json.dumps({"schema": "lean-deps-pin/v1", "deps": {k: {"package": "pkg/a", **pin, "run": "r1"}}}))
    out = tmp_path / "out"
    assert LA.run(out, tmp_path / "c" / "lean-audit", tmp_path / "c" / "lean-deps", log=lambda s: None,
                  urls={k: (tmp_path / f"{k}.tar.gz").as_uri()}, export=tmp_path / "export") == 0
    assert sorted((out / "ran.txt").read_text().split("\n")[:-1]) == ["pkg/a warm", "pkg/b cold"]
    listing = json.loads((tmp_path / "export" / "lean-deps.json").read_text())["deps"]
    assert list(listing) == [k] and listing[k]["package"] == "pkg/a" and (tmp_path / "export" / f"{k}.tar.gz").is_file()


def test_setup_takes_a_cold_packages_pinned_bundle_from_the_store_and_replaces_nothing(tmp_path, monkeypatch):
    """`--restore PKG` (`tools/verity/lean/setup.sh`): the pinned bundle becomes PKG's `.lake/packages` when its bytes are the
    pinned ones; dependencies already there, a manifest with no pin, or other bytes leave the package for lake to fetch."""
    root = _tree(tmp_path, monkeypatch)
    k = LA.deps_key(root / "pkg/a")
    (tmp_path / "made" / "dep").mkdir(parents=True)
    (tmp_path / "made" / "dep" / "built").write_text("x")
    pin = LA.bundle(tmp_path / "made", tmp_path / f"{k}.tar.gz")
    pins = lambda p: (root / LA.PIN).write_text(json.dumps({"schema": "lean-deps-pin/v1", "deps": p}))
    pins({k: {"package": "pkg/a", **pin, "run": "r1"}})
    monkeypatch.setattr(pytest.importorskip("research.store.remote"), "load_remote",
                        lambda: types.SimpleNamespace(presign=lambda key, ttl: (tmp_path / f"{k}.tar.gz").as_uri()))
    logs, deps = [], root / "pkg/a/.lake/packages"
    assert LA.restore_into("pkg/a", logs.append) == 0
    assert (deps / "dep" / "built").read_text() == "x" and sorted(p.name for p in deps.parent.iterdir()) == ["packages"]
    (deps / "dep" / "built").write_text("mine")
    assert LA.restore_into("pkg/a", logs.append) == 0 and (deps / "dep" / "built").read_text() == "mine"
    shutil.rmtree(deps.parent)
    pins({k: {"package": "pkg/a", **pin, "sha256": "0" * 64, "run": "r1"}})
    assert LA.restore_into("pkg/a", logs.append) == 1 and not deps.exists() and "not the pinned bundle" in logs[-1]
    pins({})
    assert LA.restore_into("pkg/a", logs.append) == 0 and not deps.exists() and "no bundle is pinned" in logs[-1]
    assert LA.restore_into("pkg/b", logs.append) == 0 and not (root / "pkg/b/.lake").exists()       # path dependencies only


def test_pinning_takes_only_bundles_on_the_remote_for_this_trees_manifests(tmp_path, monkeypatch, capsys):
    root = _tree(tmp_path, monkeypatch)
    k = LA.deps_key(root / "pkg/a")
    (tmp_path / "rec" / "lean-deps").mkdir(parents=True)
    entry = {"package": "pkg/a", "sha256": "ab" * 32, "bytes": 5}
    (tmp_path / "rec" / "lean-deps" / "lean-deps.json").write_text(json.dumps({"deps": {k: entry}}))
    heads = {f"objects/sha256/{'ab' * 32}": {"bytes": 5}}
    tree = [types.SimpleNamespace(path="out/audit/tool/lean-deps.json", sha256="12" * 32, bytes=7),     # the audit's copy of the pins
            types.SimpleNamespace(path=f"lean-deps/{k}.tar.gz", sha256="ab" * 32, bytes=5),
            types.SimpleNamespace(path="lean-deps/lean-deps.json", sha256="cd" * 32, bytes=9)]

    class Store:
        remote = types.SimpleNamespace(head=heads.get)

        def get_attempt(self, run):
            return {"outputs": {"run_record": "art:rec"}}

        def get_manifest(self, art):
            return types.SimpleNamespace(payload=types.SimpleNamespace(files=tree))

        def fetch(self, art, to, paths):
            return tmp_path / "rec"

    local = types.ModuleType("research.store.local")
    local.LocalStore, local.object_key = Store, lambda h: f"objects/sha256/{h}"
    monkeypatch.setitem(sys.modules, "research.store.local", local)
    stale = {"0" * 64: {"package": "pkg/a", "sha256": "ef" * 32, "bytes": 1, "run": "r0"}}
    (root / LA.PIN).write_text(json.dumps({"schema": "lean-deps-pin/v1", "deps": stale}))
    assert LA.pin("r20260929-000000-abcd") == 0
    assert LA.pins() == {k: {**entry, "run": "r20260929-000000-abcd"}}                  # the stale manifest's pin is gone
    heads.clear()
    assert LA.pin("r20260929-000000-abcd") == 1 and "not on the remote" in capsys.readouterr().err
    tree += [types.SimpleNamespace(path=f"again/{k}.tar.gz", sha256="ab" * 32, bytes=5),
             types.SimpleNamespace(path="again/lean-deps.json", sha256="cd" * 32, bytes=9)]
    assert LA.pin("r20260929-000000-abcd") == 1 and "2 bundle listings" in capsys.readouterr().err


def test_the_launcher_signs_only_this_trees_pinned_bundles(tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)
    k = LA.deps_key(root / "pkg/a")
    (root / LA.PIN).write_text(json.dumps({"schema": "lean-deps-pin/v1", "deps": {
        k: {"package": "pkg/a", "sha256": "ab" * 32, "bytes": 5, "run": "r1"},
        "0" * 64: {"package": "pkg/a", "sha256": "cd" * 32, "bytes": 5, "run": "r0"}}}))
    signed = []
    remote = types.SimpleNamespace(presign=lambda key, ttl: signed.append((key, ttl)) or f"https://r2/{key}?sig")
    monkeypatch.setattr(pytest.importorskip("research.store.remote"), "load_remote", lambda: remote)
    p = LA.url_file(tmp_path, ttl_s=60)
    assert json.loads(p.read_text()) == {k: f"https://r2/objects/sha256/{'ab' * 32}?sig"} and signed == [(f"objects/sha256/{'ab' * 32}", 60)]
    assert p.name == LA.URLS and p.stat().st_mode & 0o777 == 0o600


def _proc(root: Path, pid: int, cwd: Path | None = None, fds: tuple[Path, ...] = (), blind: bool = False) -> None:
    p = root / str(pid)
    p.mkdir(parents=True)
    (p / "cwd").symlink_to(cwd or root)
    (p / "root").symlink_to("/")
    if blind:
        (p / "fd").write_text("")
        return
    (p / "fd").mkdir()
    for i, f in enumerate(fds):
        (p / "fd" / str(i)).symlink_to(f)


def test_the_sweep_removes_only_the_scratch_trees_no_audit_holds(tmp_path):
    """A tree an audit left when it was killed goes; one whose creator lives, or a live process is inside, stays, and so does a
    tree without a pid in its name while it is young or some process can't be read."""
    base, proc = tmp_path / "cache", tmp_path / "proc"
    proc.mkdir()
    t = {n: base / f"{LA.SCRATCH}{n}" for n in ("41-dead", "42-alive", "old", "old-cwd", "old-fd", "young")}
    for d in t.values():
        (d / "pkg").mkdir(parents=True)
    for n in ("old", "old-cwd", "old-fd"):
        os.utime(t[n], (0, 0))
    now = LA.SCRATCH_GRACE_S + 10.0
    os.utime(t["young"], (now - 5, now - 5))
    _proc(proc, 42)
    _proc(proc, 7, cwd=t["old-cwd"] / "pkg")
    _proc(proc, 8, fds=(t["old-fd"] / "pkg" / "x.olean",))
    (base / "lean-deps").mkdir()

    gone = LA.sweep(base, log=lambda m: None, proc=proc, now=now)
    assert sorted(gone) == [t["41-dead"].name, t["old"].name]
    assert sorted(p.name for p in base.iterdir()) == sorted(["lean-deps", *(t[n].name for n in ("42-alive", "old-cwd", "old-fd", "young"))])

    _proc(proc, 9, blind=True)
    for n in ("43-dead", "old2"):
        (base / f"{LA.SCRATCH}{n}").mkdir()
    os.utime(base / f"{LA.SCRATCH}old2", (0, 0))
    assert LA.sweep(base, log=lambda m: None, proc=proc, now=now) == [f"{LA.SCRATCH}43-dead"]
    assert (base / f"{LA.SCRATCH}old2").is_dir()


def test_an_audit_sweeps_before_it_makes_its_scratch_tree_named_by_its_pid(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(LA, "sweep", lambda base, log=print, **k: seen.append(base))
    made = []
    real = LA.tempfile.mkdtemp
    monkeypatch.setattr(LA.tempfile, "mkdtemp", lambda prefix, dir: made.append(prefix) or real(prefix=prefix, dir=dir))
    deps = tmp_path / "cache" / "lean-deps"
    (tmp_path / "out").mkdir()
    LA._audit([], [], {}, {}, tmp_path / "out", tmp_path / "cache" / "lean-audit", deps, True, lambda m: None)
    assert seen == [deps.parent]
    assert made == [f"{LA.SCRATCH}{os.getpid()}-"]
