"""`tools/verity/lean/lean_audit.py`: audit verdicts reused only while every input is unchanged, the hermetic scratch tree as the read
guard, and dependencies kept warm across runs."""
from __future__ import annotations

import ast
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
spec = importlib.util.spec_from_file_location("lean_check_for_test_lean_audit", ROOT / "tools" / "verity" / "lean" / "check.py")
AUDIT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AUDIT)

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
(out / "argv.json").write_text(json.dumps(a))
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


def _tree(tmp: Path, monkeypatch, extra: dict[str, str] | None = None) -> Path:
    root = tmp / "repo"
    files = {
        "tools/verity/lean/check.py": FAKE_AUDIT, "tools/verity/lean/README.md": "", "tools/verity/lean/tests/fixtures/control.lean": "",
        "tools/verity/lean/lean_audit.py": "", "tools/verity/lean/lean-deps.json": "{}", "tools/verity/check/common.py": "", "tools/verity/check/check.py": "",
        LA.CONTRACT: "",
        "pkg/a/lakefile.toml": "", "pkg/a/lean-toolchain": "v1\n", "pkg/a/A.lean": "theorem a : True := trivial\n",
        "pkg/a/lake-manifest.json": json.dumps({"packages": [{"name": "dep", "type": "git", "rev": "abc"}]}),
        "pkg/a/lean-audit.json": "{}",
        "pkg/b/lakefile.toml": "", "pkg/b/lean-toolchain": "v1\n", "pkg/b/B.lean": "",
        "pkg/b/lake-manifest.json": json.dumps({"packages": [{"name": "a", "type": "path", "dir": "../a"}]}),
        "pkg/b/lean-audit.json": json.dumps({"reads": ["docs/outside.txt"]}),
        "docs/outside.txt": "read by b's audit, but no input of it",
        **(extra or {}),
    }
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    monkeypatch.setattr(LA, "ROOT", root)
    monkeypatch.setattr(LA.COMMON, "tracked_files", lambda r=None: sorted(files))
    monkeypatch.setattr(LA.COMMON, "cores", lambda: 2)
    monkeypatch.setattr(LA, "sandboxed", lambda: True)
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


def test_the_audit_replays_the_import_closure_against_the_kept_cache_and_a_fresh_one_against_its_own(tmp_path, monkeypatch):
    _tree(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    def argv(out):
        return json.loads((out / "argv.json").read_text())
    def named(a):
        return [Path(x) for i, x in enumerate(a) if not x.startswith("--") and a[i - 1] not in ("--out", "--repo",
                                                                                                  "--replay-cache", "--artifacts")]
    LA.run(tmp_path / "kept", cache / "lean-audit", cache / "lean-deps", log=lambda s: None)
    a = argv(tmp_path / "kept")
    assert "--closure" in a and Path(a[a.index("--replay-cache") + 1]).resolve() == (cache / "lean-replay").resolve()
    # outside every audited package's .lake, which check.py's `build_roots_over` refuses (no build writes a replay entry)
    assert named(a) and AUDIT.build_roots_over(Path(a[a.index("--replay-cache") + 1]), named(a)) == []
    LA.run(tmp_path / "fresh", cache / "lean-audit", cache / "lean-deps", fresh=True, log=lambda s: None)
    a = argv(tmp_path / "fresh")
    run_cache = Path(a[a.index("--replay-cache") + 1])
    # the packages of one fresh run share a cache in its scratch tree, gone with it; the kept one isn't read
    assert "--closure" in a and "--artifacts" not in a and run_cache.name == "lean-replay"
    assert run_cache.parent.name.startswith(LA.SCRATCH) and not run_cache.parent.exists()
    assert named(a) and AUDIT.build_roots_over(run_cache, named(a)) == []


#: a stand-in for check.py that knows `--regenerate`: a package whose policy says "stale" fails on a record difference, or with
#: the flag passes on the record it writes under --out (`check.py`'s `settled` and `regenerated`)
REGEN_AUDIT = r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
root, out = Path.cwd(), Path(a[a.index("--out") + 1])
named = [Path(x) for i, x in enumerate(a) if not x.startswith("--") and not (i and a[i - 1] in ("--out", "--repo", "--replay-cache", "--artifacts"))]
out.mkdir(parents=True, exist_ok=True)
reps, failed = [], []
for pkg in named:
    with open(out / "ran.txt", "a") as f:
        f.write(f"{pkg.relative_to(root)}\n")
    rep = {"package": str(pkg), "failures": []}
    if json.loads((pkg / "lean-audit.json").read_text()).get("stale"):
        rec = out / f"audit-{pkg.name}" / "lean-audit.json"
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text("{}")
        diff = ["guarantees: they no longer read definitions in P.Defs"]
        rep |= {"settled": diff, "regenerated": str(rec)} if "--regenerate" in a else {"failures": diff}
    if rep["failures"]:
        failed.append(str(pkg))
    reps.append(rep)
(out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "packages": reps}))
sys.exit(1 if failed else 0)
'''


def test_a_trains_audit_passes_on_regenerated_records_names_them_and_keeps_no_such_pass(tmp_path, monkeypatch):
    """`--regenerate` (a train's lean-audit step) reaches check.py and not the cache key: a package that passes only on
    its regenerated record is named in audit.json's `regenerated` for the train to commit, and is never kept as a reusable pass,
    since its lock changes when the record lands; the other packages' passes are kept and reused as before."""
    root = _tree(tmp_path, monkeypatch)
    (root / "tools/verity/lean/check.py").write_text(REGEN_AUDIT)
    (root / "pkg/b/lean-audit.json").write_text(json.dumps({"stale": True}))
    cache, deps = tmp_path / "cache" / "lean-audit", tmp_path / "cache" / "lean-deps"

    def go(n: int, regenerate: bool) -> tuple[int, list[str], dict]:
        out = tmp_path / f"out{n}"
        rc = LA.run(out, cache, deps, log=lambda s: None, regenerate=regenerate)
        ran = (out / "ran.txt").read_text().split() if (out / "ran.txt").exists() else []
        return rc, ran, json.loads((out / "audit.json").read_text())
    rc, ran, doc = go(1, False)
    assert rc == 1 and doc["failed"] == ["pkg/b"] and doc["regenerated"] == {}
    rc, ran, doc = go(2, True)
    assert rc == 0 and ran == ["pkg/b"] and list(doc["cached"]) == ["pkg/a"], "a's pass from run 1 is reused: same key"
    assert doc["regenerated"] == {"pkg/b": str(tmp_path / "out2" / "audit-b" / "lean-audit.json")}
    rc, ran, doc = go(3, True)
    assert rc == 0 and ran == ["pkg/b"], "a pass on a regenerated record is not kept"


#: a stand-in for check.py --regenerate [--bookkeeping warn]: pkg/b's lock fails on bookkeeping, which only the flag repairs,
#: in a regenerated lock listing two guarantees under `unwitnessed` and one warning
BOOK_AUDIT = r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
root, out = Path.cwd(), Path(a[a.index("--out") + 1])
named = [Path(x) for i, x in enumerate(a) if not x.startswith("--") and not (i and a[i - 1] in ("--out", "--repo", "--replay-cache", "--artifacts", "--bookkeeping"))]
out.mkdir(parents=True, exist_ok=True)
reps, failed = [], []
for pkg in named:
    rep = {"package": str(pkg), "failures": []}
    if json.loads((pkg / "lean-audit.json").read_text()).get("unlisted"):
        unlisted = ["open: P.Sound is a premise the lock doesn't list"]
        if a[a.index("--bookkeeping") + 1:][:1] == ["warn"] if "--bookkeeping" in a else False:
            rec = out / f"audit-{pkg.name}" / "lean-audit.json"
            rec.parent.mkdir(parents=True, exist_ok=True)
            rec.write_text(json.dumps({"unwitnessed": {"P.t": "h", "P.u": "h"}}))
            rep |= {"settled": unlisted, "regenerated": str(rec),
                    "bookkeeping": ["bookkeeping: P.Sound (@lean): listed under `open` on LEAN-186, due D (open.unlisted)"]}
        else:
            rep["failures"] = unlisted
    if rep["failures"]:
        failed.append(str(pkg))
    reps.append(rep)
(out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "packages": reps}))
sys.exit(1 if failed else 0)
'''


def test_the_lock_runs_audit_repairs_bookkeeping_and_the_caps_the_repaired_locks_break(tmp_path, monkeypatch):
    """`--bookkeeping warn` (drive's lock run) reaches check.py: a package that fails only on bookkeeping passes on its
    repaired lock, and a cap of `tests/test_lean_packages.py` that lock breaks is set to its count in a copy under `repaired/`,
    which audit.json's `repaired` names for drive to commit, with every warning under `bookkeeping`. Without the flag the
    package fails and nothing is repaired. The flag goes only with --regenerate, over every package: `--package` naming
    each, as check.py's `lean-audit:PKG` parts of a deferred run do, is every package, and one that leaves a package out
    fails."""
    caps = 'UNWITNESSED = {"pkg/b": 1}\n\nINLINE = {}\n\nROOM = {"UNWITNESSED pkg/a": 2}\n'
    root = _tree(tmp_path, monkeypatch, {LA.CAPS_TEST: caps})
    (root / "tools/verity/lean/check.py").write_text(BOOK_AUDIT)
    (root / "pkg/b/lean-audit.json").write_text(json.dumps({"unlisted": True}))
    cache, deps = tmp_path / "cache" / "lean-audit", tmp_path / "cache" / "lean-deps"

    def go(n: int, bookkeeping: bool, only: list[str] | None = None) -> tuple[int, dict]:
        out = tmp_path / f"out{n}"
        rc = LA.run(out, cache, deps, log=lambda s: None, regenerate=True, bookkeeping=bookkeeping, only=only)
        return rc, json.loads((out / "audit.json").read_text())
    rc, doc = go(1, False)
    assert rc == 1 and doc["failed"] == ["pkg/b"] and "repaired" not in doc and "bookkeeping" not in doc
    rc, doc = go(2, True)
    assert rc == 0 and doc["regenerated"] == {"pkg/b": str(tmp_path / "out2" / "audit-b" / "lean-audit.json")}
    assert doc["repaired"] == {LA.CAPS_TEST: str(tmp_path / "out2" / "repaired" / LA.CAPS_TEST)}
    assert Path(doc["repaired"][LA.CAPS_TEST]).read_text() == caps.replace('{"pkg/b": 1}', '{"pkg/b": 2}')
    assert (root / LA.CAPS_TEST).read_text() == caps, "the tree's own copy is never written"
    assert doc["bookkeeping"] == [
        "pkg/b: bookkeeping: P.Sound (@lean): listed under `open` on LEAN-186, due D (open.unlisted)",
        f"bookkeeping: UNWITNESSED['pkg/b'] in {LA.CAPS_TEST} (@lean): 1 -> 2, the count its lock has"]
    every = sorted(LA.packages(LA.COMMON.tracked_files(root)))
    assert "pkg/b" in every and len(every) > 1
    rc, named = go(3, True, every)
    assert rc == 0 and named["repaired"].keys() == doc["repaired"].keys() and named["bookkeeping"] == doc["bookkeeping"]
    logged: list[str] = []
    assert LA.run(tmp_path / "out4", cache, deps, log=logged.append, regenerate=True, bookkeeping=True, only=["pkg/b"]) == 1
    assert any("--package leaves out" in s for s in logged), logged
    with pytest.raises(SystemExit):
        LA.main(["--out", str(tmp_path / "out5"), "--bookkeeping", "warn"])


def test_held_caps_sets_each_broken_cap_to_its_count_and_leaves_the_rest():
    """A count above its cap is raised to, and a cap above its count by more than its room lowered to, the count and its
    room, which changes queued on `next` still use; a cap within its room stays, and a package no cap names with nothing
    listed gets none. A caps file without the line fails."""
    text = ('X = 1\nUNWITNESSED = {"a": 5, "b": 4, "c": 9, "e": 6}\nINLINE = {"a": 3}\n'
            'ROOM = {"UNWITNESSED b": 2, "UNWITNESSED e": 1}\n')
    locks = {"a": {"unwitnessed": {"t": "h"}, "shape": {"inline": {"t": {"x": "why", "y": "why"}}}},
             "b": {"unwitnessed": {"t": "h", "u": "h"}}, "c": {"unwitnessed": dict.fromkeys("tuvwxyzpqr", "h")}, "d": {},
             "e": {"unwitnessed": {"t": "h", "u": "h"}}}
    new, warned = LA.held_caps(text, locks)
    assert new == ('X = 1\nUNWITNESSED = {"a": 1, "b": 4, "c": 10, "e": 3}\nINLINE = {"a": 2}\n'
                   'ROOM = {"UNWITNESSED b": 2, "UNWITNESSED e": 1}\n')
    assert [w.split(" in ")[0] for w in warned] == ["bookkeeping: UNWITNESSED['a']", "bookkeeping: UNWITNESSED['c']",
                                                    "bookkeeping: UNWITNESSED['e']", "bookkeeping: INLINE['a']"]
    assert warned[2].endswith("6 -> 3, the count its lock has and its room of 1")
    assert LA.held_caps(new, locks) == (new, [])
    with pytest.raises(LookupError):
        LA.held_caps("UNWITNESSED = {}\n", locks)


def test_raised_unread_raises_the_areas_above_their_caps_and_keeps_the_rest():
    u = types.SimpleNamespace(caps_text=lambda caps: json.dumps(caps, sort_keys=True))
    unread = {"failures": ["unread: A has 3 ..."], "areas": {
        "A": {"count": {"modules": 3, "declarations": 30}, "cap": {"modules": 2, "declarations": 40}},
        "B": {"count": {"modules": 1, "declarations": 5}, "cap": {"modules": 4, "declarations": 50}},
        "C": {"count": {"modules": 2, "declarations": 6}, "cap": None},
        "D": {"count": {"modules": 0, "declarations": 0}, "cap": None}}}
    text, warned = LA.raised_unread(unread, u)
    assert json.loads(text) == {"A": {"modules": 3, "declarations": 30}, "B": {"modules": 4, "declarations": 50},
                                "C": {"modules": 2, "declarations": 6}}
    assert [w.split(" in ")[0] for w in warned] == ["bookkeeping: unread area A", "bookkeeping: unread area C"]


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
    def keys():
        return {p: LA.key(p, files, LA.FLAGS) for p in ("pkg/a", "pkg/b")}
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
    def single(d):
        return (root / "pkg/b/lakefile.toml").write_text(f"name = \"b\"\n\n[[require]]\nname = \"a\"\npath = '../{d}'\n")
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
    """No audit reads `tools/verity/lean/README.md`, `tools/verity/lean/tests/` or the pin file `lean-deps.json`
    (preflight's, read before the audit), so an edit to them audits no package again. Nor the directory's other files that
    `NOT_READ` lists, such as `merge.py` and `lean_changed.py`, which would otherwise audit every package again on every
    node. They stay out of the scratch tree too, so an audit that read one would fail."""
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
    (root / "pkg/b/lean-audit.json").write_text("{}")
    assert _run(tmp_path, 4)[1] == []                  # the policy of run 2 again: its pass answers
    files = LA.COMMON.tracked_files()
    for i, name in enumerate(sorted(LA.NOT_READ), 5):
        f = f"tools/verity/lean/{name}"
        (root / f).write_text(f"{i}\n")
        files = sorted({*files, f})
        monkeypatch.setattr(LA.COMMON, "tracked_files", lambda r=None, files=files: files)
        rc, ran, doc = _run(tmp_path, i)
        assert rc == 0 and ran == [] and sorted(doc["cached"]) == ["pkg/a", "pkg/b"], name
    (root / "pkg/b/lean-audit.json").write_text(json.dumps({"reads": [f"tools/verity/lean/{n}" for n in sorted(LA.NOT_READ)]}))
    rc, _ran, doc = _run(tmp_path, 99)
    assert rc == 1 and doc["failed"] == ["pkg/b"]
    assert doc["packages"][1]["failures"] == [f"reads tools/verity/lean/{n}" for n in sorted(LA.NOT_READ)]


def test_a_passing_audit_keeps_its_build(tmp_path, monkeypatch):
    """b201 rebuilt all 561 of `soundness`'s modules (2,508 s) because the auditor changed, though no `.olean` depends on it."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    store = tmp_path / "cache" / "lean-builds"
    def builds(n):
        return sorted((tmp_path / f"out{n}" / "builds.txt").read_text().split("\n")[:-1])
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


def test_without_the_sandbox_nothing_is_restored_and_nothing_kept(tmp_path, monkeypatch):
    """Outside the sandbox a build's compile-time code may leave any output, and the sampled recompile that binds a
    restored module to its source doesn't run. So on such a node, with a kept build of each package and an artifact cache
    present, a run takes neither, nor a bundle, gives check.py no `--artifacts`, and keeps nothing it built."""
    root = _tree(tmp_path, monkeypatch)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    store = tmp_path / "cache" / "lean-builds"
    assert _run(tmp_path, 1)[0] == 0 and sorted(p.name for p in store.iterdir() if p.is_dir())
    kept = {p: p.stat().st_mtime_ns for p in store.rglob("*")}
    (tmp_path / "cache" / "lean-artifacts").mkdir(exist_ok=True)
    (root / "tools/verity/lean/check.py").write_text(FAKE_AUDIT + "\n# the auditor changed: every package runs again\n")
    monkeypatch.setattr(LA, "sandboxed", lambda: False)
    calls = []
    monkeypatch.setattr(LA.LakeArtifacts, "take", lambda self, urls, log=print: calls.append("take") or 0)
    monkeypatch.setattr(LA.LakeArtifacts, "put", lambda self, reports, since=None, log=print: calls.append("put") or 0)
    logged = []
    out = tmp_path / "out2"
    rc = LA.run(out, tmp_path / "cache" / "lean-audit", tmp_path / "cache" / "lean-deps", log=logged.append,
                urls={"bundle": "file:///nowhere"})
    assert rc == 0 and sorted((out / "builds.txt").read_text().split("\n")[:-1]) == ["pkg/a cold", "pkg/b cold"]
    assert "--artifacts" not in json.loads((out / "argv.json").read_text()) and calls == []
    assert {p: p.stat().st_mtime_ns for p in store.rglob("*")} == kept, "nothing it built is kept"
    assert any("builds without the sandbox" in ln for ln in logged)


def _built(pkg: Path, mod: str, outputs: dict[str, str], imports: tuple[str, ...] = ()) -> None:
    """A module of ``pkg``'s build as Lake leaves it: each output beside its `<file>.hash`, the trace naming them, and its
    `setup.json`, whose `importArts` hold the `.olean` of each module it imports, transitively (``imports``)."""
    build = pkg / ".lake" / "build"
    for ext, h in outputs.items():
        f = build / ("ir" if ext == "c" else "lib/lean") / f"{mod}.{ext}"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.unlink(missing_ok=True)    # a rebuild replaces its outputs: `put` left these read-only, linked into the cache
        f.write_text(f"{mod}.{ext} {h}")
        (f.parent / f"{f.name}.hash").write_text(h)
    trace = build / "lib/lean" / f"{mod}.trace"
    trace.write_text(json.dumps({"depHash": hashlib.sha256(mod.encode()).hexdigest()[:16], "outputs": {
        "o": [f"{outputs['olean']}.olean"], "m": False, "i": f"{outputs['ilean']}.ilean", "c": f"{outputs['c']}.c"}}))
    setup = build / "ir" / f"{mod}.setup.json"
    setup.parent.mkdir(parents=True, exist_ok=True)
    setup.write_text(json.dumps({"name": mod.replace("/", "."), "package": "p", "importArts": {
        i: {"olean": str(build / "lib/lean" / f"{i.replace('.', '/')}.olean")} for i in imports}}))


#: check.py's report on a build in the sandbox, the only kind `LakeArtifacts.put` indexes
SANDBOXED = {"sandbox": True}


def _package(tmp: Path) -> Path:
    """Package `p`, library `P` (its default target): `P.A`, `P.B` importing it, and `P` importing both, all built."""
    pkg = tmp / "pkg"
    (pkg / "P").mkdir(parents=True)
    (pkg / "lakefile.toml").write_text('name = "p"\ndefaultTargets = ["P"]\n\n[[lean_lib]]\nname = "P"\n')
    (pkg / "lake-manifest.json").write_text(json.dumps({"name": "p", "packages": []}))
    (pkg / "lean-toolchain").write_text("leanprover/lean4:v4.34.0\n")
    (pkg / "P.lean").write_text("import P.A\nimport P.B\n")
    (pkg / "P" / "A.lean").write_text("theorem a : True := trivial\n")
    (pkg / "P" / "B.lean").write_text("import P.A\n")
    _built(pkg, "P/A", {"olean": "a" * 16, "ilean": "b" * 16, "c": "c" * 16})
    _built(pkg, "P/B", {"olean": "d" * 16, "ilean": "e" * 16, "c": "f" * 16}, ("P.A",))
    _built(pkg, "P", {"olean": "1" * 16, "ilean": "2" * 16, "c": "3" * 16}, ("P.A", "P.B"))
    return pkg


def _keys(pkg: Path) -> dict[str, str]:
    """{module: the entry its key names in a store}, keyed from the outputs ``pkg``'s build holds, as `put` keys them."""
    keyed = LA.MK.keys(pkg)
    outs = LA.MK.read_digests(keyed)
    return {m: LA.MK.entry_path(Path(), LA.MK.key(m, keyed, outs)).as_posix() for m in keyed}


def test_the_artifact_cache_indexes_each_module_under_its_key_after_what_it_imports(tmp_path):
    """Lake's input hash is 64 bits of a hash that isn't cryptographic, so the cache indexes a module under its key
    (`module_keys`), the task graph's of its compile reading the outputs its build holds of what it imports, with the digest
    of each artifact. A module whose build holds an output under another hash than its trace names (an edit after the
    build) stays out, and so does everything importing it."""
    pkg = _package(tmp_path)
    lib, store = pkg / ".lake/build/lib/lean", tmp_path / "store"
    (lib / "P/B.olean.hash").write_text("0" * 16)
    arts, entries = LA.LakeArtifacts(store), _keys(pkg)
    assert arts.put({pkg: SANDBOXED}) == 1 and [e for e, _ in arts.added] == [entries["P.A"]]
    names = ["a" * 16 + ".olean", "b" * 16 + ".ilean", "c" * 16 + ".c"]
    assert json.loads((store / entries["P.A"]).read_text()) == {
        "module": "P.A", "scope": "p", "hash": hashlib.sha256(b"P/A").hexdigest()[:16],
        "outputs": json.loads((lib / "P/A.trace").read_text())["outputs"],
        "digests": {n: LA.MK.C.digest(f"P/A.{n.split('.', 1)[1]} {n.split('.')[0]}".encode()) for n in names}}
    assert sorted(f.name for f in (store / "artifacts").iterdir()) == sorted(names)
    (lib / "P/B.olean.hash").write_text("d" * 16)
    assert arts.put({pkg: SANDBOXED}) == 2 and arts.put({pkg: SANDBOXED}) == 0, "P.B, then P, which imports it"
    assert sorted(f"keys/{f.name}" for f in (store / "keys").iterdir()) == sorted(entries.values())
    _built(pkg, "P/A", {"olean": "a" * 16, "ilean": "8" * 16, "c": "9" * 16})
    assert _keys(pkg) == entries and arts.put({pkg: SANDBOXED}) == 0 and arts.added[-1][0] == entries["P.A"], \
        "P.A rebuilt to the same olean: P.B and P, which read only that, keep their keys"
    assert json.loads((store / entries["P.A"]).read_text())["outputs"]["i"] == "8" * 16 + ".ilean", \
        "a newer build of a module replaces its entry"
    _built(pkg, "P/A", {"olean": "7" * 16, "ilean": "8" * 16, "c": "9" * 16})
    moved = _keys(pkg)
    assert {m for m in entries if moved[m] != entries[m]} == {"P.B", "P"}, "P.A's olean is another: what reads it is rekeyed"


def test_only_a_module_the_run_compiled_goes_into_the_artifact_cache(tmp_path):
    """A kept build holds modules an earlier run compiled from other sources that Lake's input hash took for these: indexed
    under these sources' key, they would be restored for them elsewhere. Nor does a module Lake restored go in again, or one
    whose artifact the cache holds with other bytes."""
    pkg = _package(tmp_path)
    lib = pkg / ".lake/build/lib/lean"
    for t in lib.rglob("*.trace"):
        os.utime(t, (1000, 1000))
    assert LA.LakeArtifacts(tmp_path / "s1").put({pkg: SANDBOXED}, since=1000) == 0
    os.utime(lib / "P/A.trace", (1001, 1001))
    assert LA.LakeArtifacts(tmp_path / "s1").put({pkg: SANDBOXED}, since=1000) == 1
    trace = json.loads((lib / "P/A.trace").read_text())
    (lib / "P/A.trace").write_text(json.dumps(trace | {"synthetic": True}))
    assert LA.LakeArtifacts(tmp_path / "s2").put({pkg: SANDBOXED}) == 0, "P.A restored: it, and P.B and P after it, stay out"
    (lib / "P/A.trace").write_text(json.dumps(trace))
    (tmp_path / "s3" / "artifacts").mkdir(parents=True)
    (tmp_path / "s3" / "artifacts" / ("a" * 16 + ".olean")).write_text("other bytes under P.A's artifact's name")
    assert LA.LakeArtifacts(tmp_path / "s3").put({pkg: SANDBOXED}) == 0


def test_a_build_outside_the_sandbox_indexes_nothing(tmp_path):
    """Where `unshare` is missing, check.py builds without the sandbox, and compile-time code runs as the user: it could
    leave in the build any outputs, traces and `setup.json` it chose, which `put` would index under a module's key for
    every later build on the node to restore. Only a build whose check.py report says it ran in the sandbox goes in."""
    pkg, store, said = _package(tmp_path), tmp_path / "store", []
    arts = LA.LakeArtifacts(store)
    for rep in ({"sandbox": False}, {}, {"sandbox": "true"}, None):
        assert arts.put({pkg: rep}, log=said.append) == 0 and arts.added == [], rep
    assert not (store / "keys").exists() and not (store / "artifacts").exists()
    assert len(said) == 4 and all(s.endswith("none of its modules go into the artifact cache") for s in said)
    assert arts.put({pkg: SANDBOXED}, log=said.append) == 3 and len(said) == 4


def test_the_artifact_cache_drops_its_oldest_past_the_cap_counting_only_what_it_alone_holds(tmp_path):
    store = tmp_path / "store"
    (store / "artifacts").mkdir(parents=True)
    (store / "keys").mkdir(parents=True)
    (store / "outputs" / "p").mkdir(parents=True)
    kept = tmp_path / "kept.olean"
    kept.write_bytes(b"k" * 100)
    os.link(kept, store / "artifacts" / "shared.olean")              # a kept build holds it too: no disk of the cache's own
    for i in range(4):
        f, e = store / "artifacts" / f"{i}.olean", store / "keys" / f"{i}.json"
        f.write_bytes(b"x" * 10)
        e.write_text("{}")
        os.utime(f, (i + 10, i + 10))
        os.utime(e, (i + 10, i + 10))
    legacy = store / "outputs" / "p" / "0.json"                     # Lake's own layout, which no build reads any more
    legacy.write_text("{}")
    os.utime(legacy, (5, 5))
    os.utime(store / "artifacts" / "shared.olean", (1, 1))
    assert LA.LakeArtifacts(store, cap=40).prune() == 0
    assert LA.LakeArtifacts(store, cap=30).prune() == 3               # 40 > 30: the shared one goes first, then 0 and 1
    assert sorted(f.name for f in (store / "artifacts").iterdir()) == ["2.olean", "3.olean"] and kept.is_file()
    assert sorted(f.name for f in (store / "keys").iterdir()) == ["2.json", "3.json"] and not legacy.exists()


def _bundle_of(tmp: Path, pkg: Path) -> tuple[Path, str]:
    arts = LA.LakeArtifacts(tmp / "node-a")
    arts.put({pkg: SANDBOXED})
    dest = tmp / "lean-artifacts.tar.gz"
    assert arts.bundle(dest) == len(arts.added) > 0
    return dest, hashlib.sha256(dest.read_bytes()).hexdigest()


def test_a_bundle_carries_the_module_versions_one_node_added_to_another_and_nothing_else(tmp_path):
    """Kept builds and the artifact cache are per node, and the trains spread over several machines, which share the cache
    rather than pin to one. A node takes a recorded check's bundle once, checked against its sha256, and never replaces a
    file."""
    pkg = _package(tmp_path)
    entries = _keys(pkg)
    dest, h = _bundle_of(tmp_path, pkg)
    with tarfile.open(dest) as t:
        names = t.getnames()
    assert [n.split("/")[0] for n in names] == ["keys"] * 3 + ["artifacts"] * 9, "entries before the artifacts they name"
    b = LA.LakeArtifacts(tmp_path / "node-b")
    (tmp_path / "node-b" / "artifacts").mkdir(parents=True)
    (tmp_path / "node-b" / "artifacts" / ("a" * 16 + ".olean")).write_text("node b's own")
    logged = []
    assert b.take({LA.SHARED + h: dest.as_uri(), LA.SHARED + "0" * 64: dest.as_uri(), "k": "https://deps"}, logged.append) == 3
    assert any("is not the one named" in ln for ln in logged)
    got = tmp_path / "node-b"
    assert sorted(f.relative_to(got).as_posix() for f in got.rglob("*") if f.is_file() and f.parent.name != "taken"
                  and f.name != "lock") == sorted([*(n for n in names if n.startswith("artifacts/")), *entries.values()])
    assert (got / "artifacts" / ("a" * 16 + ".olean")).read_text() == "node b's own"
    assert not os.stat(got / "artifacts" / ("d" * 16 + ".olean")).st_mode & 0o222
    assert LA.MK.stage(got, pkg, [], tmp_path / "cache")[0] == {}, \
        "P.A's artifact there is not the one its entry names: built, not restored, and P.B and P, which read it, too"
    c = LA.LakeArtifacts(tmp_path / "node-c")
    assert c.take({LA.SHARED + h: dest.as_uri()}, lambda s: None) == 3
    hashes = {m: hashlib.sha256(m.replace(".", "/").encode()).hexdigest()[:16] for m in entries}
    assert LA.MK.stage(tmp_path / "node-c", pkg, [], tmp_path / "cache-c")[0] == {("p", hashes[m]): m for m in entries}
    assert b.take({LA.SHARED + h: dest.as_uri()}, logged.append) == 0 and logged[-1].startswith("lean-audit: 3 module")
    junk = tmp_path / "junk.tar.gz"
    junk.write_bytes(b"not a tar")
    j = hashlib.sha256(junk.read_bytes()).hexdigest()
    seen = len(logged)
    assert b.take({LA.SHARED + j: junk.as_uri()}, logged.append) == 0 and "could not be unpacked" in logged[seen]
    assert b.take({LA.SHARED + j: junk.as_uri(), LA.SHARED + "0" * 64: dest.as_uri()}, logged.append) == 0
    assert len(logged) == seen + 2                         # bytes that failed once are never fetched again


def test_a_bundle_member_outside_the_cache_s_layout_or_an_entry_without_its_artifacts_is_left_out(tmp_path):
    """An artifact goes in only as one an entry before it names, with the digest it names; an entry once every artifact it
    names is here. A bundle of Lake's own layout (`outputs/`), which no build reads any more, adds nothing."""
    ok = LA.MK.C.digest(b"ok")

    def entry(art, d):
        return json.dumps({"module": "P.A", "scope": "p", "hash": "1" * 16, "outputs": {"o": [art]},
                           "digests": {art: d}}).encode()

    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as t:
        for name, data in [("keys/" + "1" * 128 + ".json", entry("e" * 16 + ".olean", ok)),
                           ("keys/" + "2" * 128 + ".json", entry("9" * 16 + ".olean", ok)),         # its artifact isn't here
                           ("keys/" + "3" * 128 + ".json", b"not json"),
                           ("keys/" + "4" * 128 + ".json", entry("8" * 16 + ".olean", LA.MK.C.digest(b"other"))),
                           ("keys/" + "6" * 64 + ".json", entry("e" * 16 + ".olean", ok)),         # a key of before
                           ("keys/../" + "5" * 128 + ".json", entry("e" * 16 + ".olean", ok)),
                           ("outputs/p/" + "6" * 16 + ".json", json.dumps({"o": ["e" * 16 + ".olean"]}).encode()),
                           ("../escape", b"x"), ("artifacts/../../x.olean", b"x"),
                           ("artifacts/" + "e" * 16 + ".olean", b"ok"), ("artifacts/" + "8" * 16 + ".olean", b"ok"),
                           ("artifacts/" + "7" * 16 + ".olean", b"ok")]:                            # no entry names it
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    h = hashlib.sha256(bad.read_bytes()).hexdigest()
    store = tmp_path / "node"
    assert LA.LakeArtifacts(store).take({LA.SHARED + h: bad.as_uri()}, lambda s: None) == 1
    assert sorted(f.relative_to(store).as_posix() for f in store.rglob("*") if f.is_file() and f.name != "lock"
                  and f.parent.name != "taken") == ["artifacts/" + "e" * 16 + ".olean", "keys/" + "1" * 128 + ".json"]
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


def _attempts(tmp: Path, rows: list[tuple[str, str, str, list[str]]]):
    """A store index holding ``rows`` (id, tool, state, argv), the newest first, as the store's `attempts` table has them."""
    import sqlite3

    db = sqlite3.connect(tmp / "index.sqlite")
    db.execute("CREATE TABLE attempts(id TEXT PRIMARY KEY, tool TEXT, state TEXT, start_utc TEXT, wall_s REAL, manifest_json TEXT)")
    for i, (rid, tool, state, argv) in enumerate(rows):
        db.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?)",
                   (rid, tool, state, f"2026-10-26T{20 - i:02d}:00:00Z", 60.0, json.dumps({"id": rid, "params": {"argv": argv}})))
    return types.SimpleNamespace(sql=lambda q, a=(): db.execute(q, a).fetchall())


#: drive's Lean check as the store records it: `slot.py` runs the check, so its tool is `python3`
DEFERRED = ["python3", "tools/verity/check/slot.py", "--priority", "--by", "ci", "--", "uv", "run", "--locked", "python",
            "tools/verity/check/check.py", "--keep-going", "--deferred", "lean-audit"]


def test_a_slot_wrapped_check_is_a_recorded_check_whose_bundle_and_verdicts_are_taken(tmp_path, monkeypatch):
    """Checks run under `slot.py`, so the store records drive's trains, Lean checks and after-landing runs with tool
    `python3`; `shared_bundles` and `gathered_packs` selecting only tool `check` would leave each node without another's
    built modules or replay verdicts, building `Security/Proofs` cold (700 to 1,300 s) and replaying some 2,000 modules."""
    index = _attempts(tmp_path, [("r-lean", "python3", "failed", DEFERRED), ("r-other", "python3", "done", ["python3", "x.py"]),
                                 ("r-lean-tool", "python3", "done", ["python3", "tools/verity/lean/check.py", "--build"]),
                                 ("r-check", "check", "done", ["check"]), ("r-running", "python3", "running", DEFERRED)])
    runs = {r: [("lean-audit/lean-artifacts.tar.gz", r[2] * 64, True)] for r in ("r-lean", "r-other", "r-lean-tool", "r-check", "r-running")}
    store = types.SimpleNamespace(remote=None, index=index, get_attempt=lambda r: {"outputs": {"run_files": f"art:{r}"}},
                                  listing=lambda art: [{"path": p, "sha256": h, "remote": on} for p, h, on in runs[art[4:]]])
    assert LA.shared_bundles(store) == ["l" * 64, "c" * 64]
    monkeypatch.setattr(CHECK, "attempt_pack", lambda store, rid, d, name="verdicts.tar.gz": d / rid)
    monkeypatch.setattr(CHECK.VERDICTS, "refusal", lambda pack: None)
    assert [rid for rid, _ in CHECK.gathered_packs(tmp_path, store)] == ["r-lean", "r-check"]


def test_only_a_recorded_check_bundles_its_built_modules_for_other_nodes(tmp_path, monkeypatch):
    """The bundle was written only when `RESEARCH_RUN_DIR` was set, which `check.env()` never passes to a step: no recorded
    check ever wrote one. `check.py` now says so with `--bundle`, inside a recorded run alone."""
    _tree(tmp_path, monkeypatch)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "run"))

    put_with = []

    def put(self, reports, since=None, log=print):
        put_with.append({p.name: rep for p, rep in reports.items()})
        self.added = [(f"keys/{'1' * 64}.json", ["e" * 16 + ".olean"])]
        return 1

    monkeypatch.setattr(LA.LakeArtifacts, "put", put)
    monkeypatch.setattr(LA.LakeArtifacts, "bundle", lambda self, dest, cap=0: dest.write_bytes(b"") or 1)
    for n, bundle in enumerate((False, True)):
        shutil.rmtree(tmp_path / "cache", ignore_errors=True)
        assert LA.run(tmp_path / f"out{n}", tmp_path / "cache" / "lean-audit", tmp_path / "cache" / "lean-deps",
                      log=lambda s: None, bundle=bundle) == 1
        assert (tmp_path / f"out{n}" / "lean-artifacts.tar.gz").exists() == bundle
        assert put_with[-1] == {"a": json.loads((tmp_path / f"out{n}" / "audit.json").read_text())["packages"][0]}, \
            "the passing package, with check.py's report on it, which says whether it was built in the sandbox"
    out = tmp_path / "steps"
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    def step():
        return dict((name, argv) for name, argv, _ in CHECK.lean_steps(out))["lean-audit"]
    assert "--bundle" not in step()
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261026-000000-abcd")
    assert "--bundle" in step()


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
    def builds(n):
        return sorted((tmp_path / f"out{n}" / "builds.txt").read_text().split("\n")[:-1])
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
    """check.py names each package by its resolved path. With the cache behind a symlink (a node's `~/.cache/verity-check`),
    the scratch tree's unresolved path would match no report, and a package that passed would fail as unreported."""
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
    """A manual stop (`research cancel`) sends SIGTERM to the check's whole process group. An audit that dies of it without
    running its finally leaves its scratch tree, about 60 GB, in `~/.cache/verity-check`, and two of them fill a node's root
    disk until a check refuses to start."""
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
    """What `lean_audit.py` loads by path is in `inputs`, and `common.py` and the task graph's contract, which `module_keys`
    loads, import only the standard library, so a key covers everything the step runs. Loading `check.py` again would need
    it back in the key."""
    import ast

    files = LA.COMMON.tracked_files()
    got = LA.inputs("Security/Proofs", files)
    loaded = re.findall(r'_module\(ROOT / "tools" / "verity" / "check" / "([a-z_]+\.py)"', (ROOT / "tools/verity/lean/lean_audit.py").read_text())
    assert loaded == ["common.py"] and all(f"tools/verity/check/{m}" in got for m in loaded)
    assert ROOT / LA.CONTRACT == LA.MK.CONTRACT and LA.CONTRACT in got
    assert "tools/verity/check/check.py" not in got
    for node in [n for f in ("tools/verity/check/common.py", LA.CONTRACT) for n in ast.walk(ast.parse((ROOT / f).read_text()))]:
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
    bundle, while a changed fetched entry needs one; a node's daily run keeps its copies under the same key."""
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
    """Every tree whose manifest or lakefile differs keeps a build of its own, and unremoved they pile up (71 of them, 19 GB,
    on a 200 GB disk). A build given back is the newest whatever its directory's mtime was; one whose lock a run
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


def test_each_packages_newest_build_stays_past_the_cap(tmp_path):
    """At 24 kept builds, fewer than the ~30 packages, every Lean check would evict some packages' only build and rebuild them
    cold (Flock alone takes about 1,100 s)."""
    store = tmp_path / "lean-builds"
    works = []
    for i, pkg in enumerate(["pkg/a", "pkg/b", "pkg/c"]):
        work = tmp_path / f"work{i}"
        (work / pkg / ".lake" / "build").mkdir(parents=True)
        (work / pkg / "lakefile.toml").write_text(f'name = "{i}"\n')
        works.append((work, pkg))
    builds = LA.WarmBuilds(store, kept=1)
    for work, pkg in works:
        builds.give_back(work, {work / pkg})
    left = {p.name for p in store.iterdir() if p.is_dir()}
    assert left == {LA.build_key(pkg, work) for work, pkg in works}
    old = store / "stale"
    old.mkdir()
    (store / "stale.pkg").write_text("pkg/a")
    os.utime(old, (1, 1))
    assert builds.evict() == 1 and not old.exists() and not (store / "stale.pkg").exists()


def test_builds_past_the_byte_cap_go_least_recently_used_first_but_never_a_packages_newest_or_a_held_one(tmp_path):
    """Kept by count alone, 48 builds of about 17 GB each fill a node's root disk. Past `max_gb` the least recently used go
    until the rest fit, but never a package's newest, and never one whose lock a run holds."""
    import fcntl

    store = tmp_path / "lean-builds"
    for i in range(5):
        home = store / f"k{i}"
        home.mkdir(parents=True)
        (home / "out.olean").write_bytes(b"x" * 400_000)
        os.utime(home, (1000 + i, 1000 + i))
    (store / "k0.pkg").write_text("pkg/a")                  # pkg/a's only build: its newest, so it stays
    size = LA.tree_bytes(store / "k1")
    assert size >= 400_000
    builds = LA.WarmBuilds(store, kept=48, max_gb=3.5 * size / 1e9)     # room for three and a half builds
    with open(store / "k1.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert builds.evict() == 2
    left = sorted(p.name for p in store.iterdir() if p.is_dir())
    assert left == ["k0", "k1", "k4"], "k2 and k3 go; k0 is pkg/a's newest and k1 is held"
    assert LA.WarmBuilds(store, kept=48, max_gb=None).evict() == 0, "no byte cap: the count alone"


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
    assert {"tools/verity/lean/upstream.py", "tools/verity/lean/setup.sh", "tools/verity/lean/cache.py"} <= set(got)
    assert not {f"tools/verity/lean/{n}" for n in LA.NOT_READ} & set(got)
    assert {f"tools/verity/lean/{n}" for n in LA.NOT_READ} <= set(files), "an entry of NOT_READ names no file: remove it"
    assert "tools/verity/lean/Unread.lean" in got and set(LA.UNREAD_FACTS) <= set(pkgs)
    lean_check = (ROOT / "tools" / "verity" / "lean" / "check.py").read_text()
    assert ast.literal_eval(re.search(r"^UNREAD_FACTS = (.+)$", lean_check, re.M).group(1)) == LA.UNREAD_FACTS, \
        "check.py writes the ratchet's facts in the audits of the packages lean_audit.py joins them from"


#: `check.py` as `FAKE_AUDIT`, and the audits of the packages the ratchet reads write its facts as the real one does
FACTS_AUDIT = FAKE_AUDIT.replace(
    '    reps.append({"package": str(pkg), "failures": [f"reads {m}" for m in missing]})\n',
    '''    rep = {"package": str(pkg), "failures": [f"reads {m}" for m in missing]}
    rel = pkg.relative_to(root).as_posix()
    if rel in ("Security", "Security/Proofs"):
        import gzip, hashlib
        tag = "audit-" + "".join(c if c.isalnum() else "_" for c in rel)
        (out / tag).mkdir(parents=True, exist_ok=True)
        gz = out / tag / "unread-facts.jsonl.gz"
        gz.write_bytes(gzip.compress((json.dumps({"module": rel, "found": True}) + "\\n").encode(), mtime=0))
        rep["unread_facts"] = {"file": f"{tag}/unread-facts.jsonl.gz", "sha256": hashlib.sha256(gz.read_bytes()).hexdigest()}
    reps.append(rep)
''')
#: a stand-in for `unread.py`'s `check`: the modules the joined facts hold, against a cap of their number
FAKE_UNREAD = r'''
import gzip, json
from pathlib import Path
CAPS = Path(__file__).resolve().parent / "unread-caps.json"
def check(root, files, facts, out, caps=CAPS):
    mods = [json.loads(line)["module"] for f in facts for line in gzip.open(f, "rt")]
    out.mkdir(parents=True, exist_ok=True)
    (out / "seen.json").write_text(json.dumps(mods))
    cap = json.loads(caps.read_text())
    return {"failures": [f"unread: {len(mods)} above {cap}"] if len(mods) > cap else [], "lower": len(mods) < cap,
            "seconds": 0.0, "areas": {"A": {"count": {"modules": len(mods), "declarations": 0}, "cap": cap}}}
'''
SPEC = {"lakefile.toml": "", "lean-toolchain": "v1\n", "lake-manifest.json": json.dumps({"packages": []}), "lean-audit.json": "{}"}


def test_the_ratchet_joins_the_facts_of_this_run_s_audits_or_of_the_passes_it_reuses(tmp_path, monkeypatch):
    """The ratchet on wholly unread modules runs after the audits on both packages' facts: those an audit wrote, else
    those kept beside the reused pass; a pass kept without its facts, or with other bytes than its entry names, is audited
    again; the caps and `unread.py` rerun no audit; a run with one of the two packages leaves the ratchet to another."""
    extra = {"tools/verity/lean/check.py": FACTS_AUDIT, "tools/verity/lean/unread.py": FAKE_UNREAD,
             "tools/verity/lean/unread-caps.json": "2", "pkg/b/lean-audit.json": "{}",
             "Security/S.lean": "", "Security/Proofs/P.lean": "",
             **{f"Security/{f}": t for f, t in SPEC.items()}, **{f"Security/Proofs/{f}": t for f, t in SPEC.items()}}
    assert FACTS_AUDIT != FAKE_AUDIT
    root = _tree(tmp_path, monkeypatch, extra)
    cache = tmp_path / "cache" / "lean-audit"

    def run(n, **kw):
        out = tmp_path / f"out{n}"
        rc = LA.run(out, cache, tmp_path / "cache" / "lean-deps", log=lambda s: None, **kw)
        ran = sorted(x.split()[0] for x in (out / "ran.txt").read_text().splitlines()) if (out / "ran.txt").exists() else []
        seen = json.loads((out / "unread" / "seen.json").read_text()) if (out / "unread" / "seen.json").exists() else None
        return rc, ran, json.loads((out / "audit.json").read_text()), seen

    keys = {p: LA.key(p, LA.COMMON.tracked_files(), LA.FLAGS) for p in ("Security", "Security/Proofs", "pkg/a")}
    rc, ran, doc, seen = run(1)
    assert rc == 0 and ran == ["Security", "Security/Proofs", "pkg/a", "pkg/b"] and seen == ["Security", "Security/Proofs"]
    assert doc["unread"]["failures"] == [] and sorted(p.stem for p in (cache / "unread").glob("*.json")) == sorted(
        [keys["Security"], keys["Security/Proofs"]])
    published = sorted(f"{keys[p]}.jsonl.gz" for p in LA.UNREAD_FACTS)
    assert sorted(f.name for f in (tmp_path / "out1" / "unread").glob("*.jsonl.gz")) == published, \
        "an audit's facts are left by their key, as the landing gate looks for them"
    rc, ran, doc, seen = run(2)
    assert rc == 0 and ran == [] and seen == ["Security", "Security/Proofs"] and len(doc["cached"]) == 4
    assert sorted(f.name for f in (tmp_path / "out2" / "unread").glob("*.jsonl.gz")) == published, "a reused pass's too"
    (root / "tools/verity/lean/unread-caps.json").write_text("1")
    rc, ran, doc, seen = run(3)
    assert rc == 1 and ran == [] and doc["failed"] == ["unread"] and doc["unread"]["failures"] == ["unread: 2 above 1"]
    (root / "tools/verity/lean/unread-caps.json").write_text("2")
    k = keys["Security/Proofs"]
    entry = cache / "unread" / f"{k}.json"
    entry.write_text(json.dumps(json.loads(entry.read_text()) | {"facts": "e30K"}))
    rc, ran, doc, seen = run(4)
    assert rc == 0 and ran == ["Security/Proofs"] and seen == ["Security", "Security/Proofs"], "other bytes: audited again"
    entry.unlink()
    assert not LA.reusable(cache, "Security/Proofs", k) and LA.reusable(cache, "pkg/a", keys["pkg/a"])
    rc, ran, doc, seen = run(5)
    assert rc == 0 and ran == ["Security/Proofs"] and entry.is_file()
    rc, ran, doc, seen = run(6, only=["Security/Proofs"])
    assert rc == 0 and doc["unread"] is None and seen is None
    assert LA.prune_facts(cache, kept=1) == 1 and len(list((cache / "unread").glob("*.json"))) == 1


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


def test_a_bundle_lists_what_it_holds_built_of_each_package_mathlibs_cache_does_not_supply(tmp_path):
    """`modules`, which `--export` lists beside each bundle and `--pin` pins: the built modules (`.olean`, not `.ilean` or a
    module's server part) and the roots of each package outside Mathlib's cache, a package nothing was built of included,
    and nothing of Mathlib's or of what its manifest pins."""
    src = _packages_dir(tmp_path / "deps" / "k1")
    for f in ("Arklib/.lake/build/lib/lean/ArkLib/Data/Code.olean", "Arklib/.lake/build/lib/lean/ArkLib/Data/Code.ilean",
              "Arklib/.lake/build/lib/lean/ArkLib/Data/Code.olean.server", "Arklib/lakefile.lean", "VCVio/VCVio.lean",
              "VCVio/ToMathlib/Control.lean"):
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_text("")
    assert LA.modules(src) == {"Arklib": {"roots": ["ArkLib"], "built": ["ArkLib", "ArkLib.Data.Code"]},
                               "VCVio": {"roots": ["VCVio"], "built": []}}


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
    def pins(p):
        return (root / LA.PIN).write_text(json.dumps({"schema": "lean-deps-pin/v1", "deps": p}))
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
    entry = {"package": "pkg/a", "sha256": "ab" * 32, "bytes": 5, "modules": {"Arklib": {"roots": ["ArkLib"], "built": []}}}
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
    (tmp_path / "rec" / "lean-deps" / "lean-deps.json").write_text(
        json.dumps({"deps": {k: {x: v for x, v in entry.items() if x != "modules"}}}))
    assert LA.pin("r20260929-000000-abcd") == 1 and "lists no modules" in capsys.readouterr().err
    (tmp_path / "rec" / "lean-deps" / "lean-deps.json").write_text(json.dumps({"deps": {k: entry}}))
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
    monkeypatch.setattr(LA, "shared_bundles", lambda: ["ef" * 32])
    p = LA.url_file(tmp_path, ttl_s=60)
    assert json.loads(p.read_text()) == {k: f"https://r2/objects/sha256/{'ab' * 32}?sig",
                                         LA.SHARED + "ef" * 32: f"https://r2/objects/sha256/{'ef' * 32}?sig"}
    assert signed == [(f"objects/sha256/{'ab' * 32}", 60), (f"objects/sha256/{'ef' * 32}", 60)]
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


def test_kept_dependencies_and_replays_past_their_caps_go_oldest_first(tmp_path):
    """Kept dependency trees and replay entries are evicted, or a node fills up with them. Kept dependencies keep the newest `kept` by use, a held one stays, and the replay cache its newest entries."""
    deps = tmp_path / "lean-deps"
    for i, k in enumerate(("a", "b", "c")):
        (deps / k / "pkg").mkdir(parents=True)
        os.utime(deps / k, (1000 + i, 1000 + i))
    held = open(deps / "a.lock", "w")
    import fcntl
    fcntl.flock(held, fcntl.LOCK_EX)
    assert LA.WarmDeps(deps, kept=1).evict() == 1 and sorted(p.name for p in deps.iterdir() if p.is_dir()) == ["a", "c"]
    held.close()
    assert LA.WarmDeps(deps, kept=1).evict() == 1 and [p.name for p in deps.iterdir() if p.is_dir()] == ["c"]
    replay = tmp_path / "lean-replay"
    replay.mkdir()
    for i in range(5):
        (replay / f"{i}.json").write_text("{}")
        os.utime(replay / f"{i}.json", (2000 + i, 2000 + i))
    assert LA.prune_replays(replay, kept=2) == 3 and sorted(p.name for p in replay.iterdir()) == ["3.json", "4.json"]
    assert LA.prune_replays(tmp_path / "absent", kept=2) == 0


def test_records_trees_beyond_the_cap_go_least_recently_used_first_unless_a_run_holds_one(tmp_path):
    """Nine idle `lean-records` trees, about 490k files each, take a node to 95% of its inodes.
    `prune_records` keeps the most recently used (by `.who`, `links.json`, `.lock`); an older one whose `.lock` a run holds
    stays; a link to a tree elsewhere goes with its target."""
    import fcntl
    store, away = tmp_path / "lean-records", tmp_path / "moved"
    store.mkdir()
    away.mkdir()
    for i, name in enumerate(["old", "held", "linked", "recent"]):
        tree = (away if name == "linked" else store) / name
        tree.mkdir()
        (tree / "f").write_text("x")
        (tree / ".who").write_text("r")
        os.utime(tree / ".who", (1000 + i, 1000 + i))
        if name == "linked":
            (store / name).symlink_to(tree)
    with open(store / "held" / ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        os.utime(store / "held" / ".lock", (1001, 1001))
        assert LA.prune_records(store, kept=1, slots=0) == 2
    assert sorted(p.name for p in store.iterdir()) == ["held", "recent"] and not (away / "linked").exists()
    assert LA.prune_records(store, kept=1, slots=0) == 1 and [p.name for p in store.iterdir()] == ["recent"]
    assert LA.prune_records(tmp_path / "absent", slots=0) == 0


def test_a_node_with_more_build_slots_than_records_kept_keeps_a_tree_per_slot_and_its_passes(tmp_path):
    """Trees seeded for `build 8` keep their source's old `.who`, and an hourly prune that keeps 3 would take some of them,
    so a waiting run would build Mathlib from scratch on the root disk. The slots' own trees stay, the
    ones a lowered cap left (`build-8`) go first whatever their use, and `passes/` is not a tree."""
    store = tmp_path / "lean-records"
    for i, name in enumerate(["build-8", *(f"build-{j}" for j in range(8)), "passes"]):
        (store / name).mkdir(parents=True)
        (store / name / ".who").write_text("r")
        t = 2000 - 100 * (name != "build-8") - i
        os.utime(store / name / ".who", (t, t))
    slots = tmp_path / "lean-slots"
    slots.write_text("check 4\nbuild 8   # one per tree\n")
    assert LA.build_slots(slots) == 8 and LA.build_slots(tmp_path / "absent") == 0
    assert LA.prune_records(store, slots=8) == 1
    assert sorted(p.name for p in store.iterdir()) == sorted([*(f"build-{j}" for j in range(8)), "passes"])
    slots.write_text("build 2\n")
    assert LA.prune_records(store, slots=LA.build_slots(slots)) == 5
    assert sorted(p.name for p in store.iterdir()) == ["build-0", "build-1", "build-2", "passes"]
