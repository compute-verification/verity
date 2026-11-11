"""`tools/verity/lean/lean_audit.py`: audit verdicts reused only while every input is unchanged, the hermetic scratch tree as the read
guard, and dependencies kept warm across runs."""
from __future__ import annotations

import ast
import fnmatch
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
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
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
    reps[-1]["sandbox"] = True
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
#: the flag passes on the record it writes under --out (`check.py`'s `settled` and `regenerated`). It stores no facts
#: document (`--facts-cache`), as check.py on a node without the sandbox
REGEN_AUDIT = r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
root, out = Path.cwd(), Path(a[a.index("--out") + 1])
named = [Path(x) for i, x in enumerate(a) if not x.startswith("--") and not (i and a[i - 1] in ("--out", "--repo", "--replay-cache", "--artifacts", "--facts-cache"))]
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
named = [Path(x) for i, x in enumerate(a) if not x.startswith("--") and not (i and a[i - 1] in ("--out", "--repo", "--replay-cache", "--artifacts", "--bookkeeping", "--facts-cache"))]
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


#: a stand-in for check.py as a lock run and the audit of its lock commit run it. A package's build is its `.lean` files'
#: bytes, and its lock records their hash for each guarantee it lists; a record that differs is a record difference, which
#: `--regenerate` settles on the lock it writes under --out. `--facts-cache DIR` stores each audit's facts document (the
#: hash) and sidecar under its facts key, as check.py does where builds are sandboxed. `--facts-from FILE` judges the named
#: package's lock from that document, with no build, and notes the locks it read of the packages it requires by path. A
#: lock's `regenerates_as` (a record) and `regenerates_with` (sections) say what its regenerated lock holds instead
LOCK_AUDIT = r'''
import gzip, hashlib, json, sys
from pathlib import Path
a = sys.argv[1:]
root, out = Path.cwd(), Path(a[a.index("--out") + 1])
named = [Path(x) for i, x in enumerate(a) if not x.startswith("--") and not (i and a[i - 1] in (
    "--out", "--repo", "--replay-cache", "--artifacts", "--bookkeeping", "--facts-cache", "--facts-from"))]
out.mkdir(parents=True, exist_ok=True)
docs = Path(a[a.index("--facts-cache") + 1]) if "--facts-cache" in a else None
judged = Path(a[a.index("--facts-from") + 1]) if "--facts-from" in a else None
if judged is None:
    (out / "argv.json").write_text(json.dumps(a))
reps, failed = [], []
for pkg in named:
    lock, rel = json.loads((pkg / "lean-audit.json").read_text()), str(pkg.relative_to(root))
    with open(out / ("ran.txt" if judged is None else "judged.txt"), "a") as f:
        f.write(rel + "\n")
    if judged is None:
        built = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(pkg.glob("*.lean")))).hexdigest()[:12]
        rep = {"package": str(pkg), "failures": [], "replay": {"checked": 1}, "seconds": {"replay": 1.0, "total": 2.0}}
        if docs is not None and not lock.get("unsandboxed"):
            docs.mkdir(parents=True, exist_ok=True)
            data = gzip.compress(json.dumps({"schema": "FACTS_SCHEMA", "built": built}).encode(), mtime=0)
            key = hashlib.sha256(rel.encode() + data).hexdigest()
            (docs / f"{key}.json.gz").write_bytes(data)
            (docs / f"{key}.json").write_text(json.dumps({"schema": "FACTS_SCHEMA", "key": key,
                                                          "sha256": hashlib.sha256(data).hexdigest()}))
            rep |= {"facts_key": key, "facts_document": str(docs / f"{key}.json.gz")}
    else:
        built = json.loads(gzip.decompress(judged.read_bytes()))["built"]
        rep = {"package": str(pkg), "failures": [], "seconds": {"total": 0.5}}
        (out / f"saw-{pkg.name}.json").write_text(json.dumps({
            d["dir"]: json.loads((pkg / d["dir"] / "lean-audit.json").read_text())
            for d in json.loads((pkg / "lake-manifest.json").read_text())["packages"] if d["type"] == "path"}))
    stale = [f"guarantees: {n}'s record differs from the build" for n, r in sorted(lock.get("guarantees", {}).items())
             if r != built]
    if stale and judged is None and "--regenerate" in a:
        rec = out / f"audit-{pkg.name}" / "lean-audit.json"
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(json.dumps(lock | {"guarantees": {n: lock.get("regenerates_as", built) for n in lock["guarantees"]},
                                          "format": 1} | lock.get("regenerates_with", {})))
        rep |= {"settled": stale, "regenerated": str(rec)}
    else:
        rep["failures"] = stale
    if rep["failures"]:
        failed.append(str(pkg))
    reps.append(rep)
(out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "packages": reps}))
sys.exit(1 if failed else 0)
'''.replace("FACTS_SCHEMA", AUDIT.FACTS_SCHEMA)
LOCK_PACKAGES = ["pkg/a", "pkg/b", "pkg/c", "pkg/p", "pkg/s"]


def _lock_tree(tmp: Path, monkeypatch, locks: dict[str, dict] | None = None) -> Path:
    """A tree under `LOCK_AUDIT` whose every lock records its build: `pkg/a`, `pkg/b` which requires it by path, `pkg/c`
    which requires nothing, and a spec `pkg/s` whose guarantees `pkg/p`, which requires it, proves. ``locks`` ({package
    name: sections}) are laid over those locks: a guarantee recorded as anything else is a record difference."""
    def package(name: str, requires: tuple[str, ...] = (), **sections) -> dict[str, str]:
        source = f"theorem {name} : True := trivial\n"
        lock = {"roots": [name.upper()], "guarantees": {f"{name.upper()}.t": hashlib.sha256(source.encode()).hexdigest()[:12]}}
        return {f"pkg/{name}/lakefile.toml": "", f"pkg/{name}/lean-toolchain": "v1\n", f"pkg/{name}/{name.upper()}.lean": source,
                f"pkg/{name}/lake-manifest.json": json.dumps({"packages": [{"name": r, "type": "path", "dir": f"../{r}"}
                                                                           for r in requires]}),
                f"pkg/{name}/lean-audit.json": json.dumps(lock | sections | (locks or {}).get(name, {}))}
    return _tree(tmp, monkeypatch, {"tools/verity/lean/check.py": LOCK_AUDIT, **package("a"), **package("b", ("a",)),
                                    **package("c"), **package("s", proved_in=["../p"]), **package("p", ("s",))})


def _lock_run(tmp: Path, name: str, **flags) -> tuple[int, Path, dict, list[str]]:
    """`LA.run` into ``tmp``'s ``name``, on the caches every run of a test shares: (its exit, its --out, its audit.json,
    its log)."""
    out, logged = tmp / name / "lean-audit", []
    rc = LA.run(out, tmp / "cache" / "lean-audit", tmp / "cache" / "lean-deps", log=logged.append, **flags)
    return rc, out, json.loads((out / "audit.json").read_text()), logged


def _lines(path: Path) -> list[str]:
    return sorted(path.read_text().split()) if path.exists() else []


def _commit_locks(root: Path, doc: dict) -> None:
    """The commit drive makes of a lock run's regenerated locks: each over its package's own."""
    for p, f in doc["regenerated"].items():
        shutil.copyfile(f, root / p / "lean-audit.json")


def test_the_audit_of_the_lock_commit_reuses_the_lock_runs_passes_and_builds_nothing(tmp_path, monkeypatch):
    """A lock run judges, from the facts its own audits stored, every package whose key its regenerated locks move, on the
    tree with those locks in place: the packages regenerated, the one requiring one of them by path, and the spec whose
    prover's lock changed. Each judgment that leaves nothing is kept as that package's pass under its key on that tree,
    so the audit of the commit drive makes of the locks reuses every pass and runs no build. A kept pass is a full
    audit's: the judgment of the committed lock with the lock run's replay, which `check.lean_fast` requires of a pass."""
    root = _lock_tree(tmp_path, monkeypatch, {"a": {"guarantees": {"A.t": "stale"}}, "p": {"guarantees": {"P.t": "stale"}}})
    before = {p: LA.key(p, LA.COMMON.tracked_files(root), LA.FLAGS) for p in LOCK_PACKAGES}
    stale = (root / "pkg/a/lean-audit.json").read_text()
    rc, out, doc, logged = _lock_run(tmp_path, "lock-run", regenerate=True)
    assert rc == 0 and sorted(doc["regenerated"]) == ["pkg/a", "pkg/p"] and _lines(out / "ran.txt") == LOCK_PACKAGES
    assert sorted(doc["rekeyed"]) == ["pkg/a", "pkg/b", "pkg/p", "pkg/s"], logged
    assert _lines(out / LA.LOCK_COMMIT / "judged.txt") == sorted(doc["rekeyed"]), "each judged once, and only those"
    assert _lines(out / "ran.txt") == LOCK_PACKAGES, "the judgments build nothing"
    # judged on the tree with the regenerated locks in place, which is never this checkout
    regenerated = json.loads(Path(doc["regenerated"]["pkg/a"]).read_text())
    assert json.loads((out / LA.LOCK_COMMIT / "saw-b.json").read_text()) == {"../a": regenerated}
    assert (root / "pkg/a/lean-audit.json").read_text() == stale
    # the lock run's own verdict and reports are as before: its pass on a regenerated lock is no pass of this tree
    a = next(r for r in doc["packages"] if r["package"] == "pkg/a")
    assert a["settled"] and a["failures"] == [] and not (tmp_path / "cache" / "lean-audit" / f"{before['pkg/a']}.json").exists()
    # the facts documents were the run's own, outside all a build writes, and are gone with it
    argv = json.loads((out / "argv.json").read_text())
    documents = Path(argv[argv.index("--facts-cache") + 1])
    assert documents.name.startswith(LA.SCRATCH) and not documents.exists()
    assert AUDIT.build_roots_over(documents, [Path(x) for x in argv if x.endswith(tuple(LOCK_PACKAGES))]) == []
    assert not list((tmp_path / "cache").glob(LA.SCRATCH + "*")) and not (out / LA.LOCK_COMMIT / "tool").exists()

    _commit_locks(root, doc)
    after = {p: LA.key(p, LA.COMMON.tracked_files(root), LA.FLAGS) for p in LOCK_PACKAGES}
    assert doc["rekeyed"] == {p: k for p, k in after.items() if k != before[p]}, "the keys of the tree drive commits"
    rc, out2, doc2, _ = _lock_run(tmp_path, "lock-commit")
    assert rc == 0 and doc2["cached"] == after and doc2["regenerated"] == {}
    assert _lines(out2 / "ran.txt") == [] and not (out2 / LA.LOCK_COMMIT).exists(), "no build and no judgment: only reuse"
    for r in doc2["packages"]:
        assert r["failures"] == [] and "settled" not in r and "regenerated" not in r and "facts_document" not in r
        assert r["replay"] == {"checked": 1} and r["seconds"]["replay"] == 1.0
        assert r.get("judged_from") == (before[r["package"]] if r["package"] in doc["rekeyed"] else None)
    assert CHECK.lean_fast(out2.parent) is None


def test_a_regenerated_lock_that_fails_its_judgment_keeps_no_pass(tmp_path, monkeypatch):
    """A package whose regenerated lock fails when it is judged on the tree with the locks in place keeps no pass under
    that tree's key, so the audit of the lock commit audits it in full (and fails it, as the judgment did). The lock run
    itself passes as before, and a package whose own judgment there leaves nothing still keeps its pass."""
    root = _lock_tree(tmp_path, monkeypatch, {"a": {"guarantees": {"A.t": "stale"}, "regenerates_as": "not the build's"}})
    rc, out, doc, logged = _lock_run(tmp_path, "lock-run", regenerate=True)
    assert rc == 0 and list(doc["regenerated"]) == ["pkg/a"] and list(doc["rekeyed"]) == ["pkg/b"]
    assert _lines(out / LA.LOCK_COMMIT / "judged.txt") == ["pkg/a", "pkg/b"]
    assert [s for s in logged if s.startswith("lean-audit: pkg/a: no pass kept") and "it fails (exit 1, 1 failures" in s], logged
    assert (out / LA.LOCK_COMMIT / "pkg_a.log").is_file(), "a judgment's output goes to a file, not into the lock run's log"
    _commit_locks(root, doc)
    rc, out2, doc2, _ = _lock_run(tmp_path, "lock-commit")
    assert rc == 1 and doc2["failed"] == ["pkg/a"] and _lines(out2 / "ran.txt") == ["pkg/a"]
    assert sorted(doc2["cached"]) == ["pkg/b", "pkg/c", "pkg/p", "pkg/s"]


def test_a_package_whose_key_the_regenerated_locks_leave_is_untouched(tmp_path, monkeypatch):
    """A package that reads no regenerated lock has the same key on the tree with them: the lock run judges it no second
    time and writes nothing for it, and its pass, stored by its audit or reused from an earlier one, answers on the lock
    commit as it is."""
    root = _lock_tree(tmp_path, monkeypatch, {"a": {"guarantees": {"A.t": "stale"}}})
    cache = tmp_path / "cache" / "lean-audit"
    keys = {p: LA.key(p, LA.COMMON.tracked_files(root), LA.FLAGS) for p in LOCK_PACKAGES}
    rc, out, doc, _ = _lock_run(tmp_path, "lock-run", regenerate=True)
    assert rc == 0 and sorted(doc["rekeyed"]) == ["pkg/a", "pkg/b"]
    assert _lines(out / LA.LOCK_COMMIT / "judged.txt") == ["pkg/a", "pkg/b"]
    kept = {p: (cache / f"{keys[p]}.json").read_bytes() for p in ("pkg/c", "pkg/p", "pkg/s")}
    assert all(json.loads(e)["report"]["seconds"] == {"replay": 1.0, "total": 2.0} and "judged_from" not in json.loads(e)["report"]
               for e in kept.values()), "the passes their own audits stored"
    assert sorted(f.name for f in cache.glob("*.json")) == sorted(
        f"{k}.json" for k in [*(keys[p] for p in kept), keys["pkg/b"], *doc["rekeyed"].values()])
    # a second lock run reuses them, and those it reused are as untouched
    rc, again, doc, _ = _lock_run(tmp_path, "lock-run-again", regenerate=True)
    assert rc == 0 and sorted(doc["cached"]) == ["pkg/b", "pkg/c", "pkg/p", "pkg/s"] and _lines(again / "ran.txt") == ["pkg/a"]
    assert {p: (cache / f"{keys[p]}.json").read_bytes() for p in kept} == kept
    _commit_locks(root, doc)
    assert {p: LA.key(p, LA.COMMON.tracked_files(root), LA.FLAGS) for p in kept} == {p: keys[p] for p in kept}
    rc, out2, doc2, _ = _lock_run(tmp_path, "lock-commit")
    assert rc == 0 and _lines(out2 / "ran.txt") == [] and {p: doc2["cached"][p] for p in kept} == {p: keys[p] for p in kept}


def test_a_regenerated_lock_that_changes_what_the_build_reads_keeps_no_pass(tmp_path, monkeypatch):
    """The facts a lock run stored are the facts of the tree with the regenerated locks only while every step before the
    judge reads the same of both locks (`check.py`'s `build_inputs`). A regenerated lock that newly lists a premise, which
    `Facts.lean` then walks, is no such lock: its package and the one whose audit reads that lock keep no pass, though a
    judgment from the old facts would leave nothing, and the lock commit audits both in full."""
    root = _lock_tree(tmp_path, monkeypatch, {"a": {"guarantees": {"A.t": "stale"},
                                                    "regenerates_with": {"assumptions": {"A.H": {"owner": "@lean"}}}}})
    rc, out, doc, logged = _lock_run(tmp_path, "lock-run", regenerate=True)
    assert rc == 0 and list(doc["regenerated"]) == ["pkg/a"] and doc["rekeyed"] == {}
    assert not (out / LA.LOCK_COMMIT).exists(), "nothing is judged"
    for p in ("pkg/a", "pkg/b"):
        assert [s for s in logged if s.startswith(f"lean-audit: {p}: no pass kept") and "the regenerated lock of pkg/a" in s
                and "build_inputs" in s], logged
    _commit_locks(root, doc)
    rc, out2, doc2, _ = _lock_run(tmp_path, "lock-commit")
    assert rc == 0 and _lines(out2 / "ran.txt") == ["pkg/a", "pkg/b"] and sorted(doc2["cached"]) == ["pkg/c", "pkg/p", "pkg/s"]

    # a lock `build_inputs` can't read (it names no roots) agrees with no lock, itself included
    root = _lock_tree(tmp_path / "unread", monkeypatch)
    (root / "pkg/a/lean-audit.json").write_text(json.dumps({"guarantees": {"A.t": "stale"}}))
    rc, out, doc, logged = _lock_run(tmp_path / "unread", "lock-run", regenerate=True)
    assert rc == 0 and list(doc["regenerated"]) == ["pkg/a"] and doc["rekeyed"] == {} and not (out / LA.LOCK_COMMIT).exists()
    assert len([s for s in logged if "no pass kept" in s and "the regenerated lock of pkg/a" in s]) == 2, logged


def test_only_a_whole_run_keeps_a_pass_and_only_of_a_package_it_audited_with_a_stored_document(tmp_path, monkeypatch):
    """What keeps no pass for the lock commit, each leaving that commit's audit to run in full as before: a package the
    lock run reused from the cache (it has no facts of this run), one whose audit stored no facts document (a node without
    the sandbox), a shard's run (another shard's regenerated locks move its keys too) and a `--fresh` run (nothing is
    cached). Neither of the last two asks check.py for documents."""
    stale = {"a": {"guarantees": {"A.t": "stale"}}}
    _lock_tree(tmp_path / "reused", monkeypatch, stale)
    rc, _, doc, _ = _lock_run(tmp_path / "reused", "check")           # a fails on its record; the others' passes are kept
    assert rc == 1 and doc["failed"] == ["pkg/a"]
    rc, out, doc, logged = _lock_run(tmp_path / "reused", "lock-run", regenerate=True)
    assert rc == 0 and sorted(doc["cached"]) == ["pkg/b", "pkg/c", "pkg/p", "pkg/s"] and list(doc["rekeyed"]) == ["pkg/a"]
    assert [s for s in logged if s.startswith("lean-audit: pkg/b: no pass kept") and "reused its pass" in s], logged

    _lock_tree(tmp_path / "unsandboxed", monkeypatch, {"a": stale["a"] | {"unsandboxed": True}})
    rc, out, doc, logged = _lock_run(tmp_path / "unsandboxed", "lock-run", regenerate=True)
    assert rc == 0 and list(doc["regenerated"]) == ["pkg/a"] and list(doc["rekeyed"]) == ["pkg/b"]
    assert [s for s in logged if s.startswith("lean-audit: pkg/a: no pass kept") and "stored no facts document" in s], logged

    _lock_tree(tmp_path / "partial", monkeypatch, stale)
    for name, flags in (("shard", {"only": ["pkg/a", "pkg/b"]}), ("fresh", {"fresh": True})):
        rc, out, doc, _ = _lock_run(tmp_path / "partial", name, regenerate=True, **flags)
        assert rc == 0 and list(doc["regenerated"]) == ["pkg/a"] and doc["rekeyed"] == {}
        assert "--facts-cache" not in json.loads((out / "argv.json").read_text()) and not (out / LA.LOCK_COMMIT).exists()
    assert "rekeyed" not in _lock_run(tmp_path / "partial", "check")[2], "only a run that regenerates lists them"


def test_a_facts_document_is_judged_from_only_as_its_sidecar_names_it(tmp_path):
    """`stored_document` gives the document a lock run's audit stored under a facts key only when its sidecar names the
    schema, that key and the SHA-256 the document's bytes have: never one whose bytes changed since, one filed under
    another key, one of another schema, or one without its sidecar (`check.py`'s `read_cached_facts`). Without the
    document it says why, which the lock run's log carries."""
    def store(name: str, data: bytes, **meta) -> None:
        (tmp_path / f"{name}.json.gz").write_bytes(data)
        (tmp_path / f"{name}.json").write_text(json.dumps({"schema": "s/v1", "key": name,
                                                           "sha256": hashlib.sha256(data).hexdigest()} | meta))

    def refused(name: str, schema: str = "s/v1") -> str:
        doc, why = LA.stored_document(tmp_path, name, schema)
        assert doc is None and why
        return why

    store("good", b"facts")
    assert LA.stored_document(tmp_path, "good", "s/v1") == (tmp_path / "good.json.gz", "")
    assert "another schema" in refused("good", "s/v2") and refused("absent").startswith("FileNotFoundError")
    (tmp_path / "good.json.gz").write_bytes(b"other facts")
    assert "SHA-256" in refused("good"), "its bytes changed"
    store("misfiled", b"facts", key="another")
    assert "key" in refused("misfiled")
    store("bare", b"facts")
    (tmp_path / "bare.json").unlink()
    assert refused("bare").startswith("FileNotFoundError")
    (tmp_path / "bare.json").write_text("[1]")
    assert refused("bare").startswith("AttributeError") and refused("None").startswith("FileNotFoundError")
    (tmp_path / "bare.json").write_text("not json")
    assert refused("bare").startswith("JSONDecodeError")


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


def test_a_packages_prose_and_pytest_files_are_no_input(tmp_path, monkeypatch):
    """No audit reads a package's `*.md`, `.gitignore`, `pyproject.toml`, `test_*.py` or `conftest.py` (`UNREAD_NAMES`), so
    an edit to one audits no package again: two READMEs and a pytest file inside packages audited all three `security`
    packages cold in a candidate that changed no Lean. A generator, a data file or a fixture the package keeps, under
    `tests/` too, still audits it again. The dropped files stay out of the scratch tree, so an audit that read one fails."""
    unread = {"pkg/a/README.md": "", "pkg/a/docs/NOTES.md": "", "pkg/a/.gitignore": "", "pkg/a/pyproject.toml": "",
              "pkg/a/tests/test_a.py": "", "pkg/a/tests/conftest.py": ""}
    kept = {"pkg/a/gen.py": "", "pkg/a/tests/difftest_vectors.py": "", "pkg/a/tests/fixtures/vectors.json": "{}",
            "pkg/a/models.json": "{}", "pkg/a/TRUSTED.sha256": ""}
    root = _tree(tmp_path, monkeypatch, {**unread, **kept})
    (root / "pkg/b/lean-audit.json").write_text("{}")
    assert _run(tmp_path, 1)[0] == 0
    files = LA.COMMON.tracked_files()
    got = LA.inputs("pkg/b", files)
    assert not set(unread) & set(got) and set(kept) <= set(got)
    keys = {p: LA.key(p, files, LA.FLAGS) for p in ("pkg/a", "pkg/b")}
    for i, f in enumerate(sorted(unread), 2):
        (root / f).write_text(f"edited {i}\n")
        assert {p: LA.key(p, files, LA.FLAGS) for p in keys} == keys, f
        rc, ran, doc = _run(tmp_path, i)
        assert rc == 0 and ran == [] and sorted(doc["cached"]) == ["pkg/a", "pkg/b"], f
    for i, f in enumerate(sorted(kept), 20):
        k = LA.key("pkg/a", files, LA.FLAGS)
        (root / f).write_text(f"edited {i}\n")
        assert LA.key("pkg/a", files, LA.FLAGS) != k, f
        rc, ran, _doc = _run(tmp_path, i)
        assert rc == 0 and {r.split()[0] for r in ran} == {"pkg/a", "pkg/b"}, f
    (root / "pkg/b/lean-audit.json").write_text(json.dumps({"reads": sorted(unread)}))
    rc, _ran, doc = _run(tmp_path, 99)
    assert rc == 1 and doc["failed"] == ["pkg/b"]
    assert doc["packages"][1]["failures"] == [f"reads {f}" for f in sorted(unread)]


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


def test_what_a_failed_or_a_killed_audit_compiled_goes_into_the_artifact_cache_on_check_py_s_word(tmp_path, monkeypatch):
    """A module is stored for what its compile read, so a package that failed (a record, a later module) and a check.py
    that was killed store what they compiled, as a pass does. What says the build ran in the sandbox is check.py's own:
    its report, or, where it left none, what it wrote in the package's work directory as the build began."""
    root = _tree(tmp_path, monkeypatch)
    put_with = []
    monkeypatch.setattr(LA.LakeArtifacts, "put", lambda self, reports, since=None, log=print: put_with.append(
        {p.name: {k: v for k, v in rep.items() if k != "package"} for p, rep in reports.items()}) or 0)
    assert _run(tmp_path, 1)[0] == 1, "b fails: it reads a file outside its inputs"
    assert put_with == [{"a": {"sandbox": True, "failures": []},
                         "b": {"sandbox": True, "failures": ["reads docs/outside.txt"]}}]
    (root / "tools/verity/lean/check.py").write_text(FAKE_AUDIT.replace(
        'reps, failed = [], []', 'reps, failed = [], []\nfor n, pkg in enumerate(named):\n'
        '    (out / f"audit-{n}").mkdir()\n'
        f'    (out / f"audit-{{n}}" / "{LA.BUILDING}").write_text(json.dumps({{"package": str(pkg), "sandbox": n == 0}}))\n'
        'sys.exit(137)'))
    assert LA.run(tmp_path / "out2", tmp_path / "cache" / "lean-audit", tmp_path / "cache" / "lean-deps",
                  log=lambda s: None) == 1
    assert put_with[1:] == [{"a": {"sandbox": True}, "b": {"sandbox": False}}], "killed: no report, so what each build began with"


def test_what_check_py_said_of_a_build_is_its_report_or_what_it_wrote_as_the_build_began(tmp_path):
    out, a, b = tmp_path / "out", tmp_path / "pkg" / "a", tmp_path / "pkg" / "b"
    assert LA.built(out) == {}
    for n, (pkg, sandbox) in enumerate(((a, True), (b, False))):
        (out / f"audit-{n}").mkdir(parents=True)
        (out / f"audit-{n}" / LA.BUILDING).write_text(json.dumps({"package": str(pkg), "sandbox": sandbox}))
    began = {a: {"package": str(a), "sandbox": True}, b: {"package": str(b), "sandbox": False}}
    assert LA.built(out) == began
    reps = [{"package": str(a), "sandbox": True, "failures": ["sampled: a: differs"]}, {"package": str(tmp_path / "c")},
            {"package": str(tmp_path / "d"), "sandbox": True}]
    (out / "audit.json").write_text(json.dumps({"packages": reps}))
    assert LA.built(out) == began | {a: reps[0], tmp_path / "d": reps[2]}, \
        "a report replaces what the build began with; one that doesn't say `sandbox` is of a package check.py didn't build"
    os.utime(out / "audit.json", ns=(1, 1))
    assert LA.built(out) == began | {tmp_path / "d": reps[2]}, "a report an earlier check.py left is not this build's"
    (out / "audit.json").write_text("{")
    assert LA.built(out) == began
    assert AUDIT.BUILDING == LA.BUILDING


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
    keyed = LA.MK.keys(pkg)
    assert json.loads((store / entries["P.A"]).read_text()) == {
        "schema": "lean-module/v4", "module": "P.A", "scope": "p", "hash": hashlib.sha256(b"P/A").hexdigest()[:16],
        "outputs": json.loads((lib / "P/A.trace").read_text())["outputs"],
        "digests": {n: LA.MK.C.digest(f"P/A.{n.split('.', 1)[1]} {n.split('.')[0]}".encode()) for n in names},
        "inputs": {"vars": {"module": "P.A", "package": "p", "library": "P"}, "env": keyed["P.A"].parts,
                   "files": {f: keyed["P.A"].files[f] for f in ("p/P/A.lean", "p/lakefile.toml")}, "reads": {}}}
    assert LA.MK.last(store, "p", "P.A") == LA.MK.record("P.A", keyed, {}) and LA.MK.last(store, "p", "P.B") is None
    assert sorted(f.name for f in (store / "artifacts").iterdir()) == sorted(names)
    (lib / "P/B.olean.hash").write_text("d" * 16)
    assert arts.put({pkg: SANDBOXED}) == 2 and arts.put({pkg: SANDBOXED}) == 0, "P.B, then P, which imports it"
    assert sorted(f"keys/{f.name}" for f in (store / "keys").iterdir()) == sorted(entries.values())
    assert set(LA.MK.last(store, "p", "P")["reads"]) == {"P.A", "P.B"}, "what its key read of each module it loads"
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


def test_a_module_whose_compile_was_stopped_stays_out_of_the_artifact_cache(tmp_path):
    """A failed or killed run's build is indexed too, so `put` must tell a module whose compile ended from one it was
    stopped in. Lake removes the outputs, runs `lean`, removes each `.hash`, writes them from the new outputs and writes
    the trace last, leaving the earlier trace and `.hash` files until then: at every point before the end the trace isn't
    whole, an output it names is missing or without its `.hash`, or an output is newer than the `.hash` beside it, which
    names the bytes that were there before."""
    pkg = _package(tmp_path)
    lib, n = pkg / ".lake/build/lib/lean", 0

    def put() -> int:
        nonlocal n
        n += 1
        return LA.LakeArtifacts(tmp_path / f"store-{n}").put({pkg: SANDBOXED})

    whole = (lib / "P/A.trace").read_text()
    (lib / "P/A.trace").write_text(whole[:40])
    assert put() == 0, "the trace was being written: P.A stays out, and P.B and P, which load it, after it"
    (lib / "P/A.trace").write_text(whole)
    olean = (lib / "P/A.olean").read_bytes()
    (lib / "P/A.olean").unlink()
    assert put() == 0, "`lean` had not written the output yet"
    (lib / "P/A.olean").write_bytes(olean[:5])
    at = (lib / "P/A.olean.hash").stat().st_mtime_ns
    os.utime(lib / "P/A.olean", ns=(at + 1, at + 1))
    assert put() == 0, "`lean` wrote the output, half of it, and the `.hash` is still the one of the compile before"
    (lib / "P/A.olean.hash").unlink()
    assert put() == 0, "the `.hash` files were removed and not yet written"
    (lib / "P/A.olean").write_bytes(olean)
    (lib / "P/A.olean.hash").write_text("a" * 16)
    assert put() == 3, "whole again"


def test_a_build_whose_sampled_recompile_differs_indexes_nothing(tmp_path):
    """The sampled recompile found a module this build restored or kept that differs from its own compile. What was
    compiled on top of it read those outputs, so nothing of that build goes in."""
    pkg, said = _package(tmp_path), []
    rep = SANDBOXED | {"failures": ["sampled: P.A: the build's P/A.olean differs from the recompile's"]}
    assert LA.LakeArtifacts(tmp_path / "store").put({pkg: rep}, log=said.append) == 0 and "(`sampled`)" in said[0]
    assert LA.LakeArtifacts(tmp_path / "store").put({pkg: SANDBOXED | {"failures": ["records: differ"]}}) == 3


def test_a_build_whose_sources_run_code_while_they_compile_indexes_nothing(tmp_path):
    """In the sandbox a build still writes its own `.lake`, so code a module runs while it compiles can leave any other
    module's outputs, hashes and trace there, as Lake would. A passing audit says its package has no such code, and a
    build is stored whatever its audit came to, so `put` reads the build's sources itself: where one runs code outside
    a listing reviewed at its digest, nothing of the build goes in, and each module it compiled is recorded as left to
    Lake, so the next build of the tree keeps it rather than compile it again."""
    pkg = _package(tmp_path)
    (pkg / "P" / "A.lean").write_text("theorem a : True := trivial\n-- run_cmd in a comment runs nothing\n/- #eval 1 -/\n")
    assert LA.runs_code(pkg) is None
    (pkg / "P" / "B.lean").write_text("import P.A\n\nopen Nat in\nrun_cmd pure ()\n")
    why = LA.runs_code(pkg)
    assert why == "pkg/P/B.lean:4 runs code while it compiles (run_cmd), outside a `compile_time` listing reviewed at its digest"
    store, said = tmp_path / "store", []
    arts = LA.LakeArtifacts(store)
    assert arts.put({pkg: SANDBOXED}, log=said.append) == 0 and not (store / "keys").exists()
    assert not (store / "artifacts").exists() and len(said) == 1 and why in said[0]
    left = [json.loads(f.read_text()) for f in sorted((store / LA.MK.UNSTORED).iterdir())]
    assert sorted(d["module"] for d in left) == ["P", "P.A", "P.B"] and {d["why"] for d in left} == {why}

    def listed(digest: str) -> None:
        (pkg / "lean-audit.json").write_text(json.dumps({"compile_time": {"P.B": {"why": "a reviewed command", "digest": digest}}}))

    listed(hashlib.sha256((pkg / "P" / "B.lean").read_bytes()).hexdigest())
    assert LA.runs_code(pkg) is None and arts.put({pkg: SANDBOXED}, log=said.append) == 3 and len(said) == 1
    assert not list((store / LA.MK.UNSTORED).iterdir()), "indexed, a module is no longer left to Lake"
    (pkg / "P" / "B.lean").write_text("import P.A\n\n#eval 1\n")
    assert "pkg/P/B.lean:3 runs code while it compiles (#eval)" in LA.runs_code(pkg), "the listing was reviewed at another digest"
    (pkg / "lean-audit.json").write_text(json.dumps({"compile_time": {"P.B": "listed, with no digest of what was reviewed"}}))
    assert "pkg/P/B.lean:3" in LA.runs_code(pkg)
    (pkg / "lean-audit.json").write_text("[")
    assert LA.runs_code(pkg) == "pkg: its sources or its lock could not be read (JSONDecodeError)"
    (pkg / "lean-audit.json").unlink()
    (pkg / "P" / "B.lean").write_text("import P.A\n")
    assert LA.runs_code(pkg) is None
    dep = tmp_path / "dep"
    (dep / "D").mkdir(parents=True)
    (dep / "D" / "X.lean").write_text("@[simp] initialize counter : IO.Ref Nat ← IO.mkRef 0\n")
    (pkg / "lake-manifest.json").write_text(json.dumps({"name": "p", "packages": [{"type": "path", "name": "d", "dir": "../dep"}]}))
    assert "dep/D/X.lean:1 runs code while it compiles (initialize)" in LA.runs_code(pkg), \
        "a build of the package writes the `.lake` of each package it requires by path"
    (dep / "D" / "X.lean").write_text("def x := 1\n")
    (dep / "lakefile.lean").write_text("import Lake\n")
    assert LA.runs_code(pkg) == "dep/lakefile.lean runs code whenever Lake loads the package"


def test_a_build_that_compiled_nothing_hashes_nothing(tmp_path, monkeypatch):
    """Every build puts now, and most compile a handful of modules: only those and what they load are read."""
    pkg = _package(tmp_path)
    lib = pkg / ".lake/build/lib/lean"
    for t in lib.rglob("*.trace"):
        os.utime(t, (1000, 1000))
    read = []
    digest = LA.MK.digest
    monkeypatch.setattr(LA.MK, "digest", lambda f: read.append(Path(f)) or digest(f))

    def outputs() -> set[str]:
        return {f.name.split(".")[0] for f in read if pkg / ".lake" in f.parents}

    assert LA.LakeArtifacts(tmp_path / "store").put({pkg: SANDBOXED}, since=1000) == 0 and outputs() == set()
    os.utime(lib / "P/B.trace", (1001, 1001))
    LA.LakeArtifacts(tmp_path / "store").put({pkg: SANDBOXED}, since=1000)
    assert outputs() == {"A", "B"}, "P.B, compiled, and P.A, which it loads; not P"


def test_a_module_a_build_compiled_and_the_cache_could_not_index_is_recorded_under_its_key(tmp_path):
    """The next build takes the trace of a module the cache lacks, so that it is compiled and indexed
    (`module_keys.settle`). One `put` can't index would then be compiled in every build, with everything above it: `put`
    records it under the key it compiled under, which that build reads and leaves the module to Lake. Only a module
    the run compiled whole is recorded, with the code that refused it (`module_keys.rules`), and a later `put` that
    indexes it removes the record."""
    pkg = _package(tmp_path)
    lib, store = pkg / ".lake/build/lib/lean", tmp_path / "store"
    setup = pkg / ".lake/build/ir/P/B.setup.json"
    whole = setup.read_text()
    setup.write_text(json.dumps(json.loads(whole) | {"importArts": {}}))        # Lake read no import where `keys` reads P.A
    arts, keys = LA.LakeArtifacts(store), _keys(pkg)

    def unstored() -> dict[str, str]:
        return {m: json.loads(f.read_text())["why"] for m, e in keys.items()
                if (f := store / LA.MK.UNSTORED / Path(e).name).is_file()}

    assert arts.put({pkg: SANDBOXED}) == 1, "P.A"
    got = unstored()
    assert set(got) == {"P.B", "P"} and got["P"] == "it loads P.B, which is not indexed" and "P.A" in got["P.B"]
    assert json.loads((store / LA.MK.UNSTORED / Path(keys["P"]).name).read_text()) == {
        "module": "P", "scope": "p", "why": got["P"], "by": LA.MK.rules()}
    setup.write_text(whole)
    assert arts.put({pkg: SANDBOXED}) == 2 and unstored() == {}, "indexed now: nothing says otherwise"
    for t in lib.rglob("*.trace"):
        os.utime(t, (1000, 1000))
    (lib / "P/B.olean.hash").write_text("0" * 16)
    other = LA.LakeArtifacts(tmp_path / "store-2")
    assert other.put({pkg: SANDBOXED}, since=1000) == 0 and not (tmp_path / "store-2" / LA.MK.UNSTORED).exists(), \
        "nothing this run compiled"
    os.utime(lib / "P/B.trace", (1001, 1001))
    assert other.put({pkg: SANDBOXED}, since=1000) == 0 and not (tmp_path / "store-2" / LA.MK.UNSTORED).exists(), \
        "P.B's compile was stopped: the next run's repair has it compiled again, and nothing is recorded of it"


def test_a_module_indexable_refuses_is_recorded_with_its_words_though_no_module_loads_it(tmp_path):
    """A build `indexable` refuses is compiled again by every build that finds no record of it, and `P`, the top of its
    build, is loaded by no module `put` indexes. It is recorded under its key with `indexable`'s words all the same, and
    the log counts and names it; a refused module Lake didn't say it built (`"compiled"`) is not recorded."""
    pkg = _package(tmp_path)
    setup = pkg / ".lake/build/ir/P.setup.json"
    setup.write_text(json.dumps(json.loads(setup.read_text()) | {"importArts": {}}))
    store, said, keys = tmp_path / "store", [], _keys(pkg)
    assert LA.LakeArtifacts(store).put({pkg: SANDBOXED | {"compiled": ["P.A", "P.B"]}}, log=said.append) == 2
    assert not (store / LA.MK.UNSTORED).exists() and said == []
    assert LA.LakeArtifacts(store).put({pkg: SANDBOXED}, log=said.append) == 0
    why = "Lake imports [] and not ['P.A', 'P.B'] of the workspace"
    assert json.loads((store / LA.MK.UNSTORED / Path(keys["P"]).name).read_text())["why"] == why
    assert said == [f"lean-audit: {pkg}: 1 modules this run compiled can't be indexed (`indexable`), 1 recorded under "
                    f"their keys: P: {why}"]


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


def test_an_artifact_was_last_used_when_an_entry_that_names_it_was(tmp_path):
    """A build touches the entries it stages, keeps and indexes, never an artifact: each tree that restored one holds
    that file by a hard link, and a newer time on it would be an output newer than its `.hash`. So the oldest artifacts
    by their entries' use go first, and one an entry used lately names stays, however long ago it was written."""
    store = tmp_path / "store"
    (store / "artifacts").mkdir(parents=True)
    (store / "keys").mkdir(parents=True)
    for i, (written, used) in enumerate([(10, 90), (20, 20), (30, 30), (40, 40)]):
        f, e = store / "artifacts" / f"{i}.olean", store / "keys" / f"{i}.json"
        f.write_bytes(b"x" * 10)
        e.write_text(json.dumps({"digests": {f.name: "sha512:0"}}))
        os.utime(f, (written, written))
        os.utime(e, (used, used))
    (store / "keys" / "also.json").write_text(json.dumps({"digests": {"2.olean": "sha512:0", "gone.olean": "sha512:0"}}))
    os.utime(store / "keys" / "also.json", (80, 80))
    (store / "keys" / "torn.json").write_text("{")
    os.utime(store / "keys" / "torn.json", (1, 1))
    (store / "keys" / "gone.json").symlink_to(store / "keys" / "no-such-entry")     # listed, then gone: a build removed it
    assert LA.LakeArtifacts(store, cap=30).prune() == 2
    assert sorted(f.name for f in (store / "artifacts").iterdir()) == ["0.olean", "2.olean"], \
        "written first and used last, 0 stays; 2 stays for the later entry that names it too"
    assert sorted(f.name for f in (store / "keys").iterdir()) == ["0.json", "also.json"], \
        "2's first entry was used no later than what went, so it goes: a miss, and the module is built"


TAKEN = "taken-v4"


def _bundle_of(tmp: Path, pkg: Path) -> tuple[Path, str]:
    arts = LA.LakeArtifacts(tmp / "node-a")
    arts.put({pkg: SANDBOXED})
    dest = tmp / "lean-artifacts.tar.gz"
    assert arts.bundle(dest) == (len(arts.added), len(arts.added)) and arts.added
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
    assert sorted(f.relative_to(got).as_posix() for f in got.rglob("*") if f.is_file() and f.parent.name != TAKEN
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
    assert b.take({LA.SHARED + j: junk.as_uri()}, logged.append) == 0
    assert len(logged) == seen + 2                         # the bytes named, which can't be unpacked: never fetched again


def test_a_bundle_whose_download_gave_other_bytes_than_those_named_is_fetched_again_by_a_later_run(tmp_path, monkeypatch):
    """A response that ends before its length does so without an error, and what arrived has another sha256 than the one
    named. Such a bundle was marked taken, so the node never fetched it again and built its modules for as long as the
    launcher listed it. Nothing tells those bytes from a bundle that is wrong where it is kept, so neither is marked: only
    bytes that are the ones named are. A chunked response that ends early raises an error that is no `OSError`, which
    ended the take and the bundles after it."""
    pkg = _package(tmp_path)
    dest, h = _bundle_of(tmp_path, pkg)
    data, asked = dest.read_bytes(), []

    class Short(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            asked.append(self.path)
            self.send_response(200)
            if self.path == "/chunked":
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data[:len(data) // 2])
            else:
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data[:len(data) // 2] if asked.count(self.path) == 1 else data)

    monkeypatch.setenv("no_proxy", "127.0.0.1")
    srv = HTTPServer(("127.0.0.1", 0), Short)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        at = f"http://127.0.0.1:{srv.server_address[1]}"
        b, logged = LA.LakeArtifacts(tmp_path / "node-b"), []
        assert b.take({LA.SHARED + h: at + "/bundle"}, logged.append) == 0 and "is not the one named" in logged[0]
        assert not list((tmp_path / "node-b").glob("taken-*/*")), "not marked taken: a later run fetches it again"
        assert not list((tmp_path / "node-b").glob(".*.part"))
        assert b.take({LA.SHARED + h: at + "/chunked"}, logged.append) == 0 and "could not be fetched (IncompleteRead)" in logged[2]
        assert b.take({LA.SHARED + h: at + "/bundle"}, logged.append) == 3, "the next run fetches it whole and takes it"
        assert b.take({LA.SHARED + h: at + "/bundle"}, logged.append) == 0 and asked == ["/bundle", "/chunked", "/bundle"]
        other = LA.LakeArtifacts(tmp_path / "node-c")
        for _ in range(2):
            assert other.take({LA.SHARED + "0" * 64: dest.as_uri()}, logged.append) == 0 and "is not the one named" in logged[-2]
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_run_that_added_more_than_one_bundle_holds_writes_as_many_as_that_takes_and_another_node_takes_them_all(tmp_path):
    """A bundle has a largest size, since a larger file of a run uploads without resume, and a cold build of a large
    package adds more than that: what was past it was in no bundle, and every other node compiled it again. So a run
    writes as many bundles as its additions take, none larger than that and each whole in itself (its entries, and every
    artifact they name), the entries added first in the first; a node that takes them all restores every module. An
    entry that is larger than a bundle alone is in none, and those after it still travel."""
    pkg = _package(tmp_path)
    entries = _keys(pkg)
    a = tmp_path / "node-a"
    arts = LA.LakeArtifacts(a)
    arts.put({pkg: SANDBOXED})
    assert [e for e, _ in arts.added] == [entries[m] for m in ("P.A", "P.B", "P")]
    sizes = [(a / e).stat().st_size + sum((a / "artifacts" / o).stat().st_size for o in outs) for e, outs in arts.added]
    room = sizes[0] + sizes[1]
    assert max(sizes) <= room < sum(sizes), "room for the first two, not for all three"
    out = tmp_path / "out"
    out.mkdir()
    dests, n = arts.bundle_into(out, room)
    assert [d.name for d in dests] == ["lean-artifacts.tar.gz", "lean-artifacts-2.tar.gz"] and n == 3
    assert all(LA.BUNDLE_FILE.fullmatch(f"lean-audit/{d.name}") for d in dests), "the names the launcher lists"
    held = []
    for d in dests:
        with tarfile.open(d) as t:
            members = t.getmembers()
            mine = [json.loads(t.extractfile(m).read()) for m in members if m.name.startswith("keys/")]
        assert sum(m.size for m in members) <= room, "no bundle holds more"
        assert [m.name.split("/")[0] for m in members] == ["keys"] * len(mine) + ["artifacts"] * (3 * len(mine))
        assert {m.name for m in members if m.name.startswith("artifacts/")} == {f"artifacts/{o}" for e in mine for o in e["digests"]}
        held.append([e["module"] for e in mine])
    assert held == [["P.A", "P.B"], ["P"]]
    urls = {LA.SHARED + hashlib.sha256(d.read_bytes()).hexdigest(): d.as_uri() for d in dests}
    b, said = LA.LakeArtifacts(tmp_path / "node-b"), []
    assert b.take(urls, said.append) == 3 and "taken from 2 of 2 artifact bundle(s)" in said[0]
    hashes = {m: hashlib.sha256(m.replace(".", "/").encode()).hexdigest()[:16] for m in entries}
    assert LA.MK.stage(tmp_path / "node-b", pkg, [], tmp_path / "cache-b")[0] == {("p", hashes[m]): m for m in entries}, \
        "every module the run added is restored on another node's tree"
    c = LA.LakeArtifacts(tmp_path / "node-c")
    assert c.take(dict(list(urls.items())[:1]), lambda s: None) == 2
    assert sorted(LA.MK.stage(tmp_path / "node-c", pkg, [], tmp_path / "cache-c")[0].values()) == ["P.A", "P.B"], \
        "the first bundle alone restores what it holds"
    again = LA.LakeArtifacts(a)
    again.added = arts.added[2:]
    dests, n = again.bundle_into(out, room)
    assert [d.name for d in dests] == ["lean-artifacts-3.tar.gz"] and n == 1, "a later audit of the run: the next free name"
    assert LA.bundled(dests, n, 1) == "1 of the 1 module version(s) it added bundled for other nodes (lean-artifacts-3.tar.gz)"
    alone = tmp_path / "alone"
    alone.mkdir()
    dests, n = arts.bundle_into(alone, sizes[2] - 1)
    assert sizes[0] < sizes[1] < sizes[2] and (len(dests), n) == (2, 2), "P alone is more than such a bundle holds; P.A and P.B travel"
    assert LA.bundled(dests, n, 3).endswith("; 1 in none: gone from the cache since, or more than a bundle holds")
    assert arts.bundle_into(alone, 1) == ([], 0) and len(list(alone.iterdir())) == 2, "nothing fits: no file"


def test_one_run_takes_the_newest_bundles_up_to_a_quarter_of_the_cache_s_cap_and_a_later_run_the_rest(tmp_path):
    """Every build bundles what it compiled, up to `BUNDLE_CAP` a bundle, and the launcher lists dozens: together more
    than the cache may hold. One run takes them in the launcher's order, newest first, and stops at its budget, a quarter
    of the cap; the bundle it stopped in is not marked, and a later run takes what is left of it. Downloads run ahead of
    the unpacking: a bundle fetched and not reached is dropped, unmarked."""
    def bundle(i: int) -> tuple[str, str]:
        f = tmp_path / f"b{i}.tar.gz"
        with tarfile.open(f, "w:gz") as t:
            arts = {f"{i}{j}".ljust(16, "0") + ext: bytes([i, j]) * 50 for j, ext in enumerate([".olean", ".ilean", ".c"])}
            e = json.dumps({"schema": LA.MK.SCHEMA, "module": f"P.M{i}", "scope": "p", "hash": str(i) * 16,
                            "outputs": {"o": sorted(arts)}, "digests": {a: LA.MK.C.digest(b) for a, b in arts.items()}}).encode()
            for name, data in [("keys/" + str(i) * 128 + ".json", e), *((f"artifacts/{a}", b) for a, b in sorted(arts.items()))]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        return LA.SHARED + hashlib.sha256(f.read_bytes()).hexdigest(), f.as_uri()

    urls = dict(bundle(i) for i in (3, 1, 2))                           # the launcher's order, not the hashes'
    store, said = tmp_path / "node", []
    arts = LA.LakeArtifacts(store, cap=2000)                            # so a run takes 500 bytes: one bundle and two thirds
    assert arts.take(urls, said.append) == 1 and len(said) == 1
    assert "1 module version(s), 0 MiB, taken from 2 of 3 artifact bundle(s)" in said[0] and "left to a later run" in said[0]
    assert sorted(f.name for f in (store / "keys").iterdir()) == ["3" * 128 + ".json"], "the first the launcher named, whole"
    assert sorted(f.name[:2] for f in (store / "artifacts").iterdir()) == ["10", "11", "30", "31", "32"], \
        "the second as far as the budget went: its entry waits for its last artifact"
    assert [f.name for f in (store / TAKEN).iterdir()] == [next(iter(urls))[len(LA.SHARED):]], "only the first is marked"
    assert not list(store.glob(".*.part")), "the third was fetched ahead and dropped"
    assert arts.take(urls, said.append) == 2 and "left to a later run" not in said[1]
    assert len(list((store / "keys").iterdir())) == 3 and len(list((store / "artifacts").iterdir())) == 9
    assert len(list((store / TAKEN).iterdir())) == 3 and arts.take(urls, said.append) == 0 and len(said) == 2
    more = dict(bundle(i) for i in range(1, 10))                        # more than are fetched at once
    assert LA.AHEAD < len(more) and LA.LakeArtifacts(tmp_path / "other").take(more, said.append) == 9
    assert "taken from 9 of 9 artifact bundle(s)" in said[2] and len(list((tmp_path / "other" / TAKEN).iterdir())) == 9


def test_a_bundle_made_under_another_schema_is_left_for_a_build_of_that_schema_to_take(tmp_path, monkeypatch):
    """While trees of two schemas build on one node, each takes the bundles of its own: a bundle whose entries are of
    another schema adds nothing, and its mark is this schema's, so a build of the other still takes it."""
    pkg = _package(tmp_path)
    dest, h = _bundle_of(tmp_path, pkg)
    store = tmp_path / "node"
    monkeypatch.setattr(LA.MK, "SCHEMA", "lean-module/v5")
    assert LA.LakeArtifacts(store).take({LA.SHARED + h: dest.as_uri()}, lambda s: None) == 0
    assert not (store / "keys").exists() and not (store / "artifacts").exists() and (store / "taken-v5" / h).is_file()
    monkeypatch.undo()
    assert LA.LakeArtifacts(store).take({LA.SHARED + h: dest.as_uri()}, lambda s: None) == 3


def test_a_bundle_member_outside_the_cache_s_layout_or_an_entry_without_its_artifacts_is_left_out(tmp_path):
    """An artifact goes in only as one an entry before it names, with the digest it names; an entry once every artifact it
    names is here. A bundle of Lake's own layout (`outputs/`), which no build reads any more, adds nothing."""
    ok = LA.MK.C.digest(b"ok")

    def entry(art, d, schema=LA.MK.SCHEMA):
        return json.dumps({"schema": schema, "module": "P.A", "scope": "p", "hash": "1" * 16, "outputs": {"o": [art]},
                           "digests": {art: d}}).encode()

    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as t:
        for name, data in [("keys/" + "1" * 128 + ".json", entry("e" * 16 + ".olean", ok)),
                           ("keys/" + "2" * 128 + ".json", entry("9" * 16 + ".olean", ok)),         # its artifact isn't here
                           ("keys/" + "3" * 128 + ".json", b"not json"),
                           ("keys/" + "4" * 128 + ".json", entry("8" * 16 + ".olean", LA.MK.C.digest(b"other"))),
                           ("keys/" + "6" * 64 + ".json", entry("e" * 16 + ".olean", ok)),         # a key of before
                           ("keys/" + "7" * 128 + ".json", entry("e" * 16 + ".olean", ok, "lean-module/v3")),  # an entry of before
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
                  and f.parent.name != TAKEN) == ["artifacts/" + "e" * 16 + ".olean", "keys/" + "1" * 128 + ".json"]
    assert not (tmp_path / "escape").exists() and not (tmp_path / "x.olean").exists()


def test_the_launcher_sends_the_newest_recorded_checks_bundles(tmp_path, monkeypatch):
    """Listed from the store's remote when it has one, as only its last `common.RECENT_S` of attempts: listing all of them took
    half a minute a launch."""
    runs = {"r3": [("lean-audit/lean-artifacts.tar.gz", "c" * 64, True), ("verdicts.tar.gz", "v" * 64, True)],
            "r2": [("lean-audit/lean-artifacts.tar.gz", "b" * 64, False)],          # not on the remote: no node could fetch it
            "r1": [("lean-audit/lean-artifacts.tar.gz", "a" * 64, True), ("lean-audit/lean-artifacts-2.tar.gz", "d" * 64, True),
                   ("lean-audit/x/lean-artifacts.tar.gz", "e" * 64, True)]}
    store = types.SimpleNamespace(
        remote=None, index=types.SimpleNamespace(sql=lambda q, a: [] if LA.RECORDS_RUNS in q else [(r,) for r in runs]),
        get_attempt=lambda r: {"outputs": {"run_files": f"art:{r}"}},
        listing=lambda art: [{"path": p, "sha256": h, "remote": on} for p, h, on in runs[art[4:]]])
    assert LA.shared_bundles(store) == ["c" * 64, "a" * 64, "d" * 64]
    assert LA.shared_bundles(store, limit=1) == ["c" * 64]
    assert LA.shared_bundles(store, limit=2) == ["c" * 64, "a" * 64, "d" * 64], "runs are counted: every bundle of the second"
    runs["r0"] = [("lean-audit/lean-artifacts-10.tar.gz", "3" * 64, True), ("lean-audit/lean-artifacts-2.tar.gz", "2" * 64, True),
                  ("lean-audit/lean-artifacts.tar.gz", "1" * 64, True)]
    assert LA.shared_bundles(store)[3:] == ["1" * 64, "2" * 64, "3" * 64], "a run's in the order it wrote them, however listed"
    del runs["r0"]
    refreshed: list[dict] = []
    store.remote, store.refresh = object(), lambda **kw: refreshed.append(kw)
    monkeypatch.setattr(LA.COMMON, "recent", lambda now: f"r-since-{now > 1.7e9}")
    assert LA.shared_bundles(store) == ["c" * 64, "a" * 64, "d" * 64]
    assert refreshed == [{"attempts_only": True, "since": "r-since-True"}]


def _attempts(tmp: Path, rows: list[tuple[str, str, str, list[str]]]):
    """A store index holding ``rows`` (id, tool, state, argv), the newest first, as the store's `attempts` table has them."""
    import sqlite3

    db = sqlite3.connect(tmp / "index.sqlite")
    db.execute("CREATE TABLE attempts(id TEXT PRIMARY KEY, tool TEXT, state TEXT, start_utc TEXT, wall_s REAL, manifest_json TEXT)")
    for i, (rid, tool, state, argv) in enumerate(rows):
        db.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?)",
                   (rid, tool, state, f"2026-10-26T{20 - i:02d}:00:00Z", 60.0, json.dumps({"id": rid, "params": {"argv": argv}})))
    return types.SimpleNamespace(sql=lambda q, a=(): db.execute(q, a).fetchall())


#: a records run as `research.queue.RECORDS_COMMAND` starts it, and one a lane wrote by hand
RECORDS_RUN = ["python3", "tools/verity/lean/lean_changed.py", "--records", "security", "--update"]
RECORDS_BY_HAND = ["bash", "-lc", "uv sync -q && python3 tools/verity/lean/lean_changed.py --records security/Proofs"]


def test_the_launcher_also_sends_the_newest_records_runs_bundles(tmp_path):
    """A records run (`lean_changed.py --records`) bundles the modules its sandboxed builds compiled, and nothing listed
    them: each node compiled again what another's records run had. `shared_bundles` lists them as one more kind of
    `PUBLISHERS`, after the recorded checks', newest first and at most ``limit``: a failed run's too, whose compile is
    worth what a passing run's is (`LakeArtifacts.put`), but not an unfinished one's or another `lean_changed.py` run's."""
    index = _attempts(tmp_path, [("r-new", "python3", "done", RECORDS_RUN), ("r-check", "check", "done", ["check"]),
                                 ("r-failed", "python3", "failed", RECORDS_RUN), ("r-running", "python3", "running", RECORDS_RUN),
                                 ("r-hand", "python3", "done", RECORDS_BY_HAND),
                                 ("r-try", "python3", "done", ["python3", "tools/verity/lean/lean_changed.py", "--try", "x"]),
                                 ("r-old", "python3", "cancelled", RECORDS_RUN)])
    sha = {"r-new": "n", "r-check": "c", "r-failed": "f", "r-running": "u", "r-hand": "h", "r-try": "t", "r-old": "o"}
    runs = {r: [("lean-audit/lean-artifacts.tar.gz", s * 64, True)] for r, s in sha.items()}
    runs["r-hand"].append(("lean-audit/lean-artifacts-2.tar.gz", "2" * 64, True))
    runs["r-old"] = [("lean-audit/lean-artifacts.tar.gz", "o" * 64, False)]            # not on the remote: no node could fetch it
    store = types.SimpleNamespace(remote=None, index=index, get_attempt=lambda r: {"outputs": {"run_files": f"art:{r}"}},
                                  listing=lambda art: [{"path": p, "sha256": h, "remote": on} for p, h, on in runs[art[4:]]])
    assert LA.shared_bundles(store) == ["c" * 64, "n" * 64, "f" * 64, "h" * 64, "2" * 64]
    assert LA.shared_bundles(store, limit=2) == ["c" * 64, "n" * 64, "f" * 64]
    assert LA.shared_bundles(store, limit=1) == ["c" * 64, "n" * 64]
    assert LA.PUBLISHERS[2][0] == LA.RECORDS_RUNS and LA.PUBLISHERS[2][1].fullmatch("lean-audit/lean-artifacts.tar.gz"), \
        "where `lean_changed.audit_records` writes it"


#: drive's Lean check as the store records it: `slot.py` runs the check, so its tool is `python3`
DEFERRED = ["python3", "tools/verity/check/slot.py", "--priority", "--by", "ci", "--", "uv", "run", "--locked", "python",
            "tools/verity/check/check.py", "--keep-going", "--deferred", "lean-audit"]


def test_a_slot_wrapped_check_is_a_recorded_check_whose_bundle_and_verdicts_are_taken(tmp_path, monkeypatch):
    """Checks run under `slot.py`, so the store records drive's trains, Lean checks and after-landing runs with tool
    `python3`; `shared_bundles` and `gathered_packs` selecting only tool `check` would leave each node without another's
    built modules or replay verdicts, building `security/Proofs` cold (700 to 1,300 s) and replaying some 2,000 modules."""
    index = _attempts(tmp_path, [("r-lean", "python3", "failed", DEFERRED), ("r-other", "python3", "done", ["python3", "x.py"]),
                                 ("r-lean-tool", "python3", "done", ["python3", "tools/verity/lean/check.py", "--build"]),
                                 ("r-check", "check", "done", ["check"]), ("r-running", "python3", "running", DEFERRED)])
    runs = {r: [("lean-audit/lean-artifacts.tar.gz", r[2] * 64, True), ("verdicts.tar.gz", r[3] * 64, True)]
            for r in ("r-lean", "r-other", "r-lean-tool", "r-check", "r-running")}
    store = types.SimpleNamespace(remote=None, index=index, get_attempt=lambda r: {"outputs": {"run_files": f"art:{r}"}},
                                  listing=lambda art: [{"path": p, "sha256": h, "remote": on, "bytes": 1} for p, h, on in runs[art[4:]]])
    assert LA.shared_bundles(store) == ["l" * 64, "c" * 64]
    monkeypatch.setattr(CHECK, "attempt_pack", lambda store, rid, d, name="verdicts.tar.gz": d / rid)
    monkeypatch.setattr(CHECK.VERDICTS, "refusal", lambda pack: None)
    assert [rid for rid, _ in CHECK.gathered_packs(tmp_path, store)] == ["r-lean", "r-check"]


def _tier(audit: bool, head: str = "a" * 40, merged: str | None = None) -> list[str]:
    """`research queue ready`'s quick tier of the PR head ``head`` as the store records it (`env` runs `slot.py`, which runs
    the tier), with the Lean audit of the merge when ``audit``; run on the merge ``merged`` of that head onto `main` when
    given, as a head behind `main` is."""
    uv = "uv run --locked --all-packages --extra torch-cpu python"
    lean = f" --lean-audit '{uv} tools/verity/lean/warm.py'" if audit else ""
    at = f"--head {merged} --base {'b' * 40} --merged {head}" if merged else f"--head {head} --base {'b' * 40}"
    return ["env", "CHECK_SLOTS=/workspace/research/locks/research/slots", "python3", "tools/verity/check/slot.py", "--by", "ci", "--",
            "sh", "-c", f'exec python3 "$RESEARCH_RUN_DIR/inputs/quick_tier.py" {at}{lean} -- '
                        f"{uv} tools/verity/check/suites.py --quick"]


def test_the_audit_of_a_ready_marks_merge_publishes_its_bundle_and_verdicts_as_a_recorded_check_does(tmp_path):
    """A change's modules are built at its ready mark so that no candidate builds them: unlisted, the mark's bundle and
    verdict pack stay on the node that ran it, and the candidate that holds the change builds it again on another. The mark's
    run is no recorded check (its tool is `env`) and keeps the audit's files under `out/lean-audit/`, in its run record when
    its result lists no files. Each kind of run has its own count, since a candidate's shards end by the dozen."""
    bundle, pack = "out/lean-audit/lean-artifacts.tar.gz", "out/lean-audit/verdicts.tar.gz"
    files = {
        "r-mark-new": {"run_files": [("out/logs/suite.log", "0" * 64, True)], "run_record": [(bundle, "1" * 64, True), (pack, "2" * 64, True)]},
        "r-check": {"run_files": [("lean-audit/lean-artifacts.tar.gz", "3" * 64, True), ("verdicts.tar.gz", "4" * 64, True)]},
        # a tier whose command names no audit publishes nothing, whatever its files are called
        "r-tier": {"run_files": [(bundle, "5" * 64, True), (pack, "6" * 64, True)]},
        "r-mark-old": {"run_files": [(bundle, "7" * 64, True), ("out/lean-audit/lean-artifacts-2.tar.gz", "8" * 64, False),
                                     ("lean-audit/lean-artifacts.tar.gz", "9" * 64, True), ("verdicts.tar.gz", "a" * 64, True),
                                     (pack, "b" * 64, True)]},
        "r-mark-running": {"run_files": [(bundle, "c" * 64, True), (pack, "d" * 64, True)]},
    }
    index = _attempts(tmp_path, [("r-mark-new", "env", "done", _tier(True)), ("r-check", "check", "done", ["check"]),
                                 ("r-tier", "env", "done", _tier(False)), ("r-mark-old", "env", "failed", _tier(True)),
                                 ("r-mark-running", "env", "running", _tier(True))])
    store = types.SimpleNamespace(
        remote=None, index=index, get_attempt=lambda r: {"outputs": {kind: f"{kind}:{r}" for kind in files[r]}},
        listing=lambda art: [{"path": p, "sha256": h, "remote": on, "bytes": 1} for p, h, on in files[art.split(":")[1]][art.split(":")[0]]])
    assert LA.shared_bundles(store) == ["3" * 64, "1" * 64, "7" * 64]
    assert LA.shared_bundles(store, limit=1) == ["3" * 64, "1" * 64]
    assert [(p["run"], p["path"], p["sha256"]) for p in CHECK.verdict_packs(store)] == [
        ("r-check", "verdicts.tar.gz", "4" * 64), ("r-mark-new", pack, "2" * 64), ("r-mark-old", pack, "b" * 64)]
    assert [p["run"] for p in CHECK.verdict_packs(store, limit=1)] == ["r-check", "r-mark-new"]
    spec = importlib.util.spec_from_file_location("lean_warm_for_test_lean_audit", ROOT / "tools" / "verity" / "lean" / "warm.py")
    warm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warm)
    assert CHECK.MARK_PACK == LA.COMMON.MARK_AUDIT_OUT + warm.PACK == pack, "the pack is listed where the audit writes it"
    assert LA.PUBLISHERS[1][1].fullmatch(LA.COMMON.MARK_AUDIT_OUT + "lean-artifacts.tar.gz"), "and the bundle where `--bundle` does"


def test_a_launch_sends_the_mark_audits_of_the_prs_its_tree_holds_however_old(tmp_path, monkeypatch):
    """A PR waits hours between its mark and the candidate that holds it, and each newer mark's audit pushes its bundle and
    pack down the newest: once past them, the candidate compiled the change again, on a node that had not run the mark. The
    marks of the heads the launcher's tree merged are listed before a kind's newest and beside its count, whether the tier
    ran on the head or on its merge onto `main`, and only the audits: a head's other runs, and other heads' marks, are not."""
    bundle, pack = "out/lean-audit/lean-artifacts.tar.gz", "out/lean-audit/verdicts.tar.gz"
    mine, behind, other = "1" * 40, "2" * 40, "3" * 40
    rows = [("r-new", "env", "done", _tier(True, other)), ("r-check", "check", "done", ["check"]),
            ("r-mine-tier", "env", "done", _tier(False, mine)),                       # a mark of this head that audited nothing
            ("r-mine-running", "env", "running", _tier(True, mine)),
            ("r-old", "env", "done", _tier(True, other)), ("r-mine", "env", "failed", _tier(True, mine)),
            ("r-behind", "env", "done", _tier(True, behind, merged="4" * 40))]
    files = {r: [(bundle, f"{i:x}" * 64, True), (pack, f"{i + 8:x}" * 64, True)] for i, (r, *_) in enumerate(rows)}
    files["r-check"] = [("lean-audit/lean-artifacts.tar.gz", "c0" * 32, True), ("verdicts.tar.gz", "c1" * 32, True)]
    store = types.SimpleNamespace(
        remote=None, index=_attempts(tmp_path, rows), get_attempt=lambda r: {"outputs": {"run_files": f"art:{r}"}},
        listing=lambda art: [{"path": p, "sha256": h, "remote": on, "bytes": 1} for p, h, on in files[art[4:]]])

    def bundles(**kw) -> list[str]:
        by = {files[r][0][1]: r for r in files}
        return [by[h] for h in LA.shared_bundles(store, **kw)]

    def packs(**kw) -> list[str]:
        return [p["run"] for p in CHECK.verdict_packs(store, **kw)]

    assert bundles(limit=1) == packs(limit=1) == ["r-check", "r-new"], "past the newest, a member's mark was not sent"
    assert bundles(limit=1, heads=[mine, behind]) == packs(limit=1, heads=[mine, behind]) == ["r-check", "r-mine", "r-behind", "r-new"]
    assert bundles(heads=[mine, behind]) == packs(heads=[mine, behind]) == ["r-check", "r-mine", "r-behind", "r-new", "r-old"], \
        "each run once, a member's first"
    assert bundles(limit=1, heads=["5" * 40]) == packs(limit=1, heads=["5" * 40]) == ["r-check", "r-new"]
    # the launcher reads the heads from its own checkout: a caller that gives a store gives its heads too
    monkeypatch.setattr(CHECK.COMMON, "merged_heads", lambda at: [behind] if at == CHECK.ROOT else [])
    monkeypatch.setattr(pytest.importorskip("research.store.local"), "LocalStore", lambda: store)
    store.remote, store.refresh = types.SimpleNamespace(presign=lambda key, ttl: f"https://r2/{key}"), lambda **kw: None

    def sent(*given) -> list[str]:
        return [p["run"] for p in json.loads(CHECK.verdict_url_file(tmp_path, *given).read_text())]

    assert sent() == ["r-check", "r-behind", "r-new", "r-old", "r-mine"]
    assert sent(store) == ["r-check", "r-new", "r-old", "r-mine", "r-behind"]
    assert sent(store, [mine]) == ["r-check", "r-mine", "r-new", "r-old", "r-behind"]
    files["r-mine"][1] = (pack, files["r-mine"][1][1], False)
    assert packs(limit=1, heads=[mine], remote=True) == ["r-check", "r-new"], "a pack the remote lacks is sent to no node"
    assert LA.COMMON.lookups(LA.COMMON.RECORDED_CHECKS, 2, [mine]) == [(LA.COMMON.RECORDED_CHECKS, (8,), 2)], "a head names no check"
    assert len(LA.COMMON.lookups(LA.COMMON.MARK_AUDITS, 2, [mine])) == 2 and len(LA.COMMON.lookups(LA.COMMON.MARK_AUDITS, 2)) == 1


def test_the_heads_a_tree_holds_are_those_the_queues_commits_name_the_newest_first(tmp_path):
    """`common.merged_heads` reads the PRs a candidate holds from the commits the verified queue made of them on its
    first-parent line, each naming the PR's head in its `Head` trailer: a merge of that head, or the one commit of a move the
    queue regenerated, which has no second parent. A tier names that head, so its mark's audit is found by it. A PR's own
    merge of a branch and a commit that only quotes a trailer are no member."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                               *args], check=True, capture_output=True, text=True).stdout.strip()

    def pr(n: int) -> str:
        git("checkout", "-q", "-b", f"pr-{n}", "main")
        git("commit", "-q", "--allow-empty", "-m", f"pr {n}")
        head = git("rev-parse", "HEAD")
        git("checkout", "-q", "main")
        return head

    def trailers(n: int, head: str) -> str:
        return f"PR: #{n}\nHead: {head}\nOwner: lane\n"

    git("init", "-q", "-b", "main")
    assert LA.COMMON.merged_heads(tmp_path) == [], "no commit yet"
    git("commit", "-q", "--allow-empty", "-m", "root")
    assert LA.COMMON.merged_heads(tmp_path) == []
    first = pr(1)
    git("merge", "-q", "--no-ff", "-m", f"Queue: #1 one\n\n{trailers(1, first)}", "pr-1")
    other = pr(2)
    git("merge", "-q", "--no-ff", "-m", f"Merge branch 'pr-2'\n\n{trailers(2, other)}", "pr-2")
    git("commit", "-q", "--allow-empty", "-m", f"Lock records regenerated for #1\n\nHead: {other}")
    moved = pr(3)
    git("commit", "-q", "--allow-empty", "-m", f"Queue: #3 a move\n\nregenerated by drive\n\n{trailers(3, moved)}Move: tools/move/m.toml\n")
    last = pr(4)
    git("merge", "-q", "--no-ff", "-m", f"Queue: #4 four\n\n{trailers(4, last)}", "pr-4")
    assert LA.COMMON.merged_heads(tmp_path) == [last, moved, first]
    assert LA.COMMON.merged_heads(tmp_path, limit=2) == [last, moved]
    assert LA.COMMON.merged_heads(tmp_path / "no-checkout") == []


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
    monkeypatch.setattr(LA.LakeArtifacts, "bundle", lambda self, dest, *at: (dest.write_bytes(b"") or 1, 1))
    for n, bundle in enumerate((False, True)):
        shutil.rmtree(tmp_path / "cache", ignore_errors=True)
        assert LA.run(tmp_path / f"out{n}", tmp_path / "cache" / "lean-audit", tmp_path / "cache" / "lean-deps",
                      log=lambda s: None, bundle=bundle) == 1
        assert (tmp_path / f"out{n}" / "lean-artifacts.tar.gz").exists() == bundle
        assert {p: (r["sandbox"], r["failures"]) for p, r in put_with[-1].items()} == {
            "a": (True, []), "b": (True, ["reads docs/outside.txt"])}, \
            "each package it built, with check.py's report on it, which says whether it was built in the sandbox"
    out = tmp_path / "steps"
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    def step():
        return dict((name, argv) for name, argv, _ in CHECK.lean_steps(out))["lean-audit"]
    assert "--bundle" not in step()
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261026-000000-abcd")
    assert "--bundle" in step()


def test_a_package_whose_sources_sit_beside_it_keeps_its_build(tmp_path):
    """`security/Proofs` has `srcDir = ".."`. Pruning looked for each kept output's source inside the package, found none,
    and deleted the whole kept build on every warm take, so each check's audit rebuilt `security/Proofs` from scratch."""
    pkg = tmp_path / "security" / "Proofs"
    (pkg / "Flock").mkdir(parents=True)
    (pkg / "lakefile.toml").write_text('name = "proofs"\nsrcDir = ".."\n\n[[lean_lib]]\nname = "Proofs"\n')
    (pkg / "Flock" / "X.lean").write_text("theorem x : True := trivial\n")
    (tmp_path / "security" / "Flock").mkdir()
    (tmp_path / "security" / "Flock" / "Y.lean").write_text("theorem y : True := trivial\n")
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
    """A test's `generate` (`runs`) reads fixtures that are no package's input (PoUW's reads `verity/definitions/silicon/captures/vectors`), so the
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
    got = LA.inputs("security/Proofs", files)
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
    assert "security/Proofs" in pkgs and not [p for p in pkgs if p.startswith("tools/verity/lean")]
    got = LA.inputs("security/Proofs", files)
    assert "security/lean-audit.json" in got and "verity/protocols/flock/verifier/FlockVerify.lean" in got
    assert "tools/verity/lean/check.py" in got and not [f for f in got if "/.lake/" in f]
    assert "tools/verity/lean/Replay.lean" in got and "tools/verity/lean/README.md" not in got
    assert not [f for f in got if f.startswith("tools/verity/lean/tests/")]
    assert {"tools/verity/lean/upstream.py", "tools/verity/lean/setup.sh", "tools/verity/lean/cache.py"} <= set(got)
    assert not {f"tools/verity/lean/{n}" for n in LA.NOT_READ} & set(got)
    assert {f"tools/verity/lean/{n}" for n in LA.NOT_READ} <= set(files), "an entry of NOT_READ names no file: remove it"
    assert not [f for f in got if not f.startswith(LA.LEAN_TOOL)
                and any(fnmatch.fnmatchcase(f.rsplit("/", 1)[-1], p) for p in LA.UNREAD_NAMES)]
    assert "tools/verity/lean/Unread.lean" in got and set(LA.UNREAD_FACTS) <= set(pkgs)
    lean_check = (ROOT / "tools" / "verity" / "lean" / "check.py").read_text()
    assert ast.literal_eval(re.search(r"^UNREAD_FACTS = (.+)$", lean_check, re.M).group(1)) == LA.UNREAD_FACTS, \
        "check.py writes the ratchet's facts in the audits of the packages lean_audit.py joins them from"


#: `check.py` as `FAKE_AUDIT`, and the audits of the packages the ratchet reads write its facts as the real one does
FACTS_AUDIT = FAKE_AUDIT.replace(
    '    reps.append({"package": str(pkg), "failures": [f"reads {m}" for m in missing]})\n',
    '''    rep = {"package": str(pkg), "failures": [f"reads {m}" for m in missing]}
    rel = pkg.relative_to(root).as_posix()
    if rel in ("security", "security/Proofs"):
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
             "security/S.lean": "", "security/Proofs/P.lean": "",
             **{f"security/{f}": t for f, t in SPEC.items()}, **{f"security/Proofs/{f}": t for f, t in SPEC.items()}}
    assert FACTS_AUDIT != FAKE_AUDIT
    root = _tree(tmp_path, monkeypatch, extra)
    cache = tmp_path / "cache" / "lean-audit"

    def run(n, **kw):
        out = tmp_path / f"out{n}"
        rc = LA.run(out, cache, tmp_path / "cache" / "lean-deps", log=lambda s: None, **kw)
        ran = sorted(x.split()[0] for x in (out / "ran.txt").read_text().splitlines()) if (out / "ran.txt").exists() else []
        seen = json.loads((out / "unread" / "seen.json").read_text()) if (out / "unread" / "seen.json").exists() else None
        return rc, ran, json.loads((out / "audit.json").read_text()), seen

    keys = {p: LA.key(p, LA.COMMON.tracked_files(), LA.FLAGS) for p in ("security", "security/Proofs", "pkg/a")}
    rc, ran, doc, seen = run(1)
    assert rc == 0 and ran == sorted(["security", "security/Proofs", "pkg/a", "pkg/b"]) and seen == ["security", "security/Proofs"]
    assert doc["unread"]["failures"] == [] and sorted(p.stem for p in (cache / "unread").glob("*.json")) == sorted(
        [keys["security"], keys["security/Proofs"]])
    published = sorted(f"{keys[p]}.jsonl.gz" for p in LA.UNREAD_FACTS)
    assert sorted(f.name for f in (tmp_path / "out1" / "unread").glob("*.jsonl.gz")) == published, \
        "an audit's facts are left by their key, as the landing gate looks for them"
    rc, ran, doc, seen = run(2)
    assert rc == 0 and ran == [] and seen == ["security", "security/Proofs"] and len(doc["cached"]) == 4
    assert sorted(f.name for f in (tmp_path / "out2" / "unread").glob("*.jsonl.gz")) == published, "a reused pass's too"
    (root / "tools/verity/lean/unread-caps.json").write_text("1")
    rc, ran, doc, seen = run(3)
    assert rc == 1 and ran == [] and doc["failed"] == ["unread"] and doc["unread"]["failures"] == ["unread: 2 above 1"]
    (root / "tools/verity/lean/unread-caps.json").write_text("2")
    k = keys["security/Proofs"]
    entry = cache / "unread" / f"{k}.json"
    entry.write_text(json.dumps(json.loads(entry.read_text()) | {"facts": "e30K"}))
    rc, ran, doc, seen = run(4)
    assert rc == 0 and ran == ["security/Proofs"] and seen == ["security", "security/Proofs"], "other bytes: audited again"
    entry.unlink()
    assert not LA.reusable(cache, "security/Proofs", k) and LA.reusable(cache, "pkg/a", keys["pkg/a"])
    rc, ran, doc, seen = run(5)
    assert rc == 0 and ran == ["security/Proofs"] and entry.is_file()
    rc, ran, doc, seen = run(6, only=["security/Proofs"])
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
    monkeypatch.setattr(LA.COMMON, "merged_heads", lambda at: [f"merged in {at.name}"])
    monkeypatch.setattr(LA, "shared_bundles", lambda heads: ["ef" * 32] if heads == ["merged in repo"] else [])
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
