"""`lean_audit`: check's Lean audit step. `tools/lean/audit.py --all --build`, with passes reused and dependencies kept warm.

    python tools/check/lean_audit.py --out DIR [--cache DIR] [--deps DIR] [--fresh] [--urls FILE] [--export DIR]
    python tools/check/lean_audit.py --pin RUN        pin the bundles run RUN exported (`lean-deps.json`)
    python tools/check/lean_audit.py --explain PKG    why PKG's audit would run: its inputs against the nearest pass kept here

A package's verdict is reused while everything its audit reads is unchanged. Its key hashes:
- the tracked files of the package, less the Lake packages nested in it (`audit.lean_modules` skips them);
- the same for each package it requires by path (`audit.path_deps`);
- `tools/lean/` (the auditor and its controls), this file and `common.py`, the helpers it imports: not `check.py`, which only
  launches it, so an edit to `check.py` alone keeps every package's verdict;
- the package's `lean-toolchain` and `lake-manifest.json`, which pin the toolchain and the fetched dependencies.

When one package misses, the audit runs on a scratch tree holding only the missed packages' inputs, from that tree's own copy of
`tools/lean`. So a read of anything outside a key finds nothing, and the audit fails there instead of a stale pass being kept
(the counterpart of `suites.py`'s read guard). Every package in the scratch tree is audited, the controls first. Each pass is
stored under its key, and a failure is never stored.

A package's fetched dependencies (`.lake/packages`: Mathlib's cache, ArkLib's build) are kept on this machine under
`lake-manifest.json` plus `lean-toolchain`. They're moved into the scratch package's own `.lake` before its build (the only place
its sandbox lets it write, never a shared symlink) and back afterwards only when that package's audit passed; after a failure
they're dropped, so a run's compile-time code cannot leave the next run a cache it wrote. Lake rebuilds nothing it built in an
earlier passing run. The audit's `dependencies` check still compares their `.olean` files with the
package's record, and the kernel replay still covers the package's own modules. Lean gets this machine's CPUs
(`LEAN_NUM_THREADS`): inside a pod, its default counts the host's.

A machine without a package's dependencies restores them from the evidence store instead of cloning them from GitHub, which
throttles anonymous clones from shared addresses. `lean-deps.json` pins one bundle per `lake-manifest.json` + `lean-toolchain`
by sha256; the launcher sends a presigned URL for each (`--urls`, `check.py --record`), and a bundle is used only when its
bytes are the pinned ones. A bundle leaves out the builds Mathlib's cache supplies: `lake exe cache get` takes those from
Mathlib's CDN, which is not GitHub. After a bump, a passing cold audit run with `--export` writes the new bundles into its run
directory, custody puts them on R2, and `--pin RUN` pins them.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEAN_TOOL = "tools/lean/"
SCHEMA = "lean-audit-cache/v1"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


#: every file this step loads is in `inputs`: this module, `common.py` and `tools/lean/` (`test_lean_audit.py` holds it)
COMMON = _module(ROOT / "tools" / "check" / "common.py", "check_common_for_lean_audit")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def packages(files: list[str]) -> list[str]:
    """The audited packages: directories whose tracked files include a `lean-audit.json`, the auditor's fixtures aside."""
    return sorted({f.rsplit("/", 1)[0] for f in files if f.endswith("/lean-audit.json") and not f.startswith(LEAN_TOOL)})


def own_files(pkg: str, files: list[str]) -> list[str]:
    """The package's tracked files, less those of the Lake packages nested in it."""
    nested = {f.rsplit("/", 1)[0] + "/" for f in files if f.startswith(pkg + "/") and f.rsplit("/", 1)[-1] in ("lakefile.toml", "lakefile.lean")
              and f.rsplit("/", 1)[0] != pkg}
    return [f for f in files if f.startswith(pkg + "/") and not any(f.startswith(n) for n in nested)]


def path_deps(pkg: str) -> list[str]:
    """The packages ``pkg`` requires by path, as `audit.path_deps` reads them (its manifest lists the inherited ones too)."""
    m = ROOT / pkg / "lake-manifest.json"
    if not m.exists():
        return []
    out = []
    for d in json.loads(m.read_text()).get("packages", []):
        if d.get("type") == "path":
            out.append(os.path.relpath(os.path.normpath(ROOT / pkg / d["dir"]), ROOT))
    return sorted(set(out))


def inputs(pkg: str, files: list[str]) -> list[str]:
    """Every tracked file an audit of ``pkg`` can read."""
    own = [f for p in (pkg, *path_deps(pkg)) for f in own_files(p, files)]
    tool = [f for f in files if f.startswith(LEAN_TOOL)]
    return sorted({*own, *tool, "tools/check/lean_audit.py", "tools/check/common.py"})


def components(pkg: str, files: list[str], flags: list[str]) -> dict:
    """What a package's key hashes: the schema, the package, the audit's flags, and each input's sha256 ("-" when absent). A pass
    keeps them beside it, so `--explain` can say what a miss changed."""
    return {"schema": SCHEMA, "package": pkg, "flags": flags,
            "files": {f: (_sha(ROOT / f) if (ROOT / f).is_file() else "-") for f in inputs(pkg, files)}}


def key_of(c: dict) -> str:
    h = hashlib.sha256(json.dumps([c["schema"], c["package"], c["flags"]]).encode())
    for f, d in sorted(c["files"].items()):
        h.update(f.encode() + b"\0" + d.encode() + b"\0")
    return h.hexdigest()


def key(pkg: str, files: list[str], flags: list[str]) -> str:
    return key_of(components(pkg, files, flags))


def explain(name: str, cache: Path) -> tuple[int, list[str]]:
    """(exit status, lines): whether package ``name`` (its path, or a unique part of one) would reuse a pass, and if not, which of
    its inputs differ from the nearest pass kept in ``cache`` (fewest differing files)."""
    files = COMMON.tracked_files(ROOT) or []
    pkgs = packages(files)
    match = [p for p in pkgs if p == name.rstrip("/")] or [p for p in pkgs if name in p]
    if len(match) != 1:
        return 2, [f"lean_audit: {name!r} matches {len(match)} packages: {', '.join(match) or ', '.join(pkgs)}"]
    pkg = match[0]
    now = components(pkg, files, ["--all", "--build"])
    k = key_of(now)
    if (cache / f"{k}.json").is_file():
        return 0, [f"{pkg}: PASS reused ({k[:12]}): every file its audit reads is unchanged"]
    kept = []
    for f in cache.glob("*.json") if cache.is_dir() else ():
        try:
            e = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if e.get("package") == pkg and e.get("components"):
            kept.append(e)
    if not kept:
        return 0, [f"{pkg}: missed ({k[:12]}); no pass of it kept in {cache} records its inputs (passes kept before `--explain` don't)"]

    def diff(was: dict) -> list[str]:
        a, b = was.get("files", {}), now["files"]
        out = [f"{'changed' if f in a and f in b else 'new' if f in b else 'no longer an input'}: {f}"
               for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
        return out + [f"{x}: {was.get(x)} -> {now[x]}" for x in ("schema", "flags") if was.get(x) != now[x]]

    best = min(kept, key=lambda e: len(diff(e["components"])))
    lines = diff(best["components"])
    return 0, [f"{pkg}: missed ({k[:12]}); the nearest pass kept here ({best['key'][:12]}) differs in {len(lines)} input(s):",
               *(f"  {line}" for line in lines)]


def deps_key(pkg: Path) -> str | None:
    """What a package's fetched dependencies are a function of: its manifest and toolchain (None: it fetches nothing)."""
    m = pkg / "lake-manifest.json"
    if not m.exists() or not [d for d in json.loads(m.read_text()).get("packages", []) if d.get("type") != "path"]:
        return None
    return hashlib.sha256(m.read_bytes() + b"\0" + (pkg / "lean-toolchain").read_bytes()).hexdigest()


PIN = "tools/check/lean-deps.json"
#: the presigned URLs sent with a recorded run, {deps_key: URL}; `check.py` moves the file out of the run directory
URLS = "lean-deps-urls.json"
URL_TTL_S = 6 * 3600


def pins() -> dict[str, dict]:
    """{deps_key: {package, sha256, bytes, run}}: the stored bundle of each manifest + toolchain."""
    p = ROOT / PIN
    return json.loads(p.read_text())["deps"] if p.is_file() else {}


def cache_covered(packages_dir: Path) -> set[str]:
    """The fetched packages whose builds `lake exe cache get` takes from Mathlib's CDN: Mathlib and what its manifest pins."""
    m = packages_dir / "mathlib" / "lake-manifest.json"
    return {"mathlib", *(d["name"] for d in json.loads(m.read_text()).get("packages", []))} if m.is_file() else set()


def bundle(src: Path, dst: Path) -> dict:
    """A package's `.lake/packages` as a gzipped tar, less the builds Mathlib's cache supplies. Every member must pass
    `tarfile`'s data filter, which `restore` extracts with."""
    skip = tuple(f"{n}/.lake/build" for n in sorted(cache_covered(src)))

    def keep(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if ti.name in skip or ti.name.startswith(tuple(s + "/" for s in skip)):
            return None
        tarfile.data_filter(ti, str(src))
        return ti

    tmp = dst.with_name(f".{dst.name}.tmp")
    with tarfile.open(tmp, "w:gz", compresslevel=1) as t:
        for name in sorted(os.listdir(src)):
            t.add(src / name, arcname=name, filter=keep)
    os.replace(tmp, dst)
    return {"sha256": COMMON.sha256(dst), "bytes": dst.stat().st_size}


def restore(store: Path, k: str, url: str, pin: dict, name: str, log=print) -> bool:
    """Download the bundle pinned under ``k`` and unpack it as ``store/k``. False, and the package builds cold, when it can't
    be fetched or isn't the pinned bundle. The URL is never logged: it grants a read until it expires."""
    part, tmp = store / f".{k}.part", store / f".{k}.tmp"
    t0 = time.monotonic()
    try:
        h, n = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                h.update(chunk)
                n += len(chunk)
                f.write(chunk)
        if (h.hexdigest(), n) != (pin["sha256"], pin["bytes"]):
            log(f"lean-audit: {name}: the store's dependencies ({k[:12]}) are not the pinned bundle; building cold")
            return False
        t1 = time.monotonic()
        shutil.rmtree(tmp, ignore_errors=True)
        with tarfile.open(part) as t:
            t.extractall(tmp, filter="data")
        os.replace(tmp, store / k)
    except (OSError, ValueError, tarfile.TarError) as e:            # urllib's errors are OSErrors
        log(f"lean-audit: {name}: the store's dependencies ({k[:12]}) could not be restored "
            f"({getattr(e, 'code', None) or type(e).__name__}); building cold")
        return False
    finally:
        part.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"lean-audit: {name}: dependencies from the store ({k[:12]}): {n >> 20} MiB fetched in {t1 - t0:.0f} s, "
        f"unpacked in {time.monotonic() - t1:.0f} s")
    return True


class WarmDeps:
    """Each package's `.lake/packages` kept under its `deps_key`, moved in before a build and back after. A run holds the
    ones it moved in; another run at the same time finds them absent and builds cold. Missing ones come from the store
    when ``urls`` has the pinned bundle's."""

    def __init__(self, store: Path, urls: dict[str, str] | None = None):
        self.store, self.urls = store, urls or {}
        self.held: dict[Path, Path] = {}

    def _lock(self, k: str):
        self.store.mkdir(parents=True, exist_ok=True)
        f = open(self.store / f"{k}.lock", "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    def take(self, pkg: Path, log=print) -> None:
        if (k := deps_key(pkg)) is None:
            return
        dst = pkg / ".lake" / "packages"
        with self._lock(k):
            src = self.store / k
            if not src.is_dir() and not dst.exists() and k in self.urls and (pin := pins().get(k)):
                restore(self.store, k, self.urls[k], pin, pkg.name, log)
            if src.is_dir() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                _move(src, dst)
                log(f"lean-audit: {pkg.name}: dependencies warm ({k[:12]})")
            else:
                log(f"lean-audit: {pkg.name}: dependencies cold ({k[:12]})")
        self.held[pkg] = self.store / k

    def give_back(self, passed: set[Path]) -> None:
        """Keep the dependencies of the packages in ``passed`` (an audit that passed, whose imported `.olean` files matched
        the record, or one that failed before compiling anything, on fetching a dependency) and drop the others': a run's
        compile-time code may write in them, so only such a run vouches for what comes back."""
        for pkg, home in self.held.items():
            src = pkg / ".lake" / "packages"
            if not src.is_dir() or pkg not in passed:
                continue
            with self._lock(home.name):
                if home.exists():
                    shutil.rmtree(home, ignore_errors=True)
                _move(src, home)

    def export(self, pkgs: dict[str, str], dest: Path, log=print) -> dict:
        """Bundle the kept dependencies of ``pkgs`` ({package: deps_key}, packages that passed) into ``dest``, with the
        listing `--pin` reads (`lean-deps.json`)."""
        dest.mkdir(parents=True, exist_ok=True)
        done: dict[str, dict] = {}
        for p, k in sorted(pkgs.items()):
            if k in done or not (self.store / k).is_dir():
                continue
            t0 = time.monotonic()
            with self._lock(k):
                done[k] = {"package": p, **bundle(self.store / k, dest / f"{k}.tar.gz")}
            log(f"lean-audit: {p}: dependencies bundled ({k[:12]}, {done[k]['bytes'] >> 20} MiB in {time.monotonic() - t0:.0f} s)")
        (dest / "lean-deps.json").write_text(json.dumps({"schema": "lean-deps/v1", "deps": done}, indent=1) + "\n")
        return done


def _move(src: Path, dst: Path) -> None:
    try:
        os.replace(src, dst)
    except OSError:                                    # another filesystem
        shutil.move(str(src), str(dst))


def scratch_tree(dest: Path, files: list[str]) -> None:
    for f in files:
        s, d = ROOT / f, dest / f
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_symlink():
            d.symlink_to(os.readlink(s))
        elif s.is_file():
            shutil.copy2(s, d)


#: a dependency fetch that failed for a reason outside the audit (GitHub throttles anonymous clones from a busy address)
TRANSIENT = ("could not read Username for 'https://github.com'", "Could not resolve host", "expected flush after ref listing",
             "Failed to connect to", "The requested URL returned error: 429", "The requested URL returned error: 5")
RETRIES = (90, 240, 480)


def transient(rep: dict) -> bool:
    """A package that failed only in setup or build, and only on fetching a dependency: no verdict, so it is audited again."""
    fs = rep.get("failures", [])
    return bool(fs) and all(f.startswith(("setup:", "build:")) and any(t in f for t in TRANSIENT) for f in fs)


def _audit(missed: list[str], files: list[str], keys: dict, reports: dict, out: Path, cache: Path, deps: Path, fresh: bool,
           log, urls: dict[str, str] | None = None, comps: dict | None = None) -> tuple[list[str], list[str]]:
    """Audit ``missed`` (and the packages they require by path) on a scratch tree of their inputs, and store each pass.
    Returns (the failed, the audited)."""
    deps.mkdir(parents=True, exist_ok=True)
    # beside the dependencies: moves, not copies. Resolved, because audit.py names each package by its resolved path, and a
    # cache reached through a symlink would otherwise match no report to its package
    work = Path(tempfile.mkdtemp(prefix="lean-audit-scratch-", dir=deps.parent)).resolve()
    warm = WarmDeps(deps, urls)
    passed: set[Path] = set()
    audited: list[str] = []
    failed: list[str] = []
    try:
        union = sorted({f for p in missed for f in inputs(p, files)})
        scratch_tree(work, union)
        audited = packages(union)
        with ThreadPoolExecutor(max_workers=max(1, len(audited))) as ex:
            list(ex.map(lambda p: warm.take(work / p, log), audited))
        env = {**os.environ, "LEAN_NUM_THREADS": str(COMMON.cores())}
        t0 = time.monotonic()
        (out / "audit.json").unlink(missing_ok=True)
        r = subprocess.run([sys.executable, str(work / "tools" / "lean" / "audit.py"), "--all", "--build", "--root", str(work),
                            "--out", str(out)], cwd=work, env=env)
        log(f"lean-audit: {', '.join(audited)} audited in {time.monotonic() - t0:.0f} s (exit {r.returncode})")
        doc = json.loads((out / "audit.json").read_text()) if (out / "audit.json").exists() else {"failed": ["audit.py"]}
        failed = [os.path.relpath(f, work) if f.startswith(str(work)) else f for f in doc.get("failed", [])]
        controls_ok = not [f for f in failed if f.startswith("controls")]
        for rep in doc.get("packages", []):
            p = os.path.relpath(rep["package"], work)
            rep["package"] = p
            reports[p] = rep
            if controls_ok and not rep["failures"] and r.returncode in (0, 1) and p in keys and not fresh:
                cache.mkdir(parents=True, exist_ok=True)
                tmp = cache / f".{keys[p]}.tmp"
                tmp.write_text(json.dumps({"schema": SCHEMA, "package": p, "key": keys[p], "report": rep,
                                           "components": (comps or {}).get(p)}))
                os.replace(tmp, cache / f"{keys[p]}.json")
        if r.returncode not in (0, 1) and not failed:
            failed = [f"audit.py exit {r.returncode}"]
        # a pass vouches for its dependencies; so does a failure only in fetching them (Lake clones every dependency, each at
        # the revision the manifest pins, before it compiles anything): keeping those, a retry fetches only what is missing
        passed = {work / p for p in audited if p in reports and ((controls_ok and not reports[p].get("failures"))
                                                                 or transient(reports[p]))}
    finally:
        warm.give_back(passed)
        shutil.rmtree(work, ignore_errors=True)
    return failed, audited


def run(out: Path, cache: Path, deps: Path, fresh: bool = False, log=print, urls: dict[str, str] | None = None,
        export: Path | None = None) -> int:
    files = COMMON.tracked_files(ROOT)
    if files is None:
        log("lean-audit: the tree's files cannot be listed; nothing is cached")
        fresh = True
        files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".lake" not in p.parts]
    pkgs = packages(files)
    comps = {p: components(p, files, ["--all", "--build"]) for p in pkgs}
    keys = {p: key_of(c) for p, c in comps.items()}
    out.mkdir(parents=True, exist_ok=True)
    reports, cached = {}, {}
    for p, k in keys.items():
        entry = cache / f"{k}.json"
        if not fresh and entry.is_file():
            reports[p] = json.loads(entry.read_text())["report"]
            cached[p] = k
            log(f"lean-audit: {p}: PASS reused ({k[:12]}): every file its audit reads is unchanged")
    todo, failed = [p for p in pkgs if p not in cached], []
    for attempt in range(len(RETRIES) + 1):
        if not todo:
            break
        now, audited = _audit(todo, files, keys, reports, out, cache, deps, fresh, log, urls, comps)
        failed = [f for f in failed if f not in audited] + now
        retry = [p for p in audited if p in now and transient(reports.get(p, {}))]
        if not retry or attempt == len(RETRIES):
            break
        log(f"lean-audit: {', '.join(retry)}: a dependency fetch failed, so no verdict; auditing again in {RETRIES[attempt]} s")
        time.sleep(RETRIES[attempt])
        todo = retry
    failed += [p for p in pkgs if p not in reports and p not in failed]
    (out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "cached": cached,
                                                "packages": [reports[p] for p in pkgs if p in reports]}, indent=1) + "\n")
    log(f"lean-audit: {'PASS' if not failed else 'FAIL (' + ', '.join(failed) + ')'}: {len(cached)} of {len(pkgs)} packages reused")
    if export is not None:
        ok = {p: k for p in pkgs if p in reports and not reports[p].get("failures") and (k := deps_key(ROOT / p))}
        WarmDeps(deps).export(ok, export, log)
    return 0 if not failed else 1


def presigned(ttl_s: int = URL_TTL_S) -> dict[str, str]:
    """{deps_key: URL} for this tree's pinned bundles, each a GET of the bundle's blob signed by the launcher's store remote.
    Empty when this machine's store has no remote that signs."""
    from research.store.local import object_key
    from research.store.remote import ConfigError, load_remote

    want = {deps_key(ROOT / p) for p in packages(COMMON.tracked_files(ROOT) or [])}
    pinned = {k: v for k, v in pins().items() if k in want}
    try:
        r = load_remote() if pinned else None
    except ConfigError:
        r = None
    if not hasattr(r, "presign"):
        return {}
    return {k: r.presign(object_key(v["sha256"]), ttl_s) for k, v in sorted(pinned.items())}


def url_file(dest: Path, ttl_s: int = URL_TTL_S) -> Path | None:
    """`presigned` written as ``dest/lean-deps-urls.json`` (mode 600), the name `check.py` looks for among a run's inputs."""
    if not (urls := presigned(ttl_s)):
        return None
    p = dest / URLS
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(urls, f)
    return p


def pin(run_id: str) -> int:
    """Pin the bundles run ``run_id`` exported: each must be a file of its run record, which custody put on the remote."""
    from research.store.local import LocalStore, object_key

    st = LocalStore()
    doc = st.get_attempt(run_id)
    rr = ((doc or {}).get("outputs") or {}).get("run_record")
    if not rr:
        print(f"lean_audit: {run_id} has no run record (launch it with --custody-r2)", file=sys.stderr)
        return 1
    files = {f.path: f for f in st.get_manifest(rr).payload.files or []}
    listing = next((p for p in files if p.endswith("/lean-deps.json") or p == "lean-deps.json"), None)
    if listing is None:
        print(f"lean_audit: {run_id} exported no bundle (`--export`)", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as d:
        got = json.loads((st.fetch(rr, Path(d), paths=[listing]) / listing).read_text())["deps"]
    base = listing[: -len("lean-deps.json")]
    current = {deps_key(ROOT / p) for p in packages(COMMON.tracked_files(ROOT) or [])}
    deps = {k: v for k, v in pins().items() if k in current}                  # a bumped manifest's bundle is no one's any more
    for k, v in sorted(got.items()):
        f = files.get(f"{base}{k}.tar.gz")
        head = st.remote.head(object_key(v["sha256"])) if st.remote else None
        if k not in current:
            print(f"lean_audit: {v['package']}: the bundle {k[:12]} is for another manifest or toolchain; not pinned", file=sys.stderr)
            return 1
        if f is None or (f.sha256, f.bytes) != (v["sha256"], v["bytes"]) or not head or head["bytes"] != v["bytes"]:
            print(f"lean_audit: {v['package']}: the bundle {k[:12]} is not on the remote as listed; not pinned", file=sys.stderr)
            return 1
        deps[k] = {"package": v["package"], "sha256": v["sha256"], "bytes": v["bytes"], "run": run_id}
        print(f"lean_audit: {v['package']}: pinned {k[:12]} ({v['bytes'] >> 20} MiB, {run_id})")
    (ROOT / PIN).write_text(json.dumps({"schema": "lean-deps-pin/v1", "deps": deps}, indent=1, sort_keys=True) + "\n")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--pin"] and len(argv) == 2:
        return pin(argv[1])
    if argv[:1] == ["--explain"] and len(argv) in (2, 4):
        cache = Path(argv[3]) if len(argv) == 4 and argv[2] == "--cache" else COMMON.cache_dir() / "lean-audit"
        rc, lines = explain(argv[1], cache)
        print("\n".join(lines), file=sys.stderr if rc == 2 else sys.stdout)
        return rc
    ap = argparse.ArgumentParser(prog="lean_audit.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=COMMON.cache_dir() / "lean-audit")
    ap.add_argument("--deps", type=Path, default=COMMON.cache_dir() / "lean-deps")
    ap.add_argument("--fresh", action="store_true", help="audit every package and store nothing")
    ap.add_argument("--urls", type=Path, help=f"{URLS}: presigned URLs for the pinned bundles ({PIN})")
    ap.add_argument("--export", type=Path, help="bundle the dependencies of every package that passed into this directory")
    a = ap.parse_args(argv)
    urls = json.loads(a.urls.read_text()) if a.urls and a.urls.is_file() else None
    return run(a.out, a.cache, a.deps, a.fresh, log=lambda s: print(s, flush=True), urls=urls, export=a.export)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
