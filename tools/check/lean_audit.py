"""`lean_audit`: check's Lean audit step. `tools/lean/audit.py --all --build`, with passes reused and dependencies kept warm.

    python tools/check/lean_audit.py --out DIR [--cache DIR] [--deps DIR] [--fresh]

A package's verdict is reused while everything its audit reads is unchanged. Its key hashes:
- the tracked files of the package, less the Lake packages nested in it (`audit.lean_modules` skips them);
- the same for each package it requires by path (`audit.path_deps`);
- `tools/lean/` (the auditor and its controls), this file and `check.py`, which it imports;
- the package's `lean-toolchain` and `lake-manifest.json`, which pin the toolchain and the fetched dependencies.

When one package misses, the audit runs on a scratch tree holding only the missed packages' inputs, from that tree's own copy of
`tools/lean`. So a read of anything outside a key finds nothing, and the audit fails there instead of a stale pass being kept
(the counterpart of `suites.py`'s read guard). Every package in the scratch tree is audited, the controls first. Each pass is
stored under its key, and a failure is never stored.

A package's fetched dependencies (`.lake/packages`: Mathlib's cache, ArkLib's build) are kept on this machine under
`lake-manifest.json` plus `lean-toolchain`. They're moved into the scratch package before its build and back afterwards, so Lake
rebuilds nothing it built in an earlier run. The audit's `dependencies` check still compares their `.olean` files with the
package's record, and the kernel replay still covers the package's own modules. Lean gets this machine's CPUs
(`LEAN_NUM_THREADS`): inside a pod, its default counts the host's.
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
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEAN_TOOL = "tools/lean/"
SCHEMA = "lean-audit-cache/v1"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CHECK = _module(ROOT / "tools" / "check" / "check.py", "check_for_lean_audit")


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
    return sorted({*own, *tool, "tools/check/lean_audit.py", "tools/check/check.py"})


def key(pkg: str, files: list[str], flags: list[str]) -> str:
    h = hashlib.sha256(json.dumps([SCHEMA, pkg, flags]).encode())
    for f in inputs(pkg, files):
        p = ROOT / f
        h.update(f.encode() + b"\0" + (_sha(p).encode() if p.is_file() else b"-") + b"\0")
    return h.hexdigest()


def deps_key(pkg: Path) -> str | None:
    """What a package's fetched dependencies are a function of: its manifest and toolchain (None: it fetches nothing)."""
    m = pkg / "lake-manifest.json"
    if not m.exists() or not [d for d in json.loads(m.read_text()).get("packages", []) if d.get("type") != "path"]:
        return None
    return hashlib.sha256(m.read_bytes() + b"\0" + (pkg / "lean-toolchain").read_bytes()).hexdigest()


class WarmDeps:
    """Each package's `.lake/packages` kept under its `deps_key`, moved in before a build and back after. A run holds the
    ones it moved in; another run at the same time finds them absent and builds cold."""

    def __init__(self, store: Path):
        self.store = store
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
            if src.is_dir() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                _move(src, dst)
                log(f"lean-audit: {pkg.name}: dependencies warm ({k[:12]})")
            else:
                log(f"lean-audit: {pkg.name}: dependencies cold ({k[:12]})")
        self.held[pkg] = self.store / k

    def give_back(self) -> None:
        for pkg, home in self.held.items():
            src = pkg / ".lake" / "packages"
            if not src.is_dir():
                continue
            with self._lock(home.name):
                if home.exists():
                    shutil.rmtree(home, ignore_errors=True)
                _move(src, home)


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


def run(out: Path, cache: Path, deps: Path, fresh: bool = False, log=print) -> int:
    files = CHECK.tracked_files(ROOT)
    if files is None:
        log("lean-audit: the tree's files cannot be listed; nothing is cached")
        fresh = True
        files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".lake" not in p.parts]
    flags = ["--all", "--build"]
    pkgs = packages(files)
    keys = {p: key(p, files, flags) for p in pkgs}
    out.mkdir(parents=True, exist_ok=True)
    reports, cached = {}, {}
    for p, k in keys.items():
        entry = cache / f"{k}.json"
        if not fresh and entry.is_file():
            reports[p] = json.loads(entry.read_text())["report"]
            cached[p] = k
            log(f"lean-audit: {p}: PASS reused ({k[:12]}): every file its audit reads is unchanged")
    missed = [p for p in pkgs if p not in cached]
    failed: list[str] = []
    if missed:
        deps.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="lean-audit-scratch-", dir=deps.parent))    # beside the dependencies: moves, not copies
        warm = WarmDeps(deps)
        try:
            union = sorted({f for p in missed for f in inputs(p, files)})
            scratch_tree(work, union)
            audited = [p for p in packages(union)]
            for p in audited:
                warm.take(work / p, log)
            env = {**os.environ, "LEAN_NUM_THREADS": str(CHECK.cores())}
            t0 = time.monotonic()
            r = subprocess.run([sys.executable, str(work / "tools" / "lean" / "audit.py"), *flags, "--root", str(work),
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
                    tmp.write_text(json.dumps({"schema": SCHEMA, "package": p, "key": keys[p], "report": rep}))
                    os.replace(tmp, cache / f"{keys[p]}.json")
            if r.returncode not in (0, 1) and not failed:
                failed = [f"audit.py exit {r.returncode}"]
        finally:
            warm.give_back()
            shutil.rmtree(work, ignore_errors=True)
    failed += [p for p in pkgs if p not in reports and p not in failed]
    (out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "cached": cached,
                                                "packages": [reports[p] for p in pkgs if p in reports]}, indent=1) + "\n")
    log(f"lean-audit: {'PASS' if not failed else 'FAIL (' + ', '.join(failed) + ')'}: {len(cached)} of {len(pkgs)} packages reused")
    return 0 if not failed else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lean_audit.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=CHECK.cache_dir() / "lean-audit")
    ap.add_argument("--deps", type=Path, default=CHECK.cache_dir() / "lean-deps")
    ap.add_argument("--fresh", action="store_true", help="audit every package and store nothing")
    a = ap.parse_args(argv)
    return run(a.out, a.cache, a.deps, a.fresh, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
