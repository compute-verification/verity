"""`lean_audit`: Lean's audit job, which `check` runs as its lean-audit step. `tools/verity/lean/check.py --build --closure` on the
packages whose passes can't be reused, with dependencies kept warm.

    python tools/verity/lean/lean_audit.py --out DIR [--cache DIR] [--deps DIR] [--fresh] [--urls FILE] [--export DIR] [--package PKG]...
                                                      (only those packages: a shard's share, `check.py --shard`)
                                                      [--regenerate] (a train's check: record differences pass, and
                                                      audit.json's `regenerated` names the records the train commits)
                                                      [--bundle] (a recorded check: the module versions it built, for other nodes)
    python tools/verity/lean/lean_audit.py --pin RUN        pin the bundles run RUN exported (`lean-deps.json`)
    python tools/verity/lean/lean_audit.py --restore PKG    PKG's pinned bundle as its `.lake/packages`, when it has none (`setup.sh`)
    python tools/verity/lean/lean_audit.py --explain PKG    why PKG's audit would run: its inputs against the nearest pass kept here
    python tools/verity/lean/lean_audit.py --prune          evict now what the caps keep (`DEPS_KEPT`, `BUILDS_KEPT`, `REPLAYS_KEPT`, `RECORDS_KEPT` or one per `build` slot, artifacts)

A package's verdict is reused while everything its audit reads is unchanged, wherever the package is. Its key hashes content,
each file under its path relative to its package (a module's is its name), never the package's own path:
- the tracked files of the package, less the Lake packages nested in it (`audit.lean_modules` skips them), its
  `lean-toolchain`, `lake-manifest.json` and `lean-audit.json` among them;
- each package its `proved_in` names (`provers`), whose build the audit reads its guarantees from, by its content (which
  names the packages it requires by theirs), under its path;
- the same for each package it requires by path (`audit.path_deps`), under the name its manifest gives it. Where the
  lakefile's `require ... path = "P"` and the manifest's `"dir": "P"` name such a package, the key reads P as that
  package's own content key (`content_key`), so moving it and rewriting the paths to it keeps every key. A path spelled
  otherwise stays a path in the key;
- `tools/verity/lean/` (the auditor and this file) and `tools/verity/check/common.py`, the helpers it imports, at their paths in the
  repository: not `check.py`, which only launches it, so an edit to `check.py` alone keeps every package's verdict. Nor
  `tools/verity/lean/tests/` (the controls, which the `tools/verity/lean` suite runs), its `*.md`, or the files `NOT_READ` lists
  (the merge driver, the lean-changed step, the daily publish, ...), which the audit never reads: the scratch tree leaves them
  out too, so a read of one would fail there.
A `git mv` of a package with its content unchanged is therefore no miss, and its kept pass answers at the new path. What the
scratch tree can't guard is a package reading beyond its directory by a relative path, which would find other files once it
moved: a `..` left (after the paths above) in its lakefile, manifest, toolchain or policy, or an `include_str` in a module,
puts the package's path back into its key.

When one package misses, the audit runs on a scratch tree holding only the missed packages' inputs, from that tree's own copy of
`tools/verity/lean` (named by its creator's pid; an audit first removes every scratch tree no audit holds, `sweep`). So a read of anything outside a key finds nothing, and the audit fails there instead of a stale pass being kept
(the counterpart of `suites.py`'s read guard). Only the missed packages are audited. A package one of them requires by path is
in the tree and built as far as it imports it, but not audited again: what vouches for its modules is its own pass, under its
own key, and the run passes only when every package has one. The dependent's axiom walk follows its declarations into those
modules, but its replay and its record's `dependencies` (fetched packages and the toolchain) don't cover them. A spec and a
package its `proved_in` names, which requires it, read the same files, so they miss together and are audited together. The controls
don't run here: they test
the auditor, so the `tools/verity/lean` suite runs them, cached on `tools/verity/lean/`, not again on every package's miss. Each pass is
stored under its key, and a failure is never stored. One read is outside the scratch tree: a package test's `generate` (`runs`
in `lean-audit.json`) reads this checkout (`check.py --repo`). Its output is used only with the SHA-256 the record holds for
it, and the key covers the record, so it cannot pass a test on other data. The package's own Python tests check that the
generator still writes that data.

A package's fetched dependencies (`.lake/packages`: Mathlib's cache, ArkLib's build) are kept on this machine under
`lake-manifest.json` plus `lean-toolchain`. They're moved into the scratch package's own `.lake` before its build (the only place
its sandbox lets it write, never a shared symlink) and back afterwards only when that package's audit passed; after a failure
they're dropped, so a run's compile-time code cannot leave the next run a cache it wrote. Lake rebuilds nothing it built in an
earlier passing run. The package's own `.lake/build` is kept the same way (`WarmBuilds`), under its toolchain, manifest and
lakefile less the paths to what it requires (`build_key`), not under its path: Lake decides what is up to date by content
hashes, not paths or mtimes, so a moved package rebuilds nothing. The audit's `dependencies` check still compares their
`.olean` files with the package's record, and the kernel replay still covers the package's own modules and their whole import
closure, the toolchain's, Mathlib's and ArkLib's modules included (`check.py --closure`): each module whose `.olean` files, and
those of every module it imports, transitively, passed an earlier replay here or in a recorded check whose verdict pack this
one imported (`check.gather_verdicts`) is reused (`check.py --replay-cache`, `lean-replay/` beside the passes, its
`REPLAYS_KEPT` last used kept), so a change to a few modules replays those and what imports them, and a dependency's
modules go through the kernel once per change of their `.olean` files. With `--fresh` the replay cache is
the scratch tree's, so the packages of one run share it and nothing outlives the run. Lean gets this
machine's CPUs (`LEAN_NUM_THREADS`): inside a pod, its default counts the host's.

A machine without a package's dependencies restores them from the evidence store instead of cloning them from GitHub, which
throttles anonymous clones from shared addresses. `lean-deps.json` pins one bundle per `lake-manifest.json`'s fetched entries + `lean-toolchain`
(`deps_key`; a `path` entry changes no key) by sha256; the launcher sends a presigned URL for each (`--urls`, `check.py --record`), and a bundle is used only when its
bytes are the pinned ones. A bundle leaves out the builds Mathlib's cache supplies: `lake exe cache get` takes those from
Mathlib's CDN, which is not GitHub. After a bump, a passing cold audit run with `--export` writes the new bundles into its run
directory, custody puts them on R2, and `--pin RUN` pins them.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEAN_TOOL = "tools/verity/lean/"
SCHEMA = "lean-audit-cache/v2"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


#: every file this step loads is in `inputs`: `tools/verity/lean/` (this module among them) and `common.py` (`test_lean_audit.py` holds it)
COMMON = _module(ROOT / "tools" / "verity" / "check" / "common.py", "check_common_for_lean_audit")


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


def provers(pkg: str) -> list[str]:
    """The packages ``pkg``'s `lean-audit.json` names under `proved_in` (`audit.provers`): the audit reads their builds."""
    p = ROOT / pkg / "lean-audit.json"
    try:
        named = json.loads(p.read_text()).get("proved_in", []) if p.is_file() else []
    except (OSError, ValueError, AttributeError):
        return []                                       # the audit fails the policy
    return sorted({os.path.relpath(os.path.normpath(ROOT / pkg / d), ROOT) for d in named if isinstance(d, str)})


#: How the audit runs, and so part of every key
FLAGS = ["--build", "--closure"]


#: the files of `tools/verity/lean/` no audit reads, each with what does: out of every key, and of the scratch tree, where a
#: read of one fails the audit
NOT_READ = {
    "lean-deps.json": "the dependency bundles' pins, which preflight reads",
    "lean_changed.py": "check's lean-changed step and the Lean fast path, which run beside the audit",
    "merge.py": "git's merge driver for lean-audit.json",
    "run_daily.py": "the build cache's daily publish",
    "tool.py": "the store's declarations of the scheduled runs",
    "Upstream.lean": "`upstream.py --lean`, the exact pass; the audit runs the no-build one",
    "pyproject.toml": "the tools/verity/lean suite's settings",
}


def tool_files(files: list[str]) -> list[str]:
    """The auditor's files: `tools/verity/lean/` less its tests, its docs and `NOT_READ`, and the helpers it imports."""
    tool = [f for f in files if f.startswith(LEAN_TOOL) and not f.startswith(LEAN_TOOL + "tests/") and not f.endswith(".md")
            and f[len(LEAN_TOOL):] not in NOT_READ]
    return sorted({*tool, "tools/verity/check/common.py"})


def inputs(pkg: str, files: list[str]) -> list[str]:
    """Every tracked file an audit of ``pkg`` can read."""
    pkgs = [pkg, *path_deps(pkg), *(d for q in provers(pkg) for d in (q, *path_deps(q)))]
    own = [f for p in dict.fromkeys(pkgs) for f in own_files(p, files)]
    return sorted({*own, *tool_files(files)})


def _required(pkg: str, files: list[str]) -> dict[str, str]:
    """{a path as the package's lakefile or manifest spells it: the repository path of the Lake package there}, for each
    `require ... path` and each path entry of the manifest that names a directory holding a tracked lakefile."""
    spelled = []
    m, lf = ROOT / pkg / "lake-manifest.json", ROOT / pkg / "lakefile.toml"
    if m.is_file():
        spelled += [d["dir"] for d in json.loads(m.read_text()).get("packages", []) if d.get("type") == "path" and "dir" in d]
    if lf.is_file():
        try:
            reqs = tomllib.loads(lf.read_text()).get("require", [])
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            reqs = []
        spelled += [r["path"] for r in reqs if isinstance(r, dict) and isinstance(r.get("path"), str)]
    out = {}
    for s in spelled:
        d = os.path.relpath(os.path.normpath(ROOT / pkg / s), ROOT)
        if any(f"{d}/{n}" in files for n in ("lakefile.toml", "lakefile.lean")) and d != pkg:
            out[s] = d
    return out


TOKEN = b"content:"


def _spelled_as(text: bytes, name: str, spelled: dict[str, str], token) -> tuple[bytes, str]:
    """``text`` (a lakefile.toml, ``name`` "path", or a manifest, ``name`` "dir") with each `name = "S"` or `"name": "S"` whose
    S is in ``spelled`` written as ``token(spelled[S])``, and its digest, under a prefix of its own so that it is never a file's
    own digest. A path spelled any other way stays as it is, and so does a file that already holds a token's prefix: the key
    then covers the path."""
    if TOKEN in text:
        return text, hashlib.sha256(text).hexdigest()
    for s, d in sorted(spelled.items()):
        lit = re.escape(json.dumps(s).encode())
        pat = rb'("dir"\s*:\s*)' + lit if name == "dir" else rb'(?m)^([ \t]*path[ \t]*=[ \t]*)' + lit
        text = re.sub(pat, lambda m, d=d: m.group(1) + json.dumps(TOKEN.decode() + token(d)).encode(), text)
    return text, hashlib.sha256(b"lean-audit/paths-as-content\0" + text).hexdigest()


#: what stands for the package's path in its `content` when something Lake or the audit reads may name a path beyond it
WHERE = "(where the package is)"
#: the files of a package Lake or the audit read paths from
CONFIG = ("lakefile.toml", "lakefile.lean", "lake-manifest.json", "lean-toolchain", "lean-audit.json")


def content(pkg: str, files: list[str], _memo: dict | None = None) -> dict[str, str]:
    """{path relative to the package: sha256} of its own files, so that the package moved to another directory with the same
    content has the same map. Its lakefile and manifest name each package it requires by path by that package's own content
    (`content_key`), not by where it is: moving the required package and rewriting the paths to it keeps the map too. A package
    that may read beyond its directory by a relative path keeps its path in the map (`WHERE`): a `..` left in a file Lake or
    the audit read paths from, or a module's `include_str`."""
    memo = {} if _memo is None else _memo
    spelled = _required(pkg, files)
    out, escapes = {}, False
    for f in own_files(pkg, files):
        rel, p = f[len(pkg) + 1:], ROOT / f
        if not p.is_file():
            out[rel] = "-"
            continue
        text = p.read_bytes()
        if rel in ("lakefile.toml", "lake-manifest.json") and spelled:
            text, out[rel] = _spelled_as(text, "path" if rel == "lakefile.toml" else "dir", spelled,
                                         lambda d: content_key(d, files, memo))
        else:
            out[rel] = hashlib.sha256(text).hexdigest()
        escapes |= (b".." in text) if rel in CONFIG else rel.endswith(".lean") and b"include_str" in text
    if escapes:
        out[WHERE] = pkg
    return out


def content_key(pkg: str, files: list[str], _memo: dict | None = None) -> str:
    memo = {} if _memo is None else _memo
    if pkg not in memo:
        memo[pkg] = hashlib.sha256(json.dumps(sorted(content(pkg, files, memo).items())).encode()).hexdigest()
    return memo[pkg]


def _lake_name(pkg: str, dep: str) -> str:
    """The name ``pkg``'s manifest gives the package it requires at ``dep`` (its path when it has none)."""
    m = ROOT / pkg / "lake-manifest.json"
    for d in json.loads(m.read_text()).get("packages", []) if m.is_file() else ():
        if d.get("type") == "path" and os.path.relpath(os.path.normpath(ROOT / pkg / d.get("dir", "")), ROOT) == dep:
            return str(d.get("name", dep))
    return dep


def components(pkg: str, files: list[str], flags: list[str]) -> dict:
    """What a package's key hashes, none of it a path of the package: the schema, the audit's flags, the package's `content`,
    that of each package it requires by path (under the name its manifest gives it), that of each package its `proved_in`
    names (under its path, which the policy already holds), and each of the auditor's files at its
    path in the repository. A pass keeps them beside it, so `--explain` can say what a miss changed."""
    memo: dict = {}
    return {"schema": SCHEMA, "flags": flags, "files": content(pkg, files, memo),
            "requires": {_lake_name(pkg, d): content(d, files, memo) for d in path_deps(pkg)},
            "proved_in": {q: content(q, files, memo) for q in provers(pkg)},
            "tool": {f: (_sha(ROOT / f) if (ROOT / f).is_file() else "-") for f in tool_files(files)}}


def key_of(c: dict) -> str:
    return hashlib.sha256(json.dumps([c["schema"], c["flags"], sorted(c["files"].items()),
                                      sorted((n, sorted(m.items())) for n, m in c["requires"].items()),
                                      sorted(c["tool"].items()),
                                      sorted((q, sorted(m.items())) for q, m in c.get("proved_in", {}).items())]).encode()).hexdigest()


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
    now = components(pkg, files, FLAGS)
    k = key_of(now)
    if (cache / f"{k}.json").is_file():
        return 0, [f"{pkg}: PASS reused ({k[:12]}): every file its audit reads is unchanged"]
    kept = []
    for f in cache.glob("*.json") if cache.is_dir() else ():
        try:
            e = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        # a pass of this package, or of one audited where no package is now (it may be this one, before it moved)
        if (e.get("components") or {}).get("schema") == SCHEMA and (e.get("package") == pkg or e.get("package") not in pkgs):
            kept.append(e)
    if not kept:
        return 0, [f"{pkg}: missed ({k[:12]}); no pass of it kept in {cache} records its inputs (passes kept before `--explain` don't)"]
    where = {_lake_name(pkg, d): d for d in path_deps(pkg)}

    def flat(c: dict) -> dict[str, str]:
        """Each input under its path now: the package's files under ``pkg``, a required package's under where it is now."""
        return {**{f"{pkg}/{r}": d for r, d in c["files"].items()}, **c["tool"],
                **{f"{where.get(n, n + ' (required by path)')}/{r}": d for n, m in c["requires"].items() for r, d in m.items()},
                **{f"{q}/{r}": d for q, m in c.get("proved_in", {}).items() for r, d in m.items()}}

    def diff(was: dict) -> list[str]:
        a, b = flat(was), flat(now)
        named = {f"{pkg}/lakefile.toml", f"{pkg}/lake-manifest.json"}
        out = [f"{'changed' if f in a and f in b else 'new' if f in b else 'no longer an input'}: {f}"
               + (" (it names each package it requires by path by that package's content)"
                  if f in named and was["requires"] != now["requires"] else "")
               for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
        return out + [f"{x}: {was.get(x)} -> {now[x]}" for x in ("schema", "flags") if was.get(x) != now[x]]

    best = min(kept, key=lambda e: (len(diff(e["components"])), e.get("package") != pkg))
    lines = diff(best["components"])
    at = "" if best.get("package") == pkg else f", audited at {best.get('package')}"
    return 0, [f"{pkg}: missed ({k[:12]}); the nearest pass kept here ({best['key'][:12]}{at}) differs in {len(lines)} input(s):",
               *(f"  {line}" for line in lines)]


def deps_key(pkg: Path) -> str | None:
    """What a package's fetched dependencies are a function of: its manifest's fetched entries and its toolchain (None: it
    fetches nothing). A `path` entry is not in it: `.lake/packages` holds only fetched packages, so requiring another
    package of this repository by path needs no new bundle."""
    m = pkg / "lake-manifest.json"
    doc = json.loads(m.read_text()) if m.exists() else {}
    fetched = [d for d in doc.get("packages", []) if d.get("type") != "path"]
    if not fetched:
        return None
    canon = json.dumps({"packagesDir": doc.get("packagesDir"), "packages": fetched}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode() + b"\0" + (pkg / "lean-toolchain").read_bytes()).hexdigest()


PIN = "tools/verity/lean/lean-deps.json"
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

    def __init__(self, store: Path, urls: dict[str, str] | None = None, kept: int | None = None):
        self.store, self.urls, self.kept = store, urls or {}, DEPS_KEPT if kept is None else kept
        self.held: dict[Path, Path] = {}

    def _lock(self, k: str):
        self.store.mkdir(parents=True, exist_ok=True)
        f = open(self.store / f"{k}.lock", "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    def take(self, pkg: Path, log=print) -> None:
        """Move ``pkg``'s kept dependencies in; when another package of this run with the same manifest and toolchain holds
        them, copy its tree (taken before either builds), never shared: each build may write in its own."""
        if (k := deps_key(pkg)) is None:
            return
        dst = pkg / ".lake" / "packages"
        with self._lock(k):
            src = self.store / k
            twin = next((p for p, h in list(self.held.items()) if h.name == k and (p / ".lake" / "packages").is_dir()), None)
            if not src.is_dir() and not dst.exists() and twin is None and k in self.urls and (pin := pins().get(k)):
                restore(self.store, k, self.urls[k], pin, pkg.name, log)
            if src.is_dir() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                _move(src, dst)
                log(f"lean-audit: {pkg.name}: dependencies warm ({k[:12]})")
            elif twin is not None and not dst.exists():
                t0 = time.monotonic()
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(twin / ".lake" / "packages", dst, symlinks=True)
                log(f"lean-audit: {pkg.name}: dependencies warm ({k[:12]}), copied from {twin.name}'s in "
                    f"{time.monotonic() - t0:.0f} s")
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
                os.utime(home)
        self.evict()

    def evict(self) -> int:
        """Remove all but the `kept` dependency trees last given back (`evict_lru`); the number removed."""
        return evict_lru(self.store, self.kept)

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


def build_key(pkg: str, root: Path) -> str:
    """What a kept build is reused under: the package's toolchain, its manifest and its lakefile, less the paths to the packages
    it requires by path, and not where the package is: a moved package takes the build it left, which Lake reuses where it
    is now. Within that, Lake's own traces (each module's source and its imports' traces) decide what it rebuilds."""
    h = hashlib.sha256(b"lean-build/v2\0")
    for f in ("lean-toolchain", "lake-manifest.json", "lakefile.toml", "lakefile.lean"):
        text = (root / pkg / f).read_bytes() if (root / pkg / f).is_file() else b"-"
        if f == "lake-manifest.json":
            text = re.sub(rb'"dir"\s*:\s*"[^"]*"', b'"dir": ""', text)
        elif f == "lakefile.toml":
            text = re.sub(rb'(?m)^([ \t]*path[ \t]*=).*$', rb"\1", text)
        h.update(f.encode() + b"\0" + text + b"\0")
    return h.hexdigest()


def source_dirs(pkg: Path) -> list[Path]:
    """The directories a package's module names are relative to: its lakefile's `srcDir` and each `lean_lib`'s under it
    (Lake's default `.` for each), as `move_check`'s `source_dirs` reads them. `Security/Proofs` has `srcDir = ".."`, so its sources
    sit beside it, not in it."""
    lake = pkg / "lakefile.toml"
    cfg = tomllib.loads(lake.read_text()) if lake.is_file() else {}
    top = pkg / cfg.get("srcDir", ".")
    return [top, *(top / lib.get("srcDir", ".") for lib in cfg.get("lean_lib", []))]


def prune(pkg: Path) -> int:
    """Remove the kept outputs of modules whose source is gone, so no `.olean` outlives its `.lean`; the number removed. A
    module's source is looked for under each of the package's `source_dirs`."""
    n, srcs = 0, source_dirs(pkg)
    for sub in ("lib/lean", "ir"):
        base = pkg / ".lake" / "build" / sub
        for f in sorted(base.rglob("*")) if base.is_dir() else ():
            src = f.relative_to(base).parent / (f.name.split(".")[0] + ".lean")
            if f.is_file() and not any((d / src).is_file() for d in srcs):
                f.unlink()
                n += 1
    return n


#: the kept builds a node holds before its least recently used go: every tree whose manifest or lakefile differs adds one
#: (a node collects dozens, tens of GB). Each package's newest stays past it (`WarmBuilds.evict`), and the limit stays above
#: the ~30 packages: below them, every check would evict some and rebuild them cold (Flock's build alone takes about 18 min)
BUILDS_KEPT = 48
#: the kept dependency trees (one per manifest and toolchain, `deps_key`) a node holds before its least recently used go:
#: Security/Proofs' is about 35k files, so a few dozen of them hold most of a node's inodes
DEPS_KEPT = 8
#: the replay cache's entries a node keeps, last used first (a replay touches each entry it reuses): one per module and key,
#: about 3k a version of the tree plus some 8k of Mathlib's and ArkLib's that every replay reuses, each a few hundred bytes;
#: a node left to grow holds over 100k
REPLAYS_KEPT = 60_000


class WarmBuilds:
    """Each package's own `.lake/build`, kept under its `build_key` by the last run whose audit of it passed, so an auditor
    change, or a change to one module, doesn't rebuild every module. An audited package's is moved in and back only when its
    audit passed: its compile-time code may write in it. A package in the tree only as a path dependency gets a copy, which
    never comes back. The audit still checks every statement against these `.olean` files and replays every declaration
    through the kernel. The newest `kept` by use stay (`evict`)."""

    def __init__(self, store: Path, kept: int = BUILDS_KEPT):
        self.store, self.kept = store, kept

    def _lock(self, k: str):
        self.store.mkdir(parents=True, exist_ok=True)
        f = open(self.store / f"{k}.lock", "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    def take(self, work: Path, pkg: str, audited: bool, log=print) -> None:
        k = build_key(pkg, work)
        src, dst = self.store / k, work / pkg / ".lake" / "build"
        with self._lock(k):
            if not src.is_dir() or dst.exists():
                log(f"lean-audit: {pkg}: build cold ({k[:12]})")
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            if audited:
                _move(src, dst)
            else:
                shutil.copytree(src, dst, symlinks=True)
                os.utime(src)
        log(f"lean-audit: {pkg}: build warm ({k[:12]}){'' if audited else ', a copy'}; {prune(work / pkg)} stale output(s) removed")

    def give_back(self, work: Path, passed: set[Path]) -> None:
        for pkg in passed:
            src = pkg / ".lake" / "build"
            if not src.is_dir():
                continue
            rel = os.path.relpath(pkg, work)
            home = self.store / build_key(rel, work)
            with self._lock(home.name):
                if home.exists():
                    shutil.rmtree(home, ignore_errors=True)
                _move(src, home)
                os.utime(home)
                (self.store / f"{home.name}.pkg").write_text(rel)
        self.evict()

    def newest_per_package(self) -> set[str]:
        """The newest kept build of each package a `NAME.pkg` beside it names."""
        newest: dict[str, tuple[float, str]] = {}
        for tag in self.store.glob("*.pkg") if self.store.is_dir() else ():
            home = self.store / tag.stem
            try:
                pkg, t = tag.read_text().strip(), home.stat().st_mtime
            except OSError:                            # taken while listed, or its build is gone
                continue
            if pkg not in newest or t > newest[pkg][0]:
                newest[pkg] = (t, home.name)
        return {name for _, name in newest.values()}

    def evict(self) -> int:
        """Remove all but the `kept` builds last given back or copied from, never a package's newest (`evict_lru`); the number
        removed."""
        return evict_lru(self.store, self.kept, keep=self.newest_per_package())


def evict_lru(store: Path, kept: int, keep: set[str] = frozenset()) -> int:
    """Remove all but the ``kept`` directories of ``store`` last used (a rename keeps a directory's mtime, so each use sets
    it), and never one named in ``keep``, each only when its lock (`NAME.lock` beside it) is free: a run holding it is taking
    it or giving it back. The number removed."""
    held = []
    for home in store.iterdir() if store.is_dir() else ():
        try:
            if home.is_dir():
                held.append((home.stat().st_mtime, home))
        except OSError:                                # taken while listed
            pass
    n = 0
    for _, home in sorted(held, reverse=True)[kept:]:
        if home.name in keep:
            continue
        with open(store / f"{home.name}.lock", "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if home.is_dir():
                shutil.rmtree(home, ignore_errors=True)
                (store / f"{home.name}.pkg").unlink(missing_ok=True)
                n += 1
    return n


#: the fast path's (`lean_changed.py --records`) trees a node keeps, its most recently used: each is a checkout with its own
#: builds, about 40 GB and 490k inodes (nine of them nearly fill a node's inodes). A node with more
#: `build` Lean slots keeps one per slot (`build_slots`)
RECORDS_KEPT = 3
#: `lean_slot.py`'s SLOTS: the node's Lean slot caps, `build N` among them
LEAN_SLOTS = Path(os.environ.get("LEAN_SLOTS", "/workspace/research/locks/lean-slots"))
#: `lean_changed.py`'s kept passes under `lean-records/`, not a tree: they have their own cap (`PASSES_KEPT`)
RECORDS_PASSES = "passes"


def build_slots(slots: Path | None = None) -> int:
    """The node's `build` Lean slots (``slots``' `build N`), 0 when it lists none or can't be read."""
    try:
        text = (slots or LEAN_SLOTS).read_text()
    except OSError:
        return 0
    for line in text.splitlines():
        f = line.split("#")[0].split()
        if len(f) == 2 and f[0] == "build" and f[1].isdigit():
            return int(f[1])
    return 0


def prune_records(store: Path, kept: int | None = None, slots: int | None = None) -> int:
    """Remove all but ``kept`` trees of ``store`` (`lean-records/`), each only when its `.lock` is free: a run holding or
    waiting on it keeps it. ``kept`` is `RECORDS_KEPT`, or the node's ``slots`` (`build_slots`) when it has more, since each
    slot's tree is the one a run there takes, and removing it hands that run a cold one (an hourly prune that kept only
    `RECORDS_KEPT` would take trees seeded for the slots). The slots' own trees (`build-I`, I below ``slots``) are kept
    first, then the rest by last use, the newest of their `.who`, `links.json` and `.lock`. A link to a tree elsewhere goes
    with its target, and `passes/` is never touched. The number removed."""
    slots = build_slots() if slots is None else slots
    kept = max(RECORDS_KEPT, slots) if kept is None else kept
    own = {f"build-{i}" for i in range(slots)}
    used = []
    for tree in store.iterdir() if store.is_dir() else ():
        try:
            if tree.is_dir() and tree.name != RECORDS_PASSES:
                stamps = [p.stat().st_mtime for p in (tree / ".who", tree / "links.json", tree / ".lock") if p.exists()]
                used.append((tree.name in own, max(stamps, default=tree.stat().st_mtime), tree))
        except OSError:
            pass
    n = 0
    for _, _, tree in sorted(used, reverse=True)[kept:]:
        try:
            with open(tree / ".lock", "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                target = tree.resolve()
                shutil.rmtree(target, ignore_errors=True)
                if tree.is_symlink():
                    tree.unlink()
                n += 1
        except (BlockingIOError, OSError):
            continue
    return n


def prune_replays(cache: Path, kept: int | None = None) -> int:
    """Remove all but the ``kept`` (`REPLAYS_KEPT`) replay entries of ``cache`` (`check.py --replay-cache`) last stored or
    reused (`check.touch_replays`): each is one module's pass under its key, so one dropped costs that module one replay. The
    number removed."""
    kept = REPLAYS_KEPT if kept is None else kept
    try:
        entries = [(e.stat().st_mtime, e) for e in os.scandir(cache) if e.name.endswith(".json")]
    except OSError:
        return 0
    n = 0
    for _, e in sorted(entries, key=lambda x: x[0], reverse=True)[kept:]:
        try:
            os.unlink(e.path)
            n += 1
        except OSError:
            pass
    return n


#: the disk the artifact cache may hold beyond what kept builds share with it (hard links), before its oldest go
ARTIFACTS_CAP = 20 << 30
ARTIFACT = re.compile(r"[0-9a-f]{16}\.[a-z.]+")
#: a bundle's members: Lake's cache layout, nothing else
MEMBER = re.compile(r"artifacts/[0-9a-f]{16}\.[a-z.]+|outputs/(?!\.\.?/)[A-Za-z0-9_.-]+/[0-9a-f]{16}\.json")
#: what one run's bundle (`LakeArtifacts.bundle`) holds at most, uncompressed: a run's files over 4 GiB upload without resume
BUNDLE_CAP = 3 << 30
#: a key of `lean-deps-urls.json` naming a recorded check's bundle by its sha256 (`shared_bundles`)
SHARED = "lean-artifacts/"
#: how many of the newest recorded checks' bundles the launcher sends
SHARED_LIMIT = 12


class LakeArtifacts:
    """The node's Lake artifact cache, in Lake's own layout (`artifacts/<hash>.<ext>`, `outputs/<package>/<input hash>.json`).
    Every audit's builds read it (`check.lake_env`) and never write it; after an audit, each module of a package whose audit
    passed goes in from its build's trace. A module version any earlier passing audit on this node built is then restored
    whatever tree comes next, not only the one its package's kept build (`WarmBuilds`) came from. Lake keys a module's
    outputs by its input hash (its source, its imports' outputs, the toolchain and the options), and the audit still checks
    every statement against the restored `.olean` files and replays every declaration through the kernel.

    Between nodes, the module versions a recorded check's audit added travel as one bundle in its run's files (`bundle`); the
    launcher sends a URL for each of the newest (`shared_bundles`), and a node takes in each it hasn't yet (`take`). A bundle is
    trusted as far as its source, as a verdict pack is (`verdicts.py`): a recorded run."""

    def __init__(self, store: Path, cap: int = ARTIFACTS_CAP):
        self.store, self.cap = store, cap
        #: (entry, its outputs) of each module version this object stored that the cache lacked
        self.added: list[tuple[str, list[str]]] = []

    def _lock(self):
        self.store.mkdir(parents=True, exist_ok=True)
        f = open(self.store / "lock", "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    def put(self, pkgs: set[Path]) -> int:
        """Store the modules of ``pkgs``' builds; the number the cache lacked. A module goes in only when each output its
        trace names is in the build with that hash (`<file>.hash`), its outputs before the entry naming them."""
        with self._lock():
            n = sum(self._module(pkg, t) for pkg in sorted(pkgs) for t in sorted((pkg / ".lake/build/lib/lean").rglob("*.trace")))
            self.prune()
        return n

    def _module(self, pkg: Path, trace: Path) -> bool:
        build = pkg / ".lake" / "build"
        try:
            d = json.loads(trace.read_text())
            scope = json.loads((pkg / "lake-manifest.json").read_text())["name"]
        except (OSError, ValueError, KeyError):
            return False
        out, h = d.get("outputs"), d.get("depHash")
        if not isinstance(out, dict) or not isinstance(h, str):
            return False
        mod = str(trace.relative_to(build / "lib" / "lean"))[: -len(".trace")]
        files = {}
        for name in _strings(out):
            if not ARTIFACT.fullmatch(name):
                continue
            digest, ext = name.split(".", 1)
            f = build / ("ir" if ext in ("c", "bc", "ltar") else "lib/lean") / f"{mod}.{ext}"
            try:
                if (f.parent / f"{f.name}.hash").read_text().strip() != digest:
                    return False
            except OSError:
                return False
            files[name] = f
        if not files:
            return False
        arts = self.store / "artifacts"
        arts.mkdir(exist_ok=True)
        for name, f in files.items():
            dst = arts / name
            if dst.exists():
                os.utime(dst)
                continue
            tmp = arts / f".{name}.{os.getpid()}.tmp"
            try:
                os.link(f, tmp)
            except OSError:
                shutil.copyfile(f, tmp)
            os.chmod(tmp, 0o444)
            os.replace(tmp, dst)
        entry = self.store / "outputs" / scope / f"{h}.json"
        entry.parent.mkdir(parents=True, exist_ok=True)
        new = not entry.exists()
        tmp = entry.parent / f".{h}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(out))
        os.replace(tmp, entry)
        if new:
            self.added.append((f"outputs/{scope}/{h}.json", sorted(files)))
        return new

    def bundle(self, dest: Path, cap: int = BUNDLE_CAP) -> int:
        """Write the module versions `added` as a gzipped tar at ``dest``, each with every output it names, outputs before
        entries, up to ``cap`` bytes uncompressed; the number of entries written (none: no file)."""
        with self._lock():
            todo, size, names = [], 0, set()
            for entry, outs in self.added:
                try:
                    more = sum((self.store / "artifacts" / n).stat().st_size for n in outs if n not in names)
                    more += (self.store / entry).stat().st_size
                except OSError:
                    continue
                if size + more > cap:
                    break
                todo.append((entry, outs))
                names.update(outs)
                size += more
            if not todo:
                return 0
            tmp = dest.with_name(f".{dest.name}.tmp")
            with tarfile.open(tmp, "w:gz", compresslevel=1) as t:
                for n in sorted(names):
                    t.add(self.store / "artifacts" / n, arcname=f"artifacts/{n}", recursive=False)
                for entry, _ in todo:
                    t.add(self.store / entry, arcname=entry, recursive=False)
            os.replace(tmp, dest)
        return len(todo)

    def take(self, urls: dict[str, str], log=print) -> int:
        """Take in each bundle ``urls`` names (`SHARED` + its sha256) that this node hasn't: downloaded, checked against its
        sha256, and unpacked member by member, each a file of Lake's layout (`MEMBER`) that the cache lacks, outputs before
        the entries naming them; a file here is never replaced. The number of entries added. A bundle that fails is skipped,
        and its modules are built; one whose bytes fail (not the sha256 named, or not unpackable) is marked taken, since they
        never change. The URL is never logged: it grants a read until it expires."""
        want = {k[len(SHARED):]: u for k, u in sorted(urls.items()) if k.startswith(SHARED)}
        taken = self.store / "taken"
        want = {h: u for h, u in want.items() if re.fullmatch(r"[0-9a-f]{64}", h) and not (taken / h).exists()}
        if not want:
            return 0
        taken.mkdir(parents=True, exist_ok=True)
        t0, n = time.monotonic(), 0

        def get(h: str, url: str) -> Path | None:
            part = self.store / f".{h}.{os.getpid()}.part"
            try:
                d = hashlib.sha256()
                with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                    for chunk in iter(lambda: r.read(1 << 20), b""):
                        d.update(chunk)
                        f.write(chunk)
                if d.hexdigest() == h:
                    return part
                log(f"lean-audit: artifact bundle {h[:12]} is not the one named; skipped")
                (taken / h).touch()
            except OSError as e:
                log(f"lean-audit: artifact bundle {h[:12]} could not be fetched ({getattr(e, 'code', None) or type(e).__name__})")
            part.unlink(missing_ok=True)
            return None

        with ThreadPoolExecutor(max_workers=4) as ex:
            parts = dict(zip(want, ex.map(lambda h: get(h, want[h]), want)))
        with self._lock():
            for h, part in parts.items():
                if part is None:
                    continue
                try:
                    n += self._unpack(part)
                    (taken / h).touch()
                except (tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile) as e:
                    log(f"lean-audit: artifact bundle {h[:12]} could not be unpacked ({type(e).__name__}); skipped")
                    (taken / h).touch()
                except OSError as e:
                    log(f"lean-audit: artifact bundle {h[:12]} could not be unpacked ({type(e).__name__}); skipped")
                finally:
                    part.unlink(missing_ok=True)
            self.prune()
        log(f"lean-audit: {n} module version(s) taken from {sum(p is not None for p in parts.values())} of {len(want)} "
            f"artifact bundle(s) in {time.monotonic() - t0:.0f} s")
        return n

    def _unpack(self, part: Path) -> int:
        n, entries = 0, []
        with tarfile.open(part, "r:gz") as t:
            for m in t:
                if not m.isfile() or not MEMBER.fullmatch(m.name):
                    continue
                if m.name.startswith("outputs/"):
                    entries.append((m.name, t.extractfile(m).read()))
                    continue
                dst = self.store / m.name
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
                with t.extractfile(m) as src, open(tmp, "wb") as f:
                    shutil.copyfileobj(src, f)
                os.chmod(tmp, 0o444)
                os.replace(tmp, dst)
        for name, data in entries:
            dst = self.store / name
            try:
                outs = [v for v in _strings(json.loads(data)) if ARTIFACT.fullmatch(v)]
            except ValueError:
                continue
            if dst.exists() or not outs or not all((self.store / "artifacts" / v).exists() for v in outs):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            n += 1
        return n

    def prune(self) -> int:
        """Remove the least recently stored artifacts while those only the cache holds exceed the cap, and the entries no
        newer than them; the number of artifacts removed. An entry whose output is gone is a miss, so Lake rebuilds."""
        arts = [(s.st_mtime, s.st_size if s.st_nlink == 1 else 0, f) for f in (self.store / "artifacts").glob("*")
                if (s := f.stat())]
        own = sum(size for _, size, _ in arts)
        if own <= self.cap:
            return 0
        n, horizon = 0, 0.0
        for mtime, size, f in sorted(arts):
            if own <= self.cap * 3 // 4:
                break
            f.unlink(missing_ok=True)
            own, n, horizon = own - size, n + 1, mtime
        for e in (self.store / "outputs").glob("*/*.json"):
            if e.stat().st_mtime <= horizon:
                e.unlink(missing_ok=True)
        return n


def _strings(j) -> list[str]:
    if isinstance(j, str):
        return [j]
    return [s for v in (j.values() if isinstance(j, dict) else j if isinstance(j, list) else ()) for s in _strings(v)]


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


SCRATCH = "lean-audit-scratch-"
#: how long a scratch tree without its creator's pid in its name is kept unheld: an older check's, which may be filling it
SCRATCH_GRACE_S = 3600


def _links(proc: Path) -> tuple[list[str], bool]:
    """(every cwd, root and open file of the processes under ``proc``, whether some process's could not be read)."""
    out, blind = [], False
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            names = [p / "cwd", p / "root", *(p / "fd").iterdir()]
        except FileNotFoundError:
            continue
        except OSError:
            blind = True
            continue
        for n in names:
            try:
                out.append(os.readlink(n))
            except FileNotFoundError:
                pass
            except OSError:
                blind = True
    return out, blind


def sweep(base: Path, log=print, proc: Path = Path("/proc"), now: float | None = None) -> list[str]:
    """Remove each ``lean-audit-scratch-*`` under ``base`` that no audit holds: one an audit left when it was killed before its
    ``finally``.  A tree is held while a live process has its cwd, root or an open file inside it, while the pid its name holds
    is alive, or, for a name without one, while some process can't be read or the tree changed within ``SCRATCH_GRACE_S``."""
    trees = [d for d in base.glob(SCRATCH + "*") if d.is_dir() and not d.is_symlink()] if base.is_dir() else []
    if not trees:
        return []
    links, blind = _links(proc)
    now = time.time() if now is None else now
    gone = []
    for d in trees:
        top = str(d.resolve())
        if any(t == top or t.startswith(top + os.sep) for t in links):
            continue
        pid = d.name[len(SCRATCH):].split("-", 1)[0]
        if pid.isdigit() and "-" in d.name[len(SCRATCH):]:
            if (proc / pid).exists():
                continue
        elif blind or now - d.stat().st_mtime < SCRATCH_GRACE_S:
            continue
        shutil.rmtree(d, ignore_errors=True)
        gone.append(d.name)
    if gone:
        log(f"lean-audit: removed {len(gone)} scratch tree(s) no audit holds: {', '.join(sorted(gone))}")
    return gone


#: a dependency fetch that failed for a reason outside the audit (GitHub throttles anonymous clones from a busy address)
TRANSIENT = ("could not read Username for 'https://github.com'", "Could not resolve host", "expected flush after ref listing",
             "Failed to connect to", "The requested URL returned error: 429", "The requested URL returned error: 5")
RETRIES = (90, 240, 480)


def transient(rep: dict) -> bool:
    """A package that failed only in setup or build, and only on fetching a dependency: no verdict, so it is audited again."""
    fs = rep.get("failures", [])
    return bool(fs) and all(f.startswith(("setup:", "build:")) and any(t in f for t in TRANSIENT) for f in fs)


def _audit(missed: list[str], files: list[str], keys: dict, reports: dict, out: Path, cache: Path, deps: Path, fresh: bool,
           log, urls: dict[str, str] | None = None, comps: dict | None = None,
           regenerate: bool = False, bundle: bool = False) -> tuple[list[str], list[str]]:
    """Audit ``missed`` on a scratch tree of their inputs (the packages they require by path included), and store each pass.
    ``regenerate`` (`check.py --regenerate`, a train's check) passes a package whose every failure is a record difference;
    such a pass is never stored, since its lock changes when the train commits the record.  ``bundle`` (a recorded check)
    writes the module versions new in the artifact cache to ``out`` for other nodes (`LakeArtifacts.bundle`).  Returns (the
    failed, the audited)."""
    deps.mkdir(parents=True, exist_ok=True)
    # beside the dependencies: moves, not copies. Resolved, because check.py names each package by its resolved path, and a
    # cache reached through a symlink would otherwise match no report to its package
    sweep(deps.parent, log)
    work = Path(tempfile.mkdtemp(prefix=f"{SCRATCH}{os.getpid()}-", dir=deps.parent)).resolve()
    warm = WarmDeps(deps, urls)
    builds = WarmBuilds(deps.parent / "lean-builds")
    artifacts = LakeArtifacts(deps.parent / "lean-artifacts")
    passed: set[Path] = set()
    audited: list[str] = []
    failed: list[str] = []
    try:
        union = sorted({f for p in missed for f in inputs(p, files)})
        scratch_tree(work, union)
        audited = sorted(missed)
        with ThreadPoolExecutor(max_workers=max(1, len(audited))) as ex:
            list(ex.map(lambda p: warm.take(work / p, log), audited))
        if not fresh:
            for p in sorted({*audited, *(d for q in audited for d in path_deps(q))}):
                builds.take(work, p, p in audited, log)
            try:
                artifacts.take(urls or {}, log)
            except OSError as e:
                log(f"lean-audit: no artifact bundle taken: {e}")
        env = {**os.environ, "LEAN_NUM_THREADS": str(COMMON.cores())}
        t0 = time.monotonic()
        (out / "audit.json").unlink(missing_ok=True)
        r = subprocess.run([sys.executable, str(work / "tools" / "verity" / "lean" / "check.py"), *FLAGS, *(str(work / p) for p in audited),
                            "--replay-cache", str((work if fresh else cache.parent) / "lean-replay"),
                            *([] if fresh else ["--artifacts", str(artifacts.store)]),
                            *(["--regenerate"] if regenerate else []), "--repo", str(ROOT), "--out", str(out)], cwd=work, env=env)
        log(f"lean-audit: {', '.join(audited)} audited in {time.monotonic() - t0:.0f} s (exit {r.returncode})")
        doc = json.loads((out / "audit.json").read_text()) if (out / "audit.json").exists() else {"failed": ["check.py"]}
        failed = [os.path.relpath(f, work) if f.startswith(str(work)) else f for f in doc.get("failed", [])]
        for rep in doc.get("packages", []):
            p = os.path.relpath(rep["package"], work)
            rep["package"] = p
            reports[p] = rep
            if not rep["failures"] and not rep.get("settled") and r.returncode in (0, 1) and p in keys and not fresh:
                cache.mkdir(parents=True, exist_ok=True)
                tmp = cache / f".{keys[p]}.tmp"
                tmp.write_text(json.dumps({"schema": SCHEMA, "package": p, "key": keys[p], "report": rep,
                                           "components": (comps or {}).get(p)}))
                os.replace(tmp, cache / f"{keys[p]}.json")
        if r.returncode not in (0, 1) and not failed:
            failed = [f"check.py exit {r.returncode}"]
        # a pass vouches for its dependencies; so does a failure only in fetching them (Lake clones every dependency, each at
        # the revision the manifest pins, before it compiles anything): keeping those, a retry fetches only what is missing
        passed = {work / p for p in audited if p in reports and (not reports[p].get("failures") or transient(reports[p]))}
    finally:
        warm.give_back(passed)
        if not fresh:
            kept = {work / p for p in audited if p in reports and not reports[p].get("failures")}
            try:
                log(f"lean-audit: {artifacts.put(kept)} module version(s) new in the artifact cache")
                if bundle and artifacts.added:
                    dest = next(out / f"lean-artifacts{'' if i == 1 else f'-{i}'}.tar.gz" for i in range(1, 99)
                                if not (out / f"lean-artifacts{'' if i == 1 else f'-{i}'}.tar.gz").exists())
                    log(f"lean-audit: {artifacts.bundle(dest)} of them bundled for other nodes ({dest.name})")
            except OSError as e:
                log(f"lean-audit: the artifact cache was not written: {e}")
            builds.give_back(work, kept)
            if n := prune_replays(cache.parent / "lean-replay"):
                log(f"lean-audit: {n} old replay entries removed")
        shutil.rmtree(work, ignore_errors=True)
    return failed, audited


def select(pkgs: list[str], only: list[str] | None) -> list[str]:
    """``pkgs`` less those ``only`` (`--package`, a shard's share) leaves out; ValueError naming one ``only`` has and ``pkgs``
    lacks."""
    if not only:
        return pkgs
    if unknown := sorted(set(only) - set(pkgs)):
        raise ValueError(f"no audited package {', '.join(unknown)} on this tree")
    return [p for p in pkgs if p in only]


def run(out: Path, cache: Path, deps: Path, fresh: bool = False, log=print, urls: dict[str, str] | None = None,
        export: Path | None = None, only: list[str] | None = None, regenerate: bool = False, bundle: bool = False) -> int:
    files = COMMON.tracked_files(ROOT)
    if files is None:
        log("lean-audit: the tree's files cannot be listed; nothing is cached")
        fresh = True
        files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".lake" not in p.parts]
    try:
        pkgs = select(packages(files), only)
    except ValueError as e:
        log(f"lean-audit: FAIL: {e}")
        return 1
    comps = {p: components(p, files, FLAGS) for p in pkgs}
    keys = {p: key_of(c) for p, c in comps.items()}
    out.mkdir(parents=True, exist_ok=True)
    reports, cached = {}, {}
    for p, k in keys.items():
        entry = cache / f"{k}.json"
        if not fresh and entry.is_file():
            reports[p] = {**json.loads(entry.read_text())["report"], "package": p}      # it may have passed elsewhere, before a move
            cached[p] = k
            log(f"lean-audit: {p}: PASS reused ({k[:12]}): every file its audit reads is unchanged")
    todo, failed = [p for p in pkgs if p not in cached], []
    for attempt in range(len(RETRIES) + 1):
        if not todo:
            break
        now, audited = _audit(todo, files, keys, reports, out, cache, deps, fresh, log, urls, comps, regenerate, bundle)
        failed = [f for f in failed if f not in audited] + now
        retry = [p for p in audited if p in now and transient(reports.get(p, {}))]
        if not retry or attempt == len(RETRIES):
            break
        log(f"lean-audit: {', '.join(retry)}: a dependency fetch failed, so no verdict; auditing again in {RETRIES[attempt]} s")
        time.sleep(RETRIES[attempt])
        todo = retry
    failed += [p for p in pkgs if p not in reports and p not in failed]
    regenerated = {p: reports[p]["regenerated"] for p in pkgs if reports.get(p, {}).get("settled")}
    for p, f in regenerated.items():
        log(f"lean-audit: {p}: passed on its regenerated record ({f}), which the train commits over {p}/lean-audit.json")
    for rep in reports.values():                        # a pass cached before check.py reported codes has none to list
        rep.setdefault("failure_codes", [None] * len(rep.get("failures", [])))
    (out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "cached": cached,
                                                "packages": [reports[p] for p in pkgs if p in reports],
                                                "regenerated": regenerated}, indent=1) + "\n")
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
        r = load_remote()
    except ConfigError:
        r = None
    if not hasattr(r, "presign"):
        return {}
    return {**{k: r.presign(object_key(v["sha256"]), ttl_s) for k, v in sorted(pinned.items())},
            **{SHARED + h: r.presign(object_key(h), ttl_s) for h in shared_bundles()}}


def shared_bundles(store=None, limit: int = SHARED_LIMIT) -> list[str]:
    """The sha256 of each artifact bundle (`LakeArtifacts.bundle`) among the newest recorded checks' run files, newest first, at
    most ``limit``: only a recorded run's (`common.RECORDED_CHECKS`), as `gather_verdicts` takes only theirs, a failed or
    cancelled run's too, since its bundle holds only modules of packages whose audit passed. Empty when the store can't be
    read: each node then reuses only what it built."""
    from research.store.local import LocalStore

    out: list[str] = []
    try:
        store = store or LocalStore()
        if store.remote is not None:
            store.refresh(attempts_only=True)
        rows = store.index.sql(f"SELECT id FROM attempts WHERE {COMMON.RECORDED_CHECKS} AND state IN ('done', 'failed', 'cancelled') "
                               "ORDER BY start_utc DESC LIMIT ?", (4 * limit,))
        for (rid,) in rows:
            art = ((store.get_attempt(rid) or {}).get("outputs") or {}).get("run_files")
            for f in store.listing(art) if art else []:
                if re.fullmatch(r"lean-audit/lean-artifacts(-\d+)?\.tar\.gz", f["path"]) and f["remote"] and f["sha256"] not in out:
                    out.append(f["sha256"])
            if len(out) >= limit:
                break
    except Exception as e:  # noqa: BLE001  (no bundle: each node reuses only what it built)
        print(f"lean-audit: no artifact bundles listed from the store ({type(e).__name__}: {e})", file=sys.stderr)
    return out[:limit]


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
    # the audit's own copy of the pin file is a `lean-deps.json` too; the export's listing is the one beside its bundles
    beside = {q.rpartition("/")[0] for q in files if q.endswith(".tar.gz")}
    listings = [p for p in files if p.rpartition("/")[2] == "lean-deps.json" and p.rpartition("/")[0] in beside]
    if len(listings) != 1:
        print(f"lean_audit: {run_id} " + (f"has {len(listings)} bundle listings ({', '.join(sorted(listings))}); not pinned"
                                          if listings else "exported no bundle (`--export`)"), file=sys.stderr)
        return 1
    listing = listings[0]
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


def restore_into(pkg: str, log=print) -> int:
    """`--restore PKG` (`tools/verity/lean/setup.sh`): PKG's pinned bundle from the store as its `.lake/packages`, when it has none.
    0 also when there is nothing to restore (no pin for its manifest and toolchain, or no store remote here): `lake` then
    fetches them itself."""
    from research.store.local import object_key
    from research.store.remote import ConfigError, load_remote

    d = (ROOT / pkg).resolve()
    if (k := deps_key(d)) is None or (d / ".lake" / "packages").exists():
        return 0
    if (p := pins().get(k)) is None:
        log(f"lean-audit: {pkg}: no bundle is pinned for its manifest and toolchain ({k[:12]}); lake fetches them")
        return 0
    try:
        r = load_remote()
    except ConfigError:
        r = None
    if not hasattr(r, "presign"):
        log(f"lean-audit: {pkg}: no store remote here to fetch its pinned bundle from; lake fetches them")
        return 0
    (d / ".lake").mkdir(exist_ok=True)
    store = Path(tempfile.mkdtemp(prefix="deps-", dir=d / ".lake"))
    try:
        if not restore(store, k, r.presign(object_key(p["sha256"]), 3600), p, pkg, log):
            return 1
        os.replace(store / k, d / ".lake" / "packages")
    finally:
        shutil.rmtree(store, ignore_errors=True)
    return 0


def main(argv: list[str]) -> int:
    # a manual stop or a runner's timeout sends SIGTERM to check.py's whole process group: as SIGINT, it runs `_audit`'s finally,
    # which removes the scratch tree (about 60 GB on a node, left behind by every stopped check otherwise)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if argv[:1] == ["--pin"] and len(argv) == 2:
        return pin(argv[1])
    if argv[:1] == ["--restore"] and len(argv) == 2:
        return restore_into(argv[1])
    if argv[:1] == ["--prune"]:
        root = Path(argv[argv.index("--cache") + 1]).parent if "--cache" in argv else COMMON.cache_dir()
        got = {"lean-deps": WarmDeps(root / "lean-deps").evict(), "lean-builds": WarmBuilds(root / "lean-builds").evict(),
               "lean-replay": prune_replays(root / "lean-replay"), "lean-records": prune_records(root / "lean-records"),
               "lean-artifacts": LakeArtifacts(root / "lean-artifacts").prune()}
        print("lean-audit --prune: removed " + ", ".join(f"{v} from {k}" for k, v in got.items()) + f" (under {root})")
        return 0
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
    ap.add_argument("--package", action="append", metavar="PKG", help="audit only this package (repeatable): a shard's share")
    ap.add_argument("--regenerate", action="store_true", help="a train's check: `check.py --regenerate`, so a package whose "
                    "every failure is a record difference passes and audit.json's `regenerated` names its record; not in the "
                    "cache key, and such a pass is never stored")
    ap.add_argument("--bundle", action="store_true", help="a recorded check's audit: write the module versions it added to the "
                    "artifact cache as lean-artifacts.tar.gz in --out, for other nodes to take (`shared_bundles`)")
    a = ap.parse_args(argv)
    urls = json.loads(a.urls.read_text()) if a.urls and a.urls.is_file() else None
    return run(a.out, a.cache, a.deps, a.fresh, log=lambda s: print(s, flush=True), urls=urls, export=a.export, only=a.package,
               regenerate=a.regenerate, bundle=a.bundle)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
