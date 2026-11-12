"""`lean_audit`: Lean's audit job, which `check` runs as its lean-audit step. `tools/verity/lean/check.py --build --closure` on the
packages whose passes can't be reused, with dependencies kept warm.

    python tools/verity/lean/lean_audit.py --out DIR [--cache DIR] [--deps DIR] [--fresh] [--urls FILE] [--export DIR] [--package PKG]...
                                                      (only those packages: a shard's share, `check.py --shard`)
                                                      [--regenerate] (the Lean check of `next`: record differences pass,
                                                      and audit.json's `regenerated` names the records drive commits,
                                                      `rekeyed` the passes kept for that commit)
                                                      [--bookkeeping warn] (with --regenerate, drive's lock run: the
                                                      bookkeeping repaired and warned, `repaired` the caps drive commits)
                                                      [--bundle] (a recorded check: the module versions it built, for other nodes)
    python tools/verity/lean/lean_audit.py --pin RUN        pin the bundles run RUN exported (`lean-deps.json`)
    python tools/verity/lean/lean_audit.py --restore PKG    PKG's pinned bundle as its `.lake/packages`, when it has none (`setup.sh`)
    python tools/verity/lean/lean_audit.py --explain PKG    why PKG's audit would run: its inputs against the nearest pass kept here
    python tools/verity/lean/lean_audit.py --prune          evict now what the caps keep (`DEPS_KEPT`, `BUILDS_KEPT` and `BUILDS_MAX_GB`, `REPLAYS_KEPT`, `RECORDS_KEPT` or one per `build` slot, artifacts, `UNREAD_KEPT`)

A package's verdict is reused while everything its audit reads is unchanged, wherever the package is. Its key hashes content,
each file under its path relative to its package (a module's is its name), never the package's own path:
- the tracked files of the package, its `lean-toolchain`, `lake-manifest.json` and `lean-audit.json` among them, less the
  Lake packages nested in it (`audit.lean_modules` skips them) and the files `UNREAD_NAMES` names, which no audit reads:
  its `*.md`, `.gitignore`, `pyproject.toml`, `test_*.py` and `conftest.py`, so a README or a pytest file changed inside
  a package audits nothing again. The scratch tree leaves them out too, so a read of one would fail;
- each package its `proved_in` names (`provers`), whose build the audit reads its guarantees from, by its content (which
  names the packages it requires by theirs), under its path;
- the same for each package it requires by path (`audit.path_deps`), under the name its manifest gives it. Where the
  lakefile's `require ... path = "P"` and the manifest's `"dir": "P"` name such a package, the key reads P as that
  package's own content key (`content_key`), so moving it and rewriting the paths to it keeps every key. A path spelled
  otherwise stays a path in the key;
- `tools/verity/lean/` (the auditor and this file) and `tools/verity/check/common.py`, the helpers it imports, at their paths in the
  repository: not `check.py`, which only launches it, so an edit to `check.py` alone keeps every package's verdict. Nor
  `tools/verity/lean/tests/` (the controls, which the `tools/verity/lean` suite runs), its `*.md`, or the files `NOT_READ` lists
  (the merge driver, the Lean fast path, the daily publish, ...), which the audit never reads: the scratch tree leaves them
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

A lock run (`--regenerate`) passes a package on its regenerated lock, the lock of another tree: the commit drive makes of
the regenerated locks, where that package, each package requiring it by path and the spec it proves have other keys, so
no pass of the run answers there. After a whole lock run passes, each such package it audited is judged on that tree
from the facts document its own audit stored (`keep_regenerated`: `check.py --facts-from` with every regenerated lock in
place, every check that reads the live locks and no build), and where that leaves nothing its pass is stored under the
new key (audit.json's `rekeyed`): the audit of the lock commit reuses it and builds nothing. A pass is kept only where
each regenerated lock the package's audit reads agrees with the lock it replaces on all the build, the seal, `Facts.lean`,
the replay and the tests take from a lock (`check.py`'s `build_inputs`), so those steps would run on the same inputs
there. That holds for a build and tests that read their lock only through the audit, as every package's do; one that
read its own `lean-audit.json` would be a read its pass does not cover, which the nightly uncached check shows as a
disagreement, as it does a read outside a key.

A package's fetched dependencies (`.lake/packages`: Mathlib's cache, ArkLib's build) are kept on this machine under
`lake-manifest.json` plus `lean-toolchain`. They're moved into the scratch package's own `.lake` before its build (the only place
its sandbox lets it write, never a shared symlink) and back afterwards only when that package's audit passed; after a failure
they're dropped, so a run's compile-time code cannot leave the next run a cache it wrote. Lake rebuilds nothing it built in an
earlier passing run. The package's own `.lake/build` is kept the same way (`WarmBuilds`), under its toolchain, manifest and
lakefile less the paths to what it requires (`build_key`), not under its path: Lake decides what is up to date by content
hashes, not paths or mtimes, so a moved package rebuilds nothing. On a node whose builds aren't sandboxed (`sandboxed`) a
run takes no kept build, no artifact bundle and no artifact cache, and keeps nothing it built: every package builds cold.
The audit's `dependencies` check still compares their `.olean` files with the package's record, and the kernel replay
still covers the package's own modules and their whole import
closure, the toolchain's, Mathlib's and ArkLib's modules included (`check.py --closure`): each module whose `.olean` files, and
those of every module it imports, transitively, passed an earlier replay here or in a recorded check whose verdict pack this
one imported (`check.verdicts_fetched`) is reused (`check.py --replay-cache`, `lean-replay/` beside the passes, its
`REPLAYS_KEPT` last used kept), so a change to a few modules replays those and what imports them, and a dependency's
modules go through the kernel once per change of their `.olean` files. With `--fresh` the replay cache is
the scratch tree's, so the packages of one run share it and nothing outlives the run. Lean gets this
machine's CPUs (`LEAN_NUM_THREADS`): inside a pod, its default counts the host's.

The audits of `security` and `security/Proofs` (`UNREAD_FACTS`) also write what each constant of the analysis mentions
(`check.py`'s `unread_facts`), and their passes are stored, and reused, only with those facts beside them (`unread/` in
the cache, `UNREAD_KEPT` last used kept). After the audits, the run joins both packages' facts, this run's or the reused
passes', and runs the ratchet on wholly unread modules (`unread_ratchet`, `unread.py`): an area of the analysis whose
wholly unread modules, or their declarations, rose above its cap in `unread-caps.json` fails the run (`unread` in
`audit.json`'s `failed`), and one that came down leaves the lowered caps in `unread/` under ``--out``. It builds nothing
and takes seconds. `unread.py` and the caps are out of every key (`NOT_READ`), so a change to either reruns no audit. The
seeds it reads outside Lean (code that names a declaration or a module, the documents a lock names) are in no key either,
so each run leaves the facts it read under ``--out`` as `unread/K.jsonl.gz`, by the package's key (`facts_file`), which a
recorded check publishes: drive's landing gate (`unread_gate.py`) runs the ratchet from them on a change to those seeds
before it lands.

A machine without a package's dependencies restores them from the evidence store instead of cloning them from GitHub, which
throttles anonymous clones from shared addresses. `lean-deps.json` pins one bundle per `lake-manifest.json`'s fetched entries + `lean-toolchain`
(`deps_key`; a `path` entry changes no key) by sha256; the launcher sends a presigned URL for each (`--urls`, `check.py --record`), and a bundle is used only when its
bytes are the pinned ones. A bundle leaves out the builds Mathlib's cache supplies: `lake exe cache get` takes those from
Mathlib's CDN, which is not GitHub. After a bump, a passing cold audit run with `--export` writes the new bundles into its run
directory, custody puts them on R2, and `--pin RUN` pins them. The rest of a fetched package is built only as far as the tree
imported it when it was exported: each pin lists what its bundle holds built (`modules`), and an import beyond that needs a
re-pin, since the Lean check's module runner mounts the bundle read-only and can't build it
(`tests/test_lean_packages.py`).
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import fnmatch
import functools
import gzip
import hashlib
import http.client
import importlib.util
import itertools
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


#: every file this step loads is in `inputs`: `tools/verity/lean/` (this module among them), `common.py` and the task graph's
#: contract, which `module_keys` loads (`test_lean_audit.py` holds it)
COMMON = _module(ROOT / "tools" / "verity" / "check" / "common.py", "check_common_for_lean_audit")
MK = _module(Path(__file__).resolve().parent / "module_keys.py", "lean_module_keys_for_lean_audit")
SETTINGS = _module(Path(__file__).resolve().parent / "settings.py", "lean_settings_for_lean_audit")
#: the task graph's contract, which `module_keys` keys each module with (`module_keys.CONTRACT`)
CONTRACT = "tools/verity/graph/verity_graph/contract.py"


@functools.cache
def lean_check():
    """`check.py` beside this file, loaded once: the auditor, for the functions of it this job calls itself."""
    return _module(Path(__file__).resolve().parent / "check.py", "lean_check_for_lean_audit")


def sandboxed() -> bool:
    """Whether check.py builds in its sandbox on this node, by its own probe (`check.sandboxed`). Outside it a build's
    compile-time code runs as the user and may leave any output, and the sampled recompile, which binds a restored module to
    its source, doesn't run, so nothing a build restores would be checked."""
    return lean_check().sandboxed()


def runs_code(pkg: Path) -> str | None:
    """The first source that runs code while it compiles, in ``pkg`` or a package it requires by path, or None: as
    check.py scans for it (`check.COMPILE_TIME`, comments aside), outside a `compile_time` listing of its package's lock
    reviewed at the file's digest (`check.listings`); or a `lakefile.lean`, which runs whenever Lake loads its package. A
    lock or a source that can't be read counts as one.

    Those packages are the ones whose `.lake` a build of ``pkg`` writes (`check.writable`), in the sandbox too, and code
    a module runs while it compiles can write there what it likes: another module's outputs, their `.hash` files and
    its trace, which nothing then tells from Lake's. A passing audit says there is no such code in the package it
    audits (`check.scans`); the artifact cache takes builds no audit passed, so `LakeArtifacts.put` asks it of every
    build, the sources being the ones the build read: the sandbox lets no build write them."""
    a = lean_check()
    for d in [pkg.resolve(), *a.path_deps(pkg)]:
        try:
            if (d / "lakefile.lean").exists():
                return f"{d.name}/lakefile.lean runs code whenever Lake loads the package"
            listed = (a.load(d).get("compile_time") or {}) if (d / a.POLICY).is_file() else {}
            for m, rel in sorted(a.lean_modules(d).items()):
                if isinstance(e := listed.get(m), dict) and e.get("digest") == a.sha256(d / rel):
                    continue
                text = a.strip_comments((d / rel).read_text(errors="replace"))
                if hit := a.COMPILE_TIME.search(text):
                    return (f"{d.name}/{rel}:{text.count(chr(10), 0, hit.start()) + 1} runs code while it compiles "
                            f"({hit.group(1) or hit.group(2)}), outside a `compile_time` listing reviewed at its digest")
        except (OSError, ValueError, AttributeError, TypeError) as e:
            return f"{d.name}: its sources or its lock could not be read ({type(e).__name__})"
    return None


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def packages(files: list[str]) -> list[str]:
    """The audited packages: directories whose tracked files include a `lean-audit.json`, the auditor's fixtures aside."""
    return sorted({f.rsplit("/", 1)[0] for f in files if f.endswith("/lean-audit.json") and not f.startswith(LEAN_TOOL)})


#: the files of a package no audit reads, by name, each with what does: out of the package's key and of the scratch tree,
#: where a read of one fails the audit. Nothing goes by directory: a generator, its helpers, data and fixtures stay in
#: wherever they sit, `tests/` included (a `runs` entry may run a generator kept there)
UNREAD_NAMES = {
    "*.md": "the package's README and other prose, for readers",
    ".gitignore": "git, which never checks out what it lists",
    "pyproject.toml": "the settings of the package's Python suite, which its suite reads",
    "test_*.py": "the package's pytest files, which its suite runs",
    "conftest.py": "pytest's fixtures for those files",
}


def own_files(pkg: str, files: list[str]) -> list[str]:
    """The package's tracked files, less those of the Lake packages nested in it and those `UNREAD_NAMES` names."""
    nested = {f.rsplit("/", 1)[0] + "/" for f in files if f.startswith(pkg + "/") and f.rsplit("/", 1)[-1] in ("lakefile.toml", "lakefile.lean")
              and f.rsplit("/", 1)[0] != pkg}
    return [f for f in files if f.startswith(pkg + "/") and not any(f.startswith(n) for n in nested)
            and not any(fnmatch.fnmatchcase(f.rsplit("/", 1)[-1], p) for p in UNREAD_NAMES)]


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
    "axiom_probe.py": "the axiom probe of `verity lean try`, which `lean_changed.py` and the CLI load",
    "lean-deps.json": "the dependency bundles' pins, which preflight reads",
    "lean_changed.py": "the Lean fast path, which runs beside the audit",
    "merge.py": "git's merge driver for lean-audit.json",
    "modules.py": "the codemod onto Lean's module system, which the ratchet and the merge driver load",
    "public_lock.py": "a lock as the public export ships it, which `tools/verity/check/public_export.py` loads",
    "run_daily.py": "the build cache's daily publish",
    "siblings.py": "the sibling check of `verity lean try --siblings`, which `lean_changed.py` and the CLI load",
    "tool.py": "the store's declarations of the scheduled runs",
    "trusted_size.py": "the size of a package's trusted text, a command of its own on a records run's facts document",
    "Upstream.lean": "`upstream.py --lean`, the exact pass; the audit runs the no-build one",
    "unread.py": "the ratchet on wholly unread modules, which `run` runs after the audits on the facts they wrote",
    "unread-caps.json": "the ratchet's caps, which `unread.py` reads",
    "unread_gate.py": "drive's landing gate on the seeds the ratchet reads, which runs `main`'s copy of this tool",
    "warm.py": "the audit of a ready mark's merge, which starts `lean_audit.py` for the packages a change reaches",
    "pyproject.toml": "the tools/verity/lean suite's settings",
}

#: the packages whose audits write `Unread.lean`'s facts (`check.py`'s `UNREAD_FACTS`): a pass of one is reused only with
#: the facts it wrote, kept beside it, and the ratchet on wholly unread modules joins theirs (`unread_ratchet`)
UNREAD_FACTS = ("security", "security/Proofs")
UNREAD = "tools/verity/lean/unread.py"
UNREAD_SCHEMA = "lean-unread-facts/v1"
#: the facts entries a node keeps beside the passes (`unread/` in the cache), last stored or reused first: security/Proofs'
#: are some 6 MB each, and a pass whose facts went is audited again
UNREAD_KEPT = 64
#: the caps `--bookkeeping warn` (drive's lock run after a landing) holds to what the run measures, each
#: repaired in a copy under the run's `repaired/` for drive to commit beside the regenerated locks: the ratchet's, and the
#: repository suite's caps on what the locks list (`tests/test_lean_packages.py`'s `_held`), each {cap: a lock's count}
UNREAD_CAPS = "tools/verity/lean/unread-caps.json"
CAPS_TEST = "tests/test_lean_packages.py"
LOCK_CAPS = {"UNWITNESSED": lambda lock: len(lock.get("unwitnessed") or {}),
             "INLINE": lambda lock: sum(map(len, ((lock.get("shape") or {}).get("inline") or {}).values()))}
#: who a repaired cap's warning names
CAPS_OWNER = "@lean"


def raised_unread(unread: dict, u) -> tuple[str, list[str]]:
    """`unread-caps.json` with each area the ratchet ``unread`` (`unread.check`'s result) failed raised to its count, and the
    rest as they were: its text, and a warning naming each raised area."""
    caps, warned = {}, []
    for area, v in sorted(unread["areas"].items()):
        count, cap = v["count"], v.get("cap")
        if cap is None or any(count[k] > cap.get(k, 0) for k in count):
            if count.get("modules"):
                caps[area] = count
                warned.append(f"bookkeeping: unread area {area} in {UNREAD_CAPS} ({CAPS_OWNER}): its cap raised to the "
                              f"{count['modules']} wholly unread modules ({count['declarations']} declarations) it has")
        elif cap.get("modules"):
            caps[area] = cap
    return u.caps_text(caps), warned


def keep_repaired(out: Path, path: str, text: str) -> str:
    """Write ``text``, the repaired copy of the repository's ``path``, under ``out``'s `repaired/`: its path there."""
    dest = out / "repaired" / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    return str(dest)


def held_caps(text: str, locks: dict[str, dict]) -> tuple[str, list[str]]:
    """``text`` (`CAPS_TEST`) with each cap of `LOCK_CAPS` that ``locks`` ({package: its lock}) break (`_held`: a count above
    its cap, raised to the count; a cap above its count by more than its `ROOM`, lowered to the count and its room, which
    changes queued on `next` still use), and a warning naming each. LookupError when a cap's line is not one dict literal
    there."""
    def literal(name: str) -> tuple[re.Match, dict]:
        if not (m := re.search(rf"^{name} = (\{{.*\}})$", text, re.M)):
            raise LookupError(f"{CAPS_TEST} has no line `{name} = {{...}}` for --bookkeeping warn to hold to the locks")
        return m, json.loads(m.group(1))
    room, warned = literal("ROOM")[1], []
    for name, count in LOCK_CAPS.items():
        m, caps = literal(name)
        got = {p: n for p, lock in sorted(locks.items()) if (n := count(lock)) or p in caps}
        new = dict(caps)
        for p, n in got.items():
            held = n if n > caps.get(p, 0) else n + room.get(f"{name} {p}", 0)
            if held != caps.get(p, 0) and (n > caps.get(p, 0) or caps.get(p, 0) > held):
                new[p] = held
                warned.append(f"bookkeeping: {name}[{p!r}] in {CAPS_TEST} ({CAPS_OWNER}): {caps.get(p, 0)} -> {held}, the "
                              f"count its lock has" + (f" and its room of {held - n}" if held > n else ""))
        if new != caps:
            text = text[:m.start(1)] + json.dumps(new) + text[m.end(1):]
    return text, warned


def reusable(cache: Path, pkg: str, k: str) -> bool:
    """Whether ``pkg``'s pass under ``k`` is kept in ``cache`` with what its reuse needs: the pass, and for a package of
    `UNREAD_FACTS` the facts its audit wrote."""
    return (cache / f"{k}.json").is_file() and (pkg not in UNREAD_FACTS or (cache / "unread" / f"{k}.json").is_file())


def keep_facts(cache: Path, pkg: str, k: str, rep: dict, out: Path) -> bool:
    """Store the facts ``pkg``'s audit wrote (its report's `unread_facts`, under ``out``) beside its pass: `unread/K.json`,
    a `.json` entry so that a verdict pack carries it. False when the audit wrote none."""
    f = rep.get("unread_facts") or {}
    src = out / str(f.get("file", ""))
    if not f or not src.is_file() or _sha(src) != f.get("sha256"):
        return False
    d = cache / "unread"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{k}.tmp"
    tmp.write_text(json.dumps({"schema": UNREAD_SCHEMA, "package": pkg, "key": k, "sha256": f["sha256"],
                               "facts": base64.b64encode(src.read_bytes()).decode()}))
    os.replace(tmp, d / f"{k}.json")
    return True


def kept_facts(cache: Path, pkg: str, k: str, out: Path, log=print) -> Path | None:
    """The facts kept beside ``pkg``'s pass under ``k``, written out to ``out`` (`unread/K.jsonl.gz`, `facts_file`), or None
    when there are none or (logged) they don't read or their bytes aren't the ones the entry's sha256 names."""
    entry = cache / "unread" / f"{k}.json"
    if not entry.is_file():
        return None
    try:
        doc = json.loads(entry.read_text())
        data = base64.b64decode(doc["facts"], validate=True)
    except (OSError, ValueError, KeyError, TypeError) as e:
        log(f"lean-audit: {pkg}: the facts kept under {k[:12]} don't read ({type(e).__name__}: {e})")
        return None
    if doc.get("schema") != UNREAD_SCHEMA or hashlib.sha256(data).hexdigest() != doc.get("sha256"):
        log(f"lean-audit: {pkg}: the facts kept under {k[:12]} aren't {UNREAD_SCHEMA} bytes of the sha256 they name")
        return None
    dst = facts_file(out, k)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    os.utime(entry)
    return dst


def facts_file(out: Path, k: str) -> Path:
    """Where a run leaves the facts the ratchet read of the package whose key is ``k``: `unread/K.jsonl.gz` under ``out``,
    which a recorded check publishes, so that drive's landing gate (`unread_gate.py`) finds them by the key alone."""
    return out / "unread" / f"{k}.jsonl.gz"


def prune_facts(cache: Path, kept: int = UNREAD_KEPT) -> int:
    """Remove all but the ``kept`` facts entries of ``cache`` (its `unread/`) last stored or reused; the number removed."""
    return prune_replays(cache / "unread", kept)


def unread_ratchet(pkgs: list[str], failed: list[str], facts: dict[str, Path], files: list[str], out: Path,
                   log=print) -> dict | None:
    """The ratchet on wholly unread modules (`unread.check`, against `unread-caps.json`) on the facts the audits of
    `UNREAD_FACTS` wrote, joined, with no build: its result, or None where it doesn't run. It runs where both packages are
    audited or reused: a shard's run that has only one leaves it to the shard that audits the spec, which audits its
    provers beside it; and a failed audit of either fails the run already."""
    if missing := [p for p in UNREAD_FACTS if p not in pkgs]:
        log(f"lean-audit: unread: not run here: {', '.join(missing)} is another run's (the ratchet joins the facts of "
            f"{' and '.join(UNREAD_FACTS)})")
        return None
    if bad := [p for p in UNREAD_FACTS if p in failed]:
        log(f"lean-audit: unread: not run: the audit of {', '.join(bad)} failed, so the facts the ratchet reads are not there")
        return None
    if lack := [p for p in UNREAD_FACTS if p not in facts]:
        return {"failures": [f"unread: {', '.join(lack)} passed with no facts for the ratchet (check.py's `unread_facts` "
                             f"writes them in each audit of {' and '.join(UNREAD_FACTS)} that replays)"]}
    try:
        u = _module(ROOT / UNREAD, "lean_unread_for_lean_audit")
        r = u.check(ROOT, files, [facts[p] for p in UNREAD_FACTS], out / "unread")
    except (OSError, EOFError, ValueError, LookupError, TypeError, AttributeError, ImportError, SyntaxError) as e:
        return {"failures": [f"unread: {type(e).__name__}: {e}"]}  # anything else raises, which fails the step too
    for f in r["failures"]:
        log(f"lean-audit: FAIL: {f}")
    if r["lower"]:
        log(f"lean-audit: unread: an area's wholly unread modules came down: {out / 'unread' / 'unread-caps.json'} holds "
            f"the caps at the counts, to copy over {u.CAPS.relative_to(ROOT)} (or `unread.py --caps "
            f"{u.CAPS.relative_to(ROOT)} --update-caps`)")
    if not r["failures"]:
        n, d = (sum(v["count"][k] for v in r["areas"].values()) for k in ("modules", "declarations"))
        log(f"lean-audit: unread: PASS in {r['seconds']} s: {n} wholly unread modules ({d} declarations), no area above "
            f"its cap")
    return r


def tool_files(files: list[str]) -> list[str]:
    """The auditor's files: `tools/verity/lean/` less its tests, its docs and `NOT_READ`, and the helpers it imports."""
    tool = [f for f in files if f.startswith(LEAN_TOOL) and not f.startswith(LEAN_TOOL + "tests/") and not f.endswith(".md")
            and f[len(LEAN_TOOL):] not in NOT_READ]
    return sorted({*tool, "tools/verity/check/common.py", CONTRACT})


def read_packages(pkg: str) -> list[str]:
    """The packages whose files an audit of ``pkg`` can read: itself, those it requires by path, and each package its
    `proved_in` names with those that one requires."""
    return list(dict.fromkeys([pkg, *path_deps(pkg), *(d for q in provers(pkg) for d in (q, *path_deps(q)))]))


def inputs(pkg: str, files: list[str]) -> list[str]:
    """Every tracked file an audit of ``pkg`` can read."""
    own = [f for p in read_packages(pkg) for f in own_files(p, files)]
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


def content(pkg: str, files: list[str], _memo: dict | None = None, locks: dict[str, bytes] | None = None) -> dict[str, str]:
    """{path relative to the package: sha256} of its own files, so that the package moved to another directory with the same
    content has the same map. Its lakefile and manifest name each package it requires by path by that package's own content
    (`content_key`), not by where it is: moving the required package and rewriting the paths to it keeps the map too. A package
    that may read beyond its directory by a relative path keeps its path in the map (`WHERE`): a `..` left in a file Lake or
    the audit read paths from, or a module's `include_str`. ``locks`` ({a lock's path in the repository: its bytes}) stand
    for those files' bytes here: the map of the tree with them written (`keep_regenerated`)."""
    memo = {} if _memo is None else _memo
    spelled = _required(pkg, files)
    out, escapes = {}, False
    for f in own_files(pkg, files):
        rel, p = f[len(pkg) + 1:], ROOT / f
        if locks and f in locks:
            text = locks[f]
        elif not p.is_file():
            out[rel] = "-"
            continue
        else:
            text = p.read_bytes()
        if rel in ("lakefile.toml", "lake-manifest.json") and spelled:
            text, out[rel] = _spelled_as(text, "path" if rel == "lakefile.toml" else "dir", spelled,
                                         lambda d: content_key(d, files, memo, locks))
        else:
            out[rel] = hashlib.sha256(text).hexdigest()
        escapes |= (b".." in text) if rel in CONFIG else rel.endswith(".lean") and b"include_str" in text
    if escapes:
        out[WHERE] = pkg
    return out


def content_key(pkg: str, files: list[str], _memo: dict | None = None, locks: dict[str, bytes] | None = None) -> str:
    memo = {} if _memo is None else _memo
    if pkg not in memo:
        memo[pkg] = hashlib.sha256(json.dumps(sorted(content(pkg, files, memo, locks).items())).encode()).hexdigest()
    return memo[pkg]


def _lake_name(pkg: str, dep: str) -> str:
    """The name ``pkg``'s manifest gives the package it requires at ``dep`` (its path when it has none)."""
    m = ROOT / pkg / "lake-manifest.json"
    for d in json.loads(m.read_text()).get("packages", []) if m.is_file() else ():
        if d.get("type") == "path" and os.path.relpath(os.path.normpath(ROOT / pkg / d.get("dir", "")), ROOT) == dep:
            return str(d.get("name", dep))
    return dep


def components(pkg: str, files: list[str], flags: list[str], locks: dict[str, bytes] | None = None) -> dict:
    """What a package's key hashes, none of it a path of the package: the schema, the audit's flags, the package's `content`,
    that of each package it requires by path (under the name its manifest gives it), that of each package its `proved_in`
    names (under its path, which the policy already holds), and each of the auditor's files at its
    path in the repository. A pass keeps them beside it, so `--explain` can say what a miss changed. With ``locks``
    (`content`), those of the tree with those locks written: the commit drive makes of a lock run's regenerated records."""
    memo: dict = {}
    return {"schema": SCHEMA, "flags": flags, "files": content(pkg, files, memo, locks),
            "requires": {_lake_name(pkg, d): content(d, files, memo, locks) for d in path_deps(pkg)},
            "proved_in": {q: content(q, files, memo, locks) for q in provers(pkg)},
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


def modules(src: Path) -> dict[str, dict[str, list[str]]]:
    """What a bundle of ``src`` (a `.lake/packages`) holds of each fetched package whose build Mathlib's cache doesn't supply:
    {package: {"roots": the first components of its module names, "built": each module with an `.olean` in its build}}. A
    root is one its built modules have or a `.lean` file at the top of its checkout, so a package nothing was built of yet
    still claims its imports."""
    out = {}
    for n in sorted(set(os.listdir(src)) - cache_covered(src)):
        if not (src / n).is_dir():
            continue
        lib = src / n / ".lake" / "build" / "lib" / "lean"
        built = sorted(".".join(f.relative_to(lib).with_suffix("").parts) for f in lib.rglob("*.olean")) if lib.is_dir() else []
        tops = {f.stem for f in (src / n).glob("*.lean") if f.stem != "lakefile"}
        out[n] = {"roots": sorted(tops | {m.split(".")[0] for m in built}), "built": built}
    return out


def restore(store: Path, k: str, url: str, pin: dict, name: str, log=print) -> bool:
    """Download the bundle pinned under ``k`` and unpack it as ``store/k``. False, and the package builds cold, when it can't
    be fetched or isn't the pinned bundle. The URL is never logged: it grants a read until it expires."""
    part, tmp = store / f".{k}.part", store / f".{k}.tmp"
    t0 = time.monotonic()
    try:
        h, n = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:  # noqa: S310  (the store's presigned URL)
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
                done[k] = {"package": p, **bundle(self.store / k, dest / f"{k}.tar.gz"), "modules": modules(self.store / k)}
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
    (Lake's default `.` for each), as `move_check`'s `source_dirs` reads them. `security/Proofs` has `srcDir = ".."`, so its sources
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
#: (dozens of them fill a 200 GB disk). Each package's newest stays past it (`WarmBuilds.evict`): a cap below the ~30
#: packages would have every check evict some and rebuild them cold
BUILDS_KEPT = 48
#: the GB the kept builds may take on a node before its least recently used go, whatever their number: a security/Proofs
#: build is about 17 GB, so `BUILDS_KEPT` of them alone would fill a node's root disk. Each package's newest stays past it too
BUILDS_MAX_GB = SETTINGS.BUILDS_MAX_GB
#: the kept dependency trees (one per manifest and toolchain, `deps_key`) a node holds before its least recently used go:
#: security/Proofs' is about 35k files, so two dozen of them take some 800k inodes
DEPS_KEPT = 8
#: the replay cache's entries a node keeps, last used first (a replay touches each entry it reuses): one per module and key,
#: about 3k a version of the tree plus some 8k of Mathlib's and ArkLib's that every replay reuses, each a few hundred bytes;
#: uncapped, a node gathers some 180k
REPLAYS_KEPT = 60_000


class WarmBuilds:
    """Each package's own `.lake/build`, kept under its `build_key` by the last run whose audit of it passed, so an auditor
    change, or a change to one module, doesn't rebuild every module. An audited package's is moved in and back only when its
    audit passed: its compile-time code may write in it. A package in the tree only as a path dependency gets a copy, which
    never comes back; a module of it that the artifact cache lacks is compiled by the first build that reaches it, and
    indexed with that build's (`module_keys.settle`). The audit still checks every statement against these `.olean`
    files and replays every declaration through the kernel, and compiles a random sample of the modules its build kept
    again (`check.sampled_recompile`). The newest `kept` by use stay (`evict`)."""

    def __init__(self, store: Path, kept: int = BUILDS_KEPT, max_gb: float | None = BUILDS_MAX_GB):
        self.store, self.kept, self.max_gb = store, kept, max_gb

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
        """Remove all but the `kept` builds last given back or copied from, and then the least recently used until they take at
        most `max_gb`, never a package's newest (`evict_lru`); the number removed."""
        max_bytes = None if self.max_gb is None else int(self.max_gb * 1e9)
        return evict_lru(self.store, self.kept, keep=self.newest_per_package(), max_bytes=max_bytes)


def tree_bytes(path: Path) -> int:
    """The bytes ``path``'s files take on disk (their allocated blocks, as `du` counts them), not following links: one `lstat`
    per entry, some seconds for a build of tens of thousands of files."""
    total = 0
    for root, dirs, files in os.walk(path):
        for name in (*dirs, *files):
            try:
                total += os.lstat(os.path.join(root, name)).st_blocks * 512
            except OSError:                            # removed while walked
                pass
    return total


def evict_lru(store: Path, kept: int, keep: set[str] = frozenset(), max_bytes: int | None = None) -> int:
    """Remove all but the ``kept`` directories of ``store`` last used (a rename keeps a directory's mtime, so each use sets
    it), and with ``max_bytes`` then the least recently used until the rest take at most that (`tree_bytes`); never one named
    in ``keep``, and each only when its lock (`NAME.lock` beside it) is free: a run holding it is taking it or giving it back.
    The number removed."""
    held = []
    for home in store.iterdir() if store.is_dir() else ():
        try:
            if home.is_dir():
                held.append((home.stat().st_mtime, home))
        except OSError:                                # taken while listed
            pass
    def remove(home: Path) -> bool:
        with open(store / f"{home.name}.lock", "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            if not home.is_dir():
                return False
            shutil.rmtree(home, ignore_errors=True)
            (store / f"{home.name}.pkg").unlink(missing_ok=True)
            return True

    newest = sorted(held, reverse=True)
    n = 0
    for _, home in newest[kept:]:
        if home.name not in keep and remove(home):
            n += 1
    if max_bytes is not None:
        rest = [(home, tree_bytes(home)) for _, home in newest[:kept] if home.is_dir()]
        total = sum(b for _, b in rest)
        for home, b in reversed(rest):                 # the least recently used first
            if total <= max_bytes:
                break
            if home.name not in keep and remove(home):
                total -= b
                n += 1
    return n


#: the fast path's (`lean_changed.py --records`) trees a node keeps, its most recently used: each is a checkout with its own
#: builds, about 40 GB and 490k inodes (nine of them take nearly all of a node's inodes). A node with more
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
    slot's tree is the one a run there takes, and removing it hands that run a cold one (the hourly prune would take
    trees seeded for a node's `build` slots). The slots' own trees (`build-I`, I below ``slots``)
    are kept first, then the rest by last use, the newest of their `.who`, `links.json` and `.lock`. A link to a tree
    elsewhere goes with its target, and `passes/` is never touched. The number removed."""
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


#: the disk the artifact cache may hold beyond what kept builds share with it (hard links), before its oldest go: more
#: than the largest package's build, with room for the versions before it, since what a tree held is the cache's alone
#: once that tree is swept (a stopped audit's), and `prune` then cuts to three quarters of this
ARTIFACTS_CAP = 64 << 30
ARTIFACT = MK.ARTIFACT
#: a bundle's members: the index's entries and the artifacts they name, nothing else
MEMBER = re.compile(r"artifacts/[0-9a-f]{16}\.[a-z.]+|" + MK.INDEX + r"/[0-9a-f]{128}\.json")
#: what one bundle (`LakeArtifacts.bundle`) holds at most, uncompressed: a run's files over 4 GiB upload without resume. A
#: run that added more writes as many bundles as that takes (`LakeArtifacts.bundle_into`)
BUNDLE_CAP = 3 << 30
#: a key of `lean-deps-urls.json` naming a bundle of one of `PUBLISHERS`' runs by its sha256 (`shared_bundles`)
SHARED = "lean-artifacts/"
#: a bundle's name in the directory its audit writes it in (`LakeArtifacts.bundle_into`)
BUNDLE_NAME = r"lean-artifacts(-\d+)?\.tar\.gz"
#: a recorded check's or a records run's file that is a bundle, in the run's `lean-audit/`
BUNDLE_FILE = re.compile("lean-audit/" + BUNDLE_NAME)
#: the store's attempts of records runs (`lean_changed.py --records`), by the command `research run` recorded
RECORDS_RUNS = "manifest_json LIKE '%tools/verity/lean/lean_changed.py%--records%'"
#: the runs `shared_bundles` lists bundles of, each kind as (the store's attempts of it, a bundle's path among its files,
#: which of the attempt's outputs list its files): a recorded check lists its own among its run's files; the audit of a
#: ready mark's merge (`warm.py`) leaves its in the `--out` its caller gave it, which a caller that doesn't list its files
#: still has in its run record; a records run writes its where a recorded check does (`lean_changed.audit_records`)
PUBLISHERS = ((COMMON.RECORDED_CHECKS, BUNDLE_FILE, ("run_files",)),
              (COMMON.MARK_AUDITS, re.compile(re.escape(COMMON.MARK_AUDIT_OUT) + BUNDLE_NAME), ("run_files", "run_record")),
              (RECORDS_RUNS, BUNDLE_FILE, ("run_files",)))
#: how many of the newest runs of each kind the launcher sends the bundles of, every bundle of each: a candidate's shards end
#: by the dozen, and counted with them a mark's bundles would be gone before the candidate that holds its change is built.
#: The marks of the PRs the launcher's tree holds are sent beside these, however old (`shared_bundles`' ``heads``). Runs
#: are counted and not bundles, or the count would cut a run's later bundles off its first
SHARED_LIMIT = 12
#: how many bundles `LakeArtifacts.take` downloads at once, ahead of the one it unpacks
AHEAD = 4


class Unindexable(str):
    """Why a module a build compiled whole can't be indexed (`module_keys.indexable`): a reason its next compile gives again."""


class LakeArtifacts:
    """The node's Lake artifact cache: Lake's artifacts (`artifacts/<hash>.<ext>`) and an index of the modules they are the
    outputs of, each under its key (`keys/<key>.json`, `module_keys`): the module, its package, Lake's input hash, the
    outputs its trace names and the digest of each, and what its key read (`module_keys.record`), against which a later
    build says why it didn't restore the module (`modules/<package>/<module>.json` names the entry a module was last
    indexed under). An entry of another schema (`module_keys.SCHEMA`) is under a key no build computes: it is never read,
    never taken from a bundle, and goes with the least recently used (`prune`). `unstored/<key>.json` records a module
    a build compiled under that key and `put` could not index, which the next build leaves to Lake
    (`module_keys.settle`); it stays on its node. The builds of check's audit and of a
    records run read it through a Lake cache directory of their own, which `module_keys.stage` fills with the modules
    whose keys it indexes (`check.py --artifacts`), and never write it; after every build of check's audit, of a records run, of `--try` and of `--build`, whatever
    became of the run, each module the build compiled in the sandbox goes in (`put`), unless a source of the build runs
    code while it compiles (`runs_code`). A module is stored for what its compile read, and no verdict of a run is read
    from here, so a failed run's compile is worth what a passing run's is. A
    module version any earlier run on this node compiled is then restored whatever tree comes next, not only the one
    its package's kept build (`WarmBuilds`) came from. The audit still checks every statement against the restored `.olean` files and replays
    every declaration through the kernel, and compiles a random sample of the modules it restored again, failing on any
    output that differs (`check.sampled_recompile`), which is what binds a restored `.olean` to its source.

    Between nodes, the module versions a recorded check's audit added travel as bundles in its run's files (`bundle_into`), as
    do those the audit of a ready mark's merge added (`warm.py`) and those a records run added (`lean_changed.py
    --records`); the launcher of a check, and `research run` for a records run, sends a URL for each of the newest of each
    and for those of the marks of the PRs its tree holds (`shared_bundles`), and a node takes in each it hasn't yet (`take`).
    A bundle is trusted as far as its source, as a verdict pack is (`verdicts.py`): a recorded run."""

    def __init__(self, store: Path, cap: int = ARTIFACTS_CAP):
        self.store, self.cap = store, cap
        #: (entry, its artifacts) of each module version this object indexed
        self.added: list[tuple[str, list[str]]] = []

    def _lock(self):
        self.store.mkdir(parents=True, exist_ok=True)
        f = open(self.store / MK.LOCK, "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    def put(self, reports: dict[Path, dict], since: float | None = None, log=print) -> int:
        """Index each module of the builds of ``reports``' packages (each with what check.py said of its build: its report,
        or `builds`' record of a run that didn't end) that the build compiled (a trace written after ``since``, and not
        one Lake restored) under its key; the number of entries new to the index. Whether the run passed is not read: a
        key names what a compile read, and a failed or killed run's compile of those is the compile a passing run's is,
        given the two things a passing audit said of the build, which are asked here of every build instead.
        Of a build that names the modules Lake said it built (`"compiled"`: one that read Lake's own cache, where an
        archive of main's published build unpacks to outputs, fresh `.hash` files and the publisher's trace, which nothing
        in the build tells from a compile), only those go in.
        A build that says where it read its tree (`"bound"`: the fixed path and the tree, check.py's `bound`) has the
        paths Lake wrote under the first read as the second's (`module_keys.indexable`); the log says how many of its
        `setup.json` name a path under the tree's own instead, each read as it is.
        Only a build that says it ran in the sandbox (`"sandbox": true`) goes in: outside it, compile-time
        code runs as the user and could leave any outputs, traces and `setup.json` it chose in the build, each then indexed
        as its module's. And only a build none of whose sources runs code while it compiles, outside a reviewed listing
        (`runs_code`): in the sandbox such code can still write the build's own `.lake`, every other module's outputs
        there included. A module goes in only when `module_keys.indexable`, when its trace is whole and each output it
        names is in the build with that hash (`<file>.hash`) and no newer than the file that says so (`_built`: Lake
        writes the outputs, then their hashes, then the trace, so a build that was stopped leaves no module that passes),
        and once every workspace module it loads is indexed, under the key the
        outputs the build holds of those give it (`module_keys.read_digests`); its artifacts before
        the entry naming them, one the cache holds already kept only with the same bytes. A newer build replaces the entry
        under the same key, so the entries of a module and its imports stay one build's. A module the run compiled whole
        and that still can't go in (not `indexable`, a module it loads not indexed, an artifact's name taken) is recorded
        under its key (`module_keys.UNSTORED`) with why, whatever loads it, so the next build leaves it to Lake rather than
        compile it again for nothing (`module_keys.settle`) and its stage step gives that reason, and so is each module of
        a build refused for the code its sources run. The log says how many compiled modules `indexable` refused and
        names the first few with its words. Only what
        the compiled modules load is read, so a build that compiled nothing costs its traces' `stat`s."""
        with self._lock():
            n = 0
            for pkg in sorted(reports):
                if not isinstance(rep := reports[pkg], dict) or rep.get("sandbox") is not True:
                    log(f"lean-audit: {pkg}: its report doesn't say it was built in the sandbox, so none of its modules go "
                        "into the artifact cache")
                    continue
                if any(str(f).startswith("sampled:") for f in rep.get("failures") or ()):
                    log(f"lean-audit: {pkg}: a module its build didn't compile differs from its recompile (`sampled`), so "
                        "none of the modules compiled on top of it go into the artifact cache")
                    continue
                ws = MK.workspace(pkg)
                if isinstance(ws, str):
                    continue
                keyed, owners, own = MK.keys(pkg, MK.policy_roots(pkg)), {}, set()
                at = tuple(b) if isinstance(b := rep.get("bound"), list) and len(b) == 2 and all(
                    isinstance(s, str) for s in b) else None
                with ThreadPoolExecutor(max_workers=8) as ex:
                    built = dict(zip(keyed, ex.map(lambda m: self._built(m, keyed, ws[0], since, owners, at, own), keyed)))
                if at:
                    log(f"lean-audit: {pkg}: {len(own)} setup.json of this build, which read its tree at {at[0]}, name a "
                        f"path under the tree's own ({at[1]}), each read as it is")
                if isinstance(only := rep.get("compiled"), list):
                    built = {m: b if isinstance(b, str) and not isinstance(b, Unindexable) or m in only
                             else "not compiled by this run: Lake didn't say it built it" for m, b in built.items()}
                refusals = {m: b for m, b in built.items() if isinstance(b, Unindexable)}
                need, todo = set(), [m for m, b in built.items() if not isinstance(b, str) or m in refusals]
                while todo:
                    if (m := todo.pop()) not in need:
                        need.add(m)
                        todo += MK.loads(m, keyed)
                if refused := runs_code(pkg) if need else None:
                    log(f"lean-audit: {pkg}: {refused}. Such code can write any module's outputs in the build, so none of "
                        "its modules go into the artifact cache")
                outs = MK.read_digests({m: k for m, k in keyed.items() if m in need},
                                       {(m, a.split(".", 1)[1]): d for m, b in built.items() if not isinstance(b, str)
                                        for a, (_, d) in b[2].items()})
                indexed, recorded = set(), 0
                for mod, k in keyed.items():
                    if mod not in need:
                        continue
                    loaded = MK.loads(mod, keyed)
                    key = MK.key(mod, keyed, outs, loaded)
                    got = self._index(mod, key, k.package.name, built[mod], MK.record(mod, keyed, outs, loaded)) \
                        if refused is None and key is not None and all(i in indexed for i in loaded) else None
                    if got is not None:
                        indexed.add(mod)
                        n += got
                        MK.unstored_path(self.store, key).unlink(missing_ok=True)
                    elif key is not None and (not isinstance(b := built[mod], str) or isinstance(b, Unindexable)):
                        self._unstored(mod, key, k.package.name, refused or (b if isinstance(b, str) else next(
                            (f"it loads {i}, which is not indexed" for i in loaded if i not in indexed),
                            "the cache holds an artifact of its name with other bytes")))
                        recorded += mod in refusals
                if refusals:
                    log(f"lean-audit: {pkg}: {len(refusals)} modules this run compiled can't be indexed (`indexable`), "
                        f"{recorded} recorded under their keys: "
                        + "; ".join(f"{m}: {why}" for m, why in sorted(refusals.items())[:3]))
            self.prune()
        return n

    @staticmethod
    def _built(mod: str, keyed: dict, pkgs: list, since: float | None, owners: dict | None = None,
               at: tuple[str, str] | None = None,
               own: set[str] | None = None) -> tuple[str, dict, dict[str, tuple[Path, str]]] | str:
        """(Lake's input hash, the outputs its trace names, {artifact: (its file, its digest)}) of ``mod`` as its build
        compiled it, or why it isn't indexed: not compiled since ``since``, restored, not `indexable` (``owners``; ``at``:
        the report's `bound`, the fixed path the build read its tree at and the tree; ``own``, which gets ``mod`` when its
        `setup.json` names the tree's own path), or
        not whole. Lake removes a module's outputs, runs `lean`, hashes each output into its `.hash`, and only then writes
        the trace naming those hashes (`Module.buildLean`, `buildAction`), and it leaves the previous trace and `.hash`
        files in place until then. So a compile that was stopped leaves a trace that isn't JSON, an output that is
        missing, or one newer than the `.hash` beside it, which names what was there before: none is indexed."""
        build = keyed[mod].package.build()
        trace = build / "lib" / "lean" / MK.path_of(mod, ".trace")
        try:
            st = trace.stat()
        except OSError:
            return "no trace"
        if since is not None and st.st_mtime <= since:
            return "not compiled by this run"
        try:
            d = json.loads(trace.read_text())
        except (OSError, ValueError) as e:
            return f"a trace that isn't JSON: {e}"
        if not isinstance(d, dict) or d.get("synthetic"):
            return "restored, or a trace of another shape"
        out, h = d.get("outputs"), d.get("depHash")
        if not isinstance(out, dict) or not isinstance(h, str) or not MK.HASH.fullmatch(h):
            return "a trace without its outputs or input hash"
        if why := MK.indexable(mod, keyed, pkgs, owners, at, own):
            return Unindexable(why)
        files = {}
        for name in sorted({s for s in _strings(out) if ARTIFACT.fullmatch(s)}):
            f = MK.output(build, mod, name)
            hashed = f.parent / f"{f.name}.hash"
            if not f.is_file() or not hashed.is_file() or hashed.read_text().strip() != name.split(".", 1)[0]:
                return f"{f.name} isn't in the build under {name}"
            if f.stat().st_mtime_ns > hashed.stat().st_mtime_ns:
                return f"{f.name} was written after {hashed.name}, which names what was there before"
            files[name] = (f, MK.digest(f))
        return (h, out, files) if files else "a trace that names no artifact"

    def _index(self, mod: str, key: str, scope: str, built, inputs: dict | None = None) -> bool | None:
        """Index ``mod`` of package ``scope`` under ``key`` from ``built`` (`_built`), with what the key read (``inputs``,
        `module_keys.record`): True for a new entry, False for one kept or replaced, None when it has none. The entry a
        compiled module is indexed under becomes the one its next miss is explained against (`module_keys.last`)."""
        at = MK.entry_path(self.store, key)
        old = MK.entry(self.store, key, mod, scope)
        if isinstance(built, str):
            return None if old is None else False
        h, out, files = built
        shas = {n: s for n, (_, s) in files.items()}
        if old is not None and old["hash"] == h and old["digests"] == shas:
            os.utime(at)
            self._last(mod, key, scope)
            return False
        arts = self.store / "artifacts"
        arts.mkdir(exist_ok=True)
        for name, (f, sha) in files.items():
            dst = arts / name
            if dst.exists():
                if not os.path.samefile(dst, f) and MK.digest(dst) != sha:
                    return None if old is None else False
                continue
            tmp = arts / f".{name}.{os.getpid()}.tmp"
            try:
                os.link(f, tmp)
            except OSError:
                shutil.copyfile(f, tmp)
            os.chmod(tmp, 0o444)
            os.replace(tmp, dst)
        at.parent.mkdir(exist_ok=True)
        tmp = at.parent / f".{at.stem}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"schema": MK.SCHEMA, "module": mod, "scope": scope, "hash": h, "outputs": out,
                                   "digests": shas, "inputs": inputs}, sort_keys=True))
        os.replace(tmp, at)
        self._last(mod, key, scope)
        self.added.append((at.relative_to(self.store).as_posix(), sorted(files)))
        return old is None

    def _unstored(self, mod: str, key: str, scope: str, why: str) -> None:
        f = MK.unstored_path(self.store, key)
        f.parent.mkdir(exist_ok=True)
        tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"module": mod, "scope": scope, "why": why, "by": MK.rules()}, sort_keys=True))
        os.replace(tmp, f)

    def _last(self, mod: str, key: str, scope: str) -> None:
        if (f := MK.last_path(self.store, scope, mod)) is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"key": key}))
            os.replace(tmp, f)

    def bundle(self, dest: Path, cap: int = BUNDLE_CAP, start: int = 0) -> tuple[int, int]:
        """Write the entries `added` from its index ``start`` on as a gzipped tar at ``dest``, each with every artifact it
        names, the entries before the artifacts, up to ``cap`` bytes uncompressed: (the number of entries written (none: no
        file), the index in `added` of the first entry this bundle had no room for, where the next one starts). A bundle
        is whole in itself: an artifact that entries of two bundles name is in both. An entry that is gone from the
        cache, or that with its artifacts is more than a bundle holds, is in no bundle, and the entries after it still are."""
        with self._lock():
            todo, size, names, stop = [], 0, set(), len(self.added)
            for i in range(start, len(self.added)):
                entry, outs = self.added[i]
                try:
                    more = sum((self.store / "artifacts" / n).stat().st_size for n in outs if n not in names)
                    more += (self.store / entry).stat().st_size
                except OSError:
                    continue
                if size + more > cap:
                    if todo:
                        stop = i
                        break
                    continue
                todo.append((entry, outs))
                names.update(outs)
                size += more
            if not todo:
                return 0, stop
            tmp = dest.with_name(f".{dest.name}.tmp")
            with tarfile.open(tmp, "w:gz", compresslevel=1) as t:
                for entry, _ in todo:
                    t.add(self.store / entry, arcname=entry, recursive=False)
                for n in sorted(names):
                    t.add(self.store / "artifacts" / n, arcname=f"artifacts/{n}", recursive=False)
            os.replace(tmp, dest)
        return len(todo), stop

    def bundle_into(self, out: Path, cap: int = BUNDLE_CAP) -> tuple[list[Path], int]:
        """Every entry `added` in bundles of at most ``cap`` bytes each (`bundle`), as many as that takes, at the free
        names `shared_bundles` lists in ``out`` in their order (`lean-artifacts.tar.gz`, then `-2`, ...), the entries
        added first in the first: (those paths, the number of entries written)."""
        free = (d for i in itertools.count(1)
                if not (d := out / f"lean-artifacts{'' if i == 1 else f'-{i}'}.tar.gz").exists())
        dests, n, at = [], 0, 0
        while at < len(self.added):
            wrote, at = self.bundle(dest := next(free), cap, at)
            if wrote:
                dests.append(dest)
                n += wrote
        return dests, n

    def take(self, urls: dict[str, str], log=print, budget: int | None = None) -> int:
        """Take in each bundle ``urls`` names (`SHARED` + its sha256) that this node hasn't: downloaded, checked against its
        sha256, and unpacked member by member (`_unpack`). The number of entries added. A bundle that fails is skipped,
        and its modules are built. One whose bytes are the ones named and can't be unpacked is marked taken, since they
        never change. One whose download gave other bytes is not: a response that ends early, with an error or without
        one, gives those too, so a later run fetches it again, for as long as the launcher lists it (`shared_bundles`).
        The URL is never logged: it grants a read until it expires.

        The bundles are unpacked one at a time in the order ``urls`` names them, which is the launcher's, newest first
        (`shared_bundles`), until the bytes written to the cache reach ``budget``: a quarter of the cap unless given. A
        bundle holds up to `BUNDLE_CAP` and the launcher lists dozens, more than the cache's cap together; a quarter is
        what the prune that follows can make room for (it stops at three quarters, least recently used first) and still
        leave half the cap to what the node itself used last. The bundle the budget stops in, and those after it, are not
        marked: a later run takes what is left of them, when the launcher still lists them.

        Downloads run ahead of the unpacking, `AHEAD` at once and no further than that past the bundle being unpacked,
        so dozens of small bundles cost a quarter of what one at a time would (about 3 s a bundle from the store), and a
        run that stops at its budget has fetched at most `AHEAD` - 1 bundles it then drops.

        A bundle is marked taken under this schema alone (`module_keys.SCHEMA`): one made under another adds nothing
        here, and a build of a tree with that schema on this node takes it for itself."""
        want = {k[len(SHARED):]: u for k, u in urls.items() if k.startswith(SHARED)}
        taken = self.store / f"taken-{MK.SCHEMA.rsplit('/', 1)[-1]}"
        want = {h: u for h, u in want.items() if re.fullmatch(r"[0-9a-f]{64}", h) and not (taken / h).exists()}
        if not want:
            return 0
        taken.mkdir(parents=True, exist_ok=True)
        budget = self.cap // 4 if budget is None else budget
        t0, n, size, got, stopped = time.monotonic(), 0, 0, 0, False

        def get(h: str, url: str) -> Path | None:
            part = self.store / f".{h}.{os.getpid()}.part"
            try:
                d = hashlib.sha256()
                with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:  # noqa: S310  (a presigned URL)
                    for chunk in iter(lambda: r.read(1 << 20), b""):
                        d.update(chunk)
                        f.write(chunk)
                if d.hexdigest() == h:
                    return part
                log(f"lean-audit: artifact bundle {h[:12]}: what was fetched is not the one named (other bytes, or a download "
                    f"that ended early); skipped, and a later run fetches it again")
            except (OSError, http.client.HTTPException) as e:
                log(f"lean-audit: artifact bundle {h[:12]} could not be fetched ({getattr(e, 'code', None) or type(e).__name__})")
            part.unlink(missing_ok=True)
            return None

        todo, ahead = list(want.items()), {}
        with ThreadPoolExecutor(max_workers=AHEAD) as ex:
            for i, (h, _) in enumerate(todo):
                if stopped := stopped or size >= budget:
                    break
                for j in range(i, min(i + AHEAD, len(todo))):
                    if j not in ahead:
                        ahead[j] = ex.submit(get, *todo[j])
                if (part := ahead.pop(i).result()) is None:
                    continue
                got += 1
                with self._lock():
                    try:
                        more, wrote, whole = self._unpack(part, budget - size)
                        n, size, stopped = n + more, size + wrote, not whole
                        if whole:
                            (taken / h).touch()
                    except (tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile) as e:
                        log(f"lean-audit: artifact bundle {h[:12]} could not be unpacked ({type(e).__name__}); skipped")
                        (taken / h).touch()
                    except OSError as e:
                        log(f"lean-audit: artifact bundle {h[:12]} could not be unpacked ({type(e).__name__}); skipped")
                    finally:
                        part.unlink(missing_ok=True)
            for f in ahead.values():
                if (part := f.result()) is not None:
                    part.unlink(missing_ok=True)
        with self._lock():
            self.prune()
        log(f"lean-audit: {n} module version(s), {size / 2**20:.0f} MiB, taken from {got} of {len(want)} artifact bundle(s) in "
            f"{time.monotonic() - t0:.0f} s" + (f"; the rest is left to a later run: one run takes {budget / 2**20:.0f} MiB "
                                                 f"at most" if stopped else ""))
        return n

    def _unpack(self, part: Path, budget: int) -> tuple[int, int, bool]:
        """Take in the entries of the bundle at ``part`` (`MEMBER`, of the shape `module_keys.valid` reads, so of this
        schema: a bundle made under another adds nothing) under keys this cache lacks, and the artifacts they name that it
        lacks, each only with a digest an entry names for it; an entry once every artifact it names is here. A file here
        is never replaced. It stops before the artifact that would take what it wrote past ``budget`` bytes. (The number
        of entries added, the bytes written, whether it read the bundle to its end.)"""
        entries: dict[str, dict] = {}
        want: dict[str, set[str]] = {}
        wrote, whole = 0, True
        with tarfile.open(part, "r:gz") as t:
            for m in t:
                if not m.isfile() or not MEMBER.fullmatch(m.name):
                    continue
                dst = self.store / m.name
                if m.name.startswith(MK.INDEX + "/"):
                    try:
                        e = json.loads(t.extractfile(m).read())
                    except ValueError:
                        continue
                    if not dst.exists() and MK.valid(e):
                        entries[m.name] = e
                        for a, d in e["digests"].items():
                            want.setdefault(a, set()).add(d)
                    continue
                if dst.exists() or m.name.split("/", 1)[1] not in want:
                    continue
                if wrote + m.size > budget:
                    whole = False
                    break
                wrote += m.size
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
                with t.extractfile(m) as src, open(tmp, "wb") as f:
                    shutil.copyfileobj(src, f, 1 << 20)
                if MK.digest(tmp) not in want[m.name.split("/", 1)[1]]:
                    tmp.unlink()
                    continue
                os.chmod(tmp, 0o444)
                os.replace(tmp, dst)
        n = 0
        for name, e in entries.items():
            dst = self.store / name
            if dst.exists() or not all((self.store / "artifacts" / a).exists() for a in e["digests"]):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(e, sort_keys=True))
            os.replace(tmp, dst)
            n += 1
        return n, wrote, whole

    def prune(self) -> int:
        """Remove the least recently used artifacts while those only the cache holds exceed the cap, and the entries used no
        later than them; the number of artifacts removed. An entry whose artifact is gone is a miss, so the module is built.

        An artifact was last used when an entry that names it was (a build touches each entry it stages, keeps or
        indexes), or when it was written if that is later. Its own time is never moved: every tree that restored it holds
        the same file by a hard link, beside a `.hash` written at the restore, and an output newer than its `.hash` is how
        a half-written module is told (`lean_changed.damaged`, `_built`)."""
        found = [(f, s) for f in (self.store / "artifacts").glob("*") if (s := f.stat())]
        if sum(s.st_size for _, s in found if s.st_nlink == 1) <= self.cap:
            return 0
        used: dict[str, float] = {}
        for e in (self.store / MK.INDEX).glob("*.json"):
            try:
                at = e.stat().st_mtime
                doc = json.loads(e.read_text())
            except (FileNotFoundError, ValueError):     # gone since it was listed (`module_keys.entry` removes one under
                e.unlink(missing_ok=True)               # the tree's lock, not this one), or damaged on disk, as it takes it
                continue
            names = doc.get("digests") if isinstance(doc, dict) else None
            for a in names if isinstance(names, dict) else ():
                used[a] = max(used.get(a, 0.0), at)
        arts = [(max(s.st_mtime, used.get(f.name, 0.0)), s.st_size if s.st_nlink == 1 else 0, f) for f, s in found]
        own = sum(size for _, size, _ in arts)
        n, horizon = 0, 0.0
        for mtime, size, f in sorted(arts):
            if own <= self.cap * 3 // 4:
                break
            f.unlink(missing_ok=True)
            own, n, horizon = own - size, n + 1, mtime
        for e in [*(self.store / MK.INDEX).glob("*.json"), *(self.store / "outputs").glob("*/*.json"),
                  *(self.store / MK.UNSTORED).glob("*.json")]:
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


#: what check.py writes in a package's work directory (`audit-<tag>/` under its `--out`) once it knows whether the package's
#: builds run in the sandbox, before the first of them (check.py's `BUILDING`)
BUILDING = "building.json"


def built(out: Path) -> dict[Path, dict]:
    """{package: what check.py said of its build} for each package the check.py with its reports under ``out`` built, or
    began to build: its report in `audit.json`, which says `sandbox` of a package it built, else what it wrote as the build
    began (`BUILDING`: the package, and whether its builds run in the sandbox), which is all a check.py that was killed
    leaves, and which is the later of the two when an earlier check.py left its report there. `LakeArtifacts.put` takes
    either. Both are check.py's own words: ``out`` is outside every directory a build in the sandbox can write."""
    found: dict[Path, tuple[int, dict]] = {}

    def said(f: Path) -> tuple[int, dict]:
        try:
            at, d = f.stat().st_mtime_ns, json.loads(f.read_text())
        except (OSError, ValueError):
            return 0, {}
        return (at, d) if isinstance(d, dict) else (0, {})

    for at, d in map(said, sorted(out.glob(f"audit-*/{BUILDING}"))):
        if isinstance(d.get("package"), str):
            found[Path(d["package"])] = (at, d)
    at, doc = said(out / "audit.json")
    reps = doc.get("packages")
    for r in reps if isinstance(reps, list) else ():
        if isinstance(r, dict) and isinstance(r.get("package"), str) and "sandbox" in r \
                and at >= found.get(pkg := Path(r["package"]), (0, {}))[0]:
            found[pkg] = (at, r)
    return {pkg: d for pkg, (_, d) in found.items()}


def transient(rep: dict) -> bool:
    """A package that failed only in setup or build, and only on fetching a dependency: no verdict, so it is audited again."""
    fs = rep.get("failures", [])
    return bool(fs) and all(f.startswith(("setup:", "build:")) and any(t in f for t in TRANSIENT) for f in fs)


def keep_pass(cache: Path, pkg: str, k: str, rep: dict, comps: dict | None, out: Path, log=print) -> bool:
    """Store ``pkg``'s pass, its report ``rep``, in ``cache`` under its key ``k``, with what the key hashes (``comps``)
    beside it; for a package of `UNREAD_FACTS`, first the facts its audit wrote under ``out`` (`keep_facts`), without which
    (logged) no pass is stored.  Whether it is stored."""
    if pkg in UNREAD_FACTS and not keep_facts(cache, pkg, k, rep, out):
        log(f"lean-audit: {pkg}: its pass is not stored: its audit wrote no facts for the ratchet (`unread_facts`)")
        return False
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f".{k}.tmp"
    tmp.write_text(json.dumps({"schema": SCHEMA, "package": pkg, "key": k, "report": rep, "components": comps}))
    os.replace(tmp, cache / f"{k}.json")
    return True


def _audit(missed: list[str], files: list[str], keys: dict, reports: dict, out: Path, cache: Path, deps: Path, fresh: bool,
           log, urls: dict[str, str] | None = None, comps: dict | None = None,
           regenerate: bool = False, bundle: bool = False, bookkeeping: bool = False,
           documents: Path | None = None) -> tuple[list[str], list[str]]:
    """Audit ``missed`` on a scratch tree of their inputs (the packages they require by path included), and store each pass.
    ``regenerate`` (`check.py --regenerate`, the Lean check of `next`) passes a package whose every failure is a record
    difference; such a pass is never stored here, since its lock changes when drive commits the record (`keep_regenerated`
    stores the pass of the tree with that lock, from the facts document this audit leaves in ``documents``, `check.py
    --facts-cache`).  What each package's build compiled in the sandbox goes into the artifact cache whether the package
    passed, failed or check.py was killed (`built`, `LakeArtifacts.put`), and ``bundle`` (a recorded check) writes the
    module versions it indexed to ``out`` for other nodes (`LakeArtifacts.bundle`).  Where builds aren't sandboxed
    (`sandboxed`) none of that happens and no kept build or artifact cache is read: every package builds cold.  Returns
    (the failed, the audited)."""
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
    since = time.time()
    cold = not fresh and not sandboxed()
    try:
        union = sorted({f for p in missed for f in inputs(p, files)})
        scratch_tree(work, union)
        audited = sorted(missed)
        with ThreadPoolExecutor(max_workers=max(1, len(audited))) as ex:
            list(ex.map(lambda p: warm.take(work / p, log), audited))
        if cold:
            log("lean-audit: this node builds without the sandbox, so no kept build, artifact bundle or artifact cache is "
                "used and none is kept: every package builds cold")
        elif not fresh:
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
                            *([] if fresh or cold else ["--artifacts", str(artifacts.store)]),
                            *(["--regenerate"] if regenerate else []), *(["--bookkeeping", "warn"] if bookkeeping else []),
                            *(["--facts-cache", str(documents)] if documents else []),
                            "--repo", str(ROOT), "--out", str(out)], cwd=work, env=env)
        log(f"lean-audit: {', '.join(audited)} audited in {time.monotonic() - t0:.0f} s (exit {r.returncode})")
        doc = json.loads((out / "audit.json").read_text()) if (out / "audit.json").exists() else {"failed": ["check.py"]}
        failed = [os.path.relpath(f, work) if f.startswith(str(work)) else f for f in doc.get("failed", [])]
        for rep in doc.get("packages", []):
            p = os.path.relpath(rep["package"], work)
            rep["package"] = p
            rep.pop("facts_document", None)             # a path under ``documents``, gone with this run; `facts_key` names it
            reports[p] = rep
            if not rep["failures"] and not rep.get("settled") and r.returncode in (0, 1) and p in keys and not fresh:
                keep_pass(cache, p, keys[p], rep, (comps or {}).get(p), out, log)
        if r.returncode not in (0, 1) and not failed:
            failed = [f"check.py exit {r.returncode}"]
        # a pass vouches for its dependencies; so does a failure only in fetching them (Lake clones every dependency, each at
        # the revision the manifest pins, before it compiles anything): keeping those, a retry fetches only what is missing
        passed = {work / p for p in audited if p in reports and (not reports[p].get("failures") or transient(reports[p]))}
    finally:
        warm.give_back(passed)
        if not fresh and not cold:
            ok = [p for p in audited if p in reports and not reports[p].get("failures")]
            kept = {work / p for p in ok}
            try:
                mine = {at: rep for at, rep in built(out).items() if work in at.resolve().parents}
                log(f"lean-audit: {artifacts.put(mine, since, log)} module version(s) new in the artifact cache")
                if bundle and artifacts.added:
                    log(f"lean-audit: {bundled(*artifacts.bundle_into(out), len(artifacts.added))}")
            except OSError as e:
                log(f"lean-audit: the artifact cache was not written: {e}")
            builds.give_back(work, kept)
        if not fresh:
            if n := prune_replays(cache.parent / "lean-replay"):
                log(f"lean-audit: {n} old replay entries removed")
            if n := prune_facts(cache):
                log(f"lean-audit: {n} old facts entries removed")
        shutil.rmtree(work, ignore_errors=True)
    return failed, audited


#: under a lock run's ``--out``: the judgments of the tree with its regenerated locks written (`keep_regenerated`), each
#: package's report and log. Not ``--out`` itself, where a second audit of a package removes the lock the first regenerated
LOCK_COMMIT = "lock-commit"
#: what a full audit's report holds of the build, the seal, the kernel replay and the tests, which `check.py --facts-from`
#: runs none of (`audit_from_facts`): a pass `keep_regenerated` keeps has them from the lock run's own audit of the package
BUILT = ("sandbox", "build_threads", "staged", "sampled", "replay", "replay_cache", "pipeline", "proved_in", "unread_facts",
         "peak_gib", "facts_key")


def stored_document(documents: Path, k: str, schema: str) -> tuple[Path | None, str]:
    """(The facts document `check.py --facts-cache` stored in ``documents`` under the facts key ``k``, "") when its sidecar
    names ``schema``, that key and the SHA-256 its bytes have (`check.py`'s `read_cached_facts`, without decoding it); else
    (None, why not)."""
    doc, side = documents / f"{k}.json.gz", documents / f"{k}.json"
    try:
        meta = json.loads(side.read_text())
        named = meta.get("schema") == schema and meta.get("key") == k and _sha(doc) == meta.get("sha256")
    except (OSError, ValueError, AttributeError) as e:
        return None, f"{type(e).__name__}: {e}"
    return (doc, "") if named else (None, f"{side.name} names another schema, key or SHA-256 than {doc.name} has")


def keep_regenerated(pkgs: list[str], files: list[str], keys: dict, reports: dict, regenerated: dict[str, str],
                     audited: set[str], documents: Path, out: Path, cache: Path, log=print) -> dict[str, str]:
    """After a lock run that passed: store the passes of the tree with its regenerated locks written, the commit drive
    makes of them, so that commit's audit reuses them and builds nothing.  {package: its key on that tree} for each kept.

    A lock run's pass on a regenerated lock is of another tree than the one audited, so `_audit` stores none, and the
    regenerated lock moves the key of its package, of each package requiring it by path and of the spec it proves.  Each
    such package is judged here on that tree from the facts document its own audit in this run stored (``documents``,
    `check.py --facts-cache`): `check.py --facts-from` on a scratch tree of its inputs with every regenerated lock in
    place, which runs every check of a full audit that reads the live locks (`audit_from_facts`) and no build.  Its pass
    is stored under the new key only when that leaves nothing (no failure, nothing settled or regenerated again), and its
    report is that judgment's with the lock run's record of the build, the replay and the tests (`BUILT`, `seconds`).

    That is a full audit's verdict of the new tree only if everything before the judge would run the same there, so a
    package keeps no pass unless each regenerated lock its audit reads (`read_packages`) agrees with the lock it replaces
    on `check.py`'s `build_inputs`: a lock that newly lists a premise, a guarantee or an inline name changes what
    `Facts.lean` walks, and that package is audited in full on the new tree, as before.  Nor does a package this run
    reused from the cache (no facts of this run), one whose audit stored no document (a node without the sandbox, a
    tamper), or any package of a run that failed or audited a shard (`run` calls this for neither)."""
    try:
        locks = {f"{p}/lean-audit.json": Path(f).read_bytes() for p, f in regenerated.items()}
    except OSError as e:
        log(f"lean-audit: no pass kept for the commit of the regenerated records: {type(e).__name__}: {e}")
        return {}
    new = {p: components(p, files, FLAGS, locks) for p in pkgs}
    moved = [p for p in pkgs if key_of(new[p]) != keys[p]]
    chk = lean_check()

    def same_build(p: str) -> bool:
        """Whether ``p``'s regenerated lock gives every step before the judge what its lock gives; a lock that can't be
        read, or has no `roots`, gives none of them anything."""
        try:
            return chk.build_inputs(chk.load(ROOT / p)) == chk.build_inputs(chk.load(Path(regenerated[p]).parent))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False

    same = {p: same_build(p) for p in regenerated}
    todo: dict[str, Path] = {}
    for p in moved:
        rep = reports.get(p) or {}
        doc, missing = stored_document(documents, str(rep.get("facts_key")), chk.FACTS_SCHEMA)
        if p not in audited:
            why = "this run reused its pass, so it has no facts of this run to judge from"
        elif rep.get("failures") != []:
            why = "its audit here did not pass"
        elif doc is None:
            why = f"its audit stored no facts document to judge from ({rep.get('facts_cache') or missing})"
        elif differ := [d for d in read_packages(p) if d in regenerated and not same[d]]:
            why = (f"the regenerated lock of {', '.join(differ)} changes what the build, the facts or the tests read "
                   f"(`check.py`'s `build_inputs`: a guarantee, a premise or an inline name it newly lists or drops)")
        else:
            todo[p] = doc
            continue
        log(f"lean-audit: {p}: no pass kept for the commit of the regenerated records, whose audit runs in full: {why}")
    if not todo:
        return {}
    kept: dict[str, str] = {}
    dest = out / LOCK_COMMIT
    dest.mkdir(parents=True, exist_ok=True)
    tree = Path(tempfile.mkdtemp(prefix=f"{SCRATCH}{os.getpid()}-", dir=documents.parent)).resolve()
    try:
        union = sorted({f for p in todo for f in inputs(p, files)})
        scratch_tree(tree, union)
        for f, data in locks.items():
            if f in union:
                (tree / f).write_bytes(data)
        for p, doc in todo.items():
            t0 = time.monotonic()
            (dest / "audit.json").unlink(missing_ok=True)
            # its output goes to a file: an `AUDIT pkg: PASS|FAIL` line in this job's log is the lock run's own verdict
            r = subprocess.run([sys.executable, str(tree / "tools" / "verity" / "lean" / "check.py"), str(tree / p),
                                "--facts-from", str(doc), "--repo", str(ROOT), "--out", str(dest)], cwd=tree,
                               capture_output=True, text=True)
            (dest / f"{re.sub(r'[^A-Za-z0-9]+', '_', p)}.log").write_text(r.stdout + r.stderr)
            took = round(time.monotonic() - t0, 1)
            try:
                (rep,) = json.loads((dest / "audit.json").read_text())["packages"]
            except (OSError, ValueError, KeyError, TypeError):
                rep = {"failures": [f"check.py --facts-from exit {r.returncode} with no report"]}
            if r.returncode or rep.get("failures") != [] or rep.get("settled") or rep.get("regenerated") \
                    or os.path.relpath(str(rep.get("package")), tree) != p:
                log(f"lean-audit: {p}: no pass kept for the commit of the regenerated records, whose audit runs in full: "
                    f"judged on that tree from this run's facts, it fails (exit {r.returncode}, "
                    f"{len(rep.get('failures') or []) + len(rep.get('settled') or [])} failures; {dest})")
                continue
            was, k = reports[p], key_of(new[p])
            rep = rep | {"package": p} | {f: was[f] for f in BUILT if f in was} | {
                "seconds": {**was.get("seconds", {}), "regenerated-lock": took}, "judged_from": keys[p]}
            if keep_pass(cache, p, k, rep, new[p], out, log):
                kept[p] = k
                log(f"lean-audit: {p}: PASS kept for the commit of the regenerated records ({k[:12]}): judged on that tree "
                    f"from this run's facts in {took:.0f} s with no build, nothing left")
    finally:
        shutil.rmtree(tree, ignore_errors=True)
        shutil.rmtree(dest / "tool", ignore_errors=True)
    return kept


def select(pkgs: list[str], only: list[str] | None) -> list[str]:
    """``pkgs`` less those ``only`` (`--package`, a shard's share) leaves out; ValueError naming one ``only`` has and ``pkgs``
    lacks."""
    if not only:
        return pkgs
    if unknown := sorted(set(only) - set(pkgs)):
        raise ValueError(f"no audited package {', '.join(unknown)} on this tree")
    return [p for p in pkgs if p in only]


def run(out: Path, cache: Path, deps: Path, fresh: bool = False, log=print, urls: dict[str, str] | None = None,
        export: Path | None = None, only: list[str] | None = None, regenerate: bool = False, bundle: bool = False,
        bookkeeping: bool = False) -> int:
    """The job (the module's docstring).  With ``regenerate`` and a cache, the audits leave their facts documents in a
    directory of this run's own beside the scratch trees (outside all a build writes, in no verdict pack, swept with them
    if this process is killed), from which `keep_regenerated` judges the tree with the regenerated locks."""
    if not regenerate or fresh:
        return _run(out, cache, deps, fresh, log, urls, export, only, regenerate, bundle, bookkeeping, None)
    deps.mkdir(parents=True, exist_ok=True)
    documents = Path(tempfile.mkdtemp(prefix=f"{SCRATCH}{os.getpid()}-facts-", dir=deps.parent)).resolve()
    try:
        return _run(out, cache, deps, fresh, log, urls, export, only, regenerate, bundle, bookkeeping, documents)
    finally:
        shutil.rmtree(documents, ignore_errors=True)


def _run(out: Path, cache: Path, deps: Path, fresh: bool, log, urls: dict[str, str] | None, export: Path | None,
         only: list[str] | None, regenerate: bool, bundle: bool, bookkeeping: bool, documents: Path | None) -> int:
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
    if bookkeeping and (left := sorted(set(packages(files)) - set(pkgs))):
        log(f"lean-audit: FAIL: --bookkeeping warn holds the caps to every package's lock, and --package leaves out "
            f"{', '.join(left)}: drive's lock run audits every package")
        return 1
    if fresh or len(pkgs) != len(packages(files)):      # nothing is cached; a shard: another's regenerated locks move its keys
        documents = None
    comps = {p: components(p, files, FLAGS) for p in pkgs}
    keys = {p: key_of(c) for p, c in comps.items()}
    out.mkdir(parents=True, exist_ok=True)
    reports, cached, facts = {}, {}, {}
    ran: set[str] = set()
    for p, k in keys.items():
        entry = cache / f"{k}.json"
        if not fresh and entry.is_file():
            if p in UNREAD_FACTS:
                if (f := kept_facts(cache, p, k, out, log)) is None:
                    log(f"lean-audit: {p}: its pass ({k[:12]}) is kept without the facts the ratchet reads: audited again")
                    continue
                facts[p] = f
            reports[p] = {**json.loads(entry.read_text())["report"], "package": p}      # it may have passed elsewhere, before a move
            cached[p] = k
            log(f"lean-audit: {p}: PASS reused ({k[:12]}): every file its audit reads is unchanged")
    todo, failed = [p for p in pkgs if p not in cached], []
    for attempt in range(len(RETRIES) + 1):
        if not todo:
            break
        now, audited = _audit(todo, files, keys, reports, out, cache, deps, fresh, log, urls, comps, regenerate, bundle,
                              bookkeeping, documents)
        ran |= set(audited)
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
        log(f"lean-audit: {p}: passed on its regenerated record ({f}), which the queue commits over {p}/lean-audit.json")
    for rep in reports.values():                        # a pass cached before check.py reported codes has none to list
        rep.setdefault("failure_codes", [None] * len(rep.get("failures", [])))
    for p in UNREAD_FACTS:
        f = (reports.get(p) or {}).get("unread_facts") or {}
        if p not in facts and p not in cached and f.get("file") and (out / f["file"]).is_file():
            facts[p] = facts_file(out, keys[p])
            facts[p].parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / f["file"], facts[p])
    unread = unread_ratchet(pkgs, failed, facts, files, out, log)
    repaired, warned = {}, [f"{p}: {w}" for p in pkgs for w in (reports.get(p) or {}).get("bookkeeping", [])]
    if bookkeeping and unread and unread["failures"] and "areas" in unread:
        text, lines = raised_unread(unread, _module(ROOT / UNREAD, "lean_unread_for_lean_audit"))
        repaired[UNREAD_CAPS] = keep_repaired(out, UNREAD_CAPS, text)
        warned += lines
    elif unread and unread["failures"]:
        failed.append("unread")
    if bookkeeping and not failed:
        locks = {p: json.loads((Path(regenerated[p]) if p in regenerated else ROOT / p / "lean-audit.json").read_text())
                 for p in packages(files)}
        try:
            text, lines = held_caps((ROOT / CAPS_TEST).read_text(), locks)
        except (OSError, LookupError, ValueError) as e:
            log(f"lean-audit: FAIL: bookkeeping: {e}")
            failed.append("bookkeeping")
        else:
            if lines:
                repaired[CAPS_TEST] = keep_repaired(out, CAPS_TEST, text)
                warned += lines
    for w in warned:
        log(f"lean-audit: WARN: {w}")
    rekeyed: dict[str, str] = {}
    if documents and regenerated and not failed:
        try:                                            # it changes no verdict of this run, so its own error fails none
            rekeyed = keep_regenerated(pkgs, files, keys, reports, regenerated, ran, documents, out, cache, log)
        except (OSError, EOFError, ValueError, LookupError, TypeError, AttributeError, ImportError, SyntaxError) as e:
            log(f"lean-audit: WARN: no pass kept for the commit of the regenerated records, whose audit runs in full: "
                f"{type(e).__name__}: {e}")
    (out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "cached": cached,
                                                "packages": [reports[p] for p in pkgs if p in reports],
                                                "regenerated": regenerated, "unread": unread} | (
                                                    {"rekeyed": rekeyed} if regenerate else {}) | (
                                                    {"repaired": repaired, "bookkeeping": warned} if bookkeeping else {}),
                                               indent=1) + "\n")
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
            **{SHARED + h: r.presign(object_key(h), ttl_s) for h in shared_bundles(heads=COMMON.merged_heads(ROOT))}}


def shared_bundles(store=None, limit: int = SHARED_LIMIT, heads: list[str] | tuple[str, ...] = ()) -> list[str]:
    """The sha256 of each artifact bundle (`LakeArtifacts.bundle`) among the files of the newest runs of each kind of
    `PUBLISHERS`, the recorded checks' first, each kind's newest first and at most ``limit`` runs with a bundle, every
    bundle of each in the order its run wrote them (`LakeArtifacts.bundle_into`; a node takes them in this order up to
    its budget, `LakeArtifacts.take`): only a recorded run's, as
    `check.verdict_packs` takes only theirs, a failed or cancelled run's too, since its bundle holds only modules compiled
    in the sandbox, each under the key of what its compile read (`LakeArtifacts.put`). Before the newest mark audits' come
    those of the marks of ``heads`` (`common.lookups`; the launcher gives the heads of the PRs its tree holds,
    `common.merged_heads`), however old and beside the ``limit``: a PR waits hours between its mark and its candidate,
    newer marks push its bundle out of the newest, and a tree whose kept build is in use compiles every module above one
    its node lacks. When the store has a remote, the attempts started in the last `common.RECENT_S` are refreshed from it
    first. Empty when the store can't be read: each node then reuses only what it built."""
    from research.store.local import LocalStore

    ended = "state IN ('done', 'failed', 'cancelled') ORDER BY start_utc DESC LIMIT ?"
    out: list[str] = []
    try:
        store = store or LocalStore()
        if store.remote is not None:
            store.refresh(attempts_only=True, since=COMMON.recent(time.time()))
        for runs, bundle, listed in PUBLISHERS:
            for which, args, most in COMMON.lookups(runs, limit, heads):
                found: list[str] = []
                taken = 0
                for (rid,) in store.index.sql(f"SELECT id FROM attempts WHERE {which} AND {ended}", args):
                    outputs = (store.get_attempt(rid) or {}).get("outputs") or {}
                    mine: dict[str, int] = {}
                    for art in dict.fromkeys(a for a in map(outputs.get, listed) if a):
                        for f in store.listing(art):
                            if (m := bundle.fullmatch(f["path"])) and f["remote"] and f["sha256"] not in (*out, *found):
                                mine.setdefault(f["sha256"], int(m[1][1:]) if m[1] else 1)
                    found += sorted(mine, key=mine.__getitem__)
                    taken += bool(mine)
                    if most and taken >= most:
                        break
                out += found
    except Exception as e:  # noqa: BLE001  (no bundle: each node reuses only what it built)
        print(f"lean-audit: no artifact bundles listed from the store ({type(e).__name__}: {e})", file=sys.stderr)
    return out


def bundled(dests: list[Path], n: int, added: int) -> str:
    """What a run says of the bundles it wrote (`LakeArtifacts.bundle_into`'s, and how many entries it had ``added``)."""
    return (f"{n} of the {added} module version(s) it added bundled for other nodes "
            f"({', '.join(d.name for d in dests) or 'no bundle'})"
            + ("" if n == added else f"; {added - n} in none: gone from the cache since, or more than a bundle holds"))


def url_file(dest: Path, ttl_s: int = URL_TTL_S) -> Path | None:
    """`presigned` written as ``dest/lean-deps-urls.json`` (mode 600), the name `check.py` and a records run
    (`lean_changed.sent_urls`) look for among a run's inputs."""
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
        if "modules" not in v:
            print(f"lean_audit: {v['package']}: the bundle {k[:12]} lists no modules (an export before `modules`); export "
                  "again", file=sys.stderr)
            return 1
        deps[k] = {"package": v["package"], "sha256": v["sha256"], "bytes": v["bytes"], "run": run_id, "modules": v["modules"]}
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
               "lean-artifacts": LakeArtifacts(root / "lean-artifacts").prune(),
               "lean-audit/unread": prune_facts(root / "lean-audit")}
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
    ap.add_argument("--regenerate", action="store_true", help="the Lean check of `next`: `check.py --regenerate`, so a package whose "
                    "every failure is a record difference passes and audit.json's `regenerated` names its record; not in the "
                    "cache key, and such a pass is never stored")
    ap.add_argument("--bundle", action="store_true", help="a recorded check's audit: write the module versions it added to the "
                    "artifact cache as lean-artifacts.tar.gz in --out, and as lean-artifacts-2.tar.gz and so on when one "
                    "bundle can't hold them, for other nodes to take (`shared_bundles`)")
    ap.add_argument("--bookkeeping", choices=["warn"], help="with --regenerate, drive's lock run after a landing: "
                    "`check.py --bookkeeping warn`, so a package that fails only on its lock's bookkeeping passes on the lock "
                    "with it repaired; an unread area above its cap, and a cap of `tests/test_lean_packages.py` the locks "
                    "left break (UNWITNESSED, INLINE), is raised to the count, or lowered to the count and its room, in a copy under --out's "
                    "`repaired/`, which audit.json's `repaired` names for drive to commit; each repair is a warning in "
                    "audit.json's `bookkeeping`. Every package is audited (`--package` may name them all, as check.py's parts do): the "
                    "caps count every package")
    a = ap.parse_args(argv)
    if a.bookkeeping and not a.regenerate:
        ap.error("--bookkeeping warn goes with --regenerate: drive's lock run, which commits the locks it regenerates")
    urls = json.loads(a.urls.read_text()) if a.urls and a.urls.is_file() else None
    return run(a.out, a.cache, a.deps, a.fresh, log=lambda s: print(s, flush=True), urls=urls, export=a.export, only=a.package,
               regenerate=a.regenerate, bundle=a.bundle, bookkeeping=bool(a.bookkeeping))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
