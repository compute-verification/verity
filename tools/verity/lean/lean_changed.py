"""`lean_changed`: Lean's early job, which `check` runs as its lean-changed step. `lake build` of every Lake package whose Lean
audit misses (`lean_audit.py`: its inputs changed since a pass kept on this machine), on the step tree (`lean-changed` under
the check cache, or `--tree`), which this machine keeps built, so a module that no longer compiles (two merges git combines
cleanly that Lean doesn't) fails the run in minutes rather than when the audit's cold build reaches it.  Its `--records`,
`--try` and `--build` work on a slot tree instead, one per `build` slot (below).

    python tools/verity/lean/lean_changed.py --cache DIR --deps DIR --tree DIR
    python tools/verity/lean/lean_changed.py --records PKG [--update [--moved FILE]] [--out DIR] [--dev-cache] [--rebuild]   a lane's records
    python tools/verity/lean/lean_changed.py --try FILE [--siblings]   a lane's edit loop on a node (`verity lean try FILE --on NODE`)
    python tools/verity/lean/lean_changed.py [--no-wait] --build FILE...   the ready mark's preflight's Lean step (`build_files`)
    python tools/verity/lean/lean_changed.py --inputs-key PKG... [--head SHA] [--repo DIR]   each PKG's inputs key at SHA

`--try` elaborates one file (FILE: its path from the repository's root) on the slot tree `--records` builds on, under the same
`build` slot, tree lock, sync, `repair` and dependencies, for the closure of the package that owns FILE only: there it runs
`lake build +MODULE:deps` (FILE's imports; Lake rebuilds only what changed since the slot's last build) and then
`lake env lean` with the package's `[leanOptions]` (`siblings.options`) in that package's directory on a copy of FILE with
the axiom probe appended (`axiom_probe.py`: its last line names each theorem whose axioms go beyond the standard three,
`sorryAx` among them, or says none do), and exits with lean's
status on FILE's own lines (with `--siblings`, that copy's build then goes through the sibling check, `siblings.py`, whose
status it takes); when that status is an import clash (an
import declaring what an earlier one did), the two modules it names lose their traces (`clashed`), so the next try rebuilds
them against the outputs in place. It never runs check.py, so it reads and writes
no record; where the node grants no `build` slot it refuses, naming the machines whose Lean slots file has that pool (each
one `slot.py`'s PEERS lists, read over ssh: `pool_machines`), else how to list a machine's Lean pools. Dependencies the
slot tree holds that no `--records` run took (not in its `links.json`) are kept as they are; without a sandbox Lake runs
unsandboxed, and fetches dependencies the node does not keep.

`--records` is a lane's fast path to the records in PKG's `lean-audit.json` (PKG: its path, or a unique part of one):
`tools/verity/lean/check.py --build --no-replay --no-runs` on PKG in a slot tree (one per slot), which prints what the audit prints,
`--update`'s review included, after a build of only what changed since that slot's last run, which restores each module the
node's artifact cache holds under its key (`check.py --artifacts`, `lean_audit.LakeArtifacts`, the one the audit
reads); each package whose check.py passes, its report saying it built in the sandbox, puts the modules its build
compiled there. It first waits for one of the
node's Lean slots of pool `build` (`lean_slot.py`; pool `any` never stands in for it), never one of the `audit` pool's, where
hour-long audits hold every slot, whatever slot (LEAN_SLOT) the process that started it holds. Where SLOTS lists no `build`
pool, or with `--dev-cache` (a slot tree whose dependencies are fetched cold builds them from source), it takes the `--dev-cache` path
instead (`dev_records`): `check.py --dev-cache --build --no-replay --no-runs` on this
checkout's PKG, then its requirers, each build restored from the published cache (Security: about 40 minutes), with the same
reports, record keep and kind; that path builds in this checkout, so a run takes it only with `--cwd clone`. It says which
path it took, and a recorded run's `result.json` names it (`path`: `slot` or `dev-cache`). Slot `build/I`'s slot tree is
`lean-records/build-I` under the check cache, whoever holds that slot's lock owns it, and check.py inherits the lock's
descriptor, so the slot is held until check.py exits; with no SLOTS (a laptop) it is `lean-records/local`. `--tree` names
another, only under `lean-records/`. A slot tree's fetched dependencies are its own copy of the audit's (taken once per
manifest and toolchain, else fetched cold), never links (the step tree's are links, below), so what Lake writes there stays
out of what the audit keeps. That bounds Lake's own writes, not compile-time code: check.py builds in its sandbox only where the
machine allows one, and without it that code can write anything this user can, the audit's kept dependencies and passes
included, as any build this user runs by hand can. `--update` writes the record back into this checkout, but not in a recorded
run, whose source tree the commit's other runs share: there it is in the run's `lean-audit/audit-*/`, and on either path a
record that changed is also kept as `lean-audit/record/<path>/lean-audit.json` (PKG's also as
`lean-audit/record/lean-audit.json`), which `research queue`'s records gate reads, each with a line saying what to commit it
over. A run marks the slot tree (`DIRTY`: its id, package and phase) while it works there and clears the mark when it ends
without a check.py killed, so the next run knows a slot tree left mid-way (a run cancelled, timed out or killed for memory) and
says so. Before each build it `repair`s the slot tree: a module an interrupted build or a restore from Lake's cache left half-written (an output its trace lists missing, an `.olean` that is
not one, a `.hash` that is not the trace's or is older than its output) loses its trace and `.hash` files, so Lake rebuilds
it, as does a module's `.hash` file when the module has no trace, and after a run left mid-way, a package's fetched dependencies that run was taking, or that hold such a
module or a checkout other than the manifest's, are taken again. A spec whose `proved_in` package's build did not pass
never reads that package's `.lake` (`check.proved`), which an earlier or interrupted build left.

Then it audits, the same way, every package whose guarantees' records read PKG's build (`requirers`: PKG is in its
`closure`, by path or by `proved_in`), each after those it requires and on the records this run regenerated before it, its
reports under `requirer-<path>/` beside PKG's. Of the facts documents, only PKG's own stays beside its reports
(`check_argv`: `check.py --facts named`, which `trusted_size.py --run` reads); its provers' and the requirers' are not
kept. A change to what another package's pinned guarantees read
(`Security/Properties`' when `Security` changes) fails here, not at the Lean check of `next`. With `--update`, a requirer whose every
failure is a record difference (`settles`) is audited again with `--update`, so one run regenerates every lock the change
moved; one failing for any other reason fails. With `--moved` too, each requirer's first audit takes `--update --moved`
(`first_update`; check.py reads `--moved` only with `--update`), so a rename its records read resolves before its
failures are judged. The exit status is the first failure's. A slot tree
keeps the copies of the packages one run audits (about 170k inodes, 8 GB each), so auditing another package that fetches
dependencies there drops the rest first.

A record difference is not a failure of the run. A package that any other run of check.py fails only on record differences
(check.py's report names the record that settles them, `regenerated`) has that record kept the same way, so a run without
`--update` regenerates too. The run's outcome (`result.json`'s `outcome`, and its last line, `OUTCOME_LINE` and JSON) is its
packages' worst: `pass`, `regenerated` (each changed lock kept under `record/`), `failed` (a failure of any other kind; a
record kept for such a package goes again) or `refused` (nothing ran). It exits 0 on `pass` and `regenerated`, check.py's
first failing status on `failed` and 2 on `refused`.

A package whose records need no run is skipped, and with none left nothing is built and no slot taken (`reused`; the
`result.json`'s `path` is `reused`). Each package whose check.py passes is kept under `lean-records/passes/`, keyed on the
content of every file its records read (`inputs_key`: its `closure`'s files and the auditor's, this file included), with
each lock this run regenerated as check.py left it, so a commit of the records `--update` wrote finds it; `result.json`'s
`passes` ({key: package}) lists the same keys, and each record its package's `inputs_key` as the commit has it. A package
needs no run when such a pass, or the Lean check's own pass (`lean_audit.key`), is kept on these inputs on this node, or a
result the store holds (`stored`, by `research data select`: the store's index where the machine has one, its catalog
otherwise) lists its key under `passes` (`kept`; the record names that run and commit, `reused_from`), or
when HEAD merges a commit of `main` into one whose records passed, possibly through further such merges, and each merge took
every Lean input of the package whole from main's side, its locks aside, which must be `merge.py`'s merge of the two sides
(`merged`). That lock is a candidate, never a verdict: the Lean check of `next` rebuilds every record and regenerates one
that differs. `--rebuild`, and `--moved`, run every package.

`--inputs-key` prints the key of each package at a commit from git's objects alone (`keys_at`), so a caller with only a
clone, its worktree not checked out, can ask the store for a pass on it before any run.

Its result is of kind `FAST_KIND`, never a Lean verdict: no kernel replay and no tests. The kind is written into the
`audit.json` and each package's `report.json` it leaves (`--out`, else the run's `lean-audit/`, else a temporary directory) and,
in a recorded run, into its `result.json`, the Attempt's result. `check` fails its lean-audit step on a report of that kind or
one without the replay (`check.lean_fast`), and `research merge` refuses an attempt whose result is of that kind
(`research.merge.REFUSED_KINDS`), whatever `--tool` recorded it. The compiled modules it leaves are not a verdict either: in a
run they go into its `lean-audit/` as a bundle (`lean_audit.LakeArtifacts.bundle_into`), and a node that restores one still
replays every declaration through the kernel.

The step tree holds the tracked files of each package whose audit misses, and those of the packages it requires by path or
names under `proved_in`, at their paths in this repository, with each package's own `.lake`: Lake rebuilds only the modules that changed since its last build there and
what imports them. A package's fetched dependencies are hard links to the ones the audit keeps (`--deps`, `lean_audit.WarmDeps`),
taken under their lock while the step tree has none for its manifest and toolchain; the build runs in the audit's sandbox
(`tools/verity/lean/sandbox.sh`), where they are mounted read-only, so it cannot write what the audit reads, and a concurrent audit can
move them away without touching the links. The first build of a package here is cold; later ones are not.

A package with guarantees then has its roots built (`lake build +ROOT`, as the audit does) and its guarantees' records
compared with that build (`records`): `check.py`'s own reading of the `.olean` files (`Facts.lean`, in the sandbox) and its
judgment of the guarantees, without the kernel replay, so a record two merges left stale (a statement or a definition a
guarantee reads changed, a guarantee gone, a `reads` digest no longer its definitions') fails the run before the audit's cold
build reaches it.

The audit still builds and audits every package cold and decides the run: a pass here vouches for nothing. So this step fails
only on a Lean error at a position in one of the step tree's own `.lean` files, or on a guarantee the audit fails (`judge`'s
failures whose code is under `guarantees.` or `assumptions.`). Anything else that stops a build (no sandbox or `lake` here, dependencies this machine doesn't keep
yet, another check holding the step tree, a dependency the links lack) skips the package by name and drops its links, so the next
run takes the audit's again; anything that stops a reading (a policy that can't be read, `Facts.lean` failing) skips its
guarantees by name.
`check.py` ends the run when this step fails, even under `--keep-going`. Delete the step tree to start it over.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import hashlib
import importlib.util
import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LA = _module(ROOT / "tools" / "verity" / "lean" / "lean_audit.py", "lean_audit_for_lean_changed")
#: `tools/verity/lean/check.py`, this step's own copy: `records` runs its commands in the sandbox
AUDIT = _module(ROOT / "tools" / "verity" / "lean" / "check.py", "lean_audit_tool_for_lean_changed")
SIBLINGS = _module(ROOT / "tools" / "verity" / "lean" / "siblings.py", "lean_siblings_for_lean_changed")
PROBE = _module(ROOT / "tools" / "verity" / "lean" / "axiom_probe.py", "axiom_probe_for_lean_changed")
#: where `Facts.lean` writes a package's facts: inside its `.lake`, which the sandbox leaves writable and `sync` leaves alone
FACTS = Path(".lake") / "lean-changed-facts"
#: the audit's build sandbox, as `tools/verity/lean/check.py` runs it: writable only in the `.lake` directories it is given
UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child"]
SANDBOX = ROOT / "tools" / "verity" / "lean" / "sandbox.sh"
#: run inside the sandbox before the build: `ro DIR... -- CMD`, each DIR bind-mounted read-only over itself
READ_ONLY = ('ro=(); while [ "$1" != -- ]; do ro+=("$1"); shift; done; shift; '
             'for d in "${ro[@]}"; do mount --bind "$d" "$d" && mount -o remount,bind,ro "$d" || exit 97; done; exec "$@"')
#: Lake's line for a Lean error, the file relative to its package: `error: Aa/Two.lean:2:39: Unknown identifier ...`
ERROR = re.compile(r"^error: (?:\./)*(\S+?\.lean):\d+:\d+: ")
#: {package: the dependencies' key its links were taken under}, beside the packages in the tree
LINKS = "links.json"


def missed(files: list[str], cache: Path) -> list[str]:
    """The packages whose audit would run, those another requires by path first, and a spec after the packages its
    `proved_in` names."""
    pkgs = [p for p in LA.packages(files) if not (cache / f"{LA.key(p, files, LA.FLAGS)}.json").is_file()]
    return sorted(pkgs, key=lambda p: (rank(p), p))


def rank(pkg: str) -> int:
    """How deep ``pkg`` sits among the packages it requires by path and those its `proved_in` names: each sorts after them."""
    return max([len(LA.path_deps(pkg)), *(len(LA.path_deps(q)) + 1 for q in LA.provers(pkg))])


def read_here(f: str) -> bytes | None:
    """A repository path's content in this checkout, as git keeps it (a symbolic link's target); None when it is absent."""
    p = ROOT / f
    if p.is_symlink():
        return os.readlink(p).encode()
    return p.read_bytes() if p.is_file() else None


def path_deps(pkg: str, read=read_here) -> list[str]:
    """The packages ``pkg`` requires by path, as `lean_audit.path_deps` reads them from its manifest, through ``read``."""
    b = read(f"{pkg}/lake-manifest.json")
    if b is None:
        return []
    return sorted({posixpath.normpath(posixpath.join(pkg, d["dir"])) for d in json.loads(b).get("packages", [])
                   if d.get("type") == "path"})


def provers(pkg: str, read=read_here) -> list[str]:
    """The packages ``pkg``'s `lean-audit.json` names under `proved_in`, as `lean_audit.provers` reads them, through ``read``:
    none when it can't be read (said on stderr), and the audit then fails its policy."""
    try:
        b = read(f"{pkg}/lean-audit.json")
        named = json.loads(b).get("proved_in", []) if b is not None else []
    except (OSError, ValueError, AttributeError) as e:
        print(f"lean_changed: {pkg}/lean-audit.json can't be read ({e!r}); it is taken to name no proved_in", file=sys.stderr)
        return []
    return sorted({posixpath.normpath(posixpath.join(pkg, d)) for d in named if isinstance(d, str)})


def closure(pkg: str, read=read_here) -> list[str]:
    """``pkg``, the packages it requires by path, and those its `proved_in` names with theirs, whose builds `records` reads;
    its manifests and locks read through ``read`` (this checkout's by default)."""
    return list(dict.fromkeys([pkg, *path_deps(pkg, read),
                               *(d for q in provers(pkg, read) for d in (q, *path_deps(q, read)))]))


def lists_guarantees(pkg: str) -> bool:
    """Whether ``pkg``'s `lean-audit.json` lists a guarantee; True when it can't be read, which the audit then fails."""
    try:
        return bool(json.loads((ROOT / pkg / "lean-audit.json").read_text()).get("guarantees"))
    except (OSError, ValueError, AttributeError):
        return True


def requirers(pkg: str, pkgs: list[str]) -> list[str]:
    """The other packages among ``pkgs`` whose guarantees' records read ``pkg``'s build (``pkg`` is in their `closure`), each
    after those it requires, as `missed` orders them."""
    return sorted((q for q in pkgs if q != pkg and pkg in closure(q) and lists_guarantees(q)), key=lambda p: (rank(p), p))


def sync(pkgs: list[str], files: list[str], tree: Path) -> int:
    """Make ``tree``'s copy of ``pkgs`` this checkout's tracked files, leaving each `.lake`; the number of files written."""
    tracked, n = set(files), 0
    for p in pkgs:
        for d, dirs, names in os.walk(tree / p):
            dirs[:] = [x for x in dirs if x != ".lake"]
            for f in names:
                rel = os.path.relpath(os.path.join(d, f), tree)
                if rel not in tracked:
                    os.unlink(tree / rel)
        for f in LA.own_files(p, files):
            src, dst = ROOT / f, tree / f
            if src.is_symlink():
                if dst.is_symlink() and os.readlink(dst) == os.readlink(src):
                    continue
                dst.unlink(missing_ok=True)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(os.readlink(src))
            elif src.is_file():
                data = src.read_bytes()
                if dst.is_file() and not dst.is_symlink() and dst.read_bytes() == data:
                    continue
                dst.unlink(missing_ok=True)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
            else:
                continue
            n += 1
    return n


def link(pkg: str, tree: Path, deps: Path, links: dict[str, str], log=print, copy: bool = False) -> bool:
    """``tree/pkg/.lake/packages`` as hard links to the audit's kept dependencies for its manifest and toolchain (``copy``: a
    copy of them, taken after their lock is released from hard links made under it, `packages.snap` beside it, so that the
    audit waits on the lock only as long as linking takes; where they can't be linked, the copy is taken under the lock);
    False when this machine keeps none (another run holds them, or none has passed here yet)."""
    if (k := LA.deps_key(tree / pkg)) is None:
        return True
    dst = tree / pkg / ".lake" / "packages"
    if links.get(pkg) == k and dst.is_dir():
        return True
    unlink(pkg, tree, links)
    verb, how = ("copy", "copied") if copy else ("link", "linked")
    snap = dst.with_name("packages.snap")
    shutil.rmtree(snap, ignore_errors=True)
    try:
        with LA.WarmDeps(deps)._lock(k):
            src = deps / k
            if not src.is_dir():
                log(f"lean-changed: {pkg}: this machine keeps no dependencies for it ({k[:12]}) to {verb}")
                return False
            t0 = time.monotonic()
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copytree(src, snap if copy else dst, symlinks=True, copy_function=os.link)
            except (OSError, shutil.Error):
                if not copy:
                    raise
                shutil.rmtree(snap, ignore_errors=True)
                shutil.copytree(src, dst, symlinks=True, copy_function=shutil.copy2)
        if snap.is_dir():
            shutil.copytree(snap, dst, symlinks=True, copy_function=shutil.copy2)
    except (OSError, shutil.Error) as e:
        shutil.rmtree(dst, ignore_errors=True)
        log(f"lean-changed: {pkg}: the dependencies ({k[:12]}) could not be {how} ({e})"
            + ("" if copy else "; a link needs one filesystem"))
        return False
    finally:
        shutil.rmtree(snap, ignore_errors=True)
    links[pkg] = k
    log(f"lean-changed: {pkg}: dependencies {how} ({k[:12]}) in {time.monotonic() - t0:.0f} s")
    return True


def unlink(pkg: str, tree: Path, links: dict[str, str]) -> None:
    shutil.rmtree(tree / pkg / ".lake" / "packages", ignore_errors=True)
    links.pop(pkg, None)


def read_links(tree: Path) -> dict[str, str]:
    """The tree's `LINKS`; none when the file is missing or not a JSON object (a write in place that failed after its
    truncation, on a full disk or a stopped run, left it empty), so each package's dependencies are taken again, as in a
    new tree."""
    try:
        links = json.loads((tree / LINKS).read_text())
    except (FileNotFoundError, ValueError):
        return {}
    return links if isinstance(links, dict) else {}


def write_links(tree: Path, links: dict[str, str]) -> None:
    """`LINKS`, replaced whole, so a write that fails (a full disk, a stopped run) leaves the previous file."""
    tmp = tree / f"{LINKS}.tmp"
    tmp.write_text(json.dumps(links, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, tree / LINKS)


_MOVED = None


def _moved():
    """`moved.py`, loaded by what carries a build and not with this file: `--inputs-key` reads no tree and no recorded
    move, and whoever runs it from a commit's key code alone (research's records gate) needs no file for it."""
    global _MOVED
    if _MOVED is None:
        _MOVED = _module(Path(__file__).resolve().parent / "moved.py", "lean_moved_for_lean_changed")
    return _MOVED


def carry_builds(tree: Path, log=print) -> None:
    """Under ``tree``'s lock, before anything reads `LINKS` or syncs a package: a build directory that a layout move left
    in ``tree`` at its package's former path moves to where this checkout has the package (`moved.carry`), with its entry
    of `LINKS`. A tree is keyed by package paths and `.lake` is no tracked file: without this the first run after a move
    would sync the package to its new path, find no build there and compile every module of it, the old build left beside
    it. It reads the tree, the checkout and the recorded moves, and starts no program. A tree that is the checkout itself
    keeps what else its former directory holds."""
    moved = _moved()
    moves = moved.recorded(ROOT)
    if not moves or not tree.is_dir():
        return
    built = []
    for d, dirs, _ in os.walk(tree):
        if ".lake" in dirs and Path(d) != tree:
            built.append(Path(d).relative_to(tree).as_posix())
        dirs[:] = [x for x in dirs if not x.startswith(".")]
    pkgs = sorted({q for f in built if (q := moved.now(f, moves)) != f
                   and any((ROOT / q / n).is_file() for n in moved.LAKEFILES)})
    got = moved.carry(tree, pkgs, moves, log, prune=tree.resolve() != ROOT.resolve()) if pkgs else {}
    if not got:
        return
    links = read_links(tree)
    for old, new in got.items():
        if old in links:
            links[new] = links.pop(old)
    write_links(tree, links)


def in_sandbox(pkg: str, tree: Path, argv: list[str]) -> list[str]:
    """``argv`` in the sandbox, writable in the `.lake` of ``tree/pkg`` and the packages it requires by path but not in their
    fetched dependencies. The paths are resolved, as mountinfo names them: a slot tree is a link into the node's cache root."""
    rw = [str((tree / p / ".lake").resolve()) for p in closure(pkg)]
    ro = [str(Path(d) / "packages") for d in rw if (Path(d) / "packages").is_dir()]
    return [*UNSHARE, "--", "bash", str(SANDBOX), *rw, "--", "bash", "-c", READ_ONLY, "ro", *ro, "--", *argv]


def build(pkg: str, tree: Path, lake: str, targets: list[str] | None = None, sandbox: bool = True,
          cmd: list[str] | None = None) -> tuple[int, str]:
    """`lake build` (of ``targets``, else the default ones; ``cmd``: that `lake` command instead) of ``tree/pkg``, in the
    sandbox unless ``sandbox`` is False; Lake's output is echoed as it comes. Before the default targets it links the
    dependencies' compiler plugins alone (`check.py`'s `plugin_targets`)."""
    if cmd is None and targets is None and (plugins := AUDIT.plugin_targets(tree / pkg)):
        rc, text = build(pkg, tree, lake, plugins, sandbox)
        if rc:
            return rc, text
    argv = [lake, *(cmd or ["build", *(targets or [])])]
    argv = in_sandbox(pkg, tree, argv) if sandbox else argv
    env = {**os.environ, "LEAN_NUM_THREADS": os.environ.get("LEAN_NUM_THREADS") or
           str(AUDIT.build_threads(tree / pkg) or LA.COMMON.cores())}
    p = subprocess.Popen(argv, cwd=tree / pkg, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    out = []
    for line in p.stdout:
        print(line, end="", flush=True)
        out.append(line)
    return p.wait(), "".join(out)


def errors(text: str, pkg: str, files: list[str]) -> list[str]:
    """Lake's Lean errors in ``text`` that sit in a `.lean` file of ``pkg`` or a package it requires by path."""
    own = {f[len(p) + 1:] for p in closure(pkg) for f in LA.own_files(p, files) if f.endswith(".lean")}
    return [line for line in text.splitlines() if (m := ERROR.match(line)) and m.group(1) in own]


def guarantee_roots(pkg: str, tree: Path) -> list[str]:
    """The roots of ``tree/pkg``'s policy when it lists a guarantee, which `records` reads; none otherwise."""
    policy = AUDIT.load(tree / pkg)
    return list(policy["roots"]) if policy.get("guarantees") else []


def records(pkg: str, tree: Path, lake: str) -> list[str]:
    """The audit's failures about the guarantees on the tree's build of ``pkg`` (`check.py`'s `facts_of` and `judge`, without
    the kernel replay): a guarantee missing, no longer a theorem or with no record, using an axiom the policy does not allow or
    assuming an unnamed `Prop`, whose statement or the definitions it reads no longer match its record, or a `reads` entry
    that no longer lists the modules the guarantees read or their definitions' digest, or a named assumption whose record
    no longer matches the definition the guarantees read. A spec's
    guarantees are read from the builds of the packages its `proved_in` names (`audit.proved`), built before it (`missed`).
    Raises RuntimeError when `Facts.lean` cannot read a build."""
    policy = AUDIT.load(tree / pkg)
    work = tree / pkg / FACTS
    work.mkdir(parents=True, exist_ok=True)
    plain, tool, programs, scratch = AUDIT.run, AUDIT.TOOL, AUDIT.PROGRAMS, AUDIT.SCRATCH
    # the sandbox mounts a private /tmp, where this checkout may be; the copy sits in the writable `.lake`, as the audit runs
    # its tool from a copy
    shutil.copy2(tool / "Facts.lean", work / "Facts.lean")

    def lake_in_sandbox(argv: list[str], cwd: Path, *a, **kw) -> tuple[int, str]:
        # the audit runs `lake env` unsandboxed on a tree of its own; here `.lake/packages` holds links to the files it keeps
        return plain(in_sandbox(pkg, tree, [lake, *argv[1:]] if argv[:1] == ["lake"] else argv), cwd, *a, **kw)

    # `Facts.lean`, in the sandbox, writes its document under `work` for the same reason
    AUDIT.run, AUDIT.TOOL, AUDIT.PROGRAMS, AUDIT.SCRATCH = lake_in_sandbox, work, work, work
    try:
        facts = AUDIT.facts_of(tree / pkg, policy, work, keep=False)
        cross: list[str] = []
        if "proved_in" in policy:
            cross, facts, _ = AUDIT.proved(tree / pkg, policy, facts, work, keep=False)
    finally:
        AUDIT.run, AUDIT.TOOL, AUDIT.PROGRAMS, AUDIT.SCRATCH = plain, tool, programs, scratch
    fails, _ = AUDIT.judge(policy, facts, AUDIT.lean_modules(tree / pkg))
    return [f for f in cross + fails if (AUDIT.code_of(f) or "").split(".")[0] in ("guarantees", "assumptions")]


def sandboxed() -> bool:
    try:
        return subprocess.run([*UNSHARE, "--", "true"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


#: `--records`: its trees, one per Lean slot (`build-I`; `local` where the node has no SLOTS), under the check cache
RECORDS = "lean-records"
AUDIT_PY = ROOT / "tools" / "verity" / "lean" / "check.py"
LEAN_SLOT = _module(ROOT / "tools" / "verity" / "check" / "lean_slot.py", "lean_slot_for_lean_changed")
#: the Lean slot pool `--records` takes, apart from the `audit` pool's hour-long audits and the `check` pool's merge checks;
#: $LEAN_RECORDS_POOL names another, such as a pool a node reserves for one lane, whose trees stay warm for that lane alone
FAST_POOL = os.environ.get("LEAN_RECORDS_POOL") or "build"
#: the kind of `--records`' result: an audit without the kernel replay or the tests, which no merge check accepts
FAST_KIND = "lean-fast/v1"
#: what a `--records` run found (its `result.json`'s `outcome`, and each package's under `records`), the better first: every
#: package passed on the commit's own lock; every failure was a record difference, and each record that settles it is kept
#: under `record/`; a failure of another kind; nothing ran
PASS, REGENERATED, FAILED, REFUSED = OUTCOMES = ("pass", "regenerated", "failed", "refused")
#: `--records`' last line, then a JSON object: its outcome, its package and its `records`
OUTCOME_LINE = "lean-records: OUTCOME "
#: how long `--records` waits for another run's hold on its tree (a node with no SLOTS shares `local` among every run;
#: without the wait, runs behind an hour-long one would be refused at once), and how often it looks
TREE_WAIT_S, TREE_POLL_S = 3 * 3600, 20
_sleep, _clock = time.sleep, time.monotonic


def hold_tree(held, tree: Path, log=print) -> bool:
    """Take ``held``'s lock (the tree's `.lock`), waiting up to `TREE_WAIT_S` for the run that holds it, whose id `.who`
    beside it names; False when it is still held then. With the lock, the builds a layout move left behind in the tree
    are brought to their packages (`carry_builds`), before the caller reads the tree."""
    who, start, told = tree / ".who", _clock(), None
    while True:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            holder = who.read_text().strip() if who.is_file() else "unknown"
            if _clock() - start >= TREE_WAIT_S:
                log(f"lean-records: another run holds {tree} ({holder}), still after {TREE_WAIT_S} s: nothing ran")
                return False
            if holder != told:
                log(f"lean-records: another run holds {tree} ({holder}); waiting, retrying every {TREE_POLL_S} s")
                told = holder
            _sleep(TREE_POLL_S)
    who.write_text(f"{os.environ.get('RESEARCH_RUN_ID') or 'a run'} (pid {os.getpid()})\n")
    carry_builds(tree, log)
    return True


#: beside the packages in a `--records` tree while a run works on it: the run, the package and the phase it is in. The tree's
#: lock is held while it exists, so a run that takes the lock and finds it took the tree from one that ended mid-way (killed,
#: cancelled, timed out, out of memory, its node stopped), and `repair`s it first
DIRTY = "dirty.json"


def mark(tree: Path, **what) -> None:
    """`DIRTY`, replaced whole: this run and ``what`` (`package`, `phase`)."""
    doc = {"run": os.environ.get("RESEARCH_RUN_ID") or "a run", "pid": os.getpid(),
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **what}
    tmp = tree / f"{DIRTY}.tmp"
    tmp.write_text(json.dumps(doc, sort_keys=True) + "\n")
    os.replace(tmp, tree / DIRTY)


def left_by(tree: Path) -> dict | None:
    """The `DIRTY` a run left in ``tree`` (``{}`` when it can't be read), or None when the last run ended cleanly."""
    try:
        doc = json.loads((tree / DIRTY).read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


#: the outputs a module's trace names that Lake writes beside it in `lib/lean`: its oleans (`o`: `.olean`, and in a `module`
#: file `.olean.server` and `.olean.private`), `.ilean` (`i`), and in a `module` file `.ir` (`r`) and `.ir.sig` (`rs`). All
#: but the `.ilean` are in the olean format, which starts `olean`.
LIB_OUTPUTS = ("o", "i", "r", "rs")


def damaged(lib: Path) -> list[Path]:
    """The traces under ``lib`` (a `.lake/build/lib/lean`) whose module an interrupted build or a restore left half-written:
    an output the trace lists beside it that is missing (`LIB_OUTPUTS`), one in the olean format without its header, an
    output whose `.hash` is not the trace's or is older than the output, or a trace that is not JSON. Lake rebuilds a module
    whose outputs are missing, but takes one whose trace matches and whose files exist as built, half-written or not, and
    reads an output's hash from its `.hash` without hashing the output. Every Lake path writes the `.hash` after its output,
    but two leave an older one: a build killed after `lean` wrote its outputs, and a restore from Lake's cache that unpacks
    a `.ltar` over the outputs.
    A module that imports one then records the old hash for the new bytes, and a later build on the other side of the
    change replays it beside a module built against the new ones (a clash such as `Proofs.Pouw`'s on
    `W1.Instr.exec.eq_10`). So each loses its trace and its `.hash` files (`forget`) for Lake to rebuild it."""
    out = []
    for trace in sorted(lib.rglob("*.trace")) if lib.is_dir() else []:
        base = trace.with_suffix("")
        try:
            doc = json.loads(trace.read_text())
            outputs = doc.get("outputs") if isinstance(doc, dict) else None
            names = [n for k, v in outputs.items() if k in LIB_OUTPUTS for n in (v if isinstance(v, list) else [v])
                     if isinstance(n, str)] if isinstance(outputs, dict) else []
            for name in names:
                digest, _, ext = name.partition(".")
                f = base.with_name(f"{base.name}.{ext}")
                if not f.is_file() or (ext != "ilean" and f.open("rb").read(5) != b"olean"):
                    raise ValueError(f"{f.name} missing or not in the olean format")
                if (h := f.with_name(f"{f.name}.hash")).is_file():
                    if h.read_text().strip() != digest:
                        raise ValueError(f"{h.name} is not the trace's")
                    if f.stat().st_mtime_ns > h.stat().st_mtime_ns:
                        raise ValueError(f"{h.name} is older than {f.name}")
        except (OSError, ValueError, UnicodeDecodeError):
            out.append(trace)
    return out


def forget(lib: Path, trace: Path) -> None:
    """Drop ``trace`` (a module's, under ``lib``) and the `.hash` file of each of the module's outputs, in ``lib`` and in
    the build's `ir/`. A restore from Lake's cache that finds an output in place keeps it and its `.hash` as they are, so
    a stale `.hash` left beside the trace would outlive the rebuild."""
    rel = trace.relative_to(lib).with_suffix("")
    for d in (lib, lib.parent.parent / "ir"):
        for h in (d / rel).parent.glob(f"{glob.escape(rel.name)}.*.hash"):
            h.unlink(missing_ok=True)
    trace.unlink(missing_ok=True)


def orphaned(lib: Path) -> dict[str, list[Path]]:
    """The `.hash` files under ``lib`` (a `.lake/build/lib/lean`) and the build's `ir/` whose module has no trace there,
    by module: what a `repair` that dropped only a trace left. Lake rebuilds or restores such a module, but a `.ltar`
    restore keeps an output's `.hash` when it finds one, so the old hash would come back with a trace beside new bytes."""
    out: dict[str, list[Path]] = {}
    for d in (lib, lib.parent.parent / "ir"):
        for h in sorted(d.rglob("*.hash")) if d.is_dir() else []:
            rel = h.parent.relative_to(d) / h.name.split(".")[0]
            if not lib.joinpath(rel).with_suffix(".trace").exists():
                out.setdefault(str(rel), []).append(h)
    return dict(sorted(out.items()))


#: Lean refusing an import that declares what an earlier import already did: (the refused module, the earlier one)
CLASH = re.compile(r"import (\S+) failed, environment already contains '[^']*' from (\S+)")


def clashed(tree: Path, pkgs: list[str], text: str) -> list[str]:
    """The modules of ``tree``'s ``pkgs`` that an import clash in ``text`` names, each `forget`-ten so the next build
    there rebuilds it against the outputs in place."""
    out = []
    for m in dict.fromkeys(n for pair in CLASH.findall(text) for n in pair):
        for p in pkgs:
            lib = tree / p / ".lake" / "build" / "lib" / "lean"
            if (t := lib.joinpath(*m.split(".")).with_suffix(".trace")).is_file():
                forget(lib, t)
                out.append(m)
                break
    return out


def repair(tree: Path, pkgs: list[str], links: dict[str, str], left: dict | None, log=print) -> None:
    """Make ``tree``'s copy of ``pkgs`` one Lake builds from, whatever a run left there: each module whose outputs `damaged`
    finds loses its trace and `.hash` files (`forget`), so Lake rebuilds it, and a module with no trace loses its `.hash`
    files (`orphaned`). Where a run ended mid-way (``left``, its `DIRTY`), also each package whose
    fetched dependencies hold a damaged module or a checkout that is not its manifest's, and the one whose dependencies it
    was taking (`phase` `deps`), has them dropped, so they are taken again."""
    if left is not None:
        what = f"{left.get('package', '?')} in phase {left.get('phase', '?')}" if left else "an unreadable record"
        log(f"lean-records: {tree} was left mid-way by {left.get('run', 'a run')} (pid {left.get('pid', '?')}, "
            f"{what}, {left.get('at', '?')}"
            + (f", check.py exit {left['killed']}" if "killed" in left else "") + "): checking it before it is built")
    for p in pkgs:
        bad = damaged(lib := tree / p / ".lake" / "build" / "lib" / "lean")
        for t in bad:
            forget(lib, t)
        if bad:
            names = [str(t.relative_to(tree / p / ".lake" / "build" / "lib" / "lean").with_suffix("")) for t in bad]
            log(f"lean-records: {p}: {len(bad)} module(s) half-written by an earlier build, rebuilt: "
                f"{', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
        if lone := orphaned(lib):
            for h in (h for hs in lone.values() for h in hs):
                h.unlink(missing_ok=True)
            names = list(lone)
            log(f"lean-records: {p}: {len(names)} module(s) without a trace kept their .hash files, dropped: "
                f"{', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
        if left is None:
            continue
        why = "the run was taking them" if left.get("phase") == "deps" and left.get("package") == p else None
        packages = tree / p / ".lake" / "packages"
        for d in sorted(packages.iterdir()) if why is None and packages.is_dir() else []:
            if damaged(d / ".lake" / "build" / "lib" / "lean"):
                why = f"{d.name} holds a module half-written"
            elif (want := checkout(p, tree, d.name)) and (d / ".git").exists() and git_head(d) != want:
                why = f"{d.name} is not the checkout its manifest pins ({want[:12]})"
            if why:
                break
        if why and packages.exists():
            unlink(p, tree, links)
            log(f"lean-records: {p}: its fetched dependencies were dropped ({why}); they are taken again")


def checkout(pkg: str, tree: Path, name: str) -> str | None:
    """The revision ``tree/pkg``'s manifest pins for its fetched dependency ``name``."""
    try:
        doc = json.loads((tree / pkg / "lake-manifest.json").read_text())
    except (OSError, ValueError):
        return None
    return next((d.get("rev") for d in doc.get("packages", []) if d.get("name") == name and d.get("type") == "git"), None)


def git_head(d: Path) -> str | None:
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def records_tree() -> Path:
    """The tree of the Lean slot this process holds (LEAN_SLOT, `build/1` -> `build-1`), else `local`."""
    return LA.COMMON.cache_dir() / RECORDS / (os.environ.get(LEAN_SLOT.HELD) or "local").replace("/", "-")


def made(tree: Path) -> Path:
    """``tree``, made. A node may link a slot's tree onto a larger volume; when the link's target is gone, the target is made
    again, where `mkdir` on the link raises FileExistsError and fails every run handed that slot."""
    tree.resolve().mkdir(parents=True, exist_ok=True)
    return tree


#: `--records`' kept passes, beside its trees: `<inputs_key>.json` for each package whose records passed on those inputs, the
#: newest `PASSES_KEPT` of them
PASSES, PASSES_KEPT = "passes", 4000
PASS_SCHEMA = "lean-records-pass/v2"
#: `--records`' facts cache, beside its trees and shared by every slot's: each audited package's sealed facts document
#: (`check.py --facts-cache`), named by its build's content address (`facts_key`), outside every slot tree's build roots
FACTS_CACHE = "facts"
SELF = "tools/verity/lean/lean_changed.py"
LOCK = "lean-audit.json"


def tool_inputs(files: list[str]) -> list[str]:
    """The files besides the packages' that decide a records run: the auditor's (`lean_audit.tool_files`) and this one."""
    return sorted({*LA.tool_files(files), SELF})


def closure_inputs(pkg: str, files: list[str], read=read_here) -> list[str]:
    """Every file of ``files`` a records run of ``pkg`` reads: the packages of its `closure`, and `tool_inputs`."""
    return sorted({*(f for q in closure(pkg, read) for f in LA.own_files(q, files)), *tool_inputs(files)})


def inputs_key(pkg: str, files: list[str], read=read_here, locks: dict[str, bytes | None] | None = None) -> str:
    """``pkg`` and what a records run of it reads (`closure_inputs`), each file by its path and content: ``read`` gives a path's
    bytes (None: absent), and ``locks`` ({path: bytes}) stands in for some, the record a run wrote in place of the committed one.
    Two packages can read the same files (`Security` and `Security/Properties`, each in the other's `closure`), and one's pass
    says nothing of the other's records. The same files read from a commit's objects (`keys_at`) give the same key."""
    locks = locks or {}

    def seen(f: str) -> bytes | None:
        return locks[f] if f in locks else read(f)

    got = {f: "-" if (b := seen(f)) is None else hashlib.sha256(b).hexdigest() for f in closure_inputs(pkg, files, seen)}
    return hashlib.sha256(json.dumps([PASS_SCHEMA, pkg, sorted(got.items())]).encode()).hexdigest()


def passes_dir() -> Path:
    return LA.COMMON.cache_dir() / RECORDS / PASSES


def facts_dir() -> Path:
    return LA.COMMON.cache_dir() / RECORDS / FACTS_CACHE


def git(*args: str) -> str | None:
    """git's output in this checkout, None when it fails (no checkout, no such commit)."""
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(ROOT), *args], capture_output=True)
    return r.stdout.decode(errors="replace").strip() if r.returncode == 0 else None


#: how long `stored` waits on the store's answer before the run builds instead
STORE_WAIT_S = 60


def stored(pkg: str, key: str, log=print) -> dict | None:
    """{`run`, `commit`} of a `FAST_KIND` result that passed ``pkg``'s records on inputs ``key`` (its `passes`), by `research
    data select`, which reads the store's index where this machine has one and its catalog otherwise (on a node, the
    results of that node's runs); None when the store holds none or can't be asked (said), and the run then builds: a
    lookup never fails open."""
    research = shutil.which("research")
    if research is None:
        log(f"lean-records: {pkg}: no `research` on PATH, so the store isn't asked for a pass on these inputs")
        return None
    try:
        r = subprocess.run([research, "data", "select", "--kind", FAST_KIND, "--where", f"passes.{key}={pkg}", "--json"],
                           capture_output=True, text=True, timeout=STORE_WAIT_S, stdin=subprocess.DEVNULL)
        if r.returncode:
            log(f"lean-records: {pkg}: the store couldn't be asked for a pass on these inputs (`research data select` exit "
                f"{r.returncode}: {r.stderr.strip()[-300:]})")
            return None
        rows = json.loads(r.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        log(f"lean-records: {pkg}: the store couldn't be asked for a pass on these inputs ({e!r})")
        return None
    for row in rows if isinstance(rows, list) else []:
        meta = (row.get("manifest") or {}).get("meta") if isinstance(row, dict) else None
        if isinstance(meta, dict) and (meta.get("validation") or {}).get("status") == "passed" and meta.get("run_id") \
                and (meta.get("passes") or {}).get(key) == pkg:
            return {"run": meta["run_id"], "commit": meta.get("commit")}
    return None


def kept(pkg: str, files: list[str], read=read_here, store: bool = False, log=print) -> tuple[str, dict | None] | None:
    """(why ``pkg``'s records need no run on these inputs, {`run`, `commit`} of the pass that says so or None), or None: a
    records run passed on the same inputs (`passes_dir`; with ``store``, also one the store holds, `stored`, which says
    through ``log`` when it can't be asked), or the Lean check's audit did (its kept pass, `lean_audit.key`). ``read``: the
    inputs, this checkout's by default."""
    key = inputs_key(pkg, files, read)
    try:
        doc = json.loads((passes_dir() / f"{key}.json").read_text())
        src = {"run": doc.get("run"), "commit": doc.get("commit")}
        return f"its records passed on the same Lean inputs in {src['run'] or 'a run'} at {str(src['commit'])[:12]}", src
    except (OSError, ValueError, AttributeError):
        pass
    if read is read_here and (LA.COMMON.cache_dir() / "lean-audit" / f"{LA.key(pkg, files, LA.FLAGS)}.json").is_file():
        return "the Lean check's audit, kernel replay included, passed on the same inputs here", None
    if store and (src := stored(pkg, key, log)):
        return (f"its records passed on the same Lean inputs in {src['run']} at {str(src['commit'])[:12]}, a result the "
                "store holds"), src
    return None


def remember(pkg: str, keys: list[str]) -> None:
    """Keep that ``pkg``'s records passed on each inputs key of ``keys``; the oldest passes beyond `PASSES_KEPT` go."""
    d = passes_dir()
    d.mkdir(parents=True, exist_ok=True)
    doc = {"schema": PASS_SCHEMA, "package": pkg, "run": os.environ.get("RESEARCH_RUN_ID"), "commit": git("rev-parse", "HEAD"),
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for k in keys:
        tmp = d / f".{os.getpid()}.tmp"
        tmp.write_text(json.dumps(doc) + "\n")
        os.replace(tmp, d / f"{k}.json")
    kept_ = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for p in kept_[:max(0, len(kept_) - PASSES_KEPT)]:
        p.unlink(missing_ok=True)


def at_commit(commit: str, prefixes: list[str], repo: Path | None = None) -> tuple[list[str], dict[str, str]] | None:
    """(the files under ``prefixes`` at ``commit``, {path: blob id}), from git in ``repo`` (this checkout by default); None
    when git can't say."""
    out = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo or ROOT), "ls-tree", "-r", "-z", commit, "--",
                          *prefixes], capture_output=True)
    if out.returncode:
        return None
    blobs = {}
    for e in out.stdout.split(b"\0"):
        if e and (meta := e.split(b"\t", 1)) and len(meta) == 2 and (m := meta[0].split()) and m[1] == b"blob":
            blobs[meta[1].decode()] = m[2].decode()
    return sorted(blobs), blobs


def blob_reader(blobs: dict[str, str], repo: Path | None = None):
    """``read`` for `inputs_key` over a commit's ``blobs``, all read by one `git cat-file --batch` in ``repo``."""
    ids = sorted(set(blobs.values()))
    out = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo or ROOT), "cat-file", "--batch"],
                         input="".join(f"{i}\n" for i in ids).encode(), capture_output=True, check=True).stdout
    data, n = {}, 0
    for i in ids:
        head_end = out.index(b"\n", n)
        size = int(out[n:head_end].split()[2])
        data[i] = out[head_end + 1:head_end + 1 + size]
        n = head_end + 1 + size + 1
    return lambda f: data.get(blobs[f]) if f in blobs else None


def lake_name(pkg: Path) -> str | None:
    """The Lake package name ``pkg``'s `lakefile.toml` declares (its top-level ``name``), or None without one."""
    f = pkg / "lakefile.toml"
    return tomllib.loads(f.read_text()).get("name") if f.is_file() else None


def matching(name: str, pkgs: list[str], root: Path | None = None) -> list[str]:
    """The packages of ``pkgs`` that ``name`` names: the one it is, else (with ``root``, the checkout they are in) the one
    whose Lake package is named ``name``, else each it is a part of."""
    name = name.rstrip("/")
    return ([p for p in pkgs if p == name]
            or ([p for p in pkgs if lake_name(root / p) == name] if root is not None else [])
            or [p for p in pkgs if name in p])


def keys_at(names: list[str], commit: str = "HEAD", repo: Path | None = None) -> dict[str, str]:
    """{package: its `inputs_key`} for each of ``names`` (a package's path, or a unique part of one, as `--records` takes it)
    at ``commit`` in the git repository ``repo`` (this checkout's by default), read from its objects alone: no checkout of it
    and no build, so a clone with no worktree serves. Equal to the key a records run of that commit keeps. ValueError when git
    can't list ``commit`` or a name matches no single package."""
    got = at_commit(commit, [], repo)
    if got is None:
        raise ValueError(f"git can't list {commit} in {repo or ROOT}")
    files, blobs = got
    pkgs = LA.packages(files)
    picked = []
    for n in names:
        if len(m := matching(n, pkgs)) != 1:
            raise ValueError(f"{n!r} matches {len(m)} packages at {commit}: {', '.join(m or pkgs)}")
        picked.append(m[0])
    meta = blob_reader({f: b for f, b in blobs.items() if f.endswith(("/lake-manifest.json", f"/{LOCK}"))}, repo)
    need = {f for p in picked for f in closure_inputs(p, files, meta)}
    read = blob_reader({f: blobs[f] for f in need if f in blobs}, repo)
    return {p: inputs_key(p, files, read) for p in picked}


#: the refs whose history is `main`'s, the first that exists
MAIN = ("refs/remotes/origin/main", "refs/heads/main")
#: how many merges back `merged` looks along the first parents for the records run they all merged from
MERGES_BACK = 8
_MERGE = None


def _merge():
    global _MERGE
    if _MERGE is None:
        _MERGE = _module(Path(__file__).resolve().parent / "merge.py", "lean_merge_for_lean_changed")
    return _MERGE


def merged(pkg: str, files: list[str]) -> str | None:
    """Why this checkout's records of ``pkg`` need no run although its inputs are new, or None: HEAD merges a commit of `main`
    into a commit whose records passed (`kept`), possibly through further such merges, and each merge takes every Lean input
    of ``pkg`` from one side whole, its locks aside, which are `merge.py`'s merge of the two sides. The lock is then a candidate,
    never a verdict: the Lean check of `next` rebuilds every record and regenerates one that differs."""
    prefixes = sorted({*closure(pkg), LA.LEAN_TOOL.rstrip("/"), *tool_inputs(files)})
    if git("status", "--porcelain", "--untracked-files=no", "--", *prefixes) != "":
        return None
    main = next((r for r in MAIN if git("rev-parse", "-q", "--verify", r) is not None), None)
    c = git("rev-parse", "HEAD")
    if main is None or c is None:
        return None
    took = []
    for _ in range(MERGES_BACK):
        parents = (git("rev-parse", f"{c}^@") or "").split()
        if len(parents) != 2 or git("merge-base", "--is-ancestor", parents[1], main) is None:
            return None
        b, m = parents
        base = git("merge-base", b, m)
        sides = [at_commit(x, prefixes) for x in (c, b, m)]
        if base is None or None in sides:
            return None
        (fc, hc), (fb, hb), (fm, hm) = sides
        read = {*closure_inputs(pkg, fc), *closure_inputs(pkg, fb), *closure_inputs(pkg, fm)}
        for f in sorted(read):
            x, y, z = hc.get(f), hb.get(f), hm.get(f)
            if x == y:
                continue
            if f.rsplit("/", 1)[-1] == LOCK:
                try:
                    load = _merge().load
                    sides_ = [load_at(load, r, f) for r in (base, b, m)]
                    got, why = _merge().merge(*sides_)
                    if why or got != load_at(load, c, f):
                        return None
                except (OSError, ValueError, subprocess.CalledProcessError):
                    return None
            elif x != z:
                return None
        took.append(m[:12])
        found = kept(pkg, fb, blob_reader({f: hb[f] for f in closure_inputs(pkg, fb) if f in hb}))
        if hit := found and found[0]:
            return (f"HEAD merges main's {', '.join(took)} into {b[:12]}, where {hit}; every Lean input it changed is main's "
                    f"and its lock merge.py's merge: a candidate, which the Lean check of `next` rebuilds and compares (a record "
                    f"that differs is regenerated there)")
        c = b
    return None


def load_at(load, commit: str, f: str) -> dict:
    """``f``'s record at ``commit`` as `merge.load` reads it (empty when absent)."""
    text = git("show", f"{commit}:{f}")
    with tempfile.NamedTemporaryFile("w", suffix=".json") as t:
        t.write(text or "")
        t.flush()
        return load(Path(t.name))


def passed(p: str, files: list[str], committed: dict[str, bytes | None], new: dict[str, bytes], log=print) -> list[str]:
    """The inputs keys ``p``'s records passed on, each `remember`ed: its own lock read as check.py left it (``new``'s record
    when this run regenerated it, else the ``committed`` one), and each other lock this run regenerated read both as
    committed and as ``new``, since a package's records read nothing of another's lock; so the commit of every record the
    run kept finds it. ``committed`` holds each lock of the run's packages as the run found it, so a lock check.py rewrote in
    this checkout since is read as it was."""
    own = {f"{p}/{LOCK}": (committed | new)[f"{p}/{LOCK}"]}
    locks = [committed | own, committed | new]
    keys = list(dict.fromkeys(inputs_key(p, files, read_here, lk) for lk in locks))
    try:
        remember(p, keys)
    except OSError as e:
        log(f"lean-records: {p}: its pass could not be kept ({e!r}); the next run on these inputs runs it again")
    return keys


def reused(todo: list[str], files: list[str], log=print) -> dict[str, tuple[str, dict | None]]:
    """{package of ``todo``: (why its records need no run, {`run`, `commit`} of the pass that says so or None)}: `kept`, the
    store's included, else `merged`, each said."""
    out = {}
    for p in todo:
        try:
            got = kept(p, files, store=True, log=log) or ((why, None) if (why := merged(p, files)) else None)
        except (OSError, ValueError, subprocess.CalledProcessError) as e:
            log(f"lean-records: {p}: whether its records ran on these inputs could not be read ({e!r}); they run")
            got = None
        if got:
            out[p] = got
            log(f"lean-records: {p}: reused, nothing built: {got[0]}")
    return out


def label(out: Path) -> None:
    """Write ``FAST_KIND`` into the reports check.py left under ``out``: `audit.json`, its packages, and each `report.json`."""
    for p in [out / "audit.json", *sorted(out.glob("audit-*/report.json"))]:
        if not p.is_file():
            continue
        doc = json.loads(p.read_text())
        doc["kind"] = FAST_KIND
        for rep in doc.get("packages", []):
            rep["kind"] = FAST_KIND
        p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


def write_result(run_dir: Path, run_id: str, pkg: str, outcome: str, rc: int, out: Path, update: bool, path: str = "slot",
                 records: dict[str, dict] | None = None, passes: dict[str, str] | None = None) -> None:
    """A recorded run's `result.json`, of kind ``FAST_KIND``: what `research merge` and `research queue`'s records gate read
    of the Attempt. ``outcome`` is one of `OUTCOMES`, and its `validation.status` is `passed` exactly when it is `PASS` or
    `REGENERATED`; ``records`` is each package's (`verdicts`), and ``passes`` {inputs key: package} each key a package's
    records passed on in this run (`passed`), which the store indexes as `passes.<key>` (`stored`). ``path``: `slot` (a
    `build` slot's tree), `dev-cache` (`dev_records`) or `reused` (no package ran: `reused`)."""
    rel = os.path.relpath(out, run_dir)
    ran = ("nothing ran: every package's records passed on the same Lean inputs or are a merge's candidate" if path == "reused"
           else "nothing ran" if outcome == REFUSED else f"check.py's first failing exit {rc} on the {path} path" if rc
           else f"{outcome} on the {path} path")
    doc = {"schema": "research/result/v0.1", "kind": FAST_KIND, "run_id": run_id, "commit": git("rev-parse", "HEAD"),
           "outcome": outcome, "workload_fingerprint": {"package": pkg, "update": update, "replay": False, "runs": False,
                                                        "path": path},
           "validation": {"status": "passed" if outcome in (PASS, REGENERATED) else "failed",
                          "checker": "tools/verity/lean/lean_changed.py --records",
                          "detail": f"{ran}, without the kernel replay or the tests: not a Lean verdict, and no merge accepts "
                                    "it"},
           "records": records or {}, "passes": passes or {},
           "artifacts": [] if rel.startswith("..") or not out.exists() else [rel]}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run_dir / "result.json")


def finish(pkg: str, outcome: str, rc: int, out: Path, update: bool, path: str, records: dict[str, dict], log=print,
           passes: dict[str, str] | None = None) -> int:
    """End a `--records` run: in a recorded run its `result.json` (`write_result`), and on every path its last line,
    `OUTCOME_LINE` and JSON (``outcome``, ``pkg``, ``records``, ``passes``); ``rc``, its exit status."""
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if run_dir and run_id:
        write_result(Path(run_dir), run_id, pkg, outcome, rc, out, update, path, records, passes)
    log(OUTCOME_LINE + json.dumps({"outcome": outcome, "package": pkg, "records": records, "passes": passes or {}},
                                  sort_keys=True))
    return rc


def differences(out: Path, base: Path, log=print) -> dict[str, Path] | None:
    """{each package check.py failed in its reports under ``out``, by its path from ``base``: the record it wrote that settles
    it}, when each such failure is a record difference only (check.py's report then names that record, `regenerated`, which
    is what `--update` writes); None when one is of another kind, or the reports can't be read (said)."""
    try:
        bad = [r for r in json.loads((out / "audit.json").read_text())["packages"] if r.get("failures")]
        if not all(isinstance(r.get("regenerated"), str) and Path(r["regenerated"]).is_file() for r in bad):
            return None
        return {os.path.relpath(Path(str(r["package"])).resolve(), base.resolve()): Path(r["regenerated"]) for r in bad}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        log(f"lean-records: check.py's reports under {out} can't be read ({e!r}), so its failures count as failures of "
            "another kind")
        return None


def keep(q: str, pkg: str, src: Path, out: Path) -> bytes:
    """Keep ``src``, the record a run wrote for package ``q``, under ``out``'s `record/<q>/` (PKG's also as
    `record/lean-audit.json`, the path `research queue`'s records gate reads for PKG); its bytes."""
    kept = out / "record" / q / LOCK
    for k in [kept, *([out / "record" / LOCK] if q == pkg else [])]:
        k.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, k)
    return kept.read_bytes()


def unkeep(q: str, pkg: str, out: Path) -> None:
    """Remove what `keep` kept of ``q``'s record under ``out``."""
    for k in [out / "record" / q / LOCK, *([out / "record" / LOCK] if q == pkg else [])]:
        k.unlink(missing_ok=True)


def settled(p: str, pkg: str, got: int, mine_out: Path, base: Path, out: Path, new: dict[str, bytes], own: bool = False,
            log=print) -> str:
    """``p``'s outcome once check.py on it exited ``got``, its reports under ``mine_out`` and its packages under ``base``,
    ``own`` saying whether `checked` kept the record its `--update` wrote (in ``new``): `PASS`; `REGENERATED` when ``own``
    or every failure in its reports is a record difference (`differences`), each such record then kept (`keep`) and put in
    ``new``; `FAILED` otherwise, with ``own``'s record dropped again, since `record/` holds only records the gate may take."""
    diff = {} if not got else None if AUDIT.killed(got) else differences(mine_out, base, log) or None
    if diff is None:
        if p != pkg:
            log(f"lean-records: {p} requires {pkg}, and its records fail on this build, not only on record differences, "
                f"which take a person (reports in {mine_out})")
        if own:
            unkeep(p, pkg, out)
            new.pop(f"{p}/{LOCK}", None)
            log(f"lean-records: {p}: its records fail on this build, not only on record differences, so the record check.py "
                "--update wrote is not kept")
        return FAILED
    for q, src in diff.items():
        if not (own and q == p):
            new[f"{q}/{LOCK}"] = keep(q, pkg, src, out)
    for q in {**({p: None} if own else {}), **diff}:
        log(f"lean-records: {q}: the regenerated record is {out / 'record' / q / LOCK}; commit it over {q}/{LOCK}")
    if diff:
        log(f"lean-records: {p}: every failure of its records is a record difference, which the kept record"
            f"{'s settle' if len(diff) > 1 else ' settles'}: not a failure of the run")
    return REGENERATED if own or diff else PASS


def verdicts(todo: dict[str, str], new: dict[str, bytes], out: Path | None, base: Path, files: list[str],
             reused_: dict[str, tuple[str, dict | None]] | None = None, log=print,
             committed: dict[str, bytes | None] | None = None) -> dict[str, dict]:
    """`result.json`'s `records`: {package: {`outcome`, `inputs_key` (this checkout's as the run found it, its locks read as
    ``committed`` holds them), and, when this run kept its record, `record` (its path in the run's files, else under ``out``)
    and `record_sha256`; `reused` (why) for one that needed no run, and `reused_from` ({`run`, `commit`}) when a records
    run's pass says so}}, for each package of ``todo`` ({package: its outcome}), each of ``reused_`` (`PASS`), and each
    package check.py audited beside one, its outcome read from its report under ``out`` (`PASS` when it failed nothing,
    `REGENERATED` when it failed only on record differences, which check.py's `regenerated` says, and this run kept its
    record, `FAILED` otherwise), the worse when it was audited twice; with no ``out`` (nothing ran), none of those. A record
    kept of a package that ends `FAILED` (audited beside one that settled it, and failing otherwise beside another) is
    removed: `record/` holds only `REGENERATED` packages'."""
    got = {**{p: PASS for p in reused_ or {}}, **todo}
    for a in [out / "audit.json", *sorted(out.glob("requirer-*/audit.json"))] if out else []:
        try:
            reps = json.loads(a.read_text()).get("packages") or []
        except (OSError, ValueError, AttributeError) as e:
            log(f"lean-records: {a} can't be read ({e!r}); the packages it reports beside this run's are left out of its "
                "records")
            continue
        for r in reps if isinstance(reps, list) else []:
            q = os.path.relpath(Path(str(r.get("package"))).resolve(), base.resolve()) if isinstance(r, dict) else ".."
            if q in todo or q.startswith(".."):
                continue
            o = PASS if not r.get("failures") else REGENERATED if r.get("regenerated") and f"{q}/{LOCK}" in new else FAILED
            got[q] = max(got.get(q, PASS), o, key=OUTCOMES.index)
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    recs: dict[str, dict] = {}
    for q, o in got.items():
        try:
            key = inputs_key(q, files, read_here, committed)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            key = None
            log(f"lean-records: {q}: its inputs key could not be read ({e!r})")
        rec: dict = {"outcome": o, "inputs_key": key}
        if out and f"{q}/{LOCK}" in new:
            kept = out / "record" / q / LOCK
            if o == FAILED:
                kept.unlink(missing_ok=True)
                log(f"lean-records: {q}: its records fail beside another package, not only on record differences, so its "
                    f"record is not kept")
            else:
                rec |= {"record": os.path.relpath(kept, run_dir) if run_dir else str(kept),
                        "record_sha256": hashlib.sha256(new[f"{q}/{LOCK}"]).hexdigest()}
        if q in (reused_ or {}):
            why, src = reused_[q]
            rec |= {"reused": why, **({"reused_from": src} if src else {})}
        recs[q] = rec
    return recs


def as_found(todo: list[str]) -> dict[str, bytes | None]:
    """{lock: its bytes in this checkout now} for each package of the `closure` of each of ``todo``: what a run's keys read
    of the locks, which check.py `--update` may rewrite in this checkout as the run goes (the `--dev-cache` path)."""
    return {f"{q}/{LOCK}": read_here(f"{q}/{LOCK}") for q in dict.fromkeys(c for t in todo for c in closure(t))}


def overall(records: dict[str, dict]) -> str:
    """A run's outcome: its packages' worst."""
    return max((r["outcome"] for r in records.values()), key=OUTCOMES.index, default=PASS)


def audit_records(name: str, deps: Path, tree: Path | None = None, update: bool = False, moved: Path | None = None,
                  out: Path | None = None, log=print, dev_cache: bool = False, rebuild: bool = False) -> int:
    """`--records`: `check.py --build --no-replay --no-runs` on the package ``name`` names, in ``tree`` (default: the tree of the
    `build` slot it takes; any other must be under `RECORDS`), or `dev_records` where the node's Lean slots grant no `build`
    slot or ``dev_cache`` asks for it. Its outcome is its packages' worst (`settled`, `verdicts`), and `finish` says it: it
    exits 0 on `PASS` and on `REGENERATED`, check.py's first failing status on `FAILED`, and 2 on `REFUSED`, when ``tree`` is
    not under `RECORDS`, ``name`` matches no single package, the memory refusal stops it or another run holds the tree past
    `TREE_WAIT_S`. A package whose records need no run (`reused`) is skipped, and with none left nothing runs and no slot is
    taken; ``rebuild`` and ``moved`` run them all. Each package whose records passed or were regenerated is kept (`passed`).
    In a run, the module versions its passing builds added to the node's artifact cache (`LakeArtifacts.put`) go into ``out``
    as one bundle for other nodes (`LakeArtifacts.bundle_into`), as a recorded check's do."""
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    out = (out or (Path(run_dir) / "lean-audit" if run_dir else Path(tempfile.mkdtemp(prefix="lean-records-")))).resolve()

    def refused(why: str | None, pkg: str = name) -> int:
        if why:
            log(f"lean-records: {why}")
        return finish(pkg, REFUSED, 2, out, update, "slot", {}, log)

    base = (LA.COMMON.cache_dir() / RECORDS).resolve()
    if tree is not None and base not in tree.resolve().parents:
        return refused(f"{tree} is not under {base}: the fast path builds only on its own trees, never on check's "
                       "(lean-changed), whose dependencies are links to the audit's")
    files = LA.COMMON.tracked_files(ROOT)
    if files is None:
        return refused("the tree's files cannot be listed")
    pkgs = LA.packages(files)
    match = matching(name, pkgs, ROOT)
    if len(match) != 1:
        return refused(f"{name!r} matches {len(match)} packages: {', '.join(match or pkgs)}")
    pkg = match[0]
    todo, rc = [pkg, *requirers(pkg, pkgs)], 0
    skip = {} if rebuild or moved else reused(todo, files, log)
    if not (todo := [p for p in todo if p not in skip]):
        log(f"lean-records: nothing to build: the records of {', '.join(skip)} need no run (--rebuild runs them anyway)")
        return finish(pkg, PASS, 0, out, update, "reused", verdicts({}, {}, None, ROOT, files, skip, log), log)
    here = ROOT / pkg
    audited = [here, *(q for _, q in AUDIT.provers(here, AUDIT.load(here)))] if (here / AUDIT.POLICY).is_file() else [here]
    if why := AUDIT.memory_refusal(audited, AUDIT.memory_gib()):
        return refused(why, pkg)
    if dev_cache:
        return dev_records(pkg, todo, update, moved, out, "--dev-cache asks for it", log, files, skip)
    os.environ.pop(LEAN_SLOT.HELD, None)
    try:
        slot = LEAN_SLOT.hold(FAST_POOL, or_any=False)
    except LEAN_SLOT.Refused as e:
        return dev_records(pkg, todo, update, moved, out, str(e), log, files, skip)
    artifacts = LA.LakeArtifacts((deps.parent / "lean-artifacts").resolve())
    try:
        tree = tree or records_tree()
        log(f"lean-records: taking the slot path: check.py on {tree}, which keeps its build between runs")
        made(tree)
        with open(tree / ".lock", "w") as held:
            if not hold_tree(held, tree, log):
                return refused(None, pkg)
            links = read_links(tree)
            committed = as_found(todo)
            new: dict[str, bytes] = {}
            outcomes: dict[str, str] = {}
            left, killed, ok = left_by(tree), {}, []
            env, fds = {**os.environ, "LEAN_NUM_THREADS": str(LA.COMMON.cores())}, () if slot is None else (slot,)
            for p in todo:
                t0 = time.monotonic()
                mark(tree, package=p, phase="sync")
                n = sync(closure(p), files, tree)
                for f in (f"{q}/{LOCK}" for q in closure(p)):
                    if f in new:
                        (tree / f).write_bytes(new[f])
                if left is not None:
                    there = [q for q in [*(c for t in todo for c in closure(t)), left.get("package")]
                             if isinstance(q, str) and (tree / q).is_dir()]
                    repair(tree, list(dict.fromkeys(there)), links, left, log)
                    write_links(tree, links)
                    left = None
                else:
                    repair(tree, closure(p), links, None, log)
                mark(tree, package=p, phase="deps")
                for other in [q for q in links if q not in todo and LA.deps_key(tree / p)]:
                    unlink(other, tree, links)
                if not link(p, tree, deps, links, log, copy=True) and (k := LA.deps_key(tree / p)):
                    links[p] = k
                    log(f"lean-records: {p}: its dependencies are fetched cold, once for this tree")
                write_links(tree, links)
                mark(tree, package=p, phase="check")
                mine_out, up, since = reports(out, pkg, p), first_update(p, pkg, update, moved, log), time.time()
                got = subprocess.run(check_argv(tree / p, mine_out, up, moved, p == pkg, artifacts=artifacts.store,
                                                facts_cache=facts_dir()), cwd=tree, env=env, pass_fds=fds).returncode
                if again := not up and regenerates(p, pkg, got, update, mine_out, tree / p, log):
                    up, got = True, subprocess.run(check_argv(tree / p, mine_out, True, moved, p == pkg,
                                                              artifacts=artifacts.store, facts_cache=facts_dir()),
                                                   cwd=tree, env=env, pass_fds=fds).returncode
                if AUDIT.killed(got):
                    killed[p] = got
                label(mine_out)
                log(f"lean-records: {p}: check.py{' --update' if again else ''} exit {got} in {time.monotonic() - t0:.0f} s on "
                    f"{tree} ({n} files changed there), without the kernel replay or the tests: a {FAST_KIND} result, which "
                    f"check and research merge refuse; reports in {mine_out}")
                # Keep the regenerated record now: the next requirer iteration's `sync` restores the committed one over
                # it, and a recorded run never writes ROOT (the commit's shared source tree), so without this copy the
                # record a lane must commit survives nowhere the run keeps.
                if (b := checked(p, pkg, got, tree / p / LOCK, committed[f"{p}/{LOCK}"], up, update, out, log)) is not None:
                    new[f"{p}/{LOCK}"] = b
                outcomes[p] = settled(p, pkg, got, mine_out, tree, out, new, b is not None, log)
                if b is not None and outcomes[p] != FAILED:
                    write_back(p, tree / p / LOCK, log)
                if got == 0:
                    try:
                        log(f"lean-records: {p}: {artifacts.put({tree / p: report(mine_out, tree / p)}, since, log)} "
                            "module version(s) new in the artifact cache")
                    except (OSError, ValueError) as e:
                        log(f"lean-records: {p}: the artifact cache was not written: {e}")
                if outcomes[p] == FAILED:
                    rc = rc or got
                else:
                    ok.append((p, got))
            passes = {k: p for p, got in ok for k in passed(p, files, committed if got == 0 else committed | new, new, log)}
            if killed:
                p, got = next(iter(killed.items()))
                mark(tree, package=p, phase="check", killed=got)
                log(f"lean-records: check.py on {', '.join(killed)} was killed (exit {got}): the next run on {tree} checks "
                    "it before it is built")
            else:
                (tree / DIRTY).unlink(missing_ok=True)
            if run_dir and artifacts.added and out.is_relative_to(Path(run_dir).resolve()):
                try:
                    dest, n = artifacts.bundle_into(out)
                    log(f"lean-records: {n} module version(s) it added bundled for other nodes ({dest.name})")
                except OSError as e:
                    log(f"lean-records: nothing bundled for other nodes: {e}")
            records = verdicts(outcomes, new, out, tree, files, skip, log, committed)
            outcome = overall(records)
            return finish(pkg, outcome, (rc or 1) if outcome == FAILED else 0, out, update, "slot", records, log, passes)
    finally:
        if slot is not None:
            os.close(slot)
            os.environ.pop(LEAN_SLOT.HELD, None)


def dev_records(pkg: str, todo: list[str], update: bool, moved: Path | None, out: Path, why: str, log=print,
                files: list[str] | None = None, skip: dict[str, tuple[str, dict | None]] | None = None) -> int:
    """`--records` where the node's Lean slots grant no `build` slot (``why``): `check.py --dev-cache --build --no-replay
    --no-runs` on this checkout's ``todo`` (PKG, then its requirers; ``skip`` those `reused` left out), each build restored
    from the published cache, with the same reports, record keep, outcome, exit status and result kind as the slot path;
    `REFUSED` (exit 2) in a tree `research run` shipped (`READY.json`), which every run of its commit shares and check.py
    would build in."""
    if (ROOT / "READY.json").is_file():
        log(f"lean-records: {why}; the --dev-cache path builds in this checkout, which `research run` shipped and every run "
            "of its commit shares: nothing ran; rerun with --cwd clone")
        return finish(pkg, REFUSED, 2, out, update, "dev-cache", {}, log)
    log(f"lean-records: {why}; taking the --dev-cache path: check.py --dev-cache on {ROOT}, each build restored from the "
        "published cache")
    committed = as_found(todo)
    new: dict[str, bytes] = {}
    outcomes: dict[str, str] = {}
    rc, ok, passes = 0, [], {}
    for p in todo:
        t0 = time.monotonic()
        mine_out, up = reports(out, pkg, p), first_update(p, pkg, update, moved, log)
        got = subprocess.run(check_argv(ROOT / p, mine_out, up, moved, p == pkg, dev=True, facts_cache=facts_dir()),
                             cwd=ROOT).returncode
        if again := not up and regenerates(p, pkg, got, update, mine_out, ROOT / p, log):
            up, got = True, subprocess.run(check_argv(ROOT / p, mine_out, True, moved, p == pkg, dev=True,
                                                      facts_cache=facts_dir()), cwd=ROOT).returncode
        label(mine_out)
        log(f"lean-records: {p}: check.py --dev-cache{' --update' if again else ''} exit {got} in {time.monotonic() - t0:.0f} s "
            f"on {ROOT}, without the kernel replay or the tests: a {FAST_KIND} result, which check and research merge refuse; "
            f"reports in {mine_out}")
        if (b := checked(p, pkg, got, ROOT / p / LOCK, committed[f"{p}/{LOCK}"], up, update, out, log)) is not None:
            new[f"{p}/{LOCK}"] = b
        outcomes[p] = settled(p, pkg, got, mine_out, ROOT, out, new, b is not None, log)
        if outcomes[p] == FAILED:
            rc = rc or got
        else:
            ok.append((p, got))
    for p, got in ok if files is not None else []:
        passes |= dict.fromkeys(passed(p, files, committed if got == 0 else committed | new, new, log), p)
    records = verdicts(outcomes, new, out, ROOT, files, skip, log, committed) if files is not None else \
        {p: {"outcome": o, "inputs_key": None} for p, o in outcomes.items()}
    outcome = overall(records)
    return finish(pkg, outcome, (rc or 1) if outcome == FAILED else 0, out, update, "dev-cache", records, log, passes)


#: what lists a machine's Lean pools, which `--try` names when it is refused a `build` slot and no machine's could be read
SLOTS_STATUS = "python3 tools/verity/check/lean_slot.py --status"


def slots_there(ssh: list[str]) -> str:
    """The Lean slots file (LEAN_SLOT.SLOTS_PATH, whatever LEAN_SLOTS says here) of the machine ``ssh`` reaches (its PEERS
    line's ssh options and destination), as `slot.py` reaches a twin; OSError or a SubprocessError when it can't be read."""
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", *ssh, f"cat {shlex.quote(LEAN_SLOT.SLOTS_PATH)}"],
                       capture_output=True, text=True, timeout=LEAN_SLOT.SLOT.TWIN_SSH_S)
    if r.returncode:
        raise OSError(f"exit {r.returncode}: {(r.stderr.strip().splitlines() or ['no output'])[-1]}")
    return r.stdout


def pool_machines(pool: str, peers: dict[str, list[str]] | None = None, read=None) -> tuple[list[str], dict[str, str]]:
    """The machines `slot.py`'s PEERS lists whose Lean slots file grants ``pool`` a slot, each read at once with ``read``
    (`slots_there`), and why each it couldn't read wasn't (this machine's own line among them, where its key doesn't reach
    itself)."""
    peers = LEAN_SLOT.SLOT.peers() if peers is None else peers
    read = read or slots_there

    def one(name: str) -> tuple[int, str | None]:
        try:
            return LEAN_SLOT.parse(read(peers[name])).get(pool, 0), None
        except (OSError, subprocess.SubprocessError) as e:
            return 0, str(e) or type(e).__name__

    with ThreadPoolExecutor(max_workers=max(1, len(peers))) as ex:
        got = dict(zip(peers, ex.map(one, peers)))
    return sorted(n for n, (k, _) in got.items() if k > 0), {n: why for n, (_, why) in sorted(got.items()) if why}


def pool_hint(pool: str, file: str) -> str:
    """Where ``file``'s `--try` can take a slot of ``pool``: the machines `pool_machines` finds, else how to list a machine's
    pools."""
    have, unread = pool_machines(pool)
    if have:
        return (f"the machines whose Lean slots have `{pool}`: {', '.join(have)} (`verity lean try {file} --on MACHINE`)"
                + (f"; not read: {', '.join(unread)}" if unread else ""))
    return (f"no machine in {LEAN_SLOT.SLOT.PEERS} has `{pool}`" + (f" that could be read ({len(unread)} not)" if unread else "")
            + f"; `{SLOTS_STATUS}` on a machine lists its Lean pools")


def owner(file: str, pkgs: list[str]) -> str | None:
    """The package whose directory holds ``file`` (a path from the repository's root), the deepest when packages nest."""
    return max((p for p in pkgs if file.startswith(p + "/")), key=len, default=None)


def module_of(pkg: str, file: str) -> str:
    """``file``'s module name in ``pkg``: its path from the package's `srcDir` (lakefile.toml or lakefile.lean), dotted."""
    src = "."
    if (toml := ROOT / pkg / "lakefile.toml").is_file():
        import tomllib
        src = tomllib.loads(toml.read_text()).get("srcDir", ".")
    elif (lean := ROOT / pkg / "lakefile.lean").is_file() and (m := re.search(r'srcDir\s*:=\s*"([^"]*)"', lean.read_text())):
        src = m.group(1)
    rel = os.path.relpath(ROOT / file, os.path.normpath(ROOT / pkg / src))
    return rel.removesuffix(".lean").replace(os.sep, ".")


def try_file(file: str, deps: Path, tree: Path | None = None, log=print, siblings: bool = False) -> int:
    """`--try`: on the tree of the `build` slot it takes (as `--records` does), with the closure of the package that owns
    ``file`` synced from this checkout and its fetched dependencies taken, `lake build +MODULE:deps` (its imports, of which
    Lake rebuilds only what changed) then `lake env lean` with the package's Lean options (`SIBLINGS.options`) on FILE's
    probed copy in that package's directory (`PROBE.probe`, whose line is the last), and with ``siblings`` the sibling check (`siblings.py`) on that copy's build; lean's exit
    status on FILE's own lines (the check's when it runs), or 2 when ``file`` is in no package, no `build` slot is granted,
    or the tree's dependencies can't be had. Never check.py."""
    file = file.removeprefix("./")
    base = (LA.COMMON.cache_dir() / RECORDS).resolve()
    if tree is not None and base not in tree.resolve().parents:
        log(f"lean-try: {tree} is not under {base}, where the trees `--records` and `--try` keep are")
        return 2
    files = LA.COMMON.tracked_files(ROOT)
    if files is None:
        log("lean-try: the tree's files cannot be listed")
        return 2
    pkg = owner(file, LA.packages(files))
    if pkg is None or not (ROOT / file).is_file() or not file.endswith(".lean"):
        log(f"lean-try: {file} is not a .lean file in a Lake package with a lean-audit.json")
        return 2
    lake = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(lake).is_file():
        log("lean-try: no `lake` here")
        return 2
    mod, rel = module_of(pkg, file), os.path.relpath(file, pkg)
    os.environ.pop(LEAN_SLOT.HELD, None)
    try:
        slot = LEAN_SLOT.hold(FAST_POOL, or_any=False)
    except LEAN_SLOT.Refused as e:
        log(f"lean-try: no `{FAST_POOL}` Lean slot here ({e}): nothing ran; {pool_hint(FAST_POOL, file)}")
        return 2
    try:
        tree = tree or records_tree()
        made(tree)
        with open(tree / ".lock", "w") as held:
            if not hold_tree(held, tree, log):
                return 2
            t0, links, left = time.monotonic(), read_links(tree), left_by(tree)
            mark(tree, package=pkg, phase="sync")
            n = sync(closure(pkg), files, tree)
            there = [q for q in [*closure(pkg), (left or {}).get("package")] if isinstance(q, str) and (tree / q).is_dir()]
            repair(tree, list(dict.fromkeys(there)), links, left, log)
            write_links(tree, links)
            mark(tree, package=pkg, phase="deps")
            sandbox = sandboxed()
            packages = tree / pkg / ".lake" / "packages"
            # dependencies a `--records` run did not take (not in LINKS) are kept as they are, and stay out of LINKS, whose
            # packages the next `--records` run on another package drops
            if not (packages.is_dir() and pkg not in links) and not link(pkg, tree, deps, links, log, copy=True):
                if sandbox:
                    log(f"lean-try: {pkg}: no dependencies to take on this node, and the sandbox can't fetch them: nothing "
                        f"ran; run `lean_changed.py --records {pkg}` here once")
                    (tree / DIRTY).unlink(missing_ok=True)
                    return 2
                links[pkg] = LA.deps_key(tree / pkg)
                log(f"lean-try: {pkg}: its dependencies are fetched cold, once for this tree")
            write_links(tree, links)
            mark(tree, package=pkg, phase="try")
            opts = SIBLINGS.options(tree / pkg)
            log(f"lean-try: {file} on {tree} ({n} files changed there, {time.monotonic() - t0:.0f} s; "
                f"{'' if sandbox else 'no sandbox here, '}lake build +{mod}:deps, then lake env lean "
                f"{''.join(o + ' ' for o in opts)}on {PROBE.COPIES / rel}, {rel} with the axiom probe after its last line"
                f"{', then the sibling check' if siblings else ''})")
            step, (rc, text) = f"lake build +{mod}:deps", build(pkg, tree, lake, [f"+{mod}:deps"], sandbox)
            plan = SIBLINGS.plan(tree / pkg, rel) if siblings and rc == 0 else None
            if isinstance(plan, str):
                log(f"lean-try: {file}: no sibling check: {plan}")
            checked, said = isinstance(plan, SIBLINGS.Plan), ""
            with SIBLINGS.scratch(tree / pkg) if checked else contextlib.nullcontext() as work:
                args = opts + (SIBLINGS.lean_args(tree / pkg, plan, work) if checked else [])
                at = os.path.relpath(SIBLINGS.source(plan, work), tree / pkg / PROBE.COPIES) if checked else rel
                if rc == 0:
                    step, (rc, said, text) = "lean", PROBE.probe(
                        tree / pkg, at, (ROOT / file).read_text(),
                        lambda copy: build(pkg, tree, lake, sandbox=sandbox, cmd=["env", "lean", *args, copy]))
                log(f"lean-try: {file}: {step} exit {rc} in {time.monotonic() - t0:.0f} s")
                if rc and (healed := clashed(tree, closure(pkg), text)):
                    log(f"lean-try: {file}: an import clash between {' and '.join(healed)}, which this tree can leave when "
                        "one was built against another version of a module both import: they lose their traces, so the "
                        "next try rebuilds them; a clash that comes back is in the source")
                if rc == 0 and checked:
                    def lean(a: list[str]) -> tuple[int, str, float, int]:
                        argv = [lake, "env", "lean", *a]
                        return SIBLINGS.run(in_sandbox(pkg, tree, argv) if sandbox else argv, tree / pkg)

                    rc, output, line = SIBLINGS.check(plan, work, lean)
                    if output:
                        log(output.rstrip("\n"))
                    log(f"lean-try: {file}: {line}")
            if said:
                log(f"lean-try: {file}: {said}")
            (tree / DIRTY).unlink(missing_ok=True)
            return rc
    finally:
        if slot is not None:
            os.close(slot)
            os.environ.pop(LEAN_SLOT.HELD, None)


#: `--build`'s last line, then a JSON object, which the ready mark's preflight reads (`research.queue.LakeBuild`)
BUILD_LINE = "lean-build: "


def first_errors(text: str, pkg: str, tracked: set[str]) -> dict[str, str]:
    """{file: its first Lean error line} of Lake's output ``text`` for a build in ``pkg``, each file a tracked path from the
    repository's root (Lake names it from ``pkg``'s directory, or from the package of its `closure` that holds it)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not (m := ERROR.match(line)):
            continue
        rel = m.group(1)
        f = next((c for c in [os.path.normpath(os.path.join(pkg, rel)), *(f"{q}/{rel}" for q in closure(pkg))] if c in tracked), rel)
        out.setdefault(f, line.strip())
    return out


def build_files(files: list[str], deps: Path, tree: Path | None = None, log=print, wait: bool = True) -> int:
    """`--build`: the ready mark's preflight's Lean step (`research.queue.Preflight`). On the tree of the `build` slot it takes,
    as `--try` does (slot, tree lock, sync, `repair` and dependencies), `lake build +MODULE...` of ``files``' modules (paths
    from the repository's root) in each package owning one, those it requires by path first; a package whose dependencies
    this node doesn't keep is skipped, never fetched cold. Its last line is `BUILD_LINE` and a JSON object: `built` ({package:
    lake's exit status}), `failed` ({file: its first Lean error}, in a tracked `.lean` file; an import clash this tree left is
    healed as `--try` heals it and not counted) and `skipped` ({package, file or `*`: why it wasn't built}). Exit 0 when
    every build passed, 1 when one failed, 2 when none ran. Without Lean slots here (not a node) it builds nothing. Without
    ``wait`` (`--no-wait`) it builds nothing when every `build` slot is taken, rather than wait in line: the preflight then
    builds on its next node."""
    doc: dict[str, dict] = {"built": {}, "failed": {}, "skipped": {}}

    def done(rc: int) -> int:
        print(BUILD_LINE + json.dumps(doc, sort_keys=True), flush=True)
        return rc

    if not LEAN_SLOT.SLOTS.exists():
        doc["skipped"]["*"] = f"no Lean slots here ({LEAN_SLOT.SLOTS}): not a node that keeps a warm Lean tree"
        return done(2)
    tracked = LA.COMMON.tracked_files(ROOT)
    if tracked is None:
        doc["skipped"]["*"] = "the tree's files cannot be listed"
        return done(2)
    pkgs, by_pkg = LA.packages(tracked), {}
    for f in (f.removeprefix("./") for f in files):
        if (p := owner(f, pkgs)) is None or not f.endswith(".lean") or not (ROOT / f).is_file():
            doc["skipped"][f] = "not a .lean file in a Lake package with a lean-audit.json"
        else:
            by_pkg.setdefault(p, []).append(f)
    lake = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(lake).is_file():
        doc["skipped"]["*"] = "no `lake` here"
        return done(2)
    os.environ.pop(LEAN_SLOT.HELD, None)
    try:
        slot = LEAN_SLOT.hold(FAST_POOL, or_any=False, queue=wait)
    except LEAN_SLOT.Refused as e:
        doc["skipped"]["*"] = f"no `{FAST_POOL}` Lean slot here ({e})"
        return done(2)
    try:
        tree = tree or records_tree()
        made(tree)
        with open(tree / ".lock", "w") as held:
            if not hold_tree(held, tree, log):
                doc["skipped"]["*"] = f"another run held {tree}"
                return done(2)
            links, left, sandbox, have = read_links(tree), left_by(tree), sandboxed(), set(tracked)
            for pkg, fs in sorted(by_pkg.items(), key=lambda kv: (rank(kv[0]), kv[0])):
                t0 = time.monotonic()
                mark(tree, package=pkg, phase="sync")
                n = sync(closure(pkg), tracked, tree)
                there = [q for q in [*closure(pkg), (left or {}).get("package")] if isinstance(q, str) and (tree / q).is_dir()]
                repair(tree, list(dict.fromkeys(there)), links, left, log)
                left = None
                write_links(tree, links)
                mark(tree, package=pkg, phase="deps")
                packages = tree / pkg / ".lake" / "packages"
                if not (packages.is_dir() and pkg not in links) and not link(pkg, tree, deps, links, log, copy=True):
                    doc["skipped"][pkg] = "this node keeps no dependencies (Mathlib) for its manifest, and they aren't fetched cold"
                    continue
                write_links(tree, links)
                mark(tree, package=pkg, phase="build")
                mods = [f"+{module_of(pkg, f)}" for f in fs]
                log(f"lean-build: {pkg} on {tree} ({n} files changed there): lake build {' '.join(mods)}")
                rc, text = build(pkg, tree, lake, mods, sandbox)
                errs = first_errors(text, pkg, have)
                if rc and (healed := clashed(tree, closure(pkg), text)):
                    errs = {f: e for f, e in errs.items() if not CLASH.search(e)}
                    doc["skipped"][pkg] = f"an import clash this tree left ({' and '.join(healed)}), healed for the next build"
                doc["built"][pkg], doc["failed"] = rc, {**doc["failed"], **errs}
                log(f"lean-build: {pkg}: exit {rc} in {time.monotonic() - t0:.0f} s")
            (tree / DIRTY).unlink(missing_ok=True)
    finally:
        if slot is not None:
            os.close(slot)
            os.environ.pop(LEAN_SLOT.HELD, None)
    return done(1 if any(doc["built"].values()) else 0 if doc["built"] else 2)


def reports(out: Path, pkg: str, p: str) -> Path:
    """Where check.py's reports on ``p`` go: ``out`` for PKG, `requirer-<path>/` under it for a requirer."""
    return out if p == pkg else out / f"requirer-{re.sub(r'[^A-Za-z0-9]+', '_', p)}"


def report(out: Path, at: Path) -> dict:
    """check.py's report on the package at ``at`` among those it left under ``out`` (`audit.json`), or {} when it holds
    none; OSError or ValueError when `audit.json` can't be read."""
    doc = json.loads((out / "audit.json").read_text())
    reps = doc.get("packages") if isinstance(doc, dict) else None
    return next((r for r in reps if isinstance(r, dict) and Path(str(r.get("package"))).resolve() == at.resolve()), {}) \
        if isinstance(reps, list) else {}


def check_argv(at: Path, out: Path, update: bool, moved: Path | None, own: bool, dev: bool = False,
               artifacts: Path | None = None, facts_cache: Path | None = None) -> list[str]:
    """check.py on ``at`` (a package's directory), without the kernel replay or the tests, its builds reading the node's
    ``artifacts`` cache where given; with ``update``, `--update` and `--moved`. Only PKG's run (``own``) keeps a facts
    document, its own (`--facts named`, which `trusted_size.py --run` reads); a requirer's keeps none. With ``facts_cache``
    every run also stores its sealed stage-A facts document there (`--facts-cache`, keyed by its build, read back only by
    the key a tree computes: `read_cached_facts`); the run is still a records audit, never the kernel replay."""
    return [sys.executable, str(AUDIT_PY), *(["--dev-cache"] if dev else []), "--build", "--no-replay", "--no-runs",
            "--facts", "named" if own else "none", *(["--artifacts", str(artifacts)] if artifacts else []),
            *(["--facts-cache", str(facts_cache)] if facts_cache else []),
            *(["--update"] if update else []),
            *(["--moved", str(moved.resolve())] if moved and update else []), "--out", str(out), str(at)]


def settles(out: Path, at: Path) -> bool:
    """Whether check.py's reports under ``out`` fail only the package at ``at``, and it only on record differences: its report
    names the record `--update` writes (`regenerated`), which check.py does when every failure's code is one of
    `RECORD_DIFFERENCE` (a statement, a `reads` group or an assumption no longer its record), which `--update` settles from
    the build alone."""
    try:
        bad = [r for r in json.loads((out / "audit.json").read_text())["packages"] if r.get("failures")]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return len(bad) == 1 and bool(bad[0].get("regenerated")) and Path(str(bad[0].get("package"))).resolve() == at.resolve()


def first_update(p: str, pkg: str, update: bool, moved: Path | None, log=print) -> bool:
    """Whether check.py's first audit of ``p`` takes `--update`: PKG's in an `--update` run, and with ``moved`` a requirer's
    too, so that it takes `--moved`, which check.py reads only with `--update`. A rename that a requirer's records read (a
    statement, or the hash an `unwitnessed` entry holds) fails its audit without `--moved` on a code no record difference
    settles (`witnesses.unwitnessed-changed`), so the requirer would fail on that first audit and never reach
    `regenerates`."""
    if p == pkg or not update or moved is None:
        return update and p == pkg
    log(f"lean-records: {p} requires {pkg}: check.py --update --moved on it from its first audit, so the moves resolve "
        f"the names its records read before its failures are judged")
    return True


def regenerates(p: str, pkg: str, got: int, update: bool, out: Path, at: Path, log=print) -> bool:
    """Whether requirer ``p``, which check.py just audited at ``at`` (exit ``got``, its reports under ``out``), is audited again
    with `--update`: in an `--update` run, when it failed only because what its guarantees read changed (`settles`), so one
    run regenerates its lock beside PKG's instead of failing on it."""
    if p == pkg or not update or not got or AUDIT.killed(got) or not settles(out, at):
        return False
    log(f"lean-records: {p} requires {pkg}, and every failure of its records is a record difference (what its guarantees "
        f"read changed): check.py --update on it too")
    return True


def checked(p: str, pkg: str, got: int, record: Path, committed: bytes | None, up: bool, _update: bool, out: Path,
            log=print) -> bytes | None:
    """After check.py on ``p`` exited ``got``: when it ran with `--update` (``up``) and the record it wrote (``record``)
    differs from the ``committed`` bytes, that record, kept under ``out``'s `record/<p>/` (`keep`) and returned, which
    `settled` drops again when ``p`` fails otherwise; None when check.py was killed, which leaves no record to keep."""
    if not up:
        return None
    if AUDIT.killed(got):
        log(f"lean-records: {p}: check.py was killed, so what it left at {record} is not kept")
        return None
    return keep(p, pkg, record, out) if record.read_bytes() != committed else None


def write_back(p: str, record: Path, log=print) -> None:
    """Outside a recorded run, which never writes ROOT (the commit's shared source tree), write ``p``'s kept ``record`` into
    this checkout."""
    mine = ROOT / p / LOCK
    if not os.environ.get("RESEARCH_RUN_DIR") and record != mine:
        shutil.copyfile(record, mine)
        log(f"lean-records: wrote {mine}")


def run(cache: Path, deps: Path, tree: Path, log=print) -> int:
    files = LA.COMMON.tracked_files(ROOT)
    if files is None:
        log("lean-changed: skipped: the tree's files cannot be listed")
        return 0
    todo = missed(files, cache)
    if not todo:
        log("lean-changed: every package's audit is reused here; nothing to build")
        return 0
    lake = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(lake).is_file():
        log("lean-changed: skipped: no `lake` here (lean-build installs it)")
        return 0
    if not sandboxed():
        log("lean-changed: skipped: no build sandbox here (`unshare`), which keeps the audit's dependencies read-only")
        return 0
    made(tree)
    with open(tree / ".lock", "w") as held:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("lean-changed: skipped: another check on this machine holds the tree")
            return 0
        carry_builds(tree, log)
        links = read_links(tree)
        failed: list[str] = []
        built: list[str] = []
        compared: list[str] = []
        try:
            for pkg in todo:
                n = sync(closure(pkg), files, tree)
                if not link(pkg, tree, deps, links, log):
                    log(f"lean-changed: {pkg}: skipped; lean-audit builds it")
                    continue
                try:
                    roots = guarantee_roots(pkg, tree)
                except (OSError, ValueError, KeyError) as e:
                    roots = []
                    log(f"lean-changed: {pkg}: its {AUDIT.POLICY} cannot be read ({e!r}); its guarantees are not compared here")
                t0 = time.monotonic()
                rc, text = build(pkg, tree, lake)
                if rc == 0 and roots:
                    rc, more = build(pkg, tree, lake, [f"+{r}" for r in roots])
                    text += more
                took = time.monotonic() - t0
                if rc and (bad := errors(text, pkg, files)):
                    failed.append(pkg)
                    log(f"lean-changed: {pkg}: FAILED in {took:.0f} s: a module does not compile:")
                    log("\n".join(f"  {line}" for line in bad[:20]))
                    break
                if rc:
                    unlink(pkg, tree, links)
                    log(f"lean-changed: {pkg}: skipped: `lake build` exited {rc} without a Lean error in the tree's files; "
                        "its links are dropped, and lean-audit builds it")
                    continue
                built.append(pkg)
                log(f"lean-changed: {pkg}: built in {took:.0f} s ({n} files changed in the tree)")
                if not roots:
                    continue
                t0 = time.monotonic()
                try:
                    stale = records(pkg, tree, lake)
                except (RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                    log(f"lean-changed: {pkg}: its guarantees are not compared here: {str(e)[-2000:]}")
                    continue
                took = time.monotonic() - t0
                if stale:
                    failed.append(pkg)
                    log(f"lean-changed: {pkg}: FAILED in {took:.0f} s: the audit fails its guarantees on this build:")
                    log("\n".join(f"  {line}" for f in stale[:20] for line in f.splitlines()))
                    break
                compared.append(pkg)
                log(f"lean-changed: {pkg}: its guarantees match their records ({took:.0f} s)")
        finally:
            write_links(tree, links)
    log(f"lean-changed: {'FAIL (' + ', '.join(failed) + ')' if failed else 'PASS'}: built {len(built)} of the {len(todo)} "
        f"package(s) whose audit misses; compared the guarantees of {len(compared)}")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lean_changed.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=LA.COMMON.cache_dir() / "lean-audit", help="the audit's kept passes")
    ap.add_argument("--deps", type=Path, default=LA.COMMON.cache_dir() / "lean-deps", help="the audit's kept dependencies")
    ap.add_argument("--tree", type=Path, help="the step tree this step keeps built (default: lean-changed), or with --records, "
                    f"--try or --build the slot tree ({RECORDS}/build-I, the Lean slot's), under the check cache; with those it "
                    f"must be under {RECORDS}/")
    ap.add_argument("--records", metavar="PKG",
                    help=f"audit PKG's records without the kernel replay, on a tree kept per Lean slot of pool {FAST_POOL}, "
                    f"else with check.py --dev-cache on this checkout (a {FAST_KIND} result, which no merge accepts)")
    ap.add_argument("--update", action="store_true", help="with --records: check.py --update")
    ap.add_argument("--moved", type=Path, metavar="FILE", help="with --records --update: check.py --update --moved FILE on PKG "
                    "and, from their first audit, on its requirers")
    ap.add_argument("--out", type=Path, help="with --records: check.py --out DIR")
    ap.add_argument("--dev-cache", action="store_true", help="with --records: take the check.py --dev-cache path even where "
                    "a build slot is granted (its tree's dependencies are fetched cold, say)")
    ap.add_argument("--rebuild", action="store_true", help="with --records: run every package's records, even those that "
                    "passed on the same Lean inputs or that a merge of main leaves a candidate of")
    ap.add_argument("--try", dest="try_", metavar="FILE",
                    help=f"elaborate FILE (from the repository's root) on the tree of a Lean slot of pool {FAST_POOL}: "
                    "lake build of its imports, then lake env lean FILE with its package's Lean options; never check.py")
    ap.add_argument("--siblings", action="store_true", help="with --try: then import FILE's build beside every built module "
                    "of its package that doesn't import it, which refuses a name two of them declare (siblings.py)")
    ap.add_argument("--build", nargs="+", metavar="FILE",
                    help=f"lake build the modules of these files (from the repository's root) on the tree of a Lean slot of pool "
                    f"{FAST_POOL}, as --try does, with no dependency fetched cold; its last line is `{BUILD_LINE}` and JSON "
                    "(the ready mark's preflight)")
    ap.add_argument("--no-wait", action="store_true", help="with --build: build nothing when every slot of its pool is taken, "
                    "rather than wait in line")
    ap.add_argument("--inputs-key", nargs="+", metavar="PKG",
                    help="print {package: its inputs key} as JSON for each PKG at --head, the key a records run of that commit "
                    "keeps and passes on, from git's objects alone (no checkout, no build)")
    ap.add_argument("--head", default="HEAD", metavar="SHA", help="with --inputs-key: the commit (default HEAD)")
    ap.add_argument("--repo", type=Path, metavar="DIR", help="with --inputs-key: the git repository to read --head from "
                    "(default this file's checkout); a clone with no worktree checked out serves")
    a = ap.parse_args(argv)
    if a.inputs_key:
        if a.records or a.update or a.moved or a.out or a.dev_cache or a.rebuild or a.try_ or a.build or a.no_wait or a.tree:
            ap.error("--inputs-key takes only --head and --repo")
        try:
            print(json.dumps(keys_at(a.inputs_key, a.head, a.repo), sort_keys=True))
        except (OSError, ValueError, subprocess.CalledProcessError) as e:
            print(f"lean_changed: --inputs-key: {e}", file=sys.stderr)
            return 2
        return 0
    if a.head != "HEAD" or a.repo:
        ap.error("--head and --repo go with --inputs-key")
    if a.try_ or a.build:
        if a.records or a.update or a.moved or a.out or a.dev_cache or a.rebuild or (a.try_ and a.build) or (a.try_ and a.no_wait):
            ap.error("--try and --build take only --deps and --tree (--build also --no-wait), and not each other")
        if a.build:
            if a.siblings:
                ap.error("--siblings goes with --try")
            return build_files(a.build, a.deps, a.tree, log=lambda s: print(s, flush=True), wait=not a.no_wait)
        return try_file(a.try_, a.deps, a.tree, log=lambda s: print(s, flush=True), siblings=a.siblings)
    if a.siblings:
        ap.error("--siblings goes with --try")
    if not a.records:
        if a.update or a.moved or a.out or a.dev_cache or a.rebuild:
            ap.error("--update, --moved, --out, --dev-cache and --rebuild go with --records")
        return run(a.cache, a.deps, a.tree or LA.COMMON.cache_dir() / "lean-changed", log=lambda s: print(s, flush=True))
    if a.dev_cache and a.tree:
        ap.error("--dev-cache builds in this checkout, not in a --tree")
    return audit_records(a.records, a.deps, a.tree, a.update, a.moved, a.out, log=lambda s: print(s, flush=True),
                         dev_cache=a.dev_cache, rebuild=a.rebuild)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
