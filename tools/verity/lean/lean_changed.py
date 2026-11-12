"""`lean_changed`: a lane's Lean runs beside the audit (`lean_audit.py`), none of them a Lean verdict: its records
(`--records`), its edit loop (`--try`) and the ready mark's preflight's Lean step (`--build`), each on a slot tree this machine
keeps built, one per `build` slot (below), and a package's inputs key (`--inputs-key`). It runs one of those: without one it
prints its usage and exits 2.

    python tools/verity/lean/lean_changed.py --records PKG [--update [--moved FILE]] [--out DIR] [--dev-cache] [--rebuild]   a lane's records
                                             [--allow CODE:ENTRY]...   with --records: a failure it takes as known, which is no pass
    python tools/verity/lean/lean_changed.py --try FILE [--siblings]   a lane's edit loop on a node (`verity lean try FILE --on NODE`)
    python tools/verity/lean/lean_changed.py [--no-wait] --build FILE...   the ready mark's preflight's Lean step (`build_files`)
    python tools/verity/lean/lean_changed.py --inputs-key PKG... [--head SHA] [--repo DIR]   each PKG's inputs key at SHA

`--try` elaborates one file (FILE: its path from the repository's root) on the slot tree `--records` builds on, under the same
`build` slot, tree lock, sync, `repair` and dependencies, for the closure of the package that owns FILE only: there it runs
`lake build +MODULE:deps` (FILE's imports; Lake rebuilds only what changed since the slot's last build) and then
`lake env lean` with the package's `[leanOptions]` (`siblings.options`) in that package's directory on a copy of FILE with
the axiom probe appended (`axiom_probe.py`: its last line names each theorem whose axioms go beyond the standard three,
`sorryAx` among them, or says none do; where lean fails on FILE's own lines it shows no answer of `#print axioms`, and
names each theorem with an error on its own lines as failed), and exits with lean's
status on FILE's own lines (with `--siblings`, that copy's build then goes through the sibling check, `siblings.py`, whose
status it takes: its stub imports no module whose build, or the build of anything it imports in any package of the
closure, is not of its source as it is, as a tree synced across a move of names holds them, and counts those it left
out); when that status is an import clash (an
import declaring what an earlier one did), the two modules it names lose their traces (`clashed`), so the next try rebuilds
them against the outputs in place. It never runs check.py, so it reads and writes
no record; where the node grants no `build` slot it refuses, naming the machines whose Lean slots file has that pool (each
one `slot.py`'s PEERS lists, read over ssh: `pool_machines`), else how to list a machine's Lean pools. Dependencies the
slot tree holds that no `--records` run took (not in its `links.json`) are kept as they are while they hold every package
their manifest names (`cache.py`'s `fetched_deps`). A directory that lacks one, as a run killed while Lake cloned them
leaves it, in `links.json` or not, is taken as none is: the try says what it lacks and takes the ones the node keeps
(`link`). In the sandbox, where Lake clones nothing, a package still missing after that (the node keeps none, or its
copy lacks one) ends the try before Lake runs: it exits 2, which is no verdict on FILE, and its line names what is
missing and the `--records` run that fetches it onto this tree, which it names with `--tree`: a records run fetches for
the one tree it takes. `--build` takes such a directory the same way and skips a package still
missing, with a sandbox or without, since it fetches nothing cold. Without a sandbox the try's Lake runs
unsandboxed, fetches dependencies the node does not keep, and restores nothing from a cache, its own included, so it
compiles every module it needs (`build`). In the sandbox the tree's fetched dependencies are mounted
read-only, so that no try changes what the next one builds against. Before that mount a dependency with a `lakefile.lean` and
no `.lake` gets an empty one, in the tree's own directory and never in the ones the audit keeps (`READ_ONLY`): Lake makes
it before it compiles that configuration at a dependency order the tree was not loaded at, and a slot tree whose dependencies
were fetched again under the `.lake/config` it kept has none. A write Lake is still refused there gives no verdict on FILE:
the try says so, names the path and the `--records` run that builds the dependencies writable, and exits 2
(`refused_write`).

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
another, only under `lean-records/` (a slot's tree there that is a link out of the check cache, onto another volume,
too: `kept_tree`). A slot
tree's fetched dependencies are its own copy of the audit's (taken once per
manifest and toolchain, else fetched cold), never links, so what Lake writes there stays
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
never reads that package's `.lake` (`check.proved`), which an earlier or interrupted build left. When check.py fails a
package only on its sampled recompile (`check.sampled_recompile`: a module the tree kept or restored compiles from its
source to other outputs, which Lean's nondeterminism can do as staleness can), on record differences, on failures
`--allow` names and on a spec's `proved_in` package whose build that failure kept from passing
(`proved_in.unpassed`), the run drops each of those modules' trace, `.hash` files and objects in the tree
(`mismatched`, `forget`) and audits the package once more, with nothing restored from the artifact cache, so Lake
compiles them; the first audit's reports go to `mismatched/<path>/`, and a second failure fails the run.

Then it audits, the same way, every package whose guarantees' records read PKG's build (`requirers`: PKG is in its
`closure`, by path or by `proved_in`), each after those it requires and on the records this run regenerated before it, its
reports under `requirer-<path>/` beside PKG's. Of the facts documents, only PKG's own stays beside its reports
(`check_argv`: `check.py --facts named`, which `trusted_size.py --run` reads); its provers' and the requirers' are not
kept. A change to what another package's pinned guarantees read
(`security/Properties`' when `security` changes) fails here, not at the Lean check of `next`. With `--update`, a requirer whose every
failure is a record difference (`settles`) is audited again with `--update`, so one run regenerates every lock the change
moved; one failing for any other reason fails. With `--moved` too, each requirer's first audit takes `--update --moved`
(`first_update`; check.py reads `--moved` only with `--update`), so a rename its records read resolves before its
failures are judged. The exit status is the first failure's. A slot tree
keeps the copies of the packages one run audits (about 170k inodes, 8 GB each), so auditing another package that fetches
dependencies there drops the rest first: every copy its `links.json` lists but those of the packages in the `closure` of
one this run audits (a prover's among them), which its check.py builds, so that dropping one would only have it cloned
again. Such a copy is kept while its entry is the key its manifest and toolchain give now and it holds every package the
manifest names (`cache.py`'s `fetched_deps`), and dropped otherwise. Each drop says one line: the package, and whether
it was outside this run's packages, under another key, or incomplete (`drops`).

A record difference is not a failure of the run. A package that any other run of check.py fails only on record differences
(check.py's report names the record that settles them, `regenerated`) has that record kept the same way, so a run without
`--update` regenerates too. The run's outcome (`result.json`'s `outcome`, and its last line, `OUTCOME_LINE` and JSON) is its
packages' worst: `pass`, `regenerated` (each changed lock kept under `record/`), `allowed` (below), `failed` (a failure of
any other kind; a record kept for such a package goes again) or `refused` (nothing ran). It exits 0 on `pass` and
`regenerated`, 3 on `allowed` (`ALLOWED_RC`), check.py's first failing status on `failed` and 2 on `refused`.

`--allow CODE:ENTRY`, repeatable and `--records`' alone, names a failure this run takes as known, by its `Failure.code`
and `Failure.entry` as check.py's report lists them (`failure_codes`, `failure_entries`), never by its text. It stays a
failure: check.py is run as ever and reports it in `failures`. A package whose audit's reports hold one, every other
failure there a record difference the run settled, ends `allowed`, which is no pass: the run exits 3, keeps no pass of
that package (nothing under `lean-records/passes/`, no key in `passes`), so nothing takes this run for its records and the
next run audits it again, and keeps the record it regenerated for it under `allowed-record/<path>/` beside `record/` and
never in `record/`, which holds only what the records gate may take; a line says where. A record difference is never
allowed: beside allowed failures it takes `--update`, without which the package fails. The codes that may be named are a
closed list here (`ALLOWABLE`: a statement that reads the proofs tree, a guarantee changed since `unwitnessed` listed
it); any other is refused before anything runs (exit 2), a record difference, the build, the replay, an axiom, `sorry`,
`native_decide`, a kernel bypass and a malformed lock among them. An allowance that matches no failure of the run fails
it (`unmatched`), so one outlives its finding by no run. It is for a finding cured before the mark: one meant to land
goes into the lock (`trees.except`), as ever. Allowances come from the command line alone, and only a lane gives one:
this file's other modes (`--build` among them), check.py and every caller of either in the check, the merge queue and
the ready mark's preflight take none.

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
run they go into its `lean-audit/` in bundles (`lean_audit.LakeArtifacts.bundle_into`), and a node that restores one still
replays every declaration through the kernel. `research run` sends a records run the URLs of the newest such bundles and of
the checks' (`research.cli.lean_records_urls`, `lean_audit.shared_bundles`), and where builds are sandboxed the run takes
them into the node's artifact cache before its first build (`sent_urls`), so it restores what another node compiled.
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
#: `tools/verity/lean/check.py`, the audit tool, loaded as this file's own copy
AUDIT = _module(ROOT / "tools" / "verity" / "lean" / "check.py", "lean_audit_tool_for_lean_changed")
SIBLINGS = _module(ROOT / "tools" / "verity" / "lean" / "siblings.py", "lean_siblings_for_lean_changed")
PROBE = _module(ROOT / "tools" / "verity" / "lean" / "axiom_probe.py", "axiom_probe_for_lean_changed")
#: the audit's build sandbox, as `tools/verity/lean/check.py` runs it: writable only in the `.lake` directories it is given
UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child"]
SANDBOX = ROOT / "tools" / "verity" / "lean" / "sandbox.sh"
#: run inside the sandbox before the build: `ro DIR... -- CMD`, each DIR bind-mounted read-only over itself. Before its
#: mount, each package directly under DIR (a directory of its own, never a link) that has a `lakefile.lean` and no `.lake`
#: gets an empty one: Lake keeps such a package's compiled configuration in the workspace's
#: `.lake/config/<the package's place among those it loads>`, and where that place holds no trace of it (a dependency order
#: this tree was never loaded at) it first makes the package's own `.lake`, which under the mount fails the load. A package
#: nothing builds has no `.lake` unless Lake made it so. Nothing else under DIR is written; a `mkdir` that fails is left to
#: Lake's own error (`refused_write`)
READ_ONLY = ('ro=(); while [ "$1" != -- ]; do ro+=("$1"); shift; done; shift; '
             'for d in "${ro[@]}"; do '
             'for p in "$d"/*; do [ -f "$p/lakefile.lean" ] && [ ! -L "$p" ] && [ ! -e "$p/.lake" ] && mkdir "$p/.lake"; done; '
             'mount --bind "$d" "$d" && mount -o remount,bind,ro "$d" || exit 97; done; exec "$@"')
#: Lake's two lines for a write the sandbox's read-only mount refused, the second naming the path
REFUSED_WRITE = re.compile(r"^error: read-only file system \(error code: \d+\)\n[ \t]+file: (\S.*?)[ \t]*$", re.M)
#: Lake's line for a Lean error, the file relative to its package: `error: Aa/Two.lean:2:39: Unknown identifier ...`
ERROR = re.compile(r"^error: (?:\./)*(\S+?\.lean):\d+:\d+: ")
#: {package: the dependencies' key its links were taken under}, beside the packages in the tree
LINKS = "links.json"


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
    """``pkg``, the packages it requires by path, and those its `proved_in` names with theirs, whose builds the audit of
    ``pkg`` reads; its manifests and locks read through ``read`` (this checkout's by default)."""
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
    after those it requires (`rank`)."""
    return sorted((q for q in pkgs if q != pkg and pkg in closure(q) and lists_guarantees(q)), key=lambda p: (rank(p), p))


def audits(todo: list[str]) -> str:
    """The packages a `--records` run of ``todo`` audits, in its order, as its line at launch names them: each of ``todo``,
    and with it those its `proved_in` names (`provers`), which its check.py audits beside it, once more for each package
    of ``todo`` that names them."""
    return ", then ".join(f"{p} (with {', '.join(qs)})" if (qs := provers(p)) else p for p in todo)


def sync(pkgs: list[str], files: list[str], tree: Path) -> int:
    """Make ``tree``'s copy of ``pkgs`` this checkout's tracked files, leaving each `.lake`; the number of files written.
    The product of a source it unlinks, or rewrites, stays: a tree serves several branches, and Lake replays the product
    when one of them brings that source back as it was built from. Until then neither Lake nor a sibling check's stub
    loads it: each reads in the module's trace which source it was built from (`siblings.py`)."""
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


def link(pkg: str, tree: Path, deps: Path, links: dict[str, str], log=print) -> bool:
    """``tree/pkg/.lake/packages`` as a copy of the audit's kept dependencies for its manifest and toolchain, taken after
    their lock is released from hard links made under it (`packages.snap` beside it), so that the audit waits on the lock
    only as long as linking takes; where they can't be linked, the copy is taken under the lock. False when this machine
    keeps none (another run holds them, or none has passed here yet)."""
    if (k := LA.deps_key(tree / pkg)) is None:
        return True
    dst = tree / pkg / ".lake" / "packages"
    if links.get(pkg) == k and dst.is_dir():
        return True
    unlink(pkg, tree, links)
    snap = dst.with_name("packages.snap")
    shutil.rmtree(snap, ignore_errors=True)
    try:
        with LA.WarmDeps(deps)._lock(k):
            src = deps / k
            if not src.is_dir():
                log(f"lean_changed: {pkg}: this machine keeps no dependencies for it ({k[:12]}) to copy")
                return False
            t0 = time.monotonic()
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copytree(src, snap, symlinks=True, copy_function=os.link)
            except (OSError, shutil.Error):
                shutil.rmtree(snap, ignore_errors=True)
                shutil.copytree(src, dst, symlinks=True, copy_function=shutil.copy2)
        if snap.is_dir():
            shutil.copytree(snap, dst, symlinks=True, copy_function=shutil.copy2)
    except (OSError, shutil.Error) as e:
        shutil.rmtree(dst, ignore_errors=True)
        log(f"lean_changed: {pkg}: the dependencies ({k[:12]}) could not be copied ({e})")
        return False
    finally:
        shutil.rmtree(snap, ignore_errors=True)
    links[pkg] = k
    log(f"lean_changed: {pkg}: dependencies copied ({k[:12]}) in {time.monotonic() - t0:.0f} s")
    return True


def unlink(pkg: str, tree: Path, links: dict[str, str]) -> None:
    shutil.rmtree(tree / pkg / ".lake" / "packages", ignore_errors=True)
    links.pop(pkg, None)


_CACHE = None


def lacks(pkg: str, tree: Path) -> str | None:
    """What ``tree/pkg``'s `.lake/packages` lacks of the fetched dependencies its manifest names, in words, or None when it
    holds a directory for each (`cache.py`'s `unfetched`, whose `fetched_deps` is the one test of that; the file is loaded
    by what takes a tree's dependencies and not with this one, as `moved.py` is). A run killed while Lake cloned them
    leaves the ones it had cloned. Lake clones the rest where it may; in the sandbox it fails on the first one missing."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _module(Path(__file__).resolve().parent / "cache.py", "lean_cache_for_lean_changed")
    gone = _CACHE.unfetched(tree / pkg)
    names = f"{', '.join(gone[:8])}{' ...' if len(gone) > 8 else ''}"
    return f"{len(gone)} of the packages its manifest names ({names})" if gone else None


def fetched_by(pkg: str, tree: Path) -> str:
    """The run that fetches ``pkg``'s dependencies onto ``tree`` where `--try` and `--build` can't: a records run's
    check.py sets the package up outside the sandbox, where Lake clones what its manifest names. `--rebuild`, since
    records that need no run are skipped, nothing fetched; `--tree`, since a records run fetches for the one tree it
    takes, and a try on another tree of the node is then no better off."""
    return f"`lean_changed.py --records {pkg} --rebuild --tree {tree}` here fetches them onto this tree"


def drop_incomplete(pkg: str, tree: Path, links: dict[str, str], log=print) -> None:
    """Before `--try` or `--build` takes ``tree/pkg``'s dependencies: a `.lake/packages` that `lacks` a package goes, with
    its entry of ``links`` and of the tree's `LINKS`, and a line says what it lacked, so it is taken as none is: whole,
    from the ones this node keeps (`link`). Whether a run was left mid-way there (`DIRTY`) does not decide it: a run
    since can have cleared that mark and kept the directory."""
    if (tree / pkg / ".lake" / "packages").is_dir() and (what := lacks(pkg, tree)):
        unlink(pkg, tree, links)
        write_links(tree, links)
        log(f"lean-records: {pkg}: its fetched dependencies lacked {what}, as a run killed while Lake cloned them leaves "
            "them; they were dropped, and are taken whole or not at all")


def drops(pkg: str, todo: list[str], tree: Path, links: dict[str, str]) -> dict[str, str]:
    """{package: why} for each package of ``links`` whose copy of its fetched dependencies a records run of ``todo`` drops
    before it audits ``pkg``. The run keeps the copy of each package in the `closure` of one it audits, which its check.py
    builds (a prover's among them), while its entry is the key its manifest and toolchain give now and it `lacks` nothing:
    dropped, check.py would clone it again. Every other copy goes when ``pkg`` fetches dependencies, for the inodes."""
    mine = {c for t in todo for c in closure(t)}
    why = {}
    for q, k in links.items():
        if q not in mine:
            if LA.deps_key(tree / pkg):
                why[q] = "outside this run's packages"
        elif k != (now := LA.deps_key(tree / q)):
            why[q] = f"another key: {LINKS} has {str(k)[:12]}, and its manifest and toolchain now give {now[:12] if now else 'none'}"
        elif what := lacks(q, tree):
            why[q] = f"incomplete: it lacks {what}"
    return why


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


def refused_write(text: str, pkg: str, tree: Path) -> str | None:
    """The path Lake's output ``text`` says a read-only file system refused it (`REFUSED_WRITE`), when it lies in the fetched
    dependencies of ``tree/pkg`` or of a package it requires by path, which `in_sandbox` mounts read-only; else None."""
    ro = [(tree / p / ".lake" / "packages").resolve() for p in closure(pkg)]
    return next((m.group(1) for m in REFUSED_WRITE.finditer(text)
                 if any(d in Path(os.path.realpath(m.group(1))).parents for d in ro)), None)


def build(pkg: str, tree: Path, lake: str, targets: list[str] | None = None, sandbox: bool = True,
          cmd: list[str] | None = None, echo=None) -> tuple[int, str]:
    """`lake build` (of ``targets``, else the default ones; ``cmd``: that `lake` command instead) of ``tree/pkg``, in the
    sandbox unless ``sandbox`` is False; Lake's output is echoed as it comes, or each line handed to ``echo``. Before the
    default targets it links the dependencies' compiler plugins alone (`check.py`'s `plugin_targets`).

    Outside the sandbox Lake restores nothing from a cache, its own included, whatever the environment or a lakefile
    says: there it restores by a hard link, and the restored file keeps the cache file's time, which `outdated` reads as
    the time it was built. `LAKE_ARTIFACT_CACHE` turns the cache off for every package whose lakefile leaves
    `enableArtifactCache` unset; a lakefile that sets it is read before the environment, so `LAKE_CACHE_DIR` names an empty
    directory of this command's own, removed after it, and such a package finds there only what this command built."""
    if cmd is None and targets is None and (plugins := AUDIT.plugin_targets(tree / pkg)):
        rc, text = build(pkg, tree, lake, plugins, sandbox)
        if rc:
            return rc, text
    argv = [lake, *(cmd or ["build", *(targets or [])])]
    argv = in_sandbox(pkg, tree, argv) if sandbox else argv
    env = {**os.environ, "LEAN_NUM_THREADS": os.environ.get("LEAN_NUM_THREADS") or
           str(AUDIT.build_threads(tree / pkg) or LA.COMMON.cores())}
    with contextlib.nullcontext() if sandbox else tempfile.TemporaryDirectory(prefix="lake-no-cache-") as empty:
        if empty is not None:
            env |= {"LAKE_ARTIFACT_CACHE": "false", "LAKE_CACHE_DIR": empty}
        p = subprocess.Popen(argv, cwd=tree / pkg, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        out = []
        for line in p.stdout:
            if echo:
                echo(line)
            else:
                print(line, end="", flush=True)
            out.append(line)
        return p.wait(), "".join(out)


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
#: under `record/`; every failure left was one `--allow` named (no pass is kept for such a package, and a record
#: regenerated for it is kept under `ALLOWED_RECORD`, never `record/`); a failure of another kind, or an allowance no
#: failure matched; nothing ran
PASS, REGENERATED, ALLOWED, FAILED, REFUSED = OUTCOMES = ("pass", "regenerated", "allowed", "failed", "refused")
#: `--records`' exit status on `ALLOWED`, its own: not 0 (a pass or `REGENERATED`), not check.py's failing 1 or a signal's
#: status (`FAILED`), not 2 (`REFUSED`)
ALLOWED_RC = 3
#: the failures `--allow` may name, by `Failure.code`: a closed list, the lock rules a lane has had ruled open on a branch
#: and cures before its mark (a statement that reads the proofs tree, a guarantee restated since `unwitnessed` listed it).
#: Never a record difference (`AUDIT.RECORD_DIFFERENCE`, which `--update` settles), the build, the replay, an axiom,
#: `sorry`, `native_decide`, a kernel bypass or a malformed lock: `allowances` refuses every code not listed here
ALLOWABLE = AUDIT.Codes({"trees.reads", "witnesses.unwitnessed-changed"})
#: where `--records` keeps the records it regenerated, under its reports: `KEPT_RECORD` (`record/`) those of a package that
#: ended `REGENERATED`, which a lane commits and `research queue`'s records gate names; `ALLOWED_RECORD` those of a package
#: that ended `ALLOWED`, which no pass vouches for
KEPT_RECORD, ALLOWED_RECORD = "record", "allowed-record"
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
    """`DIRTY`, replaced whole: this run and ``what`` (`package`, `phase`, and for a `--records` run's build what `put_left`
    reads)."""
    doc = {"run": os.environ.get("RESEARCH_RUN_ID") or "a run", "pid": os.getpid(),
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **what}
    tmp = tree / f"{DIRTY}.tmp"
    tmp.write_text(json.dumps(doc, sort_keys=True) + "\n")
    os.replace(tmp, tree / DIRTY)


#: Lake's line for a module it compiled (`Built Mod (1.2s)`); one it restored says `Unpacked` or `Replayed`, and a facet's
#: (an object, an executable) has a colon in its name
BUILT = re.compile(r"^\s*\S+ \[\d+/\d+\] Built ([^\s:]+)(?: \(.*\))?$", re.M)


def compiled(text: str) -> list[str]:
    """The modules Lake's output ``text`` says it compiled. In the sandbox a `--try` or `--build` build reads Lake's own
    cache, so it unpacks main's published build where that holds a module's input hash, and `put` must not take such a
    module as one this node compiled: `LakeArtifacts.put` indexes only these."""
    return sorted({m.replace("«", "").replace("»", "") for m in BUILT.findall(text)})


def put_built(artifacts, reps: dict[Path, dict], since: float, log=print, who: str = "lean-records") -> None:
    """Put what the builds of ``reps`` ({package: what says its build ran in the sandbox}) compiled since ``since`` into the
    node's artifact cache (`LakeArtifacts.put`), whatever became of the run: a module is stored for what its compile read,
    which a failure elsewhere in the package, or a kill, doesn't change, unless a source of the build runs code while it
    compiles (`LA.runs_code`). Before anything writes the tree again: a key is computed from the sources the tree
    holds, and those are the sources `put` reads for such code."""
    try:
        log(f"{who}: {artifacts.put(reps, since, log)} module version(s) new in the artifact cache")
    except (OSError, ValueError) as e:
        log(f"{who}: the artifact cache was not written: {e}")


def put_left(tree: Path, left: dict | None, artifacts, log=print) -> None:
    """`put_built` what the run that left ``tree`` mid-way (``left``, its `DIRTY`) had compiled when it stopped, which it never
    stored itself: its mark names when its build began (`since`) and where check.py's reports say whether each build
    ran in the sandbox (`out`, `LA.built`). Only a `--records` run's: what a `--try` or a `--build` left is not stored,
    since Lake's output, which alone says which modules that build compiled and which it unpacked (`compiled`), ended
    with the run. A run does this before it syncs the tree. Without it a build that is killed each time (a timeout
    shorter than a cold build) would start from nothing each time, since `stage` leaves no trace of a module the cache
    doesn't hold."""
    if not left or left.get("phase") != "check" or not isinstance(since := left.get("since"), (int, float)):
        return
    reps = LA.built(Path(str(left.get("out"))))
    if reps := {at: rep for at, rep in reps.items() if tree.resolve() in at.resolve().parents}:
        put_built(artifacts, reps, since, log, f"lean-records: {tree}: of what {left.get('run', 'a run')} left")


def storable(sandbox: bool) -> bool:
    """Whether what a `--try` or `--build` build compiles may go into the artifact cache: it ran in the sandbox
    (``sandbox``), and that sandbox leaves everything outside the build read-only (check.py's `sandboxed`, the probe its
    own builds' stores rest on)."""
    return sandbox and AUDIT.sandboxed()


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
#: the keys of a module trace's `outputs` that name what `lean` wrote in the build's `ir/`: the C, and the LLVM bitcode
IR_OUTPUTS = ("c", "b")
#: what Lake compiles from them there, by suffix, and the suffix of the output each is compiled from; each has a `.hash`
#: and a `.trace` of its own beside it
OBJECTS = {"c.o.export": "c", "c.o.noexport": "c", "bc.o": "bc"}


def damaged(lib: Path) -> list[Path]:
    """The traces under ``lib`` (a `.lake/build/lib/lean`) whose module an interrupted build or a restore left half-written:
    an output the trace lists beside it that is missing (`LIB_OUTPUTS`), one in the olean format without its header, an
    output there or in the build's `ir/` (`IR_OUTPUTS`) whose `.hash` is not the trace's or is older than the output, or a
    trace that is not JSON. Lake rebuilds a module
    whose outputs are missing, but takes one whose trace matches and whose files exist as built, half-written or not, and
    reads an output's hash from its `.hash` without hashing the output. Every Lake path writes the `.hash` after its output,
    but two leave an older one: a build killed after `lean` wrote its outputs, and a restore from Lake's cache that unpacks
    a `.ltar` over the outputs.
    A module that imports one then records the old hash for the new bytes, and a later build on the other side of the
    change replays it beside a module built against the new ones (a clash such as `Proofs.Pouw`'s on
    `W1.Instr.exec.eq_10`); an object compiled from its C is recorded under the old C's hash, and is taken as built once
    the module's source is back at that C. So each loses its trace, its `.hash` files and its objects (`forget`) for Lake
    to rebuild it."""
    out = []
    ir = lib.parent.parent / "ir"
    for trace in sorted(lib.rglob("*.trace")) if lib.is_dir() else []:
        base = trace.with_suffix("")
        try:
            doc = json.loads(trace.read_text())
            outputs = doc.get("outputs") if isinstance(doc, dict) else None
            names = [(k in LIB_OUTPUTS, n) for k, v in outputs.items() if k in LIB_OUTPUTS + IR_OUTPUTS
                     for n in (v if isinstance(v, list) else [v]) if isinstance(n, str)] if isinstance(outputs, dict) else []
            for beside, name in names:
                digest, _, ext = name.partition(".")
                f = (base if beside else ir / base.relative_to(lib)).with_name(f"{base.name}.{ext}")
                if beside and (not f.is_file() or (ext != "ilean" and f.open("rb").read(5) != b"olean")):
                    raise ValueError(f"{f.name} missing or not in the olean format")
                if (h := f.with_name(f"{f.name}.hash")).is_file():
                    if h.read_text().strip() != digest:
                        raise ValueError(f"{h.name} is not the trace's")
                    if f.is_file() and f.stat().st_mtime_ns > h.stat().st_mtime_ns:
                        raise ValueError(f"{h.name} is older than {f.name}")
        except (OSError, ValueError, UnicodeDecodeError):
            out.append(trace)
    return out


def objects(base: Path) -> list[Path]:
    """The objects Lake compiled from the module whose `ir/` outputs are ``base``'s (`ir/A/B` for `A.B`), each with the
    `.hash` and the `.trace` beside it: the files that are there."""
    return [f for o in OBJECTS for end in ("", ".hash", ".trace") if (f := base.with_name(f"{base.name}.{o}{end}")).exists()]


def forget(lib: Path, trace: Path) -> None:
    """Drop ``trace`` (a module's, under ``lib``), the `.hash` file of each of the module's outputs, in ``lib`` and in
    the build's `ir/`, and the module's objects there with their traces. A restore from Lake's cache that finds an output
    in place keeps it and its `.hash` as they are, so a stale `.hash` left beside the trace would outlive the rebuild; an
    object's trace names the hash Lake read for the C when it compiled it, which a stale `.hash` made another C's, so the
    object would outlive it too."""
    rel = trace.relative_to(lib).with_suffix("")
    for d in (lib, lib.parent.parent / "ir"):
        for h in (d / rel).parent.glob(f"{glob.escape(rel.name)}.*.hash"):
            h.unlink(missing_ok=True)
    for f in objects(lib.parent.parent / "ir" / rel):
        f.unlink(missing_ok=True)
    trace.unlink(missing_ok=True)


def outdated(lib: Path) -> list[Path]:
    """The objects in the build's `ir/` (``lib`` is its `.lake/build/lib/lean`) older than the C or bitcode Lake compiles
    them from: one written before the source beside it, one with no source beside it, and one whose module has no trace,
    which Lake rebuilds, writing the source anew. Lake writes an object after its source, so such an object is from an
    earlier source, and Lake takes it as built whenever its trace names the hash the new source has. A `repair` that
    dropped a module's trace and `.hash` files and kept its objects left them so.

    The rule reads the files' times, so it rests on one premise: a file Lake restores from a cache is a copy, with the
    time of its restore. Lake restores by a hard link wherever one can be made, and a linked file keeps the cache file's
    time: a source restored so can be older than the object of the source that was there before it, which then stays,
    and an object restored so is older than its own source, and goes. The premise is kept in two places. In the sandbox
    each `.lake` is a mount of its own (`sandbox.sh`), a link into it from a cache fails, and Lake copies. Outside the
    sandbox nothing is restored into a slot tree: `build` gives Lake no cache to read there, and check.py's build of one
    stages none and runs with Lake's cache off (`lake_env`)."""
    out = []
    ir = lib.parent.parent / "ir"
    for o, of in OBJECTS.items() if ir.is_dir() else ():
        for obj in ir.rglob(f"*.{o}"):
            base = obj.with_name(obj.name.removesuffix(f".{o}"))
            src = base.with_name(f"{base.name}.{of}")
            if not (lib / base.relative_to(ir)).with_name(f"{base.name}.trace").is_file() or not src.is_file() \
                    or src.stat().st_mtime_ns > obj.stat().st_mtime_ns:
                out.append(obj)
    return sorted(out)


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
    finds loses its trace, its `.hash` files and its objects (`forget`), so Lake rebuilds it, a module with no trace loses
    its `.hash` files (`orphaned`), and an object older than the C Lake compiles it from goes with its trace (`outdated`),
    so Lake compiles it again. Where a run ended mid-way (``left``, its `DIRTY`), also each package whose
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
        if old := outdated(lib):
            for obj in old:
                for f in (obj, obj.with_name(f"{obj.name}.hash"), obj.with_name(f"{obj.name}.trace")):
                    f.unlink(missing_ok=True)
            names = [str(obj.relative_to(lib.parent.parent / "ir")) for obj in old]
            log(f"lean-records: {p}: {len(names)} object(s) older than the C Lake compiles them from, dropped: "
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


def kept_tree(tree: Path) -> bool:
    """Whether ``tree`` is one of the trees `--records` and `--try` keep: a directory under `RECORDS` in the check cache, or
    a link in `RECORDS` onto another volume, as a node makes a slot's (`made`). Such a link leads out of the check cache;
    one that leads to another part of it is not kept."""
    cache, at, to = LA.COMMON.cache_dir().resolve(), Path(os.path.abspath(tree)), tree.resolve()
    base = (cache / RECORDS).resolve()
    return base in to.parents or (at.parent.resolve() == base and cache != to and cache not in to.parents)


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
    Two packages can read the same files (`security` and `security/Properties`, each in the other's `closure`), and one's pass
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
                 records: dict[str, dict] | None = None, passes: dict[str, str] | None = None,
                 allow: list[str] | None = None, stale: list[str] | None = None) -> None:
    """A recorded run's `result.json`, of kind ``FAST_KIND``: what `research merge` and `research queue`'s records gate read
    of the Attempt. ``outcome`` is one of `OUTCOMES`, and its `validation.status` is `passed` exactly when it is `PASS` or
    `REGENERATED`; ``records`` is each package's (`verdicts`), and ``passes`` {inputs key: package} each key a package's
    records passed on in this run (`passed`), which the store indexes as `passes.<key>` (`stored`). ``path``: `slot` (a
    `build` slot's tree), `dev-cache` (`dev_records`) or `reused` (no package ran: `reused`). A run given `--allow` also
    says what it was given (``allow``, each `CODE:ENTRY`, in its fingerprint too) and which of those matched no failure
    (``stale``, `stale_allowances`)."""
    rel = os.path.relpath(out, run_dir)
    ran = ("nothing ran" if outcome == REFUSED
           else f"{len(stale)} allowance(s) matched no failure ({', '.join(stale)}), which fails the run" if stale
           else "nothing ran: every package's records passed on the same Lean inputs or are a merge's candidate" if path == "reused"
           else f"every failure left on the {path} path is one --allow named, which is no pass (exit {rc}; no pass is kept "
                f"for such a package, and its regenerated record is kept under {ALLOWED_RECORD}/)" if outcome == ALLOWED
           else f"check.py's first failing exit {rc} on the {path} path" if rc else f"{outcome} on the {path} path")
    doc = {"schema": "research/result/v0.1", "kind": FAST_KIND, "run_id": run_id, "commit": git("rev-parse", "HEAD"),
           "outcome": outcome, "workload_fingerprint": {"package": pkg, "update": update, "replay": False, "runs": False,
                                                        "path": path, **({"allow": allow} if allow else {})},
           "validation": {"status": "passed" if outcome in (PASS, REGENERATED) else "failed",
                          "checker": "tools/verity/lean/lean_changed.py --records",
                          "detail": f"{ran}, without the kernel replay or the tests: not a Lean verdict, and no merge accepts "
                                    "it"},
           "records": records or {}, "passes": passes or {},
           **({"allow": allow, "stale_allowances": stale or []} if allow else {}),
           "artifacts": [] if rel.startswith("..") or not out.exists() else [rel]}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run_dir / "result.json")


def finish(pkg: str, outcome: str, rc: int, out: Path, update: bool, path: str, records: dict[str, dict], log=print,
           passes: dict[str, str] | None = None, allow: list[str] | None = None, stale: list[str] | None = None) -> int:
    """End a `--records` run: in a recorded run its `result.json` (`write_result`), and on every path its last line,
    `OUTCOME_LINE` and JSON (``outcome``, ``pkg``, ``records``, ``passes``, and for a run given `--allow` what it was given,
    ``allow``, and which of those matched no failure, `stale`); ``rc``, its exit status."""
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if run_dir and run_id:
        write_result(Path(run_dir), run_id, pkg, outcome, rc, out, update, path, records, passes, allow, stale)
    log(OUTCOME_LINE + json.dumps({"outcome": outcome, "package": pkg, "records": records, "passes": passes or {},
                                   **({"allow": allow, "stale": stale or []} if allow else {})}, sort_keys=True))
    return rc


def allowances(args: list[str]) -> tuple[frozenset[tuple[str, str]], str | None]:
    """`--allow CODE:ENTRY...` as {(code, entry)}, each a failure's `Failure.code` and `Failure.entry` as check.py's report
    lists them (`failure_codes`, `failure_entries`), never its text; and why the run is refused, or None: an argument that
    is not a code and an entry, or a code outside `ALLOWABLE`, the closed list."""
    got = set()
    never = ("never a record difference (which --update settles), the build, the replay, an axiom, `sorry`, `native_decide`, a "
             "kernel bypass or a malformed lock")
    for a in args:
        code, sep, entry = a.partition(":")
        if not (sep and code and entry):
            return frozenset(), (f"--allow {a}: an allowance is CODE:ENTRY, a failure's code and the entry it is about, as "
                                 f"check.py's report lists them (`failure_codes`, `failure_entries`), never its text; the codes "
                                 f"that may be allowed are {', '.join(sorted(ALLOWABLE))}")
        if code not in ALLOWABLE:
            what = "a record difference, which only --update settles" if code in AUDIT.RECORD_DIFFERENCE else \
                "not a failure that may be allowed"
            return frozenset(), (f"--allow {a}: {code} is {what}: the codes that may be allowed are "
                                 f"{', '.join(sorted(ALLOWABLE))}, a closed list in lean_changed.py (`ALLOWABLE`), {never}")
        got.add((code, entry))
    return frozenset(got), None


def named(allow: frozenset[tuple[str, str]]) -> list[str]:
    """``allow`` as `--allow` takes it: each `CODE:ENTRY`, sorted."""
    return sorted(f"{c}:{e}" for c, e in allow)


def failures_of(rep: dict) -> list[tuple[str | None, str | None]]:
    """(code, entry) of each failure in check.py's report ``rep`` on a package, in its order (`failure_codes` and
    `failure_entries` beside `failures`); (None, None) for each where the report lacks either list or one is not as long as
    its failures, so that such a failure is neither a record difference nor one an allowance names."""
    fails = rep.get("failures") or []
    codes, entries = rep.get("failure_codes"), rep.get("failure_entries")
    if not (isinstance(fails, list) and isinstance(codes, list) and isinstance(entries, list)
            and len(codes) == len(entries) == len(fails)):
        return [(None, None)] * (len(fails) if isinstance(fails, list) else 1)
    return [(c if isinstance(c, str) else None, e if isinstance(e, str) else None) for c, e in zip(codes, entries)]


def unmatched(allow: frozenset[tuple[str, str]], audited: list[Path], log=print) -> list[str]:
    """Those of ``allow`` (each as `CODE:ENTRY`) that no failure has in the reports check.py left under ``audited``, this
    run's (a run that built nothing has none, so every allowance is one): stale allowances, each said with the allowable
    failures the run did find, and each a failure of the run, so that a cured finding loses its allowance."""
    seen: set[tuple[str | None, str | None]] = set()
    for d in audited if allow else []:
        try:
            reps = json.loads((d / "audit.json").read_text()).get("packages") or []
        except (OSError, ValueError, AttributeError) as e:
            log(f"lean-records: check.py's reports under {d} can't be read ({e!r}), so no allowance matches a failure of them")
            continue
        for r in reps if isinstance(reps, list) else []:
            seen.update(failures_of(r) if isinstance(r, dict) else [])
    stale = named(allow - seen)
    if stale:
        found = sorted(f"{c}:{e}" for c, e in seen if c in ALLOWABLE and e is not None)
        log(f"lean-records: --allow {', '.join(stale)}: no failure of this run has that code and entry, and an allowance "
            f"that matches no failure fails the run: drop it where the finding is cured, name the failure as its report "
            f"does, or, where its package's records needed no run or its audit stopped before that rule, run that "
            f"(--rebuild) or cure what stopped it; "
            + (f"the failures this run found that may be allowed: {', '.join(found)}" if found else
               "this run found no failure that may be allowed"))
    return stale


def differences(out: Path, base: Path, log=print,
                allow: frozenset[tuple[str, str]] = frozenset()) -> tuple[dict[str, Path], dict[str, list[str]]] | None:
    """Of the packages check.py failed in its reports under ``out``, each by its path from ``base``: ({one whose every
    failure is a record difference: the record check.py wrote that settles it} (its report then names that record,
    `regenerated`, which is what `--update` writes), {one whose every failure ``allow`` names by code and entry: those, each
    `CODE:ENTRY`}); None when a package fails any other way, or the reports can't be read (said). A package that fails on
    record differences beside allowed failures has no such record without `--update`, and an allowance never covers a
    record difference: it is of another kind, and the line says which run settles it."""
    try:
        bad = [r for r in json.loads((out / "audit.json").read_text())["packages"] if r.get("failures")]
        diff: dict[str, Path] = {}
        allowed: dict[str, list[str]] = {}
        for r in bad:
            q = os.path.relpath(Path(str(r["package"])).resolve(), base.resolve())
            fs = failures_of(r)
            if isinstance(r.get("regenerated"), str) and Path(r["regenerated"]).is_file():
                diff[q] = Path(r["regenerated"])
            elif allow and all(f in allow for f in fs):
                allowed[q] = sorted({f"{c}:{e}" for c, e in fs})
            else:
                if allow and all((c, e) in allow or c in AUDIT.RECORD_DIFFERENCE for c, e in fs):
                    log(f"lean-records: {q}: besides the failures --allow names, its records fail only on record "
                        f"differences, which no allowance covers and check.py --update settles: run `--records {q} "
                        f"--update` with the same --allow")
                return None
        return diff, allowed
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        log(f"lean-records: check.py's reports under {out} can't be read ({e!r}), so its failures count as failures of "
            "another kind")
        return None


def hits(out: Path | None, base: Path, allow: frozenset[tuple[str, str]], log=print) -> dict[str, list[str]]:
    """{package, by its path from ``base``: the failures ``allow`` names (each `CODE:ENTRY`) in any report on it that check.py
    left under ``out``}, whichever package's audit reported it: a package in it has an allowed failure, so no pass of it
    is kept. A report that can't be read is said and left out."""
    got: dict[str, set[str]] = {}
    for a in [out / "audit.json", *sorted(out.glob("requirer-*/audit.json"))] if out and allow else []:
        try:
            reps = json.loads(a.read_text()).get("packages") or []
        except (OSError, ValueError, AttributeError) as e:
            log(f"lean-records: {a} can't be read ({e!r}); the failures it reports are not matched against --allow")
            continue
        for r in reps if isinstance(reps, list) else []:
            q = os.path.relpath(Path(str(r.get("package"))).resolve(), base.resolve()) if isinstance(r, dict) else ".."
            if not q.startswith("..") and (mine := {f"{c}:{e}" for c, e in failures_of(r) if (c, e) in allow}):
                got.setdefault(q, set()).update(mine)
    return {q: sorted(fs) for q, fs in got.items()}


def withheld(outcomes: dict[str, str], ok: list[tuple[str, int]], out: Path, base: Path,
             allow: frozenset[tuple[str, str]], log=print) -> list[tuple[str, int]]:
    """``ok`` (each package whose audit did not fail, with check.py's exit) less every package that ended `ALLOWED`: the
    packages whose pass a run keeps. A package with an allowed failure in any of the run's reports (`hits`) ends `ALLOWED`
    in ``outcomes`` where it was better: one that its own audit passed and another package's audit, which audits it
    beside, failed on a failure ``allow`` names."""
    for q in hits(out, base, allow, log):
        if outcomes.get(q) in (PASS, REGENERATED):
            outcomes[q] = ALLOWED
            log(f"lean-records: {q}: another package's audit reports a failure of it that --allow names, so no pass of it "
                f"is kept")
    return [(p, got) for p, got in ok if outcomes[p] not in (ALLOWED, FAILED)]


def keep(q: str, pkg: str, record: bytes, out: Path, allowed: bool = False) -> bytes:
    """Keep ``record``, the record a run wrote for package ``q``, under ``out``'s `record/<q>/` (PKG's also as
    `record/lean-audit.json`, the path `research queue`'s records gate reads for PKG); for a package that ended `ALLOWED`
    (``allowed``), under ``out``'s `ALLOWED_RECORD/<q>/` alone, and nowhere under `record/`. Its bytes."""
    if allowed:
        unkeep(q, pkg, out)
    for k in [out / ALLOWED_RECORD / q / LOCK] if allowed else [out / KEPT_RECORD / q / LOCK,
                                                               *([out / KEPT_RECORD / LOCK] if q == pkg else [])]:
        k.parent.mkdir(parents=True, exist_ok=True)
        k.write_bytes(record)
    return record


def unkeep(q: str, pkg: str | None, out: Path) -> None:
    """Remove what `keep` kept of ``q``'s record under ``out``, in `record/` and under `ALLOWED_RECORD`."""
    for k in [out / KEPT_RECORD / q / LOCK, out / ALLOWED_RECORD / q / LOCK, *([out / KEPT_RECORD / LOCK] if q == pkg else [])]:
        k.unlink(missing_ok=True)


def settled(p: str, pkg: str, got: int, mine_out: Path, base: Path, out: Path, new: dict[str, bytes],
            own: bytes | None = None, log=print, allow: frozenset[tuple[str, str]] = frozenset()) -> str:
    """``p``'s outcome once check.py on it exited ``got``, its reports under ``mine_out`` and its packages under ``base``,
    ``own`` the record its `--update` wrote where that differs from the committed one (`checked`; in ``new``): `PASS`;
    `REGENERATED` when ``own`` or every failure in its reports is a record difference (`differences`), each such record then
    kept under `record/` (`keep`) and put in ``new``; `ALLOWED` when every failure left in its reports is a record
    difference of another package or one ``allow`` names, its own record then kept under `ALLOWED_RECORD` and never under
    `record/`, which holds only records the gate may take; `FAILED` otherwise, with ``own`` kept nowhere."""
    found = ({}, {}) if not got else None if AUDIT.killed(got) else differences(mine_out, base, log, allow)
    if found is None or (got and not (found[0] or found[1])):
        if p != pkg:
            log(f"lean-records: {p} requires {pkg}, and its records fail on this build, not only on record differences, "
                f"which take a person (reports in {mine_out})")
        if own is not None:
            new.pop(f"{p}/{LOCK}", None)
            log(f"lean-records: {p}: its records fail on this build, not only on record differences, so the record check.py "
                "--update wrote is not kept")
        return FAILED
    diff, allowed = found
    mine = {**({p: own} if own is not None else {}),
            **{q: src.read_bytes() for q, src in diff.items() if not (own is not None and q == p)}}
    for q, record in mine.items():
        new[f"{q}/{LOCK}"] = keep(q, pkg, record, out, allowed=bool(allowed) and q == p)
    for q in mine:
        if allowed and q == p:
            log(f"lean-records: {q}: the regenerated record is {out / ALLOWED_RECORD / q / LOCK}, outside {KEPT_RECORD}/, "
                f"since this run allowed a failure of its audit and keeps no pass of it: commit it over {q}/{LOCK} to go on, "
                f"and run without --allow once the allowed findings are cured")
        else:
            log(f"lean-records: {q}: the regenerated record is {out / KEPT_RECORD / q / LOCK}; commit it over {q}/{LOCK}")
    if allowed:
        log(f"lean-records: {p}: every failure left in its audit is one --allow names ("
            + "; ".join(f"{q}: {', '.join(fs)}" for q, fs in sorted(allowed.items()))
            + f"): still a failure in its report, so no pass of {p} is kept and the run exits {ALLOWED_RC}")
        return ALLOWED
    if diff:
        log(f"lean-records: {p}: every failure of its records is a record difference, which the kept record"
            f"{'s settle' if len(diff) > 1 else ' settles'}: not a failure of the run")
    return REGENERATED if mine else PASS


def verdicts(todo: dict[str, str], new: dict[str, bytes], out: Path | None, base: Path, files: list[str],
             reused_: dict[str, tuple[str, dict | None]] | None = None, log=print,
             committed: dict[str, bytes | None] | None = None,
             allow: frozenset[tuple[str, str]] = frozenset(), pkg: str | None = None) -> dict[str, dict]:
    """`result.json`'s `records`: {package: {`outcome`, `inputs_key` (this checkout's as the run found it, its locks read as
    ``committed`` holds them), and, when this run kept its record, `record` (its path in the run's files, else under ``out``)
    and `record_sha256`; `allowed` (each `CODE:ENTRY`) for one whose reports hold failures ``allow`` names; `reused` (why)
    for one that needed no run, and `reused_from` ({`run`, `commit`}) when a records
    run's pass says so}}, for each package of ``todo`` ({package: its outcome}), each of ``reused_`` (`PASS`), and each
    package check.py audited beside one, its outcome read from its report under ``out`` (`PASS` when it failed nothing,
    `REGENERATED` when it failed only on record differences, which check.py's `regenerated` says, and this run kept its
    record, `ALLOWED` when ``allow`` names its every failure, `FAILED` otherwise), the worse when it was audited twice;
    with no ``out`` (nothing ran), none of those. A record
    kept of a package that ends `FAILED` (audited beside one that settled it, and failing otherwise beside another) is
    removed, and one of a package that ends `ALLOWED` is kept under `ALLOWED_RECORD` alone (``pkg``: the run's PKG, whose
    record `keep` also puts at `record/`'s top): `record/` holds only `REGENERATED` packages'."""
    got = {**{p: PASS for p in reused_ or {}}, **todo}
    named_ = hits(out, base, allow, lambda _s: None)
    for a in [out / "audit.json", *sorted(out.glob("requirer-*/audit.json"))] if out else []:
        try:
            reps = json.loads(a.read_text()).get("packages") or []
        except (OSError, ValueError, AttributeError) as e:
            log(f"lean-records: {a} can't be read ({e!r}); the packages it reports beside this run's are left out of its "
                "records")
            continue
        for r in reps if isinstance(reps, list) else []:
            q = os.path.relpath(Path(str(r.get("package"))).resolve(), base.resolve()) if isinstance(r, dict) else ".."
            if q.startswith(".."):
                continue
            fs = failures_of(r)
            if q in todo:
                continue
            o = PASS if not r.get("failures") else REGENERATED if r.get("regenerated") and f"{q}/{LOCK}" in new else \
                ALLOWED if allow and all(f in allow for f in fs) else FAILED
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
            kept = out / (ALLOWED_RECORD if o == ALLOWED else KEPT_RECORD) / q / LOCK
            if o == FAILED:
                unkeep(q, pkg, out)
                log(f"lean-records: {q}: its records fail beside another package, not only on record differences, so its "
                    f"record is not kept")
            else:
                if o == ALLOWED and ((out / KEPT_RECORD / q / LOCK).is_file() or not kept.is_file()):
                    keep(q, pkg, new[f"{q}/{LOCK}"], out, allowed=True)
                    log(f"lean-records: {q}: one of its audits holds a failure --allow names, so its regenerated record is "
                        f"{kept}, outside {KEPT_RECORD}/")
                rec |= {"record": os.path.relpath(kept, run_dir) if run_dir else str(kept),
                        "record_sha256": hashlib.sha256(new[f"{q}/{LOCK}"]).hexdigest()}
        if q in named_:
            rec["allowed"] = named_[q]
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


def conclude(pkg: str, records: dict[str, dict], rc: int, out: Path, update: bool, path: str, log=print,
             passes: dict[str, str] | None = None, allow: frozenset[tuple[str, str]] = frozenset(),
             audited: list[Path] | None = None) -> int:
    """End a run whose packages' ``records`` are judged (`finish`): its outcome their worst (`overall`), `FAILED` too when
    an allowance of ``allow`` matched no failure in the reports this run left under ``audited`` (`unmatched`); its exit
    status 0 on `PASS` and `REGENERATED`, `ALLOWED_RC` on `ALLOWED`, and ``rc`` (check.py's first failing status), else 1,
    on `FAILED`."""
    outcome = overall(records)
    if stale := unmatched(allow, audited or [], log):
        outcome = max(outcome, FAILED, key=OUTCOMES.index)
    if outcome == ALLOWED:
        log(f"lean-records: every failure left is one --allow named, each still in its package's report: exit {ALLOWED_RC}, "
            f"which is no pass; check, research merge and the records gate take nothing from this run for "
            f"{', '.join(q for q, r in records.items() if r['outcome'] == ALLOWED)}")
    return finish(pkg, outcome, (rc or 1) if outcome == FAILED else ALLOWED_RC if outcome == ALLOWED else 0, out, update,
                  path, records, log, passes, named(allow), stale)


def audit_records(name: str, deps: Path, tree: Path | None = None, update: bool = False, moved: Path | None = None,
                  out: Path | None = None, log=print, dev_cache: bool = False, rebuild: bool = False,
                  allow: list[str] | None = None) -> int:
    """`--records`: `check.py --build --no-replay --no-runs` on the package ``name`` names, in ``tree`` (default: the tree of the
    `build` slot it takes; any other must be under `RECORDS`), or `dev_records` where the node's Lean slots grant no `build`
    slot or ``dev_cache`` asks for it. Its outcome is its packages' worst (`settled`, `verdicts`), and `conclude` says it: it
    exits 0 on `PASS` and on `REGENERATED`, `ALLOWED_RC` on `ALLOWED`, check.py's first failing status on `FAILED`, and 2 on
    `REFUSED`, when ``tree`` is
    not under `RECORDS`, ``name`` matches no single package, an allowance of ``allow`` (`--allow CODE:ENTRY`) is one that may
    not be given (`allowances`), the memory refusal stops it or another run holds the tree past
    `TREE_WAIT_S`. A package whose records need no run (`reused`) is skipped, and with none left nothing runs and no slot is
    taken; ``rebuild`` and ``moved`` run them all. Before it takes a slot, one line names the packages it audits, in
    order (`audits`). Each package whose records passed or were regenerated is kept (`passed`); one that ended `ALLOWED`,
    an allowed failure in its audit's reports, is not, so the next run audits it again.
    What each check.py's builds compiled goes into the node's artifact cache when it ends, whatever its exit (`put_built`: the
    package's, and those of the provers it audits), and what a run that was killed left goes in when the next run takes
    the tree (`put_left`). In a run, the module versions it added go into ``out``
    in bundles for other nodes (`LakeArtifacts.bundle_into`), as a recorded check's do, and on a node whose builds are
    sandboxed the bundles other nodes' runs left, whose URLs `research run` sent it (`sent_urls`), go into that cache first
    (`LakeArtifacts.take`), so its builds restore what they compiled.
    Before each package's audit, the copies of fetched dependencies `LINKS` lists that this run does not keep go, each
    with a line naming the package and why (`drops`). It keeps those of the packages in the `closure` of one it audits,
    each while its entry is the key its manifest and toolchain give now and it holds every package the manifest names;
    the others go when the package about to be audited fetches dependencies."""
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    urls = sent_urls(run_dir, log)
    out = (out or (Path(run_dir) / "lean-audit" if run_dir else Path(tempfile.mkdtemp(prefix="lean-records-")))).resolve()

    def refused(why: str | None, pkg: str = name) -> int:
        if why:
            log(f"lean-records: {why}")
        return finish(pkg, REFUSED, 2, out, update, "slot", {}, log)

    allowed, why = allowances(allow or [])
    if why:
        return refused(why)
    if tree is not None and not kept_tree(tree):
        return refused(f"{tree} is not under {(LA.COMMON.cache_dir() / RECORDS).resolve()}: the fast path builds only on its "
                       "own trees")
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
        return conclude(pkg, verdicts({}, {}, None, ROOT, files, skip, log), 0, out, update, "reused", log, allow=allowed)
    here = ROOT / pkg
    audited = [here, *(q for _, q in AUDIT.provers(here, AUDIT.load(here)))] if (here / AUDIT.POLICY).is_file() else [here]
    if why := AUDIT.memory_refusal(audited, AUDIT.memory_gib()):
        return refused(why, pkg)
    log(f"lean-records: this run audits {audits(todo)}: a `--records` run for any of them beside this one audits it again")
    if dev_cache:
        return dev_records(pkg, todo, update, moved, out, "--dev-cache asks for it", log, files, skip, allowed)
    os.environ.pop(LEAN_SLOT.HELD, None)
    try:
        slot = LEAN_SLOT.hold(FAST_POOL, or_any=False)
    except LEAN_SLOT.Refused as e:
        return dev_records(pkg, todo, update, moved, out, str(e), log, files, skip, allowed)
    artifacts = LA.LakeArtifacts((deps.parent / "lean-artifacts").resolve())
    if urls and sandboxed():
        try:
            log(f"lean-records: {artifacts.take(urls, log)} module version(s) taken from other nodes' bundles")
        except OSError as e:
            log(f"lean-records: no artifact bundle taken: {e}")
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
            put_left(tree, left, artifacts, log)
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
                for other, why in drops(p, todo, tree, links).items():
                    log(f"lean-records: {other}: its fetched dependencies are dropped from {tree} ({why})")
                    unlink(other, tree, links)
                if not link(p, tree, deps, links, log) and (k := LA.deps_key(tree / p)):
                    links[p] = k
                    log(f"lean-records: {p}: its dependencies are fetched cold, once for this tree")
                write_links(tree, links)
                mine_out, up, since = reports(out, pkg, p), first_update(p, pkg, update, moved, log), time.time()
                mark(tree, package=p, phase="check", since=since, out=str(mine_out))
                store, redone = artifacts.store, False

                def audit(with_update: bool) -> int:
                    nonlocal store, redone
                    got = subprocess.run(check_argv(tree / p, mine_out, with_update, moved, p == pkg, artifacts=store,
                                                    facts_cache=facts_dir()), cwd=tree, env=env, pass_fds=fds).returncode
                    if redone or not got or AUDIT.killed(got) or not (bad := mismatched(mine_out, tree, log, allowed)):
                        return got
                    store, redone = None, True
                    for mod, d in bad.items():
                        forget(lib := d / ".lake" / "build" / "lib" / "lean", lib / LA.MK.path_of(mod, ".trace"))
                    label(aside := set_aside(mine_out, out, p))
                    log(f"lean-records: {p}: check.py's sampled recompile compiled {', '.join(sorted(bad))} to other "
                        f"outputs than {tree} kept (Lean's output isn't reproducible for every module); each loses its "
                        f"trace, its .hash files and its objects, and check.py audits {p} once more, where Lake compiles "
                        f"them, with nothing restored from the artifact cache (the first audit's reports: {aside})")
                    return audit(with_update)

                got = audit(up)
                if again := not up and regenerates(p, pkg, got, update, mine_out, tree / p, log, allowed):
                    up, got = True, audit(True)
                put_built(artifacts, LA.built(mine_out), since, log, f"lean-records: {p}")
                if AUDIT.killed(got):
                    killed[p] = got
                label(mine_out)
                log(f"lean-records: {p}: check.py{' --update' if again else ''} exit {got} in {time.monotonic() - t0:.0f} s on "
                    f"{tree} ({n} files changed there{', audited twice' if redone else ''}), without the kernel replay or "
                    f"the tests: a {FAST_KIND} result, which check and research merge refuse; reports in {mine_out}")
                # Keep the regenerated record now: the next requirer iteration's `sync` restores the committed one over
                # it, and a recorded run never writes ROOT (the commit's shared source tree), so without this copy the
                # record a lane must commit survives nowhere the run keeps.
                if (b := checked(p, got, tree / p / LOCK, committed[f"{p}/{LOCK}"], up, log)) is not None:
                    new[f"{p}/{LOCK}"] = b
                outcomes[p] = settled(p, pkg, got, mine_out, tree, out, new, b, log, allowed)
                if b is not None and outcomes[p] in (PASS, REGENERATED):
                    write_back(p, tree / p / LOCK, log)
                if outcomes[p] == FAILED:
                    rc = rc or got
                else:
                    ok.append((p, got))
            ok = withheld(outcomes, ok, out, tree, allowed, log)
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
                    log(f"lean-records: {LA.bundled(*artifacts.bundle_into(out), len(artifacts.added))}")
                except OSError as e:
                    log(f"lean-records: nothing bundled for other nodes: {e}")
            records = verdicts(outcomes, new, out, tree, files, skip, log, committed, allowed, pkg)
            return conclude(pkg, records, rc, out, update, "slot", log, passes, allowed, [reports(out, pkg, p) for p in todo])
    finally:
        if slot is not None:
            os.close(slot)
            os.environ.pop(LEAN_SLOT.HELD, None)


def sent_urls(run_dir: str | None, log=print) -> dict[str, str]:
    """The presigned URLs `research run` sent this run (`inputs/lean-deps-urls.json`, `lean_audit.url_file`), read and deleted
    before anything else runs: each grants a read until it expires. {} outside a run, or without them."""
    sent = Path(run_dir) / "inputs" / LA.URLS if run_dir else None
    if sent is None or not sent.is_file():
        return {}
    try:
        urls = json.loads(sent.read_text())
    except (OSError, ValueError) as e:
        log(f"lean-records: the bundle URLs research run sent can't be read ({type(e).__name__}): none taken")
        urls = {}
    sent.unlink(missing_ok=True)
    return urls if isinstance(urls, dict) else {}


def dev_records(pkg: str, todo: list[str], update: bool, moved: Path | None, out: Path, why: str, log=print,
                files: list[str] | None = None, skip: dict[str, tuple[str, dict | None]] | None = None,
                allow: frozenset[tuple[str, str]] = frozenset()) -> int:
    """`--records` where the node's Lean slots grant no `build` slot (``why``): `check.py --dev-cache --build --no-replay
    --no-runs` on this checkout's ``todo`` (PKG, then its requirers; ``skip`` those `reused` left out), each build restored
    from the published cache, with the same reports, record keep, allowances (``allow``), outcome, exit status and result
    kind as the slot path;
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
        if again := not up and regenerates(p, pkg, got, update, mine_out, ROOT / p, log, allow):
            up, got = True, subprocess.run(check_argv(ROOT / p, mine_out, True, moved, p == pkg, dev=True,
                                                      facts_cache=facts_dir()), cwd=ROOT).returncode
        label(mine_out)
        log(f"lean-records: {p}: check.py --dev-cache{' --update' if again else ''} exit {got} in {time.monotonic() - t0:.0f} s "
            f"on {ROOT}, without the kernel replay or the tests: a {FAST_KIND} result, which check and research merge refuse; "
            f"reports in {mine_out}")
        if (b := checked(p, got, ROOT / p / LOCK, committed[f"{p}/{LOCK}"], up, log)) is not None:
            new[f"{p}/{LOCK}"] = b
        outcomes[p] = settled(p, pkg, got, mine_out, ROOT, out, new, b, log, allow)
        if outcomes[p] == FAILED:
            rc = rc or got
        else:
            ok.append((p, got))
    for p, got in withheld(outcomes, ok, out, ROOT, allow, log) if files is not None else []:
        passes |= dict.fromkeys(passed(p, files, committed if got == 0 else committed | new, new, log), p)
    records = verdicts(outcomes, new, out, ROOT, files, skip, log, committed, allow, pkg) if files is not None else \
        {p: {"outcome": o, "inputs_key": None} for p, o in outcomes.items()}
    return conclude(pkg, records, rc, out, update, "dev-cache", log, passes, allow, [reports(out, pkg, p) for p in todo])


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
    the tree's dependencies can't be had, or Lake wanted to write in them (`refused_write`: its line names the path and the
    records run that builds them writable). Never check.py. What the build compiled in the sandbox goes into the node's
    artifact cache (`put_built`); nothing is restored from it here, which only check.py does, with its sampled recompile."""
    file = file.removeprefix("./")
    if tree is not None and not kept_tree(tree):
        log(f"lean-try: {tree} is not under {(LA.COMMON.cache_dir() / RECORDS).resolve()}, where the trees `--records` and "
            "`--try` keep are")
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
            artifacts = LA.LakeArtifacts((deps.parent / "lean-artifacts").resolve())
            put_left(tree, left, artifacts, log)
            mark(tree, package=pkg, phase="sync")
            n = sync(closure(pkg), files, tree)
            there = [q for q in [*closure(pkg), (left or {}).get("package")] if isinstance(q, str) and (tree / q).is_dir()]
            repair(tree, list(dict.fromkeys(there)), links, left, log)
            write_links(tree, links)
            mark(tree, package=pkg, phase="deps")
            sandbox = sandboxed()

            def unfetchable(why: str) -> int:
                log(f"lean-try: {file}: not elaborated, and this says nothing about the file: {why}, and the sandbox can't "
                    f"fetch them, so nothing ran; {fetched_by(pkg, tree)}")
                (tree / DIRTY).unlink(missing_ok=True)
                return 2

            drop_incomplete(pkg, tree, links, log)
            packages = tree / pkg / ".lake" / "packages"
            # dependencies a `--records` run did not take (not in LINKS) are kept as they are, and stay out of LINKS, whose
            # packages the next `--records` run on another package drops
            if not (packages.is_dir() and pkg not in links) and not link(pkg, tree, deps, links, log):
                if sandbox:
                    return unfetchable(f"this node keeps no dependencies for {pkg} to take")
                links[pkg] = LA.deps_key(tree / pkg)
                log(f"lean-try: {pkg}: its dependencies are fetched cold, once for this tree")
            write_links(tree, links)
            if sandbox and (what := lacks(pkg, tree)):
                return unfetchable(f"the dependencies this node keeps for {pkg} lack {what}")
            since, storing = time.time(), storable(sandbox)
            mark(tree, package=pkg, phase="try")
            opts = SIBLINGS.options(tree / pkg)
            log(f"lean-try: {file} on {tree} ({n} files changed there, {time.monotonic() - t0:.0f} s; "
                f"{'' if sandbox else 'no sandbox here, '}lake build +{mod}:deps, then lake env lean "
                f"{''.join(o + ' ' for o in opts)}on {PROBE.COPIES / rel}, {rel} with the axiom probe after its last line"
                f"{', then the sibling check' if siblings else ''})")
            step, (rc, text) = f"lake build +{mod}:deps", build(pkg, tree, lake, [f"+{mod}:deps"], sandbox)
            if storing:                                 # before `lean` reads the file itself, whose code then runs
                put_built(artifacts, {tree / pkg: {"sandbox": True, "compiled": compiled(text)}}, since, log, f"lean-try: {pkg}")
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
                        lambda copy, show: build(pkg, tree, lake, sandbox=sandbox, cmd=["env", "lean", *args, copy],
                                                 echo=show))
                log(f"lean-try: {file}: {step} exit {rc} in {time.monotonic() - t0:.0f} s")
                if rc and (healed := clashed(tree, closure(pkg), text)):
                    log(f"lean-try: {file}: an import clash between {' and '.join(healed)}, which this tree can leave when "
                        "one was built against another version of a module both import: they lose their traces, so the "
                        "next try rebuilds them; a clash that comes back is in the source")
                if rc and (path := refused_write(text, pkg, tree)):
                    log(f"lean-try: {file}: not elaborated, and this says nothing about the file: Lake wanted to write "
                        f"{path}, in the dependencies this tree fetched, which a try mounts read-only so that no try "
                        f"changes what the next one builds against; `lean_changed.py --records {pkg} --rebuild` here builds "
                        "them where Lake may write, on the tree it takes")
                    rc = 2
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
    this node doesn't keep, or keeps without one its manifest names (`lacks`), is skipped, never fetched cold. Its last
    line is `BUILD_LINE` and a JSON object: `built` ({package:
    lake's exit status}), `failed` ({file: its first Lean error}, in a tracked `.lean` file; an import clash this tree left is
    healed as `--try` heals it and not counted) and `skipped` ({package, file or `*`: why it wasn't built}). Exit 0 when
    every build passed, 1 when one failed, 2 when none ran. Without Lean slots here (not a node) it builds nothing. Without
    ``wait`` (`--no-wait`) it builds nothing when every `build` slot is taken, rather than wait in line: the preflight then
    builds on its next node. What each build compiled in the sandbox goes into the node's artifact cache, as `--try`'s."""
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
            artifacts, storing = LA.LakeArtifacts((deps.parent / "lean-artifacts").resolve()), storable(sandbox)
            put_left(tree, left, artifacts, log)
            for pkg, fs in sorted(by_pkg.items(), key=lambda kv: (rank(kv[0]), kv[0])):
                t0 = time.monotonic()
                mark(tree, package=pkg, phase="sync")
                n = sync(closure(pkg), tracked, tree)
                there = [q for q in [*closure(pkg), (left or {}).get("package")] if isinstance(q, str) and (tree / q).is_dir()]
                repair(tree, list(dict.fromkeys(there)), links, left, log)
                left = None
                write_links(tree, links)
                mark(tree, package=pkg, phase="deps")
                drop_incomplete(pkg, tree, links, log)
                packages = tree / pkg / ".lake" / "packages"
                if not (packages.is_dir() and pkg not in links) and not link(pkg, tree, deps, links, log):
                    doc["skipped"][pkg] = ("this node keeps no dependencies (Mathlib) for its manifest, and they aren't fetched "
                                           f"cold; {fetched_by(pkg, tree)}")
                    continue
                write_links(tree, links)
                if what := lacks(pkg, tree):
                    doc["skipped"][pkg] = (f"the dependencies this node keeps for its manifest lack {what}, and they aren't "
                                           f"fetched cold; {fetched_by(pkg, tree)}")
                    continue
                since = time.time()
                mark(tree, package=pkg, phase="build")
                mods = [f"+{module_of(pkg, f)}" for f in fs]
                log(f"lean-build: {pkg} on {tree} ({n} files changed there): lake build {' '.join(mods)}")
                rc, text = build(pkg, tree, lake, mods, sandbox)
                if storing:
                    put_built(artifacts, {tree / pkg: {"sandbox": True, "compiled": compiled(text)}}, since, log,
                              f"lean-build: {pkg}")
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


def settles(out: Path, at: Path, allow: frozenset[tuple[str, str]] = frozenset()) -> bool:
    """Whether check.py's reports under ``out`` fail only the package at ``at``, and it only on record differences: its report
    names the record `--update` writes (`regenerated`), which check.py does when every failure's code is one of
    `RECORD_DIFFERENCE` (a statement, a `reads` group or an assumption no longer its record), which `--update` settles from
    the build alone. With ``allow``, also when it fails on record differences beside failures ``allow`` names and on
    nothing else: `--update` then settles the differences, and the allowed failures are what is left."""
    try:
        bad = [r for r in json.loads((out / "audit.json").read_text())["packages"] if r.get("failures")]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    if len(bad) != 1 or Path(str(bad[0].get("package"))).resolve() != at.resolve():
        return False
    fs = failures_of(bad[0])
    return bool(bad[0].get("regenerated")) or (bool(allow) and any(c in AUDIT.RECORD_DIFFERENCE for c, _ in fs)
                                               and all((c, e) in allow or c in AUDIT.RECORD_DIFFERENCE for c, e in fs))


def mismatched(out: Path, tree: Path, log=print, allow: frozenset[tuple[str, str]] = frozenset()) -> dict[str, Path]:
    """{module: its package's directory} of each module whose outputs in ``tree`` check.py's sampled recompile found other
    than what its source compiles to (the reports' `sampled` `mismatched`, under ``out``), when every other failure there
    is a record difference (`RECORD_DIFFERENCE`), which `--update` settles, a spec's `proved_in.unpassed`, which a
    prover's failures there cause, or one ``allow`` names by code and entry, which the second audit reports again and
    the run ends `ALLOWED` on; {} when there is none, when another failure is there, when a package it names is
    outside ``tree``, or when the reports can't be read, which it says (``log``). Lean's output is not reproducible for every module (`olean_graph`): a kept build compiled once can
    differ from a recompile of the same source, as a stale one does."""
    found: dict[str, Path] = {}
    try:
        for r in json.loads((out / "audit.json").read_text())["packages"]:
            fails, codes, s = r.get("failures") or [], r.get("failure_codes") or [], r.get("sampled")
            mods = s.get("mismatched") or {} if isinstance(s, dict) else {}
            named = failures_of(r) if allow else [(c, None) for c in codes]
            if len(codes) != len(fails) or codes.count("sampled") != len(mods) or \
                    not all(c in ("sampled", "proved_in.unpassed") or c in AUDIT.RECORD_DIFFERENCE or (c, e) in allow
                            for c, e in named):
                return {}
            found |= {m: Path(d).resolve() for m, d in mods.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        log(f"lean-records: check.py's reports in {out / 'audit.json'} can't be read for the modules its sampled recompile "
            f"failed ({type(e).__name__}: {e}), so none is compiled again")
        return {}
    return found if all(d.is_relative_to(tree.resolve()) for d in found.values()) else {}


def set_aside(mine_out: Path, out: Path, p: str) -> Path:
    """Move check.py's reports on ``p`` (its `audit.json` and `audit-*` directories under ``mine_out``) to
    `mismatched/<path>/` under ``out``, where no reader of the run's verdict looks; that directory."""
    aside = out / "mismatched" / re.sub(r"[^A-Za-z0-9]+", "_", p)
    shutil.rmtree(aside, ignore_errors=True)
    aside.mkdir(parents=True)
    for f in [mine_out / "audit.json", *sorted(mine_out.glob("audit-*"))]:
        if f.exists():
            shutil.move(f, aside / f.name)
    return aside


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


def regenerates(p: str, pkg: str, got: int, update: bool, out: Path, at: Path, log=print,
                allow: frozenset[tuple[str, str]] = frozenset()) -> bool:
    """Whether requirer ``p``, which check.py just audited at ``at`` (exit ``got``, its reports under ``out``), is audited again
    with `--update`: in an `--update` run, when it failed only because what its guarantees read changed (`settles`; with
    ``allow``, beside failures it names), so one run regenerates its lock beside PKG's instead of failing on it."""
    if p == pkg or not update or not got or AUDIT.killed(got) or not settles(out, at, allow):
        return False
    log(f"lean-records: {p} requires {pkg}, and every failure of its records is a record difference (what its guarantees "
        f"read changed){'' if settles(out, at) else ' or one --allow names'}: check.py --update on it too")
    return True


def checked(p: str, got: int, record: Path, committed: bytes | None, up: bool, log=print) -> bytes | None:
    """After check.py on ``p`` exited ``got``: when it ran with `--update` (``up``) and the record it wrote (``record``)
    differs from the ``committed`` bytes, that record's bytes, which `settled` keeps where ``p``'s outcome puts them
    (`record/`, `ALLOWED_RECORD`, or nowhere when ``p`` fails otherwise); None when check.py was killed, which leaves no
    record to keep."""
    if not up:
        return None
    if AUDIT.killed(got):
        log(f"lean-records: {p}: check.py was killed, so what it left at {record} is not kept")
        return None
    return b if (b := record.read_bytes()) != committed else None


def write_back(p: str, record: Path, log=print) -> None:
    """Outside a recorded run, which never writes ROOT (the commit's shared source tree), write ``p``'s kept ``record`` into
    this checkout."""
    mine = ROOT / p / LOCK
    if not os.environ.get("RESEARCH_RUN_DIR") and record != mine:
        shutil.copyfile(record, mine)
        log(f"lean-records: wrote {mine}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lean_changed.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--deps", type=Path, default=LA.COMMON.cache_dir() / "lean-deps", help="the audit's kept dependencies")
    ap.add_argument("--tree", type=Path, help=f"with --records, --try or --build: the slot tree (default: {RECORDS}/build-I, "
                    f"the Lean slot's, under the check cache); it must be under {RECORDS}/")
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
    ap.add_argument("--allow", action="append", default=[], metavar="CODE:ENTRY",
                    help="with --records only, and repeatable: a failure this run takes as known, named by its code and the "
                    f"entry it is about as check.py's report lists them (the codes that may be: {', '.join(sorted(ALLOWABLE))}). "
                    f"It stays a failure in its report; a run whose every other failure is settled exits {ALLOWED_RC} "
                    f"(`{ALLOWED}`), keeps no pass of that package, and keeps its regenerated record under {ALLOWED_RECORD}/, "
                    f"not {KEPT_RECORD}/; an allowance that matches no failure fails the run")
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
    if a.allow and not a.records:
        ap.error("--allow goes with --records, a lane's fast path, and with nothing else: no other mode takes an "
                 "allowance, --try, --build (the ready mark's preflight) and --inputs-key among them")
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
        ap.error("one of --records, --try, --build and --inputs-key is required")
    if a.dev_cache and a.tree:
        ap.error("--dev-cache builds in this checkout, not in a --tree")
    return audit_records(a.records, a.deps, a.tree, a.update, a.moved, a.out, log=lambda s: print(s, flush=True),
                         dev_cache=a.dev_cache, rebuild=a.rebuild, allow=a.allow)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
