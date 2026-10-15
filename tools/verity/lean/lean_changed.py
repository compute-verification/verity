"""`lean_changed`: Lean's early job, which `check` runs as its lean-changed step. `lake build` of every Lake package whose Lean
audit misses (`lean_audit.py`: its inputs changed since a pass kept on this machine), on a tree this machine keeps built, so a
module that no longer compiles (two merges git combines cleanly that Lean doesn't) fails the run in minutes rather than when
the audit's cold build reaches it.

    python tools/verity/lean/lean_changed.py --cache DIR --deps DIR --tree DIR
    python tools/verity/lean/lean_changed.py --records PKG [--update [--moved FILE]] [--out DIR]   a lane's records, on a kept tree

`--records` is a lane's fast path to the records in PKG's `lean-audit.json` (PKG: its path, or a unique part of one):
`tools/verity/lean/check.py --build --no-replay --no-runs` on PKG in a tree kept per slot, which prints what the audit prints,
`--update`'s review included, after a build of only what changed since that slot's last run. It first waits for one of the
node's Lean slots of pool `build` (`lean_slot.py`; refused when SLOTS lists no `build` pool, which pool `any` never stands in
for), never one of the `audit` pool's, where hour-long audits hold every slot, whatever slot (LEAN_SLOT) the process that
started it holds. Slot `build/I`'s tree is `lean-records/build-I` under the check cache, whoever holds that slot's lock owns
it, and audit.py inherits the lock's descriptor, so the slot is held until audit.py exits; with no SLOTS (a laptop) the tree
is `lean-records/local`. `--tree` names another only under `lean-records/`. The tree's fetched dependencies are its own copy
of the audit's (taken once per manifest and toolchain, else fetched cold), never links, so what Lake writes there stays out of
what the audit keeps. That bounds Lake's own writes, not compile-time code: audit.py builds in its sandbox only where the
machine allows one, and without it that code can write anything this user can, the audit's kept dependencies and passes
included, as any build this user runs by hand can. `--update` writes the record back into this checkout, but not in a recorded
run, whose source tree the commit's other runs share: there it is in the run's `lean-audit/audit-*/`.

Then it audits, the same way but never with `--update`, every package whose guarantees' records read PKG's build
(`requirers`: PKG is in its `closure`, by path or by `proved_in`), each after those it requires, its reports under
`requirer-<path>/` beside PKG's. A change to what another package's pinned guarantees read (`Security/Properties`' when
`Security` changes) fails here, not at a train's check; the exit status is the first failure's. A tree keeps the copies of
the packages one run audits (about 170k inodes, 8 GB each), so auditing another package that fetches dependencies there
drops the rest first.

Its result is of kind `FAST_KIND`, never a Lean verdict: no kernel replay and no tests. The kind is written into the
`audit.json` and each package's `report.json` it leaves (`--out`, else the run's `lean-audit/`, else a temporary directory) and,
in a recorded run, into its `result.json`, the Attempt's result. `check` fails its lean-audit step on a report of that kind or
one without the replay (`check.lean_fast`), and `research merge` refuses an attempt whose result is of that kind
(`research.merge.REFUSED_KINDS`), whatever `--tool` recorded it.

The tree (`--tree`) holds each such package's tracked files, and those of the packages it requires by path or names under
`proved_in`, at their paths in this repository, with each package's own `.lake`: Lake rebuilds only the modules that changed since its last build there and
what imports them. A package's fetched dependencies are hard links to the ones the audit keeps (`--deps`, `lean_audit.WarmDeps`),
taken under their lock while the tree has none for its manifest and toolchain; the build runs in the audit's sandbox
(`tools/verity/lean/sandbox.sh`), where they are mounted read-only, so it cannot write what the audit reads, and a concurrent audit can
move them away without touching the links. The first build of a package here is cold; later ones are not.

A package with guarantees then has its roots built (`lake build +ROOT`, as the audit does) and its guarantees' records
compared with that build (`records`): `audit.py`'s own reading of the `.olean` files (`Facts.lean`, in the sandbox) and its
judgment of the guarantees, without the kernel replay, so a record two merges left stale (a statement or a definition a
guarantee reads changed, a guarantee gone, a `reads` digest no longer its definitions') fails the run before the audit's cold
build reaches it.

The audit still builds and audits every package cold and decides the run: a pass here vouches for nothing. So this step fails
only on a Lean error at a position in one of the tree's own `.lean` files, or on a guarantee the audit fails (`judge`'s
`guarantees:` failures). Anything else that stops a build (no sandbox or `lake` here, dependencies this machine doesn't keep
yet, another check holding the tree, a dependency the links lack) skips the package by name and drops its links, so the next
run takes the audit's again; anything that stops a reading (a policy that can't be read, `Facts.lean` failing) skips its
guarantees by name.
`check.py` ends the run when this step fails, even under `--keep-going`. Delete the tree to start it over.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
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


def closure(pkg: str) -> list[str]:
    """``pkg``, the packages it requires by path, and those its `proved_in` names with theirs, whose builds `records` reads."""
    return list(dict.fromkeys([pkg, *LA.path_deps(pkg), *(d for q in LA.provers(pkg) for d in (q, *LA.path_deps(q)))]))


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


def in_sandbox(pkg: str, tree: Path, argv: list[str]) -> list[str]:
    """``argv`` in the sandbox, writable in the `.lake` of ``tree/pkg`` and the packages it requires by path but not in their
    fetched dependencies."""
    rw = [str(tree / p / ".lake") for p in closure(pkg)]
    ro = [str(Path(d) / "packages") for d in rw if (Path(d) / "packages").is_dir()]
    return [*UNSHARE, "--", "bash", str(SANDBOX), *rw, "--", "bash", "-c", READ_ONLY, "ro", *ro, "--", *argv]


def build(pkg: str, tree: Path, lake: str, targets: list[str] | None = None) -> tuple[int, str]:
    """`lake build` (of ``targets``, else the default ones) of ``tree/pkg`` in the sandbox; Lake's output is echoed as it
    comes."""
    argv = in_sandbox(pkg, tree, [lake, "build", *(targets or [])])
    env = {**os.environ, "LEAN_NUM_THREADS": str(LA.COMMON.cores())}
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
    """The audit's failures about the guarantees on the tree's build of ``pkg`` (`audit.py`'s `facts_of` and `judge`, without
    the kernel replay): a guarantee missing, no longer a theorem or with no record, using an axiom the policy does not allow or
    assuming an unnamed `Prop`, whose statement or the definitions it reads no longer match its record, or a `reads` entry
    that no longer lists the modules the guarantees read, the guarantees that read them or their definitions' digest, or a
    named assumption whose record no longer matches the definition the guarantees read. A spec's
    guarantees are read from the builds of the packages its `proved_in` names (`audit.proved`), built before it (`missed`).
    Raises RuntimeError when `Facts.lean` cannot read a build."""
    policy = AUDIT.load(tree / pkg)
    work = tree / pkg / FACTS
    work.mkdir(parents=True, exist_ok=True)
    plain, tool = AUDIT.run, AUDIT.TOOL
    # the sandbox mounts a private /tmp, where this checkout may be; the copy sits in the writable `.lake`, as the audit runs
    # its tool from a copy
    shutil.copy2(tool / "Facts.lean", work / "Facts.lean")

    def lake_in_sandbox(argv: list[str], cwd: Path, *a, **kw) -> tuple[int, str]:
        # the audit runs `lake env` unsandboxed on a tree of its own; here `.lake/packages` holds links to the files it keeps
        return plain(in_sandbox(pkg, tree, [lake, *argv[1:]] if argv[:1] == ["lake"] else argv), cwd, *a, **kw)

    AUDIT.run, AUDIT.TOOL = lake_in_sandbox, work
    try:
        facts = AUDIT.facts_of(tree / pkg, policy, work)
        cross: list[str] = []
        if "proved_in" in policy:
            cross, facts, _ = AUDIT.proved(tree / pkg, policy, facts, work)
    finally:
        AUDIT.run, AUDIT.TOOL = plain, tool
    fails, _ = AUDIT.judge(policy, facts, AUDIT.lean_modules(tree / pkg))
    return [f for f in cross + fails if f.startswith(("guarantees:", "assumptions:"))]


def sandboxed() -> bool:
    try:
        return subprocess.run([*UNSHARE, "--", "true"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


#: `--records`: its trees, one per Lean slot (`build-I`; `local` where the node has no SLOTS), under the check cache
RECORDS = "lean-records"
AUDIT_PY = ROOT / "tools" / "verity" / "lean" / "check.py"
LEAN_SLOT = _module(ROOT / "tools" / "verity" / "check" / "lean_slot.py", "lean_slot_for_lean_changed")
#: the Lean slot pool `--records` takes, apart from the `audit` pool's hour-long audits and the `check` pool's merge checks
FAST_POOL = "build"
#: the kind of `--records`' result: an audit without the kernel replay or the tests, which no merge check accepts
FAST_KIND = "lean-fast/v1"


def records_tree() -> Path:
    """The tree of the Lean slot this process holds (LEAN_SLOT, `build/1` -> `build-1`), else `local`."""
    return LA.COMMON.cache_dir() / RECORDS / (os.environ.get(LEAN_SLOT.HELD) or "local").replace("/", "-")


def label(out: Path) -> None:
    """Write ``FAST_KIND`` into the reports audit.py left under ``out``: `audit.json`, its packages, and each `report.json`."""
    for p in [out / "audit.json", *sorted(out.glob("audit-*/report.json"))]:
        if not p.is_file():
            continue
        doc = json.loads(p.read_text())
        doc["kind"] = FAST_KIND
        for rep in doc.get("packages", []):
            rep["kind"] = FAST_KIND
        p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


def write_result(run_dir: Path, run_id: str, pkg: str, rc: int, out: Path, update: bool) -> None:
    """A recorded run's `result.json`, of kind ``FAST_KIND``: what `research merge` reads of the Attempt."""
    rel = os.path.relpath(out, run_dir)
    doc = {"schema": "research/result/v0.1", "kind": FAST_KIND, "run_id": run_id,
           "workload_fingerprint": {"package": pkg, "update": update, "replay": False, "runs": False},
           "validation": {"status": "passed" if rc == 0 else "failed", "checker": "tools/verity/lean/lean_changed.py --records",
                          "detail": f"audit.py exit {rc} without the kernel replay or the tests: not a Lean verdict, and no "
                                    "merge accepts it"},
           "artifacts": [] if rel.startswith("..") else [rel]}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run_dir / "result.json")


def audit_records(name: str, deps: Path, tree: Path | None = None, update: bool = False, moved: Path | None = None,
                  out: Path | None = None, log=print) -> int:
    """`--records`: `audit.py --build --no-replay --no-runs` on the package ``name`` names, in ``tree`` (default: the tree of the
    `build` slot it takes; any other must be under `RECORDS`); audit.py's exit status, or 2 when ``tree`` is not under
    `RECORDS`, ``name`` matches no single package, the node's Lean slots refuse it or another run holds the tree."""
    base = (LA.COMMON.cache_dir() / RECORDS).resolve()
    if tree is not None and base not in tree.resolve().parents:
        log(f"lean-records: {tree} is not under {base}: the fast path builds only on its own trees, never on check's "
            "(lean-changed), whose dependencies are links to the audit's")
        return 2
    files = LA.COMMON.tracked_files(ROOT)
    if files is None:
        log("lean-records: the tree's files cannot be listed")
        return 2
    pkgs = LA.packages(files)
    match = [p for p in pkgs if p == name.rstrip("/")] or [p for p in pkgs if name in p]
    if len(match) != 1:
        log(f"lean-records: {name!r} matches {len(match)} packages: {', '.join(match or pkgs)}")
        return 2
    pkg = match[0]
    os.environ.pop(LEAN_SLOT.HELD, None)
    try:
        slot = LEAN_SLOT.hold(FAST_POOL, or_any=False)
    except LEAN_SLOT.Refused as e:
        log(f"lean-records: {e}")
        return 2
    try:
        tree = tree or records_tree()
        tree.mkdir(parents=True, exist_ok=True)
        with open(tree / ".lock", "w") as held:
            try:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log(f"lean-records: another run holds {tree}")
                return 2
            links = json.loads((tree / LINKS).read_text()) if (tree / LINKS).is_file() else {}
            run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
            out = (out or (Path(run_dir) / "lean-audit" if run_dir else Path(tempfile.mkdtemp(prefix="lean-records-")))).resolve()
            todo, rc = [pkg, *requirers(pkg, pkgs)], 0
            for p in todo:
                t0 = time.monotonic()
                n = sync(closure(p), files, tree)
                for other in [q for q in links if q not in todo and LA.deps_key(tree / p)]:
                    unlink(other, tree, links)
                if not link(p, tree, deps, links, log, copy=True) and (k := LA.deps_key(tree / p)):
                    links[p] = k
                    log(f"lean-records: {p}: its dependencies are fetched cold, once for this tree")
                (tree / LINKS).write_text(json.dumps(links, indent=1, sort_keys=True) + "\n")
                mine_out = out if p == pkg else out / f"requirer-{re.sub(r'[^A-Za-z0-9]+', '_', p)}"
                argv = [sys.executable, str(AUDIT_PY), "--build", "--no-replay", "--no-runs",
                        *(["--update"] if update and p == pkg else []),
                        *(["--moved", str(moved.resolve())] if moved and p == pkg else []), "--out", str(mine_out), str(tree / p)]
                got = subprocess.run(argv, cwd=tree, env={**os.environ, "LEAN_NUM_THREADS": str(LA.COMMON.cores())},
                                     pass_fds=() if slot is None else (slot,)).returncode
                label(mine_out)
                log(f"lean-records: {p}: audit.py exit {got} in {time.monotonic() - t0:.0f} s on {tree} ({n} files changed "
                    f"there), without the kernel replay or the tests: a {FAST_KIND} result, which check and research merge "
                    f"refuse; reports in {mine_out}")
                if p == pkg:
                    record, mine = tree / pkg / "lean-audit.json", ROOT / pkg / "lean-audit.json"
                    if update and record.read_bytes() != mine.read_bytes():
                        # Keep the regenerated record now: the next requirer iteration's `sync` restores the committed
                        # one over `record`, and a recorded run never writes ROOT (the commit's shared source tree), so
                        # without this copy the record a lane must commit survives nowhere the run keeps (LEAN-10).
                        kept = out / "record" / "lean-audit.json"
                        kept.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(record, kept)
                        log(f"lean-records: the regenerated record is {kept}; commit it over {pkg}/lean-audit.json")
                        if not run_dir:
                            shutil.copyfile(record, mine)
                            log(f"lean-records: wrote {mine}")
                elif got:
                    log(f"lean-records: {p} requires {pkg}, and its records fail on this tree: what its guarantees read "
                        f"changed; if that is meant, run --records {p} --update")
                rc = rc or got
            if run_dir and run_id:
                write_result(Path(run_dir), run_id, pkg, rc, out, update)
    finally:
        if slot is not None:
            os.close(slot)
            os.environ.pop(LEAN_SLOT.HELD, None)
    return rc


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
    tree.mkdir(parents=True, exist_ok=True)
    with open(tree / ".lock", "w") as held:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("lean-changed: skipped: another check on this machine holds the tree")
            return 0
        links = json.loads((tree / LINKS).read_text()) if (tree / LINKS).is_file() else {}
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
            (tree / LINKS).write_text(json.dumps(links, indent=1, sort_keys=True) + "\n")
    log(f"lean-changed: {'FAIL (' + ', '.join(failed) + ')' if failed else 'PASS'}: built {len(built)} of the {len(todo)} "
        f"package(s) whose audit misses; compared the guarantees of {len(compared)}")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lean_changed.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=LA.COMMON.cache_dir() / "lean-audit", help="the audit's kept passes")
    ap.add_argument("--deps", type=Path, default=LA.COMMON.cache_dir() / "lean-deps", help="the audit's kept dependencies")
    ap.add_argument("--tree", type=Path, help="the tree this step keeps built (default: lean-changed, or with --records "
                    f"{RECORDS}/build-I, the Lean slot's, under the check cache; with --records it must be under {RECORDS}/)")
    ap.add_argument("--records", metavar="PKG",
                    help=f"audit PKG's records without the kernel replay, on a tree kept per Lean slot of pool {FAST_POOL} "
                    f"(a {FAST_KIND} result, which no merge accepts)")
    ap.add_argument("--update", action="store_true", help="with --records: audit.py --update")
    ap.add_argument("--moved", type=Path, metavar="FILE", help="with --records --update: audit.py --moved FILE")
    ap.add_argument("--out", type=Path, help="with --records: audit.py --out DIR")
    a = ap.parse_args(argv)
    if not a.records:
        if a.update or a.moved or a.out:
            ap.error("--update, --moved and --out go with --records")
        return run(a.cache, a.deps, a.tree or LA.COMMON.cache_dir() / "lean-changed", log=lambda s: print(s, flush=True))
    return audit_records(a.records, a.deps, a.tree, a.update, a.moved, a.out, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
