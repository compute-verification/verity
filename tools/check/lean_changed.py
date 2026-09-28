"""`lean_changed`: check's early Lean step. `lake build` of every Lake package whose Lean audit misses (`lean_audit.py`: its
inputs changed since a pass kept on this machine), on a tree this machine keeps built, so a module that no longer compiles (two
merges git combines cleanly that Lean doesn't) fails the run in minutes rather than when the audit's cold build reaches it.

    python tools/check/lean_changed.py --cache DIR --deps DIR --tree DIR

The tree (`--tree`) holds each such package's tracked files, and those of the packages it requires by path, at their paths in
this repository, with each package's own `.lake`: Lake rebuilds only the modules that changed since its last build there and
what imports them. A package's fetched dependencies are hard links to the ones the audit keeps (`--deps`, `lean_audit.WarmDeps`),
taken under their lock while the tree has none for its manifest and toolchain; the build runs in the audit's sandbox
(`tools/lean/sandbox.sh`), where they are mounted read-only, so it cannot write what the audit reads, and a concurrent audit can
move them away without touching the links. The first build of a package here is cold; later ones are not.

A package with pins then has its roots built (`lake build +ROOT`, as the audit does) and its pins' records compared with that
build (`records`): `audit.py`'s own reading of the `.olean` files (`Facts.lean`, in the sandbox) and its judgment of the pins,
without the kernel replay, so a record two merges left stale (a statement or a definition a pin reads changed, a pin gone, a
`reads` digest no longer its definitions') fails the run before the audit's cold build reaches it.

The audit still builds and audits every package cold and decides the run: a pass here vouches for nothing. So this step fails
only on a Lean error at a position in one of the tree's own `.lean` files, or on a pin the audit fails (`judge`'s `pins:`
failures). Anything else that stops a build (no sandbox or `lake` here, dependencies this machine doesn't keep yet, another
check holding the tree, a dependency the links lack) skips the package by name and drops its links, so the next run takes the
audit's again; anything that stops a reading (a policy that can't be read, `Facts.lean` failing) skips its pins by name.
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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LA = _module(ROOT / "tools" / "check" / "lean_audit.py", "lean_audit_for_lean_changed")
#: `tools/lean/audit.py`, this step's own copy: `records` runs its commands in the sandbox
AUDIT = _module(ROOT / "tools" / "lean" / "audit.py", "lean_audit_tool_for_lean_changed")
#: where `Facts.lean` writes a package's facts: inside its `.lake`, which the sandbox leaves writable and `sync` leaves alone
FACTS = Path(".lake") / "lean-changed-facts"
#: the audit's build sandbox, as `tools/lean/audit.py` runs it: writable only in the `.lake` directories it is given
UNSHARE = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--kill-child"]
SANDBOX = ROOT / "tools" / "lean" / "sandbox.sh"
#: run inside the sandbox before the build: `ro DIR... -- CMD`, each DIR bind-mounted read-only over itself
READ_ONLY = ('ro=(); while [ "$1" != -- ]; do ro+=("$1"); shift; done; shift; '
             'for d in "${ro[@]}"; do mount --bind "$d" "$d" && mount -o remount,bind,ro "$d" || exit 97; done; exec "$@"')
#: Lake's line for a Lean error, the file relative to its package: `error: Aa/Two.lean:2:39: Unknown identifier ...`
ERROR = re.compile(r"^error: (?:\./)*(\S+?\.lean):\d+:\d+: ")
#: {package: the dependencies' key its links were taken under}, beside the packages in the tree
LINKS = "links.json"


def missed(files: list[str], cache: Path) -> list[str]:
    """The packages whose audit would run, those another requires by path first."""
    pkgs = [p for p in LA.packages(files) if not (cache / f"{LA.key(p, files, LA.FLAGS)}.json").is_file()]
    return sorted(pkgs, key=lambda p: (len(LA.path_deps(p)), p))


def closure(pkg: str) -> list[str]:
    return [pkg, *LA.path_deps(pkg)]


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


def link(pkg: str, tree: Path, deps: Path, links: dict[str, str], log=print) -> bool:
    """``tree/pkg/.lake/packages`` as hard links to the audit's kept dependencies for its manifest and toolchain; False when this
    machine keeps none (another run holds them, or none has passed here yet)."""
    if (k := LA.deps_key(tree / pkg)) is None:
        return True
    dst = tree / pkg / ".lake" / "packages"
    if links.get(pkg) == k and dst.is_dir():
        return True
    unlink(pkg, tree, links)
    with LA.WarmDeps(deps)._lock(k):
        src = deps / k
        if not src.is_dir():
            log(f"lean-changed: {pkg}: this machine keeps no dependencies for it ({k[:12]}) to link")
            return False
        t0 = time.monotonic()
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(src, dst, symlinks=True, copy_function=os.link)
        except (OSError, shutil.Error) as e:
            shutil.rmtree(dst, ignore_errors=True)
            log(f"lean-changed: {pkg}: the dependencies ({k[:12]}) could not be linked ({e}); a link needs one filesystem")
            return False
    links[pkg] = k
    log(f"lean-changed: {pkg}: dependencies linked ({k[:12]}) in {time.monotonic() - t0:.0f} s")
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


def pinned_roots(pkg: str, tree: Path) -> list[str]:
    """The roots of ``tree/pkg``'s policy when it pins a theorem, which `records` reads; none otherwise."""
    policy = AUDIT.load(tree / pkg)
    return list(policy["roots"]) if policy.get("pins") else []


def records(pkg: str, tree: Path, lake: str) -> list[str]:
    """The audit's failures about the pins on the tree's build of ``pkg`` (`audit.py`'s `facts_of` and `judge`, without the
    kernel replay): a pin missing, no longer a theorem or with no record, using an axiom the policy does not allow or assuming
    an unnamed `Prop`, whose statement or the definitions it reads no longer match its record, or a `reads` entry that no
    longer lists the modules the pins read, the pins that read them or their definitions' digest. Raises RuntimeError when
    `Facts.lean` cannot read the build."""
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
    finally:
        AUDIT.run, AUDIT.TOOL = plain, tool
    fails, _ = AUDIT.judge(policy, facts, AUDIT.lean_modules(tree / pkg))
    return [f for f in fails if f.startswith("pins:")]


def sandboxed() -> bool:
    try:
        return subprocess.run([*UNSHARE, "--", "true"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


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
                    roots = pinned_roots(pkg, tree)
                except (OSError, ValueError, KeyError) as e:
                    roots = []
                    log(f"lean-changed: {pkg}: its {AUDIT.POLICY} cannot be read ({e!r}); its pins are not compared here")
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
                    log(f"lean-changed: {pkg}: its pins are not compared here: {str(e)[-2000:]}")
                    continue
                took = time.monotonic() - t0
                if stale:
                    failed.append(pkg)
                    log(f"lean-changed: {pkg}: FAILED in {took:.0f} s: the audit fails its pins on this build:")
                    log("\n".join(f"  {line}" for f in stale[:20] for line in f.splitlines()))
                    break
                compared.append(pkg)
                log(f"lean-changed: {pkg}: its pins match their records ({took:.0f} s)")
        finally:
            (tree / LINKS).write_text(json.dumps(links, indent=1, sort_keys=True) + "\n")
    log(f"lean-changed: {'FAIL (' + ', '.join(failed) + ')' if failed else 'PASS'}: built {len(built)} of the {len(todo)} "
        f"package(s) whose audit misses; compared the pins of {len(compared)}")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lean_changed.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=LA.COMMON.cache_dir() / "lean-audit", help="the audit's kept passes")
    ap.add_argument("--deps", type=Path, default=LA.COMMON.cache_dir() / "lean-deps", help="the audit's kept dependencies")
    ap.add_argument("--tree", type=Path, default=LA.COMMON.cache_dir() / "lean-changed", help="the tree this step keeps built")
    a = ap.parse_args(argv)
    return run(a.cache, a.deps, a.tree, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
