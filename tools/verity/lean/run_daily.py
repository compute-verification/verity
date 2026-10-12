#!/usr/bin/env python3
"""Node 1's daily publish of the Lean build cache: `vy-lean-cache-daily.service` runs it as root, with no arguments, from
`/usr/local/lib/vy-lean-cache/run-daily`, beside the `cache.py` it runs. Root holds the key, so it never runs a tree's code.

1. Main's commit is node 1's bare repository's `refs/heads/main`, which `research merge` pushes after GitHub's.
2. `vy-lean-sandbox checkout` makes `/build/verity` a fresh clone of it, its sources root's.
3. As research, in a checkout of the commit kept at PREP, `tools/verity/lean/setup.sh` fetches each package's dependencies: the
   pinned bundle, then Lake, then Mathlib's cache. Root copies them to DEPS/<deps_key>, readable by all, and the build
   gets that copy read-only, with no network. PREP's `.lake` directories are kept across runs (only the checkout moves),
   including bundles unpacked there before the first run.
4. Per package, `vy-lean-sandbox run` builds it as `lean-build` with `cache.py build`. The build writes only the tree's
   `.lake` directories and the package's directory in `/build/out` (`cache.out_dir`), the one below `/build/out` root
   doesn't own. A build that fails has the end of Lake's output logged.
5. `vy-lean-sandbox clean` must find the tree still the commit's, with no build left running. Only then does root read the
   key, with `cache.py publish` of what built, then `cache.py prune`.

A probe that got through (the sandbox's exit 3), a refused sandbox (2) or an unclean tree stops it with nothing published.
A package that doesn't build is left out, and the run exits 1.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import pwd
import shlex
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lean_cache", HERE / "cache.py")
CACHE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CACHE)

MAX_SEC = 2 * 3600
SLOT_WAIT_S = 1800
MAIN = "refs/heads/main"
#: the sandbox's exit codes that stop the run: refused (2), a probe got through (3)
STOP = (2, 3)


def as_research() -> list[str]:
    return ["runuser", "-u", "research", "--", "env", "-i", "HOME=/home/research", "LEAN_CACHE=off",
            "UV_CACHE_DIR=/workspace/cache/uv", "PATH=/home/research/.elan/bin:/usr/local/bin:/usr/bin:/bin"]


@dataclass
class Node:
    """Where everything is on node 1; the tests point it at a temporary directory."""
    bare: Path = Path("/workspace/research/git/verity.git")
    tree: Path = Path("/build/verity")
    out: Path = Path("/build/out")
    prep: Path = Path("/workspace/research/lean-cache/prep")
    deps: Path = Path("/workspace/lean-build/deps")
    #: root replaces entries under DEPS, so every directory from here down to DEPS must be one only root writes
    trusted: Path = Path("/")
    lock: Path = Path("/run/vy-lean-cache-daily.lock")
    sandbox: list[str] = field(default_factory=lambda: ["/usr/local/bin/vy-lean-sandbox"])
    research: list[str] = field(default_factory=as_research)
    builder: str = "lean-build"
    key_file: str = CACHE.KEY_FILE
    python: str = "/usr/bin/python3"


def _run(cmd: list, log, **kw) -> int:
    log("run-daily: $ " + " ".join(shlex.quote(str(c)) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw).returncode


def _out(cmd: list, **kw) -> str:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=True, **kw).stdout.strip()


def owned_dirs(top: Path, leaf: Path) -> None:
    """Make ``top`` and every directory down to ``leaf``'s parent, and refuse any that isn't a real directory only this
    user can write: the build's user then controls ``leaf`` and nothing above it."""
    rel = leaf.parent.relative_to(top).parts
    for d in [top / Path(*rel[:i]) for i in range(len(rel) + 1)]:
        d.mkdir(mode=0o755, parents=True, exist_ok=True)
        st = os.lstat(d)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o022:
            raise SystemExit(f"run-daily: {d} is not a directory only {pwd.getpwuid(os.geteuid()).pw_name} writes")


def log_tail(d: Path, name: str, limit: int = 4096) -> str:
    """The last ``limit`` bytes of ``d/name``, a file the build's user wrote: read without following a link, and only from
    a regular file with one link, so the build can't point it at the key; "" otherwise."""
    try:
        fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            f = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        finally:
            os.close(fd)
    except OSError:
        return ""
    try:
        st = os.fstat(f)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            return ""
        os.lseek(f, max(0, st.st_size - limit), os.SEEK_SET)
        return os.read(f, limit).decode(errors="replace")
    finally:
        os.close(f)


def deps_key(pkg: Path) -> str:
    """`tools/verity/lean/lean_audit.py`'s: what a package's fetched dependencies are a function of, its manifest's fetched
    entries and its toolchain (a test holds the two equal)."""
    doc = json.loads((pkg / "lake-manifest.json").read_text())
    fetched = [d for d in doc.get("packages", []) if d.get("type") != "path"]
    canon = json.dumps({"packagesDir": doc.get("packagesDir"), "packages": fetched}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode() + b"\0" + (pkg / "lean-toolchain").read_bytes()).hexdigest()


def readable_copy(src: Path, dst: Path) -> None:
    """``src`` copied to ``dst`` as this user, every file readable by all (`a+rX`: the build's user reads it; Mathlib's
    cache writes its builds 0600) and none setuid, setgid or sticky."""
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, symlinks=True)
    for d, _, files in os.walk(tmp):
        for p in [d, *(os.path.join(d, f) for f in files)]:
            st = os.lstat(p)
            if not stat.S_ISLNK(st.st_mode):
                add = 0o555 if stat.S_ISDIR(st.st_mode) or st.st_mode & 0o111 else 0o444
                os.chmod(p, (stat.S_IMODE(st.st_mode) & 0o777) | add)
    os.replace(tmp, dst)


def prepare(n: Node, sha: str, pkgs: list[str], log) -> dict[str, Path]:
    """Each package's dependencies, fetched as research in PREP at ``sha`` and copied to DEPS/<deps_key>: {package: copy}.
    Copies of keys this run doesn't use are deleted."""
    if not (n.prep / ".git").exists():
        # PREP may already hold dependencies infra unpacked, and git won't clone into a non-empty directory.
        tmp = n.prep.with_name(n.prep.name + ".clone")
        shutil.rmtree(tmp, ignore_errors=True)
        if _run([*n.research, "git", "clone", "-q", "--shared", "--no-checkout", n.bare, tmp], log) != 0 or \
                _run([*n.research, "mkdir", "-p", n.prep], log) != 0 or \
                _run([*n.research, "mv", tmp / ".git", n.prep / ".git"], log) != 0:
            return {}
        _run([*n.research, "rmdir", tmp], log)
    if _run([*n.research, "git", "-C", n.prep, "-c", "advice.detachedHead=false", "checkout", "-q", "-f", "--detach", sha],
            log) != 0:
        return {}
    owned_dirs(n.trusted, n.deps / "x")
    deps = {}
    for pkg in pkgs:
        src = n.prep / pkg / ".lake" / "packages"
        if _run([*n.research, "bash", "tools/verity/lean/setup.sh", pkg], log, cwd=n.prep) != 0:
            log(f"run-daily: {pkg}: setup.sh failed in {n.prep}; it builds without fetched dependencies")
        elif src.is_dir():
            dst = n.deps / deps_key(n.prep / pkg)
            if not dst.is_dir():
                readable_copy(src, dst)
                log(f"run-daily: {pkg}: dependencies copied to {dst}")
            deps[pkg] = dst
    for old in n.deps.iterdir():
        if old not in deps.values():
            shutil.rmtree(old, ignore_errors=True)
    return deps


def daily(n: Node, log=print) -> int:
    with open(n.lock, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"run-daily: another run holds {n.lock}")
            return 1
        return _daily(n, log)


def _daily(n: Node, log) -> int:
    try:
        sha = _out([*n.research, "git", f"--git-dir={n.bare}", "rev-parse", "--verify", f"{MAIN}^{{commit}}"])
    except subprocess.CalledProcessError:
        log(f"run-daily: {n.bare} has no {MAIN}; nothing published")
        return 1
    log(f"run-daily: main is {sha}")
    if _run([*n.sandbox, "checkout", sha, "--repo", n.bare, "--main", MAIN, "--tree", n.tree], log) != 0:
        log("run-daily: vy-lean-sandbox checkout failed; nothing published")
        return 1
    pkgs = CACHE.packages(n.tree)
    if not pkgs:
        log(f"run-daily: {n.tree} has no Lean package; nothing published")
        return 1
    deps = prepare(n, sha, pkgs, log)
    try:
        plat = CACHE.parse_platform(_out([*n.research, "lean", "--version"], cwd=n.prep / pkgs[0]))
    except (OSError, subprocess.CalledProcessError):
        plat = None
    if not plat:
        log(f"run-daily: research's `lean --version` in {n.prep / pkgs[0]} named no platform; nothing published")
        return 1
    user = pwd.getpwnam(n.builder)
    built = []
    for pkg in pkgs:
        o = CACHE.out_dir(n.out, pkg)
        owned_dirs(n.out, o)
        shutil.rmtree(o, ignore_errors=True)
        o.mkdir(mode=0o755)
        os.chown(o, user.pw_uid, user.pw_gid)
        rc = _run([*n.sandbox, "run", "--tree", n.tree, *(["--deps", f"{pkg}={deps[pkg]}"] if pkg in deps else []),
                   "--rw", o, "-E", "LAKE_ARTIFACT_CACHE=true", "--max-sec", MAX_SEC, "--slot-wait", SLOT_WAIT_S,
                   "--by", "lean-cache-daily", "--", n.python, HERE / "cache.py", "build", n.tree / pkg, o], log)
        if rc in STOP:
            log(f"run-daily: the sandbox exited {rc} building {pkg}; nothing published")
            return rc
        if rc == 0:
            built.append(pkg)
        else:
            log(f"run-daily: {pkg} didn't build (exit {rc}); it isn't published. The end of {o / CACHE.BUILD_LOG}:\n"
                + log_tail(o, CACHE.BUILD_LOG))
    if _run([*n.sandbox, "clean", "--tree", n.tree, "--sha", sha], log) != 0:
        log("run-daily: the tree isn't clean after the builds; nothing published")
        return 1
    if not built:
        log("run-daily: nothing built; nothing published")
        return 1
    pub = _run([sys.executable, HERE / "cache.py", "publish", "--tree", n.tree, "--rev", sha, "--out", n.out,
                "--main", "refs/remotes/origin/main", "--platform", plat, "--key-file", n.key_file, *built], log)
    pruned = _run([sys.executable, HERE / "cache.py", "prune", "--key-file", n.key_file], log)
    log(f"run-daily: {sha[:12]}: {len(built)} of {len(pkgs)} packages built; publish exited {pub}, prune {pruned}")
    return 0 if pub == 0 and pruned == 0 and len(built) == len(pkgs) else 1


if __name__ == "__main__":
    sys.exit(daily(Node(), lambda m: print(m, flush=True)))
