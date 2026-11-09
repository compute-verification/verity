"""lean-build: `lake build` of the Lean package the task names, as check's lean-build step runs it (`suites.LAKE_BUILD`), then
each executable it built copied to its output, at its path in the tree, and the rest of the package's `.lake/build` packed
into its output `build` (`pack`).  The toolchain is set up first, apart (elan, and the toolchain the package's
`lean-toolchain` pins): a failure there, or a build a signal killed, is the machine's, an error; a build that fails is the
commit's.

A task that reads those outputs puts them in its tree before it runs (`place`): the executables at their paths and `build`
unpacked over the package's `.lake/build`, so that `lake build` there finds every target built and builds none, in a tree at
any path: Lake holds a target to the hashes its trace records, not to times or paths."""
from __future__ import annotations

import contextlib
import fcntl
import gzip
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from .. import ROOT, contract
from . import call

#: `tools/verity/check/suites.py`'s ELAN and LAKE_BUILD, word for word (`tests/test_build.py`): lake through elan, installed
#: when missing, and one build of a tree's package at a time
ELAN = ('command -v lake >/dev/null || command -v "$HOME/.elan/bin/lake" >/dev/null || { curl -sSf '
        'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null; }; '
        'export PATH="$HOME/.elan/bin:$PATH"; ')
LAKE_BUILD = ELAN + "mkdir -p .lake && exec flock .lake/verity-build.lock lake build"
#: the output that holds the package's `.lake/build` but the executables the task outputs, packed (`pack`)
BUILD = "build"
#: under a package: its build directory, where `build` sits in a work directory, the lock its builds take (`LAKE_BUILD`'s),
#: and the digests of the outputs its tree holds (`place`)
TREE, PACKED, LOCK, PLACED = ".lake/build", ".lake/build.tar.gz", ".lake/verity-build.lock", ".lake/verity-placed.json"


def run(task: dict, work: Path) -> dict:
    pkg = ROOT / task["id"].partition(":")[2]
    if rc := call(["bash", "-c", ELAN + "lake --version"], work, pkg):
        return {"status": "error", "detail": {"error": f"the Lean toolchain didn't set up (exit {rc})"}}
    rc = call(["bash", "-c", LAKE_BUILD], work, pkg)
    if rc < 0 or rc > 128:
        return {"status": "error", "detail": {"error": f"lake build was killed (exit {rc})"}}
    if rc:
        return {"status": "fail", "detail": {"exit": rc}}
    with locked(pkg):
        for o in task["outputs"]:
            (work / o["path"]).parent.mkdir(parents=True, exist_ok=True)
            if o["name"] == BUILD:
                pack(pkg, work / o["path"], {ROOT / x["path"] for x in task["outputs"]})
            elif (ROOT / o["path"]).is_file():
                shutil.copy2(ROOT / o["path"], work / o["path"])
        stamp(pkg, {o["name"]: contract.file_digest(work / o["path"]) for o in task["outputs"] if (work / o["path"]).is_file()})
    return {"status": "pass"}


@contextlib.contextmanager
def locked(pkg: Path):
    """The package's build lock, held: one build, or one placing, in a tree's package at a time."""
    (pkg / LOCK).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(pkg / LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def pack(pkg: Path, archive: Path, skip: set[Path]) -> None:
    """``pkg``'s `.lake/build` as a gzipped tar at ``archive``: its files but ``skip`` in path order, each with its bytes and
    whether it is executable and nothing of the machine's (owner, times, the archive's name), so one build packs to one
    archive's bytes."""
    tree = pkg / TREE
    files = sorted((p for p in tree.rglob("*") if p.is_file() and not p.is_symlink() and p not in skip),
                   key=lambda p: p.relative_to(tree).as_posix())
    with open(archive, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in files:
            info = tarfile.TarInfo(p.relative_to(tree).as_posix())
            info.size, info.mode = p.stat().st_size, 0o755 if os.access(p, os.X_OK) else 0o644
            with open(p, "rb") as f:
                tar.addfile(info, f)


def unpack(archive: Path, pkg: Path) -> None:
    """The archive `pack` wrote, over ``pkg``'s `.lake/build`: each file written beside its place and moved there whole, so a
    `lake` reading the tree meanwhile never sees one half written.  A member that is no file under the build is refused."""
    tree = pkg / TREE
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            name = PurePosixPath(m.name)
            if not m.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError(f"{archive.name}: {m.name!r} is no file of a build directory")
            dst = tree / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.graph-{os.getpid()}")
            with tar.extractfile(m) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out)
            tmp.chmod(m.mode & 0o777)
            os.replace(tmp, dst)


def stamp(pkg: Path, digests: dict[str, str]) -> None:
    """Record that ``pkg``'s tree holds the outputs ``digests`` names, {output: digest}."""
    tmp = pkg / f"{PLACED}.graph-{os.getpid()}"
    tmp.write_text(json.dumps(digests, sort_keys=True))
    os.replace(tmp, pkg / PLACED)


def place(task: dict, work: Path) -> None:
    """The lean-build outputs ``task`` reads, from its work directory into its tree, under the package's lock, unless the tree
    holds them already (the ones last built or placed there, `PLACED`): each executable at its path, executable, and `build`
    unpacked (`unpack`).  The package is the task's var `lean`, and each output sits where lean-build's lister puts it
    (`families/build.py`): an executable at `.lake/build/bin/<exe>`, `build` at `PACKED`."""
    lean = task["vars"]["lean"]
    paths = {r["output"]: f"{lean}/{PACKED}" if r["output"] == BUILD else f"{lean}/{TREE}/bin/{r['output']}" for r in task["reads"]}
    digests = {o: contract.file_digest(work / p) for o, p in paths.items()}
    pkg = ROOT / lean
    with locked(pkg):
        held = json.loads((pkg / PLACED).read_text()) if (pkg / PLACED).is_file() else {}
        if all(held.get(o) == d for o, d in digests.items()) and all((ROOT / p).is_file() for o, p in paths.items() if o != BUILD):
            return
        for o, p in paths.items():
            if o == BUILD:
                unpack(work / p, pkg)
                continue
            dst = ROOT / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.graph-{os.getpid()}")
            shutil.copyfile(work / p, tmp)
            tmp.chmod(0o755)
            os.replace(tmp, dst)
        stamp(pkg, {**held, **digests})
