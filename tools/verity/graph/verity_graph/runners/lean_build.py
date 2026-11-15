"""lean-build: `lake build` of the Lean package the task names, as check's lean-build step runs it (`suites.LAKE_BUILD`), then
each executable it built copied to its output, at its path in the tree, and the rest of the build (`TREE`: each `.lake/build`
of the packages its var `builds` names, the package's and those of the tree it requires by path, but the executables) packed
into one archive, which its readers unpack into their trees (`lean_suites.place`), so that a test's `lake build` there finds
every module built, as it does in check after its lean-build step, rather than building them all, at once with every other
test.  The archive's digest is written to `STAMP` in the package's `.lake`, so a reader in this tree, which holds the build
already, doesn't unpack it again.  The toolchain is set up first, apart (elan, and the toolchain the package's
`lean-toolchain` pins): a failure there, or a build a signal killed, is the machine's, an error; a build that fails is the
commit's."""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from .. import ROOT, contract
from . import call

#: `tools/verity/check/suites.py`'s ELAN and LAKE_BUILD, word for word (`tests/test_build.py`): lake through elan, installed
#: when missing, and one build of a tree's package at a time
ELAN = ('command -v lake >/dev/null || command -v "$HOME/.elan/bin/lake" >/dev/null || { curl -sSf '
        'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null; }; '
        'export PATH="$HOME/.elan/bin:$PATH"; ')
LAKE_BUILD = ELAN + "mkdir -p .lake && exec flock .lake/verity-build.lock lake build"
#: the output holding the build but its executables, and where in a package's `.lake` the digest of the one it holds is
TREE, STAMP = "lake-build", "graph-build.sha512"


def run(task: dict, work: Path) -> dict:
    pkg = ROOT / task["id"].partition(":")[2]
    if rc := call(["bash", "-c", ELAN + "lake --version"], work, pkg):
        return {"status": "error", "detail": {"error": f"the Lean toolchain didn't set up (exit {rc})"}}
    rc = call(["bash", "-c", LAKE_BUILD], work, pkg)
    if rc < 0 or rc > 128:
        return {"status": "error", "detail": {"error": f"lake build was killed (exit {rc})"}}
    if rc:
        return {"status": "fail", "detail": {"exit": rc}}
    exes = {o["path"] for o in task["outputs"] if o["name"] != TREE}
    for o in task["outputs"]:
        if o["name"] == TREE:
            pack(task["vars"]["builds"].split(), exes, work / o["path"])
            (pkg / ".lake" / STAMP).write_text(contract.file_digest(work / o["path"]))
        elif (ROOT / o["path"]).is_file():
            (work / o["path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / o["path"], work / o["path"])
    return {"status": "pass"}


def pack(packages: list[str], leave: set[str], out: Path) -> None:
    """Each of ``packages``' `.lake/build` into the archive ``out``, by path from the root and in path order, but the files at
    ``leave``, with no owner."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz", compresslevel=1) as tf:
        for p in packages:
            top = ROOT / p / ".lake" / "build"
            for f in sorted(top.rglob("*")) if top.is_dir() else ():
                rel = f.relative_to(ROOT).as_posix()
                if f.is_file() and not f.is_symlink() and rel not in leave:
                    info = tf.gettarinfo(f, rel)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with open(f, "rb") as fh:
                        tf.addfile(info, fh)
