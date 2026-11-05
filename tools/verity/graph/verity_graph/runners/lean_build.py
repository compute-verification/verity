"""lean-build: `lake build` of the Lean package the task names, as check's lean-build step runs it (`suites.LAKE_BUILD`), then
each executable it built copied to its output, at its path in the tree.  The toolchain is set up first, apart (elan, and the
toolchain the package's `lean-toolchain` pins): a failure there, or a build a signal killed, is the machine's, an error; a
build that fails is the commit's."""
from __future__ import annotations

import shutil
from pathlib import Path

from .. import ROOT
from . import call

#: `tools/verity/check/suites.py`'s ELAN and LAKE_BUILD, word for word (`tests/test_build.py`): lake through elan, installed
#: when missing, and one build of a tree's package at a time
ELAN = ('command -v lake >/dev/null || command -v "$HOME/.elan/bin/lake" >/dev/null || { curl -sSf '
        'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null; }; '
        'export PATH="$HOME/.elan/bin:$PATH"; ')
LAKE_BUILD = ELAN + "mkdir -p .lake && exec flock .lake/verity-build.lock lake build"


def run(task: dict, work: Path) -> dict:
    pkg = ROOT / task["id"].partition(":")[2]
    if rc := call(["bash", "-c", ELAN + "lake --version"], work, pkg):
        return {"status": "error", "detail": {"error": f"the Lean toolchain didn't set up (exit {rc})"}}
    rc = call(["bash", "-c", LAKE_BUILD], work, pkg)
    if rc < 0 or rc > 128:
        return {"status": "error", "detail": {"error": f"lake build was killed (exit {rc})"}}
    if rc:
        return {"status": "fail", "detail": {"exit": rc}}
    for o in task["outputs"]:
        if (ROOT / o["path"]).is_file():
            (work / o["path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / o["path"], work / o["path"])
    return {"status": "pass"}
