"""lean-suites: `tools/verity/check/suites.py` on suites check runs after its lean-build (`suites.AFTER_LEAN`; the task's var
`suites`), as check's lean-suites step runs them: the suite runner's (`suite.suites`) with `--lean-built` and lake on the PATH
through elan (`lean_build.ELAN`).  First each verifier executable the task reads goes to its path in the tree (its var `bin`,
lean-build's `.lake/build/bin`), where the suites look, unless the tree holds those bytes there already (lean-build built them
on this machine).  Its status is the suite runner's."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .. import ROOT, contract
from .lean_build import ELAN
from .suite import suites

REPORT = "suites-lean.json"


def place(task: dict, work: Path) -> None:
    """The executables ``task`` reads, from its work directory to their paths in the tree, executable."""
    for r in task["reads"]:
        src, dst = work / task["vars"]["bin"] / r["output"], ROOT / task["vars"]["bin"] / r["output"]
        if not (dst.is_file() and contract.file_digest(dst) == contract.file_digest(src)):
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.graph-{os.getpid()}")
            shutil.copyfile(src, tmp)
            tmp.chmod(0o755)
            os.replace(tmp, dst)
        dst.chmod(0o755)


def run(task: dict, work: Path) -> dict:
    place(task, work)
    return suites(task, work, REPORT, ELAN)
