"""lean-unit-cut: `verity/protocols/flock/agreement/unit_cut_agree.py` as check's lean-unit-cut step runs it, on the flock-verify the task
reads (at its var `verify`): the Lean partition check's codes against `verity.circuits.partition`'s on the script's random
cuts, its result.json at the task's output.

A pass is every cut agreeing, at least one checked.  A fail is the tree's: a disagreement on some cuts while others agree, or the
script stopping on an exception before it compared.  Anything else is an error, no result: a run a signal killed, an exception of
the machine's (`MACHINE`), a run that checked no cut, and a verifier that agreed on none, which didn't run here (built for another
CPU or C library) as often as it was wrong."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import ROOT
from ..graph import LOG
from . import call

SCRIPT = "verity/protocols/flock/agreement/unit_cut_agree.py"
OUT = "lean-unit-cut"
RESULT = f"{OUT}/result.json"
TRACEBACK = "Traceback (most recent call last):"
#: the exceptions a Python raises for its machine rather than for the code it runs
MACHINE = ("OSError", "PermissionError", "FileNotFoundError", "MemoryError")


def run(task: dict, work: Path) -> dict:
    verify = work / task["vars"]["verify"]
    verify.chmod(0o755)
    rc = call([sys.executable, str(ROOT / SCRIPT), "--lean", str(verify), "--out", str(work / OUT)], work, ROOT)
    if rc < 0 or rc > 128:
        return {"status": "error", "detail": {"error": f"{SCRIPT} was killed (exit {rc})"}}
    if (work / RESULT).is_file():
        res = json.loads((work / RESULT).read_text())
        cuts, agree = int(res.get("cuts") or 0), int(res.get("agree") or 0)
        detail = {"exit": rc, "cuts": cuts, "agree": agree}
        if rc == 0 and cuts and agree == cuts:
            return {"status": "pass", "detail": detail}
        if not cuts:
            return {"status": "error", "detail": {**detail, "error": f"{SCRIPT} checked no cut"}}
        if not agree:
            return {"status": "error", "detail": {**detail, "error": f"the Lean verifier agreed on none of {cuts} cuts: it didn't "
                                                                     "run here, or every cut found it wrong"}}
        if rc == 1:
            return {"status": "fail", "detail": {**detail, "disagree": [Path(f).name for f in res.get("disagree") or ()]}}
        return {"status": "error", "detail": {**detail, "error": f"{SCRIPT} exited {rc}"}}
    lines = [ln for ln in (work / LOG).read_text(errors="replace").splitlines() if ln.strip()]
    if rc and TRACEBACK in lines and not lines[-1].split(":")[0].rsplit(".", 1)[-1].startswith(MACHINE):
        return {"status": "fail", "detail": {"exit": rc, "why": lines[-1][:240]}}
    return {"status": "error", "detail": {"error": f"{SCRIPT} exited {rc} and wrote no {RESULT}"
                                                   + (f": {lines[-1][:240]}" if lines else "")}}
