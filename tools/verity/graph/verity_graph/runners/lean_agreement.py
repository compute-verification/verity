"""lean-agreement: `check.py --lean-agreement` as check's lean-agreement step runs it, on the flock-verify the task reads (at
its var `verify`) and the upstream build and inputs `upstream.json` pins, fetched from the evidence store as ci.py fetches what
it pins (`research data fetch`) and held to the pin's sha256.  Those fetches and ci.py's read the store's remote this
process's environment names: a `research run` job on a node with no store config of its own gets one with --store-read.  An
error when it couldn't run: a pin that didn't fetch or isn't the pinned bytes, a machine check.py says can't run it, a kill, and
a pass in which ci.py skipped every set (no upstream binary for any set's prover commit), which compared nothing; any other
failing exit is a fail."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from .. import ROOT
from ..graph import LOG
from . import call

PIN = "verity/protocols/flock/agreement/upstream.json"
CHECK = "tools/verity/check/check.py"
#: how check.py says this machine can't run the agreement (too little memory, an older glibc, a build for another CPU)
CANNOT = "check: lean-agreement: error: cannot run here:"
SETS = "flock-agreement/ci/sets.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def run(task: dict, work: Path) -> dict:
    pin = json.loads((ROOT / PIN).read_text())
    got = {}
    for k in ("upstream", "inputs"):
        art, d = pin[k]["art"], work / "pinned" / k
        if rc := call([sys.executable, "-m", "research", "data", "fetch", art, "--to", str(d)], work, ROOT):
            said = (work / LOG).read_text(errors="replace").splitlines()[-1]
            return {"status": "error", "detail": {"error": f"the pinned {k} ({art}) didn't fetch (exit {rc}: {said}); a machine without "
                                                           f"a store config of its own needs the store's remote for this fetch and "
                                                           f"ci.py's: `research run --store-read`"}}
        files = [p for p in d.rglob("*") if p.is_file()]
        if len(files) != 1 or _sha256(files[0]) != pin[k]["sha256"]:
            return {"status": "error", "detail": {"error": f"the pinned {k} ({art}) fetched as other bytes than {PIN} pins"}}
        got[k] = files[0]
    verify = work / task["vars"]["verify"]
    verify.chmod(0o755)
    rc = call([sys.executable, str(ROOT / CHECK), "--lean-agreement", str(work), str(got["upstream"]), str(got["inputs"]),
               str(verify)], work, ROOT)
    if rc == 0:
        every = json.loads((work / SETS).read_text())["sets"]
        sets = [s for s in every if "skipped" not in s]
        detail = {"sets": len(sets), "sessions": sum(s["sessions"] for s in sets), "agree": sum(s["agree"] for s in sets)}
        if not sets:
            return {"status": "error", "detail": {**detail, "error": f"ci.py skipped every one of its {len(every)} sets: "
                                                                     "nothing was compared"}}
        return {"status": "pass", "detail": detail}
    if rc < 0 or rc > 128 or CANNOT in (work / LOG).read_text(errors="replace"):
        return {"status": "error", "detail": {"error": f"check.py --lean-agreement couldn't run here (exit {rc})"}}
    return {"status": "fail", "detail": {"exit": rc}}
