"""flock-circuit-build: `verity/protocols/flock/prover/check_build.sh` as check's flock-circuit-build step runs it: on the Flock checkout the
script pins and keeps (`$FLOCK_CHECKOUT`), with this tree's patches and live crate, `cargo check` of flock-circuit in each CPU
feature set a pod builds.  It fails when its log shows the tree at fault (a patch that doesn't apply, a crate that doesn't
compile); any other failing exit is the machine's (rustup, the clone, a crate's download, a kill), an error."""
from __future__ import annotations

from pathlib import Path

from .. import ROOT
from ..graph import LOG
from . import call

SCRIPT = "verity/protocols/flock/prover/check_build.sh"
#: what the log shows when the tree is at fault: `git apply` refusing a patch, cargo a crate (ci.toml's `halts_on`)
FAILS = ("patch does not apply", "patch failed", "error: could not compile")


def run(task: dict, work: Path) -> dict:
    rc = call(["bash", str(ROOT / SCRIPT)], work, ROOT)
    if rc == 0:
        return {"status": "pass"}
    log = (work / LOG).read_text(errors="replace")
    if 0 < rc <= 128 and (why := next((s for s in FAILS if s in log), None)):
        return {"status": "fail", "detail": {"exit": rc, "why": why}}
    return {"status": "error", "detail": {"error": f"{SCRIPT} exited {rc} with no failure of the tree's in its log"}}
