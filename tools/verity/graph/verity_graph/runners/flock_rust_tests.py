"""flock-rust-tests: `verity/protocols/flock/prover/check_build.sh test` as check's flock-rust-tests step runs it, with cargo's build jobs and
the test threads at the task's cores: on the Flock checkout the script pins and keeps (`$FLOCK_CHECKOUT`), with this tree's
patches and live crate, the Rust tests in release.  It fails when its log shows the tree at fault: a patch that doesn't apply or a
crate that doesn't compile (`flock_circuit_build.FAILS`), or a test that failed (libtest's summary); any other failing exit is the
machine's (rustup, the clone, a crate's download, a test binary a signal killed, which cargo reports as a failed test too), an
error."""
from __future__ import annotations

from pathlib import Path

from .. import ROOT
from ..graph import LOG
from . import call
from .flock_circuit_build import FAILS as BUILD_FAILS
from .flock_circuit_build import SCRIPT

#: libtest's summary of a test binary in which a test failed: cargo's own `error: test failed` follows a binary a signal killed
FAILS = (*BUILD_FAILS, "test result: FAILED")


def run(task: dict, work: Path) -> dict:
    n = str(max(1, int(task["resources"]["cores"])))
    rc = call(["env", f"CARGO_BUILD_JOBS={n}", f"RUST_TEST_THREADS={n}", "bash", str(ROOT / SCRIPT), "test"], work, ROOT)
    if rc == 0:
        return {"status": "pass"}
    log = (work / LOG).read_text(errors="replace")
    if 0 < rc <= 128 and (why := next((s for s in FAILS if s in log), None)):
        return {"status": "fail", "detail": {"exit": rc, "why": why}}
    return {"status": "error", "detail": {"error": f"{SCRIPT} test exited {rc} with no failure of the tree's in its log"}}
