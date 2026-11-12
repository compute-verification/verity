"""`Flock.Test.RecCap`'s `#guard`s, built by neither the Lean check nor any check but the nightly cold one, which runs this test
by name with VERITY_FLOCK_SLOW=1 (check.py's `OPT_IN`): they evaluate whole V* lists in the interpreter, too slowly to build
inside every check.
They are what refuses a V* list whose RecCap statements are missing, misplaced or at another node count, and a table or unit
name `RecCoins.checkCap` / `RecCoins.check` must refuse, at the deployed shapes; no cheaper test covers those shapes
(`test_lean_rec_coins` checks the rest of `checkCap`'s refusals). Flock's `lean-audit.json` lists the module under `exempt`."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from verity_config.settings import of

LEAN = Path(__file__).resolve().parents[2] / "verifier"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.mark.skipif(not of(__file__).VERITY_FLOCK_SLOW, reason="about 20 minutes of `lake build` on a node: VERITY_FLOCK_SLOW=1")
def test_rec_caps_guards_hold():
    r = subprocess.run(["bash", "-c", "mkdir -p .lake && exec flock .lake/verity-build.lock lake build Flock.Test.RecCap"],
                       cwd=LEAN, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
