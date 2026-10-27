"""`Flock.Test.RecCap`'s `#guard`s, built after landing rather than by the Lean check: they evaluate whole V* lists in the
interpreter, 1,097-1,246 s to build on every check node (26 Oct), which gated every Flock-touching Lean check by about 18 min.
They are what refuses a V* list whose RecCap statements are missing, misplaced or at another node count, and a table or unit
name `RecCoins.checkCap` / `RecCoins.check` must refuse, at the deployed shapes; no cheaper test covers those shapes
(`test_lean_rec_coins` checks the rest of `checkCap`'s refusals). Flock's `lean-audit.json` lists the module under `exempt`."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


def test_rec_caps_guards_hold():
    r = subprocess.run(["bash", "-c", "mkdir -p .lake && exec flock .lake/verity-build.lock lake build Flock.Test.RecCap"],
                       cwd=LEAN, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
