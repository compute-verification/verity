"""`Flock.Test.RecCap`'s `#guard`s, built after landing rather than by the Lean check: they evaluate V*'s schedules at every
fast100 shape in the interpreter, too slowly to build inside every Lean check that touches Flock.
They are what refuses a V* list whose RecCap statements are missing, at another node count, repeated or past the last level,
and a table or unit name `RecCoins.checkCap` / `RecCoins.check` must refuse, at the deployed shapes; no cheaper test covers
those shapes (`test_lean_rec_coins` checks the rest of `checkCap`'s refusals). They build no whole V* list: the deployed
list is `test_rec_vstar_checks.py::test_the_verifiers_statements_are_the_ones_v_star_stages`', on the compiled verifier,
and its equality with the stager's whole list is what holds the RecCaps' order, which `RecVStar.statements`' checks don't.
Flock's `lean-audit.json` lists the module under `exempt`."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[2] / "verifier"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


def test_rec_caps_guards_hold():
    r = subprocess.run(["bash", "-c", "mkdir -p .lake && exec flock .lake/verity-build.lock lake build Flock.Test.RecCap"],
                       cwd=LEAN, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
