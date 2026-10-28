"""`Flock.Test.RecCap`'s `#guard`, built after landing rather than by the Lean check: `RecVStar.statements` at the deployed
inner shape (m 25, k_log 22, claims 8) builds V*'s whole list in the interpreter and ends it with the later levels' caps, then
level 0's, about 20 min (1,097-1,246 s on every check node, 26 Oct). It catches what `Flock.Test.RecCapSmall`, one of Flock's
audit roots, can't: `statements` itself, whose parts and the verifier's groups RecCapSmall doesn't build (over 3 min even at
m 22), holding RecCap's refusals and the checks `statements` runs to the list it assembles at m 22. Flock's
`lean-audit.json` lists the module under `exempt`."""
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
