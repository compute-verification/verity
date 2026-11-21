"""`Flock.Compose.compose`'s files, flat and templated, read back by the door's reader and agreeing with their blocks
(`Flock.Test.Compose.roundTrips`), compiled: each file's reader builds the `sha512x3` and `hm96` nets, minutes in the
interpreter the Lean check elaborates `Flock.Test.Compose` with, so its guards there leave the whole read to this test.
Only a whole read of a whole file catches a writer whose sections, ranges or wires the reader refuses."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from test_lean_verifier import LEAN, lake_build

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.mark.slow
def test_compose_round_trips():
    lake_build(LEAN, "flock-compose-check")
    p = subprocess.run([str(LEAN / ".lake" / "build" / "bin" / "flock-compose-check")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
