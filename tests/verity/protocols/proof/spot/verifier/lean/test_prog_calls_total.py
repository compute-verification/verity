"""`Prog.calls` is total: on `AB16` with its root's node 0 calling a Definition C does not have (999), which no decoded C
has, it counts node 0 as a Call, as it did, and evaluates without a panic, so its reader refuses at the callee."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import verity_difftest as D

PACKAGE = Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proof" / "spot" / "verifier" / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PROBE = """import Spot.Admitted
open Spot.OneStageZk
def main : IO Unit := do
  match Flock.Program.decode AB16.toUTF8 with
  | .error e => IO.println e
  | .ok P =>
    let P := { P with defs := P.defs.modify P.root fun d =>
      { d with body := { d.body with nodes := d.body.nodes.modify 0 fun x => { x with fn := 999 } } } }
    IO.println (toString P.calls)
"""


@pytest.mark.slow
def test_a_callee_past_the_definitions_is_a_call_and_no_panic(tmp_path):
    D.build(PACKAGE, "Spot.Admitted")
    (tmp_path / "Probe.lean").write_text(PROBE)
    # an index past an array's end prints `Error: index out of bounds`, with no `PANIC`: the switch makes it an exit
    p = subprocess.run(["lake", "env", "lean", "--run", str(tmp_path / "Probe.lean")], cwd=PACKAGE,
                       capture_output=True, text=True, env={**os.environ, "LEAN_ABORT_ON_PANIC": "1"})
    assert not any(t in p.stderr + p.stdout for t in ("PANIC", "index out of bounds")), p.stderr[-2000:]
    assert (p.returncode, p.stdout.strip()) == (0, "#[0, 2, 3]"), (p.stdout, p.stderr[-2000:])
