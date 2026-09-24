"""red-team-leaf-3: G1 re-test against lane/ajtai-leaf-3 with the PYTHON key check disabled, so the forged fixture reaches
the Rust verifier.  Runs red-team-leaf-2's ``leaf2_ajtai_decoy.py`` (B = 0 chain + decoy rows carrying the derived key)
unchanged after replacing ``AjtaiLeaf.check_system`` with a no-op in this process only.

    PYTHONPATH=<tree>:<tree>/packages/verity/src:<tree>/backends/numerical/python \\
    python leaf3_g1_rust_only.py --out /tmp/rt3/out/g1-decoy-rust --bin <ligero-verify of the same tree> [--no-decoys]

Exit 0 iff the Rust verifier ACCEPTED the forgery (G1 still open); 1 iff it refused (fixed)."""
from __future__ import annotations

import pathlib
import runpy
import sys

from backends.direct.ligero.leaf import ajtai as A

A.AjtaiLeaf.check_system = lambda self, sys_: None
sys.argv[0] = str(pathlib.Path(__file__).with_name("leaf2_ajtai_decoy.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
