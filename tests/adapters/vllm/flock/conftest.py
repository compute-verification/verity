from __future__ import annotations

import sys
from pathlib import Path

# The Flock suite's helpers (`synthetic_sets`, `test_circuit_types`, ...) live there once; pyproject.toml declares them.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "proof" / "flock" / "prover"))
sys.path.append(str(Path(__file__).resolve().parents[4] / "tests" / "verity" / "circuits"))

import stand_in  # noqa: E402,F401  (the test primitives' Boolean bodies, registered as it loads)
