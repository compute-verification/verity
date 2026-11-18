from __future__ import annotations

import sys
from pathlib import Path

# The Flock suite's helpers (`synthetic_sets`, `test_circuit_types`, ...) live there once; pyproject.toml declares them.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests" / "verity" / "protocols" / "proofs" / "flock" / "prover"))
