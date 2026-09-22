"""Candidate A's prover on the GPU: torch (+Triton) over BabyBear^6.  Run from ``backends/gkr`` as ``python -m gpu.run``.

``backends/shared`` is put on ``sys.path`` so the Merkle module imports as ``hash_gpu``.
"""
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[2] / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
