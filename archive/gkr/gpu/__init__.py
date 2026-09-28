"""A-GKR's prover on the GPU: torch (+Triton) over BabyBear^6.  Run from ``archive/gkr`` as ``python -m gpu.run``.

``kernels`` is put on ``sys.path`` so the Merkle module imports as ``hash_gpu``.
"""
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[3] / "verity" / "ml" / "kernels"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
