"""Make the C0/C1 commitment engine (`verity_vllm`) importable for these tests without editing pyproject."""

from __future__ import annotations

import sys
from pathlib import Path

_POC = Path(__file__).resolve().parents[3] / "vllm-poc"
if _POC.is_dir() and str(_POC) not in sys.path:
    sys.path.insert(0, str(_POC))
