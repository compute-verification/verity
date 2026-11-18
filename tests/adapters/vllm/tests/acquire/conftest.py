import sys
from pathlib import Path

_ROOT = (Path(__file__).resolve().parents[5] / "adapters" / "vllm")
for p in (_ROOT,):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
