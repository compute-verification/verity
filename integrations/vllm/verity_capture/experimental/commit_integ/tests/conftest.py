import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
for p in (_ROOT, _ROOT / "vllm-poc", _ROOT / "src"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
