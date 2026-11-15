import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if str(HERE.parents[2] / "adapters/vllm") not in sys.path:
    sys.path.insert(0, str(HERE.parents[2] / "adapters/vllm"))
