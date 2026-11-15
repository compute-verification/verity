import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parents[2] / "benchmarks" / "serve_tax") not in sys.path:
    sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "serve_tax"))
if str(HERE.parents[2] / "adapters/vllm") not in sys.path:
    sys.path.insert(0, str(HERE.parents[2] / "adapters/vllm"))
