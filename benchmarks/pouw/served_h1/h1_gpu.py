"""`pouw_h1_gpu.py` (the vLLM adapter's served H1 run) with kernels' producer, `producer.py` beside this file, handed to its
``main``: what its ``--receipts SO`` computes each call's receipts with (`producer.Units`) and holds them to
`h1tile.reference` with (`producer.check`). Its arguments are `pouw_h1_gpu.py`'s, from ``adapters/vllm``:

    python ../../benchmarks/pouw/served_h1/h1_gpu.py --workload fill --served --receipts SO ...
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "adapters" / "vllm")]

import producer  # noqa: E402
from tests.protocols import pouw_h1_gpu  # noqa: E402

if __name__ == "__main__":
    sys.exit(pouw_h1_gpu.main(producer=producer))
