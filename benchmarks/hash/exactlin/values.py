"""exact-fused's served values: `tests/engine/int8_exact_gpu.py` (Llama served with TargetProfile.int8_exact; the tapped
linears' y checked bit for bit against steps.py's CPU oracle, int8_exact on the CPU, a GPU rerun and every row alone) with
`engine.int8_fused`'s `row_quant` and `product`, the two ops' CUDA bodies, counting their calls.  The counts go to
OUT/exactlin_calls.json, so a run where the two Triton kernels never served shows it.  Arguments are int8_exact_gpu's,
plus `--scratch` (ignored), as served.sh passes it; run from adapters/vllm.  `--at` and `--by` are the pin entry's
resolved_at and resolved_by.

    SERVED_SCRIPT=benchmarks/hash/exactlin/values.py SERVED_KGATE=1 bash benchmarks/hash/exactlin/served.sh \
      --at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --by WHO --int8-exact linears
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from verity_vllm.engine import int8_fused as X

if TYPE_CHECKING:
    from collections.abc import Callable

CALLS = {"row_quant": 0, "product": 0}


def counted(name: str, f: Callable[..., object]) -> Callable[..., object]:
    def body(*args: object) -> object:
        CALLS[name] += 1
        return f(*args)
    return body


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch")
    a, rest = ap.parse_known_args()
    # engine.int8_ops' CUDA bodies look these two up on the module at each call
    X.row_quant, X.product = counted("row_quant", X.row_quant), counted("product", X.product)
    from tests.engine import int8_exact_gpu
    rc = int8_exact_gpu.main(["--out", a.out, *rest])
    (Path(a.out) / "exactlin_calls.json").write_text(json.dumps(CALLS) + "\n")
    print(f"exactlin calls {json.dumps(CALLS)}", flush=True)
    return rc if all(CALLS.values()) else 5


if __name__ == "__main__":
    sys.exit(main())
