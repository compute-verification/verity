"""exact-fused's served values: `tests/engine/int8_exact_gpu.py` (Llama served with TargetProfile.int8_exact; the tapped
linears' y checked bit for bit against steps.py's CPU oracle, int8_exact on the CPU, a GPU rerun and every row alone) with
`exactlin.install()` first, so every linear serves from the two Triton kernels.  The bodies count their calls, written to
OUT/exactlin_calls.json, so a run where they never served shows it.  Arguments are int8_exact_gpu's, plus `--scratch`
(ignored), as served.sh passes it; run from integrations/vllm.  `--at` and `--by` are the pin entry's resolved_at and
resolved_by.

    SERVED_SCRIPT=benchmarks/hash/exactlin/values.py SERVED_KGATE=1 bash benchmarks/hash/exactlin/served.sh \
      --at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --by WHO --int8-exact linears
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exactlin as X

CALLS = {"row_quant": 0, "product": 0}


def counted(name: str, f):
    def body(*args):
        CALLS[name] += 1
        return f(*args)
    return body


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch")
    a, rest = ap.parse_known_args()
    X.row_quant, X.product = counted("row_quant", X.row_quant), counted("product", X.product)
    X.install()
    from tests.engine import int8_exact_gpu
    rc = int8_exact_gpu.main(["--out", a.out, *rest])
    (Path(a.out) / "exactlin_calls.json").write_text(json.dumps(CALLS) + "\n")
    print(f"exactlin calls {json.dumps(CALLS)}", flush=True)
    return rc if all(CALLS.values()) else 5


if __name__ == "__main__":
    sys.exit(main())
