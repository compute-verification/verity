"""`verity-bench served-table` with the numbers `metrics.toml` holds it to (the harness's README, "Served table"):

    uv run python benchmarks/harness/served_overhead.py [--json] [--local-only]

The table of every protocol's served overhead from the store's runs, as `served-table` prints it. Under a research run it
also writes `METRICS` into the run's result (`served_render.record`): each the named row's arm over stock FP8 as served,
in time per output token, as a percent over stock.
"""

from __future__ import annotations

import sys

from verity_bench.cli import main

#: the metrics `benchmarks/harness/metrics.toml` holds this benchmark to, by id: each the served table's row at
#: ``SPEC/POINT``
METRICS = {"benchmarks/served-overhead-commit": "llama31-8b-fp8-1gpu-commit-served-width-h200/b32-p512-mixed",
           "benchmarks/served-overhead-pous-bandwidth-audit": "llama31-8b-fp8-1gpu-bandwidth-served-r6/b32-p512-mixed"}

if __name__ == "__main__":
    sys.exit(main(["served-table", *sys.argv[1:]], metrics=METRICS))
