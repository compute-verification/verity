"""`verity bench overhead points|run|table`: the one front door for overhead measurements, `verity-bench overhead` (its
README beside this file).

    verity bench overhead points [--card h200] [--model M] [--on NODE] [--gpu I] [--cores CPUS]
                                                                     the sweep for C and L, a research run on one GPU
    verity bench overhead points --store RUN                         the finished sweep's points, stored and of record
    verity bench overhead run --scheme S [--point C|L|all] [--card h200] [--model M] [--on NODE] [--gpu I] [--cores CPUS]
                                                                     S against stock at the points, a research run
    verity bench overhead run --record RUN                           label RUN a run of record
    verity bench overhead table [--json]                             γ per scheme, point and card from runs of record

Under a research run, `table` also writes `METRICS` into the run's result (`overhead_records.record`).
"""

from __future__ import annotations

import sys

from verity_bench.cli import main

#: the metrics `benchmarks/overhead/metrics.toml` holds this benchmark to, by id: each the γ of record of the table's row
#: `scheme/card/point` and what the row says of its readings (`overhead_records.metric_row`): ``own``, the kernel model's on
#: the row's card its own reading and every other model's a member, or ``members``, every reading a member
METRICS = {"benchmarks/pearl-plus-gamma-c-h200": "pearl-plus/h200/C own",
           "benchmarks/pearl-plus-gamma-l-h200": "pearl-plus/h200/L own",
           "benchmarks/pearl-plus-gamma-c": "pearl-plus/h100/C members",
           "benchmarks/pearl-plus-gamma-l": "pearl-plus/h100/L members"}

if __name__ == "__main__":
    sys.exit(main(["overhead", *sys.argv[1:]], metrics=METRICS))
