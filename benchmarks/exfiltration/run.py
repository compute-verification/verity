"""`verity bench exfiltration [ARGS]`: GLM-5.3's exfiltration capacity in GB over a monitoring period at 10% of total
compute, the compute audit's part beside the network certifier's, every input labelled with its mark
(the README beside this file). Its code is `verity_bench.exfiltration`.

    verity bench exfiltration                 the figure, the inputs, each part, and the two side by side a period
    verity bench exfiltration --json          the same as one document
    verity bench exfiltration --help          every parameter and its default

Under a research run, it also writes `METRICS` into the run's result (`verity_bench.exfiltration.record`).
"""

from __future__ import annotations

import sys

from verity_bench.exfiltration import main

#: the metrics `benchmarks/exfiltration/metrics.toml` holds this benchmark to, by id: each a figure of the document
#: over the world's year at the share (`verity_bench.exfiltration.FIGURES`)
METRICS = {"benchmarks/exfiltration-gb-world-year": "headline",
           "benchmarks/exfiltration-compute-audit-gb": "compute_audit",
           "benchmarks/exfiltration-network-certifier-gb": "network_certifier"}

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], metrics=METRICS))
