"""`verity bench exfiltration [ARGS]`: GLM-5.3's exfiltration capacity in GB over a monitoring period at 10% of total
compute, the compute audit's part beside the network attester's, every input labelled with its mark
(the README beside this file). Its code is `verity_bench.exfiltration`.

    verity bench exfiltration                           the figure, the inputs, each part, and the two side by side a period
    verity bench exfiltration --json                    the same as one document
    verity bench exfiltration --from DIR                the same from one run's record directory, with no store
    verity bench exfiltration --from benchmarks/exfiltration/made_up    a made-up record, printed as made up
    verity bench exfiltration --help                    every parameter and its default

Under a research run, it also writes `METRICS` into the run's result (`verity_bench.exfiltration.record`); with
`--from`, `report` does (`kit.report`).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

from verity_bench import kit
from verity_bench.exfiltration import main, measurements

HEADLINE = "benchmarks/exfiltration-total-gb"
COMPUTE_AUDIT = "benchmarks/exfiltration-compute-audit-gb"
NETWORK = "benchmarks/exfiltration-network-attester-gb"
#: the metrics `benchmarks/exfiltration/metrics.toml` holds this benchmark to, by id: each a figure of the document
#: at the share (`verity_bench.exfiltration.FIGURES`), recorded over each of its monitoring periods, the world's year
#: the one each title names
METRICS = {HEADLINE: "headline", COMPUTE_AUDIT: "compute_audit", NETWORK: "network_attester"}


def _of(doc: Mapping[str, Any], metric: str, figure: str) -> list[dict]:
    """The figure's measurements (`measurements`) under the metric's id, every one modeled, as `kit.report` takes them:
    their mark is its `mark`."""
    return [{k: v for k, v in m.items() if k != "mark"} for m in measurements(doc, {metric: figure})]


def measured(doc: Mapping[str, Any]) -> list[dict]:
    """The three figures' measurements, each over every monitoring period, under each metric's id."""
    return [*({**m, "name": HEADLINE, "value": m["value"]} for m in _of(doc, HEADLINE, "headline")),
            *({**m, "name": COMPUTE_AUDIT, "value": m["value"]} for m in _of(doc, COMPUTE_AUDIT, "compute_audit")),
            *({**m, "name": NETWORK, "value": m["value"]} for m in _of(doc, NETWORK, "network_attester"))]


def report(doc: Mapping[str, Any], text: str, *, made_up: bool, withdrawn: Sequence[str]) -> int:
    """`--from`'s last step: the print, the document as one line and the measurements, through `kit.report`. The
    planner's rows record no coins (`ran with: not stated`). A made-up record writes no measurement, under a research
    run or not: a metric's points are runs' figures, and its document says it is made up."""
    # the figure is modeled, every input marked: no Lean verifier passes on the planner's rows, so none refuses a number
    return kit.report("exfiltration", text, doc, () if made_up else measured(doc), mark="modeled", ran_with=None,
                      accepted=None, made_up=made_up, withdrawn=withdrawn)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], metrics=METRICS, report=report))
