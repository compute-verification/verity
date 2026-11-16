"""The network attester's part of `verity bench exfiltration` (`verity_bench.exfiltration`), and the requests both parts
count over."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity_bench.exfiltration_common import (
    MARKS,
    PERIOD_ERROR,
    RESPONSE_IDS,
    SCOPES,
    TERM_SCHEMA,
    _load,
)
from verity_bench.exfiltration_text import _n, _p


def certifier(root: Path, response_ids: int = RESPONSE_IDS, scan_delta: Fraction | None = None,
              stop: Any = None) -> dict:
    """The network attester's term at the datacenter's and the world's load (`leak_bounds.document`) for a class of
    at most ``response_ids`` ids a response, over the requests its calculator counts by default, the sampling plan's,
    with each execution phase's scan at ``scan_delta``. With None the δ is the one at which the row of the document
    that sums the most execution phases carries `PERIOD_ERROR`: that error over the row's execution phases, which the
    calculator's own document counts, no term depending on the scans' δ. The document's `scan_delta` input keeps the
    calculator's value and mark, and its source says who gave the δ and how it was chosen.

    ``stop`` is how the stop class is charged, the stop verdicts' reader's answer (`stop_charge`), which the
    calculator's rows read from verdict records are charged by; with None the calculator is handed none and charges
    them as with no record read. The document's `stop` is the calculator's account of it.

    Beside the document: the jitter bound of its term of record and the one its labelled term is at, each as the
    rows' `term` names it (`of_record`, `set_time`), and the term of record at the class the calculator assumes by
    default with its ε, at the same δ (`default_class`), which is `verity bench certifier_term`'s own figure."""

    lb = _load(root, "network_traces", "leak_bounds")
    charged = {} if stop is None else {"stop": stop}

    def read(inp: Mapping, **rows: Any) -> dict:
        doc = lb.document(inp, **rows, **charged)
        if (schema := doc["schema"]) != TERM_SCHEMA:
            raise ValueError(f"the network attester's document is `{schema}` and this command reads `{TERM_SCHEMA}`: its "
                             "term of record, its rows' terms by jitter bound, their ε and their labels may mean "
                             "something else (read the new document in benchmarks/harness/verity_bench/exfiltration.py "
                             "and name its schema there)")
        if off := sorted(name for name, k in doc["legend"].items() if k.get("mark") not in (None, *MARKS)):
            raise ValueError(f"the network attester's rows {', '.join(off)} carry a mark this command doesn't print: a "
                             f"row's mark is one of {', '.join(MARKS)}")
        return doc

    how = "given (--scan-delta)"
    if scan_delta is None:
        most = max(read(lb.inputs(response_ids=response_ids), only=SCOPES)["rows"], key=lambda r: r["phases"])
        scan_delta = PERIOD_ERROR / Fraction(most["phases"])
        how = (f"this command's, in place of the calculator's default: {_p(float(PERIOD_ERROR))}, the error a monitoring "
               f"period's figure is to carry at most, over the {_n(most['phases'])} execution phases of the "
               f"{most['scope']}'s {most['period']}, the row printed that sums the most, so that every row printed "
               "carries that error or less. The network attester's result of record takes its δ so "
               "(benchmarks/certifier_term/README.md), and no term depends on it")
    term = read(lb.inputs(response_ids=response_ids, scan_delta=scan_delta), only=SCOPES)
    default = read(lb.inputs(scan_delta=scan_delta), only=("world",), periods=("year",))
    head = next(r for r in default["rows"] if r["row"] == "scan")
    set_time, = (str(dj) for dj in lb.JITTERS if dj != lb.OF_RECORD)
    given = term["inputs"]["scan_delta"] | {"source": f"each execution phase's scan's δ, {how}"}
    return term | {"inputs": term["inputs"] | {"scan_delta": given}, "of_record": str(lb.OF_RECORD), "set_time": set_time,
                   "default_class": {"response_ids": default["inputs"]["response_ids"]["value"], "eps": head["eps"],
                                     "value": default["value"] if head["eps"] < 1 else None, "unit": default["unit"]}}
