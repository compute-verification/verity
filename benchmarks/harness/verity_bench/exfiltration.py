"""`verity bench exfiltration`: GLM-5.3's exfiltration capacity over a monitoring period at a share of total compute
(10% by default), in GB, as two parts side by side with every input labelled measured, modeled, simulated, assumed,
missing or run, not proved. It computes neither part itself: it composes the two calculators that do, and runs on any
machine that reaches the store, with no GPU and no node.

* **The share of total compute** is given (`--share`) and assumed. Beside it the command prints the share a served
  two-stage run measures (`served_share`, `--workload`): the audit's GPU-seconds on the run (its proving, staging,
  replay and interior commitment, and the serving charge over stock vLLM on the same load) over the window's serving
  GPU-seconds, each read from the run's record and marked measured, a field the record lacks marked missing with its
  gap and the share then not claimed; and where it is claimed, the compute audit's part at it beside the part at the
  given share. A part whose entries are all on 0 GPUs adds no GPU-seconds and is printed `ON_CPUS` beside its 0.
  Proving read from its sessions' seconds, their wait included (`PROVE_SOURCE`), is measured, at most, and so is the
  share that sums it. Without a record it prints the share as `NO_RUN`. With a record the resampling holes are read at the workload's
  own commits (`spot_status.served`), and where the tree holds `served_row`, its section on what the audit showed
  is printed between the figures and the holes.

* **The compute audit's part** is the break-even planner's bound, read from its rows of record. A record is named in
  the store, never in code (`served_table`'s rule): the runs that carry `run_of_record=RECORD` and no `superseded_by`,
  each holding the planner's rows in one or more of `ROWS_FILES` in its run files, one JSON object a line, a case's rows
  over a scope from the first file that holds them (a later file holding them otherwise leaves the run out). A row
  prices one case of the plan (`row`, `step`, `world`) over one scope at one pinned audit budget β (`b10`): its share
  of total compute (`total`, with its parts `PARTS`) and its bound in bits (`bound_bits`). The command takes one case
  (`ROW`, the plan of record's row, `STEP`, `WORLD`), and for each scope reads off the budget at which the total is the
  share (`at_share`): the total is linear
  in β, a least-squares line over the pinned budgets, and ln(bound) is linear in ln β between the two that bracket
  it. The row plans for the weights / λ (`weights_bits / target_bits`), and the part is λ times its bound. The requests
  a row is priced over are its sampled replays over the probability a request is sampled (`replay_sample / q`); its
  `requests` field is the planner's default scope's count, another request's, and is not read. That part is the
  headline figure; the planner's driver prints the same rows' bound at the plan's own budget (`headline plan`) and,
  times λ, at the share between the two pinned budgets that bracket it (`headline rows`), and the print says so.
* **How that part's proving is priced** is read from the row's plan: each block family's share of the plan's proving
  time, each family priced as the rows say (`split.families`: its share by way, `ways`, else its ways, `priced`); a
  way is marked measured only where it is one of `MEASURED_WAYS`. A row's `split` is its own plan where it says
  the `q` and the sampled replays it is the plan of and they are the row's; the command then reads the row at the
  pinned budget nearest the one it reads the bound at. A `split` that doesn't is not the row's plan (it is written
  before the row's levers multiply its blocks) and its shares are not read. A run of record may instead hold the plan
  itself (`PLAN_FILE`, the sampling plan the row is priced over at one audit budget, with each cell's share), which is
  held to the rows it is read with: the same requests, and the rows' bound at its budget. With neither the command
  says so and marks no proving measured.
* **What prices that part's proving** is read from the row's own plan too (`_sources`): the share each source prices
  (`SOURCES`: a block type's own steady-state row, its Definition's family row, the planner's price, and any other
  the row names, such as the Lean verifier's seconds that no row times), as the row's
  `split.sources` records it, else by its `split.basis`, and the share at the planner's assumed default
  (`DEFAULT_BASIS`). A row whose plan records neither is printed as "not recorded".
* **Row `ASSUMED_ROW`** prices its proving at the planner's prices but where a table's rows price its blocks. Beside
  another row it is read at the world's year and printed only under its label, never as the figure, and the document
  holds it as its own field (`at_assumed_prices`); as the row read, its figure carries the label. Its label is what its
  rows say of their pricing (`_row_label`): their note, which the planner's driver writes where no table's row priced
  their proving or they priced under half of it; else the share of it the table's rows price; and `ASSUMED_LABEL`
  where they record none.
* **A row's note**, which the planner's driver writes on a row from the row's own data (that no table's row priced
  its proving, or what its scale reaches over the unscaled row of its case), is printed beside the figure where the
  figure carries no label, from the row at the pinned budget nearest the one it is read at.
* **The network attester's part** is `verity bench certifier_term`'s (`benchmarks/network_traces/leak_bounds.py`:
  `inputs`, `document`, whose document is `TERM_SCHEMA`): the schedules, the fail cells and the jitter of each of its
  rows, at the same scopes, for a class of at most `RESPONSE_IDS` ids a response. Its figure of record is the term
  at the calculator's jitter bound of record (`OF_RECORD`, two release schedules). A figure at its other jitter
  bound, and every figure of a row that carries the calculator's label (the row's `with_set_time_card`), is printed
  only under that label, which the command reads from the document and never spells itself.
* **The security level** is the row's binding bound, the planner's (`_binding`: the row's `verification_phase`'s
  `binding`): the two-stage first-hit finder's bound over one verification phase of the row's design and over the
  world's year of them, each counted in calls and in compressions, the inputs it assumes, and what Lean proves of
  each, a theorem of `LEAN_NUMBERS` by name or `NOT_PROVED`. The command computes none of it, names each figure by
  what it is over, and says so where the rows hold no bound.
* **Every figure carries the error summed over everything it covers**, and a figure whose error is 1 or more is
  printed as `NO_BOUND`, with that error and no amount. The compute audit's part fails with at most its rows' δ over
  the monitoring period. The network attester's fails with at most the ε of its row (`eps`): its scans' δ summed over
  every link of the scope and every execution phase of the monitoring period, one ε a scope and monitoring period.
  The two together fail with at most the sum of both. The scans' δ is an input of the calculator, and the command
  gives it one: `--scan-delta`, or by default the δ at which the row printed that sums the most execution phases, the
  world's year, carries `PERIOD_ERROR`, so that every row printed carries that or less. The network attester's
  result of record takes its δ so (`benchmarks/certifier_term/README.md`). No term depends on the scans' δ; only the
  errors move with it. The δ the command prints is the one the document states among its inputs (`scan_delta`).

Both count the same requests: the sampling plan's, which the network attester's calculator counts by default (its
`request_count`, read from the break-even planner in the tree). The command holds no count of its own. The two parts
must cover the same GPU-seconds, the same requests (the count the rows of record are priced over) and the same
weights, or the command refuses.

A link's execution phase carries K = schedules + cells + jitter + min(entries, advice + the wrong blocks' output
bits at δ). The first three are the network attester's part, the last term's second branch is the compute audit's
part, and the command sums them per scope and monitoring period under each of the network attester's rows of
`TOGETHER` (`together`): at the jitter bound of record, and under the label at the other. It prices no advice.

* **How the stop class is charged** is the answer of the stop verdicts' reader
  (`verity/protocols/resource_accounting/network/prover/stop_verdicts.py`'s `of_runs`), which the command hands the calculator (`certifier`'s
  ``stop``). The rows the calculator reads from verdict records (its document's `stop`'s `rows`: those that hold
  the lengths by a stop block) are charged below their schedules only on the execution phases the answer holds, and
  their whole schedules on every other. The runs handed to the reader are every run that carries the record
  `VERDICTS`, with those of them another run replaces: `stop_verdicts`' own list, the one the print names. With no
  run carrying the record the reader is not asked, and the rows are the calculator's with no record read. The
  document keeps the answer as the calculator's document states it, with the runs handed (`certifier`'s `stop`),
  and the print says how the stop class is charged and why. A row whose stop class the answer charges its whole
  schedule on some execution phase is called by `WHOLE`, not by `TOGETHER`: nothing holds its lengths there.
* **The output count.** Until a served audit has sampled a stop block the server chooses a response's output count,
  which the network attester's row `OUTPUT_COUNT` charges as its schedules. The sums that take it at 0 are those
  under the rows the answer names (`at_zero_in`: the rows of the document's `stop`, where the answer holds an
  execution phase; never the rows whose schedules read 0), on the execution phases it holds. The command prints
  the output count beside those sums and lists its 0 among the inputs, marked assumed. Where the answer holds no
  execution phase no sum takes the count at 0: the print says so, and the inputs list no output count.
* **The stop class's verdicts** move that mark, and only that one. A record named in the store (`VERDICTS`), its runs
  each holding verdicts' records in `VERDICTS_FILE`, one a line in the shape `VERDICT_FIELDS` (a record lacking a
  field is refused by its name), gives the four verdicts a stop class needs (`PIECES`) and the negative controls
  (`CONTROLS`). Where every piece is accepted and complete on one class (its honest verdicts each name it,
  `CLASS_FIELDS`, with their verifier's check that their stream's registration names its boundary, and agree on its
  registration, serve, class and rate, the row's `RATE`) and every control refused by the verifier that accepted
  its piece (`stop_class`), the output count's 0 is marked `RUN_NOT_PROVED`, printed "run (Python verifier, not
  proved)" with each verdict's run and verifier and counted apart from `measured`; it is `measured` only where every
  piece's record names the Lean guarantee that proves its verifier (`proved_by`). Otherwise the print names what is
  missing and the mark stays assumed. With no such record the print names no class. The mark is the 0's, so it is
  carried only where the answer names a row that takes the count at 0.

The command refuses a record nothing carries and a scope two unsuperseded runs both hold, and says how to label it.
A run of record that is superseded, or holds no rows of the case, is left out and listed under "Refused".

Under a research run, a runner that names metric ids for the figures (`benchmarks/exfiltration/run.py`'s `METRICS`)
has `record` write them through the harness kit (`measurements`): the headline figure, the compute audit's part and
the network attester's, each over every monitoring period of the document it has a value over at the share (the
world's year always, and last, the dashboard's headline), each with its `period` and the compute it is `over`, its mark
at the head of its first input and its error over that period in the next, the document as `DOCUMENT` and what ran in
the fingerprint. What it prints is the same in a run as outside one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity_bench import kit, settings, spot_status
from verity_bench.exfiltration_common import (
    AMOUNTS,
    ASSUMED_LABEL,
    ASSUMED_ROW,
    CLASS_FIELDS,
    CLASS_NAME,
    CONTROLS,
    DEFAULT_BASIS,
    DOCUMENT,
    FIGURES,
    ISSUE_ID,
    MARKS,
    MODEL,
    NETWORK_CHECK_SHARE,
    NO_BOUND,
    NO_RUN,
    ON_CPUS,
    OUTPUT_COUNT,
    PERIOD_ERROR,
    PIECES,
    PLAN_FILE,
    PROVE_SOURCE,
    PROVE_SOURCE_GAP,
    PROVE_SOURCES,
    RECORD,
    RESPONSE_IDS,
    ROW,
    ROWS_FILES,
    RUN_NOT_PROVED,
    SCHEMA,
    SCOPES,
    SECURITY,
    SERVED_PARTS,
    SERVING_GAP,
    SHARE,
    STEP,
    TERM_SCHEMA,
    TOGETHER,
    UNSTATED,
    VARIANT,
    VERDICT_FIELDS,
    VERDICT_READS,
    VERDICTS,
    VERDICTS_FILE,
    WORLD,
    _load,
    _reader,
)
from verity_bench.exfiltration_document import _inputs, _row_label, build
from verity_bench.exfiltration_planner import _sources, at_share, record_job, record_run, rows_of_record
from verity_bench.exfiltration_render import render
from verity_bench.exfiltration_served import served_share
from verity_bench.exfiltration_text import _listed, _n, _names, _p, _part, _pct, _table
from verity_bench.exfiltration_verdicts import verdict
from verity_bench.served_table import RUN_OF_RECORD, SUPERSEDED_BY

__all__ = [
    "AMOUNTS",
    "ASSUMED_LABEL",
    "ASSUMED_ROW",
    "CLASS_FIELDS",
    "CLASS_NAME",
    "CONTROLS",
    "DEFAULT_BASIS",
    "DOCUMENT",
    "FIGURES",
    "ISSUE_ID",
    "MARKS",
    "NETWORK_CHECK_SHARE",
    "NO_BOUND",
    "NO_RUN",
    "ON_CPUS",
    "OUTPUT_COUNT",
    "PERIOD_ERROR",
    "PIECES",
    "PLAN_FILE",
    "PROVE_SOURCE",
    "PROVE_SOURCES",
    "PROVE_SOURCE_GAP",
    "RECORD",
    "RESPONSE_IDS",
    "ROW",
    "ROWS_FILES",
    "RUN_NOT_PROVED",
    "RUN_OF_RECORD",
    "SCHEMA",
    "SCOPES",
    "SECURITY",
    "SERVED_PARTS",
    "SERVING_GAP",
    "SHARE",
    "STEP",
    "SUPERSEDED_BY",
    "TERM_SCHEMA",
    "TOGETHER",
    "UNSTATED",
    "VERDICTS",
    "VERDICTS_FILE",
    "VERDICT_FIELDS",
    "VERDICT_READS",
    "WORLD",
    "_inputs",
    "_listed",
    "_load",
    "_n",
    "_names",
    "_p",
    "_part",
    "_reader",
    "_row_label",
    "_sources",
    "_table",
    "at_share",
    "build",
    "main",
    "measurements",
    "record",
    "render",
    "rows_of_record",
    "served_share",
    "verdict",
]


def measurements(doc: Mapping[str, Any], metrics: Mapping[str, str]) -> list[dict]:
    """The kit writer's measurements of a document (`build`), one a metric id of `metrics` and monitoring period of the
    document's (`periods`), its `period` and the compute it is `over` (the world's or a datacenter's): the figure of
    `FIGURES` its value names, over that period at the document's share, in GB. The other periods come in the
    document's order and the world's year last, so a reader that keys a result's measurements by name alone, the last
    of a name winning, reads the world's year. Each has its `mark`, which also heads its first input, then its error
    over its period (the compute audit's δ; the network attester's ε at its scans' δ; the headline's, the compute
    audit's part, its δ), then the runs of record it read. The network attester's part is modeled, with the mark its
    calculator gives the `scan` row where it gives one. A figure with no value over a period (a period no run of record
    holds, or the network attester's where its ε is 1 or more) is left out, but over the world's year, where it has
    no value and says why: `NO_BOUND` with that ε."""

    if bad := sorted(set(metrics.values()) - set(FIGURES)):
        raise ValueError(f"no figure of the document is named {', '.join(bad)}: a metric records one of {', '.join(FIGURES)}")
    share, cert, runs = doc["share"], doc["certifier"], list(doc["runs"])
    scan = cert["legend"]["scan"].get("mark")
    term = (f"the network attester's term of record, its `scan` row {TOGETHER['scan']} at D_j = {cert['of_record']}, on "
            f"an assumed class of at most {_n(cert['response_ids'])} ids a response")
    out = []
    for p in sorted(doc["periods"], key=lambda p: (p["compute"], p["period"]) == ("world", "year")):
        scope, span, eps = f"{p['compute']} {p['period']}", f"{p['compute']}'s {p['period']}", p["eps"]
        error = (f"its error: ε = {_p(eps)} over the {span}, at a scan's δ of {_p(cert['scan_delta']['value'])} "
                 f"({cert['scan_delta']['mark']})")
        figures: dict[str, tuple[Any, str, list[str], str | None]] = {
            "network_certifier": (p["together"]["scan"]["certifier_GB"], "modeled",
                                         [f"modeled{f' ({scan})' if scan else ''}: {term}", error],
                                         f"{NO_BOUND}: ε = {_n(eps)} over the {span}")}
        head = next((x for x in doc["compute_audit"] if x["scope"] == scope), None)
        for figure, mark, what in (("headline", doc["status"], "the headline figure"),
                                   ("compute_audit", "modeled", "the compute audit's part")):
            if head is None:
                continue
            label = (_row_label(head) if doc["row"] == ASSUMED_ROW else None) or head.get("note")
            audit = (f"the planner's row `{doc['row']}`{f', {label}' if label else ''}, read where its total is "
                     f"{_pct(share)} of compute, β = {_part(head['beta'])}{'' if head['inside'] else ' (extrapolated)'}: "
                     f"λ = {_n(head['lambda'])} times its bound, {_n(head['bound']['GB'])} GB")
            figures[figure] = (head["GB"], mark, [f"{mark}: {what}, {audit}",
                                                  (f"its error: δ = {_n(head['delta'])} over the {span}, the compute "
                                                  "audit's"), *runs], None)
        for name, figure in metrics.items():
            value, mark, inputs, why = figures.get(figure, (None,) * 4)
            if value is None and scope != "world year":
                continue
            m = {"name": name, "value": value, "unit": f"GB a {scope} at {_pct(share)} of compute", "mark": mark,
                 "period": p["period"], "over": p["compute"], "inputs": inputs}
            out.append(m if value is not None else m | {"why": why})
    return out


def record(doc: Mapping[str, Any], metrics: Mapping[str, str]) -> None:
    """Writes a document (`build`) into this research run's result (`kit.writer`): the document as `DOCUMENT`, named a
    measurement file and the validation's evidence, what ran in the fingerprint, `measurements` under `metrics`' ids,
    and the validation: the checks `build` refuses a document without, which it passed to be written."""

    writer = kit.writer(settings.RESEARCH_RUN_DIR)
    kept = writer.out / DOCUMENT
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    head = next(p for p in doc["compute_audit"] if p["scope"] == "world year")
    cert = doc["certifier"]
    writer.fingerprint(benchmark="exfiltration", row=doc["row"], step=head["step"], variant=doc["variant"],
                       share=doc["share"], network_check_share=doc["network_check"]["share"],
                       response_ids=cert["response_ids"], scan_delta=cert["scan_delta"]["value"],
                       workload=doc["compute_share"]["run"], runs=list(doc["runs"]))
    writer.measurement_file(DOCUMENT)
    for m in measurements(doc, metrics):
        writer.measurement(**m)
    writer.validation("passed", "verity_bench.exfiltration.build",
                      "the two parts cover the same GPU-seconds and requests over every scope both hold, and the same "
                      f"weights; each of the {len(doc['inputs'])} inputs carries one of {', '.join(MARKS)} with its "
                      "source", [DOCUMENT])


class _Commits:
    """The one call `spot_status.status` makes of a store, answered from a run's record directory: the commit its
    `job.json` says the run ran (`computation.tree_sha`), where it says one."""

    def __init__(self, record: Path):
        self.run = record_run(record)
        self.commit = (record_job(record).get("computation") or {}).get("tree_sha")

    def get_attempt(self, run: str) -> dict | None:
        return {"source": {"commit": self.commit}} if run == self.run else None


def main(argv: Sequence[str] | None = None, metrics: Mapping[str, str] | None = None,
         report: Callable[..., int] | None = None) -> int:
    """`verity bench exfiltration`: print the two parts (`render`), or with ``--json`` their document (`build`).
    `metrics`: a runner's metric ids, each for the figure of `FIGURES` it names, which under a research run
    (`RESEARCH_RUN_DIR` set) `record` writes into the run's result; the print is the same.

    With ``--from`` (a run's record directory, or a run id where research is installed: `kit.record`), the rows of
    record are that record's, read with no store (`build`'s ``record``), and `report`, the runner's call of
    `kit.report`, prints and writes them: ``report(doc, text, made_up=..., withdrawn=...)``, its exit code the
    command's. A record whose `job.json` says it is made up (``made_up``, what it says of itself) is reported as made
    up, and the print says what it says."""

    ap = argparse.ArgumentParser(
        prog="verity bench exfiltration",
        description=f"{MODEL}'s exfiltration capacity in GB over a week, a month and a year of the world's and a "
                    "datacenter's compute, at a share of total compute: the compute audit's part, from the break-even "
                    "planner's rows of record in the evidence store, beside the network attester's part, with every "
                    "input labelled measured, modeled, simulated, assumed, missing or run, not proved. Every figure "
                    "carries the error summed over "
                    f"everything it covers, and one whose error is 1 or more is printed as \"{NO_BOUND}\", with no "
                    "amount.")
    ap.add_argument("--json", action="store_true", help="print the document (value, unit, inputs, parts, periods)")
    ap.add_argument("--share", type=float, default=SHARE, metavar="S",
                    help=f"the share of total compute the figure is read at (default {SHARE})")
    ap.add_argument("--network-check-share", type=float, default=NETWORK_CHECK_SHARE, metavar="S",
                    help="the network check's own proving as a share of total compute, which the planner's rows don't "
                         f"count (default {NETWORK_CHECK_SHARE}, assumed)")
    ap.add_argument("--row", default=ROW, help=f"the planner's row (default `{ROW}`, the plan of record's; `{ASSUMED_ROW}` "
                                               "is printed only under its label, what its rows say of their pricing)")
    ap.add_argument("--step", default=STEP, help=f"the row's step, or the beginning of its name (default `{STEP}`)")
    ap.add_argument("--variant", help=f"the row's variant, as the planner's driver names it (`{VARIANT}`), where a run "
                                      "holds several (default: a run holds one)")
    ap.add_argument("--response-ids", type=int, default=RESPONSE_IDS, metavar="N",
                    help="the most ids a response holds in the network attester's class (default "
                         f"{RESPONSE_IDS}, assumed: the class that holds the plan of record's longest output)")
    ap.add_argument("--scan-delta", type=Fraction, metavar="D",
                    help="the δ of each execution phase's scan of the network attester, which its rows' ε sum (default: "
                         "the δ at which the row printed that sums the most execution phases, the world's year, carries "
                         f"an ε of {PERIOD_ERROR}, assumed; no term depends on it)")
    ap.add_argument("--workload", type=Path, metavar="FILE",
                    help="a served two-stage run's record as its reader gives it, one JSON object (`served_share`): the "
                         "share of compute it measures is printed beside the given share, with the compute audit's part "
                         f"at it (default: none, and the share is \"{NO_RUN}\")")
    ap.add_argument("--local-only", action="store_true", help="read this machine's catalog alone, not the store's remote")
    ap.add_argument("--from", dest="record", metavar="DIR",
                    help="read the planner's rows from one run's record directory (its job.json and out/), with no "
                         "store; a run id is fetched to one first where research is installed")
    a = ap.parse_args(argv)
    if a.record is not None and (a.json or report is None):
        ap.error("--from prints the figure and its document through the runner (benchmarks/exfiltration/run.py), "
                 "without --json: run `verity bench exfiltration --from DIR`")
    try:
        rec = None if a.record is None else kit.record(a.record)
        served = None if a.workload is None else json.loads(a.workload.read_text())
        doc = build(share=a.share, network_check=a.network_check_share, row=a.row, step=a.step, variant=a.variant,
                    response_ids=a.response_ids, scan_delta=a.scan_delta, served=served, local_only=a.local_only,
                    record=None if rec is None else rec.path)
        doc["spot_holes"] = (spot_status.served(served) if served is not None else
                             spot_status.status(doc["runs"]) if rec is None else
                             spot_status.status(doc["runs"], store=_Commits(rec.path)))
    except (ValueError, RuntimeError, OSError) as e:
        print(f"verity bench exfiltration: {e}", file=sys.stderr)
        return 2
    shown = None
    if served is not None and importlib.util.find_spec("verity_bench.served_row") is not None:
        from verity_bench import served_row  # type: ignore[attr-defined]  # optional; it reads this module's names

        shown = served_row.account(served)
    if shown is not None:
        doc["workload"] = shown
    if rec is not None:
        said = record_job(rec.path).get("made_up")
        text = render(doc)
        if said:
            doc["made_up_says"] = str(said)
            head, _, rest = text.partition("\n")
            text = f"{head}\n  what the record says of itself: {said}\n{rest}"
        text += ("" if shown is None else "\n\n" + served_row.render(shown)) + "\n\n" + spot_status.render(doc["spot_holes"])
        assert report is not None
        return report(doc, text, made_up=bool(said), withdrawn=rec.withdrawn)
    if metrics and settings.is_set("RESEARCH_RUN_DIR"):
        try:
            record(doc, metrics)
        except (ValueError, OSError) as e:
            print(f"verity bench exfiltration: {e}", file=sys.stderr)
            return 2
    print(json.dumps(doc, indent=1, ensure_ascii=False) if a.json else
          render(doc) + ("" if shown is None else "\n\n" + served_row.render(shown))
          + "\n\n" + spot_status.render(doc["spot_holes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
