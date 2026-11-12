"""`verity bench exfiltration`: GLM-5.3's exfiltration capacity over a monitoring period at a share of total compute
(10% by default), in GB, as two parts side by side with every input labelled measured, modeled or assumed. It
computes neither part itself: it composes the two calculators that do, and runs on any machine that reaches the
store, with no GPU and no node.

* **The share of total compute** is given (`--share`) and assumed. Beside it the command prints the share a served
  two-stage run measures (`served_share`, `--served-run`): the audit's GPU-seconds on the run (its proving, staging,
  replay and interior commitment, and the serving charge over stock vLLM on the same load) over the window's serving
  GPU-seconds, each read from the run's record and marked measured, a field the record lacks marked missing with its
  gap and the share then not claimed; and where it is claimed, the compute audit's part at it beside the part at the
  given share. Without a record it prints the share as `NO_RUN`.

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
  `requests` field is the planner's default scope's count, another request's, and is not read.
* **How that part's proving is priced** is read from the row's plan: each block family's share of the plan's proving
  time, each family priced as the rows say (`split.families`, `priced`). A row's `split` is its own plan where it says
  the `q` and the sampled replays it is the plan of and they are the row's; the command then reads the row at the
  pinned budget nearest the one it reads the bound at. A `split` that doesn't is not the row's plan (it is written
  before the row's levers multiply its blocks) and its shares are not read. A run of record may instead hold the plan
  itself (`PLAN_FILE`, the sampling plan the row is priced over at one audit budget, with each cell's share), which is
  held to the rows it is read with: the same requests, and the rows' bound at its budget. With neither the command
  says so and marks no proving measured.
* **What prices that part's proving** is read from the row's own plan too (`_sources`): the share each source prices
  (`SOURCES`: a block type's own steady-state row, its Definition's family row, the planner's price), as the row's
  `split.sources` records it, else by its `split.basis`, and the share at the planner's assumed default
  (`DEFAULT_BASIS`). A row whose plan records neither is printed as "not recorded".
* **Row `ASSUMED_ROW`** prices its proving at the planner's assumed prices. Beside another row it is read at the
  world's year and printed only under `ASSUMED_LABEL`, never as the figure, and the document holds it as its own field
  (`at_assumed_prices`); as the row read, its figure carries the label.
* **The network certifier's part** is `verity bench certifier_term`'s (`benchmarks/network_traces/leak_bounds.py`:
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
  the monitoring period. The network certifier's fails with at most the ε of its row (`eps`): its scans' δ summed over
  every link of the scope and every execution phase of the monitoring period, one ε a scope and monitoring period.
  The two together fail with at most the sum of both. The scans' δ is an input of the calculator, and the command
  gives it one: `--scan-delta`, or by default the δ at which the row printed that sums the most execution phases, the
  world's year, carries `PERIOD_ERROR`, so that every row printed carries that or less. The network certifier's
  result of record takes its δ so (`benchmarks/certifier_term/README.md`). No term depends on the scans' δ; only the
  errors move with it. The δ the command prints is the one the document states among its inputs (`scan_delta`).

Both count the same requests: the sampling plan's, which the network certifier's calculator counts by default (its
`request_count`, read from the break-even planner in the tree). The command holds no count of its own. The two parts
must cover the same GPU-seconds, the same requests (the count the rows of record are priced over) and the same
weights, or the command refuses.

A link's execution phase carries K = schedules + cells + jitter + min(entries, advice + the wrong blocks' output
bits at δ). The first three are the network certifier's part, the last term's second branch is the compute audit's
part, and the command sums them per scope and monitoring period under each of the network certifier's rows of
`TOGETHER` (`together`): at the jitter bound of record, and under the label at the other. It prices no advice. The
sums under the rows whose schedules are 0, the lengths held by a stop block, take a response's output count at 0.
Until a served audit has sampled a stop block the server chooses that count, which the network certifier's row
`OUTPUT_COUNT` charges as its schedules: the command prints it beside those sums, marked assumed.

* **The stop class's verdicts** move that mark, and only that one. A record named in the store (`VERDICTS`), its runs
  each holding verdicts' records in `VERDICTS_FILE`, one a line in the shape `VERDICT_FIELDS` (a record lacking a
  field is refused by its name), gives the four verdicts a stop class needs (`PIECES`) and the negative controls
  (`CONTROLS`). Where every piece is accepted and complete and every control refused by the verifier that accepted
  its piece (`stop_class`), the output count's 0 is marked `RUN_NOT_PROVED`, printed "run (Python verifier, not
  proved)" with each verdict's run and verifier and counted apart from `measured`; it is `measured` only where every
  piece's record names the Lean guarantee that proves its verifier (`proved_by`). Otherwise the print names what is
  missing and the mark stays assumed. With no such record the print is as it was without one.

The command refuses a record nothing carries and a scope two unsuperseded runs both hold, and says how to label it.
A run of record that is superseded, or holds no rows of the case, is left out and listed under "Refused".
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity_bench import jobs, spot_status
from verity_bench.served_table import RUN_OF_RECORD, SUPERSEDED_BY, record_runs, records

SCHEMA = "exfiltration/v3"
#: the network certifier's document the command reads: its `value` the term of record, its rows' `term` by jitter
#: bound and their ε (`eps`), its inputs, and the label of what rests on a set-time card, at its top and on each row
#: that carries it
TERM_SCHEMA = "certifier-term/v2"
#: the network certifier's scopes the command prints
SCOPES = ("datacenter", "world")
#: the error a monitoring period's figure is to carry at most. Given no δ for the network certifier's scans, the
#: command takes the one at which the row it prints that sums the most execution phases carries this
PERIOD_ERROR = Fraction(1, 2 ** 20)
#: what a figure is printed as where its error, summed over everything it covers, is 1 or more: it carries no amount
NO_BOUND = "no bound at this period"
#: a network certifier's row's amounts, which a row with no bound doesn't carry
AMOUNTS = ("schedules", "cells", "jitter", "term", "entries")
#: a sum's two figures, the network certifier's term and the two parts together, each with the key of its error in
#: its scope's monitoring period
ERRORS = {"certifier_GB": "eps", "GB": "error"}
MODEL = "GLM-5.3"
#: the record whose runs hold the planner's rows, and the rows' files in a run's files: the driver's default command's
#: and its rows over the world's year alone
RECORD = "exfiltration-glm53-compute-audit"
ROWS_FILES = ("out/rows.jsonl", "out/three/rows.jsonl")
#: a row's plan in a run's files: the sampling plan the row is priced over at one audit budget (`beta`), with each
#: cell's block and its share of the plan's proving time (`cells`, and `left_out` for the cells it lists apart)
PLAN_FILE = "out/mix.json"
#: a plan's bound and its rows' at the plan's budget agree to this, or it is another row's plan
PLAN_TOLERANCE = 1e-3
#: the field in which the planner's driver names a row's variant, where it writes several of one row
VARIANT = "h8"
#: what makes rows one case of the plan over one scope, which one file of a run holds
CASE_KEYS = ("world", "row", "step", "scope", VARIANT)
#: how a block family is priced where the rows don't say
UNSTATED = "in no stated way"
#: what prices a plan's proving, by the planner's driver's names (`split.sources`): a block type's own steady-state
#: row, its Definition's at the same static K, and the planner's price
SOURCES = ("own row", "family row", "planner's price")
#: the source of each `split.basis` key, by its text before any colon; every other key is the planner's price, but
#: its copy stratum's model (`COPY_BASIS`), which is a source of its own
BASIS_SOURCES = {"steady": "own row", "family": "family row"}
COPY_BASIS = "headline.copy_stratum"
#: the planner's assumed default price, a key of `split.basis`
DEFAULT_BASIS = "reprice.DEFAULT"
#: printed where a row's plan records no source
NOT_RECORDED = "not recorded"
#: the share of total compute the figure is read at
SHARE = 0.10
#: the network check's own proving as a share of total compute, which the planner's rows don't count: no run measures it
NETWORK_CHECK_SHARE = 0.003
#: the case of the plan the figure is: the planner's world (rho's world fleet), the plan of record's row, with every
#: modeled lever at its measured value, and the step of that row that prices the verifier as measured (a prefix of the
#: step's name)
WORLD, ROW, STEP = "ours", "(ii) honest", "+ the verifier as measured"
#: the row the planner prices at its assumed prices, and what every figure of it is printed under
ASSUMED_ROW = "(i) honest"
ASSUMED_LABEL = "at the planner's assumed prices"
#: the most ids a response holds in the network certifier's class the figure is read at: its class that holds the
#: longest output the plan of record serves (assumed, as every class is: no run has certified one)
RESPONSE_IDS = 1024
#: a row's parts of total compute; `wait` is a part of `serve`
PARTS = ("serve", "wait", "prove", "replay", "pool", "edge", "compile")
#: the network certifier's rows the two parts are summed under, and what each is called beside the compute audit's
TOGETHER = {"scan": "as the scan's check charges it today", "N2": "as N2 charges it today",
            "all three": "with the lengths held by a stop block, the timing taken once and the entries charged through "
                         "the compute audit",
            "design, no stop block": "by the design no run has behind it, before a served audit has sampled a stop block",
            "design": "by the design no run has behind it, once a served audit has sampled a stop block"}
#: the network certifier's row whose schedules are what a response's output count is charged while the server chooses
#: it: the design before a served audit has sampled a stop block
OUTPUT_COUNT = "design, no stop block"
GB = 8e9
#: an input whose value a run's verdicts give, each checked by a verifier no Lean proof covers yet, and how it prints
RUN_NOT_PROVED = "run, not proved"
PRINTED = {RUN_NOT_PROVED: "run (Python verifier, not proved)"}
#: an input's marks: `missing` is a served run's field its record lacks, which the share it would measure is not claimed
#: without; `RUN_NOT_PROVED` is never `measured`, which a verdict is only where its record names the Lean guarantee
#: that proves its verifier (`proved_by`)
MARKS = ("measured", "modeled", "assumed", "missing", RUN_NOT_PROVED)
#: the record whose runs hold the verdicts on the stop class, each verdict's record one JSON object a line of
#: `VERDICTS_FILE` in a run's files
VERDICTS = "exfiltration-glm53-stop-verdicts"
VERDICTS_FILE = "out/verdicts.jsonl"
#: the four verdicts on a stop class (a verdict's `piece`), each accepted and complete, before the sums under the rows
#: whose schedules are 0 take a response's output count at 0 on a run: one missing or not accepted, and none does
PIECES = {"stop_record": "the stop record", "response_tie": "the response tie", "request_tie": "the request tie",
          "second_stage": "the compute audit's second stage"}
#: the negative controls (a verdict's `control`; null on the honest run): a run on the same path with one thing changed,
#: each refused by the verifier that accepted the honest run's verdict on the same piece
CONTROLS = {"id": "a changed id", "max_tokens": "a changed `max_tokens`"}
#: a verdict: accepted and complete, accepted and not complete (which moves no figure), or refused
VERDICT_READS = ("accept", "not complete", "refuse")
#: a verdict's record: its `piece` and `control`, the `run` that produced it, the `verifier` that checked it (its
#: `module` and `commit`), the `verdict`, `proved_by` (the Lean guarantee whose proof covers that verifier, null until
#: one does) and `open` (while `proved_by` is null, what that proof waits on)
VERDICT_FIELDS = ("piece", "control", "run", "verifier", "verdict", "proved_by", "open")
VERIFIER_FIELDS = ("module", "commit")
#: the audit's GPU time on a served two-stage run, by part, which the share of compute it measures sums: what each part
#: is, and why a run's record lacks it where it does
SERVED_PARTS = {
    "prove": ("the sessions' proving", "the two-stage audit's record keeps each session's proving seconds "
              "(`audit.timing`) and not the GPUs it held"),
    "staging": ("staging the sampled proof blocks for their sessions", "the one-stage audit times its staging "
                "(`staged_s`) and its summary leaves the seconds out"),
    "replay": ("the replay of the sampled spans", "the two-stage audit reads no clock over its replay"),
    "interior": ("the interior commitment of the sampled spans' proof blocks",
                 "the two-stage audit reads no clock over its interior commitment"),
    "serving_charge": ("serving under Verity over stock vLLM serving the same load", "no stock serve of the window's "
                       "load (`stock`) and no serving cost of the served arm over stock (`serving_cost`, "
                       "`verity bench serve_tax`)"),
}
#: why a run's record lacks the window's serving, which the share is of
SERVING_GAP = ("the window's serve is not tied to the audit's record: its load's seconds and GPUs (serve_tax's "
               "`meta.json`: `wall_s`, `gpu`) are in the serve's own run")
#: what the command prints for the share a served run measures where no record is given
NO_RUN = "not measured: no served run's record"
#: the inputs both parts read: the fleet and what a request of the plan is, which give the requests, and the weights
SHARED = ("world_machines", "datacenter_machines", "machine_gpus", "mfu_serve", "peak_flops", "request_tokens",
          "token_macs", "world_requests", "request_count", "weights")
#: the part of the inputs a row's binding bound assumes, and what the bound is called
SECURITY = "security level"
#: where the Lean instances a binding bound names are declared
LEAN_NUMBERS = "Claims.Flock.Numbers"
#: a Lean instance's numerals, as a binding bound names them (`lean.past`), in the bound's own symbols
NUMERALS = {"T": "calls a run", "e": "reruns a session (e_s)", "Ns": "sessions (N_s)", "Ks": "classes of gates (K_s)",
            "Km": "classes of gates, session by session (K_m)", "c": "calls of the run's own opening checks (c)"}
#: the Lean statement whose right-hand side a binding bound counted in calls is
LEAN_BOUND = "CompiledTwo.eventTwoX_le_cls_at_budget"
#: printed where no Lean theorem gives a number at a design's sizes
NOT_PROVED = "not proved at these sizes"


def _load(root: Path, directory: str, module: str) -> Any:
    """A module of another benchmark's directory, which is no package: loaded by its path."""

    path = str(root / "benchmarks" / directory)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(module)


# ---- the compute audit's part: the planner's rows of record -----------------------------------------------------------


def _label(key: str = RUN_OF_RECORD, value: str = RECORD) -> str:
    return f"`research data label RUN {key} {value} --by LANE`"


def _run_files(store: Any, run: str, names: Sequence[str]) -> tuple[Path | None, list[str], str | None]:
    """Where the files ``names`` of a run's files that it holds are fetched to, and which it holds; or why it has none:
    the run is not in the store, or kept no run files."""

    attempt = store.get_attempt(run)
    if attempt is None:
        return None, [], "not in the store or on its remote"
    out = attempt.get("outputs") or {}
    files = out.get("run_files") or out.get("run_record")
    if not files:
        return None, [], "kept no run files"
    held = [name for name in names if store.listing(files, [name])]
    return (Path(store.fetch(files, paths=held)) if held else None), held, None


def read_run(store: Any, run: str) -> tuple[list[dict], dict | None, str | None]:
    """The planner's rows and the plan in a run's files, either of which it may lack, or why the run holds neither or
    is not read. The rows are its rows files' (`ROWS_FILES`), a case's over a scope the first file's that holds them; a
    later file that holds other rows of that case and scope leaves the run unread."""

    at, held, why = _run_files(store, run, (*ROWS_FILES, PLAN_FILE))
    if why:
        return [], None, why
    if at is None:
        return [], None, f"holds no planner rows ({', '.join(f'`{n}`' for n in ROWS_FILES)}) and no plan (`{PLAN_FILE}`)"
    rows: list[dict] = []
    first: dict[tuple, tuple[str, list[str]]] = {}
    for name in (n for n in ROWS_FILES if n in held):
        cases: dict[tuple, list[dict]] = {}
        for line in (at / name).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cases.setdefault(tuple(str(r.get(k)) for k in CASE_KEYS), []).append(r)
        for case, its in cases.items():
            text = sorted(json.dumps(r, sort_keys=True) for r in its)
            if case not in first:
                first[case] = (name, text)
                rows += its
            elif first[case][1] != text:
                return [], None, (f"holds row `{case[1]}` over the {case[3]} in `{first[case][0]}` and in `{name}`, and the "
                                  "two differ")
    return rows, json.loads((at / PLAN_FILE).read_text()) if PLAN_FILE in held else None, None


def _case(x: Mapping, row: str, step: str, world: str) -> bool:
    return (x.get("world"), x.get("row")) == (world, row) and str(x.get("step", "")).startswith(step)


def rows_of_record(store: Any, *, row: str = ROW, step: str = STEP, world: str = WORLD,
                   variant: str | None = None) -> tuple[dict, dict, list[str], list[dict]]:
    """The case's rows of record by scope, each scope's from the one run that holds it, and its plans of record the
    same way, with the runs read and the runs left out with why. A run that holds a scope's plan may hold the plan's
    own row, at the plan's budget: it is the plan's, and is not read off. With ``variant``, only the rows the
    planner's driver names so (`VARIANT`) are read."""

    labelled = record_runs(store, records(store).get(RECORD, []))
    if not labelled:
        raise ValueError(f"no planner rows of record: nothing in the store carries `{RUN_OF_RECORD}={RECORD}` "
                         f"(label each run that holds them: {_label()})")
    scopes: dict[str, tuple[str, list[dict]]] = {}
    plans: dict[str, tuple[str, dict]] = {}
    runs, refused = [], []
    for run in sorted({run for run, _ in labelled}, reverse=True):
        found, plan = [], None
        if superseded := [str(lab.value) for lab in store.labels(run, key=SUPERSEDED_BY)]:
            why = f"superseded by {', '.join(superseded)}"
        else:
            found, plan, why = read_run(store, run)
            found = [r for r in found if not r.get("error") and _case(r, row, step, world)
                     and variant in (None, r.get(VARIANT))]
            if plan is not None and _case(plan, row, step, world):
                found = [r for r in found if (r["scope"], r["b10"]) != (plan["scope"], plan["beta"])]
            else:
                plan = None
            why = why or (None if found or plan else f"holds no row `{row}` at a step `{step}...` of the world `{world}`")
        if why:
            refused.append({"target": run, "record": RECORD, "why": why})
            continue
        runs.append(run)
        if plan:
            if plan["scope"] in plans:
                raise ValueError(f"{plans[plan['scope']][0]} and {run} both hold the {plan['scope']}'s plan and neither is "
                                 f"superseded: label the one the other replaces ({_label(SUPERSEDED_BY, 'NEWER')})")
            plans[plan["scope"]] = (run, plan)
        by: dict[str, list[dict]] = {}
        for r in found:
            by.setdefault(r["scope"], []).append(r)
        for scope, its in by.items():
            if scope in scopes:
                raise ValueError(f"{scopes[scope][0]} and {run} both hold the {scope}'s rows and neither is superseded: "
                                 f"label the one the other replaces ({_label(SUPERSEDED_BY, 'NEWER')})")
            if len(steps := sorted({r["step"] for r in its})) > 1:
                raise ValueError(f"`{step}` begins {len(steps)} steps of {run}'s row `{row}`: give more of the one meant "
                                 f"(--step): {'; '.join(steps)}")
            scopes[scope] = (run, its)
    if not scopes:
        left = "; ".join(f"{x['target']} {x['why']}" for x in refused) or f"{', '.join(runs)} hold a plan alone"
        raise ValueError(f"no run of record holds the planner's rows: {left} (label one that does: {_label()})")
    for scope in sorted(set(plans) - set(scopes)):
        refused.append({"target": plans.pop(scope)[0], "record": RECORD,
                        "why": f"holds the {scope}'s plan, and no run of record holds that scope's rows"})
    read = {run for run, _ in (*scopes.values(), *plans.values())}
    return scopes, plans, [run for run in runs if run in read], refused


def _fit(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """The least-squares line through ``points``: its slope and intercept."""

    n = len(points)
    mx, my = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    return slope, my - slope * mx


def _between(rows: Sequence[Mapping], key: str, beta: float, *, log: bool = False) -> float:
    """``key`` at the budget ``beta``, on the line through the two pinned budgets that bracket it (the nearest two
    where none do): linear in β, or with ``log`` ln(key) linear in ln β."""

    i = max(0, min(len(rows) - 2, sum(r["b10"] <= beta for r in rows) - 1))
    lo, hi = rows[i], rows[i + 1]
    if log:
        t = (math.log(beta) - math.log(lo["b10"])) / (math.log(hi["b10"]) - math.log(lo["b10"]))
        return math.exp(math.log(lo[key]) + (math.log(hi[key]) - math.log(lo[key])) * t)
    return lo[key] + (hi[key] - lo[key]) * (beta - lo["b10"]) / (hi["b10"] - lo["b10"])


def _requests(x: Mapping) -> float | None:
    """The requests a row or a plan is priced over: its sampled replays over the probability a request is sampled."""

    q, sample = x.get("q"), x.get("replay_sample")
    return sample / q if q and sample else None


def _priced(shares: Mapping[str, float], said: Mapping) -> dict[str, dict]:
    """A plan's proving time by how its block families are priced: each way's share of it and its families with
    theirs, largest first. ``shares`` are the plan's, by family; how a family is priced is the rows' account
    (``said``, their `split.families`, `priced`), and `UNSTATED` for a family they don't name."""

    out: dict[str, dict] = {}
    for family, its in sorted(shares.items(), key=lambda kv: -kv[1]):
        ways = (said.get(family) or {}).get("priced")
        way = out.setdefault(" and ".join(sorted(ways, reverse=True)) if ways else UNSTATED, {"share": 0.0, "families": {}})
        way["share"] += its
        way["families"][family] = its
    return out


def _sources(split: Mapping) -> dict | None:
    """The share of a plan's proving each source prices, as its row's `split` records it: its `sources`, else its
    `basis` by `BASIS_SOURCES`, each of `SOURCES` at 0 where it prices none (`by`); and the share at the planner's
    assumed default (`default`, None where the split holds no `basis`). None where it records neither."""

    basis = split.get("basis") or {}
    by = dict(split.get("sources") or {})
    if not by:
        for key, share in basis.items():
            source = BASIS_SOURCES.get(key.split(":")[0]) or (key if key == COPY_BASIS else SOURCES[-1])
            by[source] = by.get(source, 0.0) + share
    if not by:
        return None
    return {"by": {s: by.get(s, 0.0) for s in SOURCES} | {s: v for s, v in by.items() if s not in SOURCES},
            "default": basis.get(DEFAULT_BASIS, 0.0) if basis else None}


def _own(run: str, row: Mapping) -> dict | None:
    """A row's own plan, where its `split` is one: it says the `q` and the sampled replays it is the plan of, and they
    are the row's. How its proving is priced (`_priced`) and what prices it (`_sources`), or None for a `split` that is
    not the row's plan."""

    split = row.get("split") or {}
    if not all(split.get(k) and row.get(k) and _same(split[k], row[k]) for k in ("q", "replay_sample")):
        return None
    families = split.get("families") or {}
    return {"of": "rows", "run": run, "beta": row["b10"],
            "ways": _priced({family: f["share"] for family, f in families.items()}, families), "sources": _sources(split)}


def _held(scope: str, run: str, rows: Sequence[Mapping], requests: float, of: str, plan: Mapping) -> dict:
    """``of``'s plan of a scope as these rows' plan: how its proving is priced (`_priced`), once it counts the rows'
    requests and bounds what they do at its budget. A plan records no source of its prices."""

    count, theirs = _requests(plan), _between(rows, "bound_bits", plan["beta"], log=True)
    if count is None or not _same(count, requests):
        raise ValueError(f"{of}'s plan of the {scope} is priced over {count or 0:.6g} requests and {run}'s rows over "
                         f"{requests:.6g}: it is not these rows' plan")
    if not math.isclose(plan["bound_bits"], theirs, rel_tol=PLAN_TOLERANCE):
        raise ValueError(f"{of}'s plan of the {scope} bounds {plan['bound_bits'] / GB:,.1f} GB at β = {plan['beta']:.4%}, "
                         f"where {run}'s rows give {theirs / GB:,.1f} GB: it is not these rows' plan")
    shares: dict[str, float] = {}
    for cell in (*plan["cells"], *plan.get("left_out", ())):
        family = cell["block"].split("{")[0]
        shares[family] = shares.get(family, 0.0) + cell["share"]
    near = min(rows, key=lambda r: abs(r["b10"] - plan["beta"]))
    return {"of": "plan", "run": of, "beta": plan["beta"],
            "ways": _priced(shares, (near.get("split") or {}).get("families") or {}), "sources": None}


def at_share(scope: str, run: str, rows: Sequence[Mapping], share: float,
             plan: tuple[str, Mapping] | None = None) -> dict:
    """One scope's rows read off at the audit budget where the total is ``share`` of the compute: the budget, whether
    the pinned budgets bracket it, the bound there at λ = 1 and λ times it, each part of the total, and what the
    rows hold fixed (λ, δ, the most bits a request's output holds, the scope's GPU-seconds, the requests they are
    priced over and the weights). How its proving is priced is ``plan``'s account, a run and the scope's plan it
    holds (`_held`); without one the rows' own plan's at the pinned budget nearest (`_own`), or None."""

    rows = sorted(rows, key=lambda r: r["b10"])
    if len(kinds := sorted({str(r[VARIANT]) for r in rows if r.get(VARIANT) is not None})) > 1:
        raise ValueError(f"{run}'s {scope} rows are {len(kinds)} variants of the row ({', '.join(kinds)}), each on its "
                         "own premises: a run of record holds one, or --variant names the one meant")
    counts = [_requests(r) for r in rows]
    if None in counts:
        raise ValueError(f"{run}'s {scope} rows don't say what requests they are priced over (`replay_sample / q`)")
    fixed = {"lambda": {r["weights_bits"] / r["target_bits"] for r in rows}, "delta": {r["delta_seen"] for r in rows},
             "output_bits": {r["S"] for r in rows}, "gpu_seconds": {r["gpu_seconds"] for r in rows},
             "weights_bits": {r["weights_bits"] for r in rows}}
    if len({r["b10"] for r in rows}) < 2 or len({r["b10"] for r in rows}) != len(rows):
        raise ValueError(f"{run}'s {scope} rows pin {len(rows)} audit budgets, {len({r['b10'] for r in rows})} distinct: "
                         "the total's line needs two or more, each once")
    if moved := sorted([k for k, v in fixed.items() if len(v) != 1] + ["requests"] * (not _same(min(counts), max(counts)))):
        raise ValueError(f"{run}'s {scope} rows differ in {', '.join(moved)}, which one plan holds fixed over its "
                         "audit budgets")
    slope, intercept = _fit([(r["b10"], r["total"]) for r in rows])
    beta = (share - intercept) / slope
    if not beta > 0:
        raise ValueError(f"{run}'s {scope} rows spend {intercept:.2%} of the compute before any audit: no audit budget "
                         f"brings the total to {share:.2%}")
    lam = fixed["lambda"].pop()
    bits = _between(rows, "bound_bits", beta, log=True)
    compute, _, period = scope.rpartition(" ")
    return {"scope": scope, "compute": compute, "period": period, "run": run, "step": rows[0]["step"],
            "mix": rows[0].get("mix"), "pinned": [r["b10"] for r in rows], "beta": beta,
            "inside": rows[0]["b10"] <= beta <= rows[-1]["b10"],
            "residual": max(abs(r["total"] - (intercept + slope * r["b10"])) for r in rows),
            **{k: v.pop() for k, v in fixed.items() if k != "lambda"}, "lambda": lam, "requests": counts[0],
            "bound": {"bits": bits, "GB": bits / GB}, "bits": bits * lam, "GB": bits * lam / GB,
            "parts": {p: _between(rows, p, beta) for p in PARTS if all(r.get(p) is not None for r in rows)},
            "priced": _held(scope, run, rows, counts[0], *plan) if plan
            else _own(run, min(rows, key=lambda r: abs(r["b10"] - beta))),
            "binding": _binding(run, min(rows, key=lambda r: abs(r["b10"] - beta)))}


def _binding(run: str, row: Mapping) -> dict | None:
    """A row's binding bound as the planner wrote it (its `verification_phase`'s `binding`): the two-stage first-hit
    finder's bound of one verification phase of the row's design and of its scope of them, each counted in calls and
    in compressions, with the inputs it assumes and what Lean proves of one phase; beside it the row's budget and the
    phase's counts it is at. None for a row written before the planner computed one. The command computes none."""

    phase = row.get("verification_phase") or {}
    if not (bound := phase.get("binding")):
        return None
    return {"run": run, "beta": row["b10"], "scope": row["scope"],
            **{k: phase.get(k) for k in ("name", "seconds", "machines", "sampled", "sessions")}, **bound}


# ---- the stop class's verdicts ---------------------------------------------------------------------------------------


def verdict(x: Any, where: str) -> dict:
    """A verdict's record (`VERDICT_FIELDS`), refused where it lacks a field, by the field's name, or where a field
    holds what that field can't."""

    if not isinstance(x, Mapping):
        raise ValueError(f"{where} is a verdict's record, one JSON object, not {type(x).__name__}")
    if lacks := [f"`{k}`" for k in VERDICT_FIELDS if k not in x]:
        raise ValueError(f"{where} lacks {', '.join(lacks)}: a verdict's record holds "
                         f"{', '.join(f'`{k}`' for k in VERDICT_FIELDS)} (`VERDICT_FIELDS` in "
                         "benchmarks/harness/verity_bench/exfiltration.py)")
    v = x["verifier"]
    if lacks := [f"`verifier.{k}`" for k in VERIFIER_FIELDS
                 if not (isinstance(v, Mapping) and isinstance(v.get(k), str) and v[k])]:
        raise ValueError(f"{where} lacks {', '.join(lacks)}: a verdict's verifier is its module and the commit it ran at")

    def text(k: str) -> bool:
        return isinstance(x[k], str) and bool(x[k])

    for ok, why in ((x["piece"] in PIECES, f"`piece` is one of {', '.join(PIECES)}"),
                    (x["control"] is None or x["control"] in CONTROLS,
                     f"`control` is null on the honest run, else one of {', '.join(CONTROLS)}"),
                    (text("run"), "`run` is the id of the run that produced the verdict"),
                    (x["verdict"] in VERDICT_READS, f"`verdict` is one of {', '.join(VERDICT_READS)}"),
                    (x["proved_by"] is None or text("proved_by"),
                     "`proved_by` is null, or the Lean guarantee whose proof covers the verifier"),
                    (x["proved_by"] is not None or text("open"),
                     "`open` names what the Lean proof of the verifier waits on, while `proved_by` is null")):
        if not ok:
            raise ValueError(f"{where} holds {', '.join(f'`{k}` {x[k]!r}' for k in VERDICT_FIELDS if k != 'verifier')}: "
                             f"{why}")
    return {k: x[k] for k in VERDICT_FIELDS} | {"verifier": {k: v[k] for k in VERIFIER_FIELDS}}


def _what(v: Mapping) -> str:
    return PIECES[v["piece"]] + ("" if v["control"] is None else f", the control with {CONTROLS[v['control']]}")


def stop_verdicts(store: Any) -> tuple[list[dict], list[str], list[dict]] | None:
    """The stop class's verdicts of record (`verdict`), the runs read and the runs left out with why; None where
    nothing in the store carries `run_of_record=VERDICTS`. Every labelled run not superseded is read, each verdict's
    record a line of its `VERDICTS_FILE`, and a verdict on one piece and control that two records hold is refused."""

    labelled = record_runs(store, records(store).get(VERDICTS, []))
    if not labelled:
        return None
    held: dict[tuple, tuple[str, dict]] = {}
    runs, refused = [], []
    for run in sorted({run for run, _ in labelled}):
        at, why = None, None
        if superseded := [str(lab.value) for lab in store.labels(run, key=SUPERSEDED_BY)]:
            why = f"superseded by {', '.join(superseded)}"
        else:
            at, _, why = _run_files(store, run, (VERDICTS_FILE,))
            why = why or (None if at else f"holds no verdicts (`{VERDICTS_FILE}`)")
        if why:
            refused.append({"target": run, "record": VERDICTS, "why": why})
            continue
        for n, line in enumerate((at / VERDICTS_FILE).read_text().splitlines(), 1):
            if not line.strip():
                continue
            where = f"{run}'s `{VERDICTS_FILE}` line {n}"
            try:
                v = verdict(json.loads(line), where)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where} is no JSON object: {e}") from e
            key = (v["piece"], v["control"])
            if key in held:
                other = held[key][0]
                raise ValueError(f"{run} holds two verdicts on {_what(v)}" if other == run else
                                 f"{other} and {run} both hold a verdict on {_what(v)} and neither is superseded: label "
                                 f"the one the other replaces ({_label(SUPERSEDED_BY, 'NEWER')})")
            held[key] = (run, v)
        runs.append(run)
    order = list(PIECES)
    return sorted((v for _, v in held.values()), key=lambda v: (order.index(v["piece"]), v["control"] or "")), runs, refused


def _by(v: Mapping) -> str:
    return f"`{v['verifier']['module']}` at {v['verifier']['commit']}"


def stop_class(verdicts: Sequence[Mapping], runs: Sequence[str]) -> dict:
    """Whether the stop class's ``verdicts`` (`stop_verdicts`) let the sums under the rows whose schedules are 0 take a
    response's output count at 0 on a run (`complete`): each of `PIECES` accepted and complete on its honest run, and
    each of `CONTROLS` refused, on a piece, by the verifier that accepted that piece, and none accepted. Where it is,
    `mark` is what the output count's 0 is marked: `measured` only where every piece's record names the Lean guarantee
    that proves its verifier (`proved_by`), else `RUN_NOT_PROVED`. Where it is not, `lacks` names what is missing, the
    mark is None and the output count keeps its own. `said` is what the print says of the class."""

    honest = {v["piece"]: v for v in verdicts if v["control"] is None}
    lacks = []
    for piece, what in PIECES.items():
        if piece not in honest:
            lacks.append(f"no verdict on {what}")
        elif (v := honest[piece])["verdict"] != "accept":
            lacks.append(f"{what} is `{v['verdict']}` on {v['run']}")
    for control, what in CONTROLS.items():
        tried = [v for v in verdicts if v["control"] == control]
        if not tried:
            lacks.append(f"no negative control with {what}")
        for v in tried:
            accepted = honest.get(v["piece"])
            if v["verdict"] != "refuse":
                lacks.append(f"the control with {what} on {PIECES[v['piece']]} is `{v['verdict']}` on {v['run']}, not "
                             "refused")
            elif accepted is None or accepted["verdict"] != "accept" or accepted["verifier"] != v["verifier"]:
                lacks.append(f"the control with {what} is refused on {v['run']} by {_by(v)}, not by a verifier that "
                             f"accepted the honest run's verdict on {PIECES[v['piece']]}")
    out = {"record": VERDICTS, "runs": list(runs), "complete": not lacks, "mark": None, "lacks": lacks,
           "verdicts": list(verdicts)}
    if lacks:
        return out | {"said": "not complete: " + "; ".join(lacks)}
    unproved = [piece for piece in PIECES if honest[piece]["proved_by"] is None]
    if not unproved:
        return out | {"mark": "measured", "said": "complete (measured: proved by "
                      + _listed([f"`{g}`" for g in dict.fromkeys(honest[p]["proved_by"] for p in PIECES)]) + ")"}
    return out | {"mark": RUN_NOT_PROVED,
                  "said": f"complete (run, not proved: {_listed(list(dict.fromkeys(honest[p]['open'] for p in unproved)))} "
                          f"open; Python verifier for {_listed([PIECES[p] for p in unproved])})"}


# ---- the network certifier's part, and the requests both parts count over --------------------------------------------


def certifier(root: Path, response_ids: int = RESPONSE_IDS, scan_delta: Fraction | None = None) -> dict:
    """The network certifier's term at the datacenter's and the world's load (`leak_bounds.document`) for a class of
    at most ``response_ids`` ids a response, over the requests its calculator counts by default, the sampling plan's,
    with each execution phase's scan at ``scan_delta``. With None the δ is the one at which the row of the document
    that sums the most execution phases carries `PERIOD_ERROR`: that error over the row's execution phases, which the
    calculator's own document counts, no term depending on the scans' δ. The document's `scan_delta` input keeps the
    calculator's value and mark, and its source says who gave the δ and how it was chosen.

    Beside the document: the jitter bound of its term of record and the one its labelled term is at, each as the
    rows' `term` names it (`of_record`, `set_time`), and the term of record at the class the calculator assumes by
    default with its ε, at the same δ (`default_class`), which is `verity bench certifier_term`'s own figure."""

    lb = _load(root, "network_traces", "leak_bounds")

    def read(inp: Mapping, **rows: Any) -> dict:
        doc = lb.document(inp, **rows)
        if (schema := doc["schema"]) != TERM_SCHEMA:
            raise ValueError(f"the network certifier's document is `{schema}` and this command reads `{TERM_SCHEMA}`: its "
                             "term of record, its rows' terms by jitter bound, their ε and their labels may mean "
                             "something else (read the new document in benchmarks/harness/verity_bench/exfiltration.py "
                             "and name its schema there)")
        return doc

    how = "given (--scan-delta)"
    if scan_delta is None:
        most = max(read(lb.inputs(response_ids=response_ids), only=SCOPES)["rows"], key=lambda r: r["phases"])
        scan_delta = PERIOD_ERROR / Fraction(most["phases"])
        how = (f"this command's, in place of the calculator's default: {_p(float(PERIOD_ERROR))}, the error a monitoring "
               f"period's figure is to carry at most, over the {_n(most['phases'])} execution phases of the "
               f"{most['scope']}'s {most['period']}, the row printed that sums the most, so that every row printed "
               "carries that error or less. The network certifier's result of record takes its δ so "
               "(benchmarks/certifier_term/README.md), and no term depends on it")
    term = read(lb.inputs(response_ids=response_ids, scan_delta=scan_delta), only=SCOPES)
    default = read(lb.inputs(scan_delta=scan_delta), only=("world",), periods=("year",))
    head = next(r for r in default["rows"] if r["row"] == "scan")
    set_time, = (str(dj) for dj in lb.JITTERS if dj != lb.OF_RECORD)
    given = term["inputs"]["scan_delta"] | {"source": f"each execution phase's scan's δ, {how}"}
    return term | {"inputs": term["inputs"] | {"scan_delta": given}, "of_record": str(lb.OF_RECORD), "set_time": set_time,
                   "default_class": {"response_ids": default["inputs"]["response_ids"]["value"], "eps": head["eps"],
                                     "value": default["value"] if head["eps"] < 1 else None, "unit": default["unit"]}}


# ---- the share of compute a served run measures ----------------------------------------------------------------------


def _gpu_s(entries: Any, field: str, run: str) -> tuple[float | None, list[str]]:
    """The GPU-seconds a record's field holds, a list of what held GPUs for it (one entry a session or a step, each
    `{"seconds": s, "gpus": g}`) or one such entry, and the keys an entry lacks. None and the gap where it holds none.
    An entry on 0 GPUs is a step that started no process on a card: it is read, and adds no GPU-seconds."""

    if not entries:
        return None, [field]
    lacks, total = [], 0.0
    for e in [entries] if isinstance(entries, Mapping) else entries:
        seconds, gpus = e.get("seconds"), e.get("gpus")
        lacks += [f"{field}.{k}" for k, v in (("seconds", seconds), ("gpus", gpus)) if v is None]
        if None not in (seconds, gpus):
            if not (seconds >= 0 and isinstance(gpus, int) and gpus >= 0):
                raise ValueError(f"{run}'s record holds {field} at {seconds!r} seconds on {gpus!r} GPUs: an entry is "
                                 "seconds of 0 or more on a whole count of GPUs, 0 or more")
            total += seconds * gpus
    return (None if lacks else total), sorted(set(lacks))


def served_share(record: Mapping[str, Any] | None) -> dict:
    """The audit's share of compute on a served two-stage run, from its ``record`` (the reader of a served run's
    record gives it): its GPU-seconds over the window's serving GPU-seconds. Each part of the audit (`SERVED_PARTS`)
    is read from the record and marked measured, and where the record lacks it, its gap is named and the share is not
    claimed. None: no record, and the share is `NO_RUN`.

    The record is one object: ``run``, the served run's id; ``prove``, ``staging``, ``replay`` and ``interior``, each
    a list of what held GPUs for that part, one entry a session or a step, each ``{"seconds": s, "gpus": g}`` (the
    seconds it held them and how many it held); ``serving``, one such entry, the window's load: the seconds it was
    served over and the GPUs its engine held; and for the serving charge, ``stock``, one such entry for stock vLLM
    serving the same load at the same launch settings save Verity's changes, or else ``serving_cost``, `verity bench
    serve_tax`'s cost of the served arm over stock on its own loads, as a fraction (stock's output tokens a second over
    the arm's, less one), with its ``serving_cost_source`` (the run and load it is of). A field absent or None is one
    the record lacks; so is an entry's key. The serving charge is the served GPU-seconds less stock's, measured; or
    from the serving cost c, the served GPU-seconds times c / (1 + c), modeled, since its loads are not the window's."""

    if record is None:
        return {"run": None, "value": None, "mark": None, "why": NO_RUN, "serving_gpu_s": None, "parts": {},
                "missing": []}
    if not isinstance(record, Mapping):
        raise ValueError(f"a served run's record is one JSON object, not {type(record).__name__}")
    run = str(record.get("run") or "the served run")
    serving, missing = _gpu_s(record.get("serving"), "serving", run)
    gaps = [{"field": "serving", "lacks": missing, "why": SERVING_GAP}] if missing else []
    parts: dict[str, dict] = {}
    for name, (what, why) in SERVED_PARTS.items():
        if name == "serving_charge":
            stock, lacks = _gpu_s(record.get("stock"), "stock", run)
            cost = record.get("serving_cost")
            if record.get("stock") is not None or cost is None:
                gpu_s, mark = (None if None in (stock, serving) else serving - stock), "measured"
                source = "the served GPU-seconds less those of stock vLLM serving the same load (`stock`)"
            else:
                if not (isinstance(cost, (int, float)) and cost > -1):
                    raise ValueError(f"{run}'s record holds a serving cost of {cost!r}: a fraction over -1")
                lacks, mark = [], "modeled"
                gpu_s = None if serving is None else serving * cost / (1 + cost)
                source = (f"the serving cost of the served arm over stock, {_pct(cost)} (`serving_cost`, "
                          f"{record.get('serving_cost_source') or 'its run and load unnamed'}), on its own loads, not "
                          "the window's: the served GPU-seconds times c / (1 + c)")
        else:
            (gpu_s, lacks), mark, source = _gpu_s(record.get(name), name, run), "measured", f"`{name}`"
        if lacks:
            gaps.append({"field": name, "lacks": lacks, "why": why})
        parts[name] = {"what": what, "gpu_s": gpu_s, "mark": "missing" if gpu_s is None else mark, "source": source,
                       "share": None if None in (gpu_s, serving) or not serving else gpu_s / serving}
    if serving is not None and not serving > 0:
        raise ValueError(f"{run}'s record holds the window's serving at {serving!r} GPU-seconds: the share is of more "
                         "than none")
    claimed = not gaps
    return {"run": run, "value": sum(p["share"] for p in parts.values()) if claimed else None,
            "mark": (("modeled" if any(p["mark"] == "modeled" for p in parts.values()) else "measured")
                     if claimed else None),
            "why": None if claimed else "not measured: " + "; ".join(
                f"{run}'s record lacks {', '.join(f'`{k}`' for k in g['lacks'])} ({g['why']})" for g in gaps),
            "serving_gpu_s": serving, "parts": parts, "missing": gaps}


# ---- the document ----------------------------------------------------------------------------------------------------


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9)


def _inputs(points: Mapping[str, dict], runs: Sequence[str], term: Mapping, count: Mapping, share: float,
            network_check: float, row: str, served: Mapping, stop: Mapping | None = None) -> list[dict]:
    """Every input with its mark and source: the compute audit's from its rows of record, the network certifier's
    as its calculator marks them, and those both read, among them the request the calculator counts and what it is.
    ``count`` is the world year's output count (`build`'s `output_count`), which the sums it names take at 0.
    ``served`` is the share a served run measures (`served_share`): with a run's record, each of its fields with its
    mark, a field the record lacks marked missing with its gap. ``stop`` is the stop class's verdicts of record
    (`stop_class`): where it is complete the output count's 0 carries its mark and names each verdict, its run and
    its verifier; otherwise, and with none, the output count is as without it."""

    head = points["world year"]
    budgets = sorted({b for p in points.values() for b in p["pinned"]})
    where = f"`{RECORD}` ({', '.join(runs)})"

    def one(part: str, name: str, value: Any, unit: str, mark: str, source: str) -> dict:
        return {"part": part, "input": name, "value": value, "unit": unit, "mark": mark, "source": source}

    audit, both, net = "compute audit", "both", "network certifier"
    out = [
        one(audit, "planner_rows", sum(len(p["pinned"]) for p in points.values()), "row", "modeled",
            f"{where}: the planner's row `{row}` at its step `{head['step']}`, each of {len(points)} scopes at "
            f"audit budgets pinned from {min(budgets):.3g} to {max(budgets):.3g}"),
        one(audit, "compute_share", share, "", "assumed", "the share of total compute the figure is read at (--share); "
            + ("the share a served run measures is " + served["why"] if served["value"] is None else
               f"the share {served['run']} measures is beside it (`served_compute_share`)")),
        one(audit, "delta", head["delta"], "", "assumed",
            f"{where} `delta_seen`: the audit's δ over the monitoring period, a stated parameter"),
        one(audit, "lambda", head["lambda"], "", "assumed",
            f"{where} `weights_bits / target_bits`: the row plans for the weights / λ and its part is λ times its "
            "bound, a premise no run measures"),
        one(audit, "output_bits", head["output_bits"], "bit/request", "assumed",
            f"{where} `S`: the most bits a request's output holds, which no check certifies"),
        *(one(audit, f"proving priced {way}", p["share"], "of proving time", "measured" if way == "measured" else "modeled",
              f"`{RECORD}` ({head['priced']['run']}) {_plan(head['priced'])}, "
              f"block {'families' if len(p['families']) > 1 else 'family'} {', '.join(list(p['families'])[:2])}"
              f"{f' and {len(p["families"]) - 2} smaller' if len(p['families']) > 2 else ''}; how each is priced is the "
              f"rows' account (`split.families`)") for way, p in (head["priced"] or {"ways": {}})["ways"].items()),
        one(audit, "network_check_share", network_check, "", "assumed",
            "no run: the network check's own proving as a share of total compute, which the planner's rows don't "
            "count (--network-check-share)"),
        one(audit, "advice", 0, "bit", "assumed", "not counted: no input here prices the advice"),
    ]
    charged = f"{'' if count['GB'] is None else _n(count['bits_a_request']) + ' bits a request '}as its schedules"
    if stop and stop["complete"]:
        out.append(one(net, "output_count", 0, "bit/request", stop["mark"],
                       f"taken at 0 in the sums under {_names(count['at_zero_in'])}: the stop class's verdicts of record, "
                       f"`{VERDICTS}` ({', '.join(stop['runs'])}), are {stop['said']}: "
                       + "; ".join(f"{_what(v)}, `{v['verdict']}` on {v['run']} by {_by(v)}, "
                                   + (f"proved by `{v['proved_by']}`" if v["proved_by"] else f"not proved, {v['open']} open")
                                   for v in stop["verdicts"])
                       + ". Without them the server chooses a response's output count, which the network certifier's "
                       f"row `{OUTPUT_COUNT}` charges {charged}"))
    else:
        out.append(one(net, "output_count", 0, "bit/request", "assumed",
                       f"not counted in the sums under {_names(count['at_zero_in'])}: until a served audit has sampled a "
                       "stop block the server chooses a response's output count, which the network certifier's row "
                       f"`{OUTPUT_COUNT}` charges {charged}"))
    if served["run"] is not None:
        gaps = {g["field"]: g for g in served["missing"]}
        whose = f"{served['run']}'s record"
        out += [one(audit, "served_serving_gpu_s", served["serving_gpu_s"], "GPU-second",
                    "missing" if "serving" in gaps else "measured",
                    f"{whose} `serving`: the window's serving, which the share is of"
                    + (f"; it lacks {', '.join(gaps['serving']['lacks'])}: {SERVING_GAP}" if "serving" in gaps else "")),
                *(one(audit, f"served_{name}_gpu_s", p["gpu_s"], "GPU-second", p["mark"],
                      f"{whose}: {p['what']}, {p['source']}"
                      + (f"; it lacks {', '.join(gaps[name]['lacks'])}: {gaps[name]['why']}" if name in gaps else ""))
                  for name, p in served["parts"].items()),
                one(audit, "served_compute_share", served["value"], "", served["mark"] or "missing",
                    f"{whose}: the audit's GPU-seconds over the window's serving GPU-seconds"
                    + ("" if served["value"] is not None else f", {served['why']}"))]
    for name, i in term["inputs"].items():
        out.append(one(both if name in SHARED else net, name, i["value"], i["unit"], i["mark"], i["source"]))
    for i in (head.get("binding") or {}).get("assumed", ()):
        out.append(one(SECURITY, i["input"], i["value"], i["unit"], "assumed",
                       f"`{RECORD}` ({head['binding']['run']}) `verification_phase.binding`: {i['source']}"))
    if off := sorted(i["input"] for i in out if i["mark"] not in MARKS or not i["source"]):
        raise ValueError(f"an input is marked one of {', '.join(MARKS)}, with its source: {', '.join(off)}")
    return out


def at_assumed_prices(store: Any, head: Mapping, share: float, step: str, variant: str | None) -> dict:
    """Row `ASSUMED_ROW` over the world's year at ``share``, beside the figure ``head`` and under `ASSUMED_LABEL`: its
    run, budget, δ and amount, and what prices its proving (`_sources`); or, with no amount, why it is not read: no run
    of record holds it, or its rows cover other compute or weights than the figure's."""

    out = {"row": ASSUMED_ROW, "label": ASSUMED_LABEL, "GB": None, "why": None}
    try:
        scopes, plans, _, _ = rows_of_record(store, row=ASSUMED_ROW, step=step, variant=variant)
        if "world year" not in scopes:
            return out | {"why": f"no run of record holds its world year's rows (they hold {', '.join(sorted(scopes))})"}
        p = at_share("world year", *scopes["world year"], share, plans.get("world year"))
    except ValueError as e:
        return out | {"why": str(e)}
    if moved := [k for k in ("gpu_seconds", "requests", "weights_bits") if not _same(p[k], head[k])]:
        return out | {"why": f"its rows differ from the figure's in {', '.join(moved)}"}
    return out | {k: p[k] for k in ("run", "beta", "inside", "delta", "bits", "GB")} | {
        "sources": p["priced"] and p["priced"]["sources"]}


def build(*, share: float = SHARE, network_check: float = NETWORK_CHECK_SHARE, row: str = ROW, step: str = STEP,
          variant: str | None = None, response_ids: int = RESPONSE_IDS, scan_delta: Fraction | None = None,
          served: Mapping[str, Any] | None = None, store: Any = None, root: Path | None = None,
          local_only: bool = False) -> dict:
    """The two parts from the store's rows of record and the tree: the inputs with their marks, the compute audit's
    part at ``share`` of total compute for each scope of record, the network certifier's rows at the world's year
    for a class of at most ``response_ids`` ids a response with each execution phase's scan at ``scan_delta`` (None:
    the δ `certifier` takes from `PERIOD_ERROR`), and the two side by side for each scope and monitoring period,
    summed under each of `TOGETHER`. It refuses rows of record priced over other requests than the network
    certifier's part counts.

    Every figure carries the error summed over everything it covers, and one whose error is 1 or more carries no
    amount (`NO_BOUND`). A scope's monitoring period holds the network certifier's ε (`eps`), the one its rows over
    that scope and period carry, their scans' δ summed over every link's execution phases; the compute audit's δ
    over the period (its `compute_audit`'s `delta`); and the sum of the two (`error`), what the two together fail
    with at most, None for a scope no run of record holds. Where `eps` is 1 or more the network certifier's figures
    of that scope and period, the output count among them, are None, and where `error` is, the sums are. A row of the
    network certifier's own over the world's year whose ε is 1 or more is held without its `AMOUNTS`.

    A sum is at the network certifier's jitter bound of record. Its `label` is its row's own (None on most): where
    set, every figure of the row is printed under it. Its `with_set_time_card` is the same sum at the other jitter
    bound, with the label the network certifier's document gives a figure there. A scope's `output_count` is what a
    response's output count is charged while the server chooses it, the schedules of the row `OUTPUT_COUNT`, with
    the rows whose sums take it at 0 (`at_zero_in`: those whose schedules are 0).

    Beside a ``row`` other than `ASSUMED_ROW`, `at_assumed_prices` holds that row's world year at ``share``
    (`at_assumed_prices()`), which is never the figure; as ``row`` itself, the figure's `label` is `ASSUMED_LABEL`.

    ``served`` is a served two-stage run's record (`served_share`), from which the document's `compute_share` is the
    share of compute that run measures, with each of its parts, and where it is claimed the compute audit's part over
    the world's year at that share (`GB`, with its β), beside the figure at ``share``, which stays given and assumed
    (`given`). Without a record, or where it lacks a field, the share is not claimed, and `why` says so.

    Where the store holds the stop class's verdicts (`stop_verdicts`), `stop_class` is their account (`stop_class`),
    and where it is complete the output count's input carries its mark; the runs left out are among `refused`. With
    none the document holds no `stop_class`, and is as it was without one."""

    if not (0 < share < 1 and 0 <= network_check < share):
        raise ValueError(f"the share of total compute is in (0, 1) and the network check's own below it, not {share!r} "
                         f"and {network_check!r}")
    measured = served_share(served)
    root = root or jobs.repository_root()
    completion = jobs.StoreCompletion(store, root=root, local_only=local_only)
    if not local_only:
        completion.pull_labels()
    scopes, plans, runs, refused = rows_of_record(completion.store, row=row, step=step, variant=variant)
    stop = None
    if (verdicts := stop_verdicts(completion.store)) is not None:
        stop = stop_class(*verdicts[:2])
        refused = [*refused, *verdicts[2]]
    points = {scope: at_share(scope, run, its, share, plans.get(scope)) for scope, (run, its) in scopes.items()}
    if "world year" not in points:
        held = sorted({run for run, _ in scopes.values()})
        raise ValueError(f"no run of record holds the world year's rows ({', '.join(held)} hold{'s' * (len(held) == 1)} "
                         f"{', '.join(sorted(points))}): label the run that does ({_label()})")
    head = points["world year"]
    less = at_share("world year", *scopes["world year"], share - network_check)
    assumed = None if row == ASSUMED_ROW else at_assumed_prices(completion.store, head, share, step, variant)
    measured["given"] = {"share": share, "mark": "assumed", "GB": head["GB"]}
    measured.update(GB=None, beta=None, inside=None, no_GB=None)
    if measured["value"] is not None:
        try:
            there = at_share("world year", *scopes["world year"], measured["value"])
            measured.update(GB=there["GB"], beta=there["beta"], inside=there["inside"])
        except ValueError as e:
            measured["no_GB"] = str(e)

    term = certifier(root, response_ids, scan_delta)
    of, set_time, card = term["of_record"], term["set_time"], term["with_set_time_card"]["label"]
    weights = term["inputs"]["weights"]["value"]
    if not _same(head["weights_bits"], weights):
        raise ValueError(f"the two parts are shares of different weights: the rows of record hold {head['weights_bits']:,} "
                         f"bits and the network certifier's calculator {weights:,}")
    seen = {(r["scope"], r["period"]): r for r in term["rows"]}
    order = {f"{compute} {period}": n for n, (compute, period) in enumerate(seen)}
    periods = []
    for (compute, period), at in seen.items():
        scope = f"{compute} {period}"
        point = points.get(scope)
        for name, mine, theirs in (("GPU-seconds", point and point["gpu_seconds"], at["gpu_seconds"]),
                                   ("requests", point and point["requests"], at["requests"])):
            if mine is not None and not _same(mine, theirs):
                raise ValueError(f"the two parts cover different compute over the {scope}: {mine:.6g} {name} on the compute "
                                 f"audit's side and {theirs:.6g} on the network certifier's")
        rows = {r["row"]: r for r in term["rows"] if (r["scope"], r["period"]) == (compute, period)}
        if len(several := sorted({r["eps"] for r in rows.values()})) != 1:
            raise ValueError(f"the network certifier's rows over the {scope} carry {len(several)} different ε "
                             f"({', '.join(map(_n, several))}), and this command prints one a scope and monitoring "
                             "period, its scans' δ summed over every link's execution phases (read each row's own in "
                             "benchmarks/harness/verity_bench/exfiltration.py)")
        eps = at["eps"]
        audit = point and {"GB": point["GB"], "bits": point["bits"], "beta": point["beta"], "inside": point["inside"],
                           "of_weights": point["bits"] / weights, "bits_a_request": point["bits"] / at["requests"],
                           "delta": point["delta"], "run": point["run"]}
        error = None if audit is None else eps + audit["delta"]
        together = {}
        for name in TOGETHER:
            r = rows[name]
            entries = r["entries"] and r["entries"]["bits"]
            mine = None if audit is None else audit["bits"] if entries is None else min(entries, audit["bits"])

            def summed(dj: str) -> dict:
                return {"certifier_GB": r["term"][dj]["GB"] if eps < 1 else None,
                        "GB": (r["term"][dj]["bits"] + mine) / GB if mine is not None and error < 1 else None}

            together[name] = {**summed(of), "label": r["with_set_time_card"],
                              "with_set_time_card": {**summed(set_time), "label": card}}
        count = rows[OUTPUT_COUNT]["schedules"] if eps < 1 else {"GB": None, "bits": None}
        periods.append({"compute": compute, "period": period, "gpu_seconds": at["gpu_seconds"], "requests": at["requests"],
                        "phases": at["phases"], "eps": eps, "error": error, "compute_audit": audit,
                        "output_count": {"row": OUTPUT_COUNT, "GB": count["GB"], "bits": count["bits"],
                                         "bits_a_request": count["bits"] / at["requests"]
                                         if eps < 1 and at["requests"] else None,
                                         "mark": "assumed",
                                         "at_zero_in": [name for name in TOGETHER if not rows[name]["schedules"]["bits"]]},
                        "together": together})
    year = next(p for p in periods if (p["compute"], p["period"]) == ("world", "year"))

    return {"schema": SCHEMA, "model": MODEL,
            "name": f"{MODEL}'s exfiltration capacity over the world's year at {_pct(share)} of total compute: the "
                    "compute audit's part",
            "value": head["GB"], "unit": "GB", "status": "modeled", "share": share, "row": row, "variant": variant,
            "label": ASSUMED_LABEL if row == ASSUMED_ROW else None, "at_assumed_prices": assumed, "record": RECORD,
            "runs": runs, "refused": refused, "weights": {"bits": weights, "GB": weights / GB, "label": term["weights"]["label"]},
            "inputs": _inputs(points, runs, term, year["output_count"], share, network_check, row, measured, stop),
            "compute_share": measured, **({} if stop is None else {"stop_class": stop}),
            "compute_audit": sorted(points.values(), key=lambda p: (p["scope"] not in order, order.get(p["scope"], 0))),
            "network_check": {"share": network_check, "beta": less["beta"], "inside": less["inside"], "GB": less["GB"]},
            "certifier": {"schema": term["schema"], "request_count": term["request_count"],
                          "response_ids": term["inputs"]["response_ids"]["value"],
                          "scan_delta": term["inputs"]["scan_delta"], "of_record": of, "set_time": set_time,
                          "with_set_time_card": {"label": card}, "default_class": term["default_class"],
                          "legend": term["legend"], "entries_rule": term["entries_rule"], "phase": term["phase"],
                          "rows": [r if r["eps"] < 1 else {k: v for k, v in r.items() if k not in AMOUNTS}
                                   for r in term["rows"] if (r["scope"], r["period"]) == ("world", "year")]},
            "periods": periods}


# ---- rendering -------------------------------------------------------------------------------------------------------


def _n(x: Any, none: str = "none") -> str:
    """A number to four significant figures, text as itself, and ``none`` for None."""

    if x is None:
        return none
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if abs(x) >= 1e9 or abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def _p(x: float) -> str:
    """A probability, with its power of two where it is under 1."""

    return f"{_n(x)} (2^{math.log2(x):.4g})" if 0 < x < 1 else _n(x)


def _amount(x: Any, error: float | None) -> str:
    """A figure's amount as a table prints it: `NO_BOUND` where its error, summed over everything it covers, is 1 or
    more."""

    return NO_BOUND if error is not None and error >= 1 else _n(x)


def _pct(x: float | None) -> str:
    return "none" if x is None else f"{100 * x:.4g}%"


def _part(x: float | None) -> str:
    return "none" if x is None else f"{100 * x:.3f}%"


def _table(head: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head), *("| " + " | ".join(r) + " |" for r in body)]


def _plan(priced: Mapping) -> str:
    """Which plan an account of the proving is: the rows' own at a pinned budget, or a plan of record at its own."""

    if priced["of"] == "rows":
        return f"`split`: the row's own plan at the pinned budget nearest, β = {_part(priced['beta'])}"
    return f"`{PLAN_FILE}`: the row's plan at β = {_part(priced['beta'])}"


def _how(priced: Mapping | None) -> str:
    """How a scope's proving time is priced, by its plan: each way's share and the largest families; or that no plan
    says."""

    if priced is None:
        return ("how the row's proving time is priced is not read, and none of it is marked measured: the rows' `split` "
                "is not their own plan (it doesn't say the row's `q` and `replay_sample`: it is written before the row's "
                f"levers multiply its blocks), and no run of record holds the row's plan (`{PLAN_FILE}`)")
    largest = sorted(((s, family, way) for way, p in priced["ways"].items() for family, s in p["families"].items()),
                     reverse=True)[:3]
    plan = "its own plan at the pinned budget nearest" if priced["of"] == "rows" else "its plan"
    return (f"of the row's proving time, by {plan}, β = {_part(priced['beta'])} ({priced['run']}), "
            + "the share in block families priced each way is "
            + ", ".join(f"{_pct(p['share'])} {way}" for way, p in priced["ways"].items())
            + ": the largest families are " + ", ".join(f"{family} ({_pct(s)}, {way})" for s, family, way in largest))


def _by_source(sources: Mapping | None) -> str:
    """The share of a row's proving each source prices and the share at the planner's assumed default (`_sources`),
    each `NOT_RECORDED` where the row's plan doesn't say."""

    default = None if sources is None else sources["default"]
    by = sources["by"] if sources else dict.fromkeys(SOURCES)
    return (", ".join(f"{s} {NOT_RECORDED if v is None else _pct(v)}" for s, v in by.items())
            + f"; at the planner's assumed default (`{DEFAULT_BASIS}`) {NOT_RECORDED if default is None else _pct(default)}")


def _listed(items: Sequence[str]) -> str:
    """Items in a sentence."""

    return " and ".join(items) if len(items) < 3 else ", ".join(items[:-1]) + f" and {items[-1]}"


def _names(names: Sequence[str]) -> str:
    """Rows' names in a sentence."""

    return _listed([f"`{name}`" for name in names])


def _labelled(year: Mapping, key: str, set_time: str) -> list[str]:
    """The world year's figures (``key`` of each of its sums) that are printed only under a label, one clause a
    label: every row that carries its own, and every other row at the jitter bound ``set_time``."""

    own: dict[str, list[str]] = {}
    other: dict[str, list[str]] = {}
    for name, what in TOGETHER.items():
        t = year["together"][name]
        if t["label"]:
            own.setdefault(t["label"], []).append(f"{_n(t[key])} GB {what}")
        else:
            other.setdefault(t["with_set_time_card"]["label"], []).append(f"{_n(t['with_set_time_card'][key])} GB {what}")
    return [f"{label}: " + "; and ".join(filter(None, (
                "; ".join(own.get(label, ())),
                label in other and f"at D_j = {set_time}, " + "; ".join(other[label]))))
            for label in dict.fromkeys((*own, *other))]


def _lean(lean: Mapping) -> str:
    """What Lean proves at numbers of one verification phase's conflict and first-hit part, as a binding bound says
    it: the instance the phase is within and its bits, or `NOT_PROVED` with the nearest instance and each numeral
    the phase is past."""

    if lean["theorem"]:
        return (f"`{LEAN_NUMBERS}.{lean['theorem']}` holds the part of it that is linear in the prover's calls to at "
                f"most 2^-{lean['bits']}, beside each session's knowledge error and the rewinding part")
    past = "; ".join(f"{NUMERALS.get(name, name)} at most {v['instance']:,} there and {v['design']:,} here"
                     for name, v in lean["past"].items())
    return (f"{NOT_PROVED}: the nearest instance, `{LEAN_NUMBERS}.{lean['nearest']}`, at most 2^-{lean['bits']}, has "
            f"{past}")


def _security(bound: Mapping | None) -> list[str]:
    """The security level's lines: a row's binding bound (`_binding`) over one verification phase and over its scope
    of them, each counted in calls and in compressions, what it assumes, and what Lean proves of each; or that the
    rows hold none."""

    if bound is None:
        return [f"- **the {SECURITY}**: not read. The rows of record hold no binding bound (`verification_phase.binding`): "
                f"the planner wrote them before it computed one. Run it again and label its run ({_label()})."]
    design, (calls, compressions) = bound["design"], (bound["counted"][c] for c in ("calls", "compressions"))
    scope = bound["scope"].replace(" ", "-")
    return [
        f"- **the {SECURITY}** (modeled): the probability that an accepted run opens a value off the committed trace, by "
        f"the bound Lean states of a two-stage sample counted by classes of gates (`{LEAN_BOUND}`), every run at "
        f"{design['calls_a_run']} calls of SHA-512, at the row's own design at the pinned budget nearest, β = {_part(bound['beta'])} ({bound['run']}). "
        f"**Over one verification phase: at most {calls['phase']['bound']}** counted in calls, as the Lean counts, and "
        f"{compressions['phase']['bound']} in compressions. **Over a {scope} of phases**, {_n(design['phases'])} of "
        f"them: **at most {calls['phases']['bound']}** in calls and {compressions['phases']['bound']} in compressions. "
        f"A verification phase is a {bound['name']}'s {_n(bound['seconds'])} s on {_n(bound['machines'])} machines: its "
        f"first stage samples {_n(bound['sampled'])} replay blocks, and its second {_n(bound['proof_blocks'])} of their "
        f"proof blocks in {_n(design['sessions'])} sessions, the largest of {_n(bound['tables'])} "
        f"table{'s' * (bound['tables'] != 1)}.",
        f"- the {SECURITY} assumes: "
        + "; ".join(i["input"] if i["value"] is None else f"{_n(i['value'])} {i['input']}" for i in bound["assumed"])
        + ". It counts each session's knowledge error and the run's own opening checks as 0.",
        f"- what Lean proves of the {SECURITY}: over one verification phase counted in calls, {_lean(calls['lean'])}. "
        f"Counted in compressions, {_lean(compressions['lean'])}. Over a {scope} of phases, {NOT_PROVED}: no theorem "
        "sums a monitoring period's verification phases."]


def _share(cs: Mapping) -> tuple[list[str], list[str]]:
    """The share of compute a served run measures (`build`'s `compute_share`), beside the given share every other
    figure is read at: a line, and with a run's record a table of its parts."""

    given = f"the given {_pct(cs['given']['share'])} of total compute ({cs['given']['mark']})"
    if cs["value"] is None:
        lines = [f"**The audit's share of compute: {cs['why']}.** Every figure below is read at {given}.", ""]
    else:
        parts = ", ".join(f"{name.replace('_', ' ')} {_part(p['share'])}" for name, p in cs["parts"].items())
        there = (f"its part has no figure at that share: {cs['no_GB']}" if cs["GB"] is None else
                 f"at that share, β = {_part(cs['beta'])}{'' if cs['inside'] else ' (extrapolated)'}, the compute "
                 f"audit's part over the world's year is {_n(cs['GB'])} GB")
        lines = [f"**The audit's share of compute: {_part(cs['value'])} ({cs['mark']})**, on {cs['run']}: the audit's "
                 f"GPU-seconds over the window's serving, {_n(cs['serving_gpu_s'])} GPU-seconds, in parts {parts}. "
                 f"{there[0].upper()}{there[1:]}, beside {_n(cs['given']['GB'])} GB at {given}, which every figure "
                 "below is read at.", ""]
    if cs["run"] is None:
        return lines, []
    gaps = {g["field"]: g for g in cs["missing"]}

    def source(name: str, said: str) -> str:
        return said + (f"; lacks {', '.join(f'`{k}`' for k in gaps[name]['lacks'])}: {gaps[name]['why']}"
                       if name in gaps else "")

    return lines, [f"## The audit's share of compute on {cs['run']}", "",
            "Each part of the audit's GPU time on the served run, and the window's serving it is a share of, read from "
            "the run's record. A part the record lacks is marked missing with its gap, and the share is not claimed "
            "without it.", "",
            *_table(("part", "what", "GPU-seconds", "share of the window's serving", "mark", "source"),
                    [("serving", "the window's serving", _n(cs["serving_gpu_s"]), "",
                      "missing" if "serving" in gaps else "measured", source("serving", "`serving`")),
                     *((name, p["what"], _n(p["gpu_s"]), _part(p["share"]), p["mark"], source(name, p["source"]))
                       for name, p in cs["parts"].items())]), ""]


def render(doc: Mapping[str, Any]) -> str:
    """``doc`` (`build`) as text: the figure and what it is, the inputs with their marks, each part, and the two side
    by side for every scope and monitoring period. The network certifier's figures of record come first; a figure
    that is under a label is printed in a line, a row or a column that carries that label, as the document gives
    it. Every line of the network certifier's and of the two together carries its errors and the δ of the scans they
    sum, and a figure whose error is 1 or more is printed as `NO_BOUND`, with no amount."""

    share, net, w, cert = doc["share"], doc["network_check"], doc["weights"], doc["certifier"]
    head = next(p for p in doc["compute_audit"] if p["scope"] == "world year")
    planned = [p for p in doc["compute_audit"] if p["priced"]]
    year = next(p for p in doc["periods"] if (p["compute"], p["period"]) == ("world", "year"))
    of, set_time, card, default = cert["of_record"], cert["set_time"], cert["with_set_time_card"]["label"], cert["default_class"]
    count, sums, delta, eps, error = year["output_count"], year["together"], cert["scan_delta"], year["eps"], year["error"]
    record = [name for name in TOGETHER if not sums[name]["label"]]
    labels = {name: sums[name]["label"] for name in TOGETHER if sums[name]["label"]}
    marks = {m: sum(i["mark"] == m for i in doc["inputs"]) for m in MARKS}
    stop = doc.get("stop_class")
    measured, parts = _share(doc["compute_share"])
    outside = "" if head["inside"] else ", outside the pinned budgets: extrapolated"
    what = ", the plan of record's row, with every modeled lever at its measured value" if doc["row"] == ROW else ""
    what += f", in its variant `{doc['variant']}`" if doc["variant"] else ""
    label = f", {doc['label']}" if doc["label"] else ""
    sources = head["priced"] and head["priced"]["sources"]
    plan = f" (its own plan at β = {_part(head['priced']['beta'])}, {head['priced']['run']})" if sources else ""
    a = doc["at_assumed_prices"]
    assumed = [] if a is None else [
        f"- row `{a['row']}`, {a['label']}: "
        + (f"not read: {a['why']}." if a["GB"] is None else
           f"{_n(a['GB'])} GB, at β = {_part(a['beta'])}{'' if a['inside'] else ', extrapolated'} ({a['run']}), failing "
           f"with at most δ = {_n(a['delta'])} over the monitoring period. It is not the figure. What prices its "
           f"proving time: {_by_source(a['sources'])}.")]
    on = f"modeled, on an assumed class of at most {_n(cert['response_ids'])} ids a response"
    if eps < 1:
        scan = next(r for r in cert["rows"] if r["row"] == "scan")
        certifier = [
            f"- **the network certifier's part: {_n(sums['scan']['certifier_GB'])} GB** ({on}) {TOGETHER['scan']}, "
            f"{_n(scan['term'][of]['bits_a_request'])} bits a request, and "
            + ", ".join(f"{_n(sums[name]['certifier_GB'])} GB {TOGETHER[name]}" for name in record if name != "scan")
            + f", each at D_j = {of}, its term of record, and each failing with at most ε = {_p(eps)} over the world's "
            f"year, at a scan's δ of {_p(delta['value'])}."
            + ("" if default["response_ids"] == cert["response_ids"] else
               f" On the class of at most {_n(default['response_ids'])} ids a response that `verity bench certifier_term` "
               "assumes by default, the scan's is "
               + (f"{_n(default['value'])} {default['unit']}." if default["eps"] < 1 else
                  f"{NO_BOUND}, at an ε of {_n(default['eps'])}.")),
            *(f"- the network certifier's part {clause}; ε = {_p(eps)}."
              for clause in _labelled(year, "certifier_GB", set_time))]
    else:
        certifier = [f"- **the network certifier's part: {NO_BOUND}** ({on}). Its ε over the world's year is {_n(eps)}, "
                     "and at 1 or more it bounds nothing, so no amount is printed under any of its rows."]
    if error < 1:
        together = [
            "- **together**, a link's execution phase carrying K = schedules + cells + jitter + min(entries, advice + "
            f"the compute audit's part), at D_j = {of}, with no advice priced and with a response's output count taken "
            f"at 0 under {_names(count['at_zero_in'])}: "
            + "; ".join(f"{_n(sums[name]['GB'])} GB {TOGETHER[name]}" for name in record)
            + f"; each failing with at most {_n(error)}, the two parts' errors summed.",
            *(f"- together, {clause}; error {_n(error)}." for clause in _labelled(year, "GB", set_time))]
    else:
        together = [f"- **together: {NO_BOUND}**. The two parts' errors over the world's year sum to {_n(error)}, and at "
                    "1 or more the sum bounds nothing, so no amount is printed for the two together."]
    lines = [
        f"# verity bench exfiltration: {doc['model']}'s exfiltration capacity over a monitoring period, at "
        f"{_pct(share)} of total compute", "",
        *measured,
        f"Over the world's year, at {_pct(share)} of total compute, in two parts:", "",
        f"- **the compute audit's part{label}: {_n(head['GB'])} GB** (modeled), {_n(head['GB'] / w['GB'])} copies of the weights "
        f"({w['label']}, {_n(w['GB'])} GB) and {_n(year['compute_audit']['bits_a_request'])} bits a request. It is the "
        f"planner's row `{doc['row']}`{what}, read at the audit budget where its total is {_pct(share)} of the compute, "
        f"β = {_part(head['beta'])}{outside}: its bound, "
        f"{_n(head['bound']['GB'])} GB, times λ = {_n(head['lambda'])}. The row is priced over "
        f"{_n(head['requests'])} requests, the count the network certifier's part is over.",
        f"- {_how(head['priced'])}.",
        f"- what prices the row's proving time, as its rows record it{plan}: {_by_source(sources)}.",
        *assumed,
        *certifier,
        *together,
        "- the output count: "
        + (f"{_n(count['GB'])} GB ({count['mark']}), {_n(count['bits_a_request'])} bits a request" if eps < 1 else
           f"{NO_BOUND} ({count['mark']}), under the network certifier's ε")
        + ". Until a served audit has sampled a stop block the server chooses a response's output count, and the network "
        f"certifier's row `{count['row']}` charges it as its schedules. The sums under {_names(count['at_zero_in'])} "
        "take it at 0, as they price no advice."
        + ("" if stop is None else f" The stop class's verdicts of record (`{stop['record']}`, "
           f"{', '.join(stop['runs']) or 'no run read'}): {stop['said']}."),
        f"- the compute audit's part fails with at most δ = {_n(head['delta'])} over the monitoring period. The network "
        f"certifier's ε over the world's year is each execution phase's scan's δ ({delta['mark']}), "
        f"{_p(delta['value'])}, summed over every link's execution phases, {_n(year['phases'])} of them: {_p(eps)}. "
        f"The two together fail with at most the sum of both, {_n(error)}. A figure whose error, summed over "
        f"everything it covers, is 1 or more is printed as \"{NO_BOUND}\", with that error and no amount.",
        f"- with the network check's own proving at {_pct(net['share'])} of total compute, which the planner's rows "
        f"don't count (assumed, no run), the audit budget is β = {_part(net['beta'])} and the compute audit's part "
        f"{_n(net['GB'])} GB.",
        *_security(head["binding"]),
        f"- of {len(doc['inputs'])} inputs, {marks['measured']} {'is' if marks['measured'] == 1 else 'are'} measured, "
        + _listed([f"{marks['modeled']} modeled", f"{marks['assumed']} assumed",
                   *(f"{marks[m]} {PRINTED.get(m, m)}" for m in ("missing", RUN_NOT_PROVED) if marks[m])]) + ".", "",
        f"Inputs, each a labelled record of the store (`{RUN_OF_RECORD}`):", "",
        f"- compute-audit: {', '.join(doc['runs'])} (`{doc['record']}`): the planner's rows, "
        f"{sum(len(p['pinned']) for p in doc['compute_audit'])} of row `{doc['row']}` at step `{head['step']}` over "
        f"{len(doc['compute_audit'])} scopes"
        + "".join(f", and the {p['scope']}'s plan ({p['priced']['run']})" for p in planned if p["priced"]["of"] == "plan"),
        *([] if stop is None else [
            f"- the stop class: {', '.join(stop['runs']) or 'no run read'} (`{stop['record']}`): "
            f"{len(stop['verdicts'])} verdicts, {stop['said']}"]),
        "- the network certifier's part and the requests both parts count over (modeled or assumed, read from the tree, "
        "not records): `verity bench certifier_term`'s inputs, among them the request it counts and what that request "
        "is, in the table below", "",
        "## Inputs", "",
        *_table(("part", "input", "value", "unit", "mark", "source"),
                [(i["part"], i["input"], _n(i["value"]), i["unit"], PRINTED.get(i["mark"], i["mark"]), i["source"])
                 for i in doc["inputs"]]), "",
        *parts,
        f"## The compute audit's part at {_pct(share)} of total compute", "",
        "Each scope's rows of record, read at the audit budget β where their total is the share. The requests are the "
        "count the rows are priced over. The parts are shares of total compute; the wait is part of serving.", "",
        *_table(("scope", "run", "β", "bound at λ = 1, GB", "GB", "x weights", "requests", *PARTS),
                [(p["scope"], p["run"], _part(p["beta"]) + ("" if p["inside"] else " (extrapolated)"), _n(p["bound"]["GB"]),
                  _n(p["GB"]), _n(p["bits"] / w["bits"]), _n(p["requests"]), *(_part(p["parts"].get(part)) for part in PARTS))
                 for p in doc["compute_audit"]]), "",
        *(planned and (
            "Each scope's proving time by how its block families are priced, by its plan: the rows' own at the pinned "
            "budget nearest β, or a plan of record at its own budget. How each family is priced is the rows' account.", "",
            *_table(("scope", "plan", "at β", "priced", "share of proving time", "block families", "the largest"),
                    [(p["scope"], p["priced"]["run"] + (" (rows)" if p["priced"]["of"] == "rows" else ""),
                      _part(p["priced"]["beta"]), way, _part(x["share"]), str(len(x["families"])),
                      ", ".join(f"{family} {_part(s)}" for family, s in list(x["families"].items())[:3]))
                     for p in planned for way, x in p["priced"]["ways"].items()]), "")),
        "## The network certifier's part over the world's year", "",
        f"`verity bench certifier_term`'s rows (`{cert['schema']}`) at its request count `{cert['request_count']}`, for a "
        f"class of at most {_n(cert['response_ids'])} ids a response. The term is the schedules, the fail cells and the "
        f"jitter, and the term of record is the one at D_j = {of}. The term at D_j = {set_time}, and every figure of a "
        f"row whose name carries it, is printed only under the network certifier's label, \"{card}\". The entries are "
        "the compute side's. Each row carries the δ of its scans and its ε over the world's year, their sum over every "
        "link's execution phases; a row whose ε is 1 or more has no amount.", "",
        *(f"- {name} ({k['standing']}): {k['counts']}" for name, k in cert["legend"].items()), "",
        *_table(("row", "schedules, GB", "cells, GB", f"term at D_j = {of}, GB", "x weights",
                 f"term at D_j = {set_time} {card}, GB", "entries, GB", "scan's δ", "ε"),
                [(r["row"] + (f", {r['with_set_time_card']}" if r["with_set_time_card"] else ""),
                  *((NO_BOUND,) * 6 if r["eps"] >= 1 else
                    (_n(r["schedules"]["GB"]), _n(r["cells"]["GB"]), _n(r["term"][of]["GB"]),
                     _n(r["term"][of]["of_weights"]), _n(r["term"][set_time]["GB"]),
                     _n(r["entries"] and r["entries"]["GB"], "the compute side's"))),
                  _n(delta["value"]), _n(r["eps"])) for r in cert["rows"]]), "",
        f"The entries as the scan's check charges them: {cert['entries_rule']}.", "",
        "## Each monitoring period over all compute: the two parts side by side, in GB", "",
        f"The network certifier's term is the one of record, at D_j = {of}, and `together` adds the compute audit's part "
        "to it, or the entries where the row charges them and they are fewer. The output count is the schedules of "
        f"the row `{count['row']}`, what a response's output count is charged while the server chooses it "
        f"({count['mark']}): the sum under {_names([name for name in count['at_zero_in'] if name in record])} takes it "
        "at 0.", "",
        "Each part's amounts are followed by its error over the line's scope and monitoring period: the compute "
        "audit's δ; the network certifier's ε, its scans' δ summed over every link of the scope and every execution "
        "phase of the period; and `together`'s, the sum of the two. A figure whose error is 1 or more is printed as "
        f"\"{NO_BOUND}\".", "",
        *_table(("compute", "period", "GPU-seconds", "requests", "compute audit", "bits a request", "compute audit's δ",
                 *(f"certifier, {name}" for name in record), "scan's δ", "certifier's ε",
                 *(f"together, {name}" for name in record), "together's error", f"output count ({count['mark']})"),
                [(p["compute"], p["period"], _n(p["gpu_seconds"]), _n(p["requests"]),
                  _n(p["compute_audit"] and p["compute_audit"]["GB"], "no rows of record"),
                  _n(p["compute_audit"] and p["compute_audit"]["bits_a_request"]),
                  _n(p["compute_audit"] and p["compute_audit"]["delta"]),
                  *(_amount(p["together"][name]["certifier_GB"], p["eps"]) for name in record),
                  _n(delta["value"]), _n(p["eps"]),
                  *(_amount(p["together"][name]["GB"], p["error"]) for name in record), _n(p["error"]),
                  _amount(p["output_count"]["GB"], p["eps"]))
                 for p in doc["periods"]]), "",
        f"## Each monitoring period over all compute, {card}: in GB", "",
        f"No amount of this table is of record. Each is under the network certifier's label, \"{card}\", which its "
        f"column carries: every amount of a row that carries the label, and every other row's at D_j = {set_time}."
        + (f" The sum under {_names([name for name in count['at_zero_in'] if name in labels])} takes the output count "
           "at 0 too." if any(name in labels for name in count["at_zero_in"]) else "")
        + " A line's errors are those of the table above.", "",
        *_table(("compute", "period",
                 *(f"{part}, {name}, {label}" for name, label in labels.items() for part in ("certifier", "together")),
                 *(f"{part}, {name}, D_j = {set_time} {card}" for part in ("certifier", "together") for name in record),
                 "scan's δ", "certifier's ε", "together's error"),
                [(p["compute"], p["period"],
                  *(_amount(p["together"][name][key], p[e]) for name in labels for key, e in ERRORS.items()),
                  *(_amount(p["together"][name]["with_set_time_card"][key], p[e]) for key, e in ERRORS.items()
                    for name in record),
                  _n(delta["value"]), _n(p["eps"]), _n(p["error"]))
                 for p in doc["periods"]]),
    ]
    if doc["refused"]:
        lines += ["", "Refused (not read):", "", *(f"- {x['target']} (`{x['record']}`): {x['why']}" for x in doc["refused"])]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """`verity bench exfiltration`: print the two parts (`render`), or with ``--json`` their document (`build`)."""

    ap = argparse.ArgumentParser(
        prog="verity bench exfiltration",
        description=f"{MODEL}'s exfiltration capacity in GB over a week, a month and a year of the world's and a "
                    "datacenter's compute, at a share of total compute: the compute audit's part, from the break-even "
                    "planner's rows of record in the evidence store, beside the network certifier's part, with every "
                    "input labelled measured, modeled, assumed, missing or run, not proved. Every figure carries the "
                    "error summed over "
                    f"everything it covers, and one whose error is 1 or more is printed as \"{NO_BOUND}\", with no "
                    "amount.")
    ap.add_argument("--json", action="store_true", help="print the document (value, unit, inputs, parts, periods)")
    ap.add_argument("--share", type=float, default=SHARE, metavar="S",
                    help=f"the share of total compute the figure is read at (default {SHARE})")
    ap.add_argument("--network-check-share", type=float, default=NETWORK_CHECK_SHARE, metavar="S",
                    help="the network check's own proving as a share of total compute, which the planner's rows don't "
                         f"count (default {NETWORK_CHECK_SHARE}, assumed)")
    ap.add_argument("--row", default=ROW, help=f"the planner's row (default `{ROW}`, the plan of record's; `{ASSUMED_ROW}` "
                                               f"is printed only {ASSUMED_LABEL})")
    ap.add_argument("--step", default=STEP, help=f"the row's step, or the beginning of its name (default `{STEP}`)")
    ap.add_argument("--variant", help=f"the row's variant, as the planner's driver names it (`{VARIANT}`), where a run "
                                      "holds several (default: a run holds one)")
    ap.add_argument("--response-ids", type=int, default=RESPONSE_IDS, metavar="N",
                    help="the most ids a response holds in the network certifier's class (default "
                         f"{RESPONSE_IDS}, assumed: the class that holds the plan of record's longest output)")
    ap.add_argument("--scan-delta", type=Fraction, metavar="D",
                    help="the δ of each execution phase's scan of the network certifier, which its rows' ε sum (default: "
                         "the δ at which the row printed that sums the most execution phases, the world's year, carries "
                         f"an ε of {PERIOD_ERROR}, assumed; no term depends on it)")
    ap.add_argument("--served-run", type=Path, metavar="FILE",
                    help="a served two-stage run's record as its reader gives it, one JSON object (`served_share`): the "
                         "share of compute it measures is printed beside the given share, with the compute audit's part "
                         f"at it (default: none, and the share is \"{NO_RUN}\")")
    ap.add_argument("--local-only", action="store_true", help="read this machine's catalog alone, not the store's remote")
    a = ap.parse_args(argv)
    try:
        served = None if a.served_run is None else json.loads(a.served_run.read_text())
        doc = build(share=a.share, network_check=a.network_check_share, row=a.row, step=a.step, variant=a.variant,
                    response_ids=a.response_ids, scan_delta=a.scan_delta, served=served, local_only=a.local_only)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"verity bench exfiltration: {e}", file=sys.stderr)
        return 2
    doc["spot_holes"] = spot_status.status(doc["runs"])
    shown = None
    if served is not None:
        from verity_bench import served_row  # it reads this module's names

        shown = served_row.account(served)
    if shown is not None:
        doc["served_run"] = shown
    print(json.dumps(doc, indent=1, ensure_ascii=False) if a.json else
          render(doc) + ("" if shown is None else "\n\n" + served_row.render(shown))
          + "\n\n" + spot_status.render(doc["spot_holes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
