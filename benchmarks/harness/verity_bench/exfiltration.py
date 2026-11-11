"""`verity bench exfiltration`: GLM-5.3's exfiltration capacity over a monitoring period at a share of total compute
(10% by default), in GB, as two parts side by side with every input labelled measured, modeled or assumed. It
computes neither part itself: it composes the two calculators that do, and runs on any machine that reaches the
store, with no GPU and no node.

* **The compute audit's part** is the break-even planner's bound, read from its rows of record. A record is named in
  the store, never in code (`served_table`'s rule): the runs that carry `run_of_record=RECORD` and no `superseded_by`,
  each holding the planner's rows as `ROWS_FILE` in its run files, one JSON object a line. A row prices one case of
  the plan (`row`, `step`, `world`) over one scope at one pinned audit budget β (`b10`): its share of total compute
  (`total`, with its parts `PARTS`) and its bound in bits (`bound_bits`). The command takes one case (`ROW`, `STEP`,
  `WORLD`), and for each scope reads off the budget at which the total is the share (`at_share`): the total is linear
  in β, a least-squares line over the pinned budgets, and ln(bound) is linear in ln β between the two that bracket
  it. The row plans for the weights / λ (`weights_bits / target_bits`), and the part is λ times its bound.
* **The network certifier's part** is `verity bench certifier_term`'s (`benchmarks/network_traces/leak_bounds.py`:
  `inputs`, `document`): the schedules, the fail cells and the jitter of each of its rows, at the same scopes.

Both count over the same requests: the planner's, of the plan its rows are priced on (`PLAN`, read from the planner
in the tree, `benchmarks/spot_figures/breakeven/`), handed to the network certifier's calculator as the world's
requests a year in place of the census's it would count. The two parts must cover the same GPU-seconds and the same
weights, or the command refuses.

A link's execution phase carries K = schedules + cells + jitter + min(entries, advice + the wrong blocks' output
bits at δ). The first three are the network certifier's part, the last term's second branch is the compute audit's
part, and the command sums them per scope and monitoring period under each of the network certifier's rows
(`together`). It prices no advice.

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
from pathlib import Path
from typing import Any

from verity_bench import jobs
from verity_bench.served_table import RUN_OF_RECORD, SUPERSEDED_BY, record_runs, records

SCHEMA = "exfiltration/v1"
MODEL = "GLM-5.3"
#: the record whose runs hold the planner's rows, and the rows' file in a run's files
RECORD = "exfiltration-glm53-compute-audit"
ROWS_FILE = "out/rows.jsonl"
#: the share of total compute the figure is read at
SHARE = 0.10
#: the network check's own proving as a share of total compute, which the planner's rows don't count: no run measures it
NETWORK_CHECK_SHARE = 0.003
#: the case of the plan the figure is: the planner's world (rho's world fleet), its row for the registered and served
#: cut with every modeled lever at its measured value, and the step of that row that prices the verifier as measured
#: (a prefix of the step's name)
WORLD, ROW, STEP = "ours", "(i) honest", "+ the verifier as measured"
#: the planner's case whose line carries the plan the rows of record are priced on, and so its requests
PLAN = "P5+poolwin600"
#: a row's parts of total compute; `wait` is a part of `serve`
PARTS = ("serve", "wait", "prove", "replay", "pool", "edge", "compile")
#: the network certifier's rows the two parts are summed under, and what each is called beside the compute audit's
TOGETHER = {"scan": "as the scan's check charges it today", "N2": "as N2 charges it today",
            "all three": "with the lengths held, the timing taken once and the entries charged through the compute audit",
            "design": "by the design no run has behind it"}
GB = 8e9
MARKS = ("measured", "modeled", "assumed")
#: a card's status for a figure, as its mark
_MARK = {"measured": "measured", "modeled": "modeled", "derived": "modeled", "datasheet": "assumed", "assumed": "assumed",
         "ruled": "assumed"}
#: the inputs both parts read: the fleet and what a request of the plan is, which give the requests, and the weights
SHARED = ("world_machines", "datacenter_machines", "machine_gpus", "mfu_serve", "peak_flops", "request_tokens",
          "token_macs", "world_requests", "weights")


def _load(root: Path, directory: str, module: str) -> Any:
    """A module of another benchmark's directory, which is no package: loaded by its path."""

    path = str(root / "benchmarks" / directory)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(module)


# ---- the compute audit's part: the planner's rows of record -----------------------------------------------------------


def _label(key: str = RUN_OF_RECORD, value: str = RECORD) -> str:
    return f"`research data label RUN {key} {value} --by LANE`"


def read_rows(store: Any, run: str) -> tuple[list[dict], str | None]:
    """The planner's rows in a run's files, or why the run holds none."""

    attempt = store.get_attempt(run)
    if attempt is None:
        return [], "not in the store or on its remote"
    out = attempt.get("outputs") or {}
    files = out.get("run_files") or out.get("run_record")
    if not files:
        return [], "kept no run files"
    if not store.listing(files, [ROWS_FILE]):
        return [], f"holds no planner rows (`{ROWS_FILE}`)"
    text = (Path(store.fetch(files, paths=[ROWS_FILE])) / ROWS_FILE).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()], None


def rows_of_record(store: Any, *, row: str = ROW, step: str = STEP, world: str = WORLD) -> tuple[dict, list[str], list[dict]]:
    """The case's rows of record by scope, each scope's from the one run that holds it, with the runs read and the
    runs left out with why."""

    labelled = record_runs(store, records(store).get(RECORD, []))
    if not labelled:
        raise ValueError(f"no planner rows of record: nothing in the store carries `{RUN_OF_RECORD}={RECORD}` "
                         f"(label each run that holds them: {_label()})")
    scopes: dict[str, tuple[str, list[dict]]] = {}
    runs, refused = [], []
    for run in sorted({run for run, _ in labelled}, reverse=True):
        if superseded := [str(lab.value) for lab in store.labels(run, key=SUPERSEDED_BY)]:
            found, why = [], f"superseded by {', '.join(superseded)}"
        else:
            found, why = read_rows(store, run)
            found = [r for r in found if not r.get("error") and (r.get("world"), r.get("row")) == (world, row)
                     and str(r.get("step", "")).startswith(step)]
            why = why or (None if found else f"holds no row `{row}` at a step `{step}...` of the world `{world}`")
        if why:
            refused.append({"target": run, "record": RECORD, "why": why})
            continue
        runs.append(run)
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
        left = "; ".join(f"{x['target']} {x['why']}" for x in refused)
        raise ValueError(f"no run of record holds the planner's rows: {left} (label one that does: {_label()})")
    return scopes, runs, refused


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


def _priced(row: Mapping) -> dict[str, dict]:
    """A row's proving time by how its block families are priced: each way's share of it and its families, largest
    first."""

    out: dict[str, dict] = {}
    families = ((row.get("split") or {}).get("families") or {}).items()
    for family, f in sorted(families, key=lambda kv: -kv[1]["share"]):
        way = out.setdefault(" and ".join(sorted(f["priced"], reverse=True)), {"share": 0.0, "families": []})
        way["share"] += f["share"]
        way["families"].append(family)
    return out


def at_share(scope: str, run: str, rows: Sequence[Mapping], share: float) -> dict:
    """One scope's rows read off at the audit budget where the total is ``share`` of the compute: the budget, whether
    the pinned budgets bracket it, the bound there at λ = 1 and λ times it, each part of the total, and what the
    rows hold fixed (λ, δ, the most bits a request's output holds, the scope's GPU-seconds and the weights)."""

    rows = sorted(rows, key=lambda r: r["b10"])
    fixed = {"lambda": {r["weights_bits"] / r["target_bits"] for r in rows}, "delta": {r["delta_seen"] for r in rows},
             "output_bits": {r["S"] for r in rows}, "gpu_seconds": {r["gpu_seconds"] for r in rows},
             "weights_bits": {r["weights_bits"] for r in rows}}
    if len({r["b10"] for r in rows}) < 2 or len({r["b10"] for r in rows}) != len(rows):
        raise ValueError(f"{run}'s {scope} rows pin {len(rows)} audit budgets, {len({r['b10'] for r in rows})} distinct: "
                         "the total's line needs two or more, each once")
    if moved := sorted(k for k, v in fixed.items() if len(v) != 1):
        raise ValueError(f"{run}'s {scope} rows differ in {', '.join(moved)}, which one plan holds fixed over its "
                         "audit budgets")
    slope, intercept = _fit([(r["b10"], r["total"]) for r in rows])
    beta = (share - intercept) / slope
    if not beta > 0:
        raise ValueError(f"{run}'s {scope} rows spend {intercept:.2%} of the compute before any audit: no audit budget "
                         f"brings the total to {share:.2%}")
    lam = fixed["lambda"].pop()
    bits = _between(rows, "bound_bits", beta, log=True)
    near = min(rows, key=lambda r: abs(r["b10"] - beta))
    compute, _, period = scope.rpartition(" ")
    return {"scope": scope, "compute": compute, "period": period, "run": run, "step": rows[0]["step"],
            "mix": rows[0].get("mix"), "pinned": [r["b10"] for r in rows], "beta": beta,
            "inside": rows[0]["b10"] <= beta <= rows[-1]["b10"],
            "residual": max(abs(r["total"] - (intercept + slope * r["b10"])) for r in rows),
            **{k: v.pop() for k, v in fixed.items() if k != "lambda"}, "lambda": lam,
            "bound": {"bits": bits, "GB": bits / GB}, "bits": bits * lam, "GB": bits * lam / GB,
            "parts": {p: _between(rows, p, beta) for p in PARTS if all(r.get(p) is not None for r in rows)},
            "priced": _priced(near)}


# ---- the requests both parts count over, and the network certifier's part ---------------------------------------------


def plan(root: Path) -> dict:
    """The plan the rows of record are priced on, from the planner in the tree: each scope's requests and GPU-seconds,
    and what a request is."""

    sweep = _load(root, "spot_figures", "breakeven.sweep")
    line = sweep.build(sweep.cases()[PLAN])
    tokens, more = line[1]["tokens_per_request"], line[7]
    where = f"benchmarks/spot_figures/breakeven sweep.cases()['{PLAN}']"
    return {"scopes": {name: {"requests": s.requests, "gpu_seconds": s.gpu_seconds} for name, s in more["scopes"].items()},
            "request_tokens": {"value": tokens.value, "unit": tokens.unit, "mark": _MARK[tokens.status],
                               "source": f"{where}: {tokens.source}"},
            "token_macs": {"value": more["served_macs"], "unit": "MAC", "mark": "modeled",
                           "source": f"{where}: the workload card's served MACs a token with the indexer's blocks scaled "
                                     "to the keys the plan reads (reprice.workload at reprice.C343)"},
            "source": f"{where}: the world fleet's GPU-seconds x mfu_serve x peak_flops / (request_tokens x 2 x "
                      "token_macs), the requests of the plan the rows of record are priced on"}


def certifier(root: Path, requests: float, source: str) -> dict:
    """The network certifier's term at the datacenter's and the world's load (`leak_bounds.document`), counted over
    ``requests`` a world year, which are marked modeled with ``source``."""

    lb = _load(root, "network_traces", "leak_bounds")
    inp = lb.inputs(world_requests=requests)
    inp["world_requests"] = inp["world_requests"]._replace(mark="modeled", source=source)
    return lb.document(inp, only=("datacenter", "world"))


# ---- the document ----------------------------------------------------------------------------------------------------


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9)


def _inputs(points: Mapping[str, dict], runs: Sequence[str], term: Mapping, planned: Mapping, share: float,
            network_check: float, row: str) -> list[dict]:
    """Every input with its mark and source: the compute audit's from its rows of record, the network certifier's
    as its calculator marks them, and those both read. A request's tokens and MACs are the plan's, in place of the
    census's the calculator would count the requests from."""

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
        one(audit, "compute_share", share, "", "assumed", "the share of total compute the figure is read at (--share)"),
        one(audit, "delta", head["delta"], "", "assumed",
            f"{where} `delta_seen`: the audit's δ over the monitoring period, a stated parameter"),
        one(audit, "lambda", head["lambda"], "", "assumed",
            f"{where} `weights_bits / target_bits`: the row plans for the weights / λ and its part is λ times its "
            "bound, a premise no run measures"),
        one(audit, "output_bits", head["output_bits"], "bit/request", "assumed",
            f"{where} `S`: the most bits a request's output holds, which no check certifies"),
        *(one(audit, f"proving priced {way}", p["share"], "of proving time", "measured" if way == "measured" else "modeled",
              f"{where} `split.families`: {len(p['families'])} block families, the largest "
              f"{', '.join(p['families'][:2])}") for way, p in head["priced"].items()),
        one(audit, "network_check_share", network_check, "", "assumed",
            "no run: the network check's own proving as a share of total compute, which the planner's rows don't "
            "count (--network-check-share)"),
        one(audit, "advice", 0, "bit", "assumed", "not counted: no input here prices the advice"),
    ]
    for name, i in term["inputs"].items():
        i = planned.get(name, i)
        out.append(one(both if name in SHARED else net, name, i["value"], i["unit"], i["mark"], i["source"]))
    if off := sorted(i["input"] for i in out if i["mark"] not in MARKS or not i["source"]):
        raise ValueError(f"an input is marked one of {', '.join(MARKS)}, with its source: {', '.join(off)}")
    return out


def build(*, share: float = SHARE, network_check: float = NETWORK_CHECK_SHARE, row: str = ROW, step: str = STEP,
          store: Any = None, root: Path | None = None, local_only: bool = False) -> dict:
    """The two parts from the store's rows of record and the tree: the inputs with their marks, the compute audit's
    part at ``share`` of total compute for each scope of record, the network certifier's rows at the world's year,
    and the two side by side for each scope and monitoring period, summed under each of `TOGETHER`."""

    if not (0 < share < 1 and 0 <= network_check < share):
        raise ValueError(f"the share of total compute is in (0, 1) and the network check's own below it, not {share!r} "
                         f"and {network_check!r}")
    root = root or jobs.repository_root()
    completion = jobs.StoreCompletion(store, root=root, local_only=local_only)
    if not local_only:
        completion.pull_labels()
    scopes, runs, refused = rows_of_record(completion.store, row=row, step=step)
    points = {scope: at_share(scope, run, its, share) for scope, (run, its) in scopes.items()}
    if "world year" not in points:
        raise ValueError(f"no run of record holds the world year's rows ({', '.join(runs)} hold {', '.join(sorted(points))}): "
                         f"label the run that does ({_label()})")
    head = points["world year"]
    less = at_share("world year", *scopes["world year"], share - network_check)

    planned = plan(root)
    term = certifier(root, planned["scopes"]["world year"]["requests"], planned["source"])
    weights = term["inputs"]["weights"]["value"]
    if not _same(head["weights_bits"], weights):
        raise ValueError(f"the two parts are shares of different weights: the rows of record hold {head['weights_bits']:,} "
                         f"bits and the network certifier's calculator {weights:,}")
    seen = {(r["scope"], r["period"]): r for r in term["rows"]}
    order = {f"{compute} {period}": n for n, (compute, period) in enumerate(seen)}
    periods = []
    for (compute, period), at in seen.items():
        scope = f"{compute} {period}"
        point, its = points.get(scope), planned["scopes"][scope]
        for name, mine, theirs in (("GPU-seconds", point and point["gpu_seconds"], at["gpu_seconds"]),
                                   ("GPU-seconds", its["gpu_seconds"], at["gpu_seconds"]),
                                   ("requests", its["requests"], at["requests"])):
            if mine is not None and not _same(mine, theirs):
                raise ValueError(f"the two parts cover different compute over the {scope}: {mine:.6g} {name} on the compute "
                                 f"audit's side and {theirs:.6g} on the network certifier's")
        rows = {r["row"]: r for r in term["rows"] if (r["scope"], r["period"]) == (compute, period)}
        audit = point and {"GB": point["GB"], "bits": point["bits"], "beta": point["beta"], "inside": point["inside"],
                           "of_weights": point["bits"] / weights, "bits_a_request": point["bits"] / at["requests"],
                           "run": point["run"]}
        together = {}
        for name in TOGETHER:
            r = rows[name]
            entries = r["entries"] and r["entries"]["bits"]
            mine = None if audit is None else audit["bits"] if entries is None else min(entries, audit["bits"])
            together[name] = {"certifier_GB": r["term"]["1"]["GB"],
                              "GB": None if mine is None else (r["term"]["1"]["bits"] + mine) / GB}
        periods.append({"compute": compute, "period": period, "gpu_seconds": at["gpu_seconds"], "requests": at["requests"],
                        "eps": at["eps"], "compute_audit": audit, "together": together})

    return {"schema": SCHEMA, "model": MODEL,
            "name": f"{MODEL}'s exfiltration capacity over the world's year at {_pct(share)} of total compute: the "
                    "compute audit's part",
            "value": head["GB"], "unit": "GB", "status": "modeled", "share": share, "row": row, "record": RECORD,
            "runs": runs, "refused": refused, "weights": {"bits": weights, "GB": weights / GB, "label": term["weights"]["label"]},
            "inputs": _inputs(points, runs, term, planned, share, network_check, row),
            "compute_audit": sorted(points.values(), key=lambda p: (p["scope"] not in order, order.get(p["scope"], 0))),
            "network_check": {"share": network_check, "beta": less["beta"], "inside": less["inside"], "GB": less["GB"]},
            "certifier": {"legend": term["legend"], "entries_rule": term["entries_rule"], "phase": term["phase"],
                          "rows": [r for r in term["rows"] if (r["scope"], r["period"]) == ("world", "year")]},
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


def _pct(x: float | None) -> str:
    return "none" if x is None else f"{100 * x:.4g}%"


def _part(x: float | None) -> str:
    return "none" if x is None else f"{100 * x:.3f}%"


def _table(head: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head), *("| " + " | ".join(r) + " |" for r in body)]


def render(doc: Mapping[str, Any]) -> str:
    """``doc`` (`build`) as text: the figure and what it is, the inputs with their marks, each part, and the two side
    by side for every scope and monitoring period."""

    share, net, w = doc["share"], doc["network_check"], doc["weights"]
    head = next(p for p in doc["compute_audit"] if p["scope"] == "world year")
    year = next(p for p in doc["periods"] if (p["compute"], p["period"]) == ("world", "year"))
    scan = next(r for r in doc["certifier"]["rows"] if r["row"] == "scan")
    marks = {m: sum(i["mark"] == m for i in doc["inputs"]) for m in MARKS}
    outside = "" if head["inside"] else ", outside the pinned budgets: extrapolated"
    what = ", the registered and served cut with every modeled lever at its measured value" if doc["row"] == ROW else ""
    lines = [
        f"# verity bench exfiltration: {doc['model']}'s exfiltration capacity over a monitoring period, at "
        f"{_pct(share)} of total compute", "",
        f"Over the world's year, at {_pct(share)} of total compute, in two parts:", "",
        f"- **the compute audit's part: {_n(head['GB'])} GB** (modeled), {_n(head['GB'] / w['GB'])} copies of the weights "
        f"({w['label']}, {_n(w['GB'])} GB) and {_n(year['compute_audit']['bits_a_request'])} bits a request. It is the "
        f"planner's row `{doc['row']}`{what}, read at the audit budget where its total is {_pct(share)} of the compute, "
        f"β = {_part(head['beta'])}{outside}: its bound, "
        f"{_n(head['bound']['GB'])} GB, times λ = {_n(head['lambda'])}.",
        f"- **the network certifier's part: {_n(scan['term']['1']['GB'])} GB** (modeled, on an assumed class) "
        f"{TOGETHER['scan']}, {_n(scan['term']['1']['bits_a_request'])} bits a request, and "
        + ", ".join(f"{_n(year['together'][name]['certifier_GB'])} GB {what}" for name, what in TOGETHER.items()
                    if name != "scan") + ", each at D_j = 1.",
        "- **together**, a link's execution phase carrying K = schedules + cells + jitter + min(entries, advice + the "
        "compute audit's part), with no advice priced: "
        + "; ".join(f"{_n(year['together'][name]['GB'])} GB {what}" for name, what in TOGETHER.items()) + ".",
        f"- the compute audit's part fails with at most δ = {_n(head['delta'])} over the monitoring period. The scan's ε "
        f"over the world's year, its execution phases' δ summed, is {_n(year['eps'])}: at 1 or more it bounds nothing.",
        f"- with the network check's own proving at {_pct(net['share'])} of total compute, which the planner's rows "
        f"don't count (assumed, no run), the audit budget is β = {_part(net['beta'])} and the compute audit's part "
        f"{_n(net['GB'])} GB.",
        f"- of {len(doc['inputs'])} inputs, {marks['measured']} are measured, {marks['modeled']} modeled and "
        f"{marks['assumed']} assumed.", "",
        f"Inputs, each a labelled record of the store (`{RUN_OF_RECORD}`):", "",
        f"- compute-audit: {', '.join(doc['runs'])} (`{doc['record']}`): the planner's rows, "
        f"{sum(len(p['pinned']) for p in doc['compute_audit'])} of row `{doc['row']}` at step `{head['step']}` over "
        f"{len(doc['compute_audit'])} scopes",
        "- the network certifier's part and the requests both parts count over (modeled or assumed, read from the tree, "
        "not records): `verity bench certifier_term`'s inputs and the planner's plan, in the table below", "",
        "## Inputs", "",
        *_table(("part", "input", "value", "unit", "mark", "source"),
                [(i["part"], i["input"], _n(i["value"]), i["unit"], i["mark"], i["source"]) for i in doc["inputs"]]), "",
        f"## The compute audit's part at {_pct(share)} of total compute", "",
        "Each scope's rows of record, read at the audit budget β where their total is the share. The parts are shares of "
        "total compute; the wait is part of serving.", "",
        *_table(("scope", "run", "β", "bound at λ = 1, GB", "GB", "x weights", *PARTS),
                [(p["scope"], p["run"], _part(p["beta"]) + ("" if p["inside"] else " (extrapolated)"), _n(p["bound"]["GB"]),
                  _n(p["GB"]), _n(p["bits"] / w["bits"]), *(_part(p["parts"].get(part)) for part in PARTS))
                 for p in doc["compute_audit"]]), "",
        "## The network certifier's part over the world's year", "",
        "`verity bench certifier_term`'s rows, at the planner's requests. The term is the schedules, the fail cells and "
        "the jitter; the entries are the compute side's.", "",
        *(f"- {name} ({k['standing']}): {k['counts']}" for name, k in doc["certifier"]["legend"].items()), "",
        *_table(("row", "schedules, GB", "cells, GB", "term at D_j = 1, GB", "x weights", "term at D_j = 2, GB",
                 "entries, GB"),
                [(r["row"], _n(r["schedules"]["GB"]), _n(r["cells"]["GB"]), _n(r["term"]["1"]["GB"]),
                  _n(r["term"]["1"]["of_weights"]), _n(r["term"]["2"]["GB"]),
                  _n(r["entries"] and r["entries"]["GB"], "the compute side's")) for r in doc["certifier"]["rows"]]), "",
        f"The entries as the scan's check charges them: {doc['certifier']['entries_rule']}.", "",
        "## Each monitoring period over all compute: the two parts side by side, in GB", "",
        "The network certifier's term is at D_j = 1, and `together` adds the compute audit's part to it, or the entries "
        "where the row charges them and they are fewer.", "",
        *_table(("compute", "period", "GPU-seconds", "requests", "compute audit", "bits a request",
                 *(f"certifier, {name}" for name in TOGETHER), *(f"together, {name}" for name in TOGETHER)),
                [(p["compute"], p["period"], _n(p["gpu_seconds"]), _n(p["requests"]),
                  _n(p["compute_audit"] and p["compute_audit"]["GB"], "no rows of record"),
                  _n(p["compute_audit"] and p["compute_audit"]["bits_a_request"]),
                  *(_n(p["together"][name]["certifier_GB"]) for name in TOGETHER),
                  *(_n(p["together"][name]["GB"]) for name in TOGETHER)) for p in doc["periods"]]),
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
                    "input labelled measured, modeled or assumed.")
    ap.add_argument("--json", action="store_true", help="print the document (value, unit, inputs, parts, periods)")
    ap.add_argument("--share", type=float, default=SHARE, metavar="S",
                    help=f"the share of total compute the figure is read at (default {SHARE})")
    ap.add_argument("--network-check-share", type=float, default=NETWORK_CHECK_SHARE, metavar="S",
                    help="the network check's own proving as a share of total compute, which the planner's rows don't "
                         f"count (default {NETWORK_CHECK_SHARE}, assumed)")
    ap.add_argument("--row", default=ROW, help=f"the planner's row (default `{ROW}`)")
    ap.add_argument("--step", default=STEP, help=f"the row's step, or the beginning of its name (default `{STEP}`)")
    ap.add_argument("--local-only", action="store_true", help="read this machine's catalog alone, not the store's remote")
    a = ap.parse_args(argv)
    try:
        doc = build(share=a.share, network_check=a.network_check_share, row=a.row, step=a.step, local_only=a.local_only)
    except (ValueError, RuntimeError) as e:
        print(f"verity bench exfiltration: {e}", file=sys.stderr)
        return 2
    print(json.dumps(doc, indent=1, ensure_ascii=False) if a.json else render(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
