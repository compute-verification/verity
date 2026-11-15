"""The document of `verity bench exfiltration` (`verity_bench.exfiltration`): both parts summed at each scope and
monitoring period, with every input and its mark (`build`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity_bench import jobs
from verity_bench.exfiltration_certifier import certifier
from verity_bench.exfiltration_common import (
    AMOUNTS,
    ASSUMED_LABEL,
    ASSUMED_ROW,
    DEFAULT_BASIS,
    GB,
    MARKS,
    MEASURED_WAYS,
    MODEL,
    NETWORK_CHECK_SHARE,
    ON_CPUS,
    OUTPUT_COUNT,
    RECORD,
    RESPONSE_IDS,
    ROW,
    SCHEMA,
    SECURITY,
    SERVING_GAP,
    SHARE,
    SHARED,
    SOURCES,
    STEP,
    TOGETHER,
    VERDICTS,
    WHOLE,
    _same,
)
from verity_bench.exfiltration_planner import _label, at_share, rows_of_record
from verity_bench.exfiltration_served import served_share
from verity_bench.exfiltration_text import _n, _names, _on, _part, _pct, _plan
from verity_bench.exfiltration_verdicts import (
    _by,
    _what,
    stop_charge,
    stop_class,
    stop_verdicts,
    verdict_runs,
)


def _timed_alone(priced: Mapping, share: float) -> tuple[str, str]:
    """The mark of the proving a plan prices from measured statements, ``share`` of its proving time, and what is
    said of it. Such a block type is timed alone, one session on a card. What it costs with the plan's provers
    sharing a GPU is the planner's factor, and the row's plan records the share of its proving at the factor that is
    assumed (`_sources`, `default`). The share is measured only where the plan records that and it is 0."""

    default, alone = (priced.get("sources") or {}).get("default"), "each is timed alone, one session on a card"
    if default is None:
        return "modeled", (f"{alone}, and the rows don't record how the plan prices it with its provers sharing a GPU "
                           "(no `split.basis`)")
    if not default:
        return "measured", f"{alone}, and the plan prices none of its proving at the planner's assumed default"
    least = max(0.0, share - (1 - default))
    return "modeled", (f"{alone}, and with its provers sharing a GPU the plan prices {_part(default)} of its proving "
                       f"at the planner's assumed default (`{DEFAULT_BASIS}`)"
                       + (f": at least {_part(least)} of the proving time, in these families" if least > 0 else ""))


def _inputs(points: Mapping[str, dict], runs: Sequence[str], term: Mapping, count: Mapping, share: float,
            network_check: float, row: str, served: Mapping, stop: Mapping | None = None) -> list[dict]:
    """Every input with its mark and source: the compute audit's from its rows of record, the network attester's
    as its calculator marks them, and those both read, among them the request the calculator counts and what it is.
    ``count`` is the world year's output count (`build`'s `output_count`), which the sums it names take at 0, on the
    execution phases the reader's answer holds (``term``'s `stop`); where it names none, no sum leaves the output
    count out and the inputs list none.
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
    priced = head["priced"] or {"ways": {}}
    default = (priced.get("sources") or {}).get("default")

    def how(way: str, its: float) -> tuple[str, str]:
        if way != "measured":
            return "measured" if way in MEASURED_WAYS else "modeled", ""
        mark, words = _timed_alone(priced, its)
        return mark, "; " + words

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
        *(one(audit, f"proving priced {way}", p["share"], "of proving time", how(way, p["share"])[0],
              f"`{RECORD}` ({priced['run']}) {_plan(priced)}, "
              f"block {'families' if len(p['families']) > 1 else 'family'} {', '.join(list(p['families'])[:2])}"
              f"{f' and {len(p["families"]) - 2} smaller' if len(p['families']) > 2 else ''}; how each is priced is the "
              f"rows' account (`split.families`){how(way, p['share'])[1]}") for way, p in priced["ways"].items()),
        *([one(audit, "proving priced at the planner's assumed default", default, "of proving time", "assumed",
               f"`{RECORD}` ({priced['run']}) {_plan(priced)}, `split.basis` `{DEFAULT_BASIS}`: the planner's default "
               "factor for what a block type costs with the plan's provers sharing a GPU, where no run has timed it so")]
          if default else []),
        one(audit, "network_check_share", network_check, "", "assumed",
            "no run: the network check's own proving as a share of total compute, which the planner's rows don't "
            "count (--network-check-share)"),
        one(audit, "advice", 0, "bit", "assumed", "not counted: no input here prices the advice"),
    ]
    charged = f"{'' if count['GB'] is None else _n(count['bits_a_request']) + ' bits a request '}as its schedules"
    under = count["at_zero_in"] and _names(count["at_zero_in"]) + _on(term["stop"])
    if under and stop and stop["complete"]:
        out.append(one(net, "output_count", 0, "bit/request", stop["mark"],
                       f"taken at 0 in the sums under {under}: the stop class's verdicts of record, "
                       f"`{VERDICTS}` ({', '.join(stop['runs'])}), are {stop['said']}, its boundary's registration "
                       f"`{stop['class']['registration']}`: "
                       + "; ".join(f"{_what(v)}, `{v['verdict']}` on {v['run']} by {_by(v)}, "
                                   + (f"proved by `{v['proved_by']}`" if v["proved_by"] else f"not proved: {v['open']}")
                                   for v in stop["verdicts"])
                       + ". Without them the server chooses a response's output count, which the network certifier's "
                       f"row `{OUTPUT_COUNT}` charges {charged}"))
    elif under:
        out.append(one(net, "output_count", 0, "bit/request", "assumed",
                       f"not counted in the sums under {under}: until a served audit has sampled a "
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
                      f"{whose}: {p['what']}, {p['source']}" + (f"; {ON_CPUS}" if p["on_cpus"] else "")
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


def _row_label(point: Mapping) -> str:
    """What row `ASSUMED_ROW`'s figure is printed under, from its rows read at a share (`at_share`): their note, else
    the share of its proving the table's rows price (own and family rows, `_sources`), else `ASSUMED_LABEL`."""

    if point.get("note"):
        return point["note"]
    by = ((point.get("priced") or {}).get("sources") or {}).get("by") or {}
    table = by.get(SOURCES[0], 0.0) + by.get(SOURCES[1], 0.0)
    return f"{_pct(table)} of its proving priced by the table's rows" if table else ASSUMED_LABEL


def at_assumed_prices(store: Any, head: Mapping, share: float, step: str, variant: str | None) -> dict:
    """Row `ASSUMED_ROW` over the world's year at ``share``, beside the figure ``head`` and under its label
    (`_row_label`, `ASSUMED_LABEL` where it is not read): its run, budget, δ and amount, and what prices its proving
    (`_sources`); or, with no amount, why it is not read: no run of record holds it, or its rows cover other compute or
    weights than the figure's."""

    out = {"row": ASSUMED_ROW, "label": ASSUMED_LABEL, "GB": None, "why": None}
    try:
        scopes, plans, _, _ = rows_of_record(store, row=ASSUMED_ROW, step=step, variant=variant)
        if "world year" not in scopes:
            return out | {"why": f"no run of record holds its world year's rows (they hold {', '.join(sorted(scopes))})"}
        run, rows = scopes["world year"]
        p = at_share("world year", run, rows, share, plans.get("world year"))
    except ValueError as e:
        return out | {"why": str(e)}
    if moved := [k for k in ("gpu_seconds", "requests", "weights_bits") if not _same(p[k], head[k])]:
        return out | {"why": f"its rows differ from the figure's in {', '.join(moved)}"}
    return out | {k: p[k] for k in ("run", "beta", "inside", "delta", "bits", "GB")} | {
        "label": _row_label(p), "sources": p["priced"] and p["priced"]["sources"]}


def build(*, share: float = SHARE, network_check: float = NETWORK_CHECK_SHARE, row: str = ROW, step: str = STEP,
          variant: str | None = None, response_ids: int = RESPONSE_IDS, scan_delta: Fraction | None = None,
          served: Mapping[str, Any] | None = None, store: Any = None, root: Path | None = None,
          local_only: bool = False) -> dict:
    """The two parts from the store's rows of record and the tree: the inputs with their marks, the compute audit's
    part at ``share`` of total compute for each scope of record, the network attester's rows at the world's year
    for a class of at most ``response_ids`` ids a response with each execution phase's scan at ``scan_delta`` (None:
    the δ `certifier` takes from `PERIOD_ERROR`), and the two side by side for each scope and monitoring period,
    summed under each of `TOGETHER`. It refuses rows of record priced over other requests than the network
    attester's part counts.

    Every figure carries the error summed over everything it covers, and one whose error is 1 or more carries no
    amount (`NO_BOUND`). A scope's monitoring period holds the network attester's ε (`eps`), the one its rows over
    that scope and period carry, their scans' δ summed over every link's execution phases; the compute audit's δ
    over the period (its `compute_audit`'s `delta`); and the sum of the two (`error`), what the two together fail
    with at most, None for a scope no run of record holds. Where `eps` is 1 or more the network attester's figures
    of that scope and period, the output count among them, are None, and where `error` is, the sums are. A row of the
    network attester's own over the world's year whose ε is 1 or more is held without its `AMOUNTS`.

    A sum is at the network attester's jitter bound of record. Its `label` is its row's own (None on most): where
    set, every figure of the row is printed under it. Its `with_set_time_card` is the same sum at the other jitter
    bound, with the label the network attester's document gives a figure there. A scope's `output_count` is what a
    response's output count is charged while the server chooses it, the schedules of the row `OUTPUT_COUNT`, with
    the rows whose sums take it at 0 (`at_zero_in`). Those are named from the stop verdicts' reader's answer and
    never from a schedule that reads 0: the rows of `TOGETHER` among the term document's `stop`'s `rows`, where the
    answer holds an execution phase, and none where it holds none. They take it at 0 on the execution phases the
    answer holds alone, which the document's `certifier`'s `stop` counts (`held_phases`; every one where `below`).

    The answer is `stop_charge`'s on the runs that carry the stop class's record (`verdict_runs`), the ones this
    reads or leaves out, and it is handed to the calculator (`certifier`). With no run carrying the record the
    reader is not asked and the calculator is handed none, so its rows are as with no record read. The document's
    `certifier`'s `stop` is the term document's own account of it (its `rows`, `held_phases`, `below`, `why` and
    `verdicts`), with the runs handed (`runs`) and those of them handed as superseded (`superseded`).

    Beside a ``row`` other than `ASSUMED_ROW`, `at_assumed_prices` holds that row's world year at ``share``
    (`at_assumed_prices()`), which is never the figure; as ``row`` itself, the figure's `label` is its (`_row_label`).

    ``served`` is a served two-stage run's record (`served_share`), from which the document's `compute_share` is the
    share of compute that run measures, with each of its parts, and where it is claimed the compute audit's part over
    the world's year at that share (`GB`, with its β), beside the figure at ``share``, which stays given and assumed
    (`given`). Without a record, or where it lacks a field, the share is not claimed, and `why` says so.

    Where the store holds the stop class's verdicts (`stop_verdicts`), `stop_class` is their account (`stop_class`),
    and where it is complete the output count's input, where the inputs list one, carries its mark; the runs left
    out are among `refused`. With none the document holds no `stop_class`."""

    if not (0 < share < 1 and 0 <= network_check < share):
        raise ValueError(f"the share of total compute is in (0, 1) and the network check's own below it, not {share!r} "
                         f"and {network_check!r}")
    measured = served_share(served)
    root = root or jobs.repository_root()
    completion = jobs.StoreCompletion(store, root=root, local_only=local_only)
    if not local_only:
        completion.pull_labels()
    scopes, plans, runs, refused = rows_of_record(completion.store, row=row, step=step, variant=variant)
    labelled = verdict_runs(completion.store)
    stop, answer = None, None
    if (verdicts := stop_verdicts(completion.store, labelled)) is not None:
        stop = stop_class(*verdicts[:2])
        refused = [*refused, *verdicts[2]]
        answer = stop_charge(root, completion.store, labelled)
    points = {scope: at_share(scope, run, its, share, plans.get(scope)) for scope, (run, its) in scopes.items()}
    if "world year" not in points:
        held = sorted({run for run, _ in scopes.values()})
        raise ValueError(f"no run of record holds the world year's rows ({', '.join(held)} hold{'s' * (len(held) == 1)} "
                         f"{', '.join(sorted(points))}): label the run that does ({_label()})")
    head = points["world year"]
    world_run, world_rows = scopes["world year"]
    less = at_share("world year", world_run, world_rows, share - network_check)
    assumed = None if row == ASSUMED_ROW else at_assumed_prices(completion.store, head, share, step, variant)
    measured["given"] = {"share": share, "mark": "assumed", "GB": head["GB"]}
    measured.update(GB=None, beta=None, inside=None, no_GB=None)
    if measured["value"] is not None:
        try:
            there = at_share("world year", world_run, world_rows, measured["value"])
            measured.update(GB=there["GB"], beta=there["beta"], inside=there["inside"])
        except ValueError as e:
            measured["no_GB"] = str(e)

    term = certifier(root, response_ids, scan_delta, answer)
    charged = {k: term["stop"][k] for k in ("rows", "held_phases", "below", "why", "verdicts")} | {
        "runs": list(labelled), "superseded": [run for run, by in labelled.items() if by]}
    if unnamed := [name for name in charged["rows"] if name in TOGETHER and name not in WHOLE]:
        raise ValueError(f"the network certifier's rows {_names(unnamed)} are read from verdict records, and this "
                         "command has no name for their sums where the stop class is charged its whole schedule: name "
                         "each in `WHOLE` (benchmarks/harness/verity_bench/exfiltration.py)")
    at_zero = [name for name in TOGETHER if name in charged["rows"]] if charged["held_phases"] else []
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
        for name, mine, theirs in (("GPU-seconds", None if point is None else point["gpu_seconds"], at["gpu_seconds"]),
                                   ("requests", None if point is None else point["requests"], at["requests"])):
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

            def summed(dj: str, r: Mapping = r, mine: float | None = mine, eps: float = eps,
                       error: float | None = error) -> dict:
                return {"certifier_GB": r["term"][dj]["GB"] if eps < 1 else None,
                        "GB": (r["term"][dj]["bits"] + mine) / GB if mine is not None and error is not None and error < 1 else None}

            together[name] = {**summed(of), "label": r["with_set_time_card"],
                              "with_set_time_card": {**summed(set_time), "label": card}}
        count = rows[OUTPUT_COUNT]["schedules"] if eps < 1 else {"GB": None, "bits": None}
        periods.append({"compute": compute, "period": period, "gpu_seconds": at["gpu_seconds"], "requests": at["requests"],
                        "phases": at["phases"], "eps": eps, "error": error, "compute_audit": audit,
                        "output_count": {"row": OUTPUT_COUNT, "GB": count["GB"], "bits": count["bits"],
                                         "bits_a_request": count["bits"] / at["requests"]
                                         if eps < 1 and at["requests"] else None,
                                         "mark": "assumed", "at_zero_in": at_zero},
                        "together": together})
    year = next(p for p in periods if (p["compute"], p["period"]) == ("world", "year"))

    return {"schema": SCHEMA, "model": MODEL,
            "name": f"{MODEL}'s exfiltration capacity over the world's year at {_pct(share)} of total compute: the "
                    "compute audit's part",
            "value": head["GB"], "unit": "GB", "status": "modeled", "share": share, "row": row, "variant": variant,
            "label": _row_label(head) if row == ASSUMED_ROW else None, "at_assumed_prices": assumed, "record": RECORD,
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
                          "stop": charged,
                          "rows": [r if r["eps"] < 1 else {k: v for k, v in r.items() if k not in AMOUNTS}
                                   for r in term["rows"] if (r["scope"], r["period"]) == ("world", "year")]},
            "periods": periods}
