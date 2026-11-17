"""The canonical top-k's serving cost priced from counts: a job's schedule and top-k counters times the top-k's device
time a call, in percent of the reference arm's serving time. It needs no GPU.

    priced.py OUT --units TOPK_CUT.jsonl [--prefill-units FILE] [--reference prod-trace] [--stock prod]
              [--ours ARM ...] [--tested-by RUN] [--json] [--result]

A step makes `CALLS` top-k calls on each card, each over the step's rows past `K` keys. A call costs the pass's time
with no failed row (`units`' ``base``, by rows and the longest row), and one with a failed row costs ``failed``
instead: a bisected call is set by its longest row, a refined one by its failed rows. The rows and their keys are the
step trace's (from the schedule); the failed rows are the counters' (from the model's outputs). Where a lifetime kept
the top-k's counts by step (`dsa_topk.steps_to`), each step's failed rows are its own, on its slowest card; else they
are the lifetime's totals spread over its decode steps, a lower bound by the rows read last (`rows_at`). Either way the
reader places a step's failed rows in its calls three ways, packed, a request's rows together, and one a call, the
bounds and the point, unless the counts by step hold the calls a remedy failed in (`CALLS_OF`), which it then reads.

A step that runs a prefill is priced at prefill shapes (``--prefill-units``, in `units`' form), and refused without
them; the reader then gives the prefill steps' measured added seconds, from both arms' step traces, as a term apart,
marked measured.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path

METRIC = "vllm/prod-serving-cost-glm53"
UNIT = "percent of stock's serving time"
INSTRUMENT = "priced"
#: the indexer's top-k calls a decode step: 21 of GLM-5.3's 78 layers, and the MTP layer once a step
CALLS = 22
#: the indexer's k: a row of at most K keys is not rewritten
K = 2048
#: the pass each arm's server ran, by its VERITY_DSA_TOPK (`dsa_topk.MODES`): canonical refines where the job's
#: counters count refined rows, and bisects at a commit before them
PASSES = {"canonical-bisect": "present", "canonical": "cut"}
#: the remedy each counted row was priced by: the pass whose forced failures time it
REMEDY = {"rows_bisected": "present", "rows_refined": "cut"}
#: the counts by step's name for a remedy's calls with a failed row, read where the counts hold it
CALLS_OF = {"rows_bisected": "calls_bisected", "rows_refined": "calls_refined"}
ENDS = ("lo", "mid", "hi")
#: Student's t at 97.5% by degrees of freedom, for an interval over lifetime pairs
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131)


def units(path: Path) -> dict:
    """Each pass's device time a call over the fast kernel's, in µs, by (rows, keys): ``base`` where no row failed, over
    the distributions timed, and ``failed`` where every row failed (a near miss or the pin's overflow), less ``base``;
    each (lo, hi)."""
    base: dict = {"present": {}, "cut": {}}
    whole: dict = {"present": {}, "cut": {}}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "error" in r:
            continue
        shape = (r.get("rows"), r.get("keys"))
        if "dist" in r:
            for p in base:
                c = r[p]["per_call"]
                if not c.get("rows_bisected") and not c.get("rows_refined"):
                    base[p].setdefault(shape, []).append(r[p]["added_us"])
        elif r.get("forced") in ("missed", "overflow"):
            for p in whole:
                whole[p].setdefault(shape, []).append(r[p]["us"])
    out = {}
    for p in base:
        b = {s: (min(v), max(v)) for s, v in base[p].items()}
        f = {s: (max(0.0, min(v) - b[s][1]), max(v) - b[s][0]) for s, v in whole[p].items() if s in b}
        if not b or not f:
            raise SystemExit(f"{path}: no {p} times with and without a failed row")
        out[p] = {"base": b, "failed": f}
    return out


def at(table: dict, rows: int, keys: int, end: int) -> float:
    """``table``'s µs at (rows, keys), end 0 or 1 of its range: linear between the shapes timed, rows held to the
    range timed, keys held below it and extended linearly above it."""
    rs = sorted({r for r, _ in table})
    ks = sorted({k for _, k in table})

    def along(xs: list[int], x: float, f) -> float:
        if x <= xs[0] or len(xs) == 1:
            return f(xs[0])
        i = min(max(bisect_left(xs, x), 1), len(xs) - 1)
        a, b = xs[i - 1], xs[i]
        return f(a) + (f(b) - f(a)) * (x - a) / (b - a)

    r = min(max(rows, rs[0]), rs[-1])
    return along(rs, r, lambda rr: along(ks, keys, lambda kk: table[(rr, kk)][end]))


def steps(trace: Path, draft_rows: int | None = None) -> tuple[list[dict], int]:
    """The step trace's steps, each with its ``step``, ``T`` and ``t``, its decode rows past `K` keys (``rows``) and
    their most keys (``keys``), and its prefill rows past `K` and their most keys (``prefill_rows``,
    ``prefill_keys``); and the rows a decode request runs a step (1 and its drafts, from the trace's header)."""
    out, n = [], draft_rows
    with gzip.open(trace, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "header" and n is None:
                n = 1 + ((r.get("config") or {}).get("speculative") or {}).get("num_speculative_tokens", 0)
            if r["kind"] != "step":
                continue
            rows = keys = pre = pre_keys = 0
            for _, start, m in r["rows"]:
                long_ = max(0, start + m - max(start, K))
                if m <= (n or 1):
                    rows += long_
                    keys = max(keys, start + m)
                elif long_:
                    pre += long_
                    pre_keys = max(pre_keys, start + m)
            out.append({"step": r.get("step"), "T": r.get("T"), "t": r["t"], "rows": rows, "keys": keys if rows else 0,
                        "prefill_rows": pre, "prefill_keys": pre_keys})
    return out, n or 1


def by_step(d: Path) -> dict[int, dict] | None:
    """A lifetime's top-k counts by step (`dsa_topk.steps_to`'s files beside its step trace): each step's tokens as
    its ranks counted them and each card's counts in it; None where the lifetime kept none."""
    files = sorted((d / "server").glob("dsa_topk_steps.*.jsonl"))
    if not files:
        return None
    out: dict[int, dict] = {}
    for path in files:
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        for line in lines[1:]:
            at_ = out.setdefault(line["step"], {"tokens": set(), "cards": {}})
            at_["tokens"].add(line["tokens"])
            at_["cards"][line["device"]] = {k: line[k] for k in lines[0]["stats"] if k in line}
    return out


def counted(d: Path) -> list[dict]:
    """A lifetime's counted passes, each with its window on the clock and its seconds as the cost counts them."""
    meta = json.loads((d / "meta.json").read_text())
    left = meta.get("left_out") or 0
    out = []
    for m in sorted(d.glob("*/rep-*/meta.json")):
        pm = json.loads(m.read_text())
        res = m.parent / "result.json"
        if pm["rep"] <= left or not res.is_file():
            continue
        out.append({"pass": f"{pm['load']} rep {pm['rep']}", "t0": pm["t0_unix"], "t1": pm["t0_unix"] + pm["wall_s"],
                    "seconds": json.loads(res.read_text())["duration"]})
    return out


def lifetimes(out: Path, arm: str) -> list[Path]:
    k = len(arm) + 1
    found = {int(d.name[k:]): d for d in out.glob(f"{arm}.*") if d.name[k:].isdigit() and d.is_dir()}
    return [found[n] for n in sorted(found)] or ([out / arm] if (out / arm).is_dir() else [])


def failure_s(steps_: list[tuple[int, int]], rows: float, table: dict, per_request: int, calls: int = CALLS) -> dict:
    """The seconds a card spends on ``rows`` failed rows over the decode steps ``steps_`` ((rows, keys) of each of a
    step's ``calls``), placed three ways: ``lo`` packed into the calls cheapest a row, ``mid`` a request's rows
    (``per_request``) to a call, spread evenly over the calls, ``hi`` one a call in the dearest calls."""
    if rows <= 0 or not steps_:
        return dict.fromkeys(ENDS, 0.0)
    lo, left = 0.0, rows
    for cost, r in sorted((at(table, r, k, 0) / r, r) for r, k in steps_):
        take = min(left, r * calls)
        lo, left = lo + cost * take, left - take
        if left <= 0:
            break
    hi, left = 0.0, math.ceil(rows)
    for cost in sorted((at(table, 1, k, 1) for _, k in steps_), reverse=True):
        take = min(left, calls)
        hi, left = hi + cost * take, left - take
        if left <= 0:
            break
    n = len(steps_) * calls
    mean = sum(at(table, min(per_request, r), k, 0) + at(table, min(per_request, r), k, 1) for r, k in steps_) / 2
    mid = min(max(min(rows / per_request, n) * mean / len(steps_), lo), hi)
    return {"lo": lo / 1e6, "mid": mid / 1e6, "hi": hi / 1e6}


def step_failure_s(failed: float, rows: int, keys: int, table: dict, per_request: int, read: int | None = None,
                   calls: int = CALLS) -> dict:
    """The seconds one step's ``failed`` rows add on a card, its calls each over ``rows`` rows of at most ``keys``:
    placed in as few calls as hold them, a request's rows (``per_request``) to a call, and one a call, the bounds and
    the point; in ``read`` calls where the counts by step say how many failed."""
    if failed <= 0 or rows <= 0:
        return dict.fromkeys(ENDS, 0.0)
    fewest = min(calls, math.ceil(failed / rows))
    most = max(fewest, min(calls, math.ceil(failed)))
    if read:
        fewest = most = min(calls, max(read, fewest))
    mid_n = min(max(math.ceil(failed / per_request), fewest), most)

    def cost(n: int, end: int) -> float:
        return n * at(table, failed / n, keys, end)

    lo = min(cost(fewest, 0), cost(most, 0))
    hi = max(cost(fewest, 1), cost(most, 1))
    mid = min(max((cost(mid_n, 0) + cost(mid_n, 1)) / 2, lo), hi)
    return {"lo": lo / 1e6, "mid": mid / 1e6, "hi": hi / 1e6}


def kind_seconds(trace: Path, windows: list[dict], per_request: int) -> dict:
    """The seconds between a step's end and the one before, summed over the steps in the counted ``windows`` that run
    a prefill (a row of more than ``per_request`` tokens) and over those that don't, with their numbers."""
    out = {"prefill": 0.0, "decode": 0.0, "prefill_steps": 0, "decode_steps": 0}
    prev = None
    with gzip.open(trace, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] != "step":
                continue
            if prev is not None and any(w["t0"] <= r["t"] <= w["t1"] for w in windows):
                k = "prefill" if any(m > per_request for _, _, m in r["rows"]) else "decode"
                out[k] += r["t"] - prev
                out[f"{k}_steps"] += 1
            prev = r["t"]
    return out


def trace_of(d: Path) -> Path:
    traces = sorted((d / "server").glob("step_trace.*.jsonl.gz"))
    if not traces:
        raise SystemExit(f"{d.name}: no step trace, so no schedule to price")
    return traces[0]


def added(x: dict, y: dict) -> dict:
    return {e: x[e] + y[e] for e in ENDS}


def lifetime(d: Path, ref: Path, u: dict, counters: dict | None, pu: dict | None = None) -> dict:
    """One lifetime of ours priced over the same lifetime of the reference: its added seconds on decode steps (base
    and failed, each lo, mid, hi), on prefill steps where ``pu`` gives their unit times (else None, and why), the
    prefill steps' measured added seconds, and the reference's counted seconds."""
    meta = json.loads((d / "meta.json").read_text())
    argv = meta.get("server_argv") or []
    cards = int(argv[argv.index("--tensor-parallel-size") + 1]) if "--tensor-parallel-size" in argv else 1
    mode = (meta.get("env") or {}).get("VERITY_DSA_TOPK", "")
    if mode not in PASSES:
        raise SystemExit(f"{d.name}: its server ran no canonical top-k (VERITY_DSA_TOPK={mode!r})")
    per = by_step(d)
    names = set().union(*(c for s in per.values() for c in s["cards"].values())) if per else set(counters or ())
    have = [k for k in REMEDY if k in names]
    p = PASSES[mode] if "rows_refined" in have or mode != "canonical" else "present"
    all_steps, per_request = steps(trace_of(d))
    windows = counted(d)
    mine = [s for s in all_steps if any(w["t0"] <= s["t"] <= w["t1"] for w in windows)]
    decode = [s for s in mine if s["rows"] and not s["prefill_rows"]]
    prefill = [s for s in mine if s["prefill_rows"]]
    calls = [(s["rows"], s["keys"]) for s in mine if s["rows"]]
    base = {e: CALLS * sum(at(u[p]["base"], r, k, i) for r, k in calls) / 1e6 for e, i in (("lo", 0), ("hi", 1))}
    base["mid"] = (base["lo"] + base["hi"]) / 2
    failed = dict.fromkeys(ENDS, 0.0)
    pre_failed = dict.fromkeys(ENDS, 0.0)
    rows_failed: dict = {}
    pre_rows_failed: dict = {}
    unjoined = share = None
    if per:
        unjoined = 0
        for s in mine:
            at_ = per.get(s["step"])
            if not at_ or at_["tokens"] != {s["T"]}:
                unjoined += 1
                continue
            for k in have:
                slowest = max(at_["cards"].values(), key=lambda c: c.get(k, 0))
                n = slowest.get(k, 0)
                if not n:
                    continue
                if s["prefill_rows"]:
                    pre_rows_failed[k] = pre_rows_failed.get(k, 0) + n
                    if pu:
                        pre_failed = added(pre_failed, step_failure_s(
                            n, s["prefill_rows"], s["prefill_keys"], pu[REMEDY[k]]["failed"], per_request,
                            slowest.get(CALLS_OF[k])))
                else:
                    rows_failed[k] = rows_failed.get(k, 0) + n
                    failed = added(failed, step_failure_s(n, s["rows"], s["keys"], u[REMEDY[k]]["failed"],
                                                          per_request, slowest.get(CALLS_OF[k])))
    else:
        share = sum(r for r, _ in calls) / max(1, sum(s["rows"] for s in all_steps))
        for k in have:
            rows_failed[k] = counters[k] / cards * share
            failed = added(failed, failure_s(calls, rows_failed[k], u[REMEDY[k]]["failed"], per_request))
    priced_prefill = None
    if pu and prefill:
        pb = {e: CALLS * sum(at(pu[p]["base"], s["prefill_rows"], s["prefill_keys"], i) for s in prefill) / 1e6
              for e, i in (("lo", 0), ("hi", 1))}
        pb["mid"] = (pb["lo"] + pb["hi"]) / 2
        priced_prefill = added(pb, pre_failed)
    ref_windows = counted(ref)
    ref_s = sum(w["seconds"] for w in ref_windows)
    ours_k = kind_seconds(trace_of(d), windows, per_request)
    ref_k = (kind_seconds(trace_of(ref), ref_windows, per_request)
             if any((ref / "server").glob("step_trace.*.jsonl.gz")) else None)
    pre_measured = ours_k["prefill"] - ref_k["prefill"] if ref_k else None
    priced = added(base, failed) if priced_prefill is None else added(added(base, failed), priced_prefill)
    return {"dir": d.name, "reference": ref.name, "pass": p, "cards": cards, "decode_steps": len(decode),
            "prefill_steps": len(prefill), "counts_by_step": per is not None, "unjoined_steps": unjoined,
            "counted_share": share, "rows_failed_a_card": rows_failed, "prefill_rows_failed_a_card": pre_rows_failed,
            "failed_priced": bool(have), "rows_at": None if per else counters and counters.get("rows_at"),
            "calls_kept": None if per else counters and counters.get("calls"),
            "base_s": base, "failed_s": failed, "prefill_s": priced_prefill,
            "prefill_measured_s": pre_measured, "reference_s": ref_s,
            "cost": {e: priced[e] / ref_s for e in ENDS} if ref_s else None}


def interval(ys: list[float]) -> tuple[float, list[float]] | None:
    """The mean of ``ys`` and its 95% interval over them (Student's t), or None for fewer than two."""
    if len(ys) < 2:
        return None
    m = sum(ys) / len(ys)
    se = math.sqrt(sum((y - m) ** 2 for y in ys) / (len(ys) - 1) / len(ys))
    t = T975[min(len(ys) - 2, len(T975) - 1)]
    return m, [m - t * se, m + t * se]


def layer(out: Path, ours: str, reference: str, u: dict, pu: dict | None = None) -> dict:
    """``ours`` over ``reference`` priced, a lifetime pair at a time, and the mean of the pairs' costs (lo, mid, hi);
    with the prefill steps priced where ``pu`` gives their unit times, else refused, and their measured added
    seconds' share of the reference's beside, with its interval over the pairs."""
    counters = {}
    if (out / "topk_counters.json").is_file():
        counters = json.loads((out / "topk_counters.json").read_text()).get("dirs") or {}
    mine, refs = lifetimes(out, ours), lifetimes(out, reference)
    if not mine or len(mine) != len(refs):
        raise SystemExit(f"{ours} has {len(mine)} lifetimes and {reference} {len(refs)}: no pairs to price")
    each = [lifetime(d, r, u, counters.get(d.name), pu) for d, r in zip(mine, refs)]
    costs = [x["cost"] for x in each if x["cost"]]
    pre = [x["prefill_measured_s"] / x["reference_s"] for x in each
           if x["prefill_measured_s"] is not None and x["reference_s"]]
    pre_ci = interval(pre)
    return {"ours": ours, "over": reference, "pairs": len(costs), "each": each,
            "failed_priced": all(x["failed_priced"] for x in each),
            "counts_by_step": all(x["counts_by_step"] for x in each),
            "prefill_priced": pu is not None,
            "prefill_refused": None if pu else "no prefill unit time a call (--prefill-units)",
            "prefill_measured": {"value": sum(pre) / len(pre), "ci95": pre_ci[1] if pre_ci else None,
                                 "pairs": len(pre), "mark": "measured"} if pre else None,
            "cost": {e: sum(c[e] for c in costs) / len(costs) for e in ENDS} if costs else None}


def adjusted(out: Path, ours: str, over: str) -> dict | None:
    """The job's measured cost of ``ours`` over ``over`` with each pass's seconds adjusted for its engine steps: each
    lifetime pair's mean over its counted passes, paired by rep, of ln(seconds ratio) less b × ln(steps ratio), b the
    slope of ln seconds on ln steps within each arm over the job's counted passes; mean and interval over the pairs,
    marked modelled (b is fitted). None where the report holds no passes with steps for both arms."""
    f = out / "prod_repro.json"
    if not f.is_file():
        return None
    passes = [p for p in ((json.loads(f.read_text()).get("ran") or {}).get("passes") or {}).get("passes") or ()
              if p.get("seconds") and p.get("steps") and not p.get("left_out")]
    by: dict = {}
    for p in passes:
        by.setdefault(p["arm"], []).append(p)
    if ours not in by or over not in by:
        return None
    xs, ys = [], []
    for ps in by.values():
        mx = sum(math.log(p["steps"]) for p in ps) / len(ps)
        my = sum(math.log(p["seconds"]) for p in ps) / len(ps)
        xs += [math.log(p["steps"]) - mx for p in ps]
        ys += [math.log(p["seconds"]) - my for p in ps]
    if not sum(x * x for x in xs):
        return None
    b = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    key = {(p["arm"], p["dir"].rsplit(".", 1)[-1] if p["dir"].rsplit(".", 1)[-1].isdigit() else "", p["rep"]): p
           for p in passes}
    pairs: dict = {}
    for (arm, life, rep), p in key.items():
        q = key.get((over, life, rep))
        if arm == ours and q:
            pairs.setdefault(life, []).append(math.log(p["seconds"] / q["seconds"]) - b * math.log(p["steps"] / q["steps"]))
    got = interval([sum(v) / len(v) for v in pairs.values()])
    if not got:
        return None
    m, ci = got
    return {"value": math.expm1(m), "ci95": [math.expm1(ci[0]), math.expm1(ci[1])], "pairs": len(pairs),
            "slope": b, "mark": "modelled"}


def measured(out: Path, ours: str, over: str) -> dict | None:
    """The job's own measured cost of ``ours`` over ``over``: its report's layer (`prod_repro.layers`), or its cost
    line where those are its two arms; None where the report holds neither."""
    f = out / "prod_repro.json"
    if not f.is_file():
        return None
    r = json.loads(f.read_text())
    for x in (r.get("layers") or {}).get("each") or ():
        if (x["arm"], x["over"]) == (ours, over) and x.get("value") is not None:
            return {"value": x["value"], "ci95": x.get("ci95"), "pairs": x.get("pairs"), "run": r.get("run")}
    c, a = r.get("cost") or {}, r.get("arms") or {}
    if (a.get("ours"), a.get("stock")) == (ours, over) and c.get("measured"):
        life = c.get("lifetimes") or {}
        return {"value": c["value"], "ci95": life.get("ci95"), "pairs": life.get("pairs"), "run": r.get("run")}
    return None


def point(priced: dict, below: dict, tested_by: str, units_from: str, member: str) -> dict:
    """The point under `METRIC`: ours over stock, the priced layer plus the measured layers below it (``below``,
    ours's reference over stock), marked modeled; its interval the sum of the priced bounds and the measured interval.
    Where the prefill steps were not priced, their measured added seconds are a term of it, named as measured."""
    c = priced["cost"]
    lo, hi = c["lo"], c["hi"]
    val = c["mid"]
    counts = "the top-k's counts by step" if priced["counts_by_step"] else "the top-k's counters' lifetime totals"
    inputs = [f"modeled: {priced['ours']} over {priced['over']} priced from the step trace's rows and {counts} at "
              f"{units_from}'s device times a call, over {priced['pairs']} lifetime pairs"
              + ("" if priced["prefill_priced"] else ", its decode steps alone"),
              f"tested by run {tested_by}, which measures the same arms"]
    pm = priced.get("prefill_measured")
    if not priced["prefill_priced"] and pm and pm.get("ci95"):
        val, lo, hi = val + pm["value"], lo + pm["ci95"][0], hi + pm["ci95"][1]
        inputs.append(f"measured, not priced: {priced['ours']}'s prefill steps over {priced['over']}'s, "
                      f"{100 * pm['value']:+.2f}% from both arms' step traces in run {tested_by}")
    val, lo, hi = val + below["value"], lo + below["ci95"][0], hi + below["ci95"][1]
    inputs.append(f"measured: {priced['over']} over stock, {100 * below['value']:+.2f}% in run {below['run']}")
    return {"name": METRIC, "value": 100 * val, "unit": UNIT, "mark": "modeled", "member": member,
            "ci95": [100 * lo, 100 * hi], "inputs": inputs, "tested_by": tested_by}


def text(r: dict) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:+.2f}%"

    lines = []
    for ly in r["layers"]:
        c = ly["cost"]
        lines.append(f"{ly['ours']} over {ly['over']}, priced: {pct(c['mid'])} [{pct(c['lo'])}, {pct(c['hi'])}] over "
                     f"{ly['pairs']} lifetime pairs (modelled), from "
                     + ("the top-k's counts by step" if ly["counts_by_step"] else
                        "the counters' lifetime totals, a lower bound by the rows read last")
                     + ("" if ly["failed_priced"] else "; failed rows not priced: the counters hold none")
                     + ("" if ly["prefill_priced"] else f"; prefill steps not priced: {ly['prefill_refused']}"))
        for x in ly["each"]:
            fr = ", ".join(f"{k} {v:,.0f}" for k, v in x["rows_failed_a_card"].items()) or "no failed rows counted"
            pre = (f"; prefill {x['prefill_s']['lo']:.2f}–{x['prefill_s']['hi']:.2f} s" if x["prefill_s"] else "")
            pf = ", ".join(f"{k} {v:,.0f}" for k, v in x["prefill_rows_failed_a_card"].items())
            lines.append(f"  {x['dir']} ({x['pass']}): {x['decode_steps']:,} decode and {x['prefill_steps']:,} prefill "
                         f"steps counted; base {x['base_s']['mid']:.2f} s, failed {x['failed_s']['lo']:.2f}–"
                         f"{x['failed_s']['hi']:.2f} s a card ({fr} a card on decode steps"
                         + (f", {pf} on prefill steps" if pf else "") + f"){pre}; over {x['reference_s']:.1f} s of "
                         f"{x['reference']}"
                         + (f"; {x['unjoined_steps']} counted steps not in the counts by step" if x["unjoined_steps"]
                            else ""))
        pm = ly.get("prefill_measured")
        if pm:
            ends = f" [{pct(pm['ci95'][0])}, {pct(pm['ci95'][1])}]" if pm.get("ci95") else ""
            lines.append(f"  prefill steps, measured from both arms' step traces (not priced): {pct(pm['value'])}{ends} "
                         f"over {pm['pairs']} pairs")
        m = ly.get("measured")
        if m:
            ends = f" [{pct(m['ci95'][0])}, {pct(m['ci95'][1])}]" if m.get("ci95") else ""
            lines.append(f"  measured in run {m['run']}: {pct(m['value'])}{ends} over {m['pairs']} pairs; priced less "
                         f"measured {100 * (c['mid'] - m['value']):+.2f} points")
        a = ly.get("adjusted")
        if a:
            lines.append(f"  measured, each pass adjusted for its engine steps (modelled, slope {a['slope']:.2f}): "
                         f"{pct(a['value'])} [{pct(a['ci95'][0])}, {pct(a['ci95'][1])}] over {a['pairs']} pairs")
    if r.get("point"):
        p = r["point"]
        lines.append(f"point {p['name']} ({p['member']}, modeled): {p['value']:+.2f}% "
                     f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}], tested by {p['tested_by']}")
    else:
        lines.append(f"no point: {r['no_point']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--prefill-units", type=Path)
    ap.add_argument("--reference", default="prod-trace")
    ap.add_argument("--stock", default="prod")
    ap.add_argument("--ours", nargs="+", default=["prod-record", "prod-record-bisect"])
    ap.add_argument("--tested-by")
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--card", default="h200x8")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--result", action="store_true")
    a = ap.parse_args(argv)
    u = units(a.units)
    pu = units(a.prefill_units) if a.prefill_units else None
    rep = json.loads((a.out / "prod_repro.json").read_text()) if (a.out / "prod_repro.json").is_file() else {}
    tested_by = a.tested_by or rep.get("run") or "not named"
    layers = []
    for ours in a.ours:
        if not lifetimes(a.out, ours):
            continue
        ly = layer(a.out, ours, a.reference, u, pu)
        ly["measured"] = measured(a.out, ours, a.reference)
        ly["adjusted"] = adjusted(a.out, ours, a.reference)
        layers.append(ly)
    if not layers:
        raise SystemExit(f"{a.out}: none of {', '.join(a.ours)} was served")
    first = layers[0]
    member = f"{a.model}-{a.card}-{first['ours']}-over-{a.stock}-{INSTRUMENT}"
    below = measured(a.out, a.reference, a.stock)
    why = ("no lifetime pair was priced" if not first["cost"] else
           "the counters hold no failed rows, so only the pass's base is priced" if not first["failed_priced"] else
           f"the job measured no {a.reference} over {a.stock} with an interval" if not (below and below.get("ci95"))
           else None)
    r = {"layers": layers, "point": None if why else point(first, below, tested_by, a.units.name, member),
         **({"no_point": why} if why else {})}
    (a.out / "priced.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1) if a.json else text(r))
    if a.result and r["point"]:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
        from verity_bench import kit

        p = r["point"]
        kit.writer(a.out).measurement(p["name"], p["value"], p["unit"], p["ci95"], inputs=p["inputs"], mark="modeled",
                                      member=p["member"], extra={"tested_by": p["tested_by"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
