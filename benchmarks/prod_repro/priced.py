"""The canonical top-k's serving cost priced from counts: a job's schedule and top-k counters times the top-k's device
time a call, in percent of the reference arm's serving time. It needs no GPU.

    priced.py OUT --units TOPK_CUT.jsonl [--reference prod-trace] [--stock prod] [--ours ARM ...] [--tested-by RUN]
              [--json] [--result]

A decode step makes `CALLS` top-k calls on each card, each over the step's decode rows past `K` keys. A call costs the
pass's time with no failed row (`units`' ``base``, by rows and the longest row), and one with a failed row costs
``failed`` instead: a bisected call is set by its longest row, a refined one by its failed rows. The rows and their keys
are the step trace's (from the schedule); the failed rows are the counters' (from the model's outputs), totals over a
lifetime, so the reader places them in calls three ways, packed, a request's rows together, and one a call, the bounds
and the point. Prefill steps are not priced: no device times at prefill shapes.
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
ENDS = ("lo", "mid", "hi")


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
    """The step trace's steps, each with its ``t`` and its decode rows past `K` keys (``rows``) and their most keys
    (``keys``), and the rows a decode request runs a step (1 and its drafts, from the trace's header)."""
    out, n = [], draft_rows
    with gzip.open(trace, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "header" and n is None:
                n = 1 + ((r.get("config") or {}).get("speculative") or {}).get("num_speculative_tokens", 0)
            if r["kind"] != "step":
                continue
            rows = keys = 0
            for _, start, m in r["rows"]:
                if m <= (n or 1):
                    rows += max(0, start + m - max(start, K))
                    keys = max(keys, start + m)
            out.append({"t": r["t"], "rows": rows, "keys": keys if rows else 0})
    return out, n or 1


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


def lifetime(d: Path, ref: Path, u: dict, counters: dict | None) -> dict:
    """One lifetime of ours priced over the same lifetime of the reference: its added seconds (base and failed, each
    lo, mid, hi) and the reference's counted seconds."""
    meta = json.loads((d / "meta.json").read_text())
    argv = meta.get("server_argv") or []
    cards = int(argv[argv.index("--tensor-parallel-size") + 1]) if "--tensor-parallel-size" in argv else 1
    mode = (meta.get("env") or {}).get("VERITY_DSA_TOPK", "")
    if mode not in PASSES:
        raise SystemExit(f"{d.name}: its server ran no canonical top-k (VERITY_DSA_TOPK={mode!r})")
    have = [k for k in REMEDY if counters and k in counters]
    p = PASSES[mode] if "rows_refined" in have or mode != "canonical" else "present"
    traces = sorted((d / "server").glob("step_trace.*.jsonl.gz"))
    if not traces:
        raise SystemExit(f"{d.name}: no step trace, so no schedule to price")
    all_steps, per_request = steps(traces[0])
    windows = counted(d)
    mine = [s for s in all_steps if s["rows"] and any(w["t0"] <= s["t"] <= w["t1"] for w in windows)]
    calls = [(s["rows"], s["keys"]) for s in mine]
    base = {e: CALLS * sum(at(u[p]["base"], r, k, i) for r, k in calls) / 1e6
            for e, i in (("lo", 0), ("hi", 1))}
    base["mid"] = (base["lo"] + base["hi"]) / 2
    share = sum(r for r, _ in calls) / max(1, sum(s["rows"] for s in all_steps))
    failed = dict.fromkeys(ENDS, 0.0)
    rows_failed = {}
    for k in have:
        rows_failed[k] = counters[k] / cards * share
        f = failure_s(calls, rows_failed[k], u[REMEDY[k]]["failed"], per_request)
        failed = {e: failed[e] + f[e] for e in ENDS}
    ref_s = sum(w["seconds"] for w in counted(ref))
    return {"dir": d.name, "reference": ref.name, "pass": p, "cards": cards, "decode_steps": len(mine),
            "counted_share": share, "rows_failed_a_card": rows_failed,
            "failed_priced": bool(have), "rows_at": counters and counters.get("rows_at"),
            "calls_kept": counters and counters.get("calls"),
            "base_s": base, "failed_s": failed, "reference_s": ref_s,
            "cost": {e: (base[e] + failed[e]) / ref_s for e in ENDS} if ref_s else None}


def layer(out: Path, ours: str, reference: str, u: dict) -> dict:
    """``ours`` over ``reference`` priced, a lifetime pair at a time, and the mean of the pairs' costs (lo, mid, hi)."""
    counters = {}
    if (out / "topk_counters.json").is_file():
        counters = json.loads((out / "topk_counters.json").read_text()).get("dirs") or {}
    mine, refs = lifetimes(out, ours), lifetimes(out, reference)
    if not mine or len(mine) != len(refs):
        raise SystemExit(f"{ours} has {len(mine)} lifetimes and {reference} {len(refs)}: no pairs to price")
    each = [lifetime(d, r, u, counters.get(d.name)) for d, r in zip(mine, refs)]
    costs = [x["cost"] for x in each if x["cost"]]
    return {"ours": ours, "over": reference, "pairs": len(costs), "each": each,
            "failed_priced": all(x["failed_priced"] for x in each),
            "cost": {e: sum(c[e] for c in costs) / len(costs) for e in ENDS} if costs else None}


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
    ours's reference over stock), marked modeled; its interval the sum of the priced bounds and the measured interval."""
    c = priced["cost"]
    lo, hi = c["lo"], c["hi"]
    val = c["mid"]
    inputs = [f"modeled: {priced['ours']} over {priced['over']} priced from the step trace's rows and the top-k's "
              f"counters at {units_from}'s device times a call, over {priced['pairs']} lifetime pairs",
              f"tested by run {tested_by}, which measures the same arms"]
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
                     f"{ly['pairs']} lifetime pairs (modelled)"
                     + ("" if ly["failed_priced"] else ", failed rows not priced: the counters hold none"))
        for x in ly["each"]:
            fr = ", ".join(f"{k} {v:,.0f}" for k, v in x["rows_failed_a_card"].items()) or "no failed rows counted"
            lines.append(f"  {x['dir']} ({x['pass']}): {x['decode_steps']:,} decode steps counted; base "
                         f"{x['base_s']['mid']:.2f} s, failed {x['failed_s']['lo']:.2f}–{x['failed_s']['hi']:.2f} s a "
                         f"card ({fr} a card); over {x['reference_s']:.1f} s of {x['reference']}")
        m = ly.get("measured")
        if m:
            ends = f" [{pct(m['ci95'][0])}, {pct(m['ci95'][1])}]" if m.get("ci95") else ""
            lines.append(f"  measured in run {m['run']}: {pct(m['value'])}{ends} over {m['pairs']} pairs; priced less "
                         f"measured {100 * (c['mid'] - m['value']):+.2f} points")
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
    rep = json.loads((a.out / "prod_repro.json").read_text()) if (a.out / "prod_repro.json").is_file() else {}
    tested_by = a.tested_by or rep.get("run") or "not named"
    layers = []
    for ours in a.ours:
        if not lifetimes(a.out, ours):
            continue
        ly = layer(a.out, ours, a.reference, u)
        ly["measured"] = measured(a.out, ours, a.reference)
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
