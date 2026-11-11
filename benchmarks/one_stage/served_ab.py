"""`served_ab.sh`'s records and table: what each part of the served step costs, arm by arm.

    python benchmarks/one_stage/served_ab.py life SERVE_DIR --arm ARM --round ROUND [--started EPOCH_S]
    python benchmarks/one_stage/served_ab.py table OUT [--out DIR]

`life` runs on the node, over one lifetime's serve dir, and prints its record: `served_overhead.py`'s aggregate, each
load's step times (`served_overhead.serve_dir`'s ``step_s``: the first runs from the load's first arrival to its first
step's output, so it holds the Commit window's open) and each window's and seal's timed parts, numbers alone
(`numbers`). `served_ab.sh`'s load is synthetic and its class public, so the step times leave the node here, as a served
window's never do. With ``--started`` (the node's clock when the server was started) it holds ``listen_s``, the seconds
from there to the endpoint listening (``serve.json``'s ``clocks.time``, written as the server begins to listen): what an
arm does before it listens is paid there, not in a load.

`table` reads every ``OUT/rep-<i>/<arm>.json`` (a warm round's is left out) and, for the first load after each endpoint
listens and for the later, warm loads apart, gives each arm's median, least and greatest over its loads of:

- ``wall_s``: the load's span, its first arrival to its last step's output (the window: every prompt and output token);
- ``first_token_s``: the mean over its requests of arrival to the output of the step that emitted its first token;
- ``decode_step_s``: the mean of its steps after the first;
- ``close_after_s``: the on-loop part of the window's close after its last step, when the engine can't step.

Beside the first loads it gives ``listen_s`` over the arm's lifetimes, where every one recorded it: a cost moved from a
load to the server's start shows there.

Arms are ordered stock, commit, pearl, live, ahead, each one part more than the one before (`PARTS`), so a part's cost is
its arm's median less the arm before's (less than nothing where the part saves): in seconds, as a share of the last arm's
wall, and over stock's. `table` writes
``table.json`` and ``table.md`` to ``--out`` and prints the Markdown.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

LIFE = "verity-bench/served-ab-life/v0"
TABLE = "verity-bench/served-ab-table/v0"
ORDER = ("stock", "commit", "pearl", "live", "ahead")
#: what each arm serves beyond the arm before it
PARTS = {"commit": "the Commit", "pearl": "Pearl+ under the Commit, its coins in-process",
         "live": "each window's salt coin from a live session",
         "ahead": "that session reserved and opened ahead of its window"}
MEASURES = ("wall_s", "first_token_s", "decode_step_s", "close_after_s")
HEADS = {"wall_s": "window wall, s", "first_token_s": "to the first token, s", "decode_step_s": "a decode step, s",
         "close_after_s": "on-loop close after, s"}
#: a lifetime's seconds from the server's start to its endpoint listening, and its column's head
LISTEN, LISTEN_HEAD = "listen_s", "server start to listening, s"


def numbers(x: Any) -> Any:
    """``x`` with only its numbers: a dict keeps the keys whose values hold one, a list of numbers stays, the rest is None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, dict):
        kept = {k: numbers(v) for k, v in x.items()}
        return {k: v for k, v in kept.items() if v is not None} or None
    if isinstance(x, list) and x and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x):
        return x
    return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def life(d: Path, arm: str, round_: str, started: float | None = None) -> dict[str, Any]:
    """One lifetime's record from its serve dir ``d`` (module docstring), the server started at ``started`` on the node's
    clock; `ValueError` as `served_overhead.serve_dir`'s."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import served_overhead as SO

    s, agg = SO.serve_dir(d), SO.aggregate(d)
    windows = []
    for w in _jsonl(d / SO.WINDOWS):
        pearl = w.get("pouw_pearl") or {}
        windows.append({"window": w.get("window"), "steps": w.get("steps"), "passed": w.get("passed"), "failed": "error" in w,
                        "parts": numbers({k: w.get(k) for k in ("seconds", "close_phases_s", "on_loop_phases_s", "commit_s")}),
                        "pearl": {"seconds": numbers(pearl.get("seconds")), "passes": pearl.get("passes"),
                                  "calls": pearl.get("calls"), "coin_source": pearl.get("coin_source"),
                                  "failed": "error" in pearl} if pearl else None})
    seals = [numbers({k: x.get(k) for k in ("window", "seal_s", "phases_s")}) for x in _jsonl(d / "seals.jsonl")]
    serve = json.loads((d / SO.SERVE).read_text())
    protocols = serve.get("protocols")
    listened = numbers(serve.get("clocks", {}).get("time"))
    return {"schema": LIFE, "arm": arm, "round": round_, "aggregate": agg,
            LISTEN: None if started is None or listened is None else listened - started,
            "loads": [{k: ld[k] for k in (*SO.PUBLIC_LOAD, "step_s") if k in ld} for ld in s["loads"]],
            "windows": windows, "seals": seals,
            "ready": numbers(protocols.get("ready")) if isinstance(protocols, dict) else None}


def measures(load: dict[str, Any]) -> dict[str, float]:
    """One load's `MEASURES`; `ValueError` for a load of one step, which has no decode step."""
    steps = load["step_s"]
    if len(steps) < 2:
        raise ValueError(f"a load of {len(steps)} step(s) has no decode step")
    return {"wall_s": float(load["span_s"]), "first_token_s": float(load["time_to_first_token_s"]),
            "decode_step_s": statistics.fmean(steps[1:]),
            "close_after_s": float((load.get("windows") or {}).get("close_on_loop_after_s") or 0.0)}


def summary(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def _kind(by_arm: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    """One kind of load (first or warm) over the arms that have one: each arm's summaries, and each part's cost."""
    arms = [a for a in ORDER if by_arm.get(a)]
    out: dict[str, Any] = {"arms": {a: {m: summary([x[m] for x in by_arm[a]]) for m in MEASURES} for a in arms}, "parts": []}
    for a in arms:
        if all(LISTEN in x for x in by_arm[a]):
            out["arms"][a][LISTEN] = summary([x[LISTEN] for x in by_arm[a]])
    if not arms:
        return out
    med = lambda a, m: out["arms"][a][m]["median"]   # noqa: E731
    top, base = arms[-1], arms[0]
    out["top"], out["base"] = top, base
    if base == "stock":
        for a in arms:
            out["arms"][a]["over_stock"] = med(a, "wall_s") / med("stock", "wall_s")
        out["parts"].append({"part": "the stock step", "arm": "stock", "wall_s": med("stock", "wall_s"),
                             "share_of_top": med("stock", "wall_s") / med(top, "wall_s"), "over_stock": 1.0,
                             "first_token_s": med("stock", "first_token_s"), "decode_step_s": med("stock", "decode_step_s")})
    for prev, a in zip(arms, arms[1:]):
        if ORDER.index(a) != ORDER.index(prev) + 1:
            continue                              # an arm left out between them: their difference is more than one part
        add = med(a, "wall_s") - med(prev, "wall_s")
        out["parts"].append({"part": PARTS[a], "arm": a, "wall_s": add, "share_of_top": add / med(top, "wall_s"),
                             "over_stock": add / med("stock", "wall_s") if base == "stock" else None,
                             "first_token_s": med(a, "first_token_s") - med(prev, "first_token_s"),
                             "decode_step_s": med(a, "decode_step_s") - med(prev, "decode_step_s")})
    return out


def table(out: Path) -> dict[str, Any]:
    """The table over ``out``'s rep rounds (module docstring); `ValueError` when no round left a record, or when two
    lifetimes' loads differ in requests, tokens or steps."""
    first: dict[str, list[dict[str, float]]] = {}
    warm: dict[str, list[dict[str, float]]] = {}
    shape, rounds, failed = set(), set(), []
    for f in sorted(out.glob("rep-*/*.json")):
        doc = json.loads(f.read_text())
        if doc.get("schema") != LIFE:
            continue
        arm = doc["arm"]
        rounds.add(doc["round"])
        failed += [f"{doc['round']} {arm} window {w['window']}" for w in doc["windows"]
                   if w["failed"] or w["passed"] is False or (w["pearl"] or {}).get("failed")]
        for i, ld in enumerate(doc["loads"]):
            shape.add((ld["requests"], ld["tokens"], ld["steps"]))
            listen = {LISTEN: doc[LISTEN]} if i == 0 and doc.get(LISTEN) is not None else {}
            (first if i == 0 else warm).setdefault(arm, []).append({**measures(ld), **listen})
    if not first:
        raise ValueError(f"{out}: no rep-<i>/<arm>.json record")
    if len(shape) != 1:
        raise ValueError(f"{out}: loads of different (requests, tokens, steps): {sorted(shape)}")
    r, t, s = next(iter(shape))
    return {"schema": TABLE, "load": {"requests": r, "tokens": t, "steps": s}, "rounds": len(rounds), "failed_windows": failed,
            "first": _kind(first), "warm": _kind(warm)}


def _cell(x: dict[str, Any]) -> str:
    return f"{x['median']:.4g} ({x['min']:.4g} to {x['max']:.4g})"


def _pct(x: float | None) -> str:
    return "" if x is None else f"{100 * x:.3g}%"


def markdown(t: dict[str, Any]) -> str:
    ld = t["load"]
    lines = [f"The served step arm by arm: a load of {ld['requests']} requests, {ld['tokens']} output tokens and {ld['steps']} "
             f"steps, over {t['rounds']} round(s). Each cell is the median over the arm's loads (least to greatest).", ""]
    if t["failed_windows"]:
        lines += [f"Windows that failed or didn't pass: {', '.join(t['failed_windows'])}.", ""]
    for key, title in (("first", "The first load after the endpoint listens"), ("warm", "Later loads, warm")):
        k = t[key]
        if not k["arms"]:
            continue
        listen = any(LISTEN in row for row in k["arms"].values())
        lines += [f"**{title}**", "", "| arm | loads | " + " | ".join(HEADS[m] for m in MEASURES) + " | × stock |"
                  + (f" {LISTEN_HEAD} |" if listen else ""), "|---|---|" + "---|" * (len(MEASURES) + 1 + listen)]
        for a, row in k["arms"].items():
            over = f"{row['over_stock']:.3g}×" if "over_stock" in row else ""
            lines.append(f"| {a} | {row['wall_s']['n']} | " + " | ".join(_cell(row[m]) for m in MEASURES) + f" | {over} |"
                         + (f" {_cell(row[LISTEN]) if LISTEN in row else ''} |" if listen else ""))
        if k["parts"]:
            lines += ["", f"| part | adds to the wall, s | share of the {k['top']} arm's wall | over stock's wall | "
                          "adds to the first token, s | adds to a decode step, s |", "|---|---|---|---|---|---|"]
            lines += [f"| {p['part']} | {p['wall_s']:.4g} | {_pct(p['share_of_top'])} | {_pct(p['over_stock'])} | "
                      f"{p['first_token_s']:.4g} | {p['decode_step_s']:.4g} |" for p in k["parts"]]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("life")
    p.add_argument("serve_dir", type=Path)
    p.add_argument("--arm", required=True, choices=ORDER)
    p.add_argument("--round", required=True)
    p.add_argument("--started", type=float, default=None)
    p = sub.add_parser("table")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "life":
            print(json.dumps(life(a.serve_dir, a.arm, a.round, a.started)))
            return 0
        t = table(a.out_dir)
    except (ValueError, KeyError, OSError) as e:
        print(f"served_ab.py {a.cmd}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    md = markdown(t)
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "table.json").write_text(json.dumps(t, indent=1) + "\n")
        (a.out / "table.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
