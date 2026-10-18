"""Tables from results.json files: per (model, set, context), each arm's perplexity (the mean over its seeds, with
their range), its change against bf16 and against its scope's plain int8 arm, and the significance rule's verdict
against that arm (methodology.compare). A results file whose gates failed holds no perplexities, and its table says so."""
from __future__ import annotations

import json
from pathlib import Path

import methodology


def load(paths) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in paths]


def arm_stats(r: dict, arm: str) -> dict | None:
    a = r["arms"].get(arm)
    if a is None or "seeds" not in a:
        return None
    seeds = a["seeds"]
    ppls = [s["ppl"] for s in seeds.values()]
    wins = [sum(s["window_nll"][w] for s in seeds.values()) / len(seeds) for w in range(len(next(iter(seeds.values()))["window_nll"]))]
    return {"ppl": sum(ppls) / len(ppls), "min": min(ppls), "max": max(ppls), "n": len(ppls), "seeds": ppls,
            "windows": wins, "plain": a.get("plain", "")}


def verdict(r: dict, a: str, b: str) -> str:
    x, y = arm_stats(r, a), arm_stats(r, b)
    if x is None or y is None:
        return "-"
    return methodology.compare(x["seeds"], y["seeds"], x["windows"], y["windows"])


def fmt_ppl(s: dict) -> str:
    return f"{s['ppl']:.4f}" + (f" ({s['min']:.4f}–{s['max']:.4f})" if s["n"] > 1 else "")


def pct(a: float, b: float, digits: int = 2) -> str:
    """a against b in percent, signed ("+0.00" for anything that rounds to zero)."""
    x = round(100 * (a / b - 1), digits)
    return f"{x + 0.0:+.{digits}f}"


def table(r: dict) -> str:
    head = (f"### {r['model']} · {r['set']} · {r['ctx']} ({r['n_windows']} windows, {r['scored_tokens']:,} tokens)"
            f"{'' if r['passed'] else ' — GATES FAILED: no perplexities'}")
    lines = [head, "", "| arm | seeds | perplexity | Δ% vs bf16 | Δ% vs plain int8 | vs plain int8 |", "|---|---:|---:|---:|---:|---|"]
    if not r["passed"]:
        failed = {n: g["first_failure"] for n, g in r["gates"].items() if g["failed"]}
        return "\n".join([head, "", f"failed gates: {json.dumps(failed)}"])
    bf = arm_stats(r, "bf16")
    for name in r["arms"]:
        s = arm_stats(r, name)
        p = arm_stats(r, s["plain"]) if s["plain"] else None
        db = pct(s["ppl"], bf["ppl"]) if name != "bf16" else "-"
        dp = pct(s["ppl"], p["ppl"]) if p is not None and name != s["plain"] else "-"
        v = verdict(r, name, s["plain"]) if p is not None and name != s["plain"] else "-"
        lines.append(f"| {name} | {s['n']} | {fmt_ppl(s)} | {db} | {dp} | {v} |")
    g = r["gates"]
    lines.append("")
    lines.append("gates: " + ", ".join(f"{n} {v['checked']}" + (f" (skipped: {v['skipped']})" if v.get("skipped") else "")
                                       for n, v in g.items()))
    return "\n".join(lines)


#: the baseline's arms, in its table's order
BASELINE_ARMS = ("bf16", "int8-linears", "int8-attention", "int8-all", "adopted-linears", "adopted-attention",
                 "adopted-no-smooth", "adopted")
#: the baseline's points, in its table's order
BASELINE_POINTS = (("llama-3.1-8b", "c4", 2048), ("llama-3.1-8b", "wikitext2", 2048), ("llama-3.1-8b", "wikitext2", 8192),
                   ("llama-3.1-8b", "wikitext2", 32768), ("qwen2.5-7b", "wikitext2", 2048),
                   ("qwen2.5-7b", "wikitext2", 8192), ("llama-3.1-70b", "c4", 2048))


def point_key(model: str, set_: str, ctx: int) -> str:
    return f"{model}|{set_}|{ctx}"


def lock_point(r: dict) -> dict:
    """One (model, set, context) of the baseline from a passing results.json: each baseline arm's perplexity (mean and
    range over seeds) and its plain arm, with the run's id."""
    if not r["passed"]:
        raise ValueError(f"{r['model']} {r['set']} {r['ctx']}: its gates failed, so it locks nothing")
    arms = {}
    for name in BASELINE_ARMS:
        s = arm_stats(r, name)
        if s is not None:
            arms[name] = {"ppl": s["ppl"], "min": s["min"], "max": s["max"], "n": s["n"], "plain": s["plain"]}
    return {"model": r["model"], "set": r["set"], "ctx": r["ctx"], "windows": r["n_windows"], "run": r["env"]["run"],
            "arms": arms}


def lock(old: dict, results: list[dict]) -> dict:
    """The baseline with each point that `results` measures replaced, in BASELINE_POINTS' order."""
    points = {(p["model"], p["set"], p["ctx"]): p for p in old["points"]}
    for r in results:
        k = (r["model"], r["set"], r["ctx"])
        if k not in BASELINE_POINTS:
            raise ValueError(f"{k} is not a baseline point")
        points[k] = lock_point(r)
    return {**old, "points": [points[k] for k in BASELINE_POINTS if k in points]}


def _range(s: dict) -> str:
    return f"{s['ppl']:.4f}" + (f" ({s['min']:.4f}–{s['max']:.4f})" if s.get("n", 1) > 1 else "")


def baseline_table(lk: dict) -> str:
    """The locked baseline as `run.py tables baseline` prints it: per point and arm, the bench's perplexity, its Δ
    against bf16 and plain int8, tonight's measurement of the same arm, and whether the two reproduce each other
    (methodology.reproduces)."""
    lines = ["| model · set · context | arm | perplexity | Δ% vs bf16 | Δ% vs plain int8 | tonight | tonight's method | "
             "Δ% vs tonight | reproduces |", "|---|---|---:|---:|---:|---:|---|---:|---|"]
    runs = []
    for p in lk["points"]:
        where = f"{p['model']} · {p['set']} · {p['ctx'] // 1024}K ({p['windows']} windows)"
        bf = p["arms"]["bf16"]["ppl"]
        tonight = lk["tonight"].get(point_key(p["model"], p["set"], p["ctx"]), {})
        for name in BASELINE_ARMS:
            s = p["arms"].get(name)
            if s is None:
                continue
            plain = p["arms"].get(s["plain"]) if s["plain"] and s["plain"] != name else None
            db = "-" if name == "bf16" else pct(s["ppl"], bf)
            dp = "-" if plain is None else pct(s["ppl"], plain["ppl"])
            t = tonight.get(name)
            if t is None:
                tcells = "- | - | - | -"
            else:
                same = methodology.reproduces(s["min"], s["max"], t["min"], t["max"])
                how = f"`{t['as']}`" + (f" ({t['note']})" if t.get("note") else "")
                tcells = f"{_range(t)} | {how} | {pct(s['ppl'], t['ppl'])} | {'yes' if same else 'NO'}"
            lines.append(f"| {where} | {name} | {_range(s)} | {db} | {dp} | {tcells} |")
            where = ""
        tr = sorted({t["run"] for t in tonight.values()})
        runs.append(f"- {p['model']} · {p['set']} · {p['ctx'] // 1024}K: bench `{p['run']}`"
                    + (f"; tonight {', '.join(f'`{x}`' for x in tr)}" if tr else ""))
    return "\n".join(lines + ["", "Runs:", ""] + runs)


def baseline_check(lk: dict, results: list[dict]) -> tuple[str, bool]:
    """New results against the locked baseline: each locked arm the results measure, and whether it reproduces."""
    found = {(r["model"], r["set"], r["ctx"]): r for r in results}
    lines = ["| model · set · context | arm | locked | new | Δ% | reproduces |", "|---|---|---:|---:|---:|---|"]
    ok_all, seen = True, 0
    for p in lk["points"]:
        r = found.get((p["model"], p["set"], p["ctx"]))
        if r is None:
            continue
        if not r["passed"]:
            lines.append(f"| {p['model']} · {p['set']} · {p['ctx']} | - | - | gates failed | - | NO |")
            ok_all = False
            continue
        for name, s in p["arms"].items():
            new = arm_stats(r, name)
            if new is None:
                continue
            seen += 1
            ok = methodology.reproduces(s["min"], s["max"], new["min"], new["max"])
            ok_all &= ok
            lines.append(f"| {p['model']} · {p['set']} · {p['ctx']} | {name} | {_range(s)} | {fmt_ppl(new)} | "
                         f"{pct(new['ppl'], s['ppl'], 3)} | {'yes' if ok else 'NO'} |")
    if not seen:
        ok_all = False
        lines.append("no results measure a locked arm")
    return "\n".join(lines), ok_all
