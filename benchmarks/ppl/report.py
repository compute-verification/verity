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
        db = 100 * (s["ppl"] / bf["ppl"] - 1) if name != "bf16" else None
        dp = 100 * (s["ppl"] / p["ppl"] - 1) if p is not None and name != s["plain"] else None
        v = verdict(r, name, s["plain"]) if p is not None and name != s["plain"] else "-"
        lines.append(f"| {name} | {s['n']} | {fmt_ppl(s)} | {'-' if db is None else f'{db:+.2f}'} | "
                     f"{'-' if dp is None else f'{dp:+.2f}'} | {v} |")
    g = r["gates"]
    lines.append("")
    lines.append("gates: " + ", ".join(f"{n} {v['checked']}" + (f" (skipped: {v['skipped']})" if v.get("skipped") else "")
                                       for n, v in g.items()))
    return "\n".join(lines)


def baseline_rows(lock: dict, results: list[dict]) -> str:
    """The locked baseline against a new set of results: each locked row's perplexity, the new one, and whether it
    reproduces (within the lock's tolerance: the locked seed range, at least `rel_floor`)."""
    found = {(r["model"], r["set"], r["ctx"]): r for r in results}
    lines = ["| model | set | ctx | arm | locked | new | Δ% | reproduces |", "|---|---|---:|---|---:|---:|---:|---|"]
    ok_all = True
    for row in lock["rows"]:
        r = found.get((row["model"], row["set"], row["ctx"]))
        s = arm_stats(r, row["arm"]) if r is not None and r["passed"] else None
        if s is None:
            lines.append(f"| {row['model']} | {row['set']} | {row['ctx']} | {row['arm']} | {row['ppl']:.4f} | - | - | - |")
            continue
        tol = max((row["max"] - row["min"]) / row["ppl"], lock["rel_floor"])
        d = s["ppl"] / row["ppl"] - 1
        ok = abs(d) <= tol
        ok_all &= ok
        lines.append(f"| {row['model']} | {row['set']} | {row['ctx']} | {row['arm']} | {row['ppl']:.4f} | {s['ppl']:.4f} | "
                     f"{100 * d:+.3f} | {'yes' if ok else 'NO'} |")
    return "\n".join(lines) + ("\n" if ok_all else "\nsome rows do not reproduce\n")
