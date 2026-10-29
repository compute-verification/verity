"""The H200 power ranking of PoUW designs: check a rows file and render it.

A rows file is TOML: one ``[[row]]`` per design or argument, each target a cited value or ``"?"``, and one
``[period]`` table with the compute a monitoring period scales by. ``rank.py`` refuses a row that leaves a column empty,
uses a provenance mark or Lean status outside the lists below, or ranks a design without a position, and writes
``ranking.md``, ``ranking.csv`` and ``ranking.json`` into ``--out``.

The rows hold measured numbers and the ids of the runs behind them, so they live with the evidence, not in this tree:
``uv run python benchmarks/pouw/core_rank/rank.py --rows ROWS.toml --out DIR``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tomllib
from pathlib import Path

#: Where a number was obtained. "H200-modelled" is the machine model on data/devices/sm_90; "estimate" is arithmetic on
#: measured parts with no run of its own.
CARDS = ("H200-measured", "H200-modelled", "H100-only", "4090-only", "sm_120-only", "estimate", "none")
LEAN = ("proved", "draft semantics", "locked", "refuted", "open", "none")
STATUS = ("met", "open", "fail", "n/a")
VERDICTS = ("ranked", "argument", "out", "not core")
TARGETS = ("t1", "t2", "t3")
REQUIRED = ("id", "design", "catalogs", "assumption", "rank", "lean", "lean_ids", "vllm", "verdict", "why", "cites")
WEEK_S, MONTH_S, YEAR_S = 7 * 86400, 365 * 86400 / 12, 365 * 86400


def check(rows: list[dict], period: dict) -> list[str]:
    errs: list[str] = []
    seen: set[str] = set()
    positions: list[int] = []
    for r in rows:
        rid = r.get("id", "?")
        for k in REQUIRED:
            if k not in r or r[k] in ("", []):
                errs.append(f"{rid}: '{k}' is missing or empty; write '?' where no number exists")
        if rid in seen:
            errs.append(f"{rid}: a second row with this id; ids are unique")
        seen.add(rid)
        if not (isinstance(r.get("rank"), int) and 0 <= r["rank"] <= 5):
            errs.append(f"{rid}: rank {r.get('rank')!r} is not an integer 0..5 (CORE.md's ladder; the strongest needed)")
        for t in TARGETS:
            if not r.get(t):
                errs.append(f"{rid}: '{t}' is empty; write '?' where no number exists")
            if r.get(f"{t}_status") not in STATUS:
                errs.append(f"{rid}: '{t}_status' must be one of {STATUS}")
            bad = [c for c in r.get(f"{t}_card", []) if c not in CARDS]
            if bad or not r.get(f"{t}_card"):
                errs.append(f"{rid}: '{t}_card' must be a non-empty list from {CARDS} (got {r.get(f'{t}_card')!r})")
        bad = [s for s in r.get("lean", []) if s not in LEAN]
        if bad:
            errs.append(f"{rid}: Lean status {bad} not in {LEAN}")
        if r.get("verdict") not in VERDICTS:
            errs.append(f"{rid}: verdict must be one of {VERDICTS}")
        if r.get("verdict") == "ranked":
            if not isinstance(r.get("pos"), int):
                errs.append(f"{rid}: a ranked row needs an integer 'pos'")
            else:
                positions.append(r["pos"])
        share = r.get("share")
        if share is not None and not (isinstance(share, (int, float)) and 0 <= share <= 1 and r.get("share_basis")):
            errs.append(f"{rid}: 'share' must be a fraction in [0, 1] with a 'share_basis' saying which reading it is")
    if sorted(positions) != list(range(1, len(positions) + 1)):
        errs.append(f"ranked positions {sorted(positions)} must be 1..{len(positions)} with no gap or repeat")
    for k in ("world_flop_year", "dc_gpus", "gpu_flops", "mfu", "gates_per_flop", "sources"):
        if k not in period:
            errs.append(f"[period] lacks '{k}'")
    return errs


def period_rows(rows: list[dict], period: dict) -> list[dict]:
    """Each shared row's skippable work a period: its share of the compute it covers, pooled, in AND gates."""
    world_year = period["world_flop_year"] * period["gates_per_flop"]
    dc_year = period["dc_gpus"] * period["gpu_flops"] * period["mfu"] * YEAR_S * period["gates_per_flop"]
    out = []
    for r in rows:
        if r.get("share") is None:
            continue
        s = float(r["share"])
        row = {"id": r["id"], "design": r["design"], "share": s, "basis": r["share_basis"]}
        for scope, year in (("world", world_year), ("datacenter", dc_year)):
            for name, secs in (("week", WEEK_S), ("month", MONTH_S), ("year", YEAR_S)):
                row[f"{scope}_{name}"] = s * year * secs / YEAR_S
        out.append(row)
    return out


def entry(r: dict, t: str) -> str:
    return f"{r[t]} [{', '.join(r[f'{t}_card'])}; {r[f'{t}_status']}]"


def md_table(header: list[str], body: list[list[str]]) -> list[str]:
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(esc(c) for c in row) + " |" for row in body]


def render_md(rows: list[dict], period: dict, prows: list[dict], meta: dict) -> str:
    ranked = sorted((r for r in rows if r["verdict"] == "ranked"), key=lambda r: r["pos"])
    head = ["#", "id", "design", "rank", "T1 gap, H200 time", "T2 t_H/t_S", "T3 perplexity", "Lean", "vLLM"]
    line = lambda r, pos: [pos, r["id"], r["design"], f"{r['rank']}: {r['assumption']}", entry(r, "t1"), entry(r, "t2"),
                           entry(r, "t3"), f"{', '.join(r['lean'])}: {r['lean_ids']}", r["vllm"]]
    out = [f"# {meta.get('title', 'H200 power ranking')}", ""]
    out += [meta.get("preamble", ""), ""]
    out += ["Each target reads `value [where it was obtained; status]`. Marks: " + ", ".join(f"`{c}`" for c in CARDS)
            + ". Status: met, open or fail against CORE.md's targets (T1 ≤ 1% in H200 time, T2 ≤ 100, T3 ≤ 5%).", ""]
    out += ["## Ranked designs", ""] + md_table(head, [line(r, r["pos"]) for r in ranked]) + [""]
    for verdict, title in (("argument", "Arguments (no design of their own)"), ("not core", "Outside the core problem"),
                           ("out", "Out")):
        sel = [r for r in rows if r["verdict"] == verdict]
        if sel:
            out += [f"## {title}", ""]
            out += md_table(head[1:] + ["why"], [line(r, "")[1:] + [r["why"]] for r in sel]) + [""]
    if prows:
        out += ["## Per monitoring period", "", period.get("note", ""), ""]
        fmt = lambda x: f"{x:.2e}"
        out += md_table(["id", "design", "share", "reading", "world: week", "month", "year", "datacenter: week", "month",
                         "year"],
                        [[p["id"], p["design"], f"{p['share']:.3%}", p["basis"], *(fmt(p[f"world_{n}"]) for n in
                          ("week", "month", "year")), *(fmt(p[f"datacenter_{n}"]) for n in ("week", "month", "year"))]
                         for p in prows])
        out += ["", "Sources: " + "; ".join(period["sources"]), ""]
    out += ["## Rows in detail", ""]
    for r in sorted(rows, key=lambda r: (VERDICTS.index(r["verdict"]), r.get("pos", 99), r["id"])):
        out += [f"### {r['id']}: {r['design']}", "", f"- Catalogs: {', '.join(r['catalogs'])}",
                f"- Assumption (rank {r['rank']}): {r['assumption']}"]
        out += [f"- {t.upper()}: {entry(r, t)}" for t in TARGETS]
        out += [f"- Lean: {', '.join(r['lean'])}: {r['lean_ids']}", f"- vLLM: {r['vllm']}",
                f"- Verdict: {r['verdict']}: {r['why']}"]
        if r.get("needs"):
            out += [f"- Still needs on the H200: {r['needs']}"]
        out += ["- Sources:"] + [f"  - {c}" for c in r["cites"]] + [""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    data = tomllib.loads(a.rows.read_text())
    rows, period, meta = data.get("row", []), data.get("period", {}), data.get("meta", {})
    errs = check(rows, period)
    if errs:
        print("\n".join(errs), file=sys.stderr)
        return 1
    prows = period_rows(rows, period)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ranking.md").write_text(render_md(rows, period, prows, meta))
    (a.out / "ranking.json").write_text(json.dumps({"meta": meta, "period": period, "rows": rows,
                                                    "per_period": prows}, indent=1, ensure_ascii=False))
    keys = ["pos", *REQUIRED[:5], *(f"{t}{s}" for t in TARGETS for s in ("", "_card", "_status")), *REQUIRED[5:],
            "needs", "share", "share_basis"]
    with (a.out / "ranking.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: "; ".join(v) if isinstance(v, list) else v for k, v in r.items()})
    print(f"{len(rows)} rows ({sum(r['verdict'] == 'ranked' for r in rows)} ranked) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
