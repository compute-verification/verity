"""E1 — measured production rate of the reference interpreter (gates/s, wall, peak RSS) from one or more `producer.py` output dirs, and
the MODELED full-RU time that follows from it.  Reads `progress.jsonl` (one line per root call: gates, eval_s, max_rss_gb, ...) and
`rate.json`; writes a JSON summary and prints a markdown table.  Every number is labelled: the rate rows are MEASURED on the host named
in `rate.json`; the full-RU projection is MODELED (rate x the layout's computed gate count).

    PYTHONPATH=. python -m e2e.interior.rate /tmp/e1-smoke1 /tmp/e1-4090 --out out/eod/e1/rate-summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

COMPUTED_GATES_PER_RU_4x8 = 107_636_385  # MEASURED: LazyProgram(bind(Serve, B0_CONFIG, LP=4, STEPS=8)).gates - input_gates (producer log "program")


def summarize(out_dir: Path) -> dict:
    prog = [json.loads(l) for l in (out_dir / "progress.jsonl").read_text().splitlines() if l.strip()]
    rate = json.loads((out_dir / "rate.json").read_text()) if (out_dir / "rate.json").exists() else {}
    calls = [r for r in prog if r["event"] == "root_call"]
    by_role: dict[str, dict] = defaultdict(lambda: {"calls": 0, "gates": 0, "eval_s": 0.0})
    for r in calls:
        role = r["name"].rsplit("/", 1)[-1]
        d = by_role[role]
        d["calls"] += 1
        d["gates"] += r["gates"]
        d["eval_s"] += r["eval_s"]
    for d in by_role.values():
        d["us_per_gate"] = round(1e6 * d["eval_s"] / d["gates"], 1) if d["gates"] else None
    gates = sum(r["gates"] for r in calls)
    eval_s = sum(r["eval_s"] for r in calls)
    wall = calls[-1]["elapsed_s"] - (prog[0]["elapsed_s"] if prog else 0) if calls else 0.0
    us = 1e6 * eval_s / gates if gates else None
    rss = max((r.get("max_rss_gb", 0) for r in prog), default=0)
    start = next((r for r in prog if r["event"] == "start"), {})
    done = next((r for r in prog if r["event"] == "done"), None)
    program = next((r for r in prog if r["event"] == "program"), {})
    computed = program.get("computed_gates", COMPUTED_GATES_PER_RU_4x8)
    return {"out_dir": str(out_dir), "host": rate.get("host"), "python": rate.get("python"), "record": start.get("record"), "request": start.get("request"),
            "label": "MEASURED", "root_calls": len(calls), "gates_evaluated": gates, "eval_s": round(eval_s, 1), "wall_s_incl_setup": round(wall, 1),
            "gates_per_s": round(gates / eval_s, 1) if eval_s else None, "us_per_gate": round(us, 2) if us else None, "peak_rss_gb": round(rss, 3),
            "by_role": dict(by_role), "boundary": {k: (done or calls[-1] if calls else {}).get(k) for k in ("rows_compared", "rows_agree", "rows_mismatch", "boundary_compared", "boundary_agree", "boundary_mismatch")},
            "projection": {"label": "MODELED", "computed_gates_per_ru": computed, "full_ru_hours_at_this_rate": round(computed * us / 3.6e9, 2) if us else None,
                           "weighted_note": "the RU is 67 % post-call gates (72.4 M of 107.6 M), 18 % final (19.1 M), 14 % pre (15.1 M), 1 % attention; use the post/final rates, not the mean"},
            "verdict": (done or {}).get("verdict")}


def markdown(rows: list[dict]) -> str:
    out = ["| run | host | root calls | gates | eval s | µs/gate (MEASURED) | peak RSS GB | post µs/gate | pre µs/gate | full RU h (MODELED) | boundary rows compared / agree / mismatch |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b = r["boundary"]
        bc = b.get("rows_compared") or b.get("boundary_compared")
        ba = b.get("rows_agree") or b.get("boundary_agree")
        bm = b.get("rows_mismatch") if b.get("rows_mismatch") is not None else b.get("boundary_mismatch")
        out.append(f"| `{Path(r['out_dir']).name}` | {r['host']} | {r['root_calls']} | {r['gates_evaluated']:,} | {r['eval_s']} | {r['us_per_gate']} | {r['peak_rss_gb']} | "
                   f"{r['by_role'].get('post', {}).get('us_per_gate')} | {r['by_role'].get('pre', {}).get('us_per_gate')} | {r['projection']['full_ru_hours_at_this_rate']} | {bc} / {ba} / {bm} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = [summarize(d) for d in a.dirs]
    print(markdown(rows))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"schema": "veritor/e1/rate-summary/v1", "runs": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
