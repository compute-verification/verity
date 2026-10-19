"""Render run A8's per-kind table and its gap lines (Markdown) from `a8_kinds.py`'s table and `a8.py`'s summary.

    python benchmarks/one_stage/a8_table.py --kinds kinds.json [--summary a8/summary.json] [--run RUN_ID]

A row per computing kind (Const32 families, one gate each, left out): its root Calls, gates and share of the step's
computing gates, the unit M0 was given, its ANDs and k_log, the lowering outcome, and from the audit whether its units were
staged from the step's values, proved and held by Lean. A gap line names what keeps a kind from Lean acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def short(uid: str | None, n: int = 60) -> str:
    if not uid:
        return "-"
    return uid if len(uid) <= n else uid[:n - 1] + "…"


def last(r: dict) -> dict:
    return (r.get("attempts") or [{}])[-1]


def gap(f: str, r: dict, s: dict | None) -> str | None:
    """Why kind f has no Lean-accepted unit, or None."""
    a = last(r)
    size = f"unit {short(r.get('unit'), 80)} ({r.get('unit_gates')} word gates, {r.get('units_per_call')} per Call)"
    share = f"{100 * r['share']:.3f}% of the step"
    o = r.get("outcome")
    if o == "no-piece":
        sub = r.get("sub_unit") or {}
        tail = f"; largest sub-unit with every piece: {short(sub.get('unit'), 60)} ({sub.get('outcome')})" if sub else ""
        return f"{f}: no C-Flock piece for {', '.join(r.get('missing') or [a.get('piece', '?')])}; {size}; {share}{tail}"
    if o in (None, "not probed"):
        return f"{f}: not probed; {share}"
    if o != "lowers":
        why = a.get("why") or r.get("why") or o
        return f"{f}: does not lower ({o}: {str(why)[:160]}); {size}; {share}"
    if s is None:
        return None
    v = (s.get("values") or {}).get(f)
    if v is None:
        return f"{f}: lowers, not staged in this audit; {size}; {share}"
    if not v.get("staged"):
        return f"{f}: lowers, no values in the slice ({str(v.get('why'))[:200]}); {size}; {share}"
    verdicts = (s.get("audit") or {}).get("verdicts") or {}
    lean = verdicts.get(f"lean-verify/{f}")
    if lean is None and (s.get("draw_counts") or {}).get(f, 0) == 0:
        return None
    if lean is not True:
        return f"{f}: staged but Lean did not accept ({(s.get('audit') or {}).get('why', {}).get(f'lean-verify/{f}')}); {size}; {share}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--run", default=None, help="the audit's run id, named in the table")
    a = ap.parse_args()
    t = json.loads(Path(a.kinds).read_text())
    s = json.loads(Path(a.summary).read_text()) if a.summary else None
    kinds = {f: r for f, r in t["kinds"].items() if r.get("gates", 0) > 0 and not f.startswith("Const")}
    members = {m["name"]: m for m in (s or {}).get("members", [])}
    verdicts = ((s or {}).get("audit") or {}).get("verdicts") or {}
    print(f"| kind | calls | gates | share | unit | ANDs | k_log | outcome | staged (twin) | M0 | Lean |")
    print("|---|---:|---:|---:|---|---:|---:|---|---|---|---|")
    for f, r in sorted(kinds.items(), key=lambda kv: -kv[1]["gates"]):
        a_ = last(r)
        v = ((s or {}).get("values") or {}).get(f) or {}
        m = members.get(f)
        twin = "-"
        if v.get("staged"):
            tw = v.get("twin") or {}
            twin = f"{m['n'] if m else v.get('n')} ({'agree' if tw.get('agree') else tw.get('agree')}, {tw.get('compared')} words)"
        elif v:
            twin = "no values"
        mv, lv = verdicts.get(f"m0-verifier/{f}"), verdicts.get(f"lean-verify/{f}")
        print(f"| {f} | {r['calls']} | {r['gates']:.3e} | {100 * r['share']:.3f}% | {short(r.get('unit'), 48)} | "
              f"{a_.get('ands', '-')} | {a_.get('k_log', '-')} | {r.get('outcome')} | {twin} | "
              f"{'-' if mv is None else ('accepted' if mv else 'refused')} | {'-' if lv is None else ('accepted' if lv else 'refused')} |")
    lowers = [f for f, r in kinds.items() if r.get("outcome") == "lowers"]
    print(f"\n{len(lowers)} of {len(kinds)} kinds lower ({100 * sum(kinds[f]['share'] for f in lowers):.2f}% of the step's computing "
          f"gates){'; audit ' + a.run if a.run else ''}.\n\nGaps:\n")
    for f, r in sorted(kinds.items(), key=lambda kv: -kv[1]["gates"]):
        g = gap(f, r, s)
        if g:
            print(f"- {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
