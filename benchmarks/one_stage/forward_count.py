"""One capture's forwards as `tensor_util.py` cuts them (its ``--out`` JSON) against the engine's own steps (the step trace)
over the same capture: whether the cut finds one forward a served step.

    python benchmarks/one_stage/forward_count.py --sqlite PROF.sqlite --trace ADVICE_TRACE.jsonl --tu TU.json --T 8192 --out c.json

A served step's span is the previous step's ``t`` to its own (`served_util.py`'s); ``per_span`` counts the spans of full
steps (T tokens) in the capture by how many of the cut's forwards start in them.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import served_util as su  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--tu", required=True, type=Path)
    ap.add_argument("--T", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    db = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
    z = su.utc0_ns(db)
    lo, hi = db.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    doc = json.loads(a.tu.read_text())
    fw = doc.get("forwards", [])
    starts = sorted(f["start_ns"] for f in fw)
    st = su.steps(a.trace)
    at = lambda s: int(s * 1e9) - z  # noqa: E731
    inside = [r for r in st if lo <= at(r["t"]) <= hi]
    spans = [(at(st[i - 1]["t"]), at(st[i]["t"])) for i in range(1, len(st))
             if st[i]["T"] == a.T and st[i]["step"] == st[i - 1]["step"] + 1 and lo <= at(st[i - 1]["t"]) and at(st[i]["t"]) <= hi]
    per = Counter(bisect.bisect_left(starts, b) - bisect.bisect_left(starts, x) for x, b in spans)
    s = doc.get("summary", {})
    walls = [p["wall_ms"] for p in doc.get("steps", [])]
    out = {"capture_s": round((hi - lo) / 1e9, 3), "trace_steps_in_capture": len(inside),
           "trace_full_steps_in_capture": sum(1 for r in inside if r["T"] == a.T),
           "cut_forwards": len(fw), "cut_forwards_of_T": sum(1 for f in fw if f["tokens"] == a.T),
           "cut_forward_T": dict(Counter(str(f["tokens"]) for f in fw)),
           "full_spans": len(spans), "per_span": {str(k): v for k, v in sorted(per.items())},
           "error": doc.get("error"), "steps_kept": s.get("steps"), "dropped": len(s.get("dropped") or []),
           "step_ms_median": round(statistics.median(walls), 3) if walls else None,
           "step_ms_mean": round(s["wall_ms"], 3) if s.get("wall_ms") else None,
           "util_at_clock": s.get("util_at_clock"), "util_at_sheet": s.get("util"),
           "tensor_active_pct": s.get("tensor_active"), "sm_clock_mhz": s.get("clock_mhz")}
    a.out.write_text(json.dumps(out, indent=1))
    f = lambda v, p: "-" if v is None else p.format(v)  # noqa: E731
    print(f"CUT forwards {out['cut_forwards_of_T']} of T (of {out['cut_forwards']}), trace full steps in capture "
          f"{out['trace_full_steps_in_capture']}, forwards a full span {out['per_span']}; kept {out['steps_kept']} "
          f"(dropped {out['dropped']}), step median {f(out['step_ms_median'], '{:.2f}')} ms mean {f(out['step_ms_mean'], '{:.2f}')} ms, "
          f"util {f(out['util_at_clock'], '{:.2%}')} at {f(out['sm_clock_mhz'], '{:.0f}')} MHz ({f(out['util_at_sheet'], '{:.2%}')} at 1,830), "
          f"Tensor Active {f(out['tensor_active_pct'], '{:.2f}')}%" + (f"; {out['error']}" if out["error"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
