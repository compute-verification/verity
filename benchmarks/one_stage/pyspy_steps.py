"""Where a served step's host time goes, by Python function and thread, from a `served_util.sh PYSPY=1` run: py-spy's
Chrome trace (`record --format chrometrace --idle --threads --subprocesses`) cut to the main load's full steps.

    python benchmarks/one_stage/pyspy_steps.py --trace-json pyspy.json --t0 pyspy.t0 --steps advice_trace.jsonl \
        --timings main.timings.json --T 8192 --out pyspy_steps.json

py-spy's events carry microseconds since its start, which `--t0` (the UTC second the harness launched it) puts on the step
trace's clock. The window is the longest run of consecutive full steps (T tokens) after the main load's start, from the step
before the first to the last. A thread's time between two of its samples goes to every distinct frame on its stack
(inclusive) and to its innermost frame (self); with --idle a waiting thread is sampled too, so a thread's total is the window.
Threads rank by their stack changes in the window, which idle threads barely make.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import served_util as su


def frame(e: dict) -> str:
    f = (e.get("args") or {}).get("filename") or ""
    return f"{e['name']} ({os.path.basename(f)})" if f else e["name"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace-json", required=True, type=Path)
    ap.add_argument("--t0", required=True, type=Path)
    ap.add_argument("--steps", required=True, type=Path)
    ap.add_argument("--timings", type=Path)
    ap.add_argument("--T", required=True, type=int)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = float(a.t0.read_text().strip())
    st = su.steps(a.steps)
    after = json.loads(a.timings.read_text())["start_unix"] if a.timings and a.timings.exists() else None
    idx = su.steady(st, a.T, float("-inf"), float("inf"), after)
    if not idx:
        raise SystemExit(f"no full step of {a.T} tokens after the main load's start")
    lo, hi = (st[idx[0] - 1]["t"] - t0) * 1e6, (st[idx[-1]]["t"] - t0) * 1e6
    n = len(idx)
    events = json.loads(a.trace_json.read_text())
    events.sort(key=lambda e: e["ts"])
    stack: dict[tuple, list[str]] = defaultdict(list)
    last: dict[tuple, float] = {}
    proc: dict[tuple, str] = {}
    thr: dict[tuple, str] = {}
    incl: dict[tuple, Counter] = defaultdict(Counter)
    self_: dict[tuple, Counter] = defaultdict(Counter)
    changes: Counter = Counter()
    for e in events:
        k, ts = (e["pid"], e["tid"]), float(e["ts"])
        s = stack[k]
        if k in last and s:
            dt = min(ts, hi) - max(last[k], lo)
            if dt > 0:
                for f in set(s):
                    incl[k][f] += dt
                self_[k][s[-1]] += dt
        last[k] = ts
        f = frame(e)
        if e["ph"] == "B":
            s.append(f)
            if f.startswith("thread ("):
                thr[k] = f.split(": ", 1)[-1] if ": " in f else f
            elif f.startswith("process "):
                proc[k] = f[:70]
            if lo <= ts <= hi:
                changes[k] += 1
        elif e["ph"] == "E" and s:
            s.pop()
    span = hi - lo
    out: dict = {"window_s": round(span / 1e6, 3), "full_steps": n, "step_s": round(span / 1e6 / n, 3),
                 "first_step": st[idx[0]]["step"], "last_step": st[idx[-1]]["step"], "events": len(events), "threads": []}
    for k, c in changes.most_common(a.threads):
        out["threads"].append({
            "pid": k[0], "tid": k[1], "thread": f"{proc.get(k, '?')} / {thr.get(k, '?')}", "stack_changes": c,
            "inclusive_s_per_step": [[f, round(v / 1e6 / n, 3), round(v / span, 3)] for f, v in incl[k].most_common(a.top)],
            "self_s_per_step": [[f, round(v / 1e6 / n, 3), round(v / span, 3)] for f, v in self_[k].most_common(a.top // 2)]})
    a.out.write_text(json.dumps(out, indent=1))
    print(f"PYSPY window {out['window_s']} s, {n} full steps (steps {out['first_step']}..{out['last_step']}) of {out['step_s']} s")
    for t in out["threads"]:
        print(f"THREAD pid {t['pid']} tid {t['tid']} {t['thread'][:90]}: {t['stack_changes']} stack changes")
        for f, v, sh in t["inclusive_s_per_step"]:
            print(f"  incl {v:9.3f} s/step {sh:6.1%}  {f[:150]}")
        for f, v, sh in t["self_s_per_step"]:
            print(f"  self {v:9.3f} s/step {sh:6.1%}  {f[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
