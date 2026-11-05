import os

#: the study's data directory: trace.py writes it, the rest read it (`research data fetch art:7ee7a884...` restores it)
DATA = os.environ.get("LANDING_STUDY_DIR", "/tmp/ld")
import json
import statistics
import sys
from datetime import datetime, UTC

rows = [r for r in json.load(open(sys.argv[1] if len(sys.argv) > 1 else DATA + "/trace.json")) if r["pr"]]


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def d(t):
    return datetime.fromtimestamp(t, UTC).strftime("%m-%d")


periods = [("10-21..10-28 (pre-checked)", "10-21", "10-28"), ("10-29..11-05 (checked+review)", "10-29", "11-05")]
for name, a, b in periods:
    sel = [r for r in rows if r["main_at"] and a <= d(r["main_at"]) <= b and r["ready_last"]]
    print(f"\n{name}: n={len(sel)}  ({len(sel) / 8:.0f}/day)")
    for lab, f in [("ready->main (last mark)", lambda r: r["main_at"] - r["ready_last"]),
                   ("ready->main (first mark at head)", lambda r: r["main_at"] - r["ready_at"]),
                   ("ready->reviewed", lambda r: (r["reviewed_at"] - r["ready_last"]) if r["reviewed_at"] else None),
                   ("ready->checked", lambda r: (r["checked_at"] - r["ready_last"]) if r["checked_at"] else None),
                   ("reviewed->checked", lambda r: (r["checked_at"] - max(r["reviewed_at"], r["ready_last"]))
                    if r["checked_at"] and r["reviewed_at"] else None),
                   ("checked->main", lambda r: (r["main_at"] - r["checked_at"]) if r["checked_at"] else None)]:
        xs = [x / 3600 for x in (f(r) for r in sel) if x is not None]
        if xs:
            print(f"  {lab:34s} n={len(xs):4d} p50={q(xs, .5):6.1f}h p90={q(xs, .9):6.1f}h mean={statistics.mean(xs):6.1f}h")
    rev = sum(1 for r in sel if r["reviewed_at"])
    lean = sum(1 for r in sel if r["lean"])
    print(f"  reviewed {rev}, lowrisk/unreviewed {len(sel) - rev}, lean-touching {lean}")

print("\nper day, ready(last)->main p50/p90 h, count:")
days = sorted({d(r["main_at"]) for r in rows if r["main_at"]})
for day in days:
    xs = [(r["main_at"] - r["ready_last"]) / 3600 for r in rows if r["main_at"] and d(r["main_at"]) == day and r["ready_last"]]
    if xs:
        print(f"  {day} n={len(xs):4d} p50={q(xs, .5):5.1f} p90={q(xs, .9):5.1f}")
