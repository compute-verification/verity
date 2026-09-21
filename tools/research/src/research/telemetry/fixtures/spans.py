"""Nested / repeated / overlapping spans fixture (brief §12), emitted through the real append helper from one parent and two children
(concurrent siblings), then checked with telemetry.events.spans: exclusive = inclusive - union(children); the two overlapping children are
not summed into the parent; repeated `match.load` calls are distinct; the deliberately unclosed span stays open.

    spans.py --attempt-dir D
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.telemetry.events import Emitter, read_all, spans  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--attempt-dir", required=True)
a = ap.parse_args()
em = Emitter(a.attempt_dir, proc="spans-fixture")
row = em.span_start("row")
for _ in range(2):                                   # repeated
    s = em.span_start("match.load", parent=row)
    time.sleep(0.3)
    em.span_end("match.load", s)
kids = []
for stage, dur in (("match.fold", 1.0), ("match.x09", 0.6)):   # overlapping concurrent children in child processes (own streams)
    pid = os.fork()
    if pid == 0:
        cem = Emitter(a.attempt_dir, proc="spans-child")
        s = cem.span_start(stage, parent=row)
        time.sleep(dur)
        cem.span_end(stage, s)
        os._exit(0)
    kids.append(pid)
for pid in kids:
    os.waitpid(pid, 0)
g = em.span_start("match.gm", parent=row)          # nested grandchild + an unclosed leaf
gg = em.span_start("match.gm.g7", parent=g)
time.sleep(0.2)
em.span_end("match.gm.g7", gg)
em.span_start("match.gm.g8", parent=g)             # never closed (simulated death of that step)
em.span_end("match.gm", g)
em.span_end("row", row)

acc = spans(read_all(a.attempt_dir).records)
by = {s["stage"] + ("" if s["stage"] != "match.load" else s["id"][-30:]): s for s in acc["spans"]}
rowr = next(s for s in acc["spans"] if s["stage"] == "row")
loads = [s for s in acc["spans"] if s["stage"] == "match.load"]
fold = next(s for s in acc["spans"] if s["stage"] == "match.fold")
x09 = next(s for s in acc["spans"] if s["stage"] == "match.x09")
gm = next(s for s in acc["spans"] if s["stage"] == "match.gm")
g8 = next(s for s in acc["spans"] if s["stage"] == "match.gm.g8")
checks = {
    "two distinct match.load invocations": len(loads) == 2 and loads[0]["id"] != loads[1]["id"],
    "fold and x09 overlap (concurrent siblings, own streams)": fold["start_mono"] < x09["end_mono"] and x09["start_mono"] < fold["end_mono"],
    "row exclusive = inclusive - union(children) (not minus the sum)": abs(rowr["exclusive_s"] - (rowr["inclusive_s"] - (sum(l["inclusive_s"] for l in loads) + max(fold["inclusive_s"], x09["inclusive_s"]) + gm["inclusive_s"]))) < 0.15,
    "row exclusive is small (>= 0, < 0.3 s of bookkeeping)": 0.0 <= rowr["exclusive_s"] < 0.3,
    "gm exclusive excludes g7 (nested) and the open g8": gm["exclusive_s"] < gm["inclusive_s"],
    "g8 stays open/interrupted with a last observation, no invented end": g8["outcome"] == "interrupted" and g8["end_mono"] is None and g8.get("observed_at_least_s") is not None,
    "single root -> overlap_s 0 and total_union = row inclusive": acc["overlap_s"] == 0.0 and abs(acc["total_union_s"] - rowr["inclusive_s"]) < 1e-6,
    "by_stage counts": acc["by_stage"]["match.load"]["calls"] == 2 and acc["by_stage"]["match.gm.g8"]["open"] == 1,
}
ok = all(checks.values())
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
print(f"row inclusive {rowr['inclusive_s']:.3f} exclusive {rowr['exclusive_s']:.3f}; fold {fold['inclusive_s']:.3f} x09 {x09['inclusive_s']:.3f} (summed {fold['inclusive_s'] + x09['inclusive_s']:.3f}); loads {[round(l['inclusive_s'], 3) for l in loads]}")
sys.exit(0 if ok else 1)
