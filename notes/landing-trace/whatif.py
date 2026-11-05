"""Replay the last 7 days' landed PRs with each lever's causes removed: ready-to-main median, p90 and PR-hours per week."""
import json
import statistics

D = json.load(open("/tmp/trace/landing-trace-v1.json"))
per = json.load(open("/tmp/trace/per_pr_causes.json"))
files = json.load(open("/tmp/trace/pr_files.json"))
reheads = json.load(open("/tmp/trace/reheads.json"))
merge_only_rework = {}
for r in reheads:
    if r["kind"].startswith("merge-only"):
        merge_only_rework[r["pr"]] = merge_only_rework.get(r["pr"], 0)

W0 = "2026-10-29T17:30"
landed = [r for r in D["prs"] if r["main"] and r["main"] >= W0 and r.get("r2m_h") is not None]
waiting = [r for r in D["prs"] if not r["main"] and r.get("r2m_h") is not None]


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def is_lean(n):
    f = files.get(str(n))
    return bool(f and f["lean"])


def removed(r, lever):
    c = per.get(str(r["pr"]), {})
    g = lambda k: c.get(k, 0.0)
    if lever == "A Lean check decoupled from next (non-Lean never waits on one)":
        return 0.0 if is_lean(r["pr"]) else g("to_checked: behind a Lean check (prefix waits)")
    if lever == "B admission off the tick, continuous (leave 0.25 h)":
        a = g("admission: queued (no refusal logged)") + g("admission: waiting for its preflight turn") + g("admission: preflight budget (900 s) spent")
        return max(0.0, a - 0.25)
    if lever == "C1 unreviewed bottom member parked (ci, in flight)":
        return g("to_checked: behind a PR without a land verdict")
    if lever == "C2 verdict starts at mark, staffed in 1 h (own wait capped at 1 h)":
        return max(0.0, g("review: no land verdict at its head") - 1.0)
    if lever == "D after-landing cached + automatic + revert at once (cap 0.75 h)":
        al = sum(v for k, v in c.items() if k.startswith("after_landing"))
        return max(0.0, al - 0.75)
    if lever == "E `main` = checked, a `verified` ref for citations (Daniel's CI-239 rule)":
        return sum(v for k, v in c.items() if k.startswith("after_landing"))
    if lever == "F passes under 20 min (ci, in flight): cadence and rebuild x0.4":
        return 0.6 * (g("to_checked: landing (tick cadence)") + g("to_checked: next being rebuilt / no prefix"))
    if lever == "G drive never stopped":
        return g("to_checked: drive stopped") + g("after_landing: drive stopped")
    return 0.0


LEVERS = ["A Lean check decoupled from next (non-Lean never waits on one)", "B admission off the tick, continuous (leave 0.25 h)",
          "C1 unreviewed bottom member parked (ci, in flight)", "C2 verdict starts at mark, staffed in 1 h (own wait capped at 1 h)",
          "D after-landing cached + automatic + revert at once (cap 0.75 h)",
          "E `main` = checked, a `verified` ref for citations (Daniel's CI-239 rule)",
          "F passes under 20 min (ci, in flight): cadence and rebuild x0.4", "G drive never stopped"]
base = [r["r2m_h"] for r in landed]
print(f"baseline, {len(landed)} PRs landed in the 7 days to 5 Nov 17:30Z: median {statistics.median(base):.1f} h, p90 {pct(base, .9):.1f} h, "
      f"{sum(base):,.0f} PR-h; {len(waiting)} waiting: {sum(r['r2m_h'] for r in waiting):,.0f} PR-h so far")
out = {}
for lv in LEVERS:
    xs = [max(0.0, r["r2m_h"] - removed(r, lv)) for r in landed]
    saved = sum(base) - sum(xs) + sum(removed(r, lv) for r in waiting)
    out[lv] = dict(median=statistics.median(xs), p90=pct(xs, .9), saved=saved)
    print(f"{lv[:78]:78s} median {statistics.median(xs):5.1f}  p90 {pct(xs, .9):6.1f}  saves {saved:7,.0f} PR-h/wk")
for combo, lvs in (("in flight only: C1 + F + D", ["C1 unreviewed bottom member parked (ci, in flight)", "F passes under 20 min (ci, in flight): cadence and rebuild x0.4", "D after-landing cached + automatic + revert at once (cap 0.75 h)"]),
                   ("in flight + A + B", ["C1 unreviewed bottom member parked (ci, in flight)", "F passes under 20 min (ci, in flight): cadence and rebuild x0.4", "D after-landing cached + automatic + revert at once (cap 0.75 h)", LEVERS[0], LEVERS[1]]),
                   ("all: A B C1 C2 D F G", [LEVERS[i] for i in (0, 1, 2, 3, 4, 6, 7)]),
                   ("all + E", [LEVERS[i] for i in (0, 1, 2, 3, 5, 6, 7)])):
    xs = [max(0.0, r["r2m_h"] - sum(removed(r, lv) for lv in lvs)) for r in landed]
    print(f"{combo:78s} median {statistics.median(xs):5.1f}  p90 {pct(xs, .9):6.1f}  saves {sum(base) - sum(xs):7,.0f} PR-h/wk (landed only)")

import collections
lvs = [LEVERS[i] for i in (0, 1, 2, 3, 4, 6, 7)]
res = []
for r in landed:
    rem = sum(removed(r, lv) for lv in lvs)
    res.append((max(0.0, r["r2m_h"] - rem), r))
res.sort(key=lambda x: -x[0])
tail = res[: len(res) // 10]
comp = collections.Counter()
for left, r in tail:
    c = per.get(str(r["pr"]), {})
    for k, v in c.items():
        comp[k] += v
    # subtract the removed parts approximately by listing the raw causes
print("\nresidual top decile after all levers:", len(tail), "PRs; their raw cause hours (before levers):")
for k, v in comp.most_common(12):
    print(f"  {v:8,.0f}  {k}")
print("lean share in tail:", sum(1 for _, r in tail if is_lean(r["pr"])), "of", len(tail))
print("tail PRs' final mark dates:", collections.Counter(r["final_mark"][:10] for _, r in tail).most_common(6))
