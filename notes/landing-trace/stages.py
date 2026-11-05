"""Per-PR stage durations, ready-to-main per day, and the Pareto by stage."""
import collections
import json
import statistics
from datetime import datetime, timedelta, timezone

NOW = datetime(2026, 11, 5, 17, 30, tzinfo=timezone.utc)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def h(a, b):
    return max(0.0, (b - a).total_seconds() / 3600) if a and b else None


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def pst(d):
    return d - timedelta(hours=8 if d >= datetime(2026, 11, 1, 9, tzinfo=timezone.utc) else 7)


D = json.load(open("/tmp/trace/trace_v1_raw.json"))
R = D["records"]
rows = []
for r in R:
    fm, first = ts(r["final_mark"]), ts(r["first_mark"])
    adm, ver, chk = ts(r["admitted"]), ts(r["verdict"]), ts(r["checked"])
    end = ts(r["main"]) or NOW
    if not fm:
        continue
    # clamp to the PR's order
    chk_c = min(max(chk, fm), end) if chk else None
    s = dict(pr=r["pr"], where=r["where"], fm=fm, first=first, end=end, main=ts(r["main"]))
    s["rework"] = h(first, fm) if first and first < fm else 0.0
    if chk_c:
        upto = chk_c
        if adm and fm <= adm <= upto:
            s["admission"] = h(fm, adm)
            a = adm
        else:
            s["admission"] = None
            a = fm
        if r["checked_how"] == "lowrisk" or not ver:
            s["review"] = 0.0
            v = a
        else:
            v = min(max(ver, a), upto)
            s["review"] = h(a, v)
        s["to_checked"] = h(v, upto)
        s["unsplit"] = h(fm, a) if s["admission"] is None else 0.0
        s["after_landing"] = h(upto, end)
    else:
        # not on checked yet (or no landing line): everything until end is pre-checked
        if adm and fm <= adm <= end:
            s["admission"] = h(fm, adm)
            a = adm
        else:
            s["admission"] = None
            a = fm
        if ver and ver > a:
            s["review"] = h(a, min(ver, end))
            v = min(ver, end)
        else:
            s["review"] = 0.0 if ver else h(a, end)
            v = ver if ver else end
        s["to_checked"] = h(max(v, a), end)
        s["unsplit"] = h(fm, a) if s["admission"] is None else 0.0
        s["after_landing"] = 0.0
    s["r2m"] = h(fm, end)
    s["r2m_first"] = h(first or fm, end)
    s["checked_how"] = r["checked_how"]
    rows.append(s)

json.dump([{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in s.items()} for s in rows], open("/tmp/trace/stages.json", "w"))

print("== ready-to-main per PST day of arrival on main (hours): n, median/p90 from final mark, median/p90 from first mark")
byday = collections.defaultdict(list)
for s in rows:
    if s["main"]:
        byday[pst(s["main"]).strftime("%a %m-%d")].append(s)
for d in sorted(byday, key=lambda k: k[4:]):
    xs = [s["r2m"] for s in byday[d]]
    ys = [s["r2m_first"] for s in byday[d]]
    print(f"{d}: n={len(xs):4d}  final {statistics.median(xs):6.1f} / {pct(xs, .9):6.1f}   first {statistics.median(ys):6.1f} / {pct(ys, .9):6.1f}")

for label, sel in (("all landed since 22 Oct", lambda s: s["main"]),
                   ("landed before CI-239 (main < 10-28 00:30Z)", lambda s: s["main"] and s["main"] < datetime(2026, 10, 28, 0, 30, tzinfo=timezone.utc)),
                   ("landed after CI-239", lambda s: s["main"] and s["main"] >= datetime(2026, 10, 28, 0, 30, tzinfo=timezone.utc)),
                   ("landed in the last 7 days (main >= 10-29 17:30Z)", lambda s: s["main"] and s["main"] >= datetime(2026, 10, 29, 17, 30, tzinfo=timezone.utc)),
                   ("marked, not on main yet (to now)", lambda s: not s["main"])):
    sub = [s for s in rows if sel(s)]
    tot = collections.Counter()
    for s in sub:
        for k in ("rework", "unsplit", "admission", "review", "to_checked", "after_landing"):
            tot[k] += s.get(k) or 0.0
    allh = sum(tot[k] for k in ("unsplit", "admission", "review", "to_checked", "after_landing"))
    print(f"\n== {label}: {len(sub)} PRs, {allh:,.0f} PR-hours mark->main (+{tot['rework']:,.0f} rework first->final mark)")
    for k in ("unsplit", "admission", "review", "to_checked", "after_landing"):
        print(f"  {k:14s} {tot[k]:9,.0f} PR-h  {100 * tot[k] / allh if allh else 0:5.1f}%")
    if sub and any(s["main"] for s in sub):
        xs = [s["r2m"] for s in sub]
        print(f"  ready-to-main median {statistics.median(xs):.1f} h, p90 {pct(xs, .9):.1f} h; from first mark median {statistics.median([s['r2m_first'] for s in sub]):.1f}, p90 {pct([s['r2m_first'] for s in sub], .9):.1f}")
