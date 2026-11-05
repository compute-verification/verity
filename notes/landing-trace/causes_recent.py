"""Attribute every PR-hour of each stage to a cause, tick by tick, and find the incidents that cost the most PR-hours."""
import bisect
import collections
import json
import re
from datetime import datetime, timedelta, timezone


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


T = json.load(open("/tmp/trace/ticks.json"))
ticks, stopped = T["ticks"], sorted(ts(s) for s in T["stopped"])
rows = json.load(open("/tmp/trace/stages.json"))
raw = {r["pr"]: r for r in json.load(open("/tmp/trace/trace_v1_raw.json"))["records"]}
for t in ticks:
    t["_end"], t["_prev"] = ts(t["end"]), ts(t["prev_end"])
ends = [t["_end"] for t in ticks]
NOPOST = ts("2026-11-03T19:19:00Z")
CI239 = ts("2026-10-28T00:16:00Z")
CHECKED_ERA = ts("2026-10-29T17:20:00Z")


def classify_out(why, what):
    w = why
    if "preflight onto `next` hasn't run" in w or "preflights took their" in w:
        return "admission: preflight budget (900 s) spent"
    if "conflicts with `next` in" in w:
        files = re.findall(r"in ([^()]*?)(?: \(touched| \(|$)", w)
        return "admission: conflict on next"
    if "didn't merge" in w or "isn't admitted" in w or "contains #" in w or "at an older head" in w:
        return "admission: stacked on a PR not admitted/merged"
    if "isn't marked ready" in w or "waits for your mark" in w:
        return "admission: carried PR not marked"
    if "preflight" in w and ("refused" in w or "failed" in w):
        return "admission: preflight refused"
    return "admission: other refusal"


# per-tick state
state = []
al_last, al_last_t = None, None
for i, t in enumerate(ticks):
    s = {}
    s["stopped"] = bisect.bisect_right(stopped, t["_end"]) - bisect.bisect_right(stopped, t["_prev"]) >= 2
    # to-checked blocker at the bottom of the prefix
    nv = t["no_verdict"]
    if s["stopped"]:
        s["block"] = ("drive stopped", None)
    elif nv:
        b = nv[0]
        if b["why"].startswith("held by"):
            s["block"] = ("behind a held PR", b["pr"]) if "brake" not in b["why"] and "CI-239" not in b["why"] else ("CI-239 brake (hold on next's bottom)", b["pr"])
        else:
            s["block"] = ("behind a PR without a land verdict", b["pr"])
    elif t["nothing_lands"]:
        s["block"] = ("behind a Lean check (prefix waits)", t["nothing_lands"][0]["pr"])
    elif t["landed_checked"] or t["landed_main"]:
        s["block"] = ("landing (tick cadence)", None)
    else:
        s["block"] = ("next being rebuilt / no prefix", None)
    s["lean_wait"] = set(t["lean_wait"])
    s["waiting"] = set(t["waiting_prs"])
    s["out"] = {o["pr"]: classify_out(o["why"], o["what"]) for o in t["out_next"]}
    # after-landing state
    st = None
    for x in t["stale"]:
        why = x["why"]
        red = "red: nothing named red" not in why
        if why.startswith("its run"):
            st = "after-landing run going, a red named" if red and "has not run" not in why.split("red:")[1][:40] and not why.split("red:")[1].strip().startswith("nothing") else "after-landing run going"
        elif why.startswith("no after-landing run is going"):
            if red:
                st = "red on checked, no run going (fix or revert pending)"
            else:
                st = "no run drive can see (ci-no-postland: ci's hand run)" if t["_end"] >= NOPOST else "no after-landing run going"
    if t["postland"]:
        st = "after-landing run going"
    if st is None and al_last and (t["_end"] - al_last_t) <= timedelta(hours=3):
        st = al_last
    if st is not None:
        al_last, al_last_t = st, t["_end"]
    if s["stopped"]:
        st = "drive stopped"
    s["al"] = st or ("unclassified (no ring yet)")
    state.append(s)


def spans(a, b):
    """(tick index, overlap hours) of ticks overlapping [a, b)."""
    if not a or not b or b <= a:
        return
    i = bisect.bisect_left(ends, a)
    while i < len(ticks):
        lo, hi = max(a, ticks[i]["_prev"]), min(b, ticks[i]["_end"])
        if hi > lo:
            yield i, (hi - lo).total_seconds() / 3600
        if ticks[i]["_end"] >= b:
            break
        i += 1


REC = collections.Counter(); RECN = collections.Counter()
pareto = collections.Counter()          # (stage, cause) -> PR-hours
incident = collections.Counter()        # (cause, blocker) -> PR-hours
incident_span = {}
per_pr = collections.defaultdict(collections.Counter)
week = collections.Counter()
for r in rows:
    n = r["pr"]
    fm, end = ts(r["fm"]), ts(r["end"])
    rr = raw[n]
    adm, ver, chk = ts(rr["admitted"]), ts(rr["verdict"]), ts(rr["checked"])
    chk = min(max(chk, fm), end) if chk else None
    a = adm if adm and fm <= adm <= (chk or end) else None
    v = None
    if rr["checked_how"] != "lowrisk" and ver:
        v = min(max(ver, a or fm), chk or end)
    segs = []
    if a:
        segs.append(("admission", fm, a))
    start_next = a or fm
    if v and v > start_next:
        segs.append(("review", start_next, v))
        start_next = v
    elif not ver and rr["checked_how"] != "lowrisk" and not chk:
        segs.append(("review", start_next, end))
        start_next = end
    segs.append(("to_checked", start_next, chk or end))
    if chk:
        segs.append(("after_landing", chk, end))
    for stage, lo, hi in segs:
        for i, hrs in spans(lo, hi):
            s = state[i]
            if stage == "admission":
                cause = s["out"].get(n) or ("admission: waiting for its preflight turn" if n in s["waiting"] else "admission: queued (no refusal logged)")
                blocker = None
            elif stage == "review":
                cause, blocker = "review: no land verdict at its head", n
            elif stage == "to_checked":
                if n in s["out"]:
                    cause, blocker = s["out"][n] + " (pre-1 Nov: admission unsplit)" if not a else s["out"][n], None
                elif n in s["lean_wait"]:
                    cause, blocker = "to_checked: its own Lean check", None
                elif s["block"][1] == n:
                    cause, blocker = "to_checked: it is the bottom blocker itself", n
                else:
                    cause, blocker = "to_checked: " + s["block"][0], s["block"][1]
            else:
                cause, blocker = "after_landing: " + s["al"], None
            pareto[(stage, cause)] += hrs
            if ticks[i]["_end"] >= ts("2026-11-02T00:00:00Z"): REC[cause]+=hrs; RECN[(cause,n)]+=hrs
            per_pr[n][cause] += hrs
            if ticks[i]["_end"] >= ts("2026-10-29T17:30:00Z"):
                week[(stage, cause)] += hrs
            if blocker is not None and blocker != n:
                incident[(cause, blocker)] += hrs
                lo_, hi_ = incident_span.get((cause, blocker), (ticks[i]["_prev"], ticks[i]["_end"]))
                incident_span[(cause, blocker)] = (min(lo_, ticks[i]["_prev"]), max(hi_, ticks[i]["_end"]))

tot = sum(pareto.values())

tot=sum(REC.values())
print("since 2 Nov 00:00Z: %.0f PR-h over %.1f days"%(tot,(ticks[-1]["_end"]-ts("2026-11-02T00:00:00Z")).total_seconds()/86400))
for c,v in REC.most_common(16): print("  %7.0f %5.1f%%  %s"%(v,100*v/tot,c))
json.dump({f"{c}|{n}":v for (c,n),v in RECN.items()}, open("/tmp/trace/rec_since_1102.json","w"))
