"""Join store labels, GitHub, git and drive's ticks into one landing trace: one event per PR per stage, with head SHAs."""
import bisect
import collections
import json
from datetime import datetime, timedelta, timezone

T0 = "2026-10-22T07:00:00Z"  # Thu 22 Oct 00:00 PDT


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def iso(d):
    return d.strftime("%Y-%m-%dT%H:%M:%SZ") if d else None


def hours(a, b):
    return max(0.0, (b - a).total_seconds() / 3600) if a and b else None


labels = json.load(open("/tmp/trace/labels_all.json"))
closed = json.load(open("/tmp/trace/gh_closed_rest.json"))
opened = json.load(open("/tmp/trace/gh_open_rest.json"))
T = json.load(open("/tmp/trace/ticks.json"))
ticks, stopped = T["ticks"], [ts(s) for s in T["stopped"]]
main_set = set(open("/tmp/trace/main_commits.txt").read().split())
checked_set = set(open("/tmp/trace/checked_commits.txt").read().split())
next_set = set(open("/tmp/trace/next_commits.txt").read().split())

for t in ticks:
    t["_end"], t["_start"], t["_prev"] = ts(t["end"]), ts(t["start"]), ts(t["prev_end"])
tick_ends = [t["_end"] for t in ticks]

# labels by (pr, head)
ready = collections.defaultdict(list)      # pr -> [(ts, head, value, by)]
reviews = collections.defaultdict(list)    # (pr, head) -> [(ts, value, by, ref)]
for lab in labels:
    tgt = lab.get("target", "")
    if not tgt.startswith("pr:") or "@" not in tgt:
        continue
    n, head = tgt[3:].split("@", 1)
    try:
        n = int(n)
    except ValueError:
        continue
    if lab["key"] == "ready":
        ready[n].append((ts(lab["ts"]), head, lab["value"], lab["by"]))
    elif lab["key"] == "reviewed":
        reviews[(n, head)].append((ts(lab["ts"]), lab["value"], lab["by"], (lab.get("ref") or "")[:200]))
for v in ready.values():
    v.sort()

# drive events per PR
landed_checked, landed_main, preflight, out_next, nv_block = {}, {}, collections.defaultdict(list), collections.defaultdict(list), []
for i, t in enumerate(ticks):
    for lc in t["landed_checked"]:
        for p in lc["prs"]:
            landed_checked.setdefault(p, []).append((t["_end"], lc["sha"], "lowrisk" if "low-risk" in lc["how"] else "reviewed"))
    for lm in t["landed_main"]:
        for p in lm["prs"]:
            landed_main.setdefault(p, []).append((t["_end"], lm["sha"]))
    for p, h in t["preflight"]:
        preflight[p].append((t["_end"], h))
    for o in t["out_next"]:
        out_next[o["pr"]].append((t["_end"], o["head"], o["what"], o["why"]))

gh = {x["number"]: x for x in closed}
gh.update({x["number"]: x for x in opened})


def first_after(seq, after, head=None, key=lambda e: e[0]):
    for e in seq:
        if key(e) >= after and (head is None or e[1][:len(head)] == head[:len(e[1])] or head.startswith(e[1])):
            return e
    return None


def owner_set(n, head):
    return {by.lower().lstrip("@") for (_t, h, v, by) in ready.get(n, []) if h == head and v is True}


def verdict_at(n, head, after=None):
    """The first land verdict at head by a reviewer who didn't mark it (any hold after it noted)."""
    own = owner_set(n, head)
    revs = sorted(reviews.get((n, head), []))
    land = next((r for r in revs if r[1] == "land" and r[2].lower().lstrip("@") not in own), None)
    holds = [r for r in revs if r[1] == "hold"]
    return land, holds


def stopped_between(a, b):
    i, j = bisect.bisect_left(stopped, a), bisect.bisect_right(stopped, b)
    return j - i


records, events = [], []


def ev(n, stage, at, head=None, **kw):
    if at is None:
        return
    events.append(dict(pr=n, stage=stage, ts=iso(at), head=head, **kw))


population = []
for n, x in gh.items():
    m = x.get("mergedAt")
    if m and m >= T0 and x["headRefOid"] in main_set:
        population.append((n, "main"))
    elif not m and x.get("closedAt") is None:
        marks = ready.get(n, [])
        cur = [r for r in marks if r[1] == x["headRefOid"]]
        if cur and cur[-1][2] is True:
            where = "checked" if x["headRefOid"] in checked_set else ("next" if x["headRefOid"] in next_set else "open")
            population.append((n, where))

for n, where in sorted(population):
    x = gh[n]
    head = x["headRefOid"]
    marks = ready.get(n, [])
    trues = [r for r in marks if r[2] is True]
    first_mark = trues[0] if trues else None
    final_marks = [r for r in trues if r[1] == head]
    final_mark = final_marks[0] if final_marks else None
    heads_marked = list(dict.fromkeys(r[1] for r in trues))
    unmarks = [r for r in marks if r[2] is False]
    rec = dict(pr=n, title=x["title"][:120], author=x.get("author"), where=where, head=head, opened=x["createdAt"],
               first_mark=iso(first_mark[0]) if first_mark else None, first_mark_head=first_mark[1] if first_mark else None,
               first_mark_by=first_mark[3] if first_mark else None,
               final_mark=iso(final_mark[0]) if final_mark else None, final_mark_by=final_mark[3] if final_mark else None,
               heads_marked=len(heads_marked), unmarks=len(unmarks), main=x.get("mergedAt") if where == "main" else None)
    for r in marks:
        ev(n, "marked" if r[2] is True else "unmarked", r[0], r[1], by=r[3])
    ev(n, "opened", ts(x["createdAt"]), None)
    fm = final_mark[0] if final_mark else None
    # admission: first preflight of the final head at or after its mark
    adm = None
    if fm:
        adm = next((e for e in preflight.get(n, []) if e[0] >= fm and head.startswith(e[1])), None)
        if adm is None:
            adm = next((e for e in preflight.get(n, []) if e[0] >= fm), None)
    rec["admitted"] = iso(adm[0]) if adm else None
    if adm:
        ev(n, "admitted", adm[0], head)
    land, holds = verdict_at(n, head)
    rec["verdict"] = iso(land[0]) if land else None
    rec["verdict_by"] = land[2] if land else None
    rec["holds"] = len(holds)
    if land:
        ev(n, "verdict", land[0], head, by=land[2])
    for h in holds:
        ev(n, "hold", h[0], head, by=h[2], ref=h[3][:120])
    lc = None
    if fm or where == "main":
        after = fm or ts(x["createdAt"])
        lc = next((e for e in landed_checked.get(n, []) if e[0] >= after - timedelta(hours=2)), None)
        lm = next((e for e in landed_main.get(n, []) if e[0] >= after - timedelta(hours=2)), None)
        if lc is None and lm is not None:
            lc = (lm[0], lm[1], "direct-to-main")
    rec["checked"] = iso(lc[0]) if lc else None
    rec["checked_how"] = lc[2] if lc else None
    if lc:
        ev(n, "checked", lc[0], head, sha=lc[1], how=lc[2])
    if rec["main"]:
        ev(n, "main", ts(rec["main"]), head)
    for o in out_next.get(n, []):
        if fm and o[0] >= fm:
            ev(n, "refused" if "admitted" in o[2] else "ejected", o[0], o[1], why=o[3][:200])
    records.append(rec)

json.dump(dict(records=records, events=events), open("/tmp/trace/trace_v1_raw.json", "w"))
c = collections.Counter(r["where"] for r in records)
print(len(records), c)
print("no final mark:", sum(1 for r in records if not r["final_mark"]), "no checked:", sum(1 for r in records if r["where"] == "main" and not r["checked"]))
print("events", len(events), collections.Counter(e["stage"] for e in events))
