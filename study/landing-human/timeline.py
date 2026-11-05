import json, collections, statistics as S
from datetime import datetime, timezone
def T(s): return datetime.fromisoformat(s.replace("Z","+00:00")) if s else None
rows=json.load(open("labels.json"))
L=collections.defaultdict(list)
for r in rows: L[(r["pr"],r["head"])].append(r)
land=json.load(open("landed.json"))
W0,W1=T("2026-10-22T17:00:00Z"),T("2026-11-05T17:30:00Z")
out=[]
for m in land:
    lt=T(m["on_checked"]) if m["on_checked"] else T(m["on_main"])
    if not lt or not (W0<=lt<=W1): continue
    labs=sorted(L.get((m["pr"],m["head"]),[]),key=lambda r:r["ts"])
    readys=[r for r in labs if r["key"]=="ready" and r["value"] is True and T(r["ts"])<=lt]
    mark=readys[-1] if readys else None
    markers={r["by"] for r in readys}
    verd=[r for r in labs if r["key"]=="reviewed" and r["value"]=="land" and r["by"] not in markers and T(r["ts"])<=lt]
    holds=[r for r in labs if r["key"]=="reviewed" and r["value"]=="hold"]
    v=verd[0] if verd else None
    out.append(dict(pr=m["pr"],head=m["head"],commit=m["commit"],landed=lt.isoformat(),main=m["on_main"],
        mark=mark and mark["ts"], first_mark=readys[0]["ts"] if readys else None, marker=mark and mark["by"],
        verdict=v and v["ts"], reviewer=v and v["by"], vref=v and v["ref"], nholds=len(holds),subject=m["subject"]))
json.dump(out,open("heads_landed.json","w"))
print(len(out),"landed in window;", sum(1 for o in out if o["verdict"]),"with verdict;", sum(1 for o in out if not o["verdict"]),"without (low-risk lane or own)")
def h(a,b): return (T(b)-T(a)).total_seconds()/3600
lat=[max(0,h(o["mark"],o["verdict"])) for o in out if o["verdict"] and o["mark"]]
q=lambda xs,p: sorted(xs)[int(p*(len(xs)-1))]
for name,xs in [("mark->verdict",lat),
   ("verdict->landed",[h(max(o["verdict"],o["mark"]),o["landed"]) for o in out if o["verdict"] and o["mark"]]),
   ("mark->landed(reviewed)",[h(o["mark"],o["landed"]) for o in out if o["verdict"] and o["mark"]]),
   ("mark->landed(lowrisk)",[h(o["mark"],o["landed"]) for o in out if not o["verdict"] and o["mark"]]),
   ("mark->main",[h(o["mark"],o["main"]) for o in out if o["mark"] and o["main"]])]:
    if xs: print(f"{name:26s} n={len(xs):4d} p50={q(xs,.5):6.2f} p75={q(xs,.75):6.2f} p90={q(xs,.9):6.2f} max={max(xs):6.1f} mean={S.mean(xs):5.2f} sum={sum(xs):7.0f}h")
print(collections.Counter(o["reviewer"] for o in out).most_common(25))
