import json, collections
from datetime import datetime
def T(s): return datetime.fromisoformat(s.replace("Z","+00:00")) if s else None
H=json.load(open("heads_landed.json"))
G=collections.defaultdict(list)
for o in H: G[o["landed"]].append(o)
ev=[]
for lt,ms in sorted(G.items()):
    rv=[o for o in ms if o["verdict"] and o["mark"]]
    if not rv: continue
    ready=lambda o: max(T(o["verdict"]),T(o["mark"]))
    last=max(rv,key=ready); vs=ready(last)
    gap=(T(lt)-vs).total_seconds()/3600
    waited=[(vs-ready(o)).total_seconds()/3600 for o in rv if o is not last]
    ev.append(dict(landed=lt,n=len(ms),nrv=len(rv),last_pr=last["pr"],last_reviewer=last["reviewer"],last_ready=vs.isoformat(),
        last_mark_to_verdict=(T(last["verdict"])-T(last["mark"])).total_seconds()/3600, gap_h=gap, waited_sum=sum(waited), waited_n=sum(1 for w in waited if w>0.25)))
json.dump(ev,open("land_events.json","w"),indent=0)
print(len(ev),"landing pushes with reviewed members")
# verdict-bound: the landing followed the last readiness within 90 min
vb=[e for e in ev if e["gap_h"]<1.5]
print(len(vb),"verdict-bound pushes (landing within 1.5h of the last member's verdict/mark)")
print("PR-hours waiting behind the last-ready member in verdict-bound pushes: %.0f"%sum(e["waited_sum"] for e in vb))
big=sorted(vb,key=lambda e:-e["waited_sum"])[:25]
for e in big: print(f'{e["landed"][:16]} n={e["n"]:3d} waited={e["waited_n"]:3d} PRh={e["waited_sum"]:6.0f} last=#{e["last_pr"]} by {e["last_reviewer"]} mark->verdict {e["last_mark_to_verdict"]:.1f}h gap {e["gap_h"]:.2f}h')
