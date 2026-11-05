import json,collections
rows=json.load(open("labels.json")); H=json.load(open("heads_landed.json")); E=json.load(open("ejections.json"))
landed={o["pr"]:o for o in H}
marks=collections.defaultdict(dict)
holds=collections.defaultdict(set); unready=collections.defaultdict(list)
for r in rows:
    if r["key"]=="ready" and r["value"] is True and r["ts"]>="2026-10-12T06":
        marks[r["pr"]].setdefault(r["head"],r["ts"]); marks[r["pr"]][r["head"]]=min(marks[r["pr"]][r["head"]],r["ts"])
    if r["key"]=="reviewed" and r["value"]=="hold": holds[r["pr"]].add(r["head"])
    if r["key"]=="ready" and r["value"] is False: unready[(r["pr"],r["head"])].append(r)
ej=collections.defaultdict(dict)
for e in E: ej[e["pr"]].setdefault(e["sha"],e)
n_heads=collections.Counter(); fate=collections.Counter(); per_pr=[]
for pr,o in landed.items():
    hs=sorted(marks.get(pr,{}).items(),key=lambda kv:kv[1])
    n=len(hs); n_heads[min(n,8)]+=1
    for h,ts in hs:
        if h==o["head"]: continue
        e=next((v for s,v in ej[pr].items() if h.startswith(s)),None)
        if e: f="ejected:"+e["kind"]
        elif h in holds[pr]: f="held by a reviewer"
        elif unready.get((pr,h)): f="unmarked (eject/withdraw label): "+unready[(pr,h)][0]["by"]
        else: f="re-pushed by owner (no ejection, hold or unmark)"
        fate[f]+=1
    per_pr.append(n)
print("PRs landed in window:",len(per_pr),"; marked heads per PR:",sorted(n_heads.items()), "; mean %.2f"%(sum(per_pr)/len(per_pr)))
print("re-heads (marked heads that didn't land) by fate:")
for f,c in fate.most_common(): print(f"  {c:5d} {f}")
print("total extra marked heads:",sum(fate.values()))
