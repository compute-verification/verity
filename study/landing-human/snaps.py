import json,re,collections
from datetime import datetime,UTC
d=json.load(open('thread.json'))
snaps=[]
for m in d:
    t=m.get("text","")
    if "and not landed on `checked`" not in t: continue
    ts=datetime.fromtimestamp(float(m["ts"]),UTC)
    thr=re.search(r"marked ready ([\d.]+) h ago",t)
    body=re.sub(r"\n\n_\(quiet\)_","",t.split(":",1)[1])
    groups=[]
    for g in re.split(r";\s*",body):
        prs=[int(x) for x,_ in re.findall(r"#(\d+) \(([\d.]+) h\)",g)]
        held=re.search(r"held by (.*)$",g.strip())
        groups.append((prs,held.group(1) if held else ""))
    snaps.append(dict(ts=ts,thr=float(thr.group(1)) if thr else None,groups=groups))
snaps.sort(key=lambda s:s["ts"])
print(collections.Counter(s["thr"] for s in snaps))
gaps=[(b["ts"]-a["ts"]).total_seconds()/60 for a,b in zip(snaps,snaps[1:])]
print("median gap min",sorted(gaps)[len(gaps)//2], "max", max(gaps))
# episodes: blocker = '#N at SHA: no land verdict'
ep=collections.OrderedDict()
for i,s in enumerate(snaps):
    for prs,held in s["groups"]:
        m=re.match(r"#(\d+) at ([0-9a-f]+): no land verdict",held)
        if m:
            k=(int(m.group(1)),m.group(2))
            e=ep.setdefault(k,dict(first=s["ts"],last=s["ts"],max=0,snaps=0,prs=set()))
            e["last"]=s["ts"]; e["max"]=max(e["max"],len(prs)); e["snaps"]+=1; e["prs"].update(prs)
print(len(ep),"blocking heads (no land verdict) named in snapshots")
rows=[]
for (pr,sha),e in ep.items():
    dur=(e["last"]-e["first"]).total_seconds()/3600
    rows.append((pr,sha,e["first"].isoformat()[:16],dur,e["max"],len(e["prs"]),e["snaps"]))
rows.sort(key=lambda r:-r[4])
for r in rows[:30]: print(f"#{r[0]} at {r[1]} first {r[2]} seen for {r[3]:.1f} h, max {r[4]} PRs waiting (2h+ marked), {r[5]} distinct, {r[6]} snaps")
json.dump([dict(pr=r[0],sha=r[1],first=r[2],hours=r[3],max_waiting=r[4],distinct=r[5],snaps=r[6]) for r in rows],open("blockers.json","w"),indent=0)
