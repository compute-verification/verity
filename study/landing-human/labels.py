import json, glob, os, collections
rows=[]
for f in glob.glob(os.path.expanduser("~/.research/store/labels/pr_*/*.json")):
    j=json.load(open(f))
    t=j["target"]; pr,_,head=t[3:].partition("@")
    rows.append(dict(pr=int(pr),head=head,key=j["key"],by=j["by"],ts=j["ts"],value=j["value"],ref=j.get("ref")))
json.dump(rows,open("labels.json","w"))
c=collections.Counter(r["key"] for r in rows); print(c)
since="2026-10-22"
rec=[r for r in rows if r["ts"]>=since]
print(len(rec), collections.Counter(r["key"] for r in rec))
print(collections.Counter((r["key"],str(r["value"])[:40]) for r in rec if r["key"] in("reviewed","ready")).most_common(20))
