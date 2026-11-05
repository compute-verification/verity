import subprocess, json, re, sys
R="/agent/repos/verity"
def git(*a): return subprocess.run(["git","-C",R,*a],capture_output=True,text=True).stdout
out={}
for ref in ("checked","main"):
    rows=[l.split("\t") for l in open(f"act_{ref}.tsv").read().splitlines()]
    rows.sort()
    first={}
    miss=0
    for ts,typ,before,after,actor in rows:
        if typ not in("push","force_push","merge_queue_merge","branch_creation"): continue
        rng = after if before.startswith("0000") else f"{before}..{after}"
        if typ=="branch_creation": rng=f"{after}~5..{after}"
        lst=git("rev-list","--first-parent",rng).split()
        if not lst: miss+=1
        for c in lst:
            first.setdefault(c,(ts,actor,typ))
    print(ref,len(first),"commits",miss,"empty pushes", file=sys.stderr)
    out[ref]=first
# queue merges on checked/main first-parent
recs=[]
for ref in ("checked","main"):
    for l in git("log",f"origin/{ref}","--first-parent","--since=2026-10-15","--format=%H %P\t%cI\t%s").splitlines():
        hp,ci,s=l.split("\t",2); hp=hp.split(); m=re.match(r"Queue: #(\d+)",s)
        if not m or len(hp)<3: continue
        recs.append(dict(branch=ref,commit=hp[0],head=hp[2],pr=int(m.group(1)),committed=ci,subject=s[:120],
            on_checked=out["checked"].get(hp[0],[None])[0], on_main=out["main"].get(hp[0],[None])[0]))
d={}
for r in recs: d.setdefault(r["commit"],r)
json.dump(list(d.values()),open("landed.json","w"),indent=0)
print(len(d),"queue merges", sum(1 for r in d.values() if r["on_main"]), "with main time", sum(1 for r in d.values() if r["on_checked"]),"with checked time")
