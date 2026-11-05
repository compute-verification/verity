import json,re,collections
from datetime import datetime,timezone
d=json.load(open('thread.json'))
E=[]
for m in d:
    t=re.sub(r"<!subteam\^\w+>\s*","",m.get("text",""))
    ts=datetime.fromtimestamp(float(m["ts"]),timezone.utc).isoformat()
    for mm in re.finditer(r"#(\d+) at ([0-9a-f]{7,40}) is out of `next`: (.*?)(?:\n|$)",t):
        pr,sha,why=int(mm.group(1)),mm.group(2),mm.group(3)
        kind="other"; files=[]
        if why.startswith("conflicts with `next` in"):
            kind="conflict"
            body=why[len("conflicts with `next` in "):]
            files=[f.strip() for f in re.split(r",\s*(?![^()]*\))",re.sub(r"\s*\(touched by[^)]*\)","",body)) if f.strip()]
            files=[re.sub(r" and \d+ more.*","",f).strip(" .") for f in files]
        elif "contains #" in why or "it contains" in why: kind="contains"
        elif "landing gate" in why: kind="gate"
        elif "preflight" in why: kind="preflight"
        E.append(dict(ts=ts,pr=pr,sha=sha,kind=kind,files=files,why=why[:300]))
json.dump(E,open("ejections.json","w"),indent=0)
W="2026-10-22T17:00"
E=[e for e in E if e["ts"]>=W]
# dedupe per (pr,sha)
U={}
for e in E: U.setdefault((e["pr"],e["sha"]),e)
print(len(E),"ejection messages;",len(U),"distinct (PR,head)",len({e['pr'] for e in U.values()}),"PRs")
print(collections.Counter(e["kind"] for e in U.values()))
def cat(f):
    n=f.rsplit("/",1)[-1]
    if n=="lean-audit.json": return "lean-audit.json (lock)"
    if n=="TRUSTED.sha256": return "TRUSTED.sha256 (pin)"
    if n in("GLOSSARY.md","ARCHITECTURE.md","AGENTS.md"): return n
    if n=="README.md": return "README.md (any)"
    if n.endswith(".md"): return "other .md"
    if n.endswith(".circuit-check.json") or n.endswith(".json") and "pin" in n: return "circuit-check pins/json"
    if n.endswith(".json"): return "other .json"
    if n.endswith(".lean"): return ".lean"
    if n.endswith(".py") and "/tests/" in f or n.startswith("test_"): return "test .py"
    if n.endswith(".py"): return "source .py"
    if n.endswith(".toml"): return ".toml"
    return "other"
C=collections.Counter(); F=collections.Counter(); P=collections.defaultdict(set)
for e in U.values():
    if e["kind"]!="conflict": continue
    cs={cat(f) for f in e["files"]}
    for c in cs: C[c]+=1; P[c].add(e["pr"])
    for f in e["files"]: F[f]+=1
print("conflict ejections by file category (an ejection counted once per category it touches):")
for c,n in C.most_common(): print(f"  {n:4d} ejections, {len(P[c]):4d} PRs  {c}")
print("top files:")
for f,n in F.most_common(30): print(f"  {n:4d} {f}")
only_gen=sum(1 for e in U.values() if e["kind"]=="conflict" and e["files"] and all(cat(f) in("lean-audit.json (lock)","TRUSTED.sha256 (pin)") for f in e["files"]))
print("conflict ejections whose every file is a lock or pin:",only_gen)
