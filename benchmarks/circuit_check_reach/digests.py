import json, subprocess
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True).stdout
def pins(commit):
    files = [f for f in git("ls-tree", "-r", "--name-only", commit).splitlines()
             if f.endswith(".circuit-check.json") or "/circuit_check/pins/" in f]
    out = {}
    for f in files:
        try:
            d = json.loads(git("show", f"{commit}:{f}"))
        except ValueError:
            continue
        for k, v in (d.get("definitions") or {}).items():
            if isinstance(v, dict) and "sha512" in v: out[norm(k)] = v["sha512"]
        for sec, body in d.items():
            if sec != "definitions" and isinstance(body, dict):
                for k, v in body.items():
                    if isinstance(v, dict) and "sha512" in v: out.setdefault(norm(k), v["sha512"])
    return out
def norm(s): return s.replace("(", "[").replace(")", "]")
rows = [json.loads(l) for l in open("/home/ubuntu/store/partial/cc.json.progress.jsonl")]
done = {norm(r["target"]) for r in rows}
lines = [l.split()[2] for l in open("/home/ubuntu/store/partial/cc.txt") if l.split()[:1] and l.split()[0] in ("ok", "FAIL", "unchecked")]
print("progress rows", len(rows), "log targets", len(lines))
targets = done | {norm(t) for t in lines}
a, b = pins("98e9c1de7be1"), pins("c980fe2c16d")
pa = {t for t in targets if t in a}; pb = {t for t in targets if t in b}
moved = {t for t in pa & pb if a[t] != b[t]}
print("pinned digests: at the cut", len(a), "at the run's commit", len(b))
print("run's targets with a digest pin: at the cut", len(pa), "at the run's commit", len(pb))
print("re-pinned between the cut and the run's commit (digest moved and pinned):", len(moved), sorted(moved)[:20])
