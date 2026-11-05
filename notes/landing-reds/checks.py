"""Every check.py attempt in the catalog: commit, tree, flags (no-cache, deferred, shard), state, rc, wall, steps' verdicts."""
import json, sqlite3, os, collections

db = sqlite3.connect(os.path.expanduser("~/.research/store/catalog.sqlite"))
rows = []
for id_, state, rc, start, wall, host, commit, mj in db.execute(
        "select id,state,rc,start_utc,wall_s,host,source_commit,manifest_json from attempts order by start_utc"):
    m = json.loads(mj)
    argv = m.get("argv") or []
    if not any("check.py" in a for a in argv):
        continue
    j = " ".join(argv)
    c = m.get("conditions", {})
    ex = m.get("execution", {})
    out = m.get("outcome") or m.get("result") or {}
    rows.append(dict(id=id_, state=state, rc=rc, start=start, wall_min=round((wall or 0) / 60, 1),
                     machine=c.get("exec.machine"), commit=(commit or "")[:12], tree=(c.get("source.tree") or "")[:12],
                     no_cache="--no-cache" in argv, deferred="--deferred" in argv, keep_going="--keep-going" in argv,
                     shard=next((argv[i + 1] for i, a in enumerate(argv) if a == "--shard" and i + 1 < len(argv)), None),
                     verdicts_in="--verdicts-in" in argv, record="--record" in argv, argv=j[:400]))
json.dump(rows, open("/tmp/reds/checks.json", "w"), indent=1)
print(len(rows), "check.py attempts")
print(collections.Counter((r["no_cache"], r["deferred"]) for r in rows))
print(collections.Counter(r["state"] for r in rows))
k = list(json.loads(db.execute("select manifest_json from attempts where id='r20261105-154334-d0e6'").fetchone()[0]).keys())
print(k)
