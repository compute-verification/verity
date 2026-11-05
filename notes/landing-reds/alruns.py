"""After-landing shard runs (check.py --deferred ... --shard K/N) since 29 Oct 17:00Z, grouped by commit: each shard's steps
from its result record, and whether the commit was green on the merge gate (every step passed but policy.toml's off_gate
ones). Writes /tmp/reds/alruns.json."""
import collections, json, os, sqlite3, subprocess
from concurrent.futures import ThreadPoolExecutor

OFF_GATE = {"circuit-check", "ratchet-gate"}
db = sqlite3.connect(os.path.expanduser("~/.research/store/catalog.sqlite"), check_same_thread=False)
runs = []
for id_, state, rc, start, wall, commit, mj in db.execute(
        "select id,state,rc,start_utc,wall_s,source_commit,manifest_json from attempts where start_utc >= '2026-10-29T17:00' "
        "order by start_utc"):
    m = json.loads(mj)
    argv = m.get("argv") or []
    if not any(a.endswith("check.py") for a in argv) or "--shard" not in argv or "--deferred" not in argv:
        continue
    shard = argv[argv.index("--shard") + 1]
    runs.append(dict(id=id_, state=state, rc=rc, start=start, wall_min=round((wall or 0) / 60), commit=(commit or "")[:12],
                     shard=shard, cached="--no-cache" not in argv, result=(m.get("outputs") or {}).get("result"),
                     machine=(m.get("conditions") or {}).get("exec.machine")))


def steps(r):
    p = f"/tmp/reds/results/{r['id']}.json"
    if os.path.exists(p):
        return json.load(open(p)).get("steps")
    if not r["result"]:
        return None
    out = subprocess.run(["research", "data", "show", r["result"], "--json"], capture_output=True, text=True, timeout=300)
    if out.returncode:
        return None
    meta = json.loads(out.stdout)["manifest"]["meta"]
    json.dump({k: meta.get(k) for k in ("steps", "reuse", "validation", "verdicts", "deferred")}, open(p, "w"))
    return meta.get("steps")


with ThreadPoolExecutor(16) as pool:
    for r, s in zip(runs, pool.map(steps, runs)):
        r["steps"] = {x["step"]: x.get("status") for x in (s or []) if not x["step"].startswith("preflight")}
by = collections.defaultdict(list)
for r in runs:
    by[r["commit"]].append(r)
out = []
for c, rs in by.items():
    st = {}
    for r in rs:
        for k, v in r["steps"].items():
            st[k] = "passed" if v == "passed" or st.get(k) == "passed" else v
    gated = {k: v for k, v in st.items() if k not in OFF_GATE}
    red = sorted(k for k, v in gated.items() if v not in ("passed", "skipped", None))
    out.append(dict(commit=c, first=min(r["start"] for r in rs), shards=len(rs), steps=st,
                    gated_red=red, complete=all(r["steps"] for r in rs), ids=[r["id"] for r in rs]))
json.dump(out, open("/tmp/reds/alruns.json", "w"), indent=1)
done = [o for o in out if o["complete"] and o["steps"]]
print(len(runs), "shard runs,", len(out), "commits,", len(done), "with every shard's record")
print("gated green:", sum(not o["gated_red"] for o in done), "red:", sum(bool(o["gated_red"]) for o in done))
print(collections.Counter(k for o in done for k in o["gated_red"]).most_common())
for o in sorted(done, key=lambda o: o["first"]):
    print(o["first"][:16], o["commit"], o["shards"], "GREEN" if not o["gated_red"] else "red: " + ",".join(o["gated_red"]))
