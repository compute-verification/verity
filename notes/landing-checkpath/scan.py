import json, os
root = "/workspace/research/runs"
for d in sorted(os.listdir(root)):
    if not (d.startswith("r202611") or d.startswith("r2026102") or d.startswith("r2026103")):
        continue
    p = os.path.join(root, d)
    try:
        job = json.load(open(os.path.join(p, "job.json")))
    except Exception:
        continue
    cmd = (job.get("execution") or {}).get("command") or []
    s = " ".join(cmd)
    if "check.py" not in s:
        continue
    out = {"id": d, "commit": (job.get("computation") or {}).get("tree_sha"), "cmd": s[s.find("check.py"):][:300]}
    try:
        r = json.load(open(os.path.join(p, "result.json")))
        out["kind"] = r.get("kind")
        out["steps"] = {x["step"]: [x.get("status"), x.get("seconds")] for x in r.get("steps", [])}
        out["reuse"] = (r.get("reuse") or {}).get("summary")
        dfr = r.get("deferred") or {}
        out["parts"] = dfr.get("parts"); out["shard"] = [dfr.get("k"), dfr.get("n")]
    except Exception as e:
        out["noresult"] = str(e)[:60]
    try:
        st = json.load(open(os.path.join(p, "status.json")))
        tr = st.get("transitions") or []
        out["start"] = tr[0].get("t_utc") if tr else None
        out["end"] = tr[-1].get("t_utc") if tr else None
        out["state"] = tr[-1].get("state") if tr else None
        out["rc"] = tr[-1].get("rc") if tr else None
    except Exception:
        pass
    print(json.dumps(out))
