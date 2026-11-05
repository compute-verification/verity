import json, sys, glob, os
for rid in sys.argv[1:]:
    d = f"/workspace/research/runs/{rid}"
    steps = []
    try:
        for l in open(f"{d}/steps.jsonl"):
            s = json.loads(l)
            if s.get("status") not in ("passed", "cached", "skipped"):
                steps.append((s["step"], s.get("status"), s.get("rc")))
    except Exception as e:
        steps.append(("nosteps", str(e)[:40], None))
    suites = []
    for f in glob.glob(f"{d}/*.failed.json"):
        try:
            for r in json.load(open(f)):
                ft = r.get("failed_tests") or []
                suites.append((r.get("name"), len(ft), ft[:2]))
        except Exception:
            pass
    print(json.dumps({"id": rid, "steps": steps, "suites": suites}))
