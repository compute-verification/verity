"""Active Linear projects whose plan has gone stale: target date passed, or open issues in milestones whose dates passed.
Usage: PYTHONPATH=/workspace/tools/research/src python3 project_audit.py [--json]"""
import json
import sys
from datetime import datetime, UTC

from research.linear import Linear
from research.secret_hint import recover

PQ = ('{ projects(first: 40, filter: {status: {type: {nin: ["completed", "canceled"]}}}) { nodes { id name targetDate } } }')
MQ = 'query($id: String!) { project(id: $id) { projectMilestones(first: 30) { nodes { id name targetDate } } } }'
IQ = ('query($id: String!) { project(id: $id) { issues(first: 250, filter: {state: {type: {nin: ["completed", "canceled"]}}}) '
      '{ nodes { identifier state { type } projectMilestone { id } } } } }')
LEAD = {"Served overhead": "BENCH", "Smaller benchmarks": "BENCH", "Smaller network": "NET", "Smaller Lean": "LEAN",
        "Smaller messaging": "COMM", "Code quality": "VLLM", "Linear and Slack": "TOP", "Runs and machines": "INF",
        "Training workloads": "TRN", "Repository organization": "TOP", "Accounting research": "CMP",
        "Landing and a verified main": "CI", "Deployment API": "MEM", "Two demos": "TOP", "Full proofs": "PRF",
        "CI switch": "CI"}


def main() -> int:
    recover()
    lin, today, rows = Linear(), datetime.now(UTC).date().isoformat(), []
    for p in lin.gql(PQ)["projects"]["nodes"]:
        issues = lin.gql(IQ, {"id": p["id"]})["project"]["issues"]["nodes"]
        ms = {m["id"]: m for m in lin.gql(MQ, {"id": p["id"]})["project"]["projectMilestones"]["nodes"]}
        late = {}
        for i in issues:
            m = ms.get((i.get("projectMilestone") or {}).get("id"))
            if m and m.get("targetDate") and m["targetDate"] < today:
                late.setdefault(m["name"], [m["targetDate"], []])[1].append(i["identifier"])
        live = [m for m in ms.values() if not m.get("targetDate") or m["targetDate"] >= today]
        lead = next((v for k, v in LEAD.items() if p["name"].startswith(k)), "?")
        rows.append({"project": p["name"], "lead": lead, "target": p.get("targetDate"),
                     "target_passed": bool(p.get("targetDate") and p["targetDate"] < today),
                     "open": sum(i["state"]["type"] != "backlog" for i in issues), "backlog": sum(i["state"]["type"] == "backlog" for i in issues),
                     "live_milestones": len(live),
                     "late_milestones": [{"name": n, "date": d, "open": len(ids), "teams": sorted({x.split('-')[0] for x in ids})}
                                         for n, (d, ids) in late.items()]})
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1))
        return 0
    for r in sorted(rows, key=lambda r: (r["lead"], r["project"])):
        flag = "STALE" if r["target_passed"] or r["late_milestones"] else "ok"
        print(f"{flag:5} {r['lead']:5} {r['project'][:50]:50} target {r['target']} open {r['open']:3} backlog {r['backlog']:3} "
              f"live_ms {r['live_milestones']}"
              + "".join(f"\n        late: {m['name'][:58]} ({m['date']}): {m['open']} open ({','.join(m['teams'])})"
                        for m in r["late_milestones"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
