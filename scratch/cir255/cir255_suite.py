import json, sys
from pathlib import Path
for rid in sys.argv[1:]:
    d = Path.home() / ".research/runs" / rid
    sj = next(iter([d / "suites.json"] if (d / "suites.json").exists() else []), None)
    row = {}
    if sj:
        rows = json.loads(sj.read_text())
        rows = rows if isinstance(rows, list) else rows.get("suites", [])
        row = next((r for r in rows if r.get("suite") == "tools/verity/circuit_check"), {})
    cj = next(iter(sorted(d.rglob("tools_verity_circuit_check.json"))), None)
    tot = n = None
    if cj:
        p = json.loads(cj.read_text()).get("passed") or {}
        tot, n = round(sum(p.values()), 1), len(p)
    print(rid, {k: row.get(k) for k in ("name", "status", "seconds", "workers", "reused")}, "tests", n, "sum_s", tot, "files", sj, cj)
