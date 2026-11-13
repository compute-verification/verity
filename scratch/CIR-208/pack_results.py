#!/usr/bin/env python3
"""pack_results.py WORK > results.json: the sweep's results a head, without each head's whole table of ids.

For each head: its sha, the files its merge with the base stops on, the test's line, what the test's functions find
(`new`, `gone`, `moved`, `twice`), how many ids it registers and which it adds to or takes from the base's, and the same
for the forced pass where there is one."""
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
jobs = [line.split()[:2] for line in (work / "jobs.txt").read_text().splitlines() if line.strip()]
base = set(json.loads((work / "out" / "main.json").read_text())["ids"])


def brief(r: dict) -> dict:
    out = {k: r[k] for k in ("base", "heads", "conflicts", "forced", "pytest", "pytest_failed", "superseded", "entries", "new",
                             "gone", "moved", "twice", "extra", "import_errors") if k in r}
    for k in ("merge_error", "force_error", "detail_error", "extra_error"):
        if r.get(k):
            out[k] = r[k].strip().splitlines()[-1][:200]
    if "ids" in r:
        out["registered"] = len(r["ids"])
        out["ids_added"] = sorted(set(r["ids"]) - base)
        out["ids_removed"] = sorted(base - set(r["ids"]))
    return out


res = {}
for name, sha in jobs:
    res[name] = {"sha": sha, "merged": brief(json.loads((work / "out" / f"{name}.json").read_text()))}
    f = work / "out-forced" / f"{name}.json"
    if f.exists():
        res[name]["with_its_side_of_each_conflict"] = brief(json.loads(f.read_text()))
for f in sorted((work / "out-controls").glob("*.json")):
    res[f.stem] = {"merged": brief(json.loads(f.read_text()))}
print(json.dumps(res, indent=1))
