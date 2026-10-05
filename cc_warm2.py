"""Rerun, with the cache at DIR, only the `--all` tasks listed in STORED (kind, target); write the whole task list to TASKS; print stats."""
import json, sys, time
from pathlib import Path
from circuit_check import cache as CC, checks as CK, cli
stored = {tuple(x) for x in json.loads(Path(sys.argv[1]).read_text())}
cache, tasks_out = Path(sys.argv[2]), Path(sys.argv[3])
CC.trace(cli.REPO)
real = cli._tasks
t0 = time.time()


def tasks(*a):
    ts = real(*a)
    tasks_out.write_text(json.dumps([[t[0], t[1], sorted(cli._info(t).get("units", ()))] for t in ts]))
    return [t for t in ts if (t[0], t[1]) in stored]


cli._tasks = tasks
rep = cli.run([], CK.Options(), everything=True, log=None, cache=cache)
s, c = rep["summary"], rep["cache"]
print(json.dumps({"targets": len(rep["targets"]), "hits": c["hits"], "stored": c["stored"], "missed": [t["target"] for t in rep["targets"] if "cache" not in t],
                  "untraced": c["untraced"][:3], "errors": c["errors"][:5], "changed": c["changed"][:5],
                  "failures": s["failures"], "new": len(s["new_failures"]), "warnings": {k: v["total"] for k, v in s["warnings"].items()},
                  "seconds": round(time.time() - t0, 1)}))
