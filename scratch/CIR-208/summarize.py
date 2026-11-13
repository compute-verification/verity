"""summarize.py: one line for each head the sweep has run whose result is not a plain pass (out/NAME.json)."""
import json, sys
from pathlib import Path
rows = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cir208/out").glob("*.json"))]
def plain(r):
    return (not r.get("conflicts") and r.get("pytest", "").startswith("3 passed") and not r.get("pytest_failed")
            and not r.get("detail_error") and not r.get("extra_error") and not r.get("new") and not r.get("gone") and not r.get("moved"))
bad = [r for r in rows if not plain(r)]
print(len(rows), "heads run;", len(bad), "not a plain pass")
for r in bad:
    print(" ", r["name"], r["heads"], "| conflicts", {k: v[:5] for k, v in r.get("conflicts", {}).items()}, "|", r.get("pytest", "")[:70],
          "| new", [n["id"] for n in r.get("new", [])], "| gone", r.get("gone"), "| moved", len(r.get("moved", [])),
          (r.get("detail_error") or "")[-300:], (r.get("extra_error") or "")[-200:])
