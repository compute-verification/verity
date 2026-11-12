"""Each by-calls root in a circuit-check report: its verdict (failed when it has failures, unchecked when only warnings), and
each callee row whose family is FAMILY (default DsaPositionsMerge_v1), or every row with --all.
python3 - REPORT [FAMILY|--all] < cir255_bycalls.py"""
import json
import sys

d = json.load(open(sys.argv[1]))
fam = sys.argv[2] if len(sys.argv) > 2 else "DsaPositionsMerge_v1"
print("summary", json.dumps(d.get("summary"))[:600])
for t in d["targets"]:
    if t.get("mode") != "by-calls":
        continue
    fails, warns = t.get("failures", []), t.get("warnings", [])
    verdict = "failed" if fails else ("unchecked" if warns else "passed")
    print(f"ROOT {t['definition']}: {verdict} ({len(fails)} failures, {len(warns)} warnings)")
    for f in fails:
        print("   failure", f.get("name", "")[:300])
    for r in t.get("by_calls", {}).get("callees", []):
        if fam == "--all" or r.get("family") == fam:
            print("   ", json.dumps({k: r.get(k) for k in ("callee", "family", "nodes", "binding", "sidecar", "sizes", "verdict", "why")}))
want = {r.get("binding") for t in d["targets"] if t.get("mode") == "by-calls" for r in t.get("by_calls", {}).get("callees", [])
        if r.get("family") == fam}
for t in d["targets"]:
    if t.get("definition") in want:
        print(f"RELIED {t['definition']}: failures {len(t.get('failures', []))}, warnings {[w.get('code') for w in t.get('warnings', [])]}")
