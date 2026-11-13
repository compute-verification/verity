"""Print a circuit-check report (its --out JSON) as one line a target and its summary: python3 cc_report.py REPORT.json"""
import json
import sys

r = json.load(open(sys.argv[1]))
s = r["summary"]
refused = s.get("q_call_refused", {})
print({k: v for k, v in s.items() if k != "q_call_refused"})
print("Q_call v1 refuses", len(refused), sorted(set(refused.values())))
for t in r["targets"]:
    q = (t.get("partition") or {}).get("q_call") or {}
    warn = sorted({f"{w['check']}/{w['code']}" for w in t.get("warnings", [])})
    qc = refused.get(t["target"]) or (f"{q['units']}u/{q['committed_bits']}b" if "units" in q else "by its calls")
    b = t.get("block") or {}
    print(f"{t['target']} | {t.get('conformance')} | failures {len(t.get('failures', []))} | warnings {', '.join(warn) or 'none'} "
          f"| Q_call {qc} | block k {b.get('k')} of {b.get('k_max')}")
