#!/usr/bin/env bash
# Must-reject battery against both full-VU provers (lane red-team-vu). Regenerates every adversarial witness /
# trace deterministically from the committed vu-k1536-neg operands and runs the REAL provers + verifiers:
#   A  verity-gkr run-vu on redteam.vu_forge_a directories (PCS and --clear), expected verdicts in a/manifest.json
#   B  redteam-vu-b battery (Plonky3 uni-stark prove + verify, Goldilocks and BabyBear) + challenger audit
# Exit 1 if any case's verdict differs from `expected` (a reject-expected case ACCEPTED is a forgery).
#   fixtures/redteam/vu-battery/v1/run_battery.sh [OUT_DIR]        (on a pod: research run --on vy-cpu3 -- bash <this>)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
OUT=${1:-${RESEARCH_RUN_DIR:-/tmp/vu-battery}}
export RESEARCH_RUN_DIR="$OUT"
bash "$ROOT/backends/redteam/run_vu_battery.sh"
python3 - "$HERE" "$OUT" <<'EOF'
import json, sys, pathlib
fx, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
bad = []
exp = {c["id"]: c["expected"] for c in json.load(open(fx / "a" / "manifest.json"))["cases"]}
for line in (out / "a_results.jsonl").read_text().splitlines():
    r = json.loads(line)
    got = "accept" if r["verify"].startswith("Ok") else "reject"
    if got != exp[r["id"]]:
        bad.append(f"A {r['id']} {r['mode']}: expected {exp[r['id']]}, got {r['verify']}")
b = json.load(open(out / "vu_battery_b.json"))
for c in b["cases"]:
    if c["status"] not in ("ok", "UNSAT"):
        bad.append(f"B {c['field']}/{c['id']}: {c['status']} (expected {c['expected']}, verifier accepted={c['verifier']['accepted']})")
ch = json.load(open(out / "challenger_audit.json"))
for r in ch["report"]:
    if "uniform" in r and not r["uniform"] and "UPSTREAM" not in r["challenger"]:
        bad.append(f"challenger {r['challenger']}: not uniform {r}")
print("\n".join(bad) if bad else "vu-battery: every case as expected")
sys.exit(1 if bad else 0)
EOF
