#!/usr/bin/env bash
# How many times one PoUW period check replays the auditor's ledger, against the period's windows: replays.py writes,
# per W, an honest period of W windows as PoUW's period difftest generates one, and counts the replays its reference
# (`period_check.case`) makes; the compiled check (`pouw-period serve`, built in the checkout's package) answers the same
# case under calls.py, which counts its entries into `Spot.Challenger.checked`, the replay. It runs in one-stage's
# `verifier_env` (a Lean panic aborts it), and a panic line on its stderr (`lean_child.LEAN_PANIC`), a signal or a status
# outside what `serve` documents (`verifier.SERVE_EXITS`, `lean_child.LEAN_EXIT`) ends the job before its row is written.
# Run:
#   research run --as proofs --project verity --on vy-nebius-cpu-11 --source . --pool research --cwd source \
#     --mem-wait 1800 -- bash benchmarks/verifier_cost/ledger_replay/replays.sh 1 2 3
# $OUT gets replays.jsonl (one row per W: the ledger's entries and sessions, and each side's verdict and replays),
# setup.log and build.log.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: replays.sh W..." >&2; exit 2; }
H=benchmarks/verifier_cost/ledger_replay
P=$PWD/verity/protocols/accounting/compute/pouw/verifier/lean/check
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
uv sync -q --all-packages --extra torch-cpu
bash tools/verity/lean/setup.sh "$P" > "$O/setup.log" 2>&1 || { tail -40 "$O/setup.log" >&2; exit 1; }
export PATH="$HOME/.elan/bin:$PATH"
(cd "$P" && lake build pouw-period) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
B=$P/.lake/build/bin/pouw-period
sym=$(nm "$B" | awk '$3 ~ /Spot_Challenger_checked$/ { print $3 }' | head -1)
[ -n "$sym" ] || { echo "replays.sh: $B has no function Spot.Challenger.checked to count" >&2; exit 1; }
. verity/protocols/proof/spot/verifier/one_stage/verifier_env.sh
uv run python "$H/replays.py" "$S/in" "$@" > "$S/reference.jsonl"
for w in "$@"; do
  $(verifier_env) python3 "$H/calls.py" "$B" "$sym" "$S/in/period-$w.json" "$S/reply-$w.json" serve \
    > "$S/calls-$w.json" 2> "$S/stderr-$w.txt" || { tail -40 "$S/stderr-$w.txt" >&2; exit 1; }
  uv run python - "$w" "$S/reference.jsonl" "$S/reply-$w.json" "$S/calls-$w.json" "$S/stderr-$w.txt" <<'PY' | tee -a "$O/replays.jsonl"
import json, sys
from verity.protocols.proof.spot.verifier.one_stage import lean_child
from verity.protocols.accounting.compute.pouw.verifier import verifier as PV
w, reference, reply, calls, stderr = sys.argv[1:]
traced = json.load(open(calls))
try:
    lean_child.refuse_panic(open(stderr, encoding="utf-8", errors="replace").read())
    lean_child.refuse_exit(traced["rc"], PV.SERVE_EXITS)
except lean_child.LeanPanic as e:
    sys.exit(f"replays.sh: pouw-period serve at W={w}: {e}")
row = next(r for r in map(json.loads, open(reference)) if r["windows"] == int(w))
print(json.dumps({**row, "compiled_verdict": json.load(open(reply))["verdict"], "compiled_replays": traced["entries"],
                  "compiled_rc": traced["rc"], "threads": traced["threads"]}))
PY
done
