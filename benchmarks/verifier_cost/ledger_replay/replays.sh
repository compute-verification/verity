#!/usr/bin/env bash
# How many times one PoUW period check replays the auditor's ledger, against the period's windows: replays.py writes,
# per W, an honest period of W windows as PoUW's period difftest generates one, and counts the replays its reference
# (`period_check.case`) makes; the compiled check (`pouw-period serve`, built in the checkout's package) answers the same
# case under gdb, which counts its entries into `Spot.Challenger.checked`. A node without gdb gives the reference's
# count alone, and the compiled check's is null. Run:
#   research run --as proofs --project verity --on vy-nebius-cpu-11 --source . --pool research --cwd source \
#     --mem-wait 1800 -- bash benchmarks/verifier_cost/ledger_replay/replays.sh 1 2 3
# $OUT gets replays.jsonl (one row per W: the ledger's entries and sessions, each side's verdict and replays),
# gdb-W.log, setup.log and build.log.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: replays.sh W..." >&2; exit 2; }
P=$PWD/verity/protocols/accounting/pouw/verifier/lean/check
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
uv sync -q --all-packages --extra torch-cpu
bash tools/verity/lean/setup.sh "$P" > "$O/setup.log" 2>&1 || { tail -40 "$O/setup.log" >&2; exit 1; }
export PATH="$HOME/.elan/bin:$PATH"
(cd "$P" && lake build pouw-period) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
B=$P/.lake/build/bin/pouw-period
uv run python benchmarks/verifier_cost/ledger_replay/replays.py "$S/in" "$@" > "$S/reference.jsonl"
sym=$(nm "$B" | awk '$3 ~ /Spot_Challenger_checked$/ { print $3 }' | head -1)
for w in "$@"; do
  hits=null
  if command -v gdb > /dev/null && [ -n "$sym" ]; then
    gdb -q -batch -ex "break $sym" -ex "ignore 1 1000000000" -ex "run serve < $S/in/period-$w.json > $S/reply-$w.json" \
      -ex "info breakpoints" "$B" > "$O/gdb-$w.log" 2>&1 || true
    hits=$(sed -n 's/.*breakpoint already hit \([0-9]*\) time.*/\1/p' "$O/gdb-$w.log" | head -1)
  else
    "$B" serve < "$S/in/period-$w.json" > "$S/reply-$w.json"
  fi
  python3 - "$w" "${hits:-0}" "$sym" "$S/reference.jsonl" "$S/reply-$w.json" <<'PY' | tee -a "$O/replays.jsonl"
import json, sys
w, hits, sym, reference, reply = sys.argv[1:]
row = next(r for r in map(json.loads, open(reference)) if r["windows"] == int(w))
print(json.dumps({**row, "compiled_verdict": json.load(open(reply))["verdict"],
                  "compiled_replays": None if hits == "null" else int(hits), "symbol": sym or None}))
PY
done
