#!/usr/bin/env bash
# At a head (run in its source): the suites the change from MAIN touches, whole, each suite's log under $OUT/logs as it ends;
# then circuit-check on the named sidecars; then the adapter's ratchet gate.  Usage: checks.sh MAIN SIDECAR...
set -o pipefail
MAIN="$1"; shift
git rev-parse HEAD | tee "$OUT/head.txt"
uv sync -q --all-packages --extra torch-cpu || exit 3
uv run --no-sync tools/verity/check/suites.py --changed "$MAIN" --mem-gb 60 --logs "$OUT/logs" --json "$OUT/suites.json" \
  > "$OUT/suites.txt" 2>&1
a=$?
echo "suites rc $a" | tee "$OUT/rc.txt"
uv run --no-sync circuit-check "$@" --jobs "$(nproc)" --mem-gb 60 --out "$OUT/cc.json" > "$OUT/cc.txt" 2>&1
b=$?
echo "circuit-check rc $b" | tee -a "$OUT/rc.txt"
uv run --no-sync python adapters/vllm/tools/ratchet_gate.py "$MAIN" HEAD --json > "$OUT/gate.json" 2> "$OUT/gate.err"
c=$?
uv run --no-sync python adapters/vllm/tools/ratchet_gate.py "$MAIN" HEAD > "$OUT/gate.txt" 2>&1
echo "gate rc $c" | tee -a "$OUT/rc.txt"
exit $((a | b | c))
