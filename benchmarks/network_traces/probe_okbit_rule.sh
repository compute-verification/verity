#!/usr/bin/env bash
# A probe, not for landing: builds network-okbit-probe and runs NetworkCheck.ProbeOkBit's three forms of the rule on
# each registration under $RESEARCH_RUN_DIR/inputs (NAME-T<T>[-…].json, T its scan's buckets), then one constant's
# time at each member size of the test class's and the served class's schedules. From a clone:
#   research run --on auto --source . --cwd clone --send REG.json ... -- bash benchmarks/network_traces/probe_okbit_rule.sh
# Kept in $OUT: build.log, rule.log, const.log, machine.txt (the CPU's model and count, no name).
set -uo pipefail
export PATH=$HOME/.elan/bin:$PATH
OUT=$RESEARCH_RUN_DIR/out
mkdir -p "$OUT"
L=verity/protocols/certifier/verifier/lean
SECONDS=0
(cd "$L" && lake build network-okbit-probe) > "$OUT/build.log" 2>&1 || { tail -60 "$OUT/build.log"; echo "PROBE build failed"; exit 7; }
echo "PROBE build ${SECONDS} s"
B=$L/.lake/build/bin/network-okbit-probe
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|CPU max MHz)'; } > "$OUT/machine.txt" 2>/dev/null
cat "$OUT/machine.txt"
rc=0
for f in "$RESEARCH_RUN_DIR"/inputs/*.json; do
  [ -f "$f" ] || continue
  T=$(basename "$f" | sed -E 's/.*-T([0-9]+).*/\1/')
  echo "PROBE $(basename "$f") T=$T" | tee -a "$OUT/rule.log"
  "$B" rule "$f" "$T" "${REPS:-200}" 2>&1 | tee -a "$OUT/rule.log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
done
for n in 1 14 16 30 32; do
  "$B" const "$n" "${REPS:-200}" 2>&1 | tee -a "$OUT/const.log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
done
exit "$rc"
