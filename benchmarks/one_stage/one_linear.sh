#!/usr/bin/env bash
# One real linear layer's Gemm_v2 slices (`one_linear.py`) audited on the records audit's road (`served_door_prove.sh`'s
# AUDIT_RULE branch): registered, committed, drawn on a local challenger's live coins, each drawn block cut, lowered and
# composed by Lean's door, proved in zero knowledge behind the ZK server and verified by the Lean verifier. Then the
# negatives: the partition at two levels refused at the door before any coin; a changed output word in every block
# (so in every drawn one) refused; one changed output word in one unit, accepted unless its block is drawn; a
# stratified rate refused by its kind (SUBSET_ONLY). RULE: the accept's rate over the population.
# bash benchmarks/one_stage/one_linear.sh
# research run --on auto --cwd source --mem-gb 96 --cpus 16 --timeout 4h -- bash benchmarks/one_stage/one_linear.sh
set -uo pipefail
RULE=${RULE:-rate:1/72}
B=${OUT:-${RESEARCH_RUN_DIR:-/tmp/one-linear}/out}; mkdir -p $B
UV=$(command -v uv || echo $HOME/.local/bin/uv)
echo "SCOPE one template, no wired port between units: no wire between blocks, no block of several templates, no scan"
( unset PYTHONPATH; $UV run -q python benchmarks/one_stage/one_linear.py levels ) > $B/levels.txt 2>&1
LV=$?; cat $B/levels.txt
audit() {  # name, rule, window flags, audit flags
  OUT=$B/$1 AUDIT_RULE=$2 WINDOW_ARGS="benchmarks/one_stage/one_linear.py window --rule $2 $3" AUDIT_FLAGS="$4" \
    bash benchmarks/one_stage/served_door_prove.sh > $B/$1.log 2>&1
  local r=$?; echo "$1 rc $r"; grep -h '^audit rc\|^STEP\|^VERDICT\|^DRAW\|^A5\|^REFUSED\|^UNLOWERED' $B/$1.log | cut -c1-2000
  python3 - "$B/$1/summary.json" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except OSError:
    sys.exit()
keep = ("ok", "accepted", "verdict", "refused", "epsilon", "eps", "bound", "delta", "law", "drawn", "failed_drawn", "forged")
print("SUMMARY " + json.dumps({k: s[k] for k in s if k in keep}, default=str)[:3000])
EOF
  return $r
}
audit honest $RULE "" ""; H=$?
audit forged-drawn $RULE "--forge all" "--forged"; F=$?
audit forged-one $RULE "--forge 1" ""; U=$?
audit stratified stratified-${RULE#rate:} "" ""; R=$?
echo "LAW the zero-knowledge audit's proved statement covers a subset design, so the accept is under $RULE, a subset draw over one stratum; the audit under a rated law waits on its claim"
echo "RESULT $(python3 -c 'import json,sys; print(json.dumps(dict(zip(["levels","honest","forged_drawn","forged_one","stratified"], map(int, sys.argv[1:])))))' $LV $H $F $U $R)"
[ $LV = 0 ] && [ $H = 0 ] && echo "ACCEPT accepted by the Lean verifier, in zero knowledge, about the block circuits Lean's lowering derives from the registered circuit; that the lowering agrees with the circuit rests on three open premises"
[ $LV = 0 ] && [ $H = 0 ] && [ $F = 0 ]
