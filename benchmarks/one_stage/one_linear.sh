#!/usr/bin/env bash
# One real linear layer's Gemm_v2 slices (`one_linear.py`) audited on the records audit's road (`served_door_prove.sh`'s
# AUDIT_RULE branch): registered, committed, drawn on a local challenger's live coins, each drawn block cut, lowered and
# composed by Lean's door, proved in zero knowledge behind the ZK server and verified by the Lean verifier: one honest
# run, which passes only on its accept (`one_linear_arms.py`: a crash is an ERROR, a refusal a REJECT by its name). The
# partition at two levels is refused at the door before any coin. The weight is a run input each unit commits. RULE: the
# accept's rate over the population.
# No accept is printed unless C's template rows hold against its evaluator (`door_walls.py census`, one unit a block)
# and C has neither of the two word shapes (`word_shape_scan.py`), both read from the one C.json whose sha256 is printed.
# DELTA=0.5 bash benchmarks/one_stage/one_linear.sh
# research run --on auto --cwd source --mem-gb 96 --cpus 16 --timeout 4h -- bash benchmarks/one_stage/one_linear.sh
set -uo pipefail
RULE=${RULE:-rate:1/72}
[ -n "${DELTA:-}" ] || { echo "REFUSED one_linear.sh takes no default δ: pass DELTA (the audit's --delta), the δ of record"; exit 2; }
B=${OUT:-${RESEARCH_RUN_DIR:-/tmp/one-linear}/out}; mkdir -p $B
UV=$(command -v uv || echo $HOME/.local/bin/uv)
echo "SCOPE one template, no wired port between units; activations computed, not served; challenger local; the weight is the prover's committed value; nothing ties it to the model"
echo "SCOPE an accept is about the block circuit the verifier derives from C; no theorem yet says that is C's block"
( unset PYTHONPATH; $UV run -q python benchmarks/one_stage/one_linear.py c --out $B/c ) > $B/c.log 2>&1; CC=$?
cat $B/c.log
( unset PYTHONPATH; $UV run -q python benchmarks/one_stage/door_walls.py census --circuit $B/c --blocks one-per-template \
    --json $B/census.json ) > $B/census.log 2>&1; W=$?
echo "CENSUS rc $W"; grep -h '^BUILT\|^CENSUS\|^|' $B/census.log | cut -c1-1500
ROWS=$(python3 -c 'import json, sys
print("; ".join(str(r["rows"]) for r in json.load(open(sys.argv[1]))["table"]))' $B/census.json 2>/dev/null)
echo "ROWS $ROWS"
( unset PYTHONPATH; $UV run -q python benchmarks/one_stage/word_shape_scan.py $B/c/C.json ) > $B/scan.log 2>&1; SC=$?
grep -h '^DESCRIPTION\|^COUNTS\|^SCAN' $B/scan.log | cut -c1-1500
SHAPES=$(python3 -c 'import json, sys
s = [json.loads(l[5:]) for l in open(sys.argv[1]) if l.startswith("SCAN ")][0]
print(s["shape"] + s["unaligned"])' $B/scan.log 2>/dev/null)
( unset PYTHONPATH; $UV run -q python benchmarks/one_stage/one_linear.py levels ) > $B/levels.txt 2>&1
LV=$?; cat $B/levels.txt
audit() {  # name, rule, window flags, audit flags (the one honest run)
  OUT=$B/$1 AUDIT_RULE=$2 WINDOW_ARGS="benchmarks/one_stage/one_linear.py window --rule $2 $3" AUDIT_FLAGS="--delta $DELTA $4" \
    bash benchmarks/one_stage/served_door_prove.sh > $B/$1.log 2>&1
  local r=$?; echo "$1 rc $r"; grep -h '^audit rc\|^STEP\|^VERDICT\|^DRAW\|^A5\|^REFUSED\|^UNLOWERED' $B/$1.log | cut -c1-2000
  python3 - "$B/$1/summary.json" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except OSError:
    sys.exit()
keep = ("ok", "accepted", "verdict", "refused", "epsilon", "eps", "bound", "delta", "law", "drawn")
print("SUMMARY " + json.dumps({k: s[k] for k in s if k in keep}, default=str)[:3000])
EOF
  python3 benchmarks/one_stage/one_linear_arms.py $1 $r $B/$1.log $B/$1/summary.json $B/$1/audit.err $B/$1/window.txt $B/$1/audit.log | tee $B/$1.arm
}
audit honest $RULE "--weights committed" "--skip-negatives"; H=$?
echo "LAW the zero-knowledge audit's proved statement covers a subset design, so the accept is under $RULE, a subset draw over one stratum; the audit under a rated law waits on its claim"
echo "RESULT $(python3 -c 'import json,sys; print(json.dumps(dict(zip(["levels","honest"], map(int, sys.argv[1:])))))' $LV $H)"
ML=$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["audit"].get("members_lean") or {}; print("ok" if m.get("ran") is True and m.get("ok") is True else "no: "+json.dumps(m)[:300])' "$B/honest/summary.json" 2>/dev/null)
echo "members_lean: ${ML:-no: no summary}"
RP=$(python3 -c 'import json,sys
def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "rests_on_period_check" and v:
                yield v
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)
print(json.dumps(next(walk(json.load(open(sys.argv[1]))), "")))' "$B/honest/summary.json" 2>/dev/null)
[ "${RP:-\"\"}" = '""' ] || echo "REFUSED the accept rests on a period check that is not run: rests_on_period_check $RP"
# the δ the verifier's audit record states it bounded at (`Record.record`'s `wrong_units.delta`), not this script's
SDELTA=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["audit"]["wrong_units"]["delta"])' "$B/honest/summary.json" 2>/dev/null)
OK=1; [ $CC = 0 ] && [ $W = 0 ] && [ $SC = 0 ] && [ "$SHAPES" = 0 ] && [ $LV = 0 ] && [ $H = 0 ] && [ "$ML" = ok ] \
  && [ "${RP:-\"\"}" = '""' ] && python3 -c 'import sys; from fractions import Fraction as F; sys.exit(F(sys.argv[1]) != F(sys.argv[2]))' "$SDELTA" "$DELTA" 2>/dev/null && OK=0
if [ $OK = 0 ]; then
  echo "ACCEPT the proofs are verified by the Lean verifier in zero knowledge; δ not yet chosen for the gate"
  ( unset PYTHONPATH; $UV run -q python benchmarks/one_stage/accept_line.py $B/honest/summary.json $B/c/C.json "$DELTA" ) 2>&1
  echo "ACCEPT the members' files and sessions were checked by Lean against a registration and coins the Python audit handed it; not a statement about a served run"
  echo "ACCEPT the proofs are about the block circuits Lean's lowering derives from the registered circuit; each block is single-range (no block spans two root nodes)"
  echo "ACCEPT no proved statement is about the file the door writes; every accept at the door rests on the lift preserving evaluation, by reading and by the fuzzer"
  echo "ACCEPT the lowering's rows for this template were checked against the evaluator ($ROWS), one unit a block"
  echo "ACCEPT scanned for the two word shapes (word_shape_scan.py): none (C sha256 $(sha256sum $B/c/C.json | cut -c1-64))"
  echo "ACCEPT recorded under a rate the verifier resolves to a subset of 4 of 144 blocks; the rated design is not on one-circuit"
else
  echo "NO ACCEPT c $CC census $W scan $SC shapes ${SHAPES:-?} levels $LV members_lean ${ML:-?}; $(cat $B/honest.arm 2>/dev/null || echo 'ARM honest: no verdict')"
fi
[ $OK = 0 ]
