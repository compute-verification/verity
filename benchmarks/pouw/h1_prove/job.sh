#!/usr/bin/env bash
# H1 tile units of one recorded int8 Llama-3.1-8B window through sampled proofs: drawn by the live coin, proved by C-Flock
# --zk and verified by the Lean verifier against the verifier's own public files (run.py's docstring has the protocol).
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source --mem-gb G \
#     --declared-output 'out/*' --env H1_MEM_GB=G [--env H1_K=64 --env H1_COLS=64] -- bash benchmarks/pouw/h1_prove/job.sh
#   1. hidden_zk/job.sh's builds (flock-circuit's proving and selftest builds, the Lean flock-verify, a venv)
#   2. stage.py builds the unit's circuit (the long step at K = 4096) while layer.py records the window's int8 codes from the
#      checkpoint (layer L's q/k/v/gate/up at H1's quantizers; a second window from other tokens) and run.py prepare commits,
#      serves, draws, opens, checks the openings and has the drawn units staged
#   3. one loopback --zk session per statement (the drawn units; the other window's unit), every instance drawn
#   4. the Lean checks of run.py's plan (honest ACCEPT; flipped-z, wrong-receipt, other-window REJECT; other-window-own
#      ACCEPT), as many at a time as 90% of H1_MEM_GB holds at 6x the circuit text each (two at K = 4096 and
#      H1_MEM_GB=500), and the selftest build's controls on
#      the drawn units' statement (H1_CONTROLS; at K = 4096 a selftest, prover and verifier in one process, peaks past
#      170 GB)
#   5. run.py summarize: out/summary.json
# env: H1_MEM_GB (the job's --mem-gb)  H1_K (4096; a smaller K truncates every row: a smoke)  H1_COLS (each call's first
#      N weight rows; 0 all)  H1_DRAW (8)  H1_TOKENS (16)  H1_LAYER (0)  H1_HUB (/workspace/hf/hub)
#      H1_CONTROLS ("honest output_row_committed_false" below K = 4096, else none)  H1_SERVE_PCT (36) and H1_PROVE_PCT (32):
#      a session's serve and prove caps, % of H1_MEM_GB (they run side by side, the stager gone; at K = 4096 with 8
#      instances the serve passes 131 GB resident and the prover 100 GB, so H1_MEM_GB=500)  HZ_RUNS (1)  HZ_WARM (0)
#      HZ_CACHE, HZ_STAGE: hidden_zk's
set -uo pipefail
export HZ_CONFIGS=h1 HZ_MEM_GB=${H1_MEM_GB:?} HZ_RUNS=${HZ_RUNS:-1} HZ_WARM=${HZ_WARM:-0}
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/benchmarks/pouw/hidden_zk
H=$SRC/benchmarks/pouw/h1_prove
K=${H1_K:-4096} COLS=${H1_COLS:-0} DRAW=${H1_DRAW:-8} TOKENS=${H1_TOKENS:-16} LAYER=${H1_LAYER:-0}
HUB=${H1_HUB:-/workspace/hf/hub}
if [ "$K" = 4096 ]; then CONTROLS=${H1_CONTROLS:-}; else CONTROLS=${H1_CONTROLS:-honest output_row_committed_false}; fi
SERVE_GB=$(( MEM_GB * ${H1_SERVE_PCT:-36} / 100 )) PROVE_GB=$(( MEM_GB * ${H1_PROVE_PCT:-32} / 100 ))
SW=$W/stage WK=$W/work
mkdir -p "$OUT/lean"

build_layer_venv() { local D=$1; rm -rf "$D" && uv venv -q --python 3.12 "$D" && uv pip install -q --python "$D/bin/python" numpy==2.5.3 tokenizers==0.22.1; }
# the stager needs only the venv: it builds the circuit while setup builds flock-circuit and flock-verify
[ -n "${HZ_PYTHON:-}" ] || HZ_PYTHON=$(cached venv numpy2.5.3-blake3-1.0.9 build_python)/bin/python
PY=$HZ_PYTHON TIME="$HZ_PYTHON $J/timed.py"
log "h1-prove: K=$K cols=${COLS:-all} draw=$DRAW tokens=$TOKENS layer=$LAYER controls: $CONTROLS"
# detached (not a job of this shell), so setup's `wait` doesn't wait for it; run.py and the loop below watch its pid
( $TIME "$OUT/stage.time" bash -c "$(vlimit "$STAGE_GB") exec $PY $H/stage.py $K $SW" > "$OUT/stage.out" 2> "$OUT/stage.err" &
  echo $! > "$W/stager.pid" )
STAGER=$(cat "$W/stager.pid")
setup

LPY=$(cached venv-layer numpy2.5.3-tokenizers0.22.1 build_layer_venv)/bin/python
[ -x "$LPY" ] || LPY=$PY
MODEL=$HUB FETCH=()
if ! ls -d "$HUB"/models--*/snapshots/*/ 2>/dev/null | grep -q "Llama-3.1-8B"; then
  MODEL=${HZ_CACHE}/unsloth-Llama-3.1-8B FETCH=(--fetch unsloth/Llama-3.1-8B)   # a node without the checkpoint: layer L's shard
fi
"$LPY" "$H/layer.py" "$MODEL" "$W/L1.npz" --tokens "$TOKENS" --layer "$LAYER" "${FETCH[@]}" > "$OUT/layer1.json" 2> "$OUT/layer1.err" \
  && "$LPY" "$H/layer.py" "$MODEL" "$W/L2.npz" --tokens "$TOKENS" --layer "$LAYER" --seed-tokens "$("$PY" -c 'import os; print(os.urandom(16).hex())')" \
    > "$OUT/layer2.json" 2> "$OUT/layer2.err" || { log "layer.py FAILED: $(tail -2 "$OUT"/layer*.err)"; kill $STAGER; exit 1; }
log "window: $(cut -c1-400 "$OUT/layer1.json")"
args=(--work "$WK" --stage "$SW" --lean "$LEAN" --layer "$W/L1.npz" --layer2 "$W/L2.npz" --draw "$DRAW" --stager-pid "$STAGER"
      --name "h1-${RESEARCH_RUN_ID:-local}")
[ "$K" != 4096 ] && args+=(--k "$K")
[ "$COLS" != 0 ] && args+=(--cols "$COLS")
"$PY" "$H/run.py" prepare "${args[@]}" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
touch "$SW/requests/END"
while kill -0 "$STAGER" 2>/dev/null; do sleep 2; done
cp "$SW/circuit.json" "$OUT/" 2>/dev/null
grep -E '^(BUILT|STAGED|FAILED)' "$OUT/stage.out" | cut -c1-600 | tee -a "$OUT/job.log"
cp "$WK/record.json" "$WK/plan.json" "$OUT/" 2>/dev/null
[ $rc = 0 ] || { log "prepare FAILED (exit $rc): $(tail -3 "$OUT/stage.err")"; exit 1; }
cp -r "$WK/verifier" "$OUT/verifier"

# ---- proofs: one --zk session per statement ------------------------------------------------------------------------------
while IFS=$'\t' read -r kind tag D n; do
  [ "$kind" = S ] || continue
  ln -sf "$SW/circuit.txt" "$D/circuit.txt" && cp "$SW/pin.txt" "$D/pin.txt"
  mkdir -p "$OUT/$tag"
  N=$n
  log "prove $tag: $n instances, subset:$n"
  session "$D" "$OUT/$tag" || log "prove $tag FAILED: $(tail -2 "$OUT/$tag/prove.err")"
  log "prove $tag: $(cat "$OUT/$tag/prove.time" 2>/dev/null)"
done < "$WK/plan.tsv"

# ---- Lean: the plan's checks, then the selftest build's controls on the drawn units' statement ----------------------------
each=$(( $(stat -L -c %s "$SW/circuit.txt") * 6 / 1024 / 1024 / 1024 + 4 ))
par=$(( STAGE_GB / each )); [ $par -lt 1 ] && par=1; [ $par -gt 6 ] && par=6
cap=$(( STAGE_GB / par ))
lean_check() {   # lean_check TAG D PUB STATEMENT_ID
  local tag=$1 D=$2 pub=$3 sid=$4 s
  s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tail -1)
  [ -n "$s" ] || { echo "no session under $D" > "$OUT/lean/$tag.err"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do wait -n; done
  $TIME "$OUT/lean/$tag.time" bash -c "$(vlimit $cap) exec $LEAN verify --zk --circuit $D/circuit.txt --statement $sid \
    --public $pub --session $s" > "$OUT/lean/$tag.out" 2> "$OUT/lean/$tag.err" &
}
log "lean: $par at a time, cap $cap GB each"
while IFS=$'\t' read -r kind tag D pub; do
  [ "$kind" = C ] && lean_check "$tag" "$D" "$pub" verity/flock-circuit
done < "$WK/plan.tsv"
wait
HD=$(awk -F'\t' '$1 == "S" && $2 == "honest" {print $3}' "$WK/plan.tsv")
N=$(awk -F'\t' '$1 == "S" && $2 == "honest" {print $4}' "$WK/plan.tsv")
if [ -n "$CONTROLS" ] && [ -e "$HD/inst.bin" ]; then
  CASES=$CONTROLS PROVE_GB=$STAGE_GB
  controls "$HD" "$OUT/honest" tile
  for s in "$HD"/st/*/; do
    [ -d "$s" ] || continue
    c=$(basename "$s"); pub=$HD/pub.bin; [ -e "$s/pub.bin" ] && pub=$s/pub.bin
    while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do wait -n; done
    $TIME "$OUT/lean/st-$c.time" bash -c "$(vlimit $cap) exec $LEAN verify --zk --circuit $HD/circuit.txt \
      --statement verity/flock-circuit+seed-injection --public $pub --session $s" > "$OUT/lean/st-$c.out" 2> "$OUT/lean/st-$c.err" &
  done
fi
wait
"$PY" "$H/run.py" summarize --work "$WK" --out "$OUT" --controls "$CONTROLS" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
du -sh "$SW/circuit.txt" "$HD/sessions" 2>/dev/null | tee -a "$OUT/job.log"
log "h1-prove: done (exit $rc)"
exit "$rc"
