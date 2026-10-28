#!/usr/bin/env bash
# H1 tile units of one recorded int8 Llama-3.1-8B window through sampled proofs: drawn by the live coin, proved by C-Flock
# --zk and verified by the Lean verifier against the verifier's own public files (run.py's docstring has the protocol).
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source --mem-gb G \
#     --declared-output 'out/*' --env H1_MEM_GB=G [--env H1_K=64 --env H1_COLS=64] -- bash benchmarks/pouw/h1_prove/job.sh
#   1. hidden_zk/job.sh's builds (flock-circuit's proving and selftest builds, the Lean flock-verify, a venv)
#   2. layer.py records the window's int8 codes and their f32 scales from the checkpoint (layer L's q/k/v/gate/up at H1's
#      quantizers; a second window from other tokens) and run.py prepare commits (each call's A table and z table of its
#      tiles' z/y records, and the weights' B rows), serves, draws, opens, checks the openings and has the drawn units
#      staged, one statement a drawn call, while stage.py builds the unit's circuit (the long step at K = 4096), bound to
#      the window's Program once the commitment gives its shape
#   3. one loopback --zk session per statement (each drawn call's units, honest the first drawn unit's call; the other
#      window's unit; honest's units on a tampered A row, with their ok rows committed afresh, and on in-z's window
#      computing ok = 0; the in-proof negatives on honest's units, each also as
#      <tag>-skip, proved with FC_ZK_SELF_CHECK=skip), every instance drawn, each prover under the firewall (FC_FIREWALL=1:
#      the Lean session loop, its level-0 draw and Lean finish) unless H1_FIREWALL=0
#   4. the Lean checks of run.py's plan, each with the verifier's own window Program, partition and its call's registered
#      reads, the ok port among them (--program, --partition, --registered: honest and each honest-c<c> ACCEPT with their
#      units derived; flipped-zy, wrong-receipt, other-window
#      REJECT; other-window-own ACCEPT; tampered-a, ok-zero and ok-unregistered REJECT by their registered reads, each
#      against its own public file; ok-known-root (honest's session under a file whose ok root is not its known rows',
#      `h1unit.known_ok`) refused reading --registered, exit 2 and no verdict; in-receipt, in-d, the record negatives
#      in-y, in-z, in-swap, in-call, in-z-only (and in-xa under H1_TILE=v3) and their -skip runs REFUSED by some layer),
#      as many at a time as 90% of H1_MEM_GB holds at 6x the circuit text each (two at K = 4096 and
#      H1_MEM_GB=500), and the selftest build's controls on
#      the drawn units' statement (H1_CONTROLS; at K = 4096 a selftest, prover and verifier in one process, peaks past
#      170 GB)
#   5. run.py summarize: out/summary.json, with window credit's registration step (credit._registration) on the honest
#      checks' Lean findings under the window's open root, which every per-window registered value binds
# env: H1_MEM_GB (the job's --mem-gb)  H1_K (4096; a smaller K truncates every row: a smoke)  H1_COLS (each call's first
#      N weight rows; 0 all)  H1_DRAW (8)  H1_TOKENS (16)  H1_LAYER (0)  H1_HUB (/workspace/hf/hub)
#      H1_FIREWALL (1)  H1_WIRE (0; 1: the selftest's zk_worker_wire, zk_worker_finish and zk_worker_level0 on honest's
#      statement, under the firewall, after the proofs, thirteen firewalled sessions, about 34 min at K = 64; a failure
#      fails the job)  H1_ONLY (the plan's tags to prove and check, space-separated: "honest other-window-own" skips the negatives; all)
#      HZ_RUNS (1): sessions a statement, all verified by one Lean process, which reads the circuit once
#      H1_CONTROLS ("honest output_row_committed_false" below K = 4096, else none)  H1_SERVE_PCT (36) and H1_PROVE_PCT (32):
#      a session's serve and prove caps, % of H1_MEM_GB (they run side by side, the stager gone; at K = 4096 with 8
#      instances the serve passes 131 GB resident and the prover 100 GB, so H1_MEM_GB=500)  HZ_RUNS (1)  HZ_WARM (0)
#      HZ_CACHE, HZ_STAGE: hidden_zk's
#      H1_QUANT (h1: H1's ±110/[-126, 125] quantizers; vllm: vLLM's per-row absmax → ±127, over the first H1_K entries
#      below K = 4096, so a smoke's tiles hold A = 127)
set -uo pipefail
# its checks expect the Lean verifier's ACCEPT, which a templated statement with --registered never gets
# (h1unit.layout), so this job stages only the flat statement; h1_lanes/job.sh proves the structural one end to end
case "${H1_TMPL:-flat}" in
  flat) ;;
  *) echo "H1_TMPL=$H1_TMPL: this job's checks expect ACCEPT, so it proves only the flat statement (H1_TMPL unset or flat); h1_lanes/job.sh proves the structural one" >&2; exit 2 ;;
esac
export HZ_CONFIGS=h1 HZ_MEM_GB=${H1_MEM_GB:?} HZ_RUNS=${HZ_RUNS:-1} HZ_WARM=${HZ_WARM:-0}
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/benchmarks/pouw/hidden_zk
H=$SRC/benchmarks/pouw/h1_prove
K=${H1_K:-4096} COLS=${H1_COLS:-0} DRAW=${H1_DRAW:-8} TOKENS=${H1_TOKENS:-16} LAYER=${H1_LAYER:-0}
HUB=${H1_HUB:-/workspace/hf/hub}
QUANT=(); [ "${H1_QUANT:-h1}" = vllm ] && QUANT=(--quant vllm --quant-k "$([ "$K" = 4096 ] && echo 0 || echo "$K")")
if [ "$K" = 4096 ]; then CONTROLS=${H1_CONTROLS:-}; else CONTROLS=${H1_CONTROLS:-honest output_row_committed_false}; fi
SERVE_GB=$(( MEM_GB * ${H1_SERVE_PCT:-36} / 100 )) PROVE_GB=$(( MEM_GB * ${H1_PROVE_PCT:-32} / 100 ))
SW=$W/stage WK=$W/work
mkdir -p "$OUT/lean"

build_layer_venv() { local D=$1; rm -rf "$D" && uv venv -q --python 3.12 "$D" && uv pip install -q --python "$D/bin/python" numpy==2.5.3 tokenizers==0.22.1; }
# the stager needs only the venv: it builds the circuit while setup builds flock-circuit and flock-verify
[ -n "${HZ_PYTHON:-}" ] || HZ_PYTHON=$(cached venv numpy2.5.3-blake3-1.0.9 build_python)/bin/python
PY=$HZ_PYTHON TIME="$HZ_PYTHON $J/timed.py"
log "h1-prove: K=$K cols=${COLS:-all} draw=$DRAW tokens=$TOKENS layer=$LAYER quant=${H1_QUANT:-h1} controls: $CONTROLS"
# detached (not a job of this shell), so setup's `wait` doesn't wait for it; run.py and the loop below watch its pid
( $TIME "$OUT/stage.time" bash -c "$(vlimit "$STAGE_GB") exec $PY $H/stage.py $K $SW" > "$OUT/stage.out" 2> "$OUT/stage.err" &
  echo $! > "$W/stager.pid" )
STAGER=$(cat "$W/stager.pid")
setup
FW=${H1_FIREWALL:-1}
if [ "$FW" = 1 ]; then
  export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
  (exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
    > "$OUT/build-firewall.log" 2>&1
  [ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "no firewall frame beside $LEAN"; kill $STAGER; exit 1; }
  sha256sum "$FC_FIREWALL_LEAN" | tee -a "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
fi

LPY=$(cached venv-layer numpy2.5.3-tokenizers0.22.1 build_layer_venv)/bin/python
[ -x "$LPY" ] || LPY=$PY
MODEL=$HUB FETCH=()
if ! ls -d "$HUB"/models--*/snapshots/*/ 2>/dev/null | grep -q "Llama-3.1-8B"; then
  MODEL=${HZ_CACHE}/unsloth-Llama-3.1-8B FETCH=(--fetch unsloth/Llama-3.1-8B)   # a node without the checkpoint: layer L's shard
fi
"$LPY" "$H/layer.py" "$MODEL" "$W/L1.npz" --tokens "$TOKENS" --layer "$LAYER" "${QUANT[@]}" "${FETCH[@]}" > "$OUT/layer1.json" 2> "$OUT/layer1.err" \
  && "$LPY" "$H/layer.py" "$MODEL" "$W/L2.npz" --tokens "$TOKENS" --layer "$LAYER" --seed-tokens "$("$PY" -c 'import os; print(os.urandom(16).hex())')" "${QUANT[@]}" \
    > "$OUT/layer2.json" 2> "$OUT/layer2.err" || { log "layer.py FAILED: $(tail -2 "$OUT"/layer*.err)"; kill $STAGER; exit 1; }
log "window: $(cut -c1-400 "$OUT/layer1.json")"
args=(--work "$WK" --stage "$SW" --lean "$LEAN" --layer "$W/L1.npz" --layer2 "$W/L2.npz" --draw "$DRAW" --stager-pid "$STAGER"
      --name "h1-${RESEARCH_RUN_ID:-local}")
[ "$K" != 4096 ] && args+=(--k "$K")
[ "$COLS" != 0 ] && args+=(--cols "$COLS")
[ -n "${H1_ONLY:-}" ] && args+=(--only "$H1_ONLY")
"$PY" "$H/run.py" prepare "${args[@]}" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
touch "$SW/requests/END"
while kill -0 "$STAGER" 2>/dev/null; do sleep 2; done
cp "$SW/circuit.json" "$OUT/" 2>/dev/null
grep -E '^(BUILT|STAGED|FAILED)' "$OUT/stage.out" | cut -c1-600 | tee -a "$OUT/job.log"
cp "$WK/record.json" "$WK/plan.json" "$OUT/" 2>/dev/null
[ $rc = 0 ] || { log "prepare FAILED (exit $rc): $(tail -3 "$OUT/stage.err")"; exit 1; }
mkdir -p "$OUT/verifier" && cp "$WK"/verifier/*.pub.bin "$WK"/verifier/*.registered.json "$WK/verifier/partition.json" "$OUT/verifier/"   # program.json: record.json's program_sha512, rebuilt from the shape

# ---- proofs: one --zk session per statement ------------------------------------------------------------------------------
# a firewalled prover's Lean Hello sets each session up at the verifier's own statement, partition and window Program (live
# PROTOCOL.md §10.4); the registered reads are the offline verifier's, outside the session's spec
VLIMIT=$(declare -f vlimit)
if [ "$FW" = 1 ]; then
  export FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_STATEMENT=verity/flock-circuit
  export FC_FIREWALL_HELLO_ARGS="--partition $WK/verifier/partition.json --program $WK/verifier/program.json"
  # the firewall's Lean processes are the prover's children, and under an address-space cap (ulimit -v) their runtime
  # can't create its threads: a firewalled session is bounded by the run's cgroup (--mem-gb) alone
  vlimit() { :; }
fi
log "prove: firewall $FW"
while IFS=$'\t' read -r kind tag D n; do
  [ "$kind" = S ] || continue
  ln -sf "$SW/circuit.txt" "$D/circuit.txt" && cp "$SW/pin.txt" "$D/pin.txt"
  mkdir -p "$OUT/$tag"
  N=$n
  log "prove $tag: $n instances, subset:$n"
  if [[ $tag == *-skip ]]; then export FC_ZK_SELF_CHECK=skip; else unset FC_ZK_SELF_CHECK; fi   # run.py's SKIP
  session "$D" "$OUT/$tag" || log "prove $tag FAILED: $(tail -2 "$OUT/$tag/prove.err")"
  unset FC_ZK_SELF_CHECK
  log "prove $tag: $(cat "$OUT/$tag/prove.time" 2>/dev/null)"
done < "$WK/plan.tsv"
# the wire's controls, under the same firewall: the prover's frames doctored on the wire to the session loop, each dropped
# or refused, the host's rep requests to the loop's finish doctored, each refused, and a second level-0 draw for a table,
# refused (flock-circuit's zk_worker_wire, zk_worker_finish and zk_worker_level0, live PROTOCOL.md §10.4)
WIRE=ok
if [ "$FW" = 1 ] && [ "${H1_WIRE:-0}" = 1 ]; then
  HD=$(awk -F'\t' '$1 == "S" && $2 == "honest" {print $3}' "$WK/plan.tsv")
  HN=$(awk -F'\t' '$1 == "S" && $2 == "honest" {print $4}' "$WK/plan.tsv")
  for c in zk_worker_wire zk_worker_finish zk_worker_level0; do
    FC_FIREWALL_STATEMENT=verity/flock-circuit+seed-injection \
      $TIME "$OUT/honest/selftest-$c.time" bash -c "exec $SBIN selftest --zk --instances $HD/inst.bin \
      --circuit $HD/circuit.txt --draw subset:$HN --only $c" > "$OUT/honest/selftest-$c.out" 2> "$OUT/honest/selftest-$c.err"
    grep -q '^NEG.*"pass":true}$' "$OUT/honest/selftest-$c.out" || WIRE=FAIL
    log "$c controls: $(grep -E '^NEG' "$OUT/honest/selftest-$c.out" | cut -c1-300)"
  done
fi
unset FC_FIREWALL FC_HOST_WITNESS FC_FIREWALL_STATEMENT FC_FIREWALL_HELLO_ARGS
eval "$VLIMIT"

# ---- Lean: the plan's checks, then the selftest build's controls on the drawn units' statement ----------------------------
each=$(( $(stat -L -c %s "$SW/circuit.txt") * 6 / 1024 / 1024 / 1024 + 4 ))
par=$(( STAGE_GB / each )); [ $par -lt 1 ] && par=1; [ $par -gt 6 ] && par=6
cap=$(( STAGE_GB / par ))
# the verifier's own window Program and partition (run.py's), from which Lean derives each instance's unit
BOUND="--partition $WK/verifier/partition.json --program $WK/verifier/program.json"
lean_check() {   # lean_check TAG D PUB STATEMENT_ID REGISTERED: the verifier's own registered reads of the window (run.py's)
  local tag=$1 D=$2 pub=$3 sid=$4 reg=$5 s
  s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tr '\n' ' ')
  [ -n "$s" ] || { echo "no session under $D" > "$OUT/lean/$tag.err"; return; }
  while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do wait -n; done
  $TIME "$OUT/lean/$tag.time" bash -c "$(vlimit $cap) exec $LEAN verify --zk --circuit $D/circuit.txt --statement $sid \
    --public $pub $BOUND --registered $reg --session $s" > "$OUT/lean/$tag.out" 2> "$OUT/lean/$tag.err" &
}
log "lean: $par at a time, cap $cap GB each"
while IFS=$'\t' read -r kind tag D pub reg; do
  [ "$kind" = C ] && lean_check "$tag" "$D" "$pub" verity/flock-circuit "$reg"
done < "$WK/plan.tsv"
wait
HD=$(awk -F'\t' '$1 == "S" && $2 == "honest" {print $3}' "$WK/plan.tsv")
N=$(awk -F'\t' '$1 == "S" && $2 == "honest" {print $4}' "$WK/plan.tsv")
HREG=$(awk -F'\t' '$1 == "C" && $2 == "honest" {print $5}' "$WK/plan.tsv")
if [ -n "$CONTROLS" ] && [ -e "$HD/inst.bin" ] && [ -n "$HREG" ]; then
  CASES=$CONTROLS PROVE_GB=$STAGE_GB
  controls "$HD" "$OUT/honest" tile
  for s in "$HD"/st/*/; do
    [ -d "$s" ] || continue
    c=$(basename "$s"); pub=$HD/pub.bin; [ -e "$s/pub.bin" ] && pub=$s/pub.bin
    while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do wait -n; done
    $TIME "$OUT/lean/st-$c.time" bash -c "$(vlimit $cap) exec $LEAN verify --zk --circuit $HD/circuit.txt \
      --statement verity/flock-circuit+seed-injection --public $pub $BOUND --registered $HREG \
      --session $s" > "$OUT/lean/st-$c.out" 2> "$OUT/lean/st-$c.err" &
  done
fi
wait
"$PY" "$H/run.py" summarize --work "$WK" --out "$OUT" --controls "$CONTROLS" --firewall "$FW" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
[ "$WIRE" = ok ] || { log "the wire's controls failed: $OUT/honest/selftest-zk_worker_{wire,finish}.out"; rc=1; }
du -sh "$SW/circuit.txt" "$HD/sessions" 2>/dev/null | tee -a "$OUT/job.log"
log "h1-prove: done (exit $rc)"
exit "$rc"
