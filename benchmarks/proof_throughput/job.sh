#!/usr/bin/env bash
# proof_throughput's prover point: sessions that each prove B coordinates of one workload's population, the B its loopback
# `flock-circuit serve` draws at the session's Register by the stratified law (`serve --draw-law`, the law verify.sh's Lean
# verifier derives for itself), proved by C-Flock's native prover (ZK off) on one leased GPU with live coins, WARM warm and RUNS
# timed sessions. The population (n_max instances) is staged once, whatever B (FLOCK_STAGE_CACHE). Build, staging and the
# instance check run outside the lease; the lease covers the sessions alone. The timed sessions, the circuit, the population's
# public file and the law go to the store as one bundle (out/bundle.json names it) for verify.sh, which runs the Lean verifier
# on a CPU node and writes the point's contract result. This run's own result.json is the same contract result with the Lean
# verifier not run.
# Run from the verity tree on a GPU node, as a GPU host job (its CPUs at a GPU job's weight, the GPU taken by gpu-lease):
#   research run --queue --on vy-nebius-1 --gpu-host --project verity --source . --cwd source --mem-gb 96 --cpus 32 \
#     --tool verity_proof_throughput.tool:PROVER_POINT -- bash benchmarks/proof_throughput/job.sh WORKLOAD=bf16-k6144 B=64
# (`python benchmarks/proof_throughput/run.py run` prints and launches these.)
# env: WORKLOAD= B= (required)  WARM=1 RUNS=3  IDENTITY= (the plan's point identity, recorded as the run's key param)
#      FLOCK_WORK=/workspace/jobs/proof-throughput (the prover builds, the venv and FLOCK_STAGE_CACHE under it)
#      LEASE_MAX_MIN=30  BUNDLE=1 (0: keep the sessions out of the store)  STAGE_ONLY=1 (build, stage and check; no GPU)
#      any FC_*=VALUE reaches the prover (FC_PIPELINE_DEPTH=4 and FC_HOST_PREPIN=1 by default, as the hill-climb)
#      ZK=1: configuration 2's stand-in (zk.py): M1's masked sessions (--zk) on the host's witness, under the ZK server's send
#      check unless FC_FIREWALL=0; the result is zk-point's, no contract result and no bundle
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
: "${WORKLOAD:?WORKLOAD= is required}" "${B:?B= is required}"
# the ZK server rechecks a witness the host assembled (PROTOCOL §10.2), so both arms take it: their difference is the ZK server's
[ "${ZK:-0}" = 1 ] && export FLOCK_ZK=1 FC_FIREWALL=${FC_FIREWALL:-1} FC_HOST_WITNESS=1 BUNDLE=0
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/pt-$$}/out}; mkdir -p "$O"
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proof-throughput}; mkdir -p "$FLOCK_WORK" || { echo "cannot make $FLOCK_WORK" >&2; exit 1; }
PYTHONPATH=$REPO/tools/research/src python3 -m research keep "$FLOCK_WORK" --owner @proofs --expires 14d --priority low \
  --regenerate "bash benchmarks/proof_throughput/job.sh WORKLOAD=$WORKLOAD B=$B" \
  --why "proof_throughput's prover builds, venv and stage cache: a cache, rebuilt by the next point" > "$O/keep.json" 2>&1 \
  || echo "no retention record on $FLOCK_WORK: $(tail -1 "$O/keep.json")" >&2
export FLOCK_STAGE_CACHE=${FLOCK_STAGE_CACHE:-$FLOCK_WORK/stage-cache} FC_COINS=${FC_COINS:-os}
export FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
# every instance's own rows: the verifiers refuse a plain statement's shared-row public file (ProvedScope.sharedRowsOk)
export FLOCK_OWN_ROWS=${FLOCK_OWN_ROWS:-1}
unset CARGO_TARGET_DIR
S=$(mktemp -d "$FLOCK_WORK/scratch.XXXXXX"); trap 'rm -rf "$S"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
ph() { printf '%s\t%s\t%s\n' "$1" "$2" "$(date +%s.%N)" >> "$O/phases.tsv"; }
for c in "$FLOCK_WORK/cuda-root" /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x "$c/usr/local/cuda-13.3/bin/nvcc" ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
[ -x /usr/local/cuda-13.3/bin/nvcc ] || [ -n "${CUDA_TK:-}" ] || { log "no CUDA 13.3 on this node (71-gemm-slowdown.sh unpacks it)"; exit 1; }
export PT_CUDA=13.3
SWEEP=verity/ml/flock/pod/70-class-sweep.sh
# this run's records the bundle carries for verify.sh, beside the circuit, the public file and the sessions
BUNDLED="results.jsonl law.json gpu-mem.txt gpu.csv hardware.json software.json binary.sha256 staged.jsonl instances.json
stage-rec.json phases.tsv point.json index.jsonl"

ph build start
OUT=$O/build bash $SWEEP BUILD_ONLY=1 GPU=1 SM=120 > "$O/build.txt" 2>&1 || { tail -20 "$O/build.txt"; log "build failed"; exit 1; }
BIN=$(sed -n 's/^BINARY=//p' "$O/build.txt" | tail -1); SBIN=$(sed -n 's/^SELFTEST_BINARY=//p' "$O/build.txt" | tail -1)
PYBIN=$FLOCK_WORK/flock-circuit/py/bin/python3   # the class sweep's venv (numpy, blake3), made here when this node has none yet
if [ ! -x "$PYBIN" ]; then
  UV=$(command -v uv || echo "$HOME/.local/bin/uv")
  (exec 8>"$FLOCK_WORK/flock-circuit/venv.lock"; flock 8; [ -x "$PYBIN" ] || { "$UV" venv -q --python 3.12 "$FLOCK_WORK/flock-circuit/py" \
    && "$UV" pip install -q --python "$PYBIN" numpy blake3; })
fi
[ -x "$BIN" ] && [ -x "$PYBIN" ] || { log "missing the prover ($BIN) or the venv ($PYBIN)"; exit 1; }
sha256sum "$BIN" | cut -d' ' -f1 > "$O/binary.sha256"
ph build end
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/proof_throughput
PT() { "$PYBIN" -m verity_proof_throughput "$@"; }
# the population (the class sweep's PT_VU, PT_N) and the verifier's stratified law for B (law.json)
read -r PT_VU PT_N < <(PT population --workload "$WORKLOAD" --B "$B" --out "$O") && [ -n "${PT_N:-}" ] || { log "no such workload or B"; exit 1; }
export PT_VU PT_N

# the instances: the workload's frozen population, recomputed and checked against its manifest
ph instances start
PT check-instances --workload "$WORKLOAD" --out "$O/instances.json" || { log "the population doesn't match the manifest"; exit 1; }
ph instances end

ARGS=(PROGRAM="$REPO/benchmarks/proof_throughput/verity_proof_throughput/population.py:program" PARTITION=q-template-instance
      BATCH=fixed N="$PT_N" MAX_STATEMENT_BITS=34359738368 MAX_ANDS=33554432 SELFTEST=0 WARM="${WARM:-1}" RUNS="${RUNS:-3}"
      BINARY="$BIN" SELFTEST_BINARY="$SBIN" PYBIN="$PYBIN")
ph stage start
OUT=$S/stage RESEARCH_RUN_DIR=$S/stage bash $SWEEP "${ARGS[@]}" STAGE_ONLY=1 SM=120 > "$O/stage.txt" 2>&1
cp "$S/stage/classes/staged.jsonl" "$O/staged.jsonl" 2>/dev/null
ph stage end
PT staged --results "$O/staged.jsonl" > "$O/staged.txt" || { tail -20 "$O/stage.txt"; log "staging failed"; exit 1; }
log "staged: $(cat "$O/staged.txt")"
[ "${STAGE_ONLY:-0}" = 1 ] && { log "STAGE_ONLY: done"; exit 0; }
# a firewalled prover's Lean Hello sets each session up at the verifier's own statement (live PROTOCOL.md §10.4), which takes
# a stratified sample only with the verifier's own partition, program and K = B, as verify.sh derives them
if [ "${ZK:-0}" = 1 ] && [ "$FC_FIREWALL" = 1 ]; then
  C0=$(PT staged --results "$O/staged.jsonl" --cache-dir) && PT partition --workload "$WORKLOAD" --circuit "$C0/circuit.txt" \
    --out "$S/partition.json" --program-out "$S/program.json" > "$O/partition.txt" 2>&1 \
    || { tail -5 "$O/partition.txt"; log "no partition object for the firewall's Hello"; exit 1; }
  export FC_FIREWALL_STATEMENT=verity/flock-circuit
  export FC_FIREWALL_HELLO_ARGS="--partition $S/partition.json --program $S/program.json --stratified $B"
fi

# the sessions, on one leased GPU: node 1 lends Kueue's GPUs through its own gpu-lease copy (infra/cluster submit.py)
if [ -x /workspace/verity-guest/bin/gpu-lease ] && [ -e /run/gpu-lease/pool.gpus ]; then
  LEASE=(env GPU_LEASE_DIR=/run/gpu-lease GPU_LEASE_ALLOWED_FILE=/run/gpu-lease/pool.gpus /workspace/verity-guest/bin/gpu-lease)
else
  LEASE=(gpu-lease)
fi
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-proofs/proof_throughput/${RESEARCH_RUN_ID:-local}}
ph lease wait
# one point's sessions at a time on this node (their staging overlaps): two would share its host CPUs and memory bandwidth
flock "$FLOCK_WORK/sessions.lock" "${LEASE[@]}" 1 --wait --max-min "${LEASE_MAX_MIN:-30}" -- bash -c '
  set -u; O=$1; shift
  printf "lease\tstart\t%s\n" "$(date +%s.%N)" >> "$O/phases.tsv"
  G=${GPU_LEASE_UUID:-${CUDA_VISIBLE_DEVICES:-0}}
  Q=name,uuid,driver_version,memory.total,compute_cap,clocks.sm,clocks.max.sm,power.limit,temperature.gpu
  { echo $Q; nvidia-smi -i "$G" --query-gpu=$Q --format=csv,noheader,nounits; } > "$O/gpu.csv" 2>&1
  nvidia-smi -i "$G" --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$O/gpu-mem.txt" 2>/dev/null & SM_=$!
  "$@"; rc=$?
  kill $SM_ 2>/dev/null; wait $SM_ 2>/dev/null
  printf "lease\tend\t%s\n" "$(date +%s.%N)" >> "$O/phases.tsv"
  exit $rc' _ "$O" env OUT="$S/prove" RESEARCH_RUN_DIR="$S/prove" bash $SWEEP "${ARGS[@]}" GPU=1 FLOCK_DRAW_LAW="$O/law.json" > "$O/prove.txt" 2>&1
rc=$?
log "sessions: exit $rc"
D=$(ls -d "$S"/prove/classes/*/ 2>/dev/null | head -1)
cp "$S/prove/classes/results.jsonl" "$O/results.jsonl" 2>/dev/null
[ -n "$D" ] && for f in prove.out prove.err serve.log sessions/index.jsonl; do [ -e "$D/$f" ] && cp "$D/$f" "$O/$(basename "$f")"; done
PT probe --out "$O/hardware.json" --gpu-csv "$O/gpu.csv"
printf '{"workload": "%s", "B": %s, "warm": %s, "runs": %s, "run": "%s"}\n' "$WORKLOAD" "$B" "${WARM:-1}" "${RUNS:-3}" "${RESEARCH_RUN_ID:-local}" > "$O/point.json"
if [ "${ZK:-0}" = 1 ]; then
  PT zk-point --workload "$WORKLOAD" --B "$B" --dir "$O" --out "${RESEARCH_RUN_DIR:-$O}/result.json" | tee -a "$O/job.log" || rc=1
  log "done"; [ $rc = 0 ]; exit
fi
C=$(PT staged --results "$O/results.jsonl" --cache-dir 2>/dev/null)
[ -n "$C" ] && cp "$C/rec.json" "$O/stage-rec.json"
PT point-result --workload "$WORKLOAD" --B "$B" --dir "$O" --out "${RESEARCH_RUN_DIR:-$O}/result.json"

# the bundle verify.sh needs: the statement as the verifier builds it (circuit, public file), every session and this run's records
if [ "${BUNDLE:-1}" = 1 ] && [ -n "$D" ] && [ -n "$C" ] && [ -s "$O/results.jsonl" ]; then
  BD=$S/bundle; mkdir -p "$BD"
  cp "$C/circuit.txt" "$C/pub-$PT_N.bin" "$BD/" && cp -r "$D/sessions" "$BD/sessions"
  for f in $BUNDLED; do [ -e "$O/$f" ] && cp "$O/$f" "$BD/"; done
  du -sb "$BD" | tee -a "$O/job.log"
  PT put-bundle --tree "$BD" --out "$O/bundle.json" \
    --meta "{\"what\": \"proof_throughput bundle\", \"workload\": \"$WORKLOAD\", \"B\": $B, \"run\": \"${RESEARCH_RUN_ID:-local}\"}" \
    > "$O/bundle.id" 2> "$O/bundle.err" && log "bundle: $(cat "$O/bundle.id")" || { log "bundle put failed: $(tail -3 "$O/bundle.err")"; rc=1; }
fi
log "done"
[ $rc = 0 ]
