#!/usr/bin/env bash
# Flock's zero-knowledge session cost on the run's GPU (zk_cost.py): A6's W8A8 coordinates in statements of 1-512 units, sessions
# by `flock-circuit prove --zk --gpu` with and without the ZK server's send check, the Lean verifier on them, and the fit. Run
# from the verity tree on a GPU node, as a GPU host job (the GPU taken by gpu-lease around the sessions only):
#   bash benchmarks/one_stage/zk_cost.sh
#   (a job run from a clone of the tree, on a GPU host, 32 CPUs, 96 GB, up to 240 min)
# (a guest job on a shared GPU node is preemptible; one arm a run, ZK_ARMS=fw, takes about 3.5 h with its verification, the builds cached)
# env: ZK_KS=4096,14336  ZK_SIZES=1,8,64,512  ZK_SAMPLES=1,8,64,c  ZK_ARMS=fw,nofw  ZK_COLD=2  ZK_RUNS=3  ZK_WARM=1
#      ZK_UNITS="fp8-coord:K=6144 router-row" (served types instead of A6's coordinates, zk_cost.py's --units: ZK_SIZES then
#      defaults to half,full, ZK_SAMPLES to c:dev+nofw,8:dev+nofw+fw,64:dev+nofw+fw, ZK_ARMS to dev,nofw,fw; overhead instead of fit)
#      ZK_STAGE_ONLY=1 (with ZK_UNITS: stage on this node's CPUs, keep the records, no CUDA, build or GPU)
#      ZK_SM (the prover's target, from nvidia-smi unless given)  ZK_CARD, with ZK_UNITS and no default (the card file
#      whose datasheet peak divides: rtx-pro-6000-server, or h200-sxm-141gb on an H200; the overhead step refuses one whose
#      arch isn't the run's GPU's)
#      LEASE_MAX_MIN=30 (one GPU lease a point)
#      ZK_VERIFY=0 (no Lean verify: the sessions' prices alone, each accepted by its ZK server)
#      FLOCK_WORK=/workspace/jobs/proof-throughput (the CUDA prover's build and venv, as proof_throughput's job.sh keeps them)
#      PT_WORK=/workspace/cache/proof-throughput (the Lean verifier's build, as its verify.sh keeps it)
# Kept in $OUT: the records (stage, prove and verify .jsonl), the fit (fit.json, fit.md, observations.json) and the logs;
# never a circuit, a prover's file or a proof.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
PUBF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/public_files.py
for o in ZK_DRAWS:ZK_SAMPLES; do n=${o#*:} o=${o%%:*}; [ -z "${!o+x}" ] || [ -n "${!n+x}" ] \
  || { echo "zk-cost: $o is set and $n is not: it is read as $n now; set $n instead" >&2; exit 2; }; done
[ -z "${ZK_UNITS:-}" ] || [ -n "${ZK_CARD:-}" ] \
  || { echo "zk-cost: ZK_UNITS needs ZK_CARD, the run's GPU's card file (rtx-pro-6000-server, h200-sxm-141gb)" >&2; exit 2; }
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proof-throughput}; mkdir -p "$FLOCK_WORK" || { echo "cannot make $FLOCK_WORK" >&2; exit 1; }
PT_WORK=${PT_WORK:-/workspace/cache/proof-throughput}
mkdir -p "$PT_WORK" 2>/dev/null && [ -w "$PT_WORK" ] || PT_WORK=$FLOCK_WORK/verifier
S=$(mktemp -d "$FLOCK_WORK/zk-cost.XXXXXX"); trap 'rm -rf "$S"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
export PYTHONDONTWRITEBYTECODE=1
unset CARGO_TARGET_DIR
{ lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; } > "$O/host.txt" 2>&1
PY=$FLOCK_WORK/flock-circuit/py/bin/python3
if [ ! -x "$PY" ]; then
  UV=$(command -v uv || echo "$HOME/.local/bin/uv"); mkdir -p "$FLOCK_WORK/flock-circuit"
  (exec 8>"$FLOCK_WORK/flock-circuit/venv.lock"; flock 8; [ -x "$PY" ] || { "$UV" venv -q --python 3.12 "$FLOCK_WORK/flock-circuit/py" \
    && "$UV" pip install -q --python "$PY" numpy blake3; })
fi
export PYTHONPATH=$REPO:$REPO/verity/protocols/proofs/flock/prover/python:$REPO/adapters/vllm:$REPO/adapters/torch:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/numerical/python:$REPO/benchmarks/harness:$REPO/tools/config
Z=$S/zk
ZK() { "$PY" benchmarks/one_stage/zk_cost.py "$@" --out "$Z"; }
keep() {
  (cd "$Z" && find . -type f -size -8M ! -name circuit.txt ! -name 'inst*.bin' ! -name 'pub*.bin' ! -path '*/archive/*' \
    \( ! -path '*/sessions/*/*' -o -name session.json \) -print0 | python3 "$PUBF" filter | xargs -0 -r -I{} cp --parents {} "$O/")
}
# a served type stages on CPUs alone (stage.py, no Lean), before the builds
if [ -n "${ZK_UNITS:-}" ]; then
  : "${ZK_SIZES:=half,full}" "${ZK_SAMPLES:=c:dev+nofw,8:dev+nofw+fw,64:dev+nofw+fw}" "${ZK_ARMS:=dev,nofw,fw}"
  log "zk-cost: source ${RESEARCH_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null)}; stage: units $ZK_UNITS, sizes $ZK_SIZES, draws $ZK_SAMPLES"
  ZK stage --units "$ZK_UNITS" --sizes "$ZK_SIZES" --draws "$ZK_SAMPLES" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
  rc=${PIPESTATUS[0]}
  if [ "${ZK_STAGE_ONLY:-0}" = 1 ] || [ "$rc" != 0 ]; then keep; log "zk-cost: staged only, rc $rc"; exit "$rc"; fi
fi
# CUDA 13.3: CUDA_TK where given, else the machine's toolkit at 13.3 (benchmarks/kernels/machine.py), else the one
# verity/protocols/proofs/flock/prover/pod/cuda-13.3.sh finds or unpacks under FLOCK_WORK without root (an R580 driver
# runs 13.3), one unpacking at a time
[ -n "${CUDA_TK:-}" ] || CUDA_TK=$(python3 benchmarks/kernels/machine.py toolkit 13.3 2> "$O/cuda-unpack.log") \
  || { CUDA_TK=; (exec 8>"$FLOCK_WORK/cuda-root.lock"; flock 8; . verity/protocols/proofs/flock/prover/pod/cuda-13.3.sh) \
         >> "$O/cuda-unpack.log" 2>&1 && . verity/protocols/proofs/flock/prover/pod/cuda-13.3.sh >> "$O/cuda-unpack.log" 2>&1; }
[ -n "${CUDA_TK:-}" ] || { log "no CUDA 13.3 on this node"; tail -5 "$O/cuda-unpack.log"; exit 1; }
export CUDA_TK; C13=$CUDA_TK; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}

log "zk-cost: source ${RESEARCH_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null)}, builds under $FLOCK_WORK and $PT_WORK"
OUT=$O/build bash verity/protocols/proofs/flock/prover/pod/70-class-sweep.sh BUILD_ONLY=1 GPU=1 SM="${ZK_SM:-}" > "$O/build.txt" 2>&1 &
OUT=$O/lean bash benchmarks/proof_throughput/verify.sh BUILD_ONLY=1 PT_WORK="$PT_WORK" > "$O/lean-build.txt" 2>&1 &
wait
BIN=$(sed -n 's/^BINARY=//p' "$O/build.txt" | tail -1)
LEAN=$(sed -n 's/.*BUILD_ONLY: \(\S*\) .*/\1/p' "$O/lean/job.log" 2>/dev/null | tail -1)
# the ZK server's Lean frame (FC_ZK_SERVER=1 spawns it per rep), built in the same package as verity-verify-proof
export FC_ZK_SERVER_LEAN=${LEAN%/*}/flock-zk-server
if [ -x "$LEAN" ] && [ ! -x "$FC_ZK_SERVER_LEAN" ]; then
  command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
  (exec 8>"$PT_WORK/lean.lock"; flock 8; [ -x "$FC_ZK_SERVER_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-zk-server)) \
    > "$O/lake-zk-server.log" 2>&1
fi
for p in "$BIN" "$LEAN" "$FC_ZK_SERVER_LEAN" "$PY"; do
  [ -x "$p" ] || { log "missing: '$p'"; tail -20 "$O/build.txt" "$O/lean-build.txt" "$O/lake-zk-server.log"; exit 1; }
done
sha256sum "$BIN" "$LEAN" "$FC_ZK_SERVER_LEAN" | tee "$O/binaries.sha256" | tee -a "$O/job.log"

if [ -z "${ZK_UNITS:-}" ]; then
  log "stage: K ${ZK_KS:-4096,14336}, sizes ${ZK_SIZES:-1,8,64,512}, draws ${ZK_SAMPLES:-1,8,64,c}"
  ZK stage --lean "$LEAN" --ks "${ZK_KS:-4096,14336}" --sizes "${ZK_SIZES:-1,8,64,512}" --draws "${ZK_SAMPLES:-1,8,64,c}" 2>&1 \
    | tee -a "$O/zk.log" | grep ZKCOST
  rc=${PIPESTATUS[0]}
fi
if [ "$rc" = 0 ]; then
  if [ -x /workspace/verity-guest/bin/gpu-lease ] && [ -e /run/gpu-lease/pool.gpus ]; then
    LEASE=(env GPU_LEASE_DIR=/run/gpu-lease GPU_LEASE_ALLOWED_FILE=/run/gpu-lease/pool.gpus /workspace/verity-guest/bin/gpu-lease)
  else
    LEASE=(gpu-lease)
  fi
  export GPU_LEASE_WHO=${GPU_LEASE_WHO:-proofs/zk-cost/${RESEARCH_RUN_ID:-local}}
  log "prove: arms ${ZK_ARMS:-fw,nofw}, ${ZK_COLD:-2} cold, ${ZK_WARM:-1}+${ZK_RUNS:-3} warm (waiting for a GPU)"
  # one point's sessions at a time on this node, as proof_throughput's: two would share its host CPUs and memory bandwidth;
  # a lease a point keeps each lease short
  n=$("$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$Z/plan.json")
  for i in $(seq 0 $((n - 1))); do
    flock "$FLOCK_WORK/sessions.lock" "${LEASE[@]}" 1 --wait --max-min "${LEASE_MAX_MIN:-30}" -- \
      "$PY" benchmarks/one_stage/zk_cost.py prove --bin "$BIN" --out "$Z" --points "$i" --arms "${ZK_ARMS:-fw,nofw}" \
      --cold "${ZK_COLD:-2}" --warm "${ZK_WARM:-1}" --runs "${ZK_RUNS:-3}" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
    [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  done
  if [ "${ZK_VERIFY:-1}" = 1 ]; then
    log "prove: exit $rc; verify"
    ZK verify --lean "$LEAN" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
    [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  fi
  if [ -n "${ZK_UNITS:-}" ]; then ZK overhead --card "$ZK_CARD" 2>&1 | tee -a "$O/zk.log"
  else ZK fit 2>&1 | tee -a "$O/zk.log"; fi
fi
keep
log "zk-cost: done, rc $rc"
exit "$rc"
