#!/usr/bin/env bash
# C-Flock's zero-knowledge session cost on sm_120 (zk_cost.py): A6's W8A8 coordinates in statements of 1-512 units, sessions
# by `flock-circuit prove --zk --gpu` with and without the firewall's send check, the Lean verifier on them, and the fit. Run
# from the verity tree on a GPU node, as a GPU host job (the GPU taken by gpu-lease around the sessions only):
#   research run --queue --on vy-nebius-1 --gpu-host --project verity --source . --cwd clone --mem-gb 96 --cpus 32 \
#     --max-min 240 -- bash benchmarks/one_stage/zk_cost.sh
# env: ZK_KS=4096,14336  ZK_SIZES=1,8,64,512  ZK_ARMS=fw,nofw  ZK_COLD=2  ZK_RUNS=3  ZK_WARM=1  LEASE_MAX_MIN=120
#      FLOCK_WORK=/workspace/jobs/proof-throughput (the CUDA prover's build and venv, as proof_throughput's job.sh keeps them)
#      PT_WORK=/workspace/cache/proof-throughput (the Lean verifier's build, as its verify.sh keeps it)
# Kept in $OUT: the records (stage, prove and verify .jsonl), the fit (fit.json, fit.md, observations.json) and the logs;
# never a circuit, a prover's file or a proof.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proof-throughput}; mkdir -p "$FLOCK_WORK" || { echo "cannot make $FLOCK_WORK" >&2; exit 1; }
PT_WORK=${PT_WORK:-/workspace/cache/proof-throughput}
mkdir -p "$PT_WORK" 2>/dev/null && [ -w "$PT_WORK" ] || PT_WORK=$FLOCK_WORK/verifier
S=$(mktemp -d "$FLOCK_WORK/zk-cost.XXXXXX"); trap 'rm -rf "$S"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
export PYTHONDONTWRITEBYTECODE=1
unset CARGO_TARGET_DIR
for c in "$FLOCK_WORK/cuda-root" /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x "$c/usr/local/cuda-13.3/bin/nvcc" ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
[ -x /usr/local/cuda-13.3/bin/nvcc ] || [ -n "${CUDA_TK:-}" ] || { log "no CUDA 13.3 on this node"; exit 1; }
C13=${CUDA_TK:-/usr/local/cuda-13.3}; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
{ lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; } > "$O/host.txt" 2>&1

log "zk-cost: source ${RESEARCH_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null)}, builds under $FLOCK_WORK and $PT_WORK"
OUT=$O/build bash verity/ml/flock/pod/70-class-sweep.sh BUILD_ONLY=1 GPU=1 SM=120 > "$O/build.txt" 2>&1 &
OUT=$O/lean bash benchmarks/proof_throughput/verify.sh BUILD_ONLY=1 PT_WORK="$PT_WORK" > "$O/lean-build.txt" 2>&1 &
wait
BIN=$(sed -n 's/^BINARY=//p' "$O/build.txt" | tail -1)
LEAN=$(sed -n 's/.*BUILD_ONLY: \(\S*\) .*/\1/p' "$O/lean/job.log" 2>/dev/null | tail -1)
PY=$FLOCK_WORK/flock-circuit/py/bin/python3
if [ ! -x "$PY" ]; then
  UV=$(command -v uv || echo "$HOME/.local/bin/uv")
  (exec 8>"$FLOCK_WORK/flock-circuit/venv.lock"; flock 8; [ -x "$PY" ] || { "$UV" venv -q --python 3.12 "$FLOCK_WORK/flock-circuit/py" \
    && "$UV" pip install -q --python "$PY" numpy blake3; })
fi
# the firewall's Lean frame (FC_FIREWALL=1 spawns it per rep), built in the same package as flock-verify
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
if [ -x "$LEAN" ] && [ ! -x "$FC_FIREWALL_LEAN" ]; then
  command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
  (exec 8>"$PT_WORK/lean.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
    > "$O/lake-firewall.log" 2>&1
fi
for p in "$BIN" "$LEAN" "$FC_FIREWALL_LEAN" "$PY"; do
  [ -x "$p" ] || { log "missing: '$p'"; tail -20 "$O/build.txt" "$O/lean-build.txt" "$O/lake-firewall.log"; exit 1; }
done
sha256sum "$BIN" "$LEAN" "$FC_FIREWALL_LEAN" | tee "$O/binaries.sha256" | tee -a "$O/job.log"
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/numerical/python
Z=$S/zk
ZK() { "$PY" benchmarks/one_stage/zk_cost.py "$@" --out "$Z"; }

log "stage: K ${ZK_KS:-4096,14336}, sizes ${ZK_SIZES:-1,8,64,512}"
ZK stage --lean "$LEAN" --ks "${ZK_KS:-4096,14336}" --sizes "${ZK_SIZES:-1,8,64,512}" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
rc=${PIPESTATUS[0]}
if [ "$rc" = 0 ]; then
  if [ -x /workspace/verity-guest/bin/gpu-lease ] && [ -e /run/gpu-lease/pool.gpus ]; then
    LEASE=(env GPU_LEASE_DIR=/run/gpu-lease GPU_LEASE_ALLOWED_FILE=/run/gpu-lease/pool.gpus /workspace/verity-guest/bin/gpu-lease)
  else
    LEASE=(gpu-lease)
  fi
  export GPU_LEASE_WHO=${GPU_LEASE_WHO:-proofs/zk-cost/${RESEARCH_RUN_ID:-local}}
  log "prove: arms ${ZK_ARMS:-fw,nofw}, ${ZK_COLD:-2} cold, ${ZK_WARM:-1}+${ZK_RUNS:-3} warm (waiting for a GPU)"
  # one point's sessions at a time on this node, as proof_throughput's: two would share its host CPUs and memory bandwidth
  flock "$FLOCK_WORK/sessions.lock" "${LEASE[@]}" 1 --wait --max-min "${LEASE_MAX_MIN:-120}" -- \
    "$PY" benchmarks/one_stage/zk_cost.py prove --bin "$BIN" --out "$Z" --arms "${ZK_ARMS:-fw,nofw}" --cold "${ZK_COLD:-2}" \
    --warm "${ZK_WARM:-1}" --runs "${ZK_RUNS:-3}" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
  rc=${PIPESTATUS[0]}
  log "prove: exit $rc; verify"
  ZK verify --lean "$LEAN" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  ZK fit 2>&1 | tee -a "$O/zk.log"
fi
(cd "$Z" && find . -type f -size -8M ! -name circuit.txt ! -name 'inst-*' ! -name 'pub-*' ! -path '*/archive/*' \
  \( ! -path '*/sessions/*/*' -o -name session.json \) -print0 | xargs -0 -r -I{} cp --parents {} "$O/")
log "zk-cost: done, rc $rc"
exit "$rc"
