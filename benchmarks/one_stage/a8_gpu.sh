#!/usr/bin/env bash
# A8's cost points on sm_120 (a8.py --no-audit --gpu): the opening unit's honest statement staged on the host, then
# sessions by `flock-circuit prove --zk --gpu` with and without the ZK server's send check, each prover process under a GPU
# lease (the device held for the proofs alone), Lean verifying each arm at its largest point. Builds as zk_cost.sh's. Run
# from the verity tree on a GPU node, as a GPU host job:
#   research run --queue --on vy-nebius-1 --gpu-host --project verity --source . --cwd clone --mem-gb 96 --cpus 32 \
#     --max-min 240 -- bash benchmarks/one_stage/a8_gpu.sh
# (on vy-nebius-2 a guest job adds --preemptible)
# env: A8_SAMPLES=64 (the staged units)  A8_POINTS=1,8,32,64  A8_ARMS=fw,nofw  A8_COLD=2  LEASE_MAX_MIN=60 (a prover process's)
#      FLOCK_WORK, PT_WORK as zk_cost.sh's
# Kept in $OUT: a8's records and logs and the sessions' records; never a circuit, a prover's file or a proof's archive.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proof-throughput}; mkdir -p "$FLOCK_WORK" || { echo "cannot make $FLOCK_WORK" >&2; exit 1; }
PT_WORK=${PT_WORK:-/workspace/cache/proof-throughput}
mkdir -p "$PT_WORK" 2>/dev/null && [ -w "$PT_WORK" ] || PT_WORK=$FLOCK_WORK/verifier
S=$(mktemp -d "$FLOCK_WORK/a8.XXXXXX"); trap 'rm -rf "$S"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
export PYTHONDONTWRITEBYTECODE=1
unset CARGO_TARGET_DIR
for c in "$FLOCK_WORK/cuda-root" /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x "$c/usr/local/cuda-13.3/bin/nvcc" ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
[ -x /usr/local/cuda-13.3/bin/nvcc ] || [ -n "${CUDA_TK:-}" ] || { log "no CUDA 13.3 on this node"; exit 1; }
C13=${CUDA_TK:-/usr/local/cuda-13.3}; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
{ lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; nvidia-smi -L; } > "$O/host.txt" 2>&1

log "a8-gpu: source ${RESEARCH_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null)}, builds under $FLOCK_WORK and $PT_WORK"
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
if [ -x /workspace/verity-guest/bin/gpu-lease ] && [ -e /run/gpu-lease/pool.gpus ]; then
  LEASE=(env GPU_LEASE_DIR=/run/gpu-lease GPU_LEASE_ALLOWED_FILE=/run/gpu-lease/pool.gpus /workspace/verity-guest/bin/gpu-lease)
else
  LEASE=(gpu-lease)
fi
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting/a8/${RESEARCH_RUN_ID:-local}}
# one session at a time on this node, as zk_cost.sh's: two would share its host CPUs and memory bandwidth
PREFIX=(flock "$FLOCK_WORK/sessions.lock" "${LEASE[@]}" 1 --wait --max-min "${LEASE_MAX_MIN:-60}" --)
log "a8-gpu: samples ${A8_SAMPLES:-64}, points ${A8_POINTS:-1,8,32,64}, arms ${A8_ARMS:-fw,nofw}, ${A8_COLD:-2} cold"
"$PY" benchmarks/one_stage/a8.py --bin "$BIN" --lean "$LEAN" --out "$S/a8" --no-audit --gpu --lease "$(printf '%q ' "${PREFIX[@]}")" \
  --samples "${A8_SAMPLES:-64}" --points "${A8_POINTS:-1,8,32,64}" --arms "${A8_ARMS:-fw,nofw}" --cold "${A8_COLD:-2}" \
  2>&1 | tee -a "$O/a8.log" | grep '^A8 '
rc=${PIPESTATUS[0]}
mkdir -p "$O/a8"
(cd "$S/a8" && find . -type f -size -8M ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' \
  \( ! -path '*/sessions/*/*' -o -name session.json \) -print0 | xargs -0 -r -I{} cp --parents {} "$O/a8/")
log "a8-gpu: done, rc $rc"
exit "$rc"
