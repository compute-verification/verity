#!/usr/bin/env bash
# The ingress tie in zero knowledge (n2_tie_zk.py) on a CPU machine, on hidden_zk's builds (its job.sh, sourced:
# C-Flock's flock-circuit and the Lean flock-verify) with the ZK server's Lean frame and the sampled-proofs verifier
# built in the tree, as n2_zk.sh runs n2_zk.py. From a clone, in one of the network slots:
#   research run --on MACHINE --source . --project verity --as LANE --cwd clone --timeout 180m --env N2_MEM_GB=56 -- \
#     env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/network_traces/n2_tie_zk.sh
# On a GPU node with N2_GPU=1, the prover is a8_gpu.sh's CUDA build (70-class-sweep.sh BUILD_ONLY=1, kept under
# FLOCK_WORK) on the GPU the run leased:
#   research run --on GPU_NODE --gpus 1 --cpus 32 --mem-gb 96 --source . --project verity --cwd clone --timeout 180m \
#     --env N2_MEM_GB=96 --env N2_GPU=1 --env N2_CLASS=served -- bash benchmarks/network_traces/n2_tie_zk.sh
# env: N2_MEM_GB (what job.sh's stages may take, within the slot's)  N2_PORT (--port; the members are proved one after
#      another)  N2_CACHE (the build cache, kept across runs where writable)  N2_DEBUG_CLEAR=1
#      (--debug-clear)  N2_DRY=1 (the instances and references only)  N2_CLASS (--class: test, served or warden)
#      N2_WARDEN_RUN (--warden-run, N2_CLASS=warden's warden run, an absolute path on the node)  N2_WINDOWS (--windows)
#      N2_GPU=1 (--gpu)  FLOCK_WORK (N2_GPU's build, /workspace/jobs/proof-throughput by default)
# Kept in $OUT/n2tie: the driver's records and logs, the public files and everything the verifier received; never a
# prover's file (inst-*, prover-instances.bin) nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=n2tie HZ_MEM_GB=${N2_MEM_GB:?}
if [ "${N2_GPU:-0}" = 1 ]; then
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "N2_GPU=1 needs the run's leased GPU (research run --gpus 1)" >&2; exit 7; }
  export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proof-throughput}; mkdir -p "$FLOCK_WORK" || exit 7
  unset CARGO_TARGET_DIR
  for c in "$FLOCK_WORK/cuda-root" /workspace/cache/flock/cuda-root; do
    [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x "$c/usr/local/cuda-13.3/bin/nvcc" ] \
      && export CUDA_TK=$c/usr/local/cuda-13.3
  done
  C13=${CUDA_TK:-/usr/local/cuda-13.3}; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
  G=${RESEARCH_RUN_DIR:?}/out/build-gpu; mkdir -p "$G"
  OUT=$G bash verity/ml/flock/pod/70-class-sweep.sh BUILD_ONLY=1 GPU=1 SM=${SM:-120} > "$G/build.txt" 2>&1
  FLOCK_PROVE=$(sed -n 's/^BINARY=//p' "$G/build.txt" | tail -1)
  FLOCK_SELFTEST=$(sed -n 's/^SELFTEST_BINARY=//p' "$G/build.txt" | tail -1)
  [ -x "$FLOCK_PROVE" ] || { tail -30 "$G/build.txt"; echo "no CUDA flock-circuit" >&2; exit 7; }
  export FLOCK_PROVE FLOCK_SELFTEST
fi
if [ -n "${N2_CACHE:-}" ] && mkdir -p "$N2_CACHE" 2>/dev/null && [ -w "$N2_CACHE" ]; then export HZ_CACHE=$N2_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "n2tie: no firewall frame"; exit 7; }

log "n2tie: proving${N2_DRY:+, dry}"
GPU_ARG= SMI=
if [ "${N2_GPU:-0}" = 1 ]; then
  GPU_ARG=--gpu
  { nvidia-smi -L; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; } >> "$OUT/host.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 1 -i "$CUDA_VISIBLE_DEVICES" \
    > "$OUT/gpu-samples.csv" 2>&1 &
  SMI=$!
fi
"$PY" benchmarks/network_traces/n2_tie_zk.py prove --bin "$BIN" --lean "$LEAN" --out "$W/n2tie" \
  --class "${N2_CLASS:-test}" $GPU_ARG \
  ${N2_WARDEN_RUN:+--warden-run "$N2_WARDEN_RUN"} ${N2_WINDOWS:+--windows "$N2_WINDOWS"} \
  ${N2_PORT:+--port "$N2_PORT"} ${N2_DEBUG_CLEAR:+--debug-clear} ${N2_DRY:+--dry} 2>&1 \
  | tee -a "$OUT/n2tie.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
[ -n "$SMI" ] && kill "$SMI"
mkdir -p "$OUT/n2tie"
(cd "$W/n2tie" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name prover-instances.bin ! -path '*/archive/*' \
  -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/n2tie/")
log "n2tie: done, rc $rc"
exit "$rc"
