#!/usr/bin/env bash
# run.sh inside a lease on a node no verifier machine reaches (vy-dc-h200, vy-dc-b200: no node's key opens ssh to
# them, so run.sh's ROLE=verifier tunnel has no PEER): ROLE=both, the verifier beside the prover on cores of its own,
# which remote_session/node.sh calls a smoke test, not the audit. session.cu and kv.cu are built for the leased GPU's
# arch; vLLM's venv runs the engine and the prover; the verifier's memory line takes this GPU's HBM and this host's
# DRAM. The rest is the RTX served arm's recipe, set by the caller (PLAN, WINDOW_MS, SAMPLES, ENGINE_ARGS, NODE_ARGS).
set -uo pipefail
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${GPU_LEASE_UUID:?}" | head -1 | tr -d .)
hbm=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_LEASE_UUID" | head -1)
[[ $cc =~ ^[0-9]+$ && $hbm =~ ^[0-9]+$ ]] || { echo "no compute capability or HBM for $GPU_LEASE_UUID"; exit 3; }
n=$(nproc); e=$(((n - 4) / 2))
export ARCH=${cc}a ROLE=both PY=${PY:-/workspace/cp/venv312/bin/python} VLLM_PY=${VLLM_PY:-/workspace/cp/venv312/bin/python}
export ENGINE_CPUS=${ENGINE_CPUS:-0-$((e - 1))} PROVER_CPUS=${PROVER_CPUS:-$e-$((n - 5))} VERIFIER_CPUS=${VERIFIER_CPUS:-$((n - 4))-$((n - 1))}
export VERIFIER_ARGS="--hbm-bytes $((hbm * 1048576)) --host-bytes $(awk '/^MemTotal/ {print $2 * 1024}' /proc/meminfo)"
echo "sm_$ARCH engine $ENGINE_CPUS prover $PROVER_CPUS verifier $VERIFIER_CPUS $VERIFIER_ARGS"
exec bash benchmarks/hash/memsat/run.sh
