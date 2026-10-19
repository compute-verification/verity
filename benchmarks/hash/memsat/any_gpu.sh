#!/usr/bin/env bash
# memsat's run.sh on whichever GPU the lease gave (cwd = the source tree): ENGINE_CPUS and PROVER_CPUS, 12 cores each,
# from the first core of that GPU's NUMA affinity (`nvidia-smi topo -m`), unless given.
set -uo pipefail
g=$(nvidia-smi --query-gpu=index --format=csv,noheader -i "${GPU_LEASE_UUID:?}" | head -1)
aff=$(nvidia-smi topo -m | awk -v g="GPU$g" '$1 == g {print $(NF - 2)}')
lo=${aff%%-*}
[[ $lo =~ ^[0-9]+$ ]] || { echo "no CPU affinity for GPU $g in nvidia-smi topo -m: $aff"; exit 3; }
export ENGINE_CPUS=${ENGINE_CPUS:-$lo-$((lo + 11))} PROVER_CPUS=${PROVER_CPUS:-$((lo + 12))-$((lo + 23))}
echo "GPU $g affinity $aff: ENGINE_CPUS=$ENGINE_CPUS PROVER_CPUS=$PROVER_CPUS"
exec bash benchmarks/hash/memsat/run.sh
