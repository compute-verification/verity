#!/usr/bin/env bash
# tpcomm.cu on TC_TP leased GPUs of one NUMA node (cwd = the source tree): the 70B tensor-parallel decode step's arms (base,
# nocomm, comm, and the phase filler beside every collective) at TC_BLS, one process a rank. The phase filler's receipts are
# gated against the host on every rank before anything is timed, and its finished tiles re-checked after each arm. Clocks
# are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=HC-COMM \
#     --env TC_TP=2 -- bash benchmarks/pouw/filler_comm/tpcomm.sh
#
# TC_TP (2), TC_ON (the GPUs gpu-lease may take, in order; default one NUMA node's), TC_BLS ("B:L ..."), TC_REPS, TC_SETS
# (";"-separated sets of VAR=VALUE settings, each a full pass over TC_BLS, e.g. "TPC_S=256 TPC_POLL=16;TPC_S=512"), MAXMIN.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler_comm
export B; B=$(mktemp -d /tmp/tpcomm.XXXXXX); trap 'rm -rf "$B"' EXIT
export TC_TP=${TC_TP:-2} TC_REPS=${TC_REPS:-9}
export TC_ON=${TC_ON:-4,5,6,7}
export TC_BLS=${TC_BLS:-"32:2048 64:2048 128:2048 256:2048 32:8192 128:8192"}
export TC_SETS=${TC_SETS:-"TPC_S=256"}
export DFILL_MAP=${DFILL_MAP:-2}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
nvcc --version | tail -2 > "$O/nvcc.txt"
grep -m1 -E "define NCCL_(MAJOR|VERSION_CODE)" /usr/include/nccl.h >> "$O/nvcc.txt" 2>&1
nvidia-smi topo -m > "$O/topo.txt" 2>&1
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/tpcomm" $D/tpcomm.cu -lcublasLt -lnvidia-ml -ldl -lnccl \
  > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
grep -A2 "Compiling entry function '_Z7cfill_k" "$O/build.log" | grep -E "Compiling|Used|spill" > "$O/regs.txt"
# shellcheck disable=SC2016
gpu-lease "$TC_TP" --wait --on "$TC_ON" --max-min "${MAXMIN:-12}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GPU_LEASE_UUID=$GPU_LEASE_UUID" > "$O/lease.txt"
  i=0
  IFS=";" read -ra sets <<< "$TC_SETS"
  for s in "${sets[@]}"; do
    d="$O/set$i"; mkdir -p "$d"; echo "$s" > "$d/settings.txt"
    # shellcheck disable=SC2086
    env $s TPC_OUT="$d" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING "$B/tpcomm" run "$TC_BLS" "$TC_REPS"
    echo "set$i rc $?" >> "$O/rc.txt"
    q
    i=$((i + 1))
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O/rc.txt"
for f in "$O"/set*/rank0.jsonl; do
  echo "== $f"
  grep -E "pf_gate|\"tp\"" "$f" | cut -c1-330
done
grep -hE "error|failed|WARN" "$O"/set*/rank*.err 2>/dev/null | head -10
exit "$r"
