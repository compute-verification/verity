#!/usr/bin/env bash
# tpcomm.cu on TC_TP leased GPUs of one NUMA node (cwd = the source tree): the 70B tensor-parallel decode step's arms (base,
# nocomm, comm, and the phase filler, pfill-k, beside every collective) at TC_BLS, one process a rank. The filler's gates
# run on every rank before anything is timed (fixed work against pfill's dfill_ref, the flag already up, a mid-tile cut
# against kf_ref_warp), and its records are re-checked after each arm. Before the sets, on the first leased GPU: the filler
# alone (its rate), and ncu's DRAM bytes for its launches against dcap's bw_read (1 GiB a launch) as the counter's control,
# the filler scorer's DRAM gate. The SASS of kfill_k is recorded (sass.txt: its MMA opcodes). Clocks are read, never set:
# gpu.txt has the scorer's clock fields (application clocks against their defaults) before and after each set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     --env TC_TP=2 -- bash benchmarks/pouw/filler/comm/tpcomm.sh
#
# TC_TP (2), TC_ON (the GPUs gpu-lease may take, in order; default one NUMA node's), TC_BLS ("B:L ..."; default the
# suite proposal's points for TC_TP), TC_REPS, TC_SETS
# (";"-separated sets of VAR=VALUE settings, each a full pass over TC_BLS, e.g. "TPC_S=64;TPC_S=128"), MAXMIN, TC_WAIT
# (1; 0 takes the lease only if its GPUs are free now, so a TP 8 attempt never queues ahead of other jobs), TC_NCU (1; 0
# skips the alone probe and the DRAM gate).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler/comm
export B; B=$(mktemp -d /tmp/tpcomm.XXXXXX); trap 'rm -rf "$B"' EXIT
export TC_TP=${TC_TP:-2} TC_REPS=${TC_REPS:-9}
export TC_ON=${TC_ON:-4,5,6,7}
BLS="32:2048 64:2048 128:2048 256:2048 32:8192 128:8192"
[ "$TC_TP" = 2 ] && BLS="32:2048 64:2048 128:2048 256:2048 32:8192 64:8192"  # TP 2's KV wraps at B 128, 8K
export TC_BLS=${TC_BLS:-$BLS}
export TC_SETS=${TC_SETS:-"NCCL_CGA_CLUSTER_SIZE=1 TPC_S=64 TPC_DT=8192;NCCL_CGA_CLUSTER_SIZE=1 TPC_S=64 TPC_DT=16384"}
export TC_NCU=${TC_NCU:-1}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
nvcc --version | tail -2 > "$O/nvcc.txt"
grep -m1 -E "define NCCL_(MAJOR|VERSION_CODE)" /usr/include/nccl.h >> "$O/nvcc.txt" 2>&1
nvidia-smi topo -m > "$O/topo.txt" 2>&1
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/tpcomm" $D/tpcomm.cu -lcublasLt -lnvidia-ml -ldl -lnccl \
  > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
grep -A3 -E "Compiling entry function '_Z(7kfill_k|10tp_dpass_k|10tp_drain_k)" "$O/build.log" | grep -E "Compiling|Used|spill" > "$O/regs.txt"
# the filler path (tp_dfill_k) and the deferred kernels must be only IMMA.16832.S8.S8 with C = RZ (SCORER v4 / Theorem K):
# tp_dfill_k inlines into kfill_k and tp_dpass_k, so record each launched kernel's MMAs
: > "$O/sass.txt"
for fn in kfill_k tp_dpass_k tp_drain_k; do
  echo "== $fn ==" >> "$O/sass.txt"
  cuobjdump -sass "$B/tpcomm" | awk -v f="$fn" '/Function : /{p = ($0 ~ f)} p' | grep -oE "\b[A-Z]*MMA[.A-Z0-9_]*" | sort | uniq -c >> "$O/sass.txt"
  nonrz=$(cuobjdump -sass "$B/tpcomm" | awk -v f="$fn" '/Function : /{p = ($0 ~ f)} p' | grep -E "IMMA" | grep -vc "RZ")
  echo "  non-RZ IMMA: $nonrz" >> "$O/sass.txt"
done
# shellcheck disable=SC2016
WAIT=--wait; [ "${TC_WAIT:-1}" = 0 ] && WAIT=
gpu-lease "$TC_TP" $WAIT --on "$TC_ON" --max-min "${MAXMIN:-12}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks.applications.graphics,clocks.default_applications.graphics,clocks_event_reasons.active \
    --format=csv,noheader,nounits -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GPU_LEASE_UUID=$GPU_LEASE_UUID" > "$O/lease.txt"
  if [ "$TC_NCU" = 1 ]; then
    "$B/tpcomm" alone 9 > "$O/alone.jsonl" 2> "$O/alone.err"; echo "alone rc $?" >> "$O/rc.txt"
    m=gpu__time_duration.sum,dram__bytes.sum
    ncu --clock-control none -k regex:kfill_k --launch-count 3 --metrics $m --csv "$B/tpcomm" alone 3 \
      > "$O/ncu-filler.csv" 2> "$O/ncu-filler.err"; echo "ncu filler rc $?" >> "$O/rc.txt"
    GPU_LEASE_UUID=${GPU_LEASE_UUID%%,*} ncu --clock-control none -k regex:bw_read --launch-count 3 --metrics $m --csv "$B/tpcomm" bw 1 3 \
      > "$O/ncu-control.csv" 2> "$O/ncu-control.err"; echo "ncu control rc $?" >> "$O/rc.txt"
    q
  fi
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
cat "$O/rc.txt" "$O/regs.txt" "$O/sass.txt"
cat "$O/alone.jsonl" 2>/dev/null
python3 $D/report.py "$O" 2>&1 | tail -40
grep -hE "error|failed|WARN" "$O"/set*/rank*.err 2>/dev/null | head -10
exit "$r"
