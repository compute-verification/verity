#!/usr/bin/env bash
# pfill's cost probes on one leased GPU (cwd = the source tree): dgate first, then each probe's rate alone (dcap dalone)
# and ncu's warp-state, scheduler and instruction sections for one chunk of each config in PF_NCU. Clocks are read, never
# set.  PF_ALONE ("W:NC ..."), PF_NCU ("W:NC ..."), PF_S (fragments a warp a tile), PF_NVCC (more nvcc flags).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/pfprobe.XXXXXX); trap 'rm -rf "$B"' EXIT
export PF_ALONE=${PF_ALONE:-"8:4 8:1 8:5 8:3 8:6 8:7 8:8"} PF_NCU=${PF_NCU:-"8:5 8:3"} PF_S=${PF_S:-256}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
nvcc -O3 -std=c++17 ${PF_NVCC:-} -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap" $D/dcap.cu \
  -lcublasLt -lnvidia-ml -ldl > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; exit 4; }
cuobjdump -sass "$B/dcap" | awk '/Function : /{p = /dfill_k/} p' > "$O/sass-dfill.txt" 2>&1
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-15}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$B/dcap" dgate > "$O/dgate.jsonl" 2>> "$O/errors.txt" || exit 5
  for a in $PF_ALONE; do
    "$B/dcap" dalone "${a%:*}" "${a#*:}" "$PF_S" >> "$O/dalone.jsonl" 2>> "$O/errors.txt" || echo "rc $? for dalone $a" >> "$O/errors.txt"
  done
  for a in $PF_NCU; do
    ncu --clock-control none -k regex:dfill_k --launch-count 1 --launch-skip 1 --section WarpStateStats --section SchedulerStats \
      --section InstructionStats --section ComputeWorkloadAnalysis "$B/dcap" done "${a%:*}" "${a#*:}" "$PF_S" \
      > "$O/ncu-sections-w${a%:*}-nc${a#*:}.txt" 2>&1; echo "ncu $a rc $?" >> "$O/rc.txt"
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O/dgate.jsonl" "$O/rc.txt" 2>/dev/null; cut -c1-240 "$O/dalone.jsonl" 2>/dev/null
exit "$r"
