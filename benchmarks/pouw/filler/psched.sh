#!/usr/bin/env bash
# dcap.cu's pfill mode on one leased GPU (cwd = the source tree): FILL v1's partitioned DRAM-free filler (the baseline arm)
# and the phase filler in decode's attention on the step's own SMs (HC-SCHED). Under one lease: FILL's receipt gate (dgate),
# the phase kernel's gate (pgate), CAP's gate, and nothing timed unless all pass; then each arm. The round's coins are drawn
# once per run (coins.txt). Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=HC-SCHED \
#     --send benchmarks/pouw/filler/psched.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/psched.sh"'
#
# PS_ARMS: "B:L:FILL_SMS:PMODE:PW:PNC:PS:PARAM[:ENV=V,ENV=V]" (dcap pfill's arguments; FILL v1's filler is W 8, NC 4,
# S 2048), PS_REPS, PS_PGATE_S (the phase S values the gate checks, besides 8; every PS in PS_ARMS must be among them),
# PS_NVCC (more nvcc flags, -DDFILL_V1 unless set: FILL v1's filler; empty builds pfill.cuh's).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/psched.XXXXXX); trap 'rm -rf "$B"' EXIT
export PS_ARMS=${PS_ARMS:-"32:2048:92:fixed:8:4:256:0.8"}
export PS_REPS=${PS_REPS:-15} PS_PGATE_S=${PS_PGATE_S:-"128 256"}
for arm in $PS_ARMS; do
  ps=$(cut -d: -f7 <<< "$arm")
  [[ " $PS_PGATE_S " == *" $ps "* ]] || { echo "arm $arm: PS $ps is not among PS_PGATE_S ($PS_PGATE_S)"; exit 1; }
done
git rev-parse HEAD > "$O/commit.txt" 2>&1
echo "${PS_NVCC--DDFILL_V1}" > "$O/nvcc-flags.txt"
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
nvcc --version | tail -2 > "$O/nvcc.txt"
# shellcheck disable=SC2086
nvcc -O3 -std=c++17 ${PS_NVCC--DDFILL_V1} -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl \
  > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
grep -A2 -E "Compiling entry function '_Z7dfill_" "$O/build.log" | grep -E "spill|Used" | head -12
cuobjdump -sass "$B/dcap" | awk '/Function : /{p = /dfill_[kp]/} p' > "$O/sass-dfill.txt" 2>&1
echo "dfill_k/dfill_p SASS: $(grep -c IMMA "$O/sass-dfill.txt") IMMA, $(grep -cE 'LDG|STG|LDS|STS|LDL|STL' "$O/sass-dfill.txt") memory ops"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  for g in dgate "pgate $PS_PGATE_S" gate; do
    # shellcheck disable=SC2086
    "$B/dcap" $g > "$O/${g%% *}.jsonl" 2>> "$O/errors.txt"; r=$?
    echo "${g%% *} rc $r" >> "$O/rc.txt"
    [ "$r" -eq 0 ] || exit 5
  done
  for arm in $PS_ARMS; do
    IFS=: read -r b l f pm pw pnc ps par envs <<< "$arm"
    # shellcheck disable=SC2086
    env ${envs//,/ } "$B/dcap" pfill "$b" "$l" "$f" 8 4 2048 "$pm" "$pw" "$pnc" "$ps" "$par" "$PS_REPS" \
      >> "$O/pfill.jsonl" 2>> "$O/errors.txt" || echo "rc $? for pfill $arm" >> "$O/errors.txt"
    q
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O/rc.txt" 2>/dev/null
grep -h "\"ok\": false" "$O"/*gate.jsonl 2>/dev/null | head -5
grep -v "^device" "$O/errors.txt" 2>/dev/null | tail -5
exit "$r"
