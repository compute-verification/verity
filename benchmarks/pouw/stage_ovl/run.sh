#!/usr/bin/env bash
# ovl.cu on one leased GPU (cwd = the source tree): the build and stage_cost/prep.py's constants outside the lease; under one
# lease stage.cu's gate at every M, then ovl at each M over KS. Nothing is timed unless the gate passes. Clocks are read,
# never set. Results in $OUT/*.jsonl; `python3 benchmarks/pouw/stage_ovl/report.py $OUT` renders them.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=STAGE-OVL \
#     -- bash benchmarks/pouw/stage_ovl/run.sh
# env: MS (512 1024 2048), KS (1 2 4 8 16 32), ARMS (ovl.cu's default), REPS (5), DRAW_US (300), PLAIN_RB (8192),
# MAXMIN (25), CUDA_DEVICE_MAX_CONNECTIONS (32: a hardware queue per chunk), FR_ROT (0: stage.cu's bits, the commitment's
# rotations on the FMA pipe), PARTS (thru gate ovl: shathru's SHA-512 throughput, the gate, the grid; ovl needs gate;
# beside: ovl.cu's beside at each M:K of BESIDE, 2048:4 2048:16 512:32)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
D=benchmarks/pouw/stage_ovl
export B; B=$(mktemp -d /tmp/stage-ovl.XXXXXX); trap 'rm -rf "$B"' EXIT
export MS=${MS:-512 1024 2048} KS=${KS:-1 2 4 8 16 32} REPS=${REPS:-5} PARTS=${PARTS:-thru gate ovl}
export BESIDE=${BESIDE:-2048:4 2048:16 512:32}
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version | tail -2 > "$O/nvcc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -DFR_ROT="${FR_ROT:-0}" -Xptxas -v -o "$B/ovl" $D/ovl.cu \
  -lcublasLt -lnvidia-ml -ldl > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
sha256sum "$B/ovl" | cut -c1-64 > "$O/ovl.sha256"
python3 benchmarks/pouw/stage_cost/prep.py "$B/c" --rows "${MS// /,}" > "$O/prep.log" 2>&1 || { tail -20 "$O/prep.log"; exit 6; }
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  if [[ " $PARTS " == *" thru "* ]]; then
    "$B/ovl" shathru > "$O/shathru.jsonl" 2>> "$O/errors.txt"; echo "shathru rc $?" >> "$O/rc.txt"
  fi
  if [[ " $PARTS " == *" beside "* ]]; then
    for mk in $BESIDE; do
      "$B/ovl" beside "${mk%%:*}" "${mk##*:}" r3 >> "$O/beside.jsonl" 2>> "$O/errors.txt"; echo "beside $mk rc $?" >> "$O/rc.txt"
    done
  fi
  [[ " $PARTS " == *" gate "* ]] || exit 0
  "$B/ovl" gate "$B/c" $MS > "$O/gate.jsonl" 2>> "$O/errors.txt"; g=$?
  echo "gate rc $g" >> "$O/rc.txt"
  [ "$g" -eq 0 ] || exit 5
  [[ " $PARTS " == *" ovl "* ]] || exit 0
  for M in $MS; do
    "$B/ovl" ovl "$B/c" "$M" $KS "r$REPS" >> "$O/ovl.jsonl" 2>> "$O/errors.txt"; r=$?
    echo "ovl $M rc $r" >> "$O/rc.txt"
    q
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O/rc.txt"
grep -v "^device" "$O/errors.txt" | tail -5 || true
exit "$r"
