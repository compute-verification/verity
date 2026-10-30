#!/usr/bin/env bash
# One card's sustained int8 rate per operand class (class_rate.cu), alone and beside SHA-512 at RATIOS M compressions a
# T MAC, at the declared clock: under one gpu-lease the SM clock is locked to CLOCK MHz (nvidia-smi -lgc) and reset at
# exit, then ROUNDS rounds of every class through each kernel and encoding, SECS seconds an arm. Timings only: the one
# nvidia-smi query is of clocks. The build runs outside the lease, in a scratch directory.
#
#   research run --on NODE --project verity --source . --cwd source --env GPU_LEASE_WHO=compute-accounting \
#     -- bash benchmarks/int8_class_rate/run.sh
#
# Env: CLASSES (default "s8 s2 u2 u4 zero"), ENCODINGS (wgmma's, default "s8s8 u8u8"; u8u8 runs on s8 and s2, the classes
# an offset changes), KERNELS (default "wgmma lt"), RATIOS (default "0 1.70 2.3", 0 first), SECS (15), LT_SECS (10),
# ROUNDS (2), CLOCK (1830), LEASE_MIN (30). cuBLASLt runs alone only: beside it the hash's resident CTAs, though paced,
# hold the SMs its GEMM's CTAs need, which measures scheduling, not the rate.
# Writes $OUT/rates.jsonl (class_rate's lines, each with its round), card.json and summary.txt (summarize.py's reading
# of this card).
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../.." && pwd)
O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH CUDA_MODULE_LOADING=EAGER
export CLASSES=${CLASSES:-s8 s2 u2 u4 zero} ENCODINGS=${ENCODINGS:-s8s8 u8u8} KERNELS=${KERNELS:-wgmma lt}
export RATIOS=${RATIOS:-0 1.70 2.3} SECS=${SECS:-15} LT_SECS=${LT_SECS:-10} ROUNDS=${ROUNDS:-2} CLOCK=${CLOCK:-1830}
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
nvcc --version > "$O/nvcc.txt" 2>&1
nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -I"$D" -I"$R/benchmarks/torch_gpu/sha512_rows" \
  -o "$S/class_rate" "$D/class_rate.cu" -lcublasLt > "$O/build.log" 2>&1
echo "build rc $?" | tee -a "$O/rc.txt"
[ -x "$S/class_rate" ] || exit 1
export BIN=$S/class_rate O
gpu-lease 1 --wait --max-min "${LEASE_MIN:-30}" -- bash -c '
  dev=${GPU_LEASE_UUID:+-i $GPU_LEASE_UUID}
  q="--query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics --format=csv,noheader"
  sudo -n nvidia-smi $dev -lgc "$CLOCK,$CLOCK" > "$O/clock.txt" 2>&1
  lock=$?
  trap "sudo -n nvidia-smi $dev -rgc >> \"$O/clock.txt\" 2>&1" EXIT
  nvidia-smi $dev $q > "$O/gpu-before.txt" 2>&1
  for r in $(seq 1 "$ROUNDS"); do
    for k in $KERNELS; do
      for c in $CLASSES; do
        if [ "$k" = lt ]; then
          "$BIN" lt "$c" "$LT_SECS" 0 2>> "$O/class_rate.err" | sed "s/^{/{\"round\": $r, /" >> "$O/rates.jsonl"
          echo "round $r lt $c rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
          continue
        fi
        for e in $ENCODINGS; do
          [ "$e" != s8s8 ] && [ "$c" != s8 ] && [ "$c" != s2 ] && continue
          "$BIN" wgmma "$c" "$e" "$SECS" $RATIOS 2>> "$O/class_rate.err" | sed "s/^{/{\"round\": $r, /" >> "$O/rates.jsonl"
          echo "round $r wgmma $c $e rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
        done
      done
    done
  done
  nvidia-smi $dev $q > "$O/gpu-after.txt" 2>&1
  echo "lock rc $lock" >> "$O/rc.txt"
'
echo "lease rc $?" | tee -a "$O/rc.txt"
python3 - "$O" <<'EOF'
import json, os, sys
o = sys.argv[1]
gpu = open(f"{o}/gpu-before.txt").read().strip().split(", ") if os.path.exists(f"{o}/gpu-before.txt") else []
lock = [l for l in open(f"{o}/rc.txt") if l.startswith("lock rc")]
json.dump({"card": os.environ.get("CARD") or os.uname().nodename, "uuid": gpu[1] if len(gpu) > 1 else None,
           "name": gpu[2] if len(gpu) > 2 else None, "clock_mhz": int(os.environ["CLOCK"]),
           "clock_locked": bool(lock) and lock[-1].split()[-1] == "0"}, open(f"{o}/card.json", "w"), indent=1)
EOF
python3 "$D/summarize.py" "$O" | tee "$O/summary.txt"
