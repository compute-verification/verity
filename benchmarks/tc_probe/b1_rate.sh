#!/usr/bin/env bash
# b1_rate.cu under `research run` (cwd: the root of a source tree):
#
#   research run --on MACHINE --project verity --source . --cwd source --timeout 20m \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/tc_probe/b1_rate.sh
#
# Outside the lease: the sm_90a build (ARCH), its SASS and each kernel's MMA-class and popc/LOP3 opcode counts. Under one
# gpu-lease (MAX_MIN, default 8): b1_rate over SUSTAIN_S seconds a measurement, with nvidia-smi sampling the SM clock.
# $OUT gets sass.txt, opcodes.txt, rate.jsonl, smi.csv and rc.txt.
set -uo pipefail
export O=${OUT:?}
CUDA=${CUDA:-/usr/local/cuda-13.0}
ARCH=${ARCH:-sm_90a}
export W=${RESEARCH_RUN_DIR:-$O}/work
mkdir -p "$W"
git rev-parse HEAD > "$O/commit.txt" 2>&1
: > "$O/rc.txt"
"$CUDA/bin/nvcc" --version > "$O/nvcc.txt" 2>&1
"$CUDA/bin/nvcc" -O3 -gencode "arch=compute_${ARCH#sm_},code=$ARCH" -Xptxas -v -o "$W/b1_rate" \
  benchmarks/tc_probe/b1_rate.cu > "$O/build.log" 2>&1
echo "build rc $?" >> "$O/rc.txt"
"$CUDA/bin/cuobjdump" -sass "$W/b1_rate" > "$O/sass.txt" 2>&1
echo "sass rc $?" >> "$O/rc.txt"
awk '/Function :/ { f = $3 } f && match($0, /[A-Z][A-Z0-9_]*(\.[A-Z0-9_]+)*/) {
       op = substr($0, RSTART, RLENGTH)
       if (op ~ /MMA|POPC|LOP3|BMMA|IMAD|IADD/) n[f " " op]++ }
     END { for (k in n) print k, n[k] }' "$O/sass.txt" | sort > "$O/opcodes.txt"
grep -qv ' rc 0$' "$O/rc.txt" && { cat "$O/rc.txt" "$O/build.log"; exit 1; }

LEASE=${LEASE-gpu-lease 1 --wait --max-min ${MAX_MIN:-8} --}
# shellcheck disable=SC2016
$LEASE bash -c '
  G=(); [ -n "${GPU_LEASE_UUID:-}" ] && G=(-i "$GPU_LEASE_UUID")
  nvidia-smi "${G[@]}" --query-gpu=timestamp,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv -lms 500 \
    > "$O/smi.csv" 2>&1 &
  smi=$!
  "$W/b1_rate" "${SUSTAIN_S:-2.5}" > "$O/rate.jsonl" 2> "$O/rate.err"
  echo "rate rc $?" >> "$O/rc.txt"
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt" "$O/opcodes.txt" "$O/rate.jsonl" 2> /dev/null
! grep -qv ' rc 0$' "$O/rc.txt"
