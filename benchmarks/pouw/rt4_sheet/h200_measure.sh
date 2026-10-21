#!/usr/bin/env bash
# RT4-SHEET-21: the H200 run for the store and ISETP lines and li-bulk's measured cost. From a pushed commit:
#
#   research run --on vy-dc-h200 --project verity --campaign <c> --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=<you> -- bash benchmarks/pouw/rt4_sheet/h200_measure.sh
#
# Two independent parts, each recorded whatever the other does:
#   dc/       dc/run.sh sm90 (da7f's harness) timing only the store lanes STS32, STS128, STG32, STG128 and ISETP alone
#             (s_), beside the IGMMA until it stops (wgU_) and interleaved with it (wgmix_), with LDS128 and the tensor
#             alone as same-run references (DC_ONLY; DC_PASSED_ONLY=1 so one refused probe doesn't stop the rest)
#   li_bulk/  li_bulk.cu (bit-exact against li's CPU reference, then timed alone) and li_bulk_wg.cu (beside the IGMMA),
#             built for sm_90a, their SASS and ptxas reports, sass_gate.py's verdict on each (recorded, not blocking:
#             li-bulk's loop has no exact-count expect), then under one gpu-lease: both binaries, nvidia-smi sampled
#             every 200 ms, and ncu's L2 counters on lbl and lbw when ncu is on the node and may read them.
# parts.json has each step's exit code. Exit 0 when the dc part ran and li-bulk is bit-exact, else 1.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "h200_measure.sh: no nvcc" >&2; exit 2; }
CUDA_BIN=$(dirname "$NVCC")
export PATH="$CUDA_BIN:$PATH" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/dc" "$OUT/li_bulk/sass" "$OUT/li_bulk/bin" "$OUT/env"
declare -A RC

nvidia-smi --query-gpu=name,uuid,compute_cap,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv \
  >"$OUT/env/gpu.csv" 2>&1
"$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1

LANES='sts32|sts128|stg32|stg128|isetp|lds128'
OUT="$OUT/dc" DC_PASSED_ONLY=1 \
  DC_ONLY="^(s_($LANES)|wgU_($LANES)(_r[0-9]+)?|wgmix_($LANES)_r[0-9]+|wg_s8_n256_a[01])\$" \
  bash "$HERE/../dc/run.sh" sm90 >"$OUT/dc.log" 2>&1
RC[dc]=$?

L="$OUT/li_bulk"
"$NVCC" -O3 -std=c++17 --no-compress -gencode arch=compute_90a,code=sm_90a -Xptxas -v "$HERE/li_bulk.cu" \
  -o "$L/bin/li_bulk" >"$L/build.log" 2>&1
RC[build]=$?
"$NVCC" -O3 -std=c++20 --no-compress -gencode arch=compute_90a,code=sm_90a -Xptxas -v -ldl "$HERE/li_bulk_wg.cu" \
  -o "$L/bin/li_bulk_wg" >"$L/build_wg.log" 2>&1
RC[build_wg]=$?
for b in li_bulk li_bulk_wg; do
  [ -x "$L/bin/$b" ] || continue
  sha256sum "$L/bin/$b" >"$L/bin/$b.sha256"
  "$CUDA_BIN/cuobjdump" -sass "$L/bin/$b" >"$L/sass/$b.sass" 2>&1
  python3 "$HERE/../harness/sass_gate.py" "$L/bin/$b" --json "$L/sass_gate_$b.json" >"$L/sass_gate_$b.txt" 2>&1
  RC[gate_$b]=$?
done
grep -hE "Compiling entry|registers|spill" "$L/build.log" "$L/build_wg.log" >"$L/registers.txt" 2>/dev/null

NCU=$(command -v ncu || echo "$CUDA_BIN/ncu")
[ -x "$NCU" ] || NCU=
echo "ncu=${NCU:-none}" >"$L/ncu_found.txt"
gpu-lease 1 --wait --max-min 15 -- bash -c '
  L=$0 NCU=$1
  nvidia-smi --query-gpu=name,uuid,clocks.sm --format=csv -i "${GPU_LEASE_UUID:-0}" >"$L/leased.csv" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv \
    -lms 200 >"$L/smi.csv" 2>&1 &
  smi=$!
  rc=0
  if [ -x "$L/bin/li_bulk" ]; then "$L/bin/li_bulk" >"$L/device.txt" 2>&1; echo "li_bulk=$?" >>"$L/device_rc.txt"; fi
  if [ -x "$L/bin/li_bulk_wg" ]; then "$L/bin/li_bulk_wg" >"$L/device_wg.txt" 2>&1; echo "li_bulk_wg=$?" >>"$L/device_rc.txt"; fi
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  if [ -n "$NCU" ]; then
    M=lts__t_bytes.sum,lts__t_bytes.sum.per_second,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum
    M=$M,lts__t_sectors_op_red.sum,lts__t_sectors_op_atom.sum,dram__bytes.sum,sm__cycles_elapsed.avg
    M=$M,gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed
    [ -x "$L/bin/li_bulk" ] && "$NCU" --kernel-name regex:lbl --launch-count 4 --metrics "$M" --csv \
      "$L/bin/li_bulk" >"$L/ncu_lbl.csv" 2>"$L/ncu_lbl.err"
    echo "ncu_lbl=$?" >>"$L/device_rc.txt"
    [ -x "$L/bin/li_bulk_wg" ] && "$NCU" --kernel-name regex:lbw --launch-count 2 --metrics "$M" --csv \
      "$L/bin/li_bulk_wg" >"$L/ncu_lbw.csv" 2>"$L/ncu_lbw.err"
    echo "ncu_lbw=$?" >>"$L/device_rc.txt"
  fi
' "$L" "$NCU" >"$L/lease.log" 2>&1
RC[lease]=$?
bitexact=$(grep -c '^all_bitexact=1' "$L/device.txt" 2>/dev/null)
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv >"$OUT/env/gpu_after.csv" 2>&1

{
  printf '{'
  for k in "${!RC[@]}"; do printf '"%s": %s, ' "$k" "${RC[$k]}"; done
  printf '"li_bulk_all_bitexact": %s}\n' "${bitexact:-0}"
} >"$OUT/parts.json"
cat "$OUT/parts.json"
cat "$L/device.txt" "$L/device_wg.txt" 2>/dev/null
[ "${RC[dc]}" = 0 ] && [ "${bitexact:-0}" = 1 ] && exit 0
exit 1
