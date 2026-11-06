#!/usr/bin/env bash
# The edited GEMM on one A100 (sm_80, egemm80.cu): build, gpu_kit-style SASS gate (egemm80.py gate), then under one lease
# the bit-exact gate (every arm on poisoned outputs, judged by egemm80_ref.py), the interleaved bench and each receipted
# arm held 40 s for its steady clock, with nvidia-smi sampling clock and power every 200 ms. Clocks are read, never set.
#
#   research run --on MACHINE --project verity --source /workspace --cwd source \
#     --env GPU_LEASE_WHO=ampere -- bash benchmarks/pouw/edited_gemm/run80.sh
#
# Every output is under $OUT: build.log, sass/, gate-sass.json, gate/ (the dump), gate.jsonl, ref.jsonl, bench.jsonl,
# hold-<arm>.jsonl, smi-samples.csv, summary.json. EG80_NVCC adds nvcc flags (-DEG80_NST=3: a 3-stage ring).
set -uo pipefail
OUT=${OUT:?run under research run}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CUDA=${CUDA_HOME:-/usr/local/cuda-12.4}
M=${EG80_M:-8192} N=${EG80_N:-8192} K=${EG80_K:-4096} HOLD=${EG80_HOLD:-40} FLAGS=${EG80_NVCC:-}
mkdir -p "$OUT/sass" "$OUT/gate"
echo "{\"run_id\":\"${RESEARCH_RUN_ID:-unknown}\",\"host\":\"$(hostname)\",\"start_utc\":\"$(date -u +%FT%TZ)\",\"nvcc\":\"$("$CUDA/bin/nvcc" --version | tail -2 | head -1)\",\"M\":$M,\"N\":$N,\"K\":$K,\"nvcc_flags\":\"$FLAGS\"}" >"$OUT/meta.json"

"$CUDA/bin/nvcc" -O3 -std=c++17 -arch=sm_80 --no-compress -Xptxas -v $FLAGS -o "$OUT/egemm80" "$SRC/egemm80.cu" -ldl \
  >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 2; }
"$CUDA/bin/cuobjdump" -sass "$OUT/egemm80" >"$OUT/sass/egemm80.sass" 2>&1
python3 "$SRC/egemm80.py" gate "$OUT/sass/egemm80.sass" "$OUT/build.log" >"$OUT/gate-sass.json" 2>"$OUT/gate-sass.err"
rc=$?
cat "$OUT/gate-sass.err" >&2
[ $rc -eq 0 ] || exit $rc

cat >"$OUT/gpu.sh" <<EOF
set -uo pipefail
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$GPU_LEASE_UUID" >"$OUT/smi-before.txt" 2>&1
nvidia-smi --id="\$GPU_LEASE_UUID" --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active \
  --format=csv,noheader -lms 200 >"$OUT/smi-samples.csv" 2>&1 &
SMI=\$!
"$OUT/egemm80" gate "$OUT/gate" 1024 256 512 20261017 >"$OUT/gate.jsonl" 2>&1; echo "gate rc=\$?" >>"$OUT/gpu.log"
"$OUT/egemm80" bench 10 $M $N $K >"$OUT/bench.jsonl" 2>&1; echo "bench rc=\$?" >>"$OUT/gpu.log"
for a in e32 e16 e64; do
  "$OUT/egemm80" hold $HOLD \$a $M $N $K >"$OUT/hold-\$a.jsonl" 2>&1; echo "hold \$a rc=\$?" >>"$OUT/gpu.log"
done
kill \$SMI 2>/dev/null
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE --id="\$GPU_LEASE_UUID" >"$OUT/smi-after.txt" 2>&1
EOF
gpu-lease 1 --wait --max-min 20 -- bash "$OUT/gpu.sh"
echo "gpu lease rc=$?" >>"$OUT/gpu.log"
cat "$OUT/gpu.log" "$OUT/gate.jsonl" >&2

python3 "$SRC/egemm80_ref.py" "$OUT/gate" >"$OUT/ref.jsonl" 2>"$OUT/ref.err"
echo "ref rc=$?" >>"$OUT/gpu.log"
tail -3 "$OUT/ref.err" >&2
python3 "$SRC/egemm80.py" summary "$OUT" >"$OUT/summary.json" 2>&1
rc=$?
cat "$OUT/summary.json" >&2
exit $rc
