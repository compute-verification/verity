#!/usr/bin/env bash
# A2u beside a saturating FFMA stream on sm_120 (a2u_sat.cu): each kernel's loop counted (a2u_sat_count.py: uniform,
# vector, FFMA, spills), then timed on the device; a2u.json sets each kernel's issue rate beside its count. From a
# pushed commit:
#
#   research run --queue --preemptible --on <node> --gpus 1 --mem-gb 8 --disk-gb 1 --max-min 25 --project verity \
#     --campaign eps1 --source <tree> --cwd source --env QUEUED=1 -- bash benchmarks/pouw/theory/a2u_sat.sh
#
# Exit: 0 done, 1 build failed, 2 no nvcc, 3 a CUDA error on the device.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1

if [ "${A2U_LEASED:-}" != 1 ]; then
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/bin"
  NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
  [ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "a2u_sat.sh: no nvcc (CUDA 13.0)" >&2; exit 2; }
  "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
  BIN="$OUT/bin/a2u_sat"
  "$NVCC" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v "$HERE/a2u_sat.cu" -o "$BIN" \
    >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
  sha256sum "$BIN" >"$OUT/bin/a2u_sat.sha256"
  "$(dirname "$NVCC")/cuobjdump" -sass "$BIN" >"$OUT/sass/a2u_sat.sass" 2>&1
  python3 "$HERE/a2u_sat_count.py" "$OUT/sass/a2u_sat.sass" >"$OUT/sass/counts.json"
  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 20 --)
  # under `research run --queue --gpus 1` (QUEUED=1) the queue leases the GPU around the whole job
  [ "${QUEUED:-0}" = 1 ] && lease=()
  "${lease[@]}" env A2U_LEASED=1 BIN="$BIN" bash "${BASH_SOURCE[0]}"
  rc=$?
  python3 "$HERE/a2u_sat_count.py" "$OUT/sass/a2u_sat.sass" --device "$OUT/device.txt" >"$OUT/a2u.json"
  exit $rc
fi

nvidia-smi --query-gpu=name,uuid,compute_cap,clocks.sm,clocks.max.sm --format=csv >"$OUT/env/gpu.csv" 2>&1
"$BIN" | tee "$OUT/device.txt"
exit "${PIPESTATUS[0]}"
