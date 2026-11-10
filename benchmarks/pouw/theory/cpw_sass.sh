#!/usr/bin/env bash
# The sm_120 unit with its CP-W checkpoint (cpw_sass.cu): each checkpoint route's SASS on sm_120a, counted a set
# (cpw_sass_count.py), then on the device: every route bit-exact against the host's reference, and its rate at 1, 2,
# 4 warps a SMSP and full occupancy. From a pushed commit:
#
#   research run --queue --on <node> --gpus 1 --mem-gb 8 --max-min 25 --project verity --campaign eps1 --source <tree> \
#     --cwd source --env QUEUED=1 -- bash benchmarks/pouw/theory/cpw_sass.sh
#
# It builds the binary (CUDA 13.0) into $OUT/bin and dumps its SASS into $OUT/sass; the device part runs under one
# gpu-lease (device.txt, env/gpu.csv), and cpw_sass_count.py --device sets the rates beside the counts (cpw.json).
# Exit: 0 done, 1 build failed, 2 no nvcc, 3 a route not bit-exact.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1

if [ "${CPW_LEASED:-}" != 1 ]; then
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/bin"
  NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
  [ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "cpw_sass.sh: no nvcc (CUDA 13.0)" >&2; exit 2; }
  "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
  BIN="$OUT/bin/cpw_sass"
  flags=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v)
  # st256 (mode 7) only where ptxas takes st.global.v8.b32; build_st256.log keeps a refusal
  if "$NVCC" "${flags[@]}" -DCPW_ST256 "$HERE/cpw_sass.cu" -o "$BIN" >"$OUT/build_st256.log" 2>&1; then
    mv "$OUT/build_st256.log" "$OUT/build.log"
  else
    echo "cpw_sass.sh: st.global.v8.b32 refused on sm_120a (build_st256.log); building without mode 7" >&2
    "$NVCC" "${flags[@]}" "$HERE/cpw_sass.cu" -o "$BIN" >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
  fi
  sha256sum "$BIN" >"$OUT/bin/cpw_sass.sha256"
  "$(dirname "$NVCC")/cuobjdump" -sass "$BIN" >"$OUT/sass/cpw_sass.sass" 2>&1
  python3 "$HERE/cpw_sass_count.py" "$OUT/sass/cpw_sass.sass" >"$OUT/sass/counts.json"
  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 20 --)
  # under `research run --queue --gpus 1` (QUEUED=1) the queue leases the GPU around the whole job
  [ "${QUEUED:-0}" = 1 ] && lease=()
  [ ${#lease[@]} -eq 0 ] && echo "cpw_sass.sh: no gpu-lease (or queued); running on the GPU it was given" >&2
  "${lease[@]}" env CPW_LEASED=1 BIN="$BIN" bash "${BASH_SOURCE[0]}"
  rc=$?
  python3 "$HERE/cpw_sass_count.py" "$OUT/sass/cpw_sass.sass" --device "$OUT/device.txt" >"$OUT/cpw.json"
  exit $rc
fi

# ---- leased ------------------------------------------------------------------------------------------------------------
nvidia-smi --query-gpu=name,uuid,compute_cap,clocks.sm,clocks.max.sm --format=csv >"$OUT/env/gpu.csv" 2>&1
"$BIN" | tee "$OUT/device.txt"
exit "${PIPESTATUS[0]}"
