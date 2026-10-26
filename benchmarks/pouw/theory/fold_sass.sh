#!/usr/bin/env bash
# FOLD-MIN's device check of the served receipt folds (fold_sass.cu): each fold's SASS on sm_90a, sm_100a and sm_120a,
# counted by census class a word (fold_sass_count.py), then on the device: every kernel bit-exact against the host's
# reference, its chain latency and its full-SM rate. From a pushed commit:
#
#   research run --on <node> --project verity --campaign eps1 --source <tree> --cwd source --env GPU_LEASE_WHO=<id> \
#     [--send fold_sass --env FOLD_SASS_BIN=inputs/fold_sass] -- bash benchmarks/pouw/theory/fold_sass.sh
#
# Without FOLD_SASS_BIN it builds the fat binary (CUDA 13.0) into $OUT/bin and dumps its SASS into $OUT/sass; with it
# (a path relative to the run dir: --send puts the file in inputs/) it runs that binary and builds nothing, so a node
# that takes no CPU phase (the H200s) only runs it. The device part runs under one gpu-lease (device.txt, gpu.csv),
# and fold_sass_count.py --device sets the rates beside the counts (fold.json).
# Exit: 0 done, 1 build failed, 2 no nvcc or no binary, 3 a kernel not bit-exact.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1

if [ "${FOLD_LEASED:-}" != 1 ]; then
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/bin"
  BIN=${FOLD_SASS_BIN:-}
  if [ -n "$BIN" ]; then
    case "$BIN" in /*) ;; *) BIN="$RESEARCH_RUN_DIR/$BIN" ;; esac
    [ -x "$BIN" ] || chmod +x "$BIN" 2>/dev/null
    [ -x "$BIN" ] || { echo "fold_sass.sh: no binary at $BIN" >&2; exit 2; }
    sha256sum "$BIN" >"$OUT/bin/sent.sha256"
  else
    NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
    [ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "fold_sass.sh: no nvcc (CUDA 13.0)" >&2; exit 2; }
    "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
    BIN="$OUT/bin/fold_sass"
    "$NVCC" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -gencode arch=compute_100a,code=sm_100a \
      -gencode arch=compute_120a,code=sm_120a -Xptxas -v "$HERE/fold_sass.cu" -o "$BIN" >"$OUT/build.log" 2>&1 \
      || { cat "$OUT/build.log" >&2; exit 1; }
    sha256sum "$BIN" >"$OUT/bin/fold_sass.sha256"
    "$(dirname "$NVCC")/cuobjdump" -sass "$BIN" >"$OUT/sass/fold_sass.sass" 2>&1
    python3 "$HERE/fold_sass_count.py" "$OUT/sass/fold_sass.sass" >"$OUT/sass/counts.json"
  fi
  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 30 --)
  [ ${#lease[@]} -eq 0 ] && echo "fold_sass.sh: no gpu-lease on this node; running on its GPU directly" >&2
  "${lease[@]}" env FOLD_LEASED=1 BIN="$BIN" bash "${BASH_SOURCE[0]}"
  rc=$?
  sass="$OUT/sass/fold_sass.sass"
  [ -f "$sass" ] || sass=$(ls "$RESEARCH_RUN_DIR"/inputs/*.sass 2>/dev/null | head -1)
  [ -n "$sass" ] && [ -f "$sass" ] && python3 "$HERE/fold_sass_count.py" "$sass" --device "$OUT/device.txt" >"$OUT/fold.json"
  exit $rc
fi

# ---- leased ------------------------------------------------------------------------------------------------------------
nvidia-smi --query-gpu=name,uuid,compute_cap,clocks.sm,clocks.max.sm --format=csv >"$OUT/env/gpu.csv" 2>&1
"$BIN" | tee "$OUT/device.txt"
exit "${PIPESTATUS[0]}"
