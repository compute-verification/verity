#!/usr/bin/env bash
# Lane captures (lane_capture.cu, check.py): build on a CPU node, capture on the H200 under gpu-lease.
#
#   research run --on auto --wait -- bash benchmarks/pouw/dc/lane_capture/run.sh build
#       builds lane_capture for sm_90a and writes $OUT/cap.tgz (the binary and its cuobjdump -sass)
#   research run --on vy-dc-h200-5 --project verity --source <tree> --cwd source --send cap.tgz \
#     --env GPU_LEASE_WHO=NAME -- bash benchmarks/pouw/dc/lane_capture/run.sh capture [N]
#       runs the binary on N input triples (default 65,536) into $OUT/capture.bin and checks it (check.py) into
#       $OUT/verdict.json; exit 1 when a form's output differs from its semantics
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "${1:?usage: run.sh build | capture [N]}" in
build)
  NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
  [ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "run.sh: no nvcc" >&2; exit 2; }
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  "$NVCC" -std=c++17 -O3 -gencode arch=compute_90a,code=sm_90a -o "$WORK/lane_capture" "$HERE/lane_capture.cu" || exit 1
  "$(dirname "$NVCC")/cuobjdump" -sass "$WORK/lane_capture" >"$WORK/lane_capture.sass" || exit 1
  tar -C "$WORK" -czf "$OUT/cap.tgz" lane_capture lane_capture.sass
  ;;
capture)
  IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
  [ -f "$IN/cap.tgz" ] || { echo "run.sh: no inputs/cap.tgz (pass the build's with --send)" >&2; exit 2; }
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  tar -C "$WORK" -xzf "$IN/cap.tgz"
  cp "$WORK/lane_capture.sass" "$OUT/"
  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 10 --)
  "${lease[@]}" "$WORK/lane_capture" "$OUT/capture.bin" "${2:-65536}" | tee "$OUT/capture.txt" || exit 1
  python3 "$HERE/check.py" "$OUT/capture.bin" --sass "$OUT/lane_capture.sass" --json "$OUT/verdict.json" \
    | tee "$OUT/check.txt"
  exit "${PIPESTATUS[0]}"
  ;;
esac
