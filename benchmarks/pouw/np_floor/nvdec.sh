#!/usr/bin/env bash
# nvdec.py on the H200: the video decode engines' output rate through PyNvVideoCodec (the driver's libnvcuvid), with the
# engines' utilization sampled (nvidia-smi dmon's dec column) to show the decode ran on them. PyNvVideoCodec and ffmpeg
# (imageio-ffmpeg's, with libx264 and libx265) come from a scratch venv; the streams are encoded before the lease.
#   research run --on vy-dc-h200-5 --project verity --campaign pouw-night --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=n-floor -- bash benchmarks/pouw/np_floor/nvdec.sh [W H FRAMES R K PROCS...]
# Writes $OUT/setup.log, nvdec.json, dmon.txt and smi.csv.
set -euo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
uv venv -q "$WORK/v" >>"$OUT/setup.log" 2>&1
uv pip install -q --python "$WORK/v/bin/python" PyNvVideoCodec imageio-ffmpeg numpy >>"$OUT/setup.log" 2>&1
"$WORK/v/bin/python" "$HERE/nvdec.py" make "$WORK" "${@:1:3}" >>"$OUT/setup.log" 2>&1
ls -l "$WORK"/*.mp4 >>"$OUT/setup.log"
gpu-lease 1 --wait --max-min 15 -- bash -c '
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=timestamp,clocks.sm,clocks.video,power.draw --format=csv -lms 500 >"$0/smi.csv" &
  p=$!
  nvidia-smi dmon -i "$GPU_LEASE_UUID" -s u -o T >"$0/dmon.txt" 2>&1 &
  q=$!
  "$1" "$2" "$3" "${@:4}" | tee "$0/nvdec.json"
  kill $p $q' "$OUT" "$WORK/v/bin/python" "$HERE/nvdec.py" "$WORK" "$@"
