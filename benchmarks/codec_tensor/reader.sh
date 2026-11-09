#!/usr/bin/env bash
# The device reader (reader.py) on one GPU (cwd = the source tree). Everything runs under one gpu-lease 1 --wait, which
# also waits out node 2's booked windows: build reader.cu, gate it byte for byte against the PoUB driver
# (small, under compute-sanitizer, then at --gate-blocks), then time one pass. Results go to $RESEARCH_RUN_DIR/out.
#
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=ra-codec-tensor taskset -c 48-95 bash benchmarks/codec_tensor/reader.sh
#
# ENV_DIR holds the shared torch venv (as step.sh's); GIB sizes the timed pass (default 8); SANITIZE=0 skips the
# sanitizer's gate.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
export PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
B=$(mktemp -d /tmp/reader.XXXXXX); trap 'rm -rf "$B"' EXIT
export B CT=benchmarks/codec_tensor GIB=${GIB:-8} SANITIZE=${SANITIZE:-1}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-20}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -o "$B/reader.so" \
    $CT/reader.cu > "$O/reader.ptxas" 2>&1 || { tail -40 "$O/reader.ptxas"; exit 4; }
  sha256sum "$B/reader.so" > "$O/libs.sha256"
  if [ "$SANITIZE" = 1 ]; then
    env -u RESEARCH_RUN_ID compute-sanitizer --tool memcheck --error-exitcode 9 --print-limit 20 \
      "$PY" $CT/reader.py --lib "$B/reader.so" --out "$O/smoke" --gate-blocks 256 --gib 0 > "$O/smoke.log" 2>&1 \
      || { tail -60 "$O/smoke.log"; exit 9; }
  fi
  "$PY" $CT/reader.py --lib "$B/reader.so" --out "$O" --gib "$GIB" 2>&1 | tee "$O/reader.log"
  exit "${PIPESTATUS[0]}"'
