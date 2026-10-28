#!/usr/bin/env bash
# This directory's GPU job on one leased sm_90a GPU: build.sh's ship made first, in this run and off the lease; Llama 3.1
# 8B's layer-0 int8 weights fetched into scratch (weights.py; SW_REAL=0 skips them, and a failed fetch only drops the
# real-weight gate); then bench.py with this script's arguments under one gpu-lease of at most SW_MAX_MIN (20) minutes,
# writing into $OUT/bench. VERITY_BENCH_VENV names the Python with torch (else the node's first known venv, else one made
# under /workspace/cache/kernels-sw-orbit); CUDA the toolkit (build.sh's default); SW_BUILD_ONLY=1 stops after the build,
# taking no lease.
#   research run --on vy-dc-h200-3 --project verity --campaign hash --source . --cwd source --timeout 60m \
#     --env GPU_LEASE_WHO=NAME -- bash benchmarks/torch_gpu/sw_orbit/run.sh [bench.py flags]
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
git -C "$HERE" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
PY=${VERITY_BENCH_VENV:+$VERITY_BENCH_VENV/bin/python3}
for v in /workspace/cache/kernels-j10r/venv312 /workspace/cp/venv312 /workspace/jobs/venv312; do
  [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3
done
if [ ! -x "${PY:-}" ]; then
  V=/workspace/cache/kernels-sw-orbit/venv312
  if [ ! -x "$V/bin/python3" ]; then
    echo "sw_orbit: making $V at $(date -u +%H:%M:%SZ)"
    { uv venv -q --python python3.12 "$V" && uv pip install -q --python "$V/bin/python3" numpy \
        torch --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple; } \
      > "$OUT/venv.log" 2>&1 || { tail -20 "$OUT/venv.log" >&2; rm -rf "$V"; exit 2; }
    command -v research > /dev/null && research keep "$(dirname "$V")" --owner @kernels --expires 2026-11-30 \
      --priority low --why "sw_orbit's torch venv" >> "$OUT/venv.log" 2>&1
  fi
  PY=$V/bin/python3
fi
echo "sw_orbit: build at $(date -u +%H:%M:%SZ)"
bash "$HERE/build.sh" "$OUT/ship" > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log" >&2; exit 2; }
if [ "${SW_BUILD_ONLY:-0}" = 1 ]; then grep -iE "warning|error" "$OUT/build.log" | head -40; echo "sw_orbit: built"; exit 0; fi
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$OUT/python.txt" 2>&1
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
REAL=()
if [ "${SW_REAL:-1}" = 1 ]; then
  echo "sw_orbit: weights at $(date -u +%H:%M:%SZ)"
  if "$PY" "$HERE/weights.py" --dir "$W" --layers 0 > "$OUT/weights.log" 2>&1; then REAL=(--real "$W"); else
    echo "sw_orbit: no 8B weights (weights.log): no real-weight gate"; fi
fi
MAX_MIN=${SW_MAX_MIN:-20}
echo "sw_orbit: bench at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "$MAX_MIN" -- bash -c '
  nvidia-smi -i "${GPU_LEASE_UUID:-0}" --query-gpu=index,uuid,name,driver_version,clocks.max.sm --format=csv > "$0/gpus.txt" 2>&1
  exec timeout $((60 * $1 - 30)) "$2" "$3/bench.py" --out "$0/bench" --lib "$0/ship/liborbit.so" --cubin "$0/ship/s8_wgmma.cubin" "${@:4}"' \
  "$OUT" "$MAX_MIN" "$PY" "$HERE" "${REAL[@]}" "$@" 2>&1 | tee "$OUT/bench.log"
rc=${PIPESTATUS[0]}
echo "sw_orbit: bench exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
