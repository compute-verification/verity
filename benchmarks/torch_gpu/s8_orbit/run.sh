#!/usr/bin/env bash
# This directory's GPU steps on one leased sm_90a GPU: build.sh's ship made first, in this run, and s8_linear's
# (wgmma.linear, the baseline) beside it; then each STEP (a script and its arguments in one word, e.g. "gate.py --snapshot
# DIR") under its own gpu-lease of at most S8_MAX_MIN (20) minutes, writing into $OUT/<i>-<script>, or every STEP under
# one lease with S8_ONE_LEASE=1.  @OUT@ in a step is $OUT, @CUBIN@ the orbit's cubin and @WGMMA@ wgmma.linear's, and a step
# "cpu:SCRIPT ..." runs with no lease.  A step that fails ends the run (S8_KEEP_GOING=1: the rest still run).
# VERITY_BENCH_VENV names the Python with torch; CUDA the toolkit (default /usr/local/cuda-13.0), on the path for the
# SHA-512 engine's build.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
PY=${VERITY_BENCH_VENV:+$VERITY_BENCH_VENV/bin/python3}
for v in /workspace/cache/kernels-j10r/venv312 /workspace/cp/venv312 /workspace/jobs/venv312; do
  [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3
done
[ -x "${PY:-}" ] || { echo "s8_orbit: no Python with torch (VERITY_BENCH_VENV)" >&2; exit 2; }
CUDA=${CUDA:-/usr/local/cuda-13.0}
export PATH=$(dirname "$PY"):$CUDA/bin:$PATH PYTHONPATH=$ROOT:$ROOT/tools/config
CUBIN=$OUT/ship/s8_orbit.cubin WGMMA=$OUT/s8ship/s8_wgmma.cubin MAX_MIN=${S8_MAX_MIN:-20}
if [ -z "${S8_IN_LEASE:-}" ]; then
  git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
  bash "$HERE/build.sh" "$OUT/ship" > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log" >&2; exit 2; }
  PY=/nonexistent bash "$ROOT/benchmarks/torch_gpu/s8_linear/build.sh" "$OUT/s8ship" > "$OUT/s8build.log" 2>&1 \
    || { tail -30 "$OUT/s8build.log" >&2; exit 2; }
  case " $* " in *--sha*)
    "$PY" -c "import sys; sys.path.insert(0, '$ROOT/benchmarks/torch_gpu/sha512_rows'); import engine; engine.device_lib('90a')" \
      > "$OUT/shabuild.log" 2>&1 || { tail -30 "$OUT/shabuild.log" >&2; exit 2; };;
  esac
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit --format=csv > "$OUT/gpus.txt" 2>&1
  "$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$OUT/python.txt" 2>&1
  if [ "${S8_ONE_LEASE:-0}" = 1 ]; then
    echo "s8_orbit: asking for one GPU for every step at $(date -u +%H:%M:%SZ)"
    gpu-lease 1 --wait --max-min "$MAX_MIN" -- env S8_IN_LEASE=1 bash "$HERE/run.sh" "$@"
    exit $?
  fi
else
  [ -n "${GPU_LEASE_UUID:-}" ] && nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$OUT/clocks.txt" 2>&1
fi
rc_all=0; i=0
for step in "$@"; do
  s=${step//@OUT@/$OUT}; s=${s//@CUBIN@/$CUBIN}; s=${s//@WGMMA@/$WGMMA}; lease=1; [[ $s == cpu:* ]] && { lease=0; s=${s#cpu:}; }
  [ -n "${S8_IN_LEASE:-}" ] && lease=0
  i=$((i + 1)); read -r -a argv <<< "$s"; script=${argv[0]}; dir=$OUT/$i-${script%.py}; mkdir -p "$dir"
  echo "s8_orbit: step $i ($step) at $(date -u +%H:%M:%SZ)"
  if [ "$lease" = 0 ]; then
    "$PY" "$HERE/$script" --out "$dir" "${argv[@]:1}" 2>&1 | tee "$dir/log.txt"
    rc=${PIPESTATUS[0]}
  else
    # shellcheck disable=SC2016
    gpu-lease 1 --wait --max-min "$MAX_MIN" -- bash -c '
      [ -n "${GPU_LEASE_UUID:-}" ] && nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$0/clocks.txt" 2>&1
      exec timeout $((60 * $1 - 30)) "$2" "$3" --out "$0" "${@:4}"' "$dir" "$MAX_MIN" "$PY" "$HERE/$script" \
      "${argv[@]:1}" 2>&1 | tee "$dir/log.txt"
    rc=${PIPESTATUS[0]}
  fi
  echo "s8_orbit: step $i ($step) exited $rc at $(date -u +%H:%M:%SZ)" | tee -a "$OUT/rc.txt"
  [ "$rc" = 0 ] || { rc_all=$rc; [ "${S8_KEEP_GOING:-0}" = 1 ] || break; }
done
exit "$rc_all"
