#!/usr/bin/env bash
# This directory's GPU steps on one leased sm_90a GPU: build.sh's ship made first, in this run (nothing else is compiled),
# then each STEP (a script and its arguments in one word, e.g. "gate.py --snapshot DIR") under its own gpu-lease of at
# most S8_MAX_MIN (20) minutes, writing into $OUT/<i>-<script>.  @OUT@ in a step is $OUT, @CUBIN@ the ship's cubin, and a
# step "cpu:SCRIPT ..." runs with no lease.  A step that fails ends the run (S8_KEEP_GOING=1: the rest still run).
# VERITY_BENCH_VENV names the Python with torch.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
PY=${VERITY_BENCH_VENV:+$VERITY_BENCH_VENV/bin/python3}
for v in /workspace/cache/kernels-j10r/venv312 /workspace/cp/venv312 /workspace/jobs/venv312; do
  [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3
done
[ -x "${PY:-}" ] || { echo "s8_orbit: no Python with torch (VERITY_BENCH_VENV)" >&2; exit 2; }
bash "$HERE/build.sh" "$OUT/ship" > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log" >&2; exit 2; }
CUBIN=$OUT/ship/s8_orbit.cubin
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit --format=csv > "$OUT/gpus.txt" 2>&1
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$OUT/python.txt" 2>&1
MAX_MIN=${S8_MAX_MIN:-20}
rc_all=0; i=0
for step in "$@"; do
  s=${step//@OUT@/$OUT}; s=${s//@CUBIN@/$CUBIN}; lease=1; [[ $s == cpu:* ]] && { lease=0; s=${s#cpu:}; }
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
