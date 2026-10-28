#!/usr/bin/env bash
# This directory's scripts on one leased GPU of an H200 (or another sm_90 GPU), each STEP under its own lease of at most
# S8_MAX_MIN (30) minutes, from build.sh's ship (nothing is compiled here; cwd = the source tree's root):
#   tar -C SHIP -cf /tmp/s8-ship.tar .
#   OUT=DIR S8_SHIP=/tmp/s8-ship.tar bash benchmarks/torch_gpu/s8_linear/run.sh STEP...
# Needs CUDA torch with vLLM and `gpu-lease` on PATH; OUT and S8_SHIP default to a run directory's out/ and
# inputs/s8-ship.tar (RESEARCH_RUN_DIR).
# A STEP is a script and its arguments in one word, e.g. "run.py --ms 1,7,257 --no-swap" (the canary) then "run.py" (the
# gate); each writes into $OUT/<i>-<script>. @OUT@ in a step is $OUT ("bench.py --configs @OUT@/1-tune/best.json"), @S@
# the run's scratch directory (deleted at exit, never uploaded), and a step "cpu:SCRIPT ..." runs with no lease ("cpu:realw.py
# --gen --weights @S@/realw" then "realw.py --weights @S@/realw"). A step that fails ends the run (S8_KEEP_GOING=1: the rest
# still run).
# Outside any lease: the ship unpacked and held to its MANIFEST.sha256 (s8_wgmma.cubin, its sass.json; row_table/ seeds
# ~/.cache/verity/row_table), the card and Python recorded. VERITY_BENCH_VENV names the Python with torch and vLLM.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?set OUT}/out}; mkdir -p "$OUT"
SHIP=${S8_SHIP:-${RESEARCH_RUN_DIR:?set S8_SHIP}/inputs/s8-ship.tar}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
PY=${VERITY_BENCH_VENV:+$VERITY_BENCH_VENV/bin/python3}
for v in /workspace/cp/venv312 /workspace/jobs/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
[ -x "${PY:-}" ] || { echo "s8_linear: no Python with torch (VERITY_BENCH_VENV)" >&2; exit 2; }
CUDA=${CUDA:-/usr/local/cuda-13.0}
export PATH=$(dirname "$PY"):$CUDA/bin:$PATH
export PYTHONPATH=$ROOT/integrations/vllm:$ROOT:$ROOT/tools/config
export TRITON_CACHE_DIR=$S/triton
mkdir -p "$S/ship" && tar -C "$S/ship" -xf "$SHIP" && (cd "$S/ship" && sha256sum -c --quiet MANIFEST.sha256) \
  || { echo "s8_linear: the ship $SHIP is missing or fails its MANIFEST" >&2; exit 2; }
cp "$S/ship/source.txt" "$S/ship/toolchain.txt" "$S/ship/sass.json" "$S/ship/MANIFEST.sha256" "$OUT/"
export S8_WGMMA_CUBIN=$S/ship/s8_wgmma.cubin
if [ -d "$S/ship/row_table" ]; then mkdir -p "$HOME/.cache/verity/row_table" && cp -rn "$S/ship/row_table/." "$HOME/.cache/verity/row_table/"; fi
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.max.sm,power.limit --format=csv > "$OUT/gpus.txt" 2>&1
"$PY" -c 'import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)' > "$OUT/python.txt" 2>&1
MAX_MIN=${S8_MAX_MIN:-30}
rc_all=0; i=0
for step in "$@"; do
  s=${step//@OUT@/$OUT}; s=${s//@S@/$S}; lease=1; [[ $s == cpu:* ]] && { lease=0; s=${s#cpu:}; }
  i=$((i + 1)); read -r -a argv <<< "$s"; script=${argv[0]}; dir=$OUT/$i-${script%.py}; mkdir -p "$dir"
  if [ "$lease" = 0 ]; then
    echo "s8_linear: step $i ($step) with no GPU at $(date -u +%H:%M:%SZ)"
    "$PY" "$ROOT/benchmarks/torch_gpu/s8_linear/$script" --out "$dir" "${argv[@]:1}" 2>&1 | tee "$dir/log.txt"
    rc=${PIPESTATUS[0]}
    echo "s8_linear: step $i ($step) exited $rc at $(date -u +%H:%M:%SZ)" | tee -a "$OUT/rc.txt"
    [ "$rc" = 0 ] || { rc_all=$rc; [ "${S8_KEEP_GOING:-0}" = 1 ] || break; }
    continue
  fi
  echo "s8_linear: step $i ($step) asking for one GPU at $(date -u +%H:%M:%SZ)"
  nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 \
    > "$dir/smi.csv" 2>/dev/null &
  smi=$!
  # shellcheck disable=SC2016
  gpu-lease 1 --wait --max-min "$MAX_MIN" -- bash -c '
    [ -n "${GPU_LEASE_UUID:-}" ] && nvidia-smi -i "$GPU_LEASE_UUID" -q -d CLOCK,PERFORMANCE,POWER > "$0/clocks.txt" 2>&1
    exec timeout $((60 * $1 - 30)) "$2" "$3" --out "$0" "${@:4}"' "$dir" "$MAX_MIN" "$PY" "$ROOT/benchmarks/torch_gpu/s8_linear/$script" \
    "${argv[@]:1}" 2>&1 | tee "$dir/log.txt"
  rc=${PIPESTATUS[0]}
  kill "$smi" 2>/dev/null
  echo "s8_linear: step $i ($step) exited $rc at $(date -u +%H:%M:%SZ)" | tee -a "$OUT/rc.txt"
  [ "$rc" = 0 ] || { rc_all=$rc; [ "${S8_KEEP_GOING:-0}" = 1 ] || break; }
done
echo "s8_linear: last GPU step ended at $(date -u +%H:%M:%SZ)"
exit "$rc_all"
