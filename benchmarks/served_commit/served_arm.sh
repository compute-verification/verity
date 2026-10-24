#!/usr/bin/env bash
# G2's served arm (served_arm.py, the specs *-rtx-served-arm*): SPEC through run.sh (ARGS go to it), compiling into an Inductor dir
# of this job's own. With SEED_ARGS (run.sh's arguments for a pass of the spec's bi-eager and compiled stock: `--arms` names every
# arm the `--arm-module`s define, so it takes an eager arm's module, which compiles nothing), that pass runs first, one rep, so
# every kernel compiled stock launches has its autotune winner in the dir before an arm of the measured run compiles: compiled
# stock and the compiled Commit then load the same winners (a reduction's block size reaches the numerics), as prof.sh's
# PIN_BUILD gives them from a Build of record. $OUT/inductor_winners.txt counts the dir's winners after each pass, and
# $OUT/<pass>_winners.txt lists them: a winner the measured run added is a kernel compiled stock never launched.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --timeout 1h \
#     --env VERITY_BENCH_VENV=/workspace/jobs/venv312 --env SPEC=llama31-8b-fp8-rtx-served-arm \
#     --env SEED_ARGS="--arms stock-fp8,stock-fp8@compiled,commit-rows-4k-eager \
#       --arm-module served_commit.served_arm:EAGER_ROWS_4K_FP8" -- \
#     /workspace/pouw/infra/bin/gpu-lease 1 --wait --max-min 60 --mem-gb 64 -- bash benchmarks/served_commit/served_arm.sh
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
d=$(mktemp -d /tmp/served-arm.XXXXXX)
trap 'rm -rf "$d"' EXIT
mkdir -p "$OUT" "$d/inductor"
export NATIVE_COLLECT_BUILD=$d/nc_build TORCHINDUCTOR_CACHE_DIR=$d/inductor VLLM_CACHE_ROOT=$d/vllm_cache \
  VERITY_INDUCTOR_DIR_PER_PROCESS=1
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
winners() {
  (cd "$d/inductor" && find . -name '*.best_config' | sort) > "$OUT/$1_winners.txt"
  echo "$1 $(wc -l < "$OUT/$1_winners.txt")" >> "$OUT/inductor_winners.txt"
}
if [ -n "${SEED_ARGS:-}" ]; then
  read -ra seed <<< "$SEED_ARGS"
  OUT=$OUT/seed bash benchmarks/served_commit/run.sh "${seed[@]}" --reps 1
  winners seed
fi
rc=0
bash benchmarks/served_commit/run.sh "$@" || rc=$?
winners measured
exit "$rc"
