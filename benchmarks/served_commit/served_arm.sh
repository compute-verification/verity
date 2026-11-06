#!/usr/bin/env bash
# The served arm (served_arm.py, the specs *-rtx-served-arm*): SPEC through run.sh (ARGS go to it), compiling into an Inductor dir
# of this job's own. With SEED_ARGS (run.sh's arguments for a pass of the spec's bi-eager and compiled stock: `--arms` names every
# arm the `--arm-module`s define, so it takes an eager arm's module, which compiles nothing), that pass runs first, one rep, so
# every kernel compiled stock launches has its autotune winner in the dir before an arm of the measured run compiles: compiled
# stock and the compiled Commit then load the same winners (a reduction's tile size reaches the numerics), as prof.sh's
# PIN_BUILD gives them from a Build of record. $OUT/inductor_winners.txt counts the dir's winners after each pass, and
# $OUT/<pass>_winners.txt lists them: a winner the measured run added is a kernel compiled stock never launched.
#
# With PIN_BUILD (a Build of record's build.json, `served-match`'s; PIN_BUILD_NAME names one sent with the run, as
# $RESEARCH_RUN_DIR/inputs/<name>) in place of SEED_ARGS, the dir is seeded with that Build's autotune winners (`served-pin
# seed`, as prof.sh), compile caches off, and checked against them after the run (`served-pin check` -> $OUT/pin_check.txt):
# by curve.py's rules a compiled ratio counts only when the winners are the Build's.
#
# From the source tree's root on one RTX PRO 6000 (VERITY_BENCH_VENV names the pinned-vLLM environment, as run.sh's):
#   SPEC=llama31-8b-fp8-rtx-served-arm SEED_ARGS="--arms stock-fp8,stock-fp8@compiled,commit-rows-4k-eager \
#     --arm-module served_commit.served_arm:EAGER_ROWS_4K_FP8" bash benchmarks/served_commit/served_arm.sh
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
[ -z "${PIN_BUILD_NAME:-}" ] || PIN_BUILD=${RESEARCH_RUN_DIR:?}/inputs/$PIN_BUILD_NAME
if [ -n "${PIN_BUILD:-}" ]; then
  [ -z "${SEED_ARGS:-}" ] || { echo "served_arm.sh: PIN_BUILD seeds the Inductor dir from its Build; drop SEED_ARGS" >&2; exit 2; }
  . benchmarks/harness/env.sh
  pin() { "$VENV/bin/python" -m verity_vllm.pipeline.cli served-pin "$1" --build "$PIN_BUILD" --inductor-dir "$d/inductor"; }
  pin seed
  winners seed
  export VLLM_DISABLE_COMPILE_CACHE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
elif [ -n "${SEED_ARGS:-}" ]; then
  read -ra seed <<< "$SEED_ARGS"
  OUT=$OUT/seed bash benchmarks/served_commit/run.sh "${seed[@]}" --reps 1
  winners seed
fi
rc=0
bash benchmarks/served_commit/run.sh "$@" || rc=$?
winners measured
if [ -n "${PIN_BUILD:-}" ]; then
  pin check > "$OUT/pin_check.txt" 2>&1 || echo "served_arm.sh: a winner is not the Build's (pin_check.txt)" >&2
  cat "$OUT/pin_check.txt"
fi
exit "$rc"
