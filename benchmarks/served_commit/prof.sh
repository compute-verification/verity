#!/usr/bin/env bash
# One served_commit sweep of the spec SPEC with the measurement arms of prof_arm.py (ARMS, default stock + the rows-sha512 arm),
# under nsys when NSYS=1 (then nsys_summary.py over the last profiled run of each arm, or over every run with NSYS_EACH=1). Measurement only.
# PIN_BUILD=<served-match's build.json>: every arm's process compiles into one fresh Inductor dir seeded with the Build of record's
# autotune winners (`served-pin seed`, as ops/served_commit.sh), compile caches off, and the dir is checked against them after
# (`served-pin check` -> $OUT/pin_check.txt, rc 3 when a winner differs: then the kernels are not the Build's and the number counts for nothing).
set -uo pipefail
. benchmarks/harness/env.sh
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
pin() { "$VENV/bin/python" -m verity_vllm.pipeline.cli served-pin "$1" --build "$PIN_BUILD" --inductor-dir "$OUT/inductor"; }
if [ -n "${PIN_BUILD:-}" ]; then
  rm -rf "$OUT/inductor" "$OUT/vllm_cache"
  pin seed || exit 3
  export TORCHINDUCTOR_CACHE_DIR="$OUT/inductor" VLLM_CACHE_ROOT="$OUT/vllm_cache" VERITY_INDUCTOR_DIR_PER_PROCESS=1 \
    VLLM_DISABLE_COMPILE_CACHE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
fi
ARMS=${ARMS:-stock,served-commit-rows}
mods=()
for a in ${ARMS//,/ }; do
  case "$a" in
    stock-prof) mods+=(--arm-module served_commit.prof_arm:STOCK_PROF);;
    served-commit-rows) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS);;
    served-commit-rows-base) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_BASE);;
    served-commit-rows-batch) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_BATCH);;
    served-commit-rows-ring8) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_RING8);;
    served-commit-rows-w64) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_W64);;
    served-commit-rows-cap) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_CAP);;
    served-commit-rows-ring8-cap) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_RING8_CAP);;
    served-commit-rows-4k) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_4K);;
    served-commit-rows-layer) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_LAYER);;
    served-commit-rows-layer-4k) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_LAYER_4K);;
    served-commit-rows-4k-cap) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_4K_CAP);;
    served-commit-rows-layer-cap) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_LAYER_CAP);;
    served-commit-rows-hiprio) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_HIPRIO);;
  esac
done
bench=("$VENV/bin/python" -m verity_bench vllm --preset "${SPEC:?SPEC names a served spec, e.g. llama32-1b-bf16-rtx-served-arm}" --out "$OUT/bench" --arms "$ARMS" "${mods[@]}"
       --sweep "${SWEEP:-b32}" --reps "${REPS:-1}")
if [ "${NSYS:-1}" = 1 ]; then
  nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node --trace-fork-before-exec=true -f true -o "$OUT/served" \
    "${bench[@]}" > "$OUT/bench.log" 2>&1
  echo "bench rc $?"
  nsys export --type sqlite -f true -o "$OUT/served.sqlite" "$OUT/served.nsys-rep" > "$OUT/export.log" 2>&1
  echo "export rc $?"
  for a in ${ARMS//,/ }; do
    [ "$a" = stock ] && continue
    python3 benchmarks/served_commit/nsys_summary.py "$OUT/served.sqlite" --run "run:$a" ${NSYS_EACH:+--each} --json "$OUT/summary.$a.json" \
      | tee "$OUT/summary.$a.txt"
  done
  rm -f "$OUT/served.sqlite"
else
  "${bench[@]}" > "$OUT/bench.log" 2>&1
  echo "bench rc $?"
fi
if [ -n "${PIN_BUILD:-}" ]; then
  pin check > "$OUT/pin_check.txt" 2>&1
  echo "pin check rc $?"; cat "$OUT/pin_check.txt"
fi
cat "$OUT/bench/table.md" 2>/dev/null
cat "$OUT/roots.jsonl" 2>/dev/null
