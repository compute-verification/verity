#!/usr/bin/env bash
# One served_commit sweep with the measurement arms of prof_arm.py (ARMS, default stock + the NVTX-ranged noretain arm), under nsys
# when NSYS=1 (then nsys_summary.py over the last profiled run of each arm, or over every run with NSYS_EACH=1). Measurement only.
set -uo pipefail
. benchmarks/harness/env.sh
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
ARMS=${ARMS:-stock,served-commit-prof}
mods=()
for a in ${ARMS//,/ }; do
  case "$a" in
    served-commit-prof) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_PROF);;
    stock-prof) mods+=(--arm-module served_commit.prof_arm:STOCK_PROF);;
    served-commit-switch) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_SWITCH);;
    served-commit-nativedrain) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_NATIVEDRAIN);;
    served-commit-rows) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS);;
    served-commit-hiprio) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_HIPRIO);;
    served-commit-rows-hiprio) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS_HIPRIO);;
    served-commit-fprof) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_FPROF);;
    served-commit-nomemo) mods+=(--arm-module served_commit.prof_arm:SERVED_COMMIT_NOMEMO);;
    served-commit) mods+=(--arm-module served_commit.arm:SERVED_COMMIT);;
    served-commit-noretain) mods+=(--arm-module served_commit.arm:SERVED_COMMIT_NORETAIN);;
  esac
done
bench=("$VENV/bin/python" -m verity_bench vllm --preset "${SPEC:-llama32-1b-bf16-h200-served}" --out "$OUT/bench" --arms "$ARMS" "${mods[@]}"
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
cat "$OUT/bench/table.md" 2>/dev/null
cat "$OUT/roots.jsonl" 2>/dev/null
