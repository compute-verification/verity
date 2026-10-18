#!/usr/bin/env bash
# One hash-loop iteration on one GPU (the workload `iterate.sh` launches through `research run --cwd source`): circuits'
# served_commit measurement (benchmarks/served_commit/prof.sh: Qwen3-8B, B32, vLLM compiled, 4 prefill steps of 2,560 tokens
# + 64 decode steps, arms interleaved, under nsys), then collect.py's measures into $RESEARCH_RUN_DIR/result.json.
#
# Env: ROW (the iteration), HYP (its hypothesis), ARMS (default below), REPS (3), WORK, VERITY_BENCH_VENV,
# NATIVE_COLLECT_BUILD, HF_HOME (the node's, as iterate.sh's tables set them), LEASE (gpu-lease's flags, default "--max-min 75"),
# CHECK_TESTS (the committer's GPU byte-identity tests, run in the lease before timing; a failure is exit 7 and no timing; empty
# skips them).
# On a dc VM the GPU is held only for the timed bench; a queued job (vy-nebius-2) already runs inside the queue's lease, which
# it reuses.
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
export ARMS=${ARMS:-stock,stock-prof,served-commit-rows} REPS=${REPS:-3}
PY=${VERITY_BENCH_VENV:?}/bin/python
HF_HOME=${HF_HOME:-/workspace/hf} "$PY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218',
      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.py'], cache_dir='$HF_HOME/hub'))" || exit 8
export CHECK_TESTS=${CHECK_TESTS-integrations/vllm/tests/commit/test_rows_cuda.py integrations/vllm/tests/commit/test_windowed_rows_sha512.py}
bench='
  if [ -n "$CHECK_TESTS" ]; then
    OUT="$OUT/check" TESTS="$CHECK_TESTS" bash benchmarks/served_commit/gpu_tests.sh > "$OUT/check.log" 2>&1
    tail -3 "$OUT/check.log"
    grep -q "^pytest rc 0$" "$OUT/check.log" || { echo "byte-identity tests failed ($OUT/check.log): not timed"; exit 7; }
  fi
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.before.csv"
  NSYS=1 NSYS_EACH=1 SWEEP=b32 SPEC=qwen3-8b-bf16-h200-served bash benchmarks/served_commit/prof.sh
  rc=$?
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.after.csv"
  exit $rc'
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  bash -c "$bench"
else
  read -r -a lease <<< "${LEASE:---max-min 75}"
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-hash-loop} gpu-lease 1 --wait "${lease[@]}" -- bash -c "$bench"
fi
rc=$?
echo "lease rc $rc"
python3 benchmarks/hash/loop/collect.py "$OUT" --reps "$REPS" --row "${ROW:-}" --hypothesis "${HYP:-}" --bench-rc "$rc"
exit $(( rc ? rc : $? ))
