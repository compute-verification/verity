#!/usr/bin/env bash
# One hash-loop iteration on one GPU (the workload `iterate.sh` launches through `research run --cwd source`): circuits'
# served_commit measurement (benchmarks/served_commit/prof.sh: Qwen3-8B, B32, vLLM compiled, 4 prefill steps of 2,560 tokens
# + 64 decode steps, arms interleaved, under nsys), then collect.py's measures into $RESEARCH_RUN_DIR/result.json.
#
# Env: ROW (the iteration), HYP (its hypothesis), ARMS (default below), REPS (3), WORK, VERITY_BENCH_VENV,
# NATIVE_COLLECT_BUILD, HF_HOME (the node's, as iterate.sh's tables set them), LEASE (gpu-lease's flags, default "--max-min 75"),
# CHECK_TESTS (the committer's GPU byte-identity tests, run in the lease before timing, once in each rows-sha512 leaf form,
# VERITY_ROWS_FORM=fused and split; a failure is exit 7 and no timing; empty skips them; a branch that carries rows-blake3, which
# stays off main, adds check_b3.py and check_hfirst.py).  The timed arms run the device's own form (native_rows.cu,
# verity_rows_leaves_form).
# PIN_BUILD_NAME (iterate.sh sets it on each GPU it has a Build of record for: the H200, and the RTX PRO 6000 once named): the Build sent with the run,
# as $RESEARCH_RUN_DIR/inputs/<name>; prof.sh then runs every arm under its compile artifact (PIN_BUILD) and checks the winners after.
# The loop's arms (arms.py: what-if served-commit-rows-whatif-q, and on such a branch the BLAKE3 sibling served-commit-rows-blake3)
# run through a copy of circuits' prof.sh with their case lines added; prof.sh itself is circuits' and is not edited.
# On a dc VM the GPU is held only for the timed bench; a queued job (vy-nebius-2) already runs inside the queue's lease, which
# it reuses.
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
# served-commit-rows-layer: circuits' per-layer acquisition (each layer's residual, K and V; no scheme's roots) at the committer's
# 4 KiB rows, their -layer-4k frontier (iteration 1's pinned H200 baseline r20261018-104600-f33a), under the loop's committer
export ARMS=${ARMS:-stock,stock-prof,served-commit-rows,served-commit-rows-layer} REPS=${REPS:-3}
PY=${VERITY_BENCH_VENV:?}/bin/python
HF_HOME=${HF_HOME:-/workspace/hf} "$PY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218',
      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.py'], cache_dir='$HF_HOME/hub'))" || exit 8
export CHECK_TESTS=${CHECK_TESTS-integrations/vllm/tests/commit/test_rows_cuda.py integrations/vllm/tests/commit/test_windowed_rows_sha512.py integrations/vllm/tests/commit/test_native_collect.py}
PROF=$(mktemp -d)/prof.sh
sed -e '/^    served-commit-rows) /i\    served-commit-rows-blake3) mods+=(--arm-module hash.loop.arms:SERVED_COMMIT_ROWS_BLAKE3);;' \
    -e '/^    served-commit-rows) /i\    served-commit-rows-whatif-q) mods+=(--arm-module hash.loop.arms:SERVED_COMMIT_ROWS_WHATIF_Q);;' \
    benchmarks/served_commit/prof.sh > "$PROF"
grep -q "hash.loop.arms" "$PROF" || { echo "prof.sh has no served-commit-rows case line to add the loop's arms beside"; exit 9; }
export PROF
if [ -n "${PIN_BUILD_NAME:-}" ]; then
  export PIN_BUILD=${RESEARCH_RUN_DIR:?}/inputs/$PIN_BUILD_NAME
  [ -f "$PIN_BUILD" ] || { echo "PIN_BUILD $PIN_BUILD was not sent with the run"; exit 9; }
fi
bench='
  if [ -n "$CHECK_TESTS" ]; then
    for form in fused split; do
      VERITY_ROWS_FORM=$form OUT="$OUT/check-$form" TESTS="$CHECK_TESTS" bash benchmarks/served_commit/gpu_tests.sh > "$OUT/check-$form.log" 2>&1
      tail -3 "$OUT/check-$form.log"
      grep -q "^pytest rc 0$" "$OUT/check-$form.log" || { echo "byte-identity tests failed in the $form form ($OUT/check-$form.log): not timed"; exit 7; }
    done
  fi
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.before.csv"
  NSYS=1 NSYS_EACH=1 SWEEP=b32 SPEC=qwen3-8b-bf16-h200-served bash "$PROF" 2>&1 | tee "$OUT/prof.out"
  rc=${PIPESTATUS[0]}
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
rm -rf "$OUT/inductor/triton" "$OUT/vllm_cache"
pin_rc=$(sed -n 's/^pin check rc //p' "$OUT/prof.out" 2>/dev/null | tail -1)
python3 benchmarks/hash/loop/collect.py "$OUT" --reps "$REPS" --row "${ROW:-}" --hypothesis "${HYP:-}" --bench-rc "$rc" \
  ${pin_rc:+--pin-rc "$pin_rc"}
exit $(( rc ? rc : $? ))
