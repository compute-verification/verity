#!/usr/bin/env bash
# kernels-maxtput on one GPU (the workload launch.sh sends through `research run --cwd source`): maxtput.py's engines, stock and
# the Commit's arms on one saturated load, beside an nvidia-smi log of the GPU, then row.py's result into $RESEARCH_RUN_DIR/result.json.
#
# Env: MODEL (qwen3-8b | llama31-8b), ARMS (stock first), REQUESTS, CYCLES, REPS, MAXTPUT_ARGS (more driver flags), WORK,
# VERITY_BENCH_VENV, NATIVE_COLLECT_BUILD, HF_HOME (the node's), LEASE (gpu-lease's flags on a dc VM, default "--max-min 120"),
# CHECK_TESTS (the committer's GPU byte-identity tests, run in the lease before timing in both rows-sha512 leaf forms, as the loop's
# job.sh; a failure is exit 7 and no timing; empty skips them).
# PIN_BUILD_NAME: the Build of record sent with the run ($RESEARCH_RUN_DIR/inputs/<name>): every engine compiles into one fresh
# Inductor dir seeded with its autotune winners (`served-pin seed`, as served_commit's prof.sh), and the dir is checked against
# them after (`served-pin check` -> out/pin_check.txt, rc 3 when a winner differs). Unset: a fresh Inductor dir in scratch.
# On a dc VM the GPU is held with gpu-lease; a queued job (vy-nebius-2) already runs inside the queue's lease.
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
. benchmarks/harness/env.sh
export VENV MODEL=${MODEL:-qwen3-8b} ARMS=${ARMS:-stock,sha512,sha512-layer}
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
"$VENV/bin/python" -c "
import sys
sys.path.insert(0, 'benchmarks')
from hash.maxtput.maxtput import MODELS
from huggingface_hub import snapshot_download
name, rev = MODELS['$MODEL']
print(snapshot_download(name, revision=rev, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.py'],
                        cache_dir='$HF_HOME/hub'))" || exit 8
export CHECK_TESTS=${CHECK_TESTS-integrations/vllm/tests/commit/test_rows_cuda.py integrations/vllm/tests/commit/test_windowed_rows_sha512.py integrations/vllm/tests/commit/test_native_collect.py}
pin() { "$VENV/bin/python" -m verity_vllm.pipeline.cli served-pin "$1" --build "$PIN_BUILD" --inductor-dir "$OUT/inductor"; }
if [ -n "${PIN_BUILD_NAME:-}" ]; then
  export PIN_BUILD=${RESEARCH_RUN_DIR:?}/inputs/$PIN_BUILD_NAME
  [ -f "$PIN_BUILD" ] || { echo "PIN_BUILD $PIN_BUILD was not sent with the run"; exit 9; }
  rm -rf "$OUT/inductor" "$OUT/vllm_cache"
  pin seed || exit 3
  export TORCHINDUCTOR_CACHE_DIR="$OUT/inductor" VLLM_CACHE_ROOT="$OUT/vllm_cache"
else
  export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" VLLM_CACHE_ROOT="$SCRATCH/vllm_cache"
fi
export VERITY_INDUCTOR_DIR_PER_PROCESS=1 VLLM_DISABLE_COMPILE_CACHE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
bench='
  if [ -n "$CHECK_TESTS" ] && [ "$ARMS" != stock ]; then
    for form in fused split; do
      VERITY_ROWS_FORM=$form OUT="$OUT/check-$form" TESTS="$CHECK_TESTS" bash benchmarks/served_commit/gpu_tests.sh > "$OUT/check-$form.log" 2>&1
      tail -3 "$OUT/check-$form.log"
      grep -q "^pytest rc 0$" "$OUT/check-$form.log" || { echo "byte-identity tests failed in the $form form ($OUT/check-$form.log): not timed"; exit 7; }
    done
  fi
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,memory.total,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.before.csv"
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,memory.used \
    --format=csv,noheader,nounits -lms 200 > "$OUT/gpu.csv" 2>/dev/null &
  smi=$!
  "$VENV/bin/python" -m hash.maxtput.maxtput --model "$MODEL" --arms "$ARMS" --requests "${REQUESTS:-2048}" \
    --cycles "${CYCLES:-2}" --reps "${REPS:-1}" --gpu-log "$OUT/gpu.csv" --out "$OUT" ${MAXTPUT_ARGS:-} 2>&1 | tee "$OUT/maxtput.log"
  rc=${PIPESTATUS[0]}
  kill $smi 2>/dev/null
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.after.csv"
  exit $rc'
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  bash -c "$bench"
else
  read -r -a lease <<< "${LEASE:---max-min 120}"
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-kernels-maxtput} gpu-lease 1 --wait "${lease[@]}" -- bash -c "$bench"
fi
rc=$?
echo "bench rc $rc"
pin_rc=
if [ -n "${PIN_BUILD:-}" ]; then
  pin check > "$OUT/pin_check.txt" 2>&1
  pin_rc=$?
  echo "pin check rc $pin_rc"; cat "$OUT/pin_check.txt"
fi
rm -rf "$OUT/inductor/triton" "$OUT/vllm_cache"
python3 benchmarks/hash/maxtput/row.py "$OUT" --bench-rc "$rc" ${pin_rc:+--pin-rc "$pin_rc"} --to "${RESEARCH_RUN_DIR:-.}/result.json"
exit $(( rc ? rc : pin_rc ? pin_rc : $? ))
