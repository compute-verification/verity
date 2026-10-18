#!/usr/bin/env bash
# One hash-loop iteration on one GPU (the workload `iterate.sh` launches through `research run --cwd source`): circuits'
# served_commit measurement (benchmarks/served_commit/prof.sh: Qwen3-8B, B32, vLLM compiled, 4 prefill steps of 2,560 tokens
# + 64 decode steps, arms interleaved, under nsys), then collect.py's measures into $RESEARCH_RUN_DIR/result.json.
#
# Env: ROW (the iteration), HYP (its hypothesis), ARMS (default below), REPS (3), WORK, VERITY_BENCH_VENV,
# NATIVE_COLLECT_BUILD, HF_HOME (the node's, as iterate.sh's MACHINES table sets them), LEASE ("--max-min 75" plus the node's
# flags, e.g. "--mem-gb 64 --preemptible"). The GPU is held only for the timed bench: the model check and collect.py run outside.
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
export ARMS=${ARMS:-stock,stock-prof,served-commit-rows} REPS=${REPS:-3}
PY=${VERITY_BENCH_VENV:?}/bin/python
HF_HOME=${HF_HOME:-/workspace/hf} "$PY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218',
      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.py'], cache_dir='$HF_HOME/hub'))" || exit 8
read -r -a lease <<< "${LEASE:---max-min 75}"
GPU_LEASE_WHO=${GPU_LEASE_WHO:-hash-loop} gpu-lease 1 --wait "${lease[@]}" -- bash -c '
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.before.csv"
  NSYS=1 NSYS_EACH=1 SWEEP=b32 SPEC=qwen3-8b-bf16-h200-served bash benchmarks/served_commit/prof.sh
  rc=$?
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.after.csv"
  exit $rc'
rc=$?
echo "lease rc $rc"
python3 benchmarks/hash/loop/collect.py "$OUT" --reps "$REPS" --row "${ROW:-}" --hypothesis "${HYP:-}" --bench-rc "$rc"
exit $(( rc ? rc : $? ))
