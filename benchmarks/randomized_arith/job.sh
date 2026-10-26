#!/usr/bin/env bash
# `honest.py` on one GPU of vy-dc-h200 under its own `gpu-lease 1 --wait` (`research run --on vy-dc-h200 --cwd source`,
# the two pinned ppl sets sent with it): a smoke pass on SmolLM2-135M that checks the harness's exactness, then Llama-3.1-8B (the census model)
# and Qwen3-8B. Each model's results are `$OUT/<model>/results.jsonl`; MMLU and GSM8K are fetched into the run's scratch
# and their sha256 recorded there.
#
#   RA_MODELS  which of smoke,llama,qwen to run (default all three)
#   RA_GEN     the arms GSM8K runs on Llama (default below)
#   RA_LLAMA   more `honest.py` flags for Llama, after the defaults (the last of a repeated flag wins)
#   RA_MAX_MIN the lease's bound in minutes (default 300)
set -uo pipefail
if [[ -z ${GPU_LEASE_UUID:-} ]]; then
  exec gpu-lease 1 --wait --max-min "${RA_MAX_MIN:-300}" -- bash "$0" "$@"
fi
OUT=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$OUT"
PY=/workspace/cp/venv312/bin/python
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
DATA=$(mktemp -d "${TMPDIR:-/tmp}/ra-data.XXXX") || exit 1
trap 'rm -rf "$DATA"' EXIT
trap 'exit 143' TERM INT
cp "$RESEARCH_RUN_DIR"/inputs/* "$DATA"/
snap() { ls -d "$HF_HOME"/hub/models--"$1"/snapshots/*/ | head -1; }
MODELS=${RA_MODELS:-smoke,llama,qwen}
GEN=${RA_GEN:-bf16,rtn,sr,dither,noise1,acc6,acc8,acc10,had,sr+acc6}
nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,memory.used --format=csv > "$OUT/gpu.txt" 2>&1
rc=0
run() {
  local name=$1; shift
  mkdir -p "$OUT/$name"
  "$PY" benchmarks/randomized_arith/honest.py --data "$DATA" --out "$OUT/$name" "$@" \
    > "$OUT/$name/stdout.txt" 2> "$OUT/$name/stderr.txt"
  local r=$?
  echo "$name $r" | tee -a "$OUT/status.txt"
  tail -3 "$OUT/$name/stderr.txt"
  return $r
}
if [[ $MODELS == *smoke* ]]; then
  # 2 is a task's data missing, not the harness: the models still run their other evals
  run smoke --snapshot "$(snap HuggingFaceTB--SmolLM2-135M)" --smoke --seeds 1
  r=$?
  [[ $r == 0 || $r == 2 ]] || exit 1
fi
if [[ $MODELS == *llama* ]]; then
  # shellcheck disable=SC2086  # RA_LLAMA is a list of flags
  run llama-3.1-8b --snapshot "$(snap unsloth--Llama-3.1-8B)" --family llama3 --seeds 3 --gen-arms "$GEN" \
    ${RA_LLAMA:-} || rc=1
fi
if [[ $MODELS == *qwen* ]]; then
  run qwen3-8b --snapshot "$(snap Qwen--Qwen3-8B)" --seeds 1 --evals wikitext2,mmlu,gsm8k \
    --arms bf16,rtn,sr,dither,noise1,acc6,acc8,acc10,had,sr+acc6 --gen-arms bf16,rtn,sr,acc8,had,sr+acc6 || rc=1
fi
exit $rc
