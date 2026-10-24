#!/usr/bin/env bash
# One leased GPU, a queue of GRPO runs (crux 1, training-vs-inference program). QUEUE="task:method:seed:steps[:extra args] ...",
# extra args comma-separated (e.g. gsm8k:base:0:0:--passk,64). Checkpoints go to $CKPT_ROOT, results and logs to $OUT.
set -uo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
S=$(mktemp -d "${SCRATCH_ROOT:-/tmp}/tvi-a2.XXXXXX")
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
cp "$IN"/grpo.py "$S"/
mkdir -p "$S/data"
cp "$IN"/*.jsonl "$S/data/"
cd "$S"

PY=/workspace/jobs/venv312/bin/python
if [ "${PY_PREF:-auto}" != venv312 ] && [ -f /workspace/jobs/tvi/venv/READY ] \
   && /workspace/jobs/tvi/venv/bin/python -c 'import vllm, transformers, torch' 2>/dev/null; then
  PY=/workspace/jobs/tvi/venv/bin/python
fi
echo "python: $PY"
if ! "$PY" -c 'import math_verify' 2>/dev/null; then
  uv pip install -q --python "$PY" --target "$S/extra" --no-deps math-verify latex2sympy2_extended antlr4-python3-runtime==4.13.2 \
    && export PYTHONPATH="$S/extra"
fi
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else export HF_HOME="$S/hf"; fi
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export MODEL DATA="$S/data"
mkdir -p /workspace/jobs/tvi/cache/A2
export SVD_CACHE="/workspace/jobs/tvi/cache/A2/svd16_${MODEL//\//_}.pt"
CKPT_ROOT="${CKPT_ROOT:-/workspace/jobs/tvi/ckpt/A2}"
mkdir -p "$CKPT_ROOT"
if command -v research >/dev/null; then
  research keep /workspace/jobs/tvi/ckpt/A2 --owner @training --expires 2026-10-27 --priority normal --why 'training-vs-inference program' || true
  research keep /workspace/jobs/tvi/cache/A2 --owner @training --expires 2026-10-27 --priority normal --why 'training-vs-inference program' || true
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"$PY" -c 'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__, torch.cuda.get_device_name(0))'

for item in $QUEUE; do
  IFS=: read -r task method seed steps extra <<< "$item"
  # shellcheck disable=SC2086
  "$PY" grpo.py --task "$task" --method "$method" --seed "$seed" --steps "$steps" --model "$MODEL" --data "$S/data" \
    --out "$OUT" --ckpt-root "$CKPT_ROOT" ${extra//,/ } ${EXTRA:-} > "$OUT/log-$task-$method-seed$seed.txt" 2>&1
  echo "$item rc=$?"
  tail -3 "$OUT/log-$task-$method-seed$seed.txt"
done

NODE="${NODE_NAME:-$(hostname)}" "$PY" - <<'EOF'
import glob, json, os
out = os.environ["OUT"]
rows = []
for p in sorted(glob.glob(f"{out}/result-*.json")):
    r = json.load(open(p))
    rows.append({"node": os.environ.get("NODE", ""), "run_id": os.environ.get("RESEARCH_RUN_ID", ""), "task": r["task"],
                 "method": r["method"], "seed": r.get("seed", 0), "S_bits": r["S_bits"], "n_params": r.get("n_params", 0),
                 "steps": r["steps"], "score": r["acc"], "ckpts": r.get("ckpts", {}), "model": r.get("model")})
json.dump(rows, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(rows, indent=1))
EOF
