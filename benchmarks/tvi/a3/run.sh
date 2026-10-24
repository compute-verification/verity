#!/usr/bin/env bash
# A3 (new skill) on the GPUs gpu-lease gave this run: base pass@k, then (task, S, seed) jobs claimed by NW workers per GPU.
# Env: TASKS="A:0:SFT:RL,B:1:SFT:RL" SPECS="full,lora16,..." SEEDS="0,1,2" NW (workers per GPU) BASE=1 CKPT=dir|'' EXTRA_JOBS=json
set -euo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
S=$(mktemp -d "${SCRATCH_ROOT:-/tmp}/a3.XXXXXX")
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
cp "$IN"/newskill.py "$IN"/aggregate.py "$S"/
cd "$S"

PY=/workspace/jobs/venv312/bin/python
if [ -f /workspace/jobs/tvi/venv/READY ] && [ -x /workspace/jobs/tvi/venv/bin/python ]; then PY=/workspace/jobs/tvi/venv/bin/python; fi
"$PY" -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else export HF_HOME="$S/hf"; fi
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nproc=$(nproc) host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
df -h /workspace | tail -1

"$PY" - <<'EOF'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]:
    print(snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"]))
EOF

"$PY" - <<'EOF'
import json, os
tasks = [t.split(":") for t in os.environ.get("TASKS", "A:0:600:200,B:1:600:200").split(",") if t]
specs = os.environ.get("SPECS", "full,lora16,xs:48552,xs:5000,xs:625,xs:13").split(",")
seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
lr_one = os.environ.get("LR_ONE")
jobs = []
for seed in seeds:
    for name, order, sft, rl in tasks:
        for spec in specs:
            j = {"id": f"{name}-{spec.replace(':', '')}-s{seed}", "task": name, "order": int(order), "spec": spec,
                 "seed": seed, "sft_steps": int(sft), "rl_steps": int(rl)}
            if lr_one:
                j["lrs"] = [float(lr_one)] if lr_one != "mid" else None
            jobs.append(j)
for j in json.loads(os.environ.get("EXTRA_JOBS", "[]")):
    jobs.append(j)
if lr_one == "mid":
    import newskill
    for j in jobs:
        if j.get("lrs") is None:
            j["lrs"] = [newskill.LR_GRID[j["spec"]][1]]
json.dump(jobs, open("jobs.json", "w"), indent=1)
print(len(jobs), "jobs")
EOF
cp jobs.json "$OUT/jobs.json"

CKPT="${CKPT-/workspace/jobs/tvi/ckpt/A3}"
if [ "$CKPT" = scratch ]; then CKPT="$S/ckpt"; fi
if [ -n "$CKPT" ] && [ "$CKPT" != "$S/ckpt" ]; then
  mkdir -p "$CKPT"
  research keep /workspace/jobs/tvi/ckpt/A3 --owner @training --expires 2026-10-27 --priority normal \
    --why 'training-vs-inference program' || echo "research keep failed (rc $?)"
fi

IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
pids=()
if [ "${BASE:-1}" = 1 ]; then
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PY" newskill.py base --out "$OUT/base_pass64.json" > "$OUT/base.log" 2>&1 &
  pids+=($!)
fi
rank=0
for g in "${GPUS[@]}"; do
  for _ in $(seq 1 "${NW:-2}"); do
    CUDA_VISIBLE_DEVICES="$g" "$PY" newskill.py worker --jobs jobs.json --results "$OUT/results" --claims "$S/claims" \
      --ckpt "$CKPT" --rank "$rank" ${MODEL:+--model "$MODEL"} > "$OUT/worker$rank.log" 2>&1 &
    pids+=($!)
    rank=$((rank + 1))
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "workers done rc=$rc"
"$PY" aggregate.py --results "$OUT/results" --base "$OUT/base_pass64.json" --out "$OUT" --node "${NODE_NAME:-vy-nebius-3}" || true
ls -la "$OUT" "$OUT/results" | head -80
df -h /workspace | tail -1
