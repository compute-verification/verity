#!/usr/bin/env bash
# A3 (new skill) on the GPUs gpu-lease gave this run: optional base pass@k, then (task, S, seed) jobs claimed by NW workers
# per GPU from a pool shared with every other run that names the same COORD directory.
# Env: TASKS="A:0:SFT:RL:0/1/2,..." SPECS="full,lora16,..." SKIP="C:xs:13,..." NW BASE=1 CKPT=dir|scratch|'' COORD=dir
set -euo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
S=$(mktemp -d "${SCRATCH_ROOT:-/tmp}/a3.XXXXXX")
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
cp "$IN"/newskill.py "$IN"/aggregate.py "$S"/
cd "$S"

PY=
for c in /workspace/jobs/tvi/venv/bin/python /workspace/jobs/venv312/bin/python /workspace/jobs/tvi/venv-A2/bin/python \
         /workspace/jobs/tvi/venv-r1/bin/python; do
  if [ -x "$c" ] && "$c" -c 'import torch, transformers, numpy; assert torch.cuda.is_available()' 2>/dev/null; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
  uv venv -q -p 3.12 "$S/venv"
  uv pip install -q --python "$S/venv/bin/python" torch --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python "$S/venv/bin/python" transformers numpy safetensors
  PY="$S/venv/bin/python"
fi
echo "PY=$PY"
"$PY" -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else export HF_HOME="$S/hf"; fi
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nproc=$(nproc) host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
df -h /workspace | tail -1

"$PY" - <<'EOF'
import json, os
tasks = [t.split(":") for t in os.environ.get("TASKS", "A:0:1000:200:0/1/2,B:1:1000:200:0/1/2").split(",") if t]
specs = os.environ.get("SPECS", "full,lora16,xs:48552,hl:50000,xs:5000,hl:5000,xs:625,hl:625,xs:13,hl:13").split(",")
skip = set(os.environ.get("SKIP", "").split(","))
lr_from = dict(kv.split("=") for kv in os.environ.get("LR_FROM", "").split(",") if kv)  # task -> task whose seed 0 sets the LR
jid = lambda name, spec, seed: f"{name}-{spec.replace(':', '')}-s{seed}"
jobs = []
for seed in sorted({int(s) for t in tasks for s in t[4].split("/")}):
    for name, order, sft, rl, seeds in tasks:
        if str(seed) not in seeds.split("/"):
            continue
        for spec in specs:
            if f"{name}:{spec}" in skip:
                continue
            j = {"id": jid(name, spec, seed), "task": name, "order": int(order), "spec": spec, "seed": seed,
                 "sft_steps": int(sft), "rl_steps": int(rl)}
            if name in lr_from:
                j["lr_from"] = jid(lr_from[name], spec, 0)
            elif seed > 0 and "0" in seeds.split("/"):
                j["lr_from"] = jid(name, spec, 0)
            jobs.append(j)
json.dump(jobs, open("jobs.json", "w"), indent=1)
print(len(jobs), "jobs")
EOF
cp jobs.json "$OUT/jobs.json"

CKPT="${CKPT-/workspace/jobs/tvi/ckpt/A3}"
if [ "$CKPT" = scratch ]; then CKPT="$S/ckpt"; fi
COORD="${COORD:-$S/coord}"
mkdir -p "$COORD/results" "$COORD/claims"
if [ -n "$CKPT" ] && [ "$CKPT" != "$S/ckpt" ]; then
  mkdir -p "$CKPT"
  for p in /workspace/jobs/tvi/ckpt/A3 /workspace/jobs/tvi/a3; do
    [ -d "$p" ] && { research keep "$p" --owner @training --expires 2026-10-27 --priority normal \
      --why 'training-vs-inference program' || echo "research keep $p failed (rc $?)"; }
  done
  cp newskill.py "$CKPT/newskill.py"
fi

IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
pids=()
if [ "${BASE:-0}" = 1 ]; then
  ( CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PY" newskill.py base --out "$OUT/base_pass64.json" > "$OUT/base.log" 2>&1 \
    && cp "$OUT/base_pass64.json" "$COORD/base_pass64.json" ) &
  pids+=($!)
fi
rank=0
for g in "${GPUS[@]}"; do
  for _ in $(seq 1 "${NW:-3}"); do
    CUDA_VISIBLE_DEVICES="$g" "$PY" newskill.py worker --jobs jobs.json --results "$COORD/results" --claims "$COORD/claims" \
      --copy-to "$OUT/results" --tag "$(basename "$RESEARCH_RUN_DIR")" --ckpt "$CKPT" --rank "$rank" ${MODEL:+--model "$MODEL"} \
      > "$OUT/worker$rank.log" 2>&1 &
    pids+=($!)
    rank=$((rank + 1))
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "workers done rc=$rc"
"$PY" aggregate.py --results "$COORD/results" --base "$COORD/base_pass64.json" --out "$OUT" --node "${NODE_NAME:-vy-nebius-3}" || true
ls -la "$OUT" | head -40
df -h /workspace | tail -1
