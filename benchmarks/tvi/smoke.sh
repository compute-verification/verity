# The environment's smoke test: GRPO (vLLM colocate, then HF generate) and SFT on Qwen2.5-0.5B-Instruct, GSM8K, one GPU.
# Launched under `gpu-lease 1` with --cwd source; writes $OUT/<test>/metrics.json, $OUT/<test>.log, $OUT/gpu_mem.csv.
set -u
V=${TVI_PY:-/workspace/jobs/tvi/venv/bin/python}
S=benchmarks/tvi
M=${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
CK=/workspace/jobs/tvi/smoke/$RESEARCH_RUN_ID
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} host=$(hostname)"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,memory.used,memory.total --format=csv
cp "$($V -c 'import trl,os;print(os.path.dirname(trl.__file__))')/generation/vllm_generation.py" "$OUT/" || true
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 1000 > "$OUT/gpu_mem.csv" &
POLL=$!
trap 'kill $POLL 2>/dev/null' EXIT

run() {
  local name=$1; shift
  echo "== $name $(date -u +%FT%TZ)" | tee -a "$OUT/phases.txt"
  local t=$(date +%s)
  timeout "${SMOKE_TIMEOUT:-1200}" "$V" "$@" --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc wall=$(( $(date +%s) - t ))s end=$(date -u +%FT%TZ)" | tee -a "$OUT/phases.txt"
  tail -25 "$OUT/$name.log"
}

TESTS=${SMOKE_TESTS:-grpo_vllm_lora grpo_vllm_full grpo_hf_lora grpo_hf_tinylora sft_lora sft_full sft_tinylora}
for t in $TESTS; do
  case $t in
    grpo_vllm_lora)   run $t $S/grpo_train.py --model $M --steps 20 --adapter lora --lora-r 16 --gen vllm --save-dir $CK/$t --save-merged ;;
    grpo_vllm_full)   run $t $S/grpo_train.py --model $M --steps 20 --adapter full --gen vllm ;;
    grpo_vllm_tinylora) run $t $S/grpo_train.py --model $M --steps 20 --adapter tinylora --gen vllm --save-dir $CK/$t ;;
    grpo_hf_lora)     run $t $S/grpo_train.py --model $M --steps 10 --adapter lora --lora-r 16 --gen hf ;;
    grpo_hf_tinylora) run $t $S/grpo_train.py --model $M --steps 5 --adapter tinylora --gen hf --save-dir $CK/$t ;;
    sft_lora)         run $t $S/sft_train.py --model $M --steps 20 --adapter lora --lora-r 16 --save-dir $CK/$t ;;
    sft_full)         run $t $S/sft_train.py --model $M --steps 20 --adapter full ;;
    sft_tinylora)     run $t $S/sft_train.py --model $M --steps 10 --adapter tinylora ;;
  esac
done
kill $POLL 2>/dev/null
"$V" - "$OUT" <<'EOF'
import csv, json, os, sys
from datetime import datetime, timezone
out = sys.argv[1]
mem = []
for row in csv.reader(open(os.path.join(out, "gpu_mem.csv"))):
    try:
        ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        mem.append((ts, float(row[1])))
    except Exception:
        pass
phases, cur = [], None
for line in open(os.path.join(out, "phases.txt")):
    if line.startswith("== "):
        name, start = line[3:].split()
        cur = [name, datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()]
    elif cur and line.startswith(cur[0] + " rc="):
        parts = dict(p.split("=") for p in line.split()[1:])
        end = datetime.strptime(parts["end"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        peak = max((m for ts, m in mem if cur[1] - 1 <= ts <= end + 1), default=None)
        doc = {"test": cur[0], "rc": int(parts["rc"]), "wall": parts["wall"], "nvsmi_peak_mib": peak}
        mp = os.path.join(out, cur[0], "metrics.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            doc.update({k: m.get(k) for k in ["steps", "steps_per_s", "train_runtime_s", "setup_s", "trainable_params",
                        "reward_first_quarter", "reward_last_quarter", "loss_first", "loss_last"]})
            doc["torch_max_reserved_gib"] = m.get("gpu", {}).get("max_reserved_gib")
        phases.append(doc)
json.dump(phases, open(os.path.join(out, "summary.json"), "w"), indent=1)
for p in phases:
    print(json.dumps(p))
EOF
