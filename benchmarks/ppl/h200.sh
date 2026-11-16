#!/usr/bin/env bash
# From a clone, on one H100 or H200 (README.md):  env WORK=/workspace bash benchmarks/ppl/h200.sh
set -euo pipefail
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
cd "$(dirname "$SELF")/../.." || exit 3
TREE=$PWD
OUT=${RESEARCH_RUN_DIR:+$RESEARCH_RUN_DIR/out}
OUT=${OUT:-$TREE/ppl-run/out}
MODEL=llama-3.2-1b SET=c4 CTX=2048 ARM=pearl-c-linears GPU=${GPU-}
export WORK=${WORK:-$HOME/verity-work} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

command -v nvidia-smi > /dev/null || { echo "nvidia-smi isn't on PATH: h200.sh runs on one H100 or H200" >&2; exit 3; }
if [ -z "${GPU_LEASE_UUID:-}" ] && command -v gpu-lease > /dev/null; then
  exec env GPU_LEASE_WHO="${GPU_LEASE_WHO:-ppl-bench}" gpu-lease 1 --wait ${GPU:+--on "$GPU"} \
    --max-min "${LEASE_MIN:-75}" -- bash "$SELF"
fi
[ -z "${GPU_LEASE_UUID:-}" ] || GPU=${GPU_LEASE_UUID%%,*}
card=$(nvidia-smi ${GPU:+-i "$GPU"} --query-gpu=index,uuid,compute_cap --format=csv,noheader 2> /dev/null \
  | while IFS=', ' read -r i uuid compute; do
      [ "$compute" = 9.0 ] || continue
      [ "$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader 2> /dev/null | grep -c . || true)" = 0 ] \
        && { echo "$uuid"; break; }
    done) || card=
[ -n "$card" ] || { echo "no idle H100 or H200 (compute capability 9.0) at GPU=${GPU:-any}: Pearl-C's cubin is sm_90a's;" \
  "wait for the card, or set GPU to an idle one's index (nvidia-smi -L)" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=$card

mkdir -p "$OUT/logs"
if ! (. benchmarks/harness/env.sh > /dev/null 2>&1); then
  [ -z "${VERITY_BENCH_VENV:-}" ] || { echo "VERITY_BENCH_VENV=$VERITY_BENCH_VENV isn't a venv env.sh takes" >&2; exit 3; }
  env VENV="$WORK/venv312" bash adapters/vllm/verity_vllm/ops/pod_bootstrap.sh --cases LLAMA32_1B --gpu \
    --out "$OUT/logs/bootstrap" > "$OUT/logs/bootstrap.log" 2>&1 \
    || { echo "h200.sh: bootstrap failed (its log is $OUT/logs/bootstrap.log)" >&2; exit 1; }
fi
. benchmarks/harness/env.sh
PY=$VENV/bin/python
(
  # shellcheck source=/dev/null
  . benchmarks/harness/setup/pearl_c_sm90.sh && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$OUT/logs/cubin.paths"
) > "$OUT/logs/cubin.log" 2>&1 || { echo "h200.sh: the cubin failed to build (its log is $OUT/logs/cubin.log)" >&2; exit 1; }
{ read -r PEARLC_KERNEL; read -r PEARLC_CUBIN; } < "$OUT/logs/cubin.paths"
export PEARLC_KERNEL PEARLC_CUBIN PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0}

DATA=${PPL_DATA:-$WORK/ppl-data}
"$PY" - "$MODEL" "$DATA" <<'EOF'
import os, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/ppl")
from huggingface_hub import hf_hub_download, snapshot_download
import models, windows
name, data = sys.argv[1], Path(sys.argv[2])
hub, m = Path(os.environ["HF_HOME"]) / "hub", models.get(name)
if not (models.snapshot(name, os.environ["HF_HOME"]) / "config.json").exists():
    snapshot_download(m["repo"], revision=m["revision"], allow_patterns=["*.json", "*.safetensors"], cache_dir=str(hub))
c4 = windows.DATASETS["c4"]
data.mkdir(parents=True, exist_ok=True)
if not (data / c4["file"]).exists():
    got = hf_hub_download("allenai/c4", f"en/{c4['file']}", repo_type="dataset", cache_dir=str(hub))
    (data / c4["file"]).symlink_to(got)
windows.dataset_path("c4", data)
EOF

"$PY" benchmarks/ppl/run.py eval --model "$MODEL" --set "$SET" --ctx "$CTX" --gpus 1 --data "$DATA" --out "$OUT" "$ARM" \
  || { echo "h200.sh: the eval failed or its gates did ($OUT/results.json)" >&2; exit 1; }
"$PY" benchmarks/ppl/run.py metrics "$OUT/results.json" --out "$OUT"
