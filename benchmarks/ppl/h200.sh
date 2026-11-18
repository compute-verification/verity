#!/usr/bin/env bash
# From a clone, on one H100 or H200 (README.md):  env WORK=/workspace bash benchmarks/ppl/h200.sh
set -euo pipefail
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
cd "$(dirname "$SELF")/../.." || exit 3
TREE=$PWD
OUT=${RESEARCH_RUN_DIR:+$RESEARCH_RUN_DIR/out}
OUT=${PPL_OUT:-${OUT:-$TREE/ppl-run/out}}
MODEL=llama-3.2-1b SET=${SET:-c4} CTX=2048 ARMS=${ARMS:-pearl-c-linears} GPU=${GPU-}
export WORK=${WORK:-$HOME/verity-work} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

command -v nvidia-smi > /dev/null || { echo "nvidia-smi isn't on PATH: h200.sh runs on one H100 or H200" >&2; exit 3; }
if [ -z "${GPU_LEASE_UUID:-}" ] && command -v gpu-lease > /dev/null; then
  exec env GPU_LEASE_WHO="${GPU_LEASE_WHO:-ppl-bench}" gpu-lease 1 --wait ${GPU:+--on "$GPU"} ${CORES:+--cores "$CORES"} \
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
ARROW=
if [ "$SET" = wikitext2 ] && ! "$PY" -c 'import pyarrow' 2> /dev/null; then
  ARROW=$WORK/ppl-pyarrow
  [ -d "$ARROW/pyarrow" ] || uv pip install --python "$PY" --target "$ARROW" --no-deps -q pyarrow \
    || { echo "h200.sh: pyarrow, which reads WikiText-2's parquet, failed to install into $ARROW" >&2; exit 1; }
fi
PYTHONPATH=${ARROW:+$ARROW${PYTHONPATH:+:}}${PYTHONPATH:-} "$PY" - "$MODEL" "$DATA" "$SET" <<'EOF'
import os, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/ppl")
from huggingface_hub import hf_hub_download, snapshot_download
import models, windows
name, data, set_ = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
hub, m = Path(os.environ["HF_HOME"]) / "hub", models.get(name)
if not (models.snapshot(name, os.environ["HF_HOME"]) / "config.json").exists():
    snapshot_download(m["repo"], revision=m["revision"], allow_patterns=["*.json", "*.safetensors"], cache_dir=str(hub))
path = data / windows.DATASETS[set_]["file"]
data.mkdir(parents=True, exist_ok=True)
if not path.exists() and set_ == "c4":
    path.symlink_to(hf_hub_download("allenai/c4", f"en/{path.name}", repo_type="dataset", cache_dir=str(hub)))
elif not path.exists():
    import json
    import pyarrow.parquet as pq
    got = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset",
                          revision="b08601e04326c79dfdd32d625aee71d232d685c3", cache_dir=str(hub))
    path.write_text(json.dumps(pq.read_table(got).column("text").to_pylist(), ensure_ascii=False), encoding="utf-8")
windows.dataset_path(set_, data)
EOF

# shellcheck disable=SC2086
"$PY" benchmarks/ppl/run.py eval --model "$MODEL" --set "$SET" --ctx "$CTX" --gpus 1 --data "$DATA" --out "$OUT" $ARMS \
  || { echo "h200.sh: the eval failed or its gates did ($OUT/results.json)" >&2; exit 1; }
[ "$SET" != c4 ] || "$PY" benchmarks/ppl/run.py metrics "$OUT/results.json" --out "$OUT"
