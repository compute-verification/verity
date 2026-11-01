#!/usr/bin/env bash
# One arm of cheat_llama.py on a GPU node under `research run --cwd source`, in HT-USEFUL's CUDA venv and tokenized
# FineWeb-Edu shard (built on the node by HT-USEFUL's run_arm.sh), with its outputs in $OUT/NAME.
#   bash benchmarks/hardness_prov/run_cheat.sh NAME CHEAT_LLAMA_ARGS...
set -euo pipefail
name=$1; shift
HU_BASE=${HU_BASE:-/workspace/research/lanes/compute-research/hardness-useful}
test -x "$HU_BASE/venv/bin/python" || { echo "no venv in $HU_BASE" >&2; exit 2; }
test -f "$HU_BASE/fineweb-edu-llama3/meta.json" || { echo "no FineWeb-Edu shard in $HU_BASE" >&2; exit 2; }
export PATH=$HU_BASE/venv/bin:$PATH HF_HOME=${HF_HOME:-/workspace/hf} HU_DATA=$HU_BASE/fineweb-edu-llama3
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
python "$(cd "$(dirname "$0")" && pwd)/cheat_llama.py" --out "${OUT:?}/$name" "$@"
