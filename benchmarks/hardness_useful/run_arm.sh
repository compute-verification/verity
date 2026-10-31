#!/usr/bin/env bash
# One HT-USEFUL arm on a GPU node, under `research run --cwd source`: the venv and the tokenized FineWeb-Edu shard in
# $HU_BASE (built once, under a lock), then qat.py with its outputs in $OUT/NAME.
#   bash benchmarks/hardness_useful/run_arm.sh NAME QAT_ARGS...
set -euo pipefail
name=$1; shift
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
command -v research >/dev/null && research keep "$HU_BASE" --owner @compute-research --expires 7d --priority high \
  --regenerate "bash benchmarks/hardness_useful/run_arm.sh NAME ARGS" \
  --why "HT-USEFUL: venv, tokenized FineWeb-Edu, compile caches" >/dev/null 2>&1 || true
scratch=$(mktemp -d -p "$HU_BASE")
trap 'rm -rf "$scratch"' EXIT
( flock 9; python "$here/prep.py" --scratch "$scratch" ) 9>"$HU_BASE/prep.lock"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
python -c 'import torch; from importlib.metadata import version as v
print("torch", torch.__version__, "cuda", torch.version.cuda, *(f"{p} {v(p)}" for p in ("transformers", "lm_eval")))'
python "$here/qat.py" --out "${OUT:?}/$name" "$@"
