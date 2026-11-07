#!/usr/bin/env bash
# discover.py under research run, from the shipped tree's root (--cwd source), on a pod that pod_bootstrap.sh set up:
#   [TP=2] [LAYERS=4] [STEPS=0,3] [VENV=/workspace/venv312] bash benchmarks/spot_proved/run_discover.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; mkdir -p "$OUT"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python; export PATH="$VENV/bin:$PATH"
command -v nvcc >/dev/null || for d in $(ls -d /usr/local/cuda*/bin 2>/dev/null | sort -rV); do [ -x "$d/nvcc" ] && { export PATH="$PATH:$d"; break; }; done
CFG=${CFG:-/workspace/glm53-config}
[ -f "$CFG/config.json" ] || "$PY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('zai-org/GLM-5.3', revision='aca966e4e02791568aa6a4ced368624b3d897f42', local_dir='$CFG',
      allow_patterns=['*.json', '*.jinja', '*.txt', '*.model', 'tokenizer*']))" || exit 2
"$PY" "$here/capture.py" site "$OUT/site" >/dev/null || exit 2
export PYTHONPATH="$OUT/site${PYTHONPATH:+:$PYTHONPATH}"
export VERITY_SPOT_CAPTURE=$OUT/capture VERITY_SPOT_STEPS=${STEPS:-0,3}
export TORCHINDUCTOR_CACHE_DIR=$OUT/inductor VLLM_CACHE_ROOT=$OUT/vllm-cache TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=index,name,memory.used --format=csv
"$PY" "$here/discover.py" --config "$CFG" --out "$OUT" --tp "${TP:-2}" --layers "${LAYERS:-4}" --steps "$VERITY_SPOT_STEPS"
rc=$?
"$PY" "$here/discover.py" --config "$CFG" --out "$OUT" --steps "$VERITY_SPOT_STEPS" --summarize
find "$OUT/inductor" -name "*.py" -size +2k | head -400 > "$OUT/inductor_files.txt"
du -sh "$OUT"/* 2>/dev/null
exit $rc
