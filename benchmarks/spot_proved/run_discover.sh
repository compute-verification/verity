#!/usr/bin/env bash
# discover.py under research run, from the shipped tree's root (--cwd source), on a pod that pod_bootstrap.sh set up:
#   [TP=2] [LAYERS=4] [STEPS=0,3] [VENV=/workspace/venv312] bash benchmarks/spot_proved/run_discover.sh
# The capture goes to /workspace/spot-capture/<run> (outside OUT: custody uploads OUT); each step is then staged into
# OUT/values-step<s> (stage.py), which is small.
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
CAP=${CAPTURE:-/workspace/spot-capture/$(basename "$D")}; mkdir -p "$CAP"
export VERITY_SPOT_CAPTURE=$CAP VERITY_SPOT_STEPS=${STEPS:-0,3} VERITY_SPOT_OPEN=${OPEN:-dense:1,moe:1}
export TORCHINDUCTOR_CACHE_DIR=/workspace/spot-cache/inductor VLLM_CACHE_ROOT=/workspace/spot-cache/vllm TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=index,name,memory.used --format=csv
"$PY" "$here/discover.py" --config "$CFG" --out "$OUT" --capture "$CAP" --tp "${TP:-2}" --layers "${LAYERS:-4}" \
  --steps "$VERITY_SPOT_STEPS"
rc=$?
"$PY" "$here/discover.py" --config "$CFG" --out "$OUT" --capture "$CAP" --steps "$VERITY_SPOT_STEPS" --summarize
ROOT=$(cd "$here/../.." && pwd)
for s in ${VERITY_SPOT_STEPS//,/ }; do
  "$PY" "$here/stage.py" --capture "$CAP" --step "$s" \
    --out "$OUT/values-step$s" --per-type "${PER_TYPE:-4}" --of "rehearsal: GLM-5.3's config with dummy weights, ${LAYERS:-4} layers, TP${TP:-2}" \
    --link "none: dummy weights" > "$OUT/stage-step$s.log" 2>&1 || echo "stage step $s failed"
  tail -n 30 "$OUT/stage-step$s.log"
done
du -sh "$OUT"/* "$CAP" 2>/dev/null
exit $rc
