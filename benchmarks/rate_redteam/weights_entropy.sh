#!/usr/bin/env bash
# The deployed checkpoint's FP8 payload under adaptive code lengths and lzma / bz2 / zlib (weights_entropy.py), CPU only.
# Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 48 \
#     --campaign resource-accounting --declared-output 'out/*' \
#     -- taskset -c 48-95 bash benchmarks/rate_redteam/weights_entropy.sh
#
# WORK is the e2e's vLLM environment on node 2 (benchmarks/pouw/pearl_c_vllm/setup.sh), whose checkpoint
# ra-vllm-encoded serves.
set -euo pipefail
O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
[ -x "$PY" ] && [ -f "$MODEL/model.safetensors.index.json" ] || { echo "no $PY or $MODEL"; exit 3; }
"$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(json.dumps({k: d.get(k) for k in ("model", "revision", "vllm", "torch")}))' \
  "$WORK/ready.json" > "$O/ready.json" || true
FP8=$("$PY" -c 'import importlib.util as u, os; print(os.path.join(os.path.dirname(u.find_spec("vllm").origin), "model_executor/layers/quantization/fp8.py"))')
grep -n "scaled_fp8_quant\|per_tensor\|is_checkpoint_fp8_serialized" "$FP8" > "$O/vllm-fp8-online.txt" || true
"$PY" benchmarks/rate_redteam/weights_entropy.py --model "$MODEL" --procs "${PROCS:-16}" --out "$O/entropy.json" \
  2>&1 | tee "$O/entropy.log"
