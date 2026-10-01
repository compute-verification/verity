#!/usr/bin/env bash
# py-spy over one request derive of ROW's workload (LP, T), as the Build's derive runs it (bench_build.sh's environment); collapsed
# stacks to $RESEARCH_RUN_DIR/out/. Measurement only.
set -euo pipefail
ROW=$1 LP=$2 T=$3 MML=$4
SRC=$(pwd)
RID=${RESEARCH_RUN_ID:?}
BASE=/workspace/research/build-speed
SCR=$BASE/runs/$RID
mkdir -p "$SCR/tmp" "$RESEARCH_RUN_DIR/out"
trap 'rm -rf "$SCR"' EXIT
export TMPDIR=$SCR/tmp
export PY=/workspace/jobs/venv312/bin/python
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 CUDA_HOME=/workspace/jobs/cuda-12.9 TORCH_EXTENSIONS_DIR=/workspace/jobs/torch-extensions
export OMP_NUM_THREADS=16 VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
export PATH=/workspace/jobs/venv312/bin:$CUDA_HOME/bin:$PATH
cd "$SRC/integrations/vllm"
export PYTHONPATH=.:$SRC/packages/verity/src:$SRC/tools/research/src CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/workspace/jobs/cuda-driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"$PY" -m compileall -q -j 16 "$SRC/packages/verity/src" verity_vllm "$SRC/tools/research/src" > /dev/null 2>&1 || true
TGT=$("$PY" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['target']))" "workloads/$ROW.json")
"$BASE/spyenv/bin/py-spy" record --rate 50 --format raw --nonblocking -o "$RESEARCH_RUN_DIR/out/derive_LP${LP}_T${T}.collapsed" -- \
  "$PY" -m verity_vllm.pipeline.cli build --model unsloth/Llama-3.2-1B --revision 9535bd9b1d1dea6acafbdc4813b728796aeb28da --wrapper request \
  --LP "$LP" --T "$T" --max-model-len "$MML" --workload "workloads/$ROW.json" --out "$SCR/derive" --target "$TGT" > "$RESEARCH_RUN_DIR/out/derive.log" 2>&1
python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({'export': r['export']['secs'], 'derive': r['derive_secs'], 'total': r.get('wall_secs_total'), 'digest': r['program']['digest']}))" \
  "$SCR/derive/result.json" | tee "$RESEARCH_RUN_DIR/out/derive_times.json"
