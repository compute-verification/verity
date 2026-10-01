#!/usr/bin/env bash
# One request derive of ROW's shape (LP, T) under several cyclic-GC settings at once, in parallel (same host load), each timed,
# with the Program digest it derived. Measurement only (cursor/build-speed-measure-8c79), never merged.
#   derive_ab.sh ROW LP T MML SETTING...     SETTING = default | a,b,c (gc.set_threshold) | off (gc.disable)
set -euo pipefail
ROW=$1 LP=$2 T=$3 MML=$4
shift 4
SRC=$(pwd)
RID=${RESEARCH_RUN_ID:?}
SCR=/workspace/research/build-speed/runs/$RID
mkdir -p "$SCR" "$RESEARCH_RUN_DIR/out"
trap 'rm -rf "$SCR"' EXIT
export TMPDIR=$SCR HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 CUDA_HOME=/workspace/jobs/cuda-12.9 TORCH_EXTENSIONS_DIR=/workspace/jobs/torch-extensions
export OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/workspace/jobs/cuda-driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/workspace/jobs/venv312/bin/python
export PYTHONPATH=.:$SRC/packages/verity/src:$SRC/tools/research/src
cd "$SRC/integrations/vllm"
"$PY" -m compileall -q -j 16 "$SRC/packages/verity/src" verity_vllm "$SRC/tools/research/src" > /dev/null 2>&1 || true
TGT=$("$PY" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['target']))" "workloads/$ROW.json")
i=0
pids=()
for S in "$@"; do
  i=$((i + 1))
  case "$S" in
    default) PRE="" ;;
    off) PRE="gc.disable()" ;;
    *) PRE="gc.set_threshold($S)" ;;
  esac
  (
    t0=$(date +%s.%N)
    rc=0
    $PY -c "import gc, runpy, sys; $PRE; sys.argv = ['cli', *sys.argv[1:]]; runpy.run_module('verity_vllm.pipeline.cli', run_name='__main__')" \
      build --model unsloth/Llama-3.2-1B --revision 9535bd9b1d1dea6acafbdc4813b728796aeb28da --wrapper request --LP "$LP" --T "$T" \
      --max-model-len "$MML" --workload "workloads/$ROW.json" --out "$SCR/d$i" --target "$TGT" > "$RESEARCH_RUN_DIR/out/derive_$i.log" 2>&1 || rc=$?
    t1=$(date +%s.%N)
    $PY - "$SCR/d$i/result.json" "$S" "$rc" "$t0" "$t1" > "$RESEARCH_RUN_DIR/out/derive_$i.json" << 'EOF'
import json, sys
p, s, rc, t0, t1 = sys.argv[1:]
r = json.load(open(p)) if rc == "0" else {}
print(json.dumps({"setting": s, "rc": int(rc), "wall_s": round(float(t1) - float(t0), 1), "export": (r.get("export") or {}).get("secs"),
                  "derive": r.get("derive_secs"), "digest": (r.get("program") or {}).get("digest"),
                  "correspondence_digest": (r.get("program") or {}).get("correspondence_digest")}))
EOF
    rm -rf "$SCR/d$i"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
cat "$RESEARCH_RUN_DIR"/out/derive_*.json
