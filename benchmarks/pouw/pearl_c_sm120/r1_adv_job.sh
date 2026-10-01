#!/usr/bin/env bash
# Research question: on adversarial token workloads (one token repeated, a repeated phrase, a 4-token alphabet, and the late
# positions of an 8,192-token context), does R1-H's hot-started chain stay free of aligned exact regions and repeated atom
# slices on Llama-3.1-8B's distinct rows, per linear class, as it is on WikiText-2?
#
# Node 2, CPU only: captures each workload with pearlc_capture.py (bf16 on the CPU, layers 0, 4, ..., 28) into
# $CAPTURES/<workload> (outside the run dir), then runs r1_census.py on it into $RESEARCH_RUN_DIR/out/<workload>.
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
REV=a2856192dd7c25b842431f39c179a6c2c2f627d1
SNAP="$HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/$REV"
BASE="${BASE:-/workspace/pouw/fp8-security/llama8b-w0/ids.npy}"
CAPTURES="${CAPTURES:-/workspace/pouw/pouw-design/captures}"
OUT="$RESEARCH_RUN_DIR/out"
WORKERS="${WORKERS:-16}"
CAP=(uv run --no-project --with torch --with transformers --with safetensors --with numpy --with huggingface_hub --with datasets)
CEN=(env PYTHONPATH=packages/verity/src:protocols/pouw uv run --no-project --with numpy python benchmarks/pouw/pearl_c_sm120/r1_census.py
     --weights "$SNAP" --classes o-none,o-vo,o-vo-il,down-none,down-perm,down-bh32,qkv-rot,gateup-rot --starts zero,p10
     --atoms 32 --workers "$WORKERS")
mkdir -p "$OUT" "$CAPTURES"
rc=0
for w in ${WORKLOADS:-repeat phrase lowent long}; do
  d="$CAPTURES/$w"
  if [[ ! -f "$d/manifest.json" ]]; then
    if [[ "$w" == long ]]; then
      "${CAP[@]}" python benchmarks/pouw/pearlc_capture.py --model "$MODEL" --revision "$REV" --tokens 8192 --every 4 \
        --threads "$WORKERS" --out "$d" > "$OUT/capture-$w.log" 2>&1 || { rc=1; tail -5 "$OUT/capture-$w.log" >&2; continue; }
    else
      "${CAP[@]}" python benchmarks/pouw/pearl_c_sm120/r1_workloads.py --workload "$w" --base "$BASE" --out "$d" > "$OUT/workload-$w.json"
      "${CAP[@]}" python benchmarks/pouw/pearlc_capture.py --model "$MODEL" --revision "$REV" --every 4 --threads "$WORKERS" \
        --out "$d" > "$OUT/capture-$w.log" 2>&1 || { rc=1; tail -5 "$OUT/capture-$w.log" >&2; continue; }
    fi
  fi
  cp "$d/manifest.json" "$OUT/capture-manifest-$w.json"
  first=0; [[ "$w" == long ]] && first=6144
  "${CEN[@]}" --acts "$d" --first-row "$first" --out "$OUT/$w" > "$OUT/census-$w.log" 2>&1 || { rc=1; tail -5 "$OUT/census-$w.log" >&2; }
  echo "$w: done" >&2
done
exit "$rc"
