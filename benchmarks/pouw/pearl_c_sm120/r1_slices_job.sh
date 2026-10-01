#!/usr/bin/env bash
# Research question: how much of R1-H's per-atom-slice credit do exactly repeated and near-duplicate (at most 3 of 32 codes
# apart, or D codes) slices take on Llama-3.1-8B, per linear class, on natural text and on the adversarial token workloads?
#
# Node 2, CPU only: r1_slices.py on every capture present (natural; repeat, phrase, lowent and long from r1_adv_job.sh).
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REV=a2856192dd7c25b842431f39c179a6c2c2f627d1
SNAP="$HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/$REV"
CAPTURES="${CAPTURES:-/workspace/pouw/pouw-design/captures}"
OUT="$RESEARCH_RUN_DIR/out"
RUN=(env PYTHONPATH=packages/verity/src:protocols/pouw uv run --no-project --with numpy python benchmarks/pouw/pearl_c_sm120/r1_slices.py
     --weights "$SNAP" --workers "${WORKERS:-8}" --d "${D:-3}")
mkdir -p "$OUT"
rc=0
for w in natural ${WORKLOADS:-repeat phrase lowent long}; do
  d="$CAPTURES/$w"; first=0
  [[ "$w" == natural ]] && d=/workspace/pouw/fp8-security/llama8b-w0
  [[ "$w" == long ]] && first=6144
  [[ -f "$d/manifest.json" || "$w" == natural ]] || { echo "$w: no capture" >&2; continue; }
  "${RUN[@]}" --acts "$d" --first-row "$first" --out "$OUT/$w.json" > "$OUT/$w.log" 2>&1 || { rc=1; tail -5 "$OUT/$w.log" >&2; }
done
exit "$rc"
