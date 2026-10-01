#!/usr/bin/env bash
# Research question: does a residual a faulty glue unit writes at layer l, at any norm, leave layer l + 2's or l + 3's qkv and
# gate/up rows credited and cheap under the residual-state rule (the red team's 6:17 AM PDT condition 3 against the one-layer
# bound, new-designs.md §3.7)?
#
# Node 2, CPU only: r1_carry.py on Llama-3.1-8B-Instruct over the 512 natural tokens of $CAPTURES/natural-every1-512.
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" OMP_NUM_THREADS="${THREADS:-16}"
REV=a2856192dd7c25b842431f39c179a6c2c2f627d1
SNAP="$HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/$REV"
CAPTURES="${CAPTURES:-/workspace/pouw/pouw-design/captures}"
OUT="$RESEARCH_RUN_DIR/out"
mkdir -p "$OUT"
env PYTHONPATH=packages/verity/src:protocols/pouw:benchmarks/pouw/pearl_c_sm120 \
  uv run --no-project --with torch --with transformers --with safetensors --with numpy \
  python benchmarks/pouw/pearl_c_sm120/r1_carry.py --weights "$SNAP" --ids "$CAPTURES/natural-every1-512/ids.npy" \
  --layers "${LAYERS:-4,16,27}" --after "${AFTER:-3}" --scales "${SCALES:-1,8,64,512}" --families "${FAMILIES:-pow2,flat-z50}" \
  --start "${START:-centred}" --threads "${THREADS:-16}" --out "$OUT/carry.json" 2>&1 | tee "$OUT/carry.log"
