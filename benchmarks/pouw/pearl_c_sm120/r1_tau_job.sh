#!/usr/bin/env bash
# Research question: what share of verifier tiles of honest Llama-3.1-8B rows would the per-row cheap test reject at
# tau_row = 20, 25 and 30% (a tile fails when any of its rows is cheaply S-determined on more than tau_row of its steps), in
# every class and at every layer, on natural text and the adversarial token workloads (the red team's 7:48 AM PDT point 3)?
#
# Node 2, CPU only: r1_tiles.py (centred start, 512 rows x 512 columns x 16 atoms per class, tiles of 16, 64 and 128 words)
# on model rows only: every layer of the 512-token natural capture; rows 1,024-1,535 of the 2,048-token natural capture at
# layers 0, 8, 16, 24; the repeated token, repeated phrase, 4-token draw and long context at layers 0, 8, 16, 24.  Then
# r1_tau.py's rejection rates.
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REV=a2856192dd7c25b842431f39c179a6c2c2f627d1
export SNAP="$HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/$REV"
NATURAL="${NATURAL:-/workspace/pouw/fp8-security/llama8b-w0}"
CAPTURES="${CAPTURES:-/workspace/pouw/pouw-design/captures}"
export OUT="$RESEARCH_RUN_DIR/out"
mkdir -p "$OUT"
cd benchmarks/pouw/pearl_c_sm120
uv run --no-project --with numpy python -c "import numpy" > /dev/null
{
  for layer in $(seq 0 31); do echo "every1 $CAPTURES/natural-every1-512 $layer 0"; done
  for layer in 0 8 16 24; do echo "natural-r1024 $NATURAL $layer 1024"; done
  for w in repeat phrase lowent long; do
    first=0; [[ $w == long ]] && first=6144
    for layer in 0 8 16 24; do echo "$w $CAPTURES/$w $layer $first"; done
  done
} | xargs -P "${WORKERS:-24}" -L 1 bash -c '
  name=$0 acts=$1 layer=$2 first=$3
  mkdir -p "$OUT/$name"; echo "$first" > "$OUT/$name/first-row"
  env PYTHONPATH=../../../packages/verity/src:../../../protocols/pouw:. uv run --no-project --with numpy python r1_tiles.py \
    --acts "$acts" --weights "$SNAP" --classes qkv-rot,gateup-rot,o-vo-il,down-bh32 --layers "$layer" --rows 512 \
    --cols 512 --first-row "$first" --atoms 16 --h 10 --tiles 16,64,128 --out "$OUT/$name" \
    > "$OUT/$name-L$layer.log" 2>&1 || echo "$name L$layer failed" >&2'
uv run --no-project --with numpy python r1_tau.py "$OUT"/every1 "$OUT"/natural-r1024 "$OUT"/repeat "$OUT"/phrase \
  "$OUT"/lowent "$OUT"/long --json "$OUT/tau.json" | tee "$OUT/tau.txt"
