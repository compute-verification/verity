#!/usr/bin/env bash
# Research question: can R1-H's exact-window tile rule, a debit of every step in an exact run of at least L atoms capped at
# rho of the tile's credit, reject every verifier tile a written-row region touches while
# passing model rows, and at what thresholds and false-positive rate? Per tile of 16, 64 and 128 words square, under the
# centred start: the share of words exact over 4-16 consecutive atoms, the exact and S-determined step shares, the m16n8
# sub-tiles S-determined or cheaply spotted in every word, and the domain check. Model rows: natural text, the repeated
# token and the repeated phrase at layers 0, 8, 16 and 28 (row 0, the attention sink's position, included); written rows:
# flat, flat-z50, flat-z75, pow2 and pow2-z50 at layer 8. Every class, on Llama-3.1-8B's transformed weights.
#
# Node 2, CPU only: single-threaded processes, at most WORKERS at once (512 rows x 512 columns x 16 atoms per class).
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
  for w in natural repeat phrase; do
    acts=$NATURAL; [[ $w != natural ]] && acts=$CAPTURES/$w
    for layer in 0 8 16 28; do echo "$w $acts $layer -"; done
  done
  for f in flat flat-z50 flat-z75 pow2 pow2-z50; do echo "adv-$f $NATURAL 8 $f"; done
} | xargs -P "${WORKERS:-12}" -L 1 bash -c '
  name=$0 acts=$1 layer=$2 adv=$3
  env PYTHONPATH=../../../packages/verity/src:../../../protocols/pouw:. uv run --no-project --with numpy python r1_tiles.py \
    --acts "$acts" --weights "$SNAP" --classes qkv-rot,gateup-rot,o-vo-il,down-bh32 --layers "$layer" --rows 512 \
    --cols 512 --atoms 16 --h 10 --tiles 16,64,128 $([[ $adv != - ]] && echo --adv "$adv") --out "$OUT/$name" \
    > "$OUT/$name-L$layer.log" 2>&1'
