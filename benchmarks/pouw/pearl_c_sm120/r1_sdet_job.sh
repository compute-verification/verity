#!/usr/bin/env bash
# Research question: under R1-H's hot start (p10 and p12), on Llama-3.1-8B's rotated weights, what share of steps is a
# function of the atom's exact sum (the relation attack's reach), how much of it can a prover spot cheaply, and how many
# mma.sync m16n8 tiles are cheap in every step, on natural rows and on rows a prover writes (flat, flat-z50)?
#
# Node 2, CPU only, small (128 rows x 256 columns x 8 atoms per class).
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REV=a2856192dd7c25b842431f39c179a6c2c2f627d1
SNAP="$HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/$REV"
NATURAL="${NATURAL:-/workspace/pouw/fp8-security/llama8b-w0}"
OUT="$RESEARCH_RUN_DIR/out"
mkdir -p "$OUT"
cd benchmarks/pouw/pearl_c_sm120
for h in 10 12; do
  for adv in "" flat flat-z50; do
    PYTHONPATH=../../../packages/verity/src:../../../protocols/pouw:. uv run --no-project --with numpy python r1_sdet.py \
      --acts "$NATURAL" --weights "$SNAP" --classes qkv-rot,gateup-rot,o-vo-il,down-bh32 --layers 8 --rows 128 --cols 256 \
      --atoms 8 --h "$h" ${adv:+--adv "$adv"} --out "$OUT/h$h${adv:+-$adv}" 2>&1 | tee -a "$OUT/sdet.log"
  done
done
