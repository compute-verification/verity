#!/usr/bin/env bash
# Research question: under R1-H's hot start (p10), do rows a prover writes with power-of-two codes (pow2: +-256, one 448;
# pow2-z50: half of them zero) reach the staircase floors W*(L, M) at full width (1,024 rows x the linear's own n, 32 atoms,
# windows of 4-32 atoms) in q/k/v, o and down, on Llama-3.1-8B's layer-8 rotated weights?
#
# Node 2, CPU only: one worker per (family, class), gate/up left out (its n only widens the blocks).
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REV=a2856192dd7c25b842431f39c179a6c2c2f627d1
SNAP="$HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/$REV"
NATURAL="${NATURAL:-/workspace/pouw/fp8-security/llama8b-w0}"
OUT="$RESEARCH_RUN_DIR/out"
CEN=(env PYTHONPATH=packages/verity/src:protocols/pouw uv run --no-project --with numpy python benchmarks/pouw/pearl_c_sm120/r1_census.py
     --weights "$SNAP" --acts "$NATURAL" --classes qkv-rot,o-vo-il,down-bh32 --starts p10 --atoms 32 --rows 1024 --cols 0
     --col-block 512 --layers 8 --workers 3)
mkdir -p "$OUT"
pids=()
for f in pow2 pow2-z50; do
  "${CEN[@]}" --adv "$f" --out "$OUT/adv-$f" > "$OUT/adv-$f.log" 2>&1 & pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit "$rc"
