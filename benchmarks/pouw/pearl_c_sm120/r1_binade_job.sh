#!/usr/bin/env bash
# Research question: on Llama-3.1-8B's rotated rows, over the whole chain, does R1-H's per-atom lemma (C - H = sum_t R_t
# while the accumulator keeps H's binade) hold bit for bit; what share of words leave H's binade under the current (p10),
# centred (c10) and Cauchy-Schwarz (cs) starts, on natural, repeated-token and written rows; and what is U's error with
# the floor's bias folded into H?
#
# Node 2, CPU only, single-threaded processes (128 rows x 256 columns per class, all of k). PART=model runs natural and
# repeated-token rows, one process per layer; PART=written (the default) runs the written families at layer 8.
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REV=a2856192dd7c25b842431f39c179a6c2c2f627d1
SNAP="$HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/$REV"
NATURAL="${NATURAL:-/workspace/pouw/fp8-security/llama8b-w0}"
REPEAT="${REPEAT:-/workspace/pouw/pouw-design/captures/repeat}"
OUT="$RESEARCH_RUN_DIR/out"
mkdir -p "$OUT"
cd benchmarks/pouw/pearl_c_sm120
RUN=(env PYTHONPATH=../../../packages/verity/src:../../../protocols/pouw:. uv run --no-project --with numpy python r1_binade.py
     --weights "$SNAP" --classes qkv-rot,gateup-rot,o-vo-il,down-bh32 --rows 128 --cols 256 --h 10)
pids=()
if [[ "${PART:-written}" == model ]]; then
  for w in natural repeat; do
    acts=$NATURAL; [[ $w == repeat ]] && acts=$REPEAT
    for layer in 0 8 28; do
      "${RUN[@]}" --acts "$acts" --layers "$layer" --out "$OUT/$w" > "$OUT/$w-L$layer.log" 2>&1 & pids+=($!)
    done
  done
else
  for f in flat-z75 pow2; do
    "${RUN[@]}" --acts "$NATURAL" --layers 8 --adv "$f" --out "$OUT/adv-$f" > "$OUT/adv-$f.log" 2>&1 & pids+=($!)
  done
fi
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit "$rc"
