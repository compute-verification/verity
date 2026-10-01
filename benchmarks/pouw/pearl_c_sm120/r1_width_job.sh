#!/usr/bin/env bash
# Research question: under R1-H's separable hot start (p10), at full width (1,024 rows x the linear's own n), is every block
# exact over a window of 4-12 atoms narrower than the staircase floors W*(L, M), on Llama-3.1-8B's natural rows, on the
# repeated-token workload, and on rows a prover writes in the rotated basis (flat, flat-z50, flat-z75)?
#
# Node 2, CPU only. The three groups run at once (7 + 7 + 3 x 3 workers); gate/up, the widest, goes first in each.
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
REV=a2856192dd7c25b842431f39c179a6c2c2f627d1
SNAP="$HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/$REV"
NATURAL="${NATURAL:-/workspace/pouw/fp8-security/llama8b-w0}"
REPEAT="${REPEAT:-/workspace/pouw/pouw-design/captures/repeat}"
OUT="$RESEARCH_RUN_DIR/out"
CLASSES=gateup-rot,qkv-rot,o-vo-il,down-bh32
CEN=(env PYTHONPATH=packages/verity/src:protocols/pouw uv run --no-project --with numpy python benchmarks/pouw/pearl_c_sm120/r1_census.py
     --weights "$SNAP" --classes "$CLASSES" --starts p10 --atoms 16 --rows 1024 --cols 0 --col-block 512)
mkdir -p "$OUT"
pids=()
"${CEN[@]}" --acts "$NATURAL" --workers 7 --out "$OUT/natural" > "$OUT/natural.log" 2>&1 & pids+=($!)
"${CEN[@]}" --acts "$REPEAT" --workers 7 --out "$OUT/repeat" > "$OUT/repeat.log" 2>&1 & pids+=($!)
for f in flat flat-z50 flat-z75; do
  "${CEN[@]}" --acts "$NATURAL" --adv "$f" --layers 8 --workers 3 --out "$OUT/adv-$f" > "$OUT/adv-$f.log" 2>&1 & pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit "$rc"
