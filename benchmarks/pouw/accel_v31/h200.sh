#!/usr/bin/env bash
# ACCEL's no-fold receipt (v31.cu) on the H200, sm_90a, at the PoUW filler's segments (HASH-27, CMP-172): L' 512 at t 8
# and 16, L = 4,096 and 8,192. run.sh as it stands, with GENCODE sm_90a and no CUTLASS baseline. For each t, its gates
# (one lease) and its bench (another):
#   gates  run.sh's own at L' 256 t 3, then at L' 512 t 8 and 16: one and several whole segments, a partial last one,
#          wrap-heavy and all-127 operands, a third Z digit, 8 KiB rows, and the three cheats (skip, forge, forgeO);
#          the first t runs them all, the second its own whole-segment gate (QUICK=1: each t its own gate and skip
#          cheat alone, for a knob sweep whose variants the bench's sampled tiles check)
#   bench  the frozen shape's calls at M = 16,384 (q, standing for o too, kv, up, down and the head) at ROW 1 KiB (the
#          protocol's leaf) and 8 KiB
# then h200_step.py's step (52 layers plus the head) a t and a ROW, gamma_u against wgmma.linear's 3.06 s (HASH-22).
#
#   research run --on vy-dc-h200 --project verity --campaign cmp-172 --source . --cwd source --timeout 120m \
#     --env GPU_LEASE_WHO=kernels-kcommitfb -- bash benchmarks/pouw/accel_v31/h200.sh
#
# Env: TS (default "8 16"), LP (the bench's and the per-t gates' L', default 512), VARIANTS (default "row=1024 row=8192"), REPS (default 5), QUICK, SHAPES, MODEL (h200_step.py's
# --model: h200, or 8b with SHAPES8B's calls), and run.sh's (NVFLAGS=-DV31_WG with a gemm=1 variant: wg_proto.cuh's GEMMs).
# SHAPES8B: the demo's 8B calls at M = 16,384 (Llama-3-8B: QKV, O, gate-up, down).
set -uo pipefail
D=benchmarks/pouw/accel_v31
O=${OUT:?}
PY=${PY:-uv run -q --no-project --with numpy python}
export GENCODE=arch=compute_90a,code=sm_90a CUTLASS=${CUTLASS:-0} LP=${LP:-512} ROW=1024
export VARIANTS=${VARIANTS:-row=1024 row=8192} REPS=${REPS:-5}
SHAPES8B="8b-qkv:16384:6144:4096 8b-o:16384:4096:4096 8b-up:16384:28672:4096 8b-down:16384:4096:14336"
[ "${MODEL:-h200}" = 8b ] && SHAPES=${SHAPES:-$SHAPES8B}
export SHAPES=${SHAPES:-"h200-q:16384:16384:16384 h200-kv:16384:128:16384 h200-up:16384:65536:16384
h200-down:16384:16384:65536 h200-head:16384:256:16384"}
ALL="
g-llm 256 256 1536 llm none 256 3 1024
g-uniform 256 384 1280 uniform none 256 3 1024
g-uniform512 256 512 1536 uniform none 256 3 1024
g-zero 128 128 768 zero none 256 3 1024
g-dz3 128 128 20480 llm none 256 3 1024
g-lp512 256 256 2048 llm none 512 2 1024
g-max512 256 256 2048 max none 512 2 1024
c-skip 256 256 1536 llm skip:2 256 3 1024
g-t8 256 256 16384 llm none 512 8 1024
g-t8-part 128 256 10240 llm none 512 8 1024
g-t8-uniform 256 512 8192 uniform none 512 8 1024
g-t8-dy3 128 256 32768 llm none 512 8 1024
g-t16 256 256 16384 llm none 512 16 1024
g-t16-part 128 256 20480 llm none 512 16 1024
g-t16-max 256 256 16384 max none 512 16 1024
g-t16-row8k 256 256 16384 llm none 512 16 8192
c-t8-skip 256 256 8192 llm skip:3 512 8 1024
c-t16-forge 256 256 16384 llm forge:1:0.25 512 16 1024
c-t16-forgeO 256 256 16384 llm forgeO:0.25 512 16 1024
"
first=1
for t in ${TS:-8 16}; do
  mkdir -p "$O/t$t"
  g="g-t$t 256 256 16384 llm none $LP $t 1024"
  if [ "${QUICK:-0}" = 1 ]; then g="$g
c-t$t-skip 256 256 16384 llm skip:3 $LP $t 1024"
  elif [ "$first" = 1 ]; then g=$ALL; fi
  OUT="$O/t$t" T=$t GATES="$g" bash $D/run.sh > "$O/t$t/run.log" 2>&1
  echo "t$t run.sh rc $?" >> "$O/rc.txt"
  first=0
done
$PY $D/h200_step.py --model "${MODEL:-h200}" "$O"/t*/bench-*.jsonl > "$O/step.md" 2> "$O/step.err"
echo "step rc $?" >> "$O/rc.txt"
for t in ${TS:-8 16}; do
  echo "== t$t"; cat "$O/t$t/rc.txt" 2>/dev/null; cat "$O/t$t/verdicts.txt" 2>/dev/null; cat "$O/t$t/table.md" 2>/dev/null
done
cat "$O/step.md"
cat "$O/rc.txt"
! grep -qv ' rc 0$' "$O/rc.txt"
