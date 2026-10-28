#!/usr/bin/env bash
# Dither's no-fold receipt (v31.cu) on the H200, sm_90a, at the PoUW filler's segments: each of CONFIGS,
# an L':t pair (default L' 512 at t 8, L = 4,096; t at most 8, dither/PROTOCOL.md's "The segment and the noise tile").
# run.sh as it stands, with GENCODE sm_90a and no CUTLASS baseline. For each pair, its gates (one lease) and its bench
# (another):
#   gates  run.sh's own at L' 256 t 3, then at L' 512 t 8: one and several whole segments, a partial last one,
#          wrap-heavy and all-127 operands, a third Z digit, 8 KiB rows, and the three cheats (skip, forge, forgeO);
#          the first pair runs them all and its own whole-segment gate, the others that gate alone (QUICK=1: each pair
#          its own gate and skip cheat alone, for a knob sweep whose variants the bench's sampled tiles check)
#   bench  MODEL's calls at M = 16,384: mqa16k52 (default), the frozen shape's q (standing for o too), kv, up, down and the
#          head; 8b, Llama 3 8B's QKV, O, gate and up, and down; both, the two
# then h200_step.py's step a pair, ROW and variant: gamma_u against wgmma.linear's 3.06 s for mqa16k52, and
# against the run's own cuBLASLt for each.
#
#   research run --on vy-dc-h200 --project verity --campaign CAMPAIGN --source . --cwd source --timeout 120m \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/accel_v31/h200.sh
#
# Env: CONFIGS (default "512:8"), MODEL, VARIANTS (default "row=1024 row=8192"), REPS (default 5), QUICK, SHAPES
# (in place of MODEL's), and run.sh's (NVFLAGS=-DV31_WG=1: the GEMM on warpgroup MMAs; -DV31_WG_BN=256 beside it: 128 x
# 256 tiles a CTA).
set -uo pipefail
D=benchmarks/pouw/accel_v31
O=${OUT:?}
PY=${PY:-uv run -q --no-project --with numpy python}
export GENCODE=arch=compute_90a,code=sm_90a CUTLASS=${CUTLASS:-0} ROW=1024
export VARIANTS=${VARIANTS:-row=1024 row=8192} REPS=${REPS:-5}
MQA="h200-q:16384:16384:16384 h200-kv:16384:128:16384 h200-up:16384:65536:16384 h200-down:16384:16384:65536
h200-head:16384:256:16384"
B8="h200-qkv:16384:6144:4096 h200-o:16384:4096:4096 h200-gu:16384:28672:4096 h200-down8b:16384:4096:14336"
case ${MODEL:-mqa16k52} in
  mqa16k52) s=$MQA ;;
  8b) s=$B8 ;;
  both) s="$MQA $B8" ;;
  *) echo "MODEL: mqa16k52, 8b or both" >&2; exit 2 ;;
esac
export SHAPES=${SHAPES:-$s}
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
g-t8-max 256 256 16384 max none 512 8 1024
g-t8-row8k 256 256 16384 llm none 512 8 8192
c-t8-skip 256 256 8192 llm skip:3 512 8 1024
c-t8-forge 256 256 16384 llm forge:1:0.25 512 8 1024
c-t8-forgeO 256 256 16384 llm forgeO:0.25 512 8 1024
"
CONFIGS=${CONFIGS:-512:8}
first=1
for c in $CONFIGS; do
  lp=${c%%:*} t=${c##*:} tag="lp${c%%:*}-t${c##*:}"
  mkdir -p "$O/$tag"
  g="g-$tag 256 256 16384 llm none $lp $t 1024"
  if [ "${QUICK:-0}" = 1 ]; then g="$g
c-$tag-skip 256 256 16384 llm skip:3 $lp $t 1024"
  elif [ "$first" = 1 ]; then g="$ALL
$g"; fi
  OUT="$O/$tag" LP=$lp T=$t GATES="$g" bash $D/run.sh > "$O/$tag/run.log" 2>&1
  echo "$tag run.sh rc $?" >> "$O/rc.txt"
  first=0
done
$PY $D/h200_step.py "$O"/lp*/bench-*.jsonl > "$O/step.md" 2> "$O/step.err"
echo "step rc $?" >> "$O/rc.txt"
for c in $CONFIGS; do
  tag="lp${c%%:*}-t${c##*:}"
  echo "== $tag"; cat "$O/$tag/rc.txt" 2>/dev/null; cat "$O/$tag/verdicts.txt" 2>/dev/null; cat "$O/$tag/table.md" 2>/dev/null
done
cat "$O/step.md"
cat "$O/rc.txt"
! grep -qv ' rc 0$' "$O/rc.txt"
