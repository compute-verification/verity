#!/usr/bin/env bash
# The cost of mandatory per-word lane work at H's schedule, the measured γ of
# the scalar-saturated designs on VF3q-16. duo on fx<x>'s dependent IMAD/LOP3 chains (x = 0.5, 1, 1.5, 2, 3, 4.5 ops a
# word folded into duo, so a word costs x + 3 ops) beside e_l7qs (H, x = 0), e_plains and cuBLASLt (plain), at e_l7qs's
# schedule. One build, then in one gpu-lease: `egemm gate` at a 2-atom and a 3-atom shape (every kernel's Y against
# cuBLASLt, the receipt kinds re-derived by egemm_ref.py), `egemm bench` (hot items interleaved rep by rep, each with its
# SM clock) and `egemm il` (each arm against plain) at the shapes in TSHPS, and ncu's instruction and pipe sections on
# NCU_ARMS at the first shape. Clocks are read, never set. Run from a checkout at a pushed commit:
#   research run --on vy-nebius-2 --project verity --source . --cwd source \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/edited_gemm/egemm_fill_probe.sh
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/edited_gemm
export B; B=$(mktemp -d /tmp/egfill.XXXXXX); trap 'rm -rf "$B"' EXIT
export ARMS=${ARMS:-"plain,e_plains,e_l7qs,fx05_l7qs,fx1_l7qs,fx15_l7qs,fx2_l7qs,fx3_l7qs,fx45_l7qs"}
export NCU_ARMS=${NCU_ARMS:-"e_l7qs fx1_l7qs fx2_l7qs"}
export GSHP=${GSHP:-"2048 2048 4096"} GSHP3=${GSHP3:-"2048 2048 8192"}
export TSHPS=${TSHPS:-"8192:8192:8192 2048:28672:4096"} REPS=${REPS:-20} ILREPS=${ILREPS:-30}
export SEED=${SEED:-0x5eed}
git rev-parse HEAD > "$O/commit.txt" 2>&1
echo "arms=$ARMS ncu=$NCU_ARMS gate=$GSHP gate3=$GSHP3 time=$TSHPS reps=$REPS il=$ILREPS seed=$SEED" > "$O/params.txt"
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/egemm" "$D/egemm.cu" \
  -lcuda -lcublasLt -lnvidia-ml > "$O/build.log" 2>&1 || { echo "build failed"; grep -i error "$O/build.log" | head -20; exit 4; }
grep -E "spill|registers" "$O/build.log" > "$O/ptxas.txt"
cuobjdump -sass "$B/egemm" > "$B/egemm.sass" 2>/dev/null
for a in ${ARMS//,/ }; do
  [ "$a" = plain ] && continue
  awk -v k="Function : _Z[0-9]+${a}4Args\$" '/Function : /{p = ($0 ~ k)} p' "$B/egemm.sass" > "$B/sass-$a.txt"
  printf '%s' "$a" >> "$O/sass-counts.txt"
  for op in IMMA LOP3 IMAD LEA SHF IADD3; do printf ' %s %s' "$op" "$(grep -cE "\b$op\b|\b$op\." "$B/sass-$a.txt")" >> "$O/sass-counts.txt"; done
  echo >> "$O/sass-counts.txt"
done

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-20}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  gd="$B/gate"; mkdir -p "$gd" "${gd}_3"
  "$B/egemm" gate "$gd" $GSHP $SEED > "$O/egate.jsonl" 2>> "$O/errors.txt" || echo "gate rc $?" >> "$O/rc.txt"
  "$B/egemm" gate "${gd}_3" $GSHP3 $SEED > "$O/egate3.jsonl" 2>> "$O/errors.txt" || echo "gate3 rc $?" >> "$O/rc.txt"
  python3 -B "$D/egemm_ref.py" "$gd" "${gd}_3" > "$O/eref.jsonl" 2>> "$O/errors.txt" \
    && echo "eref ok" >> "$O/rc.txt" || echo "eref rc $?" >> "$O/rc.txt"
  if grep -q "rc " "$O/rc.txt"; then echo "a gate failed: no timing" >> "$O/rc.txt"; exit 5; fi
  for s in $TSHPS; do
    shp=${s//:/ }
    EG_ARMS=$ARMS "$B/egemm" bench "$REPS" $shp >> "$O/bench.jsonl" 2>> "$O/errors.txt" || echo "bench $s rc $?" >> "$O/rc.txt"
    for a in ${ARMS//,/ }; do
      [ "$a" = plain ] && continue
      "$B/egemm" il "$ILREPS" "$a" plain $shp >> "$O/il.jsonl" 2>> "$O/errors.txt" || echo "il $a $s rc $?" >> "$O/rc.txt"
    done
  done
  shp=${TSHPS%% *}; shp=${shp//:/ }
  for a in $NCU_ARMS; do
    EG_ARMS=$a ncu --clock-control none -k "regex:^${a}(_k3)?\$" --launch-count 1 --launch-skip 2 \
      --section InstructionStats --section ComputeWorkloadAnalysis --section SchedulerStats \
      "$B/egemm" bench 1 $shp > "$O/ncu-$a.txt" 2>&1; echo "ncu $a rc $?" >> "$O/rc.txt"
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
echo "=== rc ==="; cat "$O/rc.txt" 2>/dev/null
echo "=== sass ==="; cat "$O/sass-counts.txt" 2>/dev/null
echo "=== gate ==="; grep -ho '"device_ok": [a-z]*\|"control_rejected": [a-z]*' "$O"/egate*.jsonl 2>/dev/null
echo "=== eref ==="; grep -ho '"ok": [a-z]*' "$O/eref.jsonl" 2>/dev/null | sort | uniq -c
echo "=== il ==="; cat "$O/il.jsonl" 2>/dev/null
exit "$r"
