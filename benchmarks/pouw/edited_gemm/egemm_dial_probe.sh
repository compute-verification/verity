#!/usr/bin/env bash
# The served-GEMM scalar-only fold dial, measured once for correctness and cost a
# round. PROFILE owns the honest knee sweep; this run does not duplicate it. Run from a pouw-kernels checkout with
# egemm.scalar-dial.patch applied (EG_ROUNDS added to egemm.cu's XF absorb; the egemm_ref.py mirror added to xin_receipts
# and xin13_receipts). For each R in EG_R_LIST it builds egemm with -DEG_ROUNDS=R (the XOR-in(13,19) absorb of each
# committed word looped R times, no filler MMAs; R=1 byte-identical to the fold of record), dumps the x13_l2o SASS and the
# ptxas report, then in one gpu-lease: `egemm gate` at a 2-atom (k<=4096) and a 3-atom (k>4096) shape with egemm_ref.py
# bit-exact under EG_ROUNDS=R (the served product Y against cuBLASLt and the receipt re-derived on the CPU), `egemm bench`
# (hot ms a arm) and `egemm il` (x13_l2o vs e_plain) at the prefill shape, and ncu (issue/tensor) for R in EG_NCU_R.
# Clocks are read, never set.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/pouw/edited_gemm   # exported: the gpu-lease `bash -c` subshell runs egemm_ref.py as "$D/egemm_ref.py"
export B; B=$(mktemp -d /tmp/egdial.XXXXXX); trap 'rm -rf "$B"' EXIT
export EG_R_LIST=${EG_R_LIST:-"1 2 4"} EG_NCU_R=${EG_NCU_R:-"1 2 4"}
export GSHP=${GSHP:-"2048 2048 4096"} GSHP3=${GSHP3:-"2048 2048 8192"} TSHP=${TSHP:-"8192 8192 8192"}
export SEED=${SEED:-0x5eed}
git rev-parse HEAD > "$O/commit.txt" 2>&1
echo "R_list=$EG_R_LIST ncu_R=$EG_NCU_R gate=$GSHP gate3=$GSHP3 time=$TSHP seed=$SEED" > "$O/params.txt"
NVCC="nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a"

for R in $EG_R_LIST; do
  $NVCC -DEG_ROUNDS="$R" -Xptxas -v -o "$B/egemm_$R" "$D/egemm.cu" -lcuda -lcublasLt -lnvidia-ml > "$O/build-R$R.log" 2>&1 \
    || { echo "build R=$R failed"; grep -iE "error" "$O/build-R$R.log" | head -20; exit 4; }
  cuobjdump -sass "$B/egemm_$R" | awk '/Function : /{p = (/x13_l2o/ && $0 !~ /k3/)} p' > "$O/sass-x13-R$R.txt" 2>&1
  { echo "SHF  $(grep -cE '\bSHF'  "$O/sass-x13-R$R.txt")";
    echo "LOP3 $(grep -cE '\bLOP3' "$O/sass-x13-R$R.txt")";
    echo "IMAD $(grep -cE '\bIMAD' "$O/sass-x13-R$R.txt")";
    echo "LEA  $(grep -cE '\bLEA'  "$O/sass-x13-R$R.txt")";
    grep -E "registers|stack frame" "$O/build-R$R.log" | head -3; } > "$O/sass-x13-R$R.counts" 2>/dev/null || true
done

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for R in $EG_R_LIST; do
    gd="$B/gate_$R"; mkdir -p "$gd" "${gd}_3"
    "$B/egemm_$R" gate "$gd"    $GSHP  $SEED > "$O/egate-R$R.jsonl"  2>> "$O/errors.txt" || echo "gate R=$R rc $?"  >> "$O/rc.txt"
    "$B/egemm_$R" gate "${gd}_3" $GSHP3 $SEED > "$O/egate3-R$R.jsonl" 2>> "$O/errors.txt" || echo "gate3 R=$R rc $?" >> "$O/rc.txt"
    EG_ROUNDS=$R python3 -B "$D/egemm_ref.py" "$gd" "${gd}_3" > "$O/eref-R$R.jsonl" 2>> "$O/errors.txt" \
      && echo "eref R=$R ok" >> "$O/rc.txt" || echo "eref R=$R rc $?" >> "$O/rc.txt"
    EG_ARMS="e_plain,x13_l2o" "$B/egemm_$R" bench 20 $TSHP >> "$O/bench.jsonl" 2>> "$O/errors.txt" || echo "bench R=$R rc $?" >> "$O/rc.txt"
    "$B/egemm_$R" il 30 x13_l2o e_plain $TSHP >> "$O/il.jsonl" 2>> "$O/errors.txt" || echo "il R=$R rc $?" >> "$O/rc.txt"
  done
  for R in $EG_NCU_R; do
    EG_ARMS="e_plain,x13_l2o" ncu --clock-control none -k regex:x13_l2o --launch-count 1 --launch-skip 2 \
      --section WarpStateStats --section SchedulerStats --section InstructionStats --section ComputeWorkloadAnalysis \
      "$B/egemm_$R" bench 1 $TSHP > "$O/ncu-R$R.txt" 2>&1; echo "ncu R=$R rc $?" >> "$O/rc.txt"
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
echo "=== rc ==="; cat "$O/rc.txt" 2>/dev/null
echo "=== sass op counts ==="; for R in $EG_R_LIST; do echo "R=$R:"; cat "$O/sass-x13-R$R.counts" 2>/dev/null; done
echo "=== gate device_ok ==="; grep -ho '"device_ok": [a-z]*\|"control_rejected": [a-z]*' "$O"/egate*-R*.jsonl 2>/dev/null
echo "=== eref ok ==="; grep -ho '"ok": [a-z]*' "$O"/eref-R*.jsonl 2>/dev/null
echo "=== il ==="; cat "$O/il.jsonl" 2>/dev/null
echo "=== bench ==="; cut -c1-240 "$O/bench.jsonl" 2>/dev/null
exit "$r"
