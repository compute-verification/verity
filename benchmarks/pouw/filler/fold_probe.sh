#!/usr/bin/env bash
# The R-round fold's cost a word a round, beside the saturated noise IMMA.
# For each R in PF_R_LIST it builds dcap with -DPF_ROUNDS=R and the chosen fold (PF_FOLD = XIN13 default, or LI), dumps
# the dfill_k SASS and the ptxas register/instruction report, then in one gpu-lease runs `dcap dalone 8 4 S` (share of
# int8 peak) for every R, `dcap dgate` for R=1 and the largest R (GPU vs host dfill_ref bit-exact under rounds), and ncu
# (instruction/compute/scheduler/warp sections) for R in PF_NCU_R. Clocks are read, never set.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/foldprobe.XXXXXX); trap 'rm -rf "$B"' EXIT
export PF_R_LIST=${PF_R_LIST:-"1 2 3 4 6 8"} PF_NCU_R=${PF_NCU_R:-"1 2 4"} PF_S=${PF_S:-256}
export PF_FOLD=${PF_FOLD:-XIN13}   # XIN13 or LI
case "$PF_FOLD" in XIN13) FOLDFLAG="-DPFILL_XIN13";; LI) FOLDFLAG="-DPFILL_LI";; *) echo "bad PF_FOLD" >&2; exit 2;; esac
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
echo "fold=$PF_FOLD R_list=$PF_R_LIST S=$PF_S" > "$O/params.txt"

Rmax=1
for R in $PF_R_LIST; do [ "$R" -gt "$Rmax" ] && Rmax=$R; done

for R in $PF_R_LIST; do
  nvcc -O3 -std=c++17 $FOLDFLAG -DPF_ROUNDS="$R" -gencode arch=compute_120a,code=sm_120a -Xptxas -v \
    -o "$B/dcap_$R" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl > "$O/build-R$R.log" 2>&1 \
    || { echo "build R=$R failed"; grep -E "error" "$O/build-R$R.log" | head -20; exit 4; }
  cuobjdump -sass "$B/dcap_$R" | awk '/Function : /{p = /dfill_k/} p' > "$O/sass-R$R.txt" 2>&1
  # the fold loop's op count a MMA: SHF (rotl), LOP3 (xor), IMAD (mul-add) in the dfill_k<4> body
  grep -cE "\bSHF\b" "$O/sass-R$R.txt" > "$O/sass-R$R.counts" 2>/dev/null || true
  { echo "SHF  $(grep -cE 'SHF' "$O/sass-R$R.txt")";
    echo "LOP3 $(grep -cE 'LOP3' "$O/sass-R$R.txt")";
    echo "IMAD $(grep -cE 'IMAD' "$O/sass-R$R.txt")";
    grep -E "registers|Used .* registers|stack frame" "$O/build-R$R.log" | head -3; } > "$O/sass-R$R.counts"
done

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for R in $PF_R_LIST; do
    "$B/dcap_$R" dgate > "$O/dgate-R$R.jsonl" 2>> "$O/errors.txt" && echo "dgate R=$R ok" >> "$O/rc.txt" || echo "dgate R=$R rc $?" >> "$O/rc.txt"
    for rep in 1 2; do
      "$B/dcap_$R" dalone 8 4 "$PF_S" >> "$O/dalone.jsonl" 2>> "$O/errors.txt" \
        || echo "dalone R=$R rc $?" >> "$O/errors.txt"
    done
  done
  for R in $PF_NCU_R; do
    ncu --clock-control none -k regex:dfill_k --launch-count 1 --launch-skip 1 \
      --section WarpStateStats --section SchedulerStats --section InstructionStats --section ComputeWorkloadAnalysis \
      "$B/dcap_$R" done 8 4 "$PF_S" > "$O/ncu-R$R.txt" 2>&1; echo "ncu R=$R rc $?" >> "$O/rc.txt"
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
echo "=== dgate / rc ==="; cat "$O/rc.txt" 2>/dev/null
echo "=== sass op counts ==="; for R in $PF_R_LIST; do echo "R=$R:"; cat "$O/sass-R$R.counts" 2>/dev/null; done
echo "=== dalone ==="; cut -c1-200 "$O/dalone.jsonl" 2>/dev/null
exit "$r"
