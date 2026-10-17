#!/usr/bin/env bash
# FILLER (note:20261017T0652Z-draft-filler): the whole decode step co-resident with a 1-block pump on every SM, its
# linears through flin's FLIN_FIT tiles (a CTA fits beside the pump block: ≤ 32 K registers, ≤ 41 KB of shared memory)
# and lm_head too (FLIN_LM=1), attention dcap's (it already fits, capfill_fit_probe.sh). Gates: fgate (every fitted
# tile bit for bit, every row) and dgate on the fit build. Then flin alone at each fitted M, and fstep co (FILL_SMS -1,
# FW = AFW = 0: the step beside the pump at the lowest priority) at each decode point, against the cuBLASLt step alone on
# the whole GPU; the control is the same co step with flin's default tiles. Clocks are read, never set.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/capfillf.XXXXXX); trap 'rm -rf "$B"' EXIT
export CP_POINTS=${CP_POINTS:-"1:2048 32:2048 128:2048 256:2048"} CP_CTL=${CP_CTL:-"32:2048"}
export CP_PUMP=${CP_PUMP:-"8:2:256"} CP_PAD=${CP_PAD:-4} CP_FGATE=${CP_FGATE:-1}
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
for v in CP_POINTS CP_CTL CP_PUMP CP_PAD CP_FGATE; do echo "$v=${!v}"; done > "$O/params.txt"

declare -A FLAGS=([fit]="-DPF_CTAS=1 -DPF_MINB=2 -DFLIN_FIT" [c1]="-DPF_CTAS=1 -DPF_MINB=2")
pids=()
for b in fit c1; do
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -DPFILL_BU -DPF_PW=4 ${FLAGS[$b]} -Xcompiler -mpopcnt -gencode arch=compute_120a,code=sm_120a \
    -Xptxas -v -o "$B/dcap_$b" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl > "$O/build-$b.log" 2>&1 &
  pids+=("$!")
done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
for b in fit c1; do
  [ -x "$B/dcap_$b" ] || { echo "build $b failed"; grep -E "error" "$O/build-$b.log" | head -20; bad=1; }
done
[ "$bad" = 0 ] || exit 4
grep -A4 -E "Compiling entry function '_Z6flin_kILi(32|64|128)ELi[0-9]+ELi[0-9]+ELi4ELi[12]ELi0E" "$O/build-fit.log" \
  | grep -E "Compiling|registers" > "$O/flin-fit-regs.txt"

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$B/dcap_fit" dgate > "$O/dgate-fit.jsonl" 2>> "$O/errors.txt"; echo "dgate fit rc $?" >> "$O/rc.txt"
  if [ "$CP_FGATE" = 1 ]; then
    "$B/dcap_fit" fgate > "$O/fgate-fit.jsonl" 2>> "$O/errors.txt"; echo "fgate fit rc $?" >> "$O/rc.txt"
  fi
  for m in 32 64 128; do
    "$B/dcap_fit" flin "$m" 0 2 0 >> "$O/flin-fit.jsonl" 2>> "$O/errors.txt"; echo "flin fit $m rc $?" >> "$O/rc.txt"
  done
  run_co() {  # build B L
    FSTEP_PUMP=$CP_PUMP DFILL_PAD_KB=$CP_PAD FLIN_LM=1 "$B/dcap_$1" fstep "$2" "$3" 0 2 0 0 -1 2>> "$O/errors.txt" \
      | sed "s/^{\"mode\": \"fstep\"/{\"mode\": \"fstep\", \"build\": \"$1\", \"pad_kb\": $CP_PAD/" >> "$O/fstep.jsonl"
    echo "fstep co $1 $2 $3 rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
  }
  for pt in $CP_POINTS; do IFS=: read -r bb l <<< "$pt"; run_co fit "$bb" "$l"; done
  for pt in $CP_CTL; do IFS=: read -r bb l <<< "$pt"; run_co c1 "$bb" "$l"; done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
echo "=== rc ==="; cat "$O/rc.txt" 2>/dev/null
echo "=== regs ==="; cat "$O/flin-fit-regs.txt" 2>/dev/null
echo "=== fgate fails ==="; grep -c '"ok": false' "$O/fgate-fit.jsonl" 2>/dev/null
echo "=== flin ==="; cut -c1-300 "$O/flin-fit.jsonl" 2>/dev/null
echo "=== fstep ==="; cut -c1-600 "$O/fstep.jsonl" 2>/dev/null
exit "$r"
