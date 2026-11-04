#!/usr/bin/env bash
# The whole decode step co-resident with a 1-block pump on every SM, its
# linears through flin's FLIN_FIT tiles (a CTA fits beside the pump block: ≤ 32 K registers, ≤ 41 KB of shared memory)
# and lm_head too (FLIN_LM=1), attention dcap's (it already fits, capfill_fit_probe.sh). Gates: dgate on the fit build,
# fgate (every fitted tile bit for bit, every row) on the CP_FGATE builds. Then flin alone at each fitted M (CP_FLIN
# builds), and fstep co (FILL_SMS -1, FW = AFW = 0: the step beside the pump at the lowest priority) for each CP_RUNS
# entry, against the cuBLASLt step alone on the whole GPU; the control (c1) is the same co step with flin's default
# tiles. A co run may take only the linears or only attention (FSTEP_KIND), flin without PDL, and another pump shape.
# Builds: fit (FLIN_FIT), fit2 (FLIN_FIT=2), fitx (FLIN_FIT, the pump's NC 4 the provisional XIN13 fold), fitxy...
# and fitxtNS... (fitx, its pump waiting out, or sleeping NS ns a read on, the SMs flin's CTAs occupy: PF_YIELD; below),
# c1 (default tiles), all with the 1-block pump; dgate runs on every build, and every build's dfill_k registers and SASS are kept.
# (A software-pipelined pump, B fragment j + 1's MMAs under j's folds, was slower alone and beside the step:
# ptxas already spaces the MMAs, and duo's fold is bound by the ALU-heavy pipe. Attention's warps
# marking their SMs for PF_YIELD cost attention 1.13× against 1.05× beside a running pump and never held the pump off.)
# CP_ALONE
# ("BUILD:NC:S ...") runs the pump alone (dalone) and CP_NCU ("BUILD:NC:S ...") ncu over one of its launches, with
# CP_NCU_METRICS (ncu's --metrics) added to its sections and the raw page kept. Clocks are read, never set.
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/capfillf.XXXXXX); trap 'rm -rf "$B"' EXIT
# CP_RUNS: BUILD,B,L,FSTEP_KIND,FLIN_PDL,W:NC:S[,FILL_SMS[,FLIN_GRID]] a step each, co unless FILL_SMS > 0 puts the pump on
# a split, FLIN_GRID flin's CTAs (default one an SM)
# (default: the fit build at each decode point, then the control at B 32, whole steps beside the pump at 8:2:256)
export CP_RUNS=${CP_RUNS:-"fit,1,2048,2,1,8:2:256 fit,32,2048,2,1,8:2:256 fit,128,2048,2,1,8:2:256 fit,256,2048,2,1,8:2:256 c1,32,2048,2,1,8:2:256"}
export CP_PAD=${CP_PAD:-4} CP_FGATE=${CP_FGATE:-fit} CP_FLIN=${CP_FLIN:-fit} CP_ALONE=${CP_ALONE:-} CP_NCU=${CP_NCU:-}
export CP_NCU_METRICS=${CP_NCU_METRICS:-}
[ "$CP_FGATE" = none ] && CP_FGATE=""
[ "$CP_FLIN" = none ] && CP_FLIN=""
git rev-parse HEAD > "$O/commit.txt" 2>&1
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
for v in CP_RUNS CP_PAD CP_FGATE CP_FLIN CP_ALONE CP_NCU CP_NCU_METRICS; do echo "$v=${!v}"; done > "$O/params.txt"

declare -A FLAGS=([fit]="-DFLIN_FIT" [fit2]="-DFLIN_FIT=2" [fitx]="-DFLIN_FIT -DPFILL_XIN13" [c1]="")
builds=$(for x in fit $CP_FGATE $CP_FLIN $CP_RUNS $CP_ALONE $CP_NCU; do x=${x%%,*}; echo "${x%%:*}"; done | sort -u)
# fitx(y|tNS)[jJ][e]: fitx whose pump yields the SMs flin's CTAs occupy (PF_YIELD): y waits the mark out, tNS sleeps NS ns
# a read while it holds; jJ reads it every J B fragments (default 16); e marks from the CTA's start
for b in $builds; do
  if [[ $b =~ ^fitx(y|t([0-9]+))(j([0-9]+))?(e)?$ ]]; then
    f="-DFLIN_FIT -DPFILL_XIN13"
    if [ "${BASH_REMATCH[1]}" = y ]; then f+=" -DPF_YIELD=1"; else f+=" -DPF_YIELD=2 -DPF_YIELD_NS=${BASH_REMATCH[2]}"; fi
    [ -n "${BASH_REMATCH[4]}" ] && f+=" -DPF_YIELD_J=${BASH_REMATCH[4]}"
    [ -n "${BASH_REMATCH[5]}" ] && f+=" -DPF_YIELD_EARLY=1"
    FLAGS[$b]=$f
  fi
done
# shellcheck disable=SC2086
export BUILDS; BUILDS=$(echo $builds)
pids=()
for b in $builds; do
  [ -n "${FLAGS[$b]+x}" ] || { echo "unknown build $b"; exit 4; }
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -DPFILL_BU -DPF_PW=4 -DPF_CTAS=1 -DPF_MINB=2 ${FLAGS[$b]} -Xcompiler -mpopcnt \
    -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap_$b" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl \
    > "$O/build-$b.log" 2>&1 &
  pids+=("$!")
done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
for b in $builds; do
  [ -x "$B/dcap_$b" ] || { echo "build $b failed"; grep -E "error" "$O/build-$b.log" | head -20; bad=1; }
  grep -A4 -E "Compiling entry function '_Z6flin_kILi(32|64|128)ELi[0-9]+ELi[0-9]+ELi[24]ELi[12]ELi0E" "$O/build-$b.log" \
    | grep -E "Compiling|registers" > "$O/flin-regs-$b.txt"
  grep -A4 -E "Compiling entry function '_Z7dfill_kILi[0-9]+EE" "$O/build-$b.log" \
    | grep -E "Compiling|registers|spill" > "$O/pump-regs-$b.txt"
  cuobjdump -sass "$B/dcap_$b" 2> /dev/null | awk '/Function : /{p = ($0 ~ /dfill_k/)} p' > "$O/sass-$b.txt"
done
[ "$bad" = 0 ] || exit 4

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-30}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  for b in $BUILDS; do
    "$B/dcap_$b" dgate > "$O/dgate-$b.jsonl" 2>> "$O/errors.txt"; echo "dgate $b rc $?" >> "$O/rc.txt"
  done
  for b in $CP_FGATE; do
    "$B/dcap_$b" fgate > "$O/fgate-$b.jsonl" 2>> "$O/errors.txt"; echo "fgate $b rc $?" >> "$O/rc.txt"
  done
  for b in $CP_FLIN; do
    for m in 32 64 128; do
      "$B/dcap_$b" flin "$m" 0 2 0 >> "$O/flin-$b.jsonl" 2>> "$O/errors.txt"; echo "flin $b $m rc $?" >> "$O/rc.txt"
    done
  done
  for a in $CP_ALONE; do
    IFS=: read -r b nc s <<< "$a"
    DFILL_PAD_KB=$CP_PAD "$B/dcap_$b" dalone 8 "$nc" "$s" 2>> "$O/errors.txt" \
      | sed "s/^{/{\"build\": \"$b\", \"pad_kb\": $CP_PAD, /" >> "$O/dalone.jsonl"
    echo "dalone $a rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
  done
  for a in $CP_NCU; do
    IFS=: read -r b nc s <<< "$a"
    r="$O/ncu-$b-nc$nc"
    # shellcheck disable=SC2086
    DFILL_PAD_KB=$CP_PAD ncu --clock-control none -k regex:dfill_k --launch-count 1 --launch-skip 1 \
      --section InstructionStats --section SchedulerStats --section WarpStateStats --section ComputeWorkloadAnalysis \
      ${CP_NCU_METRICS:+--metrics $CP_NCU_METRICS} -o "$r" "$B/dcap_$b" done 8 "$nc" "$s" > "$r.log" 2>&1
    echo "ncu $a rc $?" >> "$O/rc.txt"
    ncu --import "$r.ncu-rep" --page details > "$r.txt" 2>&1
    ncu --import "$r.ncu-rep" --page raw --csv > "$r-raw.csv" 2>&1
  done
  for run in $CP_RUNS; do
    IFS=, read -r b bb l kind pdl pump fsm grid <<< "$run"
    fold=duo; case $b in fitx*) fold=xin13 ;; esac
    case $pump in *:9:*) fold=arx ;; esac
    FSTEP_KIND=$kind FLIN_PDL=$pdl FSTEP_PUMP=$pump DFILL_PAD_KB=$CP_PAD FLIN_LM=1 FLIN_GRID=${grid:-0} \
      "$B/dcap_$b" fstep "$bb" "$l" 0 2 0 0 \
      "${fsm:--1}" 2>> "$O/errors.txt" \
      | sed "s/^{\"mode\": \"fstep\"/{\"mode\": \"fstep\", \"build\": \"$b\", \"fold\": \"$fold\", \"pad_kb\": $CP_PAD/" \
      >> "$O/fstep.jsonl"
    echo "fstep $run rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
echo "=== rc ==="; cat "$O/rc.txt" 2>/dev/null
echo "=== regs ==="; cat "$O"/flin-regs-*.txt "$O"/pump-regs-*.txt 2>/dev/null
echo "=== dalone ==="; cut -c1-300 "$O/dalone.jsonl" 2>/dev/null
echo "=== fgate fails ==="; cat "$O"/fgate-*.jsonl 2>/dev/null | grep -c '"ok": false'
echo "=== flin ==="; cut -c1-300 "$O"/flin-*.jsonl 2>/dev/null
echo "=== fstep ==="; cut -c1-600 "$O/fstep.jsonl" 2>/dev/null
exit "$r"
