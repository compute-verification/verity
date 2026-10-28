#!/usr/bin/env bash
# dcap.cu's fused filler (flin.cuh) on one leased GPU (cwd = the source tree): the decode linears as a warp-specialized GEMM
# whose filler warps run the DRAM-free chain while its consumers wait on memory. Under one lease: fgate (every linear's
# output on every row and the filler's receipts, bit for bit) and CAP's gate first, and nothing is timed unless both pass;
# then each linear's 32 layers against cuBLASLt (flin), ncu's DRAM and tensor-pipe counters for a few configurations, and
# the decode step with its linears through flin (fstep). The round's coins are drawn once per run (coins.txt). Clocks are
# read, never set.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     --send benchmarks/pouw/filler/flin.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/flin.sh"'
#
# FL_FLIN ("M:FW:NC:THR[:PDL] ..."; "none" skips), FL_STEP ("B:L:FW:NC:THR:AFW[:PDL[:FILL_SMS[:LT[:FP[:ANC[:APS[:ALZ[:LM[:ST]]]]]]]]] ..."), PDL 0
# to launch each linear after the previous grid finishes (FLIN_PDL), FILL_SMS > 0 to run the fused step on a green split
# beside the dfill pump on that many SMs, LT the linears that stay on cuBLASLt (FLIN_LT: 1 qkv, 2 o, 4 gate_up, 8 down), FP the
# attention filler blocks' PDL (FATT_PDL: 0 none, 1 with, 2 with their filler warps starting early), ANC their warps'
# chains (FATT_NC, 4 unless set), APS 1 those blocks persistent on the step's SMs (FATT_PERSIST), ALZ their
# filler warps' lazy done test (FATT_LAZY: 0, 2, 4), LM 1 lm_head through flin (FLIN_LM), ST a pfill build's FLIN_STOP, FL_NCU ("M:FW:NC:THR ...", ncu over flin's first launches; "none" skips), FL_NCU_STEP
# ("B:L:FW:NC:THR:AFW[:ANC[:APS[:ALZ]]] ...", ncu's tensor-pipe counters over one step's attention blocks and first linear, each with
# filler warps; FL_NCU_SECTIONS adds those ncu sections over the attention blocks only and keeps the report with its
# details, raw and SASS pages), FL_REPS,
# FL_GATE (0 skips CAP's gate), FL_NVCC (more nvcc flags, -DDFILL_V1 unless set: FILL v1's filler; empty builds pfill.cuh's,
# -DPFILL_BITS its {0,1} tile)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/filler
export B; B=$(mktemp -d /tmp/flin.XXXXXX); trap 'rm -rf "$B"' EXIT
export FL_FLIN=${FL_FLIN:-"32:4:2:0 32:8:4:0 32:8:4:1 64:4:2:0 64:4:4:0 64:8:4:0 128:4:2:0 128:4:4:0 128:4:4:1 256:2:2:0 256:2:4:0"}
export FL_STEP=${FL_STEP:-"32:2048:8:4:0:0 32:2048:8:4:0:2 32:2048:8:4:0:4 32:8192:8:4:0:2 128:2048:4:4:0:2 256:2048:2:4:0:2"}
export FL_NCU=${FL_NCU:-"32:8:4:0 128:4:4:0"}
export FL_REPS=${FL_REPS:-15} FL_GATE=${FL_GATE:-1}
git rev-parse HEAD > "$O/commit.txt" 2>&1
echo "${FL_NVCC--DDFILL_V1}" > "$O/nvcc-flags.txt"
export DFILL_COINS; DFILL_COINS=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
echo "$DFILL_COINS" > "$O/coins.txt"
nvcc --version | tail -2 > "$O/nvcc.txt"
# shellcheck disable=SC2086
nvcc -O3 -std=c++17 ${FL_NVCC--DDFILL_V1} -gencode arch=compute_120a,code=sm_120a -Xptxas -v -o "$B/dcap" $D/dcap.cu -lcublasLt -lnvidia-ml -ldl \
  > "$O/build.log" 2>&1 || { grep -E "error" "$O/build.log" | head -40; tail -20 "$O/build.log"; exit 4; }
cuobjdump -sass "$B/dcap" | awk '/Function : /{p = /flin_k/} p' > "$O/sass-flin.txt" 2>&1
echo "flin_k SASS: $(grep -c IMMA "$O/sass-flin.txt") IMMA, $(grep -cE 'LDL|STL' "$O/sass-flin.txt") local memory ops"
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  "$B/dcap" fgate > "$O/fgate.jsonl" 2>> "$O/errors.txt"; g=$?
  echo "fgate rc $g" >> "$O/rc.txt"
  [ "$g" -eq 0 ] || exit 5
  if [ "$FL_GATE" = 1 ]; then
    "$B/dcap" gate > "$O/gate.jsonl" 2>> "$O/errors.txt"; g=$?
    echo "gate rc $g" >> "$O/rc.txt"
    [ "$g" -eq 0 ] || exit 5
  fi
  for c in ${FL_FLIN/none/}; do
    IFS=: read -r m fw nc thr pdl <<< "$c"
    FLIN_PDL=${pdl:-1} "$B/dcap" flin "$m" "$fw" "$nc" "$thr" "$FL_REPS" >> "$O/flin.jsonl" 2>> "$O/errors.txt" || echo "rc $? for flin $c" >> "$O/errors.txt"
  done
  q
  for c in ${FL_NCU/none/}; do
    IFS=: read -r m fw nc thr <<< "$c"
    # the first 64 launches are qkv with the filler warps (32 layers), then qkv alone (32 layers)
    ncu --clock-control none -k regex:flin_k --launch-count 64 \
      --metrics gpu__time_duration.sum,dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,lts__t_bytes.sum \
      --csv "$B/dcap" flin "$m" "$fw" "$nc" "$thr" 1 > "$O/ncu-flin-$m-$fw-$nc-$thr.csv" 2>&1
    echo "ncu flin $c rc $?" >> "$O/rc.txt"
  done
  for c in ${FL_NCU_STEP:-}; do
    IFS=: read -r b l fw nc thr afw anc aps alz <<< "$c"
    n="$O/ncu-fstep-$b-$l-$fw-$afw-${anc:-4}-${aps:-0}-${alz:-0}"
    # fstep first runs the attention blocks of layer 0 with their filler warps, then its first linear with filler warps
    nf() { FATT_NC=${anc:-4} FATT_PERSIST=${aps:-0} FATT_LAZY=${alz:-0} ncu --clock-control none -k regex:"$1" --launch-count "$2" \
      --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \
      "${@:3}" "$B/dcap" fstep "$b" "$l" "$fw" "$nc" "$thr" "$afw" 0 1; }
    if [ -z "${FL_NCU_SECTIONS:-}" ]; then
      nf "flin_k|fqk_k|fpv_k" 3 --csv > "$n.csv" 2>&1
    else
      # the attention blocks only: flin_k fails to launch under the full sections
      s=(); for x in $FL_NCU_SECTIONS; do s+=(--section "$x"); done
      nf "fqk_k|fpv_k" 2 "${s[@]}" -o "$n" > "$n.log" 2>&1
      ncu --import "$n.ncu-rep" --csv --page details > "$n.csv" 2>&1 &&
        ncu --import "$n.ncu-rep" --csv --page raw > "$n-raw.csv" 2>&1 &&
        ncu --import "$n.ncu-rep" --csv --page source --print-source sass > "$n-sass.csv" 2>&1
    fi
    echo "ncu fstep $c rc $?" >> "$O/rc.txt"
  done
  q
  for c in $FL_STEP; do
    IFS=: read -r b l fw nc thr afw pdl fsm lt fp anc aps alz lm st <<< "$c"
    FLIN_PDL=${pdl:-1} FLIN_LT=${lt:-0} FATT_PDL=${fp:-0} FATT_NC=${anc:-4} FATT_PERSIST=${aps:-0} FATT_LAZY=${alz:-0} FLIN_LM=${lm:-0} \
      FLIN_STOP=${st:-0} \
      "$B/dcap" fstep "$b" "$l" "$fw" "$nc" "$thr" "${afw:-0}" "${fsm:-0}" "$FL_REPS" \
      >> "$O/fstep.jsonl" 2>> "$O/errors.txt" ||
      echo "rc $? for fstep $c" >> "$O/errors.txt"
    q
  done
'
r=$?
echo "lease rc $r" >> "$O/rc.txt"
cat "$O/fgate.jsonl" "$O/rc.txt" 2>/dev/null | cut -c1-260
cut -c1-400 "$O/flin.jsonl" 2>/dev/null
grep -v "^device" "$O/errors.txt" 2>/dev/null | tail -5
exit "$r"
