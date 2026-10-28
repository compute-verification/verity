#!/usr/bin/env bash
# sq.cu on one GPU (cwd = the source tree): squarings mod 2^$BITS - $C (default 2^512 - 569) for every (FOLD, ILP) in
# $FORMS, each gated against Python's pow (512 words at K = 1, 777 and 1,000) before it is timed, with the SASS opcode
# mix of every kernel, ncu's pipe and issue counts and its stall reasons; then each pipe probe in $PROBES (P:W:A; set
# it empty for none), its digests checked against the host's before it is timed. Timed and profiled in blocks of $TPB
# threads (default 128, the most the kernels take); the gates use 128. Results in $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' --env GPU_LEASE_WHO=ra-squaring-ceiling -- taskset -c 48-95 bash benchmarks/rate_squaring/sq.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export D=benchmarks/rate_squaring
export B; B=$(mktemp -d /tmp/rate-squaring.XXXXXX); trap 'rm -rf "$B"' EXIT
export FORMS=${FORMS:-"0:1 0:2 1:1 1:2 1:4 2:1 2:2 2:4 3:1 3:2 4:1 4:2"}
export PROBES=${PROBES-"0:16:0 0:16:16 0:16:32 1:16:0 1:16:16"}
BITS=${BITS:-512}; C=${C:-569}; export TPB=${TPB:-128}
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v -DSQ_BITS="$BITS" -DSQ_C="$C" -o "$B/sq" $D/sq.cu \
  > "$O/build.log" 2>&1 \
  || { tail -40 "$O/build.log"; exit 4; }
nvcc --version > "$O/nvcc.txt"
cuobjdump -sass "$B/sq" > "$B/sq.sass" 2>&1
python3 $D/sass_mix.py "$B/sq.sass" > "$O/sass-mix.json"
LEASE=(gpu-lease 1 --wait --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  "$B/sq" clock > "$O/clock-before.json" 2>&1
  for fi in $FORMS; do
    f=${fi%%:*}; i=${fi##*:}; ok=1
    for k in 1 777 1000; do
      "$B/sq" gate $f $i $k 512 > "$B/gate-$f-$i-$k.txt" 2>> "$O/errors.txt" || echo "gate $f $i $k rc $?" >> "$O/errors.txt"
      python3 $D/sq_gate.py < "$B/gate-$f-$i-$k.txt" > "$O/gate-f$f-i$i-k$k.json" 2>&1 || ok=0
    done
    if [ $ok = 1 ]; then
      for w in 2 4; do
        "$B/sq" time $f $i 4000 $TPB $w 5 >> "$O/time.jsonl" 2>> "$O/errors.txt" || echo "time $f $i $w rc $?" >> "$O/errors.txt"
      done
    else
      echo "fold $f ilp $i failed its gate: not timed" >> "$O/errors.txt"
    fi
  done
  for pwa in $PROBES; do
    IFS=: read -r p w a <<< "$pwa"
    "$B/sq" probe $p $w $a 20000 4 >> "$O/probe.jsonl" 2>> "$O/errors.txt" || echo "probe $pwa rc $?" >> "$O/errors.txt"
  done
  "$B/sq" clock > "$O/clock-after.json" 2>&1
  nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  M=smsp__inst_executed.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__inst_executed_pipe_fmalite.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__warps_active.avg.per_cycle_active,sm__cycles_elapsed.avg
  for fi in $FORMS; do
    f=${fi%%:*}; i=${fi##*:}
    ncu --metrics $M --csv -k regex:chain -c 1 "$B/sq" time $f $i 400 $TPB 2 1 > "$O/ncu-f$f-i$i.csv" 2>&1 || echo "ncu $f $i rc $?" >> "$O/errors.txt"
    ncu --section WarpStateStats --csv --page raw -k regex:chain -c 1 "$B/sq" time $f $i 400 $TPB 2 1 \
      > "$O/ncu-stalls-f$f-i$i.csv" 2>&1 || echo "ncu stalls $f $i rc $?" >> "$O/errors.txt"
  done
  for pwa in $PROBES; do
    IFS=: read -r p w a <<< "$pwa"
    ncu --metrics $M --csv -k regex:probe --launch-skip 1 -c 1 "$B/sq" probe $p $w $a 2000 4 > "$O/ncu-probe-$p-$w-$a.csv" 2>&1 || echo "ncu probe $pwa rc $?" >> "$O/errors.txt"
  done'
