#!/usr/bin/env bash
# kv.cu's encode throughput against the stated squaring ceiling on one GPU (cwd = the source tree): kv.cu built at the
# stack's codec (p1024, masked, FOLD 6, 32 query heads), its ptxas report and its chain kernels' SASS opcode mix; sq.cu's
# FOLD 6, 11 and 12 (one, two and three levels of Karatsuba) at 2^1024 - 105, gated and timed, the same GPU's squaring
# reference; encode_rate.py (ARGS passes through); and Nsight Compute's issue and pipe counts and stall reasons on one
# launch each of the chains (square, encode, products, fold; NCU=0 skips them). Results in $RESEARCH_RUN_DIR/out. Under
# the queue (QUEUED=1) the GPU is leased around the whole job and gpu-lease is skipped.
#
#   RESEARCH_RUN_DIR=DIR QUEUED=1 bash benchmarks/word_codec/encode_rate.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export WC=benchmarks/word_codec RS=benchmarks/rate_squaring PY
export B; B=$(mktemp -d /tmp/encode-rate.XXXXXX); trap 'rm -rf "$B"' EXIT
NV=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas -v)
nvcc "${NV[@]}" -Xcompiler -fPIC -shared -DKV_FIELD=1024 -DKV_MASK=1 -DKV_FOLD6=1 -DKV_QHEADS=32 -o "$B/kv.so" \
  $WC/kv.cu > "$O/kv.ptxas" 2>&1 || { tail -30 "$O/kv.ptxas"; exit 4; }
nvcc "${NV[@]}" -DSQ_BITS=1024 -DSQ_C=105 -o "$B/sq" $RS/sq.cu > "$O/sq.ptxas" 2>&1 || { tail -30 "$O/sq.ptxas"; exit 4; }
cuobjdump -sass "$B/kv.so" | awk '/Function : /{p = ($0 ~ /chain_kernel/)} p' > "$O/kv-chain.sass"
cuobjdump -sass "$B/sq" | awk '/Function : /{p = ($0 ~ /chainILi6ELi1E/)} p' > "$O/sq-fold6.sass"
python3 $RS/sass_mix.py "$O/kv-chain.sass" > "$O/kv-sass-mix.json"
python3 $RS/sass_mix.py "$O/sq-fold6.sass" > "$O/sq-sass-mix.json"
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
[ "${QUEUED:-0}" = 1 ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader >> "$O/gpu.txt" 2>&1
  for f in 6 11 12; do
    ok=1
    for k in 1 777 1000; do
      "$B/sq" gate $f 1 $k 512 > "$B/gate-$f-$k.txt" 2>> "$O/errors.txt" || ok=0
      python3 $RS/sq_gate.py < "$B/gate-$f-$k.txt" > "$O/sq-gate-f$f-k$k.json" 2>&1 || ok=0
    done
    [ $ok = 1 ] || { echo "sq FOLD $f failed its gate: not timed" >> "$O/errors.txt"; continue; }
    for w in 2 4; do "$B/sq" time $f 1 4000 128 $w 5 >> "$O/sq-time.jsonl" 2>> "$O/errors.txt"; done
  done
  "$PY" $WC/encode_rate.py --lib "p1024=$B/kv.so" --out "$O" ${ARGS:-} || exit 5
  [ "${NCU:-1}" = 0 ] && exit 0
  M=smsp__inst_executed.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__inst_executed_pipe_fmalite.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__warps_active.avg.per_cycle_active,sm__cycles_elapsed.avg
  ncu --clock-control none --metrics $M --csv -k regex:chain_kernel "$PY" $WC/encode_rate.py --lib "p1024=$B/kv.so" \
    --out "$B/ncu" --ncu > "$O/ncu-kv.csv" 2>&1 || echo "ncu rc $?" >> "$O/errors.txt"
  ncu --clock-control none --section WarpStateStats --csv --page raw -k regex:chain_kernel "$PY" $WC/encode_rate.py \
    --lib "p1024=$B/kv.so" --out "$B/ncu" --ncu > "$O/ncu-kv-stalls.csv" 2>&1 || echo "ncu stalls rc $?" >> "$O/errors.txt"
  ncu --clock-control none --metrics $M --csv -k regex:chain -c 1 "$B/sq" time 6 1 400 128 2 1 > "$O/ncu-sq.csv" 2>&1 \
    || echo "ncu sq rc $?" >> "$O/errors.txt"
  exit 0'
