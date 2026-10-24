#!/usr/bin/env bash
# fused.cu's one-GPU screen (cwd = the source tree): builds kv.so at the session's codec (run.sh's flags), its FOLD 6
# twin, fused.cu at both, and session.so; then under one gpu-lease runs fused_bench.py's gate and timing, its ops under
# Nsight Compute (HBM bytes), and the pass beside the engine stand-in (fused_bench.py --mode serve, fused_beside.py).
# Results in $RESEARCH_RUN_DIR/out: fused_bench.json, ncu.json, beside.json, their logs and the builds' ptxas.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --declared-output 'out/*' \
#     --env GPU_LEASE_WHO=memory-accounting-fused -- bash benchmarks/remote_session/fused.sh
#
# SECTIONS (default gate,time,ncu,beside; each after a passed gate, which `gate` writes), MS and ARMS (fused_bench.py's
# --ms and --arms), PHASES (fused_beside.py's), NCU_MS, MAXMIN (the lease's, default 60), VPY (vLLM's venv: the GEMM
# vLLM runs), PY (the prover's torch venv), ENGINE_CPUS and PROVER_CPUS (taskset lists, default unpinned).
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
export PY=${PY:-$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python} VPY=${VPY:-/workspace/jobs/venv312/bin/python}
[ -x "$PY" ] && [ -x "$VPY" ] || { echo "no venv at $PY or $VPY"; exit 3; }
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export B=$(mktemp -d /tmp/fused.XXXXXX); trap 'rm -rf "$B"' EXIT
export RS=benchmarks/remote_session SECTIONS=${SECTIONS:-gate,time,ncu,beside}
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
KV=(-DKV_FIELD=1024 -DKV_MASK=1)
P=()
nvcc "${FLAGS[@]}" "${KV[@]}" -o "$B/kv.so" benchmarks/word_codec/kv.cu > "$O/kv.ptxas" 2>&1 & P+=($!)
nvcc "${FLAGS[@]}" "${KV[@]}" -DKV_FOLD6=1 -o "$B/kv6.so" benchmarks/word_codec/kv.cu > "$O/kv6.ptxas" 2>&1 & P+=($!)
nvcc "${FLAGS[@]}" "${KV[@]}" -o "$B/fused.so" $RS/fused.cu > "$O/fused.ptxas" 2>&1 & P+=($!)
nvcc "${FLAGS[@]}" "${KV[@]}" -DKV_FOLD6=1 -o "$B/fused6.so" $RS/fused.cu > "$O/fused6.ptxas" 2>&1 & P+=($!)
nvcc "${FLAGS[@]}" -o "$B/session.so" $RS/session.cu > "$O/session.ptxas" 2>&1 & P+=($!)
for p in "${P[@]}"; do wait "$p" || { tail -30 "$O"/*.ptxas; exit 4; }; done
sha256sum "$B"/*.so > "$O/libs.sha256"
# shellcheck disable=SC2016
/workspace/pouw/infra/bin/gpu-lease 1 --wait --preemptible --max-min "${MAXMIN:-60}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  LIBS=(--lib "kv=$B/kv.so" --lib "kv6=$B/kv6.so" --lib "fused=$B/fused.so" --lib "fused6=$B/fused6.so")
  EPIN=(); [ -n "${ENGINE_CPUS:-}" ] && EPIN=(taskset -c "$ENGINE_CPUS")
  PPIN=(); [ -n "${PROVER_CPUS:-}" ] && PPIN=(taskset -c "$PROVER_CPUS")
  BENCH=("${EPIN[@]}" "$VPY" $RS/fused_bench.py "${LIBS[@]}" --out "$O" ${MS:+--ms "$MS"} ${ARMS:+--arms "$ARMS"})
  M=$(tr , "\n" <<< "$SECTIONS" | grep -E "^(gate|time)$" | paste -sd,)
  if [ -n "$M" ]; then "${BENCH[@]}" --mode "$M" || exit $?; fi
  if [[ ,$SECTIONS, == *,ncu,* ]]; then
    ncu --target-processes all --clock-control none --cache-control all --print-units base --csv \
      --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum --log-file "$O/ncu.csv" \
      "${BENCH[@]}" --mode ncu --ms "${NCU_MS:-1,8,32,64}" > "$O/ncu.log" 2>&1 || { tail -20 "$O/ncu.log"; exit 5; }
    "$VPY" $RS/fused_bench.py --mode ncu-report --ncu-csv "$O/ncu.csv" --out "$O" > /dev/null || exit 5
  fi
  if [[ ,$SECTIONS, == *,beside,* ]]; then
    mkdir -p "$B/serve"
    env -u PYTORCH_CUDA_ALLOC_CONF "${BENCH[@]}" --mode serve --serve-dir "$B/serve" > "$O/serve.log" 2>&1 & S=$!
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PPIN[@]}" "$PY" $RS/fused_beside.py --dir "$B/serve" \
      --lib "$B/session.so" --kv "$B/kv.so" --out "$O" ${PHASES:+--phases "$PHASES"} > "$O/beside.log" 2>&1; R=$?
    tail -3 "$O/beside.log"
    [ $R = 0 ] || kill $S 2> /dev/null
    wait $S
    exit $R
  fi
'
