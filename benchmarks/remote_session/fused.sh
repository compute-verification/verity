#!/usr/bin/env bash
# fused.cu's one-GPU screen (cwd = the source tree): builds kv.so at the session's codec (run.sh's flags), its FOLD 6
# twin, fused.cu at both, and session.so; then under one gpu-lease runs fused_bench.py's gate and timing, its ops under
# Nsight Compute (HBM bytes), and the pass beside the engine stand-in (fused_bench.py --mode serve, fused_beside.py).
# Results in $RESEARCH_RUN_DIR/out: fused_bench.json, ncu.json, beside-CFG/beside.json, their logs and the builds' ptxas.
# BESIDE (default nomps) runs the beside section once a configuration: nomps, the two processes time-sliced; mps, both
# clients of a private MPS daemon on the leased GPU (pipe and logs in this run's scratch, started and quit inside the
# lease); mpsE.P, the same with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE E for the engine stand-in and P for the store. Each
# configuration's passes are checked against the configuration's before it too (fused_beside.py --reference).
# SESSION_PATCH builds session.so from session.cu with that patch applied (pass_sm.patch: kernels' %smid placement), and
# a configuration's suffix @Q.D (mps@1.4) runs the store with PASS_SM=Q/D, its pass leaving the SMs whose %smid % D < Q.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --declared-output 'out/*' \
#     --env GPU_LEASE_WHO=memory-accounting-fused -- bash benchmarks/remote_session/fused.sh
#
# SECTIONS (default gate,time,beside,ncu, in that order; beside and ncu after a passed gate, which `gate` writes), MS and ARMS (fused_bench.py's
# --ms and --arms), PHASES (fused_beside.py's), NCU_MS, MAXMIN (the lease's, default 60), LEASE_ARGS (default --wait
# --preemptible; without --wait the lease takes a free GPU at once or the run ends), VPY (vLLM's venv: the GEMM
# vLLM runs), PY (the prover's torch venv), ENGINE_CPUS and PROVER_CPUS (taskset lists, default unpinned), GPU_LEASE
# (default the gpu-lease on PATH, else pouw's). Host threads default to one (OMP_NUM_THREADS, MKL_NUM_THREADS).
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
export PY=${PY:-$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python} VPY=${VPY:-/workspace/jobs/venv312/bin/python}
[ -x "$PY" ] && [ -x "$VPY" ] || { echo "no venv at $PY or $VPY"; exit 3; }
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
GPU_LEASE=${GPU_LEASE:-$(command -v gpu-lease || echo /workspace/pouw/infra/bin/gpu-lease)}
export B=$(mktemp -d /tmp/fused.XXXXXX)
trap '[ -e "$B/mps/pipe/control" ] && echo quit | CUDA_MPS_PIPE_DIRECTORY=$B/mps/pipe nvidia-cuda-mps-control; rm -rf "$B"' EXIT
export RS=benchmarks/remote_session SECTIONS=${SECTIONS:-gate,time,beside,ncu} BESIDE=${BESIDE:-nomps}
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
KV=(-DKV_FIELD=1024 -DKV_MASK=1)
P=()
nvcc "${FLAGS[@]}" "${KV[@]}" -o "$B/kv.so" benchmarks/word_codec/kv.cu > "$O/kv.ptxas" 2>&1 & P+=($!)
nvcc "${FLAGS[@]}" "${KV[@]}" -DKV_FOLD6=1 -o "$B/kv6.so" benchmarks/word_codec/kv.cu > "$O/kv6.ptxas" 2>&1 & P+=($!)
nvcc "${FLAGS[@]}" "${KV[@]}" -o "$B/fused.so" $RS/fused.cu > "$O/fused.ptxas" 2>&1 & P+=($!)
nvcc "${FLAGS[@]}" "${KV[@]}" -DKV_FOLD6=1 -o "$B/fused6.so" $RS/fused.cu > "$O/fused6.ptxas" 2>&1 & P+=($!)
SESSION=$RS/session.cu
if [ -n "${SESSION_PATCH:-}" ]; then
  patch -s -o "$B/session.cu" "$SESSION" < "$SESSION_PATCH" || exit 4
  SESSION=$B/session.cu
fi
nvcc "${FLAGS[@]}" -I $RS -o "$B/session.so" "$SESSION" > "$O/session.ptxas" 2>&1 & P+=($!)
for p in "${P[@]}"; do wait "$p" || { tail -30 "$O"/*.ptxas; exit 4; }; done
sha256sum "$B"/*.so > "$O/libs.sha256"
# shellcheck disable=SC2016
"$GPU_LEASE" 1 ${LEASE_ARGS---wait --preemptible} --max-min "${MAXMIN:-60}" -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  LIBS=(--lib "kv=$B/kv.so" --lib "kv6=$B/kv6.so" --lib "fused=$B/fused.so" --lib "fused6=$B/fused6.so")
  EPIN=(); [ -n "${ENGINE_CPUS:-}" ] && EPIN=(taskset -c "$ENGINE_CPUS")
  PPIN=(); [ -n "${PROVER_CPUS:-}" ] && PPIN=(taskset -c "$PROVER_CPUS")
  BENCH=("${EPIN[@]}" "$VPY" $RS/fused_bench.py "${LIBS[@]}" --out "$O" ${MS:+--ms "$MS"} ${ARMS:+--arms "$ARMS"})
  M=$(tr , "\n" <<< "$SECTIONS" | grep -E "^(gate|time)$" | paste -sd,)
  if [ -n "$M" ]; then "${BENCH[@]}" --mode "$M" || exit $?; fi
  "$VPY" -c "import json, sys; sys.exit(not json.load(open(\"$O/fused_bench.json\"))[\"gate\"][\"ok\"])" ||
    { echo "no passed gate in $O/fused_bench.json: nothing timed"; exit 4; }
  R=0
  # Every client names this pipe directory, so with no daemon in it a client runs without MPS, whatever else runs on
  # the node.
  export CUDA_MPS_PIPE_DIRECTORY=$B/mps/pipe CUDA_MPS_LOG_DIRECTORY=$B/mps/log
  # An MPS client names its GPU within the daemon visible set, where the lease index is invalid unless it is 0 (the
  # client fails with CUDA error 100); the UUID names the same GPU everywhere.
  export CUDA_VISIBLE_DEVICES=$GPU_LEASE_UUID
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  # The daemon leaves its control socket behind when it quits, so it is found by its environment instead.
  mps_pids() {
    for p in /proc/[0-9]*; do
      case $(cat "$p/comm" 2> /dev/null) in nvidia-cuda-mps*)
        tr "\0" "\n" < "$p/environ" 2> /dev/null | grep -qx "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY" &&
          echo "${p#/proc/}" ;;
      esac
    done
  }
  mps_stop() {
    [ -n "$(mps_pids)" ] || return 0
    echo quit | nvidia-cuda-mps-control
    for _ in $(seq 150); do [ -n "$(mps_pids)" ] || return 0; sleep 0.2; done
    echo "the MPS daemon of $CUDA_MPS_PIPE_DIRECTORY did not quit: killing $(mps_pids)"; kill $(mps_pids); return 1
  }
  trap mps_stop EXIT
  trap "mps_stop; exit 143" TERM
  if [[ ,$SECTIONS, == *,beside,* ]]; then
    REF=
    for CFG in ${BESIDE//,/ }; do
      SM=; C=$CFG
      case $CFG in *@*) SM=${CFG#*@}; SM=${SM/./\/}; C=${CFG%@*} ;; esac
      case $C in
        nomps) mps_stop; EP=; PP= ;;
        mps*) [ -n "$(mps_pids)" ] || nvidia-cuda-mps-control -d || { R=6; break; }
              P=${C#mps}; EP=${P%.*}; PP=${P#*.} ;;
        *) echo "BESIDE: $CFG is not nomps, mps or mpsE.P, each optionally @Q.D"; R=6; break ;;
      esac
      D=$O/beside-$CFG; mkdir -p "$D" "$B/serve-$CFG"; cp "$O/fused_bench.json" "$D/"
      env -u PYTORCH_CUDA_ALLOC_CONF ${EP:+CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$EP} "${BENCH[@]}" --out "$D" \
        --mode serve --serve-dir "$B/serve-$CFG" > "$D/serve.log" 2>&1 & S=$!
      env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PP:+CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PP} \
        ${SM:+PASS_SM=$SM} "${PPIN[@]}" \
        "$PY" $RS/fused_beside.py --dir "$B/serve-$CFG" --lib "$B/session.so" --kv "$B/kv.so" --out "$D" --wait-s 300 \
        ${PHASES:+--phases "$PHASES"} ${REF:+--reference "$REF"} --engine-pid $S > "$D/beside.log" 2>&1; RC=$?
      tail -3 "$D/beside.log"
      [ $RC = 0 ] && REF=$D/beside.json || { kill $S 2> /dev/null; R=$RC; }
      wait $S
      cp "$CUDA_MPS_LOG_DIRECTORY"/*.log "$D/" 2> /dev/null
    done
    mps_stop || R=6
    cp "$CUDA_MPS_LOG_DIRECTORY"/*.log "$O/" 2> /dev/null
  fi
  if [[ ,$SECTIONS, == *,ncu,* ]]; then
    ncu --target-processes all --clock-control none --cache-control all --print-units base --csv \
      --metrics dram__bytes.sum,gpu__time_duration.sum --log-file "$O/ncu.csv" \
      "${BENCH[@]}" --mode ncu --ms "${NCU_MS:-1,8,32,64}" > "$O/ncu.log" 2>&1 &&
      "$VPY" $RS/fused_bench.py --mode ncu-report --ncu-csv "$O/ncu.csv" --out "$O" > "$O/ncu-report.log" 2>&1 ||
      { tail -n 20 "$O/ncu.log" "$O/ncu-report.log"; [ $R = 0 ] && R=5; }
  fi
  exit $R
'
