#!/usr/bin/env bash
# The honest joint run's per-resource utilization on one RTX PRO 6000 (sm_120), CMP-12 W9: Pearl-C's call loop alone,
# the bandwidth audit's prover passes alone, and both on one GPU (one process on two streams, `joint*`; two processes,
# `procs*`), each timed bare, then under nsys's GPU-metric sampler (`nsys:*`), and each protocol's kernels counted by
# ncu (`ncu:*`). Built off the lease (Pearl-C's ship by its build.sh, session.cu as remote_session/run.sh builds it);
# the GPU part under one gpu-lease 1; the reports reduced after it (nsys_summary.py, ncu_summary.py). cwd = the tree.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone --env GPU_LEASE_WHO=<id> \
#     --send benchmarks/resource_accounting/joint_util/run.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run.sh"'
#
# PHASES (comma-separated): pouw<M>, audit, joint<M>, procs<M>, gemm, gemmjoint, each optionally nsys:, nsysk: (nsys
# with each graph's kernels traced) or ncu: prefixed; M is Pearl-C's m (n = k = 8192). Suffixes: _h9 the audit at
# h0 = 9 (the pass's unfused tree path), _p0 its side stream at priority 0, _q Pearl-C's stream at the greatest priority,
# _bf16 the GEMM in BF16, _d<P> the GEMM at P% duty. SECONDS_RATE, SECONDS_NSYS: each window; WEIGHTS: Pearl-C's distinct weights; GIB: the store.
set -uo pipefail
ulimit -c 0
OUT=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$OUT"
J=benchmarks/resource_accounting/joint_util
CUDA=/usr/local/cuda-13.0
PY=${PY:-/workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python}
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
NSYS=${NSYS:-$(command -v nsys || echo "$CUDA/bin/nsys")}
D=$(mktemp -d /tmp/joint-util.XXXXXX); trap 'rm -rf "$D"' EXIT
export OUT J CUDA PY NSYS D
export PHASES=${PHASES:-pouw32,pouw128,audit,joint32,joint128,procs128,nsys:pouw32,nsys:pouw128,nsys:audit,nsys:joint32,nsys:joint128,ncu:pouw32,ncu:pouw128,ncu:audit}
export SECONDS_RATE=${SECONDS_RATE:-20} SECONDS_NSYS=${SECONDS_NSYS:-8} WEIGHTS=${WEIGHTS:-4} GIB=${GIB:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export METRICS=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,lts__cycles_elapsed.avg,\
lts__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg,dram__cycles_elapsed.avg.per_second,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed.avg.pct_of_peak_sustained_elapsed,sm__inst_executed.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.sum,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_aluheavy_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_xu.sum,\
sm__inst_executed_pipe_uniform.sum,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.sum,\
lts__t_sectors_op_red.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_op_red.sum,lts__t_requests_op_red.sum,\
lts__t_sectors_op_atom.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_op_atom.sum,\
lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.avg.pct_of_peak_sustained_elapsed,dram__bytes_op_read.sum,\
dram__bytes_op_write.sum,dram__bytes.sum.peak_sustained,lts__t_sectors.sum.peak_sustained

# A. off the lease: Pearl-C's ship (cubin, SASS gate, fixture) and session.so
nice -n 19 bash benchmarks/pouw/pearl_c_sm120/build.sh "$D/ship" > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 4; }
cp "$D/ship/toolchain.txt" "$D/ship/source.txt" "$D/ship/sass.json" "$D/ship/MANIFEST.sha256" "$OUT/"
nice -n 19 "$CUDA/bin/nvcc" -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v \
  -o "$D/session.so" benchmarks/remote_session/session.cu > "$OUT/session.ptxas" 2>&1 || { tail -40 "$OUT/session.ptxas"; exit 4; }
sha256sum "$D/session.so" "$D/ship/pearl_c_sm120.cubin" > "$OUT/libs.sha256"
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true

# B. the lease: every phase in order, each logged; a failed phase is recorded and the next runs
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-25}" -- bash -c '
  export PEARLC_EXPECT_UUID=$GPU_LEASE_UUID
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw,power.limit,\
enforced.power.limit,power.max_limit,temperature.gpu,memory.used,memory.total --format=csv -i "$GPU_LEASE_UUID" > "$OUT/gpu.txt" 2>&1
  L=("$PY" "$J/loop.py" --ship "$D/ship" --lib "$D/session.so" --weights "$WEIGHTS" --gib "$GIB")
  args() {  # a phase body (pouw32, audit, joint128, audit_h9, joint128_p0, gemmjoint_h9, ...): loop.py arguments
    local b=$1 x=
    case $b in *_h9*) x="--kept-from 9" ;; esac
    case $b in *_p0*) x="$x --side-priority 0" ;; esac
    case $b in *_q*) x="$x --pouw-priority -99" ;; esac
    case $b in *_bf16*) x="$x --gemm-dtype bf16" ;; esac
    case $b in *_d[0-9]*) x="$x --gemm-duty $(echo "$b" | sed -n "s/.*_d\([0-9]*\).*/\1/p" | awk "{print \$1 / 100}")" ;; esac
    b=${b%%_*}
    case $b in
      gemmjoint) echo --mode gemmjoint $x ;;
      gemm) echo --mode gemm $x ;;
      pouw*) echo --mode pouw --shape "${b#pouw}x8192x8192" $x ;;
      joint*) echo --mode joint --shape "${b#joint}x8192x8192" $x ;;
      audit) echo --mode audit $x ;;
    esac
  }
  T=(timeout -k 10)
  for ph in ${PHASES//,/ }; do
    name=${ph//:/_}; body=${ph#*:}; t0=$(date +%s)
    case $ph in
      ncu:*)
        units=$(case ${body%%_*} in audit | gemm) echo 2 ;; *) echo 4 ;; esac)
        "${T[@]}" 900 "$CUDA/bin/ncu" --profile-from-start off --clock-control none --cache-control none --replay-mode kernel \
          --metrics "$METRICS" -f -o "$D/$name" "${L[@]}" $(args "$body") --no-graph --count "$units" --out "$OUT/$name.json" \
          > "$OUT/$name.log" 2>&1
        rc=$?; echo "$units" > "$D/$name.units" ;;
      nsys:* | nsysk:*)
        g=$([ "${ph%%:*}" = nsysk ] && echo --cuda-graph-trace=node)
        "${T[@]}" $((SECONDS_NSYS + 240)) "$NSYS" profile --capture-range=cudaProfilerApi --capture-range-end=stop \
          --gpu-metrics-devices=cuda-visible --gpu-metrics-set=gb20x --gpu-metrics-frequency=10000 --trace=cuda $g \
          --sample=none --cpuctxsw=none --force-overwrite=true -o "$D/$name" "${L[@]}" $(args "$body") \
          --seconds "$SECONDS_NSYS" --watchdog 40 --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1
        rc=$? ;;
      procs*)
        m=${body#procs}; sfx=${m#"${m%%_*}"}; m=${m%%_*}; rm -f "$D"/r1 "$D"/r2 "$D"/go
        "${T[@]}" $((SECONDS_RATE + 180)) "${L[@]}" --mode pouw --shape "${m}x8192x8192" --seconds "$SECONDS_RATE" \
          --watchdog 40 --ready "$D/r1" --go "$D/go" --out "$OUT/${name}_pouw.json" > "$OUT/${name}_pouw.log" 2>&1 & p1=$!
        "${T[@]}" $((SECONDS_RATE + 180)) "${L[@]}" $(args "audit$sfx") --seconds "$SECONDS_RATE" --watchdog 40 \
          --ready "$D/r2" --go "$D/go" --out "$OUT/${name}_audit.json" > "$OUT/${name}_audit.log" 2>&1 & p2=$!
        while { [ ! -e "$D/r1" ] || [ ! -e "$D/r2" ]; } && kill -0 $p1 2> /dev/null && kill -0 $p2 2> /dev/null; do sleep 0.2; done
        touch "$D/go"; wait $p1; r1=$?; wait $p2; rc=$((r1 | $?)) ;;
      *)
        "${T[@]}" $((SECONDS_RATE + 180)) "${L[@]}" $(args "$ph") --seconds "$SECONDS_RATE" --watchdog 40 \
          --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1
        rc=$? ;;
    esac
    echo "$ph rc=$rc $(( $(date +%s) - t0 ))s" | tee -a "$OUT/phases.txt"
    tail -2 "$OUT/$name.log" 2> /dev/null | cut -c1-600
  done
  nvidia-smi --query-gpu=index,uuid,clocks.sm,power.draw,temperature.gpu --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$OUT/gpu.txt" 2>&1
'
echo "lease rc=$?" | tee -a "$OUT/phases.txt"

# C. off the lease: the reports reduced
for r in "$D"/nsys*.nsys-rep; do
  [ -e "$r" ] || continue
  n=$(basename "$r" .nsys-rep)
  nice -n 19 "$NSYS" export --type sqlite --force-overwrite=true -o "$D/$n.sqlite" "$r" > /dev/null 2>&1 \
    && python3 "$J/nsys_summary.py" "$D/$n.sqlite" > "$OUT/$n.gpu.json" || echo "export of $n failed" | tee -a "$OUT/phases.txt"
done
for r in "$D"/ncu_*.ncu-rep; do
  [ -e "$r" ] || continue
  n=$(basename "$r" .ncu-rep)
  "$CUDA/bin/ncu" -i "$r" --csv --page raw --print-units base > "$OUT/$n.csv" 2> "$OUT/$n.export.log" \
    && python3 "$J/ncu_summary.py" "$OUT/$n.csv" "$(cat "$D/$n.units")" > "$OUT/$n.sum.json" \
    || echo "reduction of $n failed" | tee -a "$OUT/phases.txt"
done
ls -la "$OUT" | tail -60
