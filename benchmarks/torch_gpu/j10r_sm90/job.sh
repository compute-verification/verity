#!/usr/bin/env bash
# HASH-22's jobs (cwd = the source tree's root):
#   build  (a CPU node) the fallback, j10r_ref.cu, at sm_90a: $OUT/lib/sm_90a/j10r_ref.so, ptxas's report, BUILD.txt
#   cpu    build, then fixtures.py expected into $OUT/fixtures (uv sync first; FIX_ARGS, JOBS default 30)
#   gpu    (vy-dc-h200-3) check.py, then bench.py, on the files sent (LIBS, default $RESEARCH_RUN_DIR/inputs):
#          expected.npz and fixtures.json, j10r_ref.so as the fallback, every h1u_*.so as an h1u build (and its _sm
#          launch), s8_simple.py (#4472's) as the bench's plain arm. Fixtures not sent are computed here and the fallback
#          not sent is built here (BUILD_REF=0: not), both before the lease, as are the operands, in a scratch directory
#          removed at exit.
#
#   research run --on vy-nebius-cpu-4 --as kernels-kp2dec --project verity --source . --cwd source -- \
#     env CHECK_SLOTS=/workspace/research/locks/kernels/slots python3 tools/verity/check/slot.py --by kernels -- \
#     bash benchmarks/torch_gpu/j10r_sm90/job.sh cpu
#   research run --on vy-dc-h200-3 --project verity --campaign e2e-demo --source . --cwd source --timeout 60m \
#     --env GPU_LEASE_WHO=kernels-kp2dec --send CPU_RUN/out/fixtures/expected.npz --send .../fixtures.json \
#     --send CPU_RUN/out/lib/sm_90a/j10r_ref.so [--send .../h1u_kr.so ...] -- bash benchmarks/torch_gpu/j10r_sm90/job.sh gpu
# Env: NVCC (default CUDA 13.0's), STEPS (gpu's, default "check bench"), CHECK_ARGS, BENCH_ARGS, LEASE_MIN (default 25),
# PY (default /workspace/cp/venv312's python), JOBS.
set -uo pipefail
WHAT=${1:?usage: job.sh build|cpu|gpu}
D=benchmarks/torch_gpu/j10r_sm90
ROOT=$PWD
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"

build() {
  local nv=${NVCC:-/usr/local/cuda-13.0/bin/nvcc} lib=$OUT/lib/sm_90a
  [ -x "$nv" ] || nv=$(command -v nvcc) || { echo "job.sh: no nvcc" >&2; return 2; }
  mkdir -p "$lib"
  "$nv" -O3 -std=c++17 -shared -Xcompiler -fPIC -gencode arch=compute_90a,code=sm_90a -Xptxas -v \
    "$D/j10r_ref.cu" -o "$lib/j10r_ref.so" > "$lib/ptxas.txt" 2>&1 || { cat "$lib/ptxas.txt"; return 2; }
  { echo "commit $(cat "$OUT/commit.txt")"; "$nv" --version | tail -2
    echo "flags -O3 -std=c++17 -shared -Xcompiler -fPIC -gencode arch=compute_90a,code=sm_90a"
    sha256sum "$lib/j10r_ref.so" "$D/j10r_ref.cu" benchmarks/codec_tensor/sha512.cuh; } > "$lib/BUILD.txt"
  cat "$lib/BUILD.txt"; grep -E 'registers|spill' "$lib/ptxas.txt"
}

case $WHAT in
  build) build ;;
  cpu)
    build || exit 2
    uv sync -q --all-packages --extra torch-cpu || exit 2
    uv run python "$D/fixtures.py" expected --out "$OUT/fixtures" --jobs "${JOBS:-30}" ${FIX_ARGS:-}
    rc=$?
    ls -l "$OUT/fixtures"
    exit $rc ;;
  gpu)
    PY=${PY:-/workspace/cp/venv312/bin/python}
    LIBS=${LIBS:-${RESEARCH_RUN_DIR:-.}/inputs}
    export PYTHONPATH=$ROOT:$ROOT/$D PYTHONDONTWRITEBYTECODE=1
    { echo "python $PY"; $PY -c 'import sys, torch, numpy; print(sys.version.split()[0], torch.__version__, torch.version.cuda, numpy.__version__)'
      nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv; ls -l "$LIBS"; } > "$OUT/env.txt" 2>&1
    S=$(mktemp -d /tmp/j10r-sm90.XXXXXX); trap 'rm -rf "$S"' EXIT
    : > "$OUT/rc.txt"
    FIX=$LIBS
    if [ ! -f "$FIX/expected.npz" ]; then
      FIX=$OUT/fixtures
      $PY "$D/fixtures.py" expected --out "$FIX" --jobs "${JOBS:-$(nproc)}" ${FIX_ARGS:-} > "$OUT/fixtures.out" 2>&1
      echo "fixtures rc $? ($(nproc) CPUs)" >> "$OUT/rc.txt"; tail -3 "$OUT/fixtures.out"
    fi
    REF=$LIBS/j10r_ref.so
    if [ ! -f "$REF" ] && [ "${BUILD_REF:-1}" = 1 ]; then
      NVCC=${NVCC:-$(for c in /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { echo "$c/bin/nvcc"; break; }; done)}
      build > "$OUT/build.out" 2>&1; echo "build rc $?" >> "$OUT/rc.txt"; cat "$OUT/build.out"
      REF=$OUT/lib/sm_90a/j10r_ref.so
    fi
    LA=()
    [ -f "$REF" ] && LA+=(--ref "$REF")
    SM=
    for f in "$LIBS"/h1u_*.so; do [ -f "$f" ] && { LA+=(--h1u "$(basename "$f" .so)=$f"); SM=--sm; }; done
    [ -n "$SM" ] && LA+=("$SM")
    $PY "$D/check.py" --fixtures "$FIX" --out "$OUT" --scratch "$S" --operands-only ${CHECK_ARGS:-} > "$OUT/operands.out" 2>&1
    echo "operands rc $?" >> "$OUT/rc.txt"
    BA=${BENCH_ARGS:-}
    [ -f "$LIBS/s8_simple.py" ] && BA="--s8-simple $LIBS/s8_simple.py $BA"
    export O=$OUT PY D S STEPS=${STEPS:-check bench} CHECK_ARGS=${CHECK_ARGS:-} BENCH_ARGS=$BA FIX
    export LA_S="${LA[*]}"
    # shellcheck disable=SC2016
    gpu-lease 1 --wait --max-min "${LEASE_MIN:-25}" -- bash -c '
      nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit --format=csv \
        -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
      for s in $STEPS; do
        case $s in
          check) $PY "$D/check.py" --fixtures "$FIX" --out "$O" --scratch "$S" $LA_S $CHECK_ARGS > "$O/check.out" 2> "$O/check.err" ;;
          bench) [ -f "$D/bench.py" ] || continue
                 $PY "$D/bench.py" --fixtures "$FIX" --out "$O" --scratch "$S" --check "$O/check.json" $LA_S $BENCH_ARGS \
                   > "$O/bench.jsonl" 2> "$O/bench.err" ;;
        esac
        echo "$s rc $?" >> "$O/rc.txt"
      done
      nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1'
    echo "lease rc $?" >> "$OUT/rc.txt"
    cat "$OUT/rc.txt" "$OUT/operands.out"
    tail -40 "$OUT/check.out" 2>/dev/null | cut -c1-600
    tail -20 "$OUT/check.err" 2>/dev/null
    cut -c1-600 "$OUT/bench.jsonl" 2>/dev/null | tail -40
    tail -20 "$OUT/bench.err" 2>/dev/null ;;
  *) echo "job.sh: no $WHAT (build, cpu or gpu)" >&2; exit 2 ;;
esac
