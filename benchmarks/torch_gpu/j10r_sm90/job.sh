#!/usr/bin/env bash
# The jobs (cwd = the source tree's root):
#   build  (a CPU node) the fallback, j10r_ref.cu, at sm_90a: $OUT/lib/sm_90a/j10r_ref.so, ptxas's report, BUILD.txt
#   cpu    build, then fixtures.py expected into $OUT/fixtures (uv sync first; FIX_ARGS, JOBS default 30)
#   gpu    (an H200) check.py, then bench.py, on the files in LIBS (default a run directory's inputs/,
#          RESEARCH_RUN_DIR): expected.npz and fixtures.json, j10r_ref.so as the fallback, every h1u_*.so as an h1u
#          build (and its _sm launch), and s8_linear's build.sh, s8_wgmma.cu, s8_acc.h, sass_gate.py, wgmma.py and
#          configs.py as the bench's wgmma arm, built here by that build.sh (its plain arm is this tree's
#          s8_linear/int_mm.py). Fixtures not in LIBS are computed here, the fallback not in LIBS is built here
#          (BUILD_REF=0: not) and with no h1u_*.so in LIBS h1u_build makes the arms H1U_ARMS names from this tree
#          (BUILD_H1U=0: not), all before the lease, as are the operands, in a scratch directory removed at exit.
#
#   OUT=CPU_OUT bash benchmarks/torch_gpu/j10r_sm90/job.sh cpu   # a CPU node with CUDA 13's nvcc and uv
#   mkdir -p LIBS && cp CPU_OUT/fixtures/expected.npz CPU_OUT/fixtures/fixtures.json \
#     CPU_OUT/lib/sm_90a/j10r_ref.so LIBS/
#   OUT=DIR LIBS=LIBS bash benchmarks/torch_gpu/j10r_sm90/job.sh gpu   # an H200 with nvcc and `gpu-lease` on PATH
# gpu's steps (STEPS, default "check bench") run in that order under one lease; "prof" is prof.py's one call a linear of
# the h1u build PROF_ARM (default kraq) under ncu (NCU, default the ncu beside NVCC; NCU_SET, default --set full; NCU_K,
# its kernel filter), --clock-control none, into $OUT/prof/units.ncu-rep, its raw, details and SASS pages and
# summary.json written after the lease.
# Env: NVCC (default CUDA 13.0's), H1U_ARMS (NAME=DEFINES ..., default H1U_DEFAULT's five), STEPS, CHECK_ARGS, BENCH_ARGS,
# PROF_ARGS, LEASE_MIN (default 25),
# PY (default /workspace/cp/venv312's or /workspace/jobs/venv312's python, else torch and numpy from PyPI in a uv venv
# in the scratch directory), JOBS.
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
    sha256sum "$lib/j10r_ref.so" "$D/j10r_ref.cu" verity/ml/kernels/sha512.cuh; } > "$lib/BUILD.txt"
  cat "$lib/BUILD.txt"; grep -E 'registers|spill' "$lib/ptxas.txt"
}

# sm_90a arms of benchmarks/hash/dither/h1u.cu, compiled from this tree with
# benchmarks/hash/dither/j10r90.sh's defines and dither/job.sh's flags plus --no-compress: $OUT/lib/sm_90a/h1u_ARM.so,
# their resource usage and BUILD.txt; then benchmarks/pouw/harness/sass_gate.py on them as sass_pins.json stands
# (sass-gate.json, a diagnostic only)
H1U_BASE=-DH1U_KSPLIT=4,-DH1K_AUTO=1,-DH1K_RONLY=1
H1U_A=$H1U_BASE,-DH1U_JCTA=1,-DH1U_WARPS=4,-DH1U_JX=1,-DH1U_JBW=1
H1U_Q=$H1U_A,-DH1U_PF=1,-DH1U_JPI=1
# kr, kra and kraq, then the fastest exact arms without and with JW: kraq4 (JCS 4) and kraqwxsc (JW, JWS and JPC, with JXS)
H1U_DEFAULT="kr=$H1U_BASE kra=$H1U_A kraq=$H1U_Q,-DH1U_JCS=2 kraq4=$H1U_Q,-DH1U_JCS=4
  kraqwxsc=$H1U_Q,-DH1U_JW=1,-DH1U_JXS=1,-DH1U_JWS=1,-DH1U_JPC=1"
h1u_build() {
  local nv=${NVCC:-/usr/local/cuda-13.0/bin/nvcc} lib=$OUT/lib/sm_90a arm def rc=0 p pids=() arms=${H1U_ARMS:-$H1U_DEFAULT}
  [ -x "$nv" ] || nv=$(command -v nvcc) || { echo "job.sh: no nvcc" >&2; return 2; }
  mkdir -p "$lib"
  local flags=(-O3 -std=c++17 -shared -Xcompiler -fPIC --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a
               --no-compress)
  for v in $arms; do
    arm=${v%%=*}; def=${v#*=}
    # shellcheck disable=SC2086
    "$nv" "${flags[@]}" -I benchmarks/hash/dither ${def//,/ } benchmarks/hash/dither/h1u.cu -o "$lib/h1u_$arm.so" \
      > "$lib/build-h1u_$arm.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=2; done
  [ $rc = 0 ] || tail -30 "$lib"/build-h1u_*.log
  local deps
  deps=$("$nv" -M -std=c++17 benchmarks/hash/dither/h1u.cu 2>/dev/null | tr ' \\' '\n\n' | grep -v -E ':$|\.o$' | grep . \
    | xargs -r realpath -m --relative-to="$ROOT" | grep -v '^\.\./' | sort -u)
  { echo "commit $(cat "$OUT/commit.txt")"; "$nv" --version | tail -2; echo "flags ${flags[*]}"
    for v in $arms; do echo "arm $v"; done
    sha256sum "$lib"/h1u_*.so $deps; } > "$lib/BUILD-h1u.txt"
  for f in "$lib"/h1u_*.so; do
    "$(dirname "$nv")/cuobjdump" --dump-resource-usage "$f" > "${f%.so}.res.txt" 2>&1
  done
  cat "$lib/BUILD-h1u.txt"; grep -h -A1 -E 'Function.*(h1umj|h1jr|h1_seeds_c)' "$lib"/h1u_*.res.txt | head -40
  POUW_CUDA_BIN=$(dirname "$nv") POUW_SASS_GATE_CACHE=none python3 benchmarks/pouw/harness/sass_gate.py "$lib"/h1u_*.so \
    --json "$OUT/sass-gate.json" > "$OUT/sass-gate.txt" 2>&1
  echo "sass gate (sass_pins.json as it stands, diagnostic): rc $?"; tail -20 "$OUT/sass-gate.txt"
  return $rc
}

# s8_linear's exact plain linear (benchmarks/torch_gpu/s8_linear), given as files in LIBS: its build.sh run unchanged
# from a scratch copy, $OUT/s8ship (the cubin, its SASS gate's sass.json, MANIFEST.sha256); prints the wgmma directory
s8_build() {
  local w=$S/s8/benchmarks/torch_gpu/s8_linear nv=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
  mkdir -p "$w" && cp "$LIBS"/{build.sh,s8_wgmma.cu,s8_acc.h,sass_gate.py,wgmma.py,configs.py} "$w/" || return 2
  CUDA=$(dirname "$(dirname "$nv")") PY=/nonexistent bash "$w/build.sh" "$OUT/s8ship" >&2 || return 2
  echo "$w"
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
    LIBS=${LIBS:-${RESEARCH_RUN_DIR:-.}/inputs}
    S=$(mktemp -d /tmp/j10r-sm90.XXXXXX); trap 'rm -rf "$S"' EXIT
    : > "$OUT/rc.txt"
    PY=${PY:-}
    for v in /workspace/cp/venv312 /workspace/jobs/venv312; do [ -z "$PY" ] && [ -x "$v/bin/python" ] && PY=$v/bin/python; done
    if [ -z "$PY" ]; then
      uv venv -q -p 3.12 "$S/venv" && uv pip install -q -p "$S/venv" torch numpy > "$OUT/venv.out" 2>&1
      echo "venv rc $?" >> "$OUT/rc.txt"
      PY=$S/venv/bin/python
    fi
    export PYTHONPATH=$ROOT:$ROOT/$D PYTHONDONTWRITEBYTECODE=1
    { echo "python $PY"; $PY -c 'import sys, torch, numpy; print(sys.version.split()[0], torch.__version__, torch.version.cuda, numpy.__version__)'
      nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv; ls -l "$LIBS"; } > "$OUT/env.txt" 2>&1
    FIX=$LIBS
    if [ ! -f "$FIX/expected.npz" ]; then
      FIX=$OUT/fixtures
      $PY "$D/fixtures.py" expected --out "$FIX" --jobs "${JOBS:-$(nproc)}" ${FIX_ARGS:-} > "$OUT/fixtures.out" 2>&1
      echo "fixtures rc $? ($(nproc) CPUs)" >> "$OUT/rc.txt"; tail -3 "$OUT/fixtures.out"
    fi
    NVCC=${NVCC:-$(for c in /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { echo "$c/bin/nvcc"; break; }; done)}
    export NVCC
    REF=$LIBS/j10r_ref.so
    if [ ! -f "$REF" ] && [ "${BUILD_REF:-1}" = 1 ]; then
      build > "$OUT/build.out" 2>&1; echo "build rc $?" >> "$OUT/rc.txt"; cat "$OUT/build.out"
      REF=$OUT/lib/sm_90a/j10r_ref.so
    fi
    H1US=("$LIBS"/h1u_*.so)
    if [ ! -f "${H1US[0]}" ] && [ "${BUILD_H1U:-1}" = 1 ]; then
      h1u_build > "$OUT/h1u-build.out" 2>&1; echo "h1u build rc $?" >> "$OUT/rc.txt"; cat "$OUT/h1u-build.out"
      H1US=("$OUT"/lib/sm_90a/h1u_*.so)
    fi
    LA=()
    [ -f "$REF" ] && LA+=(--ref "$REF")
    SM=
    for f in "${H1US[@]}"; do [ -f "$f" ] && { LA+=(--h1u "$(basename "$f" .so)=$f"); SM=--sm; }; done
    [ -n "$SM" ] && LA+=("$SM")
    $PY "$D/check.py" --fixtures "$FIX" --out "$OUT" --scratch "$S" --operands-only ${CHECK_ARGS:-} > "$OUT/operands.out" 2>&1
    echo "operands rc $?" >> "$OUT/rc.txt"
    BA=${BENCH_ARGS:-}
    if [ -f "$LIBS/s8_wgmma.cu" ]; then
      W8=$(s8_build 2> "$OUT/s8-build.out"); echo "s8 build rc $?" >> "$OUT/rc.txt"; tail -5 "$OUT/s8-build.out"
      [ -f "$OUT/s8ship/s8_wgmma.cubin" ] && { export S8_WGMMA_CUBIN=$OUT/s8ship/s8_wgmma.cubin; BA="--wgmma $W8 $BA"; }
    fi
    PL=()
    for f in "${H1US[@]}"; do [ "$(basename "$f" .so)" = "h1u_${PROF_ARM:-kraq}" ] && PL=(--h1u "${PROF_ARM:-kraq}=$f" $SM); done
    NCU=${NCU:-$(dirname "$NVCC")/ncu}
    mkdir -p "$OUT/prof"
    export O=$OUT PY D S STEPS=${STEPS:-check bench} CHECK_ARGS=${CHECK_ARGS:-} BENCH_ARGS=$BA FIX NCU
    export LA_S="${LA[*]}" PL_S="${PL[*]}" PROF_ARGS=${PROF_ARGS:-} NCU_SET=${NCU_SET:---set full}
    export NCU_K=${NCU_K:-regex:h1umj|h1jr|h1_seeds}
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
          prof) [ -n "$PL_S" ] || { echo "prof: no h1u build named ${PROF_ARM:-kraq}" > "$O/prof.err"; false; } &&
                timeout 900 "$NCU" --profile-from-start off --clock-control none $NCU_SET --import-source no --nvtx \
                  -k "$NCU_K" -c 64 -f -o "$O/prof/units" $PY "$D/prof.py" run --fixtures "$FIX" --scratch "$S" \
                  --out "$O/prof" $PL_S $PROF_ARGS > "$O/prof.out" 2> "$O/prof.err" ;;
        esac
        echo "$s rc $?" >> "$O/rc.txt"
      done
      nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1'
    echo "lease rc $?" >> "$OUT/rc.txt"
    if [ -f "$OUT/prof/units.ncu-rep" ]; then
      "$NCU" -i "$OUT/prof/units.ncu-rep" --page raw --csv > "$OUT/prof/raw.csv" 2> "$OUT/prof/pages.err"
      "$NCU" -i "$OUT/prof/units.ncu-rep" --page details --print-details all > "$OUT/prof/details.txt" 2>> "$OUT/prof/pages.err"
      "$NCU" -i "$OUT/prof/units.ncu-rep" --page source --print-source sass --csv 2>> "$OUT/prof/pages.err" \
        | gzip > "$OUT/prof/source-sass.csv.gz"
      $PY "$D/prof.py" summary "$OUT/prof/raw.csv" --out "$OUT/prof/summary.json" > /dev/null
      echo "prof pages rc $?" >> "$OUT/rc.txt"
    fi
    tail -5 "$OUT/prof.err" 2>/dev/null
    cat "$OUT/rc.txt" "$OUT/operands.out"
    tail -40 "$OUT/check.out" 2>/dev/null | cut -c1-600
    tail -20 "$OUT/check.err" 2>/dev/null
    cut -c1-600 "$OUT/bench.jsonl" 2>/dev/null | tail -40
    tail -20 "$OUT/bench.err" 2>/dev/null ;;
  *) echo "job.sh: no $WHAT (build, cpu or gpu)" >&2; exit 2 ;;
esac
