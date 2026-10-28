#!/usr/bin/env bash
# The dither-hb benches. Libraries are built on a CPU host and copied to the GPU host, so nothing but the bench runs on
# the H200 or the B200 (no CPU phase there):
#   build    (a CPU host) every library for each ARCH in a comma list: $OUT/lib/ARCH/{kb,kb_pg,h1noise,h1u}.so and the
#            h1u variants in $VARIANTS (NAME=FLAGS, e.g. pd4=-DH1U_PF=1,-DH1U_PD=4 builds h1u_pd4.so), their SASS and
#            resource usage, and BUILD.txt: the commit and arch, and per library the sha256 of the repository files
#            nvcc -M names (a bench refuses a library whose sources differ at its own commit)
#   kbench   kb.so (and kb_pg.so on sm_100a): hb_kbench.py (the honest GEMMs gated, then timed against the plain int8 GEMMs);
#            verify.py (the numpy twin) runs on the bench's out/ afterwards, off the GPU host
#   h1       h1noise.so: h1bench.py (H1's per-unit noise gated, then timed against the plain int8 GEMMs)
#   h1u      h1u.so, h1noise.so and every h1u_* / h1noise_* variant shipped: h1ubench.py (H1 + fx1's fused unit, and
#            each build's two-launch kh_unit_ws as arm NAME-ws, gated against receipts.py and h1tile.reference, the
#            noise floor gated as h1bench.py does, then all timed against the plain int8 GEMMs)
#   prof     every h1u*.so shipped: hb_prof.py under ncu (--set full, --clock-control none), each build gated as h1ubench.py
#            gates it, then profiled once at the served shapes ($OUT/prof.ncu-rep and its text pages in $OUT/prof/)
#   units    h1u_kr.so (dither-a's served receipts-only build, the base) and every h1u_kr*.so variant shipped:
#            unitsbench.py (the served noise and units launches at the served window, each build gated against
#            receipts.py and the base's every word, then timed: each call's units and the step's 128)
#   unitsprof  the same builds under ncu (--set full; kernels h1u and h1k): unitsbench.py --profile, then the text pages
#   partials h1u.so and every h1u_*.so shipped: benchmarks/pouw/tests/test_partials.py and test_lane_refs.py with
#            PARTIALS_LIB each (option A's partials: kh_seeds_at and kh_lanes byte for byte served_h1/partials.py's cpu
#            backend and SERVE-REG's gated lane rows)
# Benches in a comma list (kbench,h1u) run one after another under one lease.
#
#   OUT=B bash benchmarks/hash/dither/job.sh sm_90a,sm_100a build          (a CPU host with CUDA 13.0, up to 20 minutes)
#   OUT=R LIBS=B/lib/sm_90a bash benchmarks/hash/dither/job.sh sm_90a h1u  (an H200 under a GPU lease, up to 40 minutes)
#
# Env: VARIANTS (default below), BUILDS (the base libraries a build makes, default "kb h1noise h1u" and
# kb_pg on sm_100a; set empty, only the variants), NVCC_EXTRA (nvcc flags every build adds, e.g. --no-compress for the
# SASS gate), JOBS (builds at once, default 8), BENCH_ARGS (extra flags for the bench, e.g. "--ms 1,32 --linears o"), HB_WHO (the lease owner, default dither/ and the run's id), LEASE (gpu-lease flags, default
# "--max-min 14"), LIBS (the shipped libraries, default $RESEARCH_RUN_DIR/inputs), VERITY_BENCH_VENV (default
# /workspace/cp/venv312), NVCC (default CUDA 13.0's), NCU (default the ncu beside NVCC).
# Exit: the bench's (4 a gate failed), 2 a build failed or the shipped libraries are missing or another commit's.
set -uo pipefail
ARCH=${1:?usage: job.sh ARCH[,ARCH] build | job.sh sm_90a|sm_100a|sm_120a kbench|h1|h1u|prof|units|unitsprof|partials[,...]}
BENCH=${2:-kbench}
HERE=$(cd "$(dirname "$0")" && pwd)
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
LIBS=${LIBS:-${RESEARCH_RUN_DIR:-.}/inputs}
SHA=${RESEARCH_SOURCE_SHA:-$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)}
ROOT=$(pwd -P)
mkdir -p "$OUT/env"
srchash() { (cd "$ROOT" && cat "$@" | sha256sum | cut -c1-16); }

if [ "$BENCH" = build ]; then
  "$NVCC" --version > "$OUT/env/nvcc.txt" 2>&1
  LIB64=$(cd "$(dirname "$NVCC")/../lib64" && pwd -P)
  # one library: NAME[:FILE:FLAG] (a variant is FILE.cu built with FLAG) for one arch; its BUILD.txt line to NAME.line
  build_one() {
    local arch=$1 dir=$2 src file def deps
    IFS=: read -r src file def <<< "$3"
    local flags=(-O3 -std=c++17 -shared -Xcompiler -fPIC --expt-relaxed-constexpr -gencode "arch=compute_${arch#sm_},code=$arch")
    [ "$src" = kb ] && flags+=(-lcublasLt -Xlinker "-rpath=$LIB64")
    file=${file:-$src}
    [ -n "$def" ] && flags+=(${def//,/ })
    [ -n "${NVCC_EXTRA:-}" ] && flags+=($NVCC_EXTRA)
    "$NVCC" "${flags[@]}" "$HERE/$file.cu" -o "$dir/$src.so" 2> "$dir/build-$src.log" || { echo "build $arch $src failed:"; cat "$dir/build-$src.log"; return 2; }
    "$(dirname "$NVCC")/cuobjdump" --dump-resource-usage "$dir/$src.so" > "$dir/$src.res.txt" 2>&1
    "$(dirname "$NVCC")/cuobjdump" -sass "$dir/$src.so" | gzip > "$dir/$src.sass.gz"
    "$NVCC" -M -std=c++17 "${flags[@]:6:2}" "$HERE/$file.cu" > "$dir/$src.deps" 2>&1 || { cat "$dir/$src.deps"; return 2; }
    deps=$(tr ' \\' '\n\n' < "$dir/$src.deps" | grep -v ':$' | grep . | xargs -r realpath -m | while read -r f; do [ -f "$f" ] && echo "$f"; done | grep "^$ROOT/" | sed "s#^$ROOT/##" | sort -u)
    echo "src $src $(srchash $deps) $(echo $deps)" > "$dir/$src.line"
  }
  brc=0
  for arch in ${ARCH//,/ }; do
    dir="$OUT/lib/$arch"
    mkdir -p "$dir"
    read -r -a builds <<< "${BUILDS-kb h1noise h1u}"
    [ "$arch" = sm_100a ] && [ -z "${BUILDS+set}" ] && builds+=(kb_pg)
    # h1u.cu's knobs: H1U_MINB (CTAs an SM the registers leave room for), H1U_PF (the fused form prefetches too), H1U_PD
    # (segments of A and B in flight), H1U_L2PF (L2 prefetch 16 segments ahead and of the next chunk's workspace), H1U_WARPS,
    # H1U_SPLIT (h1p: a pair of warps a unit), H1U_KSPLIT (h1k: the one-stream unit's K split, 2 or 4 producer warps, for
    # calls of at most H1K_MAXU live units), H1K_AUTO (with H1U_KSPLIT = 4: 4 producers where one wave holds the call's
    # units, else 2)
    for v in ${VARIANTS:-b3=-DH1U_MINB=3 pf=-DH1U_PF=1 pd2=-DH1U_PF=1,-DH1U_PD=2 pd4=-DH1U_PF=1,-DH1U_PD=4 \
             pd8=-DH1U_PF=1,-DH1U_PD=8 l2=-DH1U_PF=1,-DH1U_L2PF=1 pd2l2=-DH1U_PF=1,-DH1U_PD=2,-DH1U_L2PF=1 \
             pd4l2=-DH1U_PF=1,-DH1U_PD=4,-DH1U_L2PF=1 pd2b3=-DH1U_PF=1,-DH1U_PD=2,-DH1U_MINB=3 \
             pd4w2=-DH1U_PF=1,-DH1U_PD=4,-DH1U_WARPS=2 sp2=-DH1U_SPLIT=1,-DH1U_PF=1,-DH1U_PD=2 \
             sp4=-DH1U_SPLIT=1,-DH1U_PF=1,-DH1U_PD=4 sp4l2=-DH1U_SPLIT=1,-DH1U_PF=1,-DH1U_PD=4,-DH1U_L2PF=1 \
             sp4w2=-DH1U_SPLIT=1,-DH1U_PF=1,-DH1U_PD=4,-DH1U_WARPS=2 sp4b3=-DH1U_SPLIT=1,-DH1U_PF=1,-DH1U_PD=4,-DH1U_MINB=3}; do
      builds+=("h1u_${v%%=*}:h1u:${v#*=}")
    done
    for b in "${builds[@]}"; do
      while [ "$(jobs -rp | wc -l)" -ge "${JOBS:-8}" ]; do wait -n || brc=2; done
      build_one "$arch" "$dir" "$b" &
    done
  done
  while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || brc=2; done
  wait || brc=2
  [ $brc = 0 ] || exit 2
  for arch in ${ARCH//,/ }; do
    dir="$OUT/lib/$arch"
    { echo "$SHA $arch"; cat "$dir"/*.line; } > "$dir/BUILD.txt"
    rm "$dir"/*.line
    grep -h "Function\|REG" "$dir"/*.res.txt
  done
  exit 0
fi

NEED=()
CMDS=()
for b in ${BENCH//,/ }; do
  case $b in
    kbench)
      NEED+=(kb)
      PG=""
      [ "$ARCH" = sm_100a ] && { NEED+=(kb_pg); PG="--pg-lib $LIBS/kb_pg.so"; }
      CMDS+=("$PY $HERE/hb_kbench.py --lib $LIBS/kb.so $PG --out $OUT ${BENCH_ARGS:-}") ;;
    h1) NEED+=(h1noise); CMDS+=("$PY $HERE/h1bench.py --lib $LIBS/h1noise.so --out $OUT ${BENCH_ARGS:-}") ;;
    h1u)
      NEED+=(h1u h1noise)
      vars=""
      for f in "$LIBS"/h1u_*.so; do
        [ -f "$f" ] || continue
        v=$(basename "$f" .so); NEED+=("$v"); vars+=" --variant ${v#h1u_}=$f"
      done
      for f in "$LIBS"/h1noise_*.so; do
        [ -f "$f" ] || continue
        v=$(basename "$f" .so); NEED+=("$v"); vars+=" --noise-variant ${v#h1noise_}=$f"
      done
      CMDS+=("$PY $HERE/h1ubench.py --lib $LIBS/h1u.so --noise-lib $LIBS/h1noise.so$vars --out $OUT ${BENCH_ARGS:-}") ;;
    units|unitsprof)
      # h1u_kr.so (today's served receipts-only build) as the base, then every other h1u_kr*.so shipped
      ul=""
      for f in "$LIBS"/h1u_kr.so "$LIBS"/h1u_kr?*.so; do
        [ -f "$f" ] || continue
        v=$(basename "$f" .so); NEED+=("$v"); ul+=" --lib ${v#h1u_}=$f"
      done
      [ -f "$LIBS/h1u_kr.so" ] || { echo "job.sh: $b needs $LIBS/h1u_kr.so, the base" >&2; exit 2; }
      if [ "$b" = units ]; then
        CMDS+=("$PY $HERE/unitsbench.py$ul --out $OUT ${BENCH_ARGS:-}")
      else
        NCU=${NCU:-$(dirname "$NVCC")/ncu}
        CMDS+=("$NCU --profile-from-start off --clock-control none --set full --nvtx -k regex:h1u|h1k -f -o $OUT/prof $PY $HERE/unitsbench.py$ul --profile --out $OUT ${BENCH_ARGS:-}")
        CMDS+=("$PY $HERE/hb_prof.py --pages $OUT/prof.ncu-rep --ncu $NCU --out $OUT")
      fi ;;
    partials)
      # every h1u*.so shipped: test_partials.py and test_lane_refs.py with PARTIALS_LIB, their gpu tests (kh_seeds_at and
      # kh_lanes, against the cpu's and SERVE-REG's lane rows) on each
      for f in "$LIBS"/h1u.so "$LIBS"/h1u_*.so; do
        [ -f "$f" ] || continue
        v=$(basename "$f" .so); NEED+=("$v")
        CMDS+=("env PARTIALS_LIB=$f $PY -m pytest -q -rA -p no:cacheprovider $ROOT/benchmarks/pouw/tests/test_partials.py $ROOT/benchmarks/pouw/tests/test_lane_refs.py")
      done
      [ -f "$LIBS/h1u.so" ] || { echo "job.sh: partials needs $LIBS/h1u.so" >&2; exit 2; } ;;
    prof)
      plibs=""
      for f in "$LIBS"/h1u.so "$LIBS"/h1u_*.so; do
        [ -f "$f" ] || continue
        v=$(basename "$f" .so); NEED+=("$v"); plibs+=" --lib $v=$f"
      done
      [ -n "$plibs" ] || { echo "job.sh: prof needs an h1u*.so in $LIBS" >&2; exit 2; }
      NCU=${NCU:-$(dirname "$NVCC")/ncu}
      CMDS+=("$NCU --profile-from-start off --clock-control none --set full --nvtx -k regex:h1u|h1p|h1s_noise -f -o $OUT/prof $PY $HERE/hb_prof.py$plibs --out $OUT ${BENCH_ARGS:-}")
      CMDS+=("$PY $HERE/hb_prof.py --pages $OUT/prof.ncu-rep --ncu $NCU --out $OUT") ;;
    *) echo "job.sh: no bench $b" >&2; exit 2 ;;
  esac
done
[ -f "$LIBS/BUILD.txt" ] || { echo "job.sh: no $LIBS/BUILD.txt: point LIBS at the build's lib/$ARCH" >&2; exit 2; }
cp "$LIBS/BUILD.txt" "$OUT/env/BUILD.txt"
read -r bsha barch < "$LIBS/BUILD.txt"
[ "$barch" = "$ARCH" ] || { echo "job.sh: the libraries are $barch's, not $ARCH's" >&2; exit 2; }
for name in $(printf '%s\n' "${NEED[@]}" | sort -u); do
  [ -f "$LIBS/$name.so" ] || { echo "job.sh: no $LIBS/$name.so: build it on a CPU host (job.sh $ARCH build) and point LIBS at its lib/$ARCH" >&2; exit 2; }
  read -r _ _ h paths <<< "$(grep "^src $name " "$LIBS/BUILD.txt")"
  [ -n "$h" ] && [ "$(srchash $paths)" = "$h" ] || {
    echo "job.sh: $name.so was built (commit $bsha) from sources that differ at $SHA: rebuild" >&2; exit 2; }
done
export PYTHONPATH=$PWD/integrations/vllm:$PWD:$PWD/tools/config${PYTHONPATH:+:$PYTHONPATH}
printf '%s\n' "${CMDS[@]}" > "$OUT/env/cmds.txt"
bench='
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.before.csv"
  rc=0
  while read -r cmd; do
    $cmd < /dev/null 2>&1 | tee -a "$OUT/bench.out"
    r=${PIPESTATUS[0]}
    echo "bench rc $r: $cmd" | tee -a "$OUT/bench.out"
    [ "$rc" = 0 ] && rc=$r
  done < "$OUT/env/cmds.txt"
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.after.csv"
  exit $rc'
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  bash -c "$bench"
else
  read -r -a lease <<< "${LEASE:---max-min 14}"
  GPU_LEASE_WHO=${HB_WHO:-dither/${RESEARCH_RUN_ID:-local}} gpu-lease 1 --wait "${lease[@]}" -- bash -c "$bench"
fi
rc=$?
echo "lease rc $rc"
exit $rc
