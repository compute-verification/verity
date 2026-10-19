#!/usr/bin/env bash
# The dither-hb benches. Libraries are built on a CPU node and shipped to the dc node, so nothing but the bench runs on
# vy-dc-h200 / vy-dc-b200 (no CPU phase there):
#   build    (vy-nebius-2) every library for each ARCH in a comma list: $OUT/lib/ARCH/{kb,kb_pg,h1noise,h1u}.so, their
#            SASS and resource usage, and BUILD.txt (this commit, the arch)
#   kbench   kb.so (and kb_pg.so on sm_100a): kbench.py (the honest GEMMs gated, then timed against the plain int8 GEMMs);
#            verify.py (CA's numpy twin) runs after a `research fetch`, off the node
#   h1       h1noise.so: h1bench.py (H1's per-unit noise gated, then timed against the plain int8 GEMMs)
#   h1u      h1u.so and h1noise.so: h1ubench.py (H1 + fx1's fused unit gated against receipts.py and h1tile.reference,
#            then timed against the plain int8 GEMMs and the noise floor)
#
#   research run --on vy-nebius-2 --preemptible --project verity --campaign dither-loop --source . --cwd source \
#     --timeout 20m -- bash benchmarks/hash/dither/job.sh sm_90a,sm_100a build
#   research run --on vy-dc-h200 --project verity --campaign dither-loop --source . --cwd source --timeout 40m \
#     --send BUILD_RUN/out/lib/sm_90a/h1u.so --send ... --send BUILD_RUN/out/lib/sm_90a/BUILD.txt \
#     -- bash benchmarks/hash/dither/job.sh sm_90a h1u
#
# Env: BENCH_ARGS (extra flags for the bench, e.g. "--ms 1,32 --linears o"), LEASE (gpu-lease flags, default
# "--max-min 14"), LIBS (the shipped libraries, default $RESEARCH_RUN_DIR/inputs), VERITY_BENCH_VENV (default
# /workspace/cp/venv312), NVCC (default CUDA 13.0's).
# Exit: the bench's (4 a gate failed), 2 a build failed or the shipped libraries are missing or another commit's.
set -uo pipefail
ARCH=${1:?usage: job.sh ARCH[,ARCH] build | job.sh sm_90a|sm_100a kbench|h1|h1u}
BENCH=${2:-kbench}
HERE=$(cd "$(dirname "$0")" && pwd)
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
LIBS=${LIBS:-${RESEARCH_RUN_DIR:-.}/inputs}
SHA=${RESEARCH_SOURCE_SHA:-$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)}
mkdir -p "$OUT/env"

if [ "$BENCH" = build ]; then
  "$NVCC" --version > "$OUT/env/nvcc.txt" 2>&1
  LIB64=$(cd "$(dirname "$NVCC")/../lib64" && pwd -P)
  for arch in ${ARCH//,/ }; do
    dir="$OUT/lib/$arch"
    mkdir -p "$dir"
    for src in kb kb_pg h1noise h1u; do
      [ "$src" = kb_pg ] && [ "$arch" != sm_100a ] && continue
      flags=(-O3 -std=c++17 -shared -Xcompiler -fPIC --expt-relaxed-constexpr -gencode "arch=compute_${arch#sm_},code=$arch")
      [ "$src" = kb ] && flags+=(-lcublasLt -Xlinker "-rpath=$LIB64")
      "$NVCC" "${flags[@]}" "$HERE/$src.cu" -o "$dir/$src.so" 2> "$dir/build-$src.log" || { cat "$dir/build-$src.log"; exit 2; }
      "$(dirname "$NVCC")/cuobjdump" --dump-resource-usage "$dir/$src.so" > "$dir/$src.res.txt" 2>&1
      "$(dirname "$NVCC")/cuobjdump" -sass "$dir/$src.so" | gzip > "$dir/$src.sass.gz"
    done
    echo "$SHA $arch" > "$dir/BUILD.txt"
    grep -h "Function\|REG" "$dir"/*.res.txt
  done
  exit 0
fi

case $BENCH in
  kbench) NEED=(kb); [ "$ARCH" = sm_100a ] && NEED+=(kb_pg) ;;
  h1) NEED=(h1noise) ;;
  h1u) NEED=(h1u h1noise) ;;
  *) echo "job.sh: no bench $BENCH" >&2; exit 2 ;;
esac
for name in "${NEED[@]}"; do
  [ -f "$LIBS/$name.so" ] || { echo "job.sh: no $LIBS/$name.so: build on vy-nebius-2 (job.sh $ARCH build) and --send lib/$ARCH/$name.so" >&2; exit 2; }
done
[ -f "$LIBS/BUILD.txt" ] || { echo "job.sh: no $LIBS/BUILD.txt: --send the build's lib/$ARCH/BUILD.txt" >&2; exit 2; }
cp "$LIBS/BUILD.txt" "$OUT/env/BUILD.txt"
read -r bsha barch < "$LIBS/BUILD.txt"
[ "$barch" = "$ARCH" ] || { echo "job.sh: the libraries are $barch's, not $ARCH's" >&2; exit 2; }
[ "$bsha" = "$SHA" ] || { echo "job.sh: the libraries are commit $bsha's, this run is $SHA: rebuild at this commit" >&2; exit 2; }
case $BENCH in
  kbench)
    PG=""
    [ "$ARCH" = sm_100a ] && PG="--pg-lib $LIBS/kb_pg.so"
    CMD="$PY $HERE/kbench.py --lib $LIBS/kb.so $PG --out $OUT ${BENCH_ARGS:-}" ;;
  h1) CMD="$PY $HERE/h1bench.py --lib $LIBS/h1noise.so --out $OUT ${BENCH_ARGS:-}" ;;
  h1u) CMD="$PY $HERE/h1ubench.py --lib $LIBS/h1u.so --noise-lib $LIBS/h1noise.so --out $OUT ${BENCH_ARGS:-}" ;;
esac
export PYTHONPATH=$PWD/integrations/vllm:$PWD${PYTHONPATH:+:$PYTHONPATH} CMD
bench='
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.before.csv"
  $CMD 2>&1 | tee "$OUT/bench.out"
  rc=${PIPESTATUS[0]}
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$OUT/clocks.after.csv"
  exit $rc'
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  bash -c "$bench"
else
  read -r -a lease <<< "${LEASE:---max-min 14}"
  GPU_LEASE_WHO=kernels-dither-hb gpu-lease 1 --wait "${lease[@]}" -- bash -c "$bench"
fi
rc=$?
echo "lease rc $rc"
exit $rc
