#!/usr/bin/env bash
# The dither-hb benches. Libraries are built on a CPU node and shipped to the dc node, so nothing but the bench runs on
# vy-dc-h200 / vy-dc-b200 (no CPU phase there):
#   build    (vy-nebius-2) every library for each ARCH in a comma list: $OUT/lib/ARCH/{kb,kb_pg,h1noise,h1u}.so and the
#            variants h1u_b3 (H1U_MINB=3), h1u_pf (H1U_PF=1), h1u_pN and h1noise_pN (H1_PIPE=N, N in $PIPES and $NOISE_PIPES), their
#            SASS and resource usage, and BUILD.txt: the commit and arch, and per library the sha256 of the repository
#            files nvcc -M names (a bench refuses a library whose sources differ at its own commit)
#   kbench   kb.so (and kb_pg.so on sm_100a): kbench.py (the honest GEMMs gated, then timed against the plain int8 GEMMs);
#            verify.py (CA's numpy twin) runs after a `research fetch`, off the node
#   h1       h1noise.so: h1bench.py (H1's per-unit noise gated, then timed against the plain int8 GEMMs)
#   h1u      h1u.so, h1noise.so and every h1u_* / h1noise_* variant shipped: h1ubench.py (H1 + fx1's fused unit, and
#            each build's two-launch kh_unit_ws as arm NAME-ws, gated against receipts.py and h1tile.reference, the
#            noise floor gated as h1bench.py does, then all timed against the plain int8 GEMMs)
# Benches in a comma list (kbench,h1u) run one after another under one lease.
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
ARCH=${1:?usage: job.sh ARCH[,ARCH] build | job.sh sm_90a|sm_100a kbench|h1|h1u[,...]}
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
  for arch in ${ARCH//,/ }; do
    dir="$OUT/lib/$arch"
    mkdir -p "$dir"
    # NAME[:FILE:FLAG]: a variant is FILE.cu built with FLAG
    builds=(kb kb_pg h1noise h1u h1u_b3:h1u:-DH1U_MINB=3 h1u_pf:h1u:-DH1U_PF=1)
    # H1_PIPE's bits (h1sha.cuh): 4 Sigma1, 8 Sigma0, 16 sigma0, 32 sigma1 and (h1u.cu) 64 the fold's rotations on IMAD
    for p in ${PIPES:-4 12 36 76 100}; do builds+=("h1u_p$p:h1u:-DH1_PIPE=$p"); done
    for p in ${NOISE_PIPES:-4 12 36}; do builds+=("h1noise_p$p:h1noise:-DH1_PIPE=$p"); done
    for b in "${builds[@]}"; do
      IFS=: read -r src file def <<< "$b"
      [ "$src" = kb_pg ] && [ "$arch" != sm_100a ] && continue
      flags=(-O3 -std=c++17 -shared -Xcompiler -fPIC --expt-relaxed-constexpr -gencode "arch=compute_${arch#sm_},code=$arch")
      [ "$src" = kb ] && flags+=(-lcublasLt -Xlinker "-rpath=$LIB64")
      file=${file:-$src}
      [ -n "$def" ] && flags+=("$def")
      "$NVCC" "${flags[@]}" "$HERE/$file.cu" -o "$dir/$src.so" 2> "$dir/build-$src.log" || { cat "$dir/build-$src.log"; exit 2; }
      "$(dirname "$NVCC")/cuobjdump" --dump-resource-usage "$dir/$src.so" > "$dir/$src.res.txt" 2>&1
      "$(dirname "$NVCC")/cuobjdump" -sass "$dir/$src.so" | gzip > "$dir/$src.sass.gz"
      "$NVCC" -M -std=c++17 "${flags[@]:6:2}" "$HERE/$file.cu" > "$dir/$src.deps" 2>&1 || { cat "$dir/$src.deps"; exit 2; }
      deps=$(tr ' \\' '\n\n' < "$dir/$src.deps" | grep -v ':$' | grep . | xargs -r realpath -m | while read -r f; do [ -f "$f" ] && echo "$f"; done | grep "^$ROOT/" | sed "s#^$ROOT/##" | sort -u)
      echo "src $src $(srchash $deps) $(echo $deps)" >> "$dir/BUILD.lines"
    done
    { echo "$SHA $arch"; cat "$dir/BUILD.lines"; } > "$dir/BUILD.txt"
    rm "$dir/BUILD.lines"
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
      CMDS+=("$PY $HERE/kbench.py --lib $LIBS/kb.so $PG --out $OUT ${BENCH_ARGS:-}") ;;
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
    *) echo "job.sh: no bench $b" >&2; exit 2 ;;
  esac
done
[ -f "$LIBS/BUILD.txt" ] || { echo "job.sh: no $LIBS/BUILD.txt: --send the build's lib/$ARCH/BUILD.txt" >&2; exit 2; }
cp "$LIBS/BUILD.txt" "$OUT/env/BUILD.txt"
read -r bsha barch < "$LIBS/BUILD.txt"
[ "$barch" = "$ARCH" ] || { echo "job.sh: the libraries are $barch's, not $ARCH's" >&2; exit 2; }
for name in $(printf '%s\n' "${NEED[@]}" | sort -u); do
  [ -f "$LIBS/$name.so" ] || { echo "job.sh: no $LIBS/$name.so: build on vy-nebius-2 (job.sh $ARCH build) and --send lib/$ARCH/$name.so" >&2; exit 2; }
  read -r _ _ h paths <<< "$(grep "^src $name " "$LIBS/BUILD.txt")"
  [ -n "$h" ] && [ "$(srchash $paths)" = "$h" ] || {
    echo "job.sh: $name.so was built (commit $bsha) from sources that differ at $SHA: rebuild" >&2; exit 2; }
done
export PYTHONPATH=$PWD/integrations/vllm:$PWD${PYTHONPATH:+:$PYTHONPATH}
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
  GPU_LEASE_WHO=kernels-dither-hb gpu-lease 1 --wait "${lease[@]}" -- bash -c "$bench"
fi
rc=$?
echo "lease rc $rc"
exit $rc
