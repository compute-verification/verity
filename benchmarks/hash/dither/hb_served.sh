#!/usr/bin/env bash
# Dither's served standard number on the H200 and the B200: kernels-dither-rtx's fused served linear (h1f.cu, h1c.cu,
# dither_fused.py) and its served bench (sbench.py), run from that lane's pushed commit, not copied. The run's source is a
# clean worktree of origin/cursor/kernels-dither-rtx-15ad at a pinned commit; this script and hb_sbench.py are sent with
# --send from this branch's pushed commit, so the run records both. Two modes, so no CPU phase runs on a dc node:
#   build ARCH[,ARCH]  (vy-nebius-2) libh1c.so, libh1f.so and libh1o.so for each arch into $OUT/lib/ARCH, their SASS and resource
#                      usage, and BUILD.txt: the source commit and arch, and per library the sha256 of the files nvcc -M names
#   run ARCH [ARGS]    (vy-dc-h200 sm_90a, vy-dc-b200 sm_100a) sbench.py through hb_sbench.py on one leased GPU for at most
#                      14 minutes, every shipped library checked against the source's files first; ARGS go to sbench.py
#
#   git worktree add --detach /tmp/rtx-src <rtx commit>
#   research run --on vy-nebius-2 --preemptible --queue --phase cpu --cpus 4 --mem-gb 8 --disk-gb 1 --project verity \
#     --campaign dither-loop --source /tmp/rtx-src --cwd source --timeout 20m \
#     --send benchmarks/hash/dither/hb_served.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/hb_served.sh" build sm_90a,sm_100a'
#   research run --on vy-dc-h200 --project verity --campaign dither-loop --source /tmp/rtx-src --cwd source --timeout 60m \
#     --send benchmarks/hash/dither/hb_served.sh --send benchmarks/hash/dither/hb_sbench.py \
#     --send BUILD_RUN/out/lib/sm_90a/libh1c.so --send BUILD_RUN/out/lib/sm_90a/libh1f.so --send BUILD_RUN/out/lib/sm_90a/BUILD.txt \
#     --env DITHER_NSYS=1 -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/hb_served.sh" run sm_90a --arms stock,exact,dither --gate ...'
#
# The stock arm is native compiled vLLM int8 (compressed-tensors W8A8) on TritonInt8ScaledMMLinearKernel, whose dequant
# order is the int8-exact path's: sbench disables vLLM's first choice, CutlassInt8ScaledMMLinearKernel, which refuses int8
# on sm_100a ("Int8 not supported on SM100") and sm_120. On sm_90a hb_sbench.py's stock-cutlass arm keeps it (vLLM's
# default there). Every arm samples without FlashInfer's JIT sampler, as served_h1 and native.sh do.
# hb_sbench.py adds five arms: stock-cutlass (sm_90a); dither-hb, dither-side's receipts on this lane's kh_unit_ws
# (hb_side.py and an h1u .so from job.sh's build on this branch, both sent; HB_H1U names the .so); dither-1s-hb,
# dither-1s with this lane's one-stream kh_one in place of h1o_units (hb_one.py and hb_side.py, the same .so);
# dither-1s-hbp, dither-1s-hb peeled (kh_one_peel: z = A B^T, y today's; hb_peel.py's gate, sent too), from the .so
# HB_H1U_PEEL names if it's set (the fastest peeled build need not be the fastest unpeeled one); and dither-1s-hbk,
# dither-1s-hbp on the unit's K split (an H1U_KSPLIT build, the .so HB_H1U_KSPLIT names). The
# dither-1s arms' gate also runs CA's H1V3 h1tile.reference at H1V3_SHA (default bd702a2c46), and dither-1s-hbp's at
# H1V3_PEEL_SHA too (default 4733b9ab0c, v3 with the peel), each from a sent h1v3-SHA.tar.gz (`git archive SHA verity |
# gzip`; a --cwd source tree has no history), else the source's git.
# Env: HB_WHO (the lease owner, default kernels-dither-hb), DITHER_NSYS=1 (sbench's nsys capture of the gate's kernels; kernels.py counts each dither arm's inside its NVTX
# range, so the source must be dither-rtx's f2c0071c7c or later), LEASE (default "--max-min 14"), RUN_TIMEOUT (seconds
# inside the lease, default 820), VERITY_BENCH_VENV (default /workspace/cp/venv312), NVCC (default CUDA 13.0's).
# Exit: sbench's (3 an engine failed to build, 4 the gate failed), 2 a build failed or a shipped library is missing or
# another source's.
set -uo pipefail
MODE=${1:?usage: hb_served.sh build ARCH[,ARCH] | hb_served.sh run sm_90a|sm_100a [sbench args]}
ARCH=${2:?arch}
shift 2
D=${RESEARCH_RUN_DIR:?}
OUT=$D/out
INPUTS=$D/inputs
ROOT=$(pwd -P)
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
CUDA=$(cd "$(dirname "$NVCC")/.." && pwd -P)
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
SHA=${RESEARCH_SOURCE_SHA:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)}
DITHER=benchmarks/hash/dither
mkdir -p "$OUT/env"
srchash() { (cd "$ROOT" && cat "$@" | sha256sum | cut -c1-16); }
[ -n "${HB_H1U:-}" ] && HB_H1U=$INPUTS/$(basename "$HB_H1U")
[ -n "${HB_H1U_PEEL:-}" ] && HB_H1U_PEEL=$INPUTS/$(basename "$HB_H1U_PEEL")
[ -n "${HB_H1U_KSPLIT:-}" ] && HB_H1U_KSPLIT=$INPUTS/$(basename "$HB_H1U_KSPLIT")
sha256sum "$0" "$INPUTS"/hb_sbench.py "$INPUTS"/hb_side.py "$INPUTS"/hb_one.py "$INPUTS"/hb_peel.py ${HB_H1U:+"$HB_H1U"} \
  ${HB_H1U_PEEL:+"$HB_H1U_PEEL"} ${HB_H1U_KSPLIT:+"$HB_H1U_KSPLIT"} 2>/dev/null | sed "s#$INPUTS/##" > "$OUT/env/wrapper.txt"
echo "source $SHA" >> "$OUT/env/wrapper.txt"

if [ "$MODE" = build ]; then
  "$NVCC" --version > "$OUT/env/nvcc.txt" 2>&1
  build_one() {
    local arch=$1 lib=$2 dir="$OUT/lib/$1" deps
    local flags=(-O3 -std=c++17 -gencode "arch=compute_${arch#sm_},code=$arch" -Xptxas -v -Xcompiler -fPIC -shared)
    "$NVCC" "${flags[@]}" -o "$dir/lib$lib.so" "$DITHER/$lib.cu" > "$dir/build-$lib.log" 2>&1 \
      || { echo "build $arch $lib failed:"; grep -v "^ptxas info" "$dir/build-$lib.log" | tail -30; return 2; }
    "$CUDA/bin/cuobjdump" --dump-resource-usage "$dir/lib$lib.so" > "$dir/lib$lib.res.txt" 2>&1
    "$CUDA/bin/cuobjdump" -sass "$dir/lib$lib.so" | gzip > "$dir/lib$lib.sass.gz"
    "$NVCC" -M -std=c++17 "${flags[@]:2:2}" "$DITHER/$lib.cu" > "$dir/lib$lib.deps" 2>&1 || { cat "$dir/lib$lib.deps"; return 2; }
    deps=$(tr ' \\' '\n\n' < "$dir/lib$lib.deps" | grep -v ':$' | grep . | xargs -r realpath -m | while read -r f; do [ -f "$f" ] && echo "$f"; done | grep "^$ROOT/" | sed "s#^$ROOT/##" | sort -u)
    echo "src lib$lib $(srchash $deps) $(echo $deps)" > "$dir/lib$lib.line"
  }
  brc=0
  for arch in ${ARCH//,/ }; do
    mkdir -p "$OUT/lib/$arch"
    for lib in h1c h1f h1o; do build_one "$arch" "$lib" & done
  done
  while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || brc=2; done
  wait || brc=2
  [ $brc = 0 ] || exit 2
  for arch in ${ARCH//,/ }; do
    dir="$OUT/lib/$arch"
    { echo "$SHA $arch"; cat "$dir"/*.line; } > "$dir/BUILD.txt"
    rm "$dir"/*.line
    grep -h -A1 "h1f_k\|h1r_units_k\|h1o_units_k\|h1c" "$dir"/*.res.txt | grep -o "Function [^:]*\|REG:[0-9]* STACK:[0-9]* SHARED:[0-9]* LOCAL:[0-9]*" | paste - - | cut -c1-150
  done
  exit 0
fi

[ "$MODE" = run ] || { echo "hb_served.sh: no mode $MODE" >&2; exit 2; }
[ -f "$INPUTS/BUILD.txt" ] || { echo "hb_served.sh: no inputs/BUILD.txt: --send the build's lib/$ARCH/BUILD.txt" >&2; exit 2; }
[ -f "$INPUTS/hb_sbench.py" ] || { echo "hb_served.sh: --send benchmarks/hash/dither/hb_sbench.py" >&2; exit 2; }
case " $* " in *dither-hb*)
  [ -f "$INPUTS/hb_side.py" ] && [ -f "${HB_H1U:-/nonexistent}" ] || {
    echo "hb_served.sh: dither-hb needs --send benchmarks/hash/dither/hb_side.py, a job.sh build's h1u .so and --env HB_H1U=ITS_NAME" >&2
    exit 2; }
  export HB_H1U ;;
esac
case " $* " in *dither-1s-hb*)
  [ -f "$INPUTS/hb_one.py" ] && [ -f "$INPUTS/hb_side.py" ] && [ -f "${HB_H1U:-/nonexistent}" ] || {
    echo "hb_served.sh: dither-1s-hb needs --send benchmarks/hash/dither/hb_one.py and hb_side.py, a job.sh build's h1u .so and --env HB_H1U=ITS_NAME" >&2
    exit 2; }
  export HB_H1U ;;
esac
case " $* " in *dither-1s-hbp*)
  [ -f "$INPUTS/hb_peel.py" ] && [ -f "${HB_H1U_PEEL:-${HB_H1U:-/nonexistent}}" ] || {
    echo "hb_served.sh: dither-1s-hbp needs --send benchmarks/hash/dither/hb_peel.py too (and HB_H1U_PEEL's .so, if set)" >&2
    exit 2; }
  [ -n "${HB_H1U_PEEL:-}" ] && export HB_H1U_PEEL ;;
esac
case " $* " in *dither-1s-hbk*)
  [ -f "$INPUTS/hb_peel.py" ] && [ -f "${HB_H1U_KSPLIT:-/nonexistent}" ] || {
    echo "hb_served.sh: dither-1s-hbk needs --send benchmarks/hash/dither/hb_peel.py and an H1U_KSPLIT build's .so, --env HB_H1U_KSPLIT=ITS_NAME" >&2
    exit 2; }
  export HB_H1U_KSPLIT ;;
esac
cp "$INPUTS/BUILD.txt" "$OUT/env/BUILD.txt"
read -r bsha barch < "$INPUTS/BUILD.txt"
[ "$barch" = "$ARCH" ] || { echo "hb_served.sh: the libraries are $barch's, not $ARCH's" >&2; exit 2; }
for lib in libh1c libh1f libh1o; do
  [ -f "$INPUTS/$lib.so" ] || { echo "hb_served.sh: no inputs/$lib.so: --send the build's lib/$ARCH/$lib.so" >&2; exit 2; }
  read -r _ _ h paths <<< "$(grep "^src $lib " "$INPUTS/BUILD.txt")"
  [ -n "$h" ] && [ "$(srchash $paths)" = "$h" ] || {
    echo "hb_served.sh: $lib.so was built (source $bsha) from files that differ at $SHA: rebuild" >&2; exit 2; }
done
case $ARCH in
  sm_90a) FI=9.0a ;;
  sm_100a) FI=10.0a ;;
  *) echo "hb_served.sh: no arch $ARCH" >&2; exit 2 ;;
esac
S=$(mktemp -d "${TMPDIR:-/tmp}/dither-hb-served.XXXXXX")
export VLLM_CACHE_ROOT=$S/vllm-cache TORCHINDUCTOR_CACHE_DIR=$S/inductor-cache
trap 'rm -rf "$S"' EXIT
export DITHER_H1C=$INPUTS/libh1c.so DITHER_H1F=$INPUTS/libh1f.so DITHER_H1O=$INPUTS/libh1o.so VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_FLASHINFER_SAMPLER=0
# CA's H1V3 (cursor/h1-v3-1172) at its pinned head, as the source's sbench.sh extracts it: dither_one's gate runs its
# h1tile.reference(stream=True) on a few units
H1V3_SHA=${H1V3_SHA:-bd702a2c46}
mkdir -p "$S/h1v3"
if [ -f "$INPUTS/h1v3-$H1V3_SHA.tar.gz" ]; then
  tar -xzf "$INPUTS/h1v3-$H1V3_SHA.tar.gz" -C "$S/h1v3"
else
  git -C "$ROOT" archive "$H1V3_SHA" verity 2>/dev/null | tar -x -C "$S/h1v3" 2>/dev/null
fi
[ -f "$S/h1v3/verity/core/protocols/pouw/circuit/h1tile.py" ] \
  && export DITHER_H1V3_TREE=$S/h1v3 DITHER_H1V3_SHA=$H1V3_SHA || echo "hb_served: no H1V3 tree at $H1V3_SHA"
echo "h1v3 ${DITHER_H1V3_SHA:-none}" >> "$OUT/env/wrapper.txt"
# H1V3 with the peel (z = A B^T, CA 12:03Z) at H1V3_PEEL_SHA: dither-1s-hbp's gate holds its z and receipts to that tree's
# h1tile.reference itself; dither-1s-hb, which serves the dithered product, stays on H1V3_SHA's
H1V3_PEEL_SHA=${H1V3_PEEL_SHA:-4733b9ab0c}
mkdir -p "$S/h1v3p"
if [ -f "$INPUTS/h1v3-$H1V3_PEEL_SHA.tar.gz" ]; then
  tar -xzf "$INPUTS/h1v3-$H1V3_PEEL_SHA.tar.gz" -C "$S/h1v3p"
else
  git -C "$ROOT" archive "$H1V3_PEEL_SHA" verity 2>/dev/null | tar -x -C "$S/h1v3p" 2>/dev/null
fi
grep -qs "^def peel_seg" "$S/h1v3p/verity/core/protocols/pouw/circuit/h1tile.py" \
  && export HB_H1V3_PEEL_TREE=$S/h1v3p HB_H1V3_PEEL_SHA=$H1V3_PEEL_SHA
echo "h1v3_peel ${HB_H1V3_PEEL_SHA:-none}" >> "$OUT/env/wrapper.txt"
case " $* " in *dither-1s-hbp*|*dither-1s-hbk*)
  [ -n "${HB_H1V3_PEEL_TREE:-}" ] || {
    echo "hb_served.sh: dither-1s-hbp needs H1V3 with the peel: --send h1v3-$H1V3_PEEL_SHA.tar.gz (git archive $H1V3_PEEL_SHA verity | gzip)" >&2
    exit 2; } ;;
esac
export PATH=$(dirname "$PY"):$CUDA/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=$FI HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/$DITHER HB_DITHER_DIR=$ROOT/$DITHER
NSYS=()
if [ "${DITHER_NSYS:-}" = 1 ]; then
  NSYS=("$CUDA/bin/nsys" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop -o "$S/gate" --force-overwrite true)
fi
cd integrations/vllm
printf '%s\n' "$PY $INPUTS/hb_sbench.py --out $OUT --scratch $S $*" > "$OUT/env/cmd.txt"
read -r -a lease <<< "${LEASE:---max-min 14}"
echo "hb_served: asking for one GPU at $(date -u +%H:%M:%SZ)"
GPU_LEASE_WHO=${HB_WHO:-kernels-dither-hb} gpu-lease 1 --wait "${lease[@]}" -- bash -c '
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$0/clocks.before.csv"
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -l 2 > "$0/smi.csv" 2>/dev/null &
  smi=$!
  timeout "${RUN_TIMEOUT:-820}" "$@"
  r=$?
  kill "$smi" 2>/dev/null
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$0/clocks.after.csv"
  exit $r' "$OUT" "${NSYS[@]}" "$PY" "$INPUTS/hb_sbench.py" --out "$OUT" --scratch "$S" "$@"
rc=$?
echo "hb_served: exited $rc at $(date -u +%H:%M:%SZ)"
if [ -f "$S/gate.nsys-rep" ]; then
  "$CUDA/bin/nsys" stats --report cuda_gpu_kern_sum --format csv -o "$OUT/gate" "$S/gate.nsys-rep" > "$OUT/nsys-stats.log" 2>&1 \
    || echo "hb_served: nsys stats rc $?"
  "$PY" "$ROOT/$DITHER/kernels.py" "$S/gate.sqlite" | tee "$OUT/kernels.json"
fi
exit "$rc"
