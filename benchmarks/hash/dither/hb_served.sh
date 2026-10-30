#!/usr/bin/env bash
# Dither's served standard number on the H200 and the B200: the fused served linear (h1f.cu, h1c.cu, dither_fused.py,
# dither_one.py) and the served bench (sbench.py) from the run's source tree. A tree with hb_sbench.py and its hb_*.py
# serves them from itself; an older one needs them in $RESEARCH_RUN_DIR/inputs, as the example below puts them. DITHER_NSYS=prof captures sbench's prof op instead of the gate (steps.py, stepprof.py), and an
# exact-fused or -xl arm runs only after exactlin's kgate.py passes in the same lease (HB_KGATE=0|1 overrides).
# Two modes, so no CPU phase runs on the GPU host:
#   build ARCH[,ARCH]  (a CPU host) libh1c.so, libh1f.so and libh1o.so for each arch into $OUT/lib/ARCH, their SASS and resource
#                      usage, and BUILD.txt: the source commit and arch, and per library the sha256 of the files nvcc -M names
#                      (and its nvcc flags); with HB_KC_FLAGS set (even empty) the dither-a-hbc arm's h1u_kc.so too, and
#                      with HB_KC_H1C_FLAGS its libh1c_kc.so, built as sbench.sh builds them (HB_H1U_KC, DITHER_H1C_KC);
#                      with HB_KC_FLAGS_B a second h1u_kc, h1u_kcb.so (HB_H1U_KC_B: sbench.py's ARM-kcb arms)
#   run ARCH [ARGS]    (an H200 for sm_90a, a B200 for sm_100a) sbench.py through hb_sbench.py on one leased GPU for at most
#                      14 minutes, every shipped library checked against the source's files first; ARGS go to sbench.py
#                      (HB_SCRIPT=h1agate.py, or another script of this directory, runs it in hb_sbench.py's place with
#                      the same libraries, environment and lease)
#
#   git worktree add --detach /tmp/rtx-src <rtx commit> && cd /tmp/rtx-src   (H: this tree's benchmarks/hash/dither)
#   RESEARCH_RUN_DIR=B bash H/hb_served.sh build sm_90a,sm_100a              (a CPU host, up to 20 minutes)
#   mkdir -p R/inputs && cp B/out/lib/sm_90a/* H/hb_sbench.py R/inputs/
#   RESEARCH_RUN_DIR=R DITHER_NSYS=1 bash H/hb_served.sh run sm_90a --arms stock,exact,dither --gate ...
#     (an H200 under a GPU lease, up to 60 minutes)
#
# The stock arm is native compiled vLLM int8 (compressed-tensors W8A8) on TritonInt8ScaledMMLinearKernel, whose dequant
# order is the int8-exact path's: sbench disables vLLM's first choice, CutlassInt8ScaledMMLinearKernel, which refuses int8
# on sm_100a ("Int8 not supported on SM100") and sm_120. On sm_90a hb_sbench.py's stock-cutlass arm keeps it (vLLM's
# default there). Every arm samples without FlashInfer's JIT sampler, as served_h1 and native.sh do.
# hb_sbench.py adds five arms: stock-cutlass (sm_90a); dither-hb, dither-side's receipts on h1u.cu's kh_unit_ws
# (hb_side.py and an h1u .so from job.sh's build, both sent; HB_H1U names the .so); dither-1s-hb,
# dither-1s with h1u.cu's one-stream kh_one in place of h1o_units (hb_one.py and hb_side.py, the same .so);
# dither-1s-hbp, dither-1s-hb peeled (kh_one_peel: z = A B^T, y today's; hb_peel.py's gate, sent too), from the .so
# HB_H1U_PEEL names if it's set (the fastest peeled build need not be the fastest unpeeled one); and dither-1s-hbk,
# dither-1s-hbp on the unit's K split (an H1U_KSPLIT build, the .so HB_H1U_KSPLIT names). The
# dither-1s arms' gate also runs H1V3's h1tile.reference: this source's own (PouwH1Tile_v4, ungated, z = A B^T) unless
# H1V3_SHA names a commit (one before the peel, as bd702a2c46, z = X (B + c)^T, holds an unpeeled arm's z to H1V3's code
# directly; against a peeled tree that z is compared through the CPU's unpeel), and dither-1s-hbp's and -hbk's the tree
# with the peel's too, the source's unless H1V3_PEEL_SHA names one; each from a sent h1v3-SHA.tar.gz (`git archive SHA
# verity | gzip`; a --cwd source tree has no history), else the source's git.
# Inside the lease everything runs on the leased GPU's NUMA node (numabind.py: that node's CPUs among the run's, and its
# memory; the source's, else one sent), or unbound with DITHER_NUMA=0 or without numactl, a node or numabind.py;
# env/numa.json, the `numa:` line and summary.json's "numa" say which.
# Env: HB_WHO (the lease owner, default dither/ and the run's id), DITHER_NSYS=1 (sbench's nsys capture of the gate's kernels; kernels.py counts each dither arm's inside its NVTX
# range, so the source must be dither-rtx's f2c0071c7c or later; HB_KEEP_SQLITE=1 keeps its sqlite, gzipped), LEASE (default "--max-min 14"), RUN_TIMEOUT (seconds
# inside the lease, default 820), VERITY_BENCH_VENV (default /workspace/cp/venv312), NVCC (default CUDA 13.0's),
# POUW_CHALLENGER and POUW_CHALLENGER_SESSION, or DITHER_CHALLENGERS (the windows' coins off the node, sessions from
# infra: sbench.py's "Coins"), J10R_SHA (a DITHER_SO_UNITS=j10r arm's J10R Definition: a sent j10r-SHA.tar.gz, else the
# repository's git), HB_H1U_KC, HB_H1U_KC_B and DITHER_H1C_KC (the KC builds, each held to its BUILD.txt line).
# Exit: sbench's (3 an engine failed to build, 4 the gate failed), 2 a build failed or a shipped library is missing or
# another source's, 15 a challenger's port never opened or a session lacks its id or address.
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
[ -n "${HB_H1U_KRONLY:-}" ] && HB_H1U_KRONLY=$INPUTS/$(basename "$HB_H1U_KRONLY") && export HB_H1U_KRONLY
[ -n "${HB_H1U_KC:-}" ] && HB_H1U_KC=$INPUTS/$(basename "$HB_H1U_KC") && export HB_H1U_KC
[ -n "${HB_H1U_KC_B:-}" ] && HB_H1U_KC_B=$INPUTS/$(basename "$HB_H1U_KC_B") && export HB_H1U_KC_B
[ -n "${DITHER_H1C_KC:-}" ] && DITHER_H1C_KC=$INPUTS/$(basename "$DITHER_H1C_KC") && export DITHER_H1C_KC
# the hb_*.py files: the source tree's where it has them (it is first on sbench's sys.path), else the ones sent
HB=$ROOT/$DITHER
[ -f "$HB/hb_sbench.py" ] || HB=$INPUTS
sha256sum "$0" "$HB"/hb_sbench.py "$HB"/hb_side.py "$HB"/hb_one.py "$HB"/hb_peel.py ${HB_H1U:+"$HB_H1U"} \
  ${HB_H1U_PEEL:+"$HB_H1U_PEEL"} ${HB_H1U_KSPLIT:+"$HB_H1U_KSPLIT"} ${HB_H1U_KRONLY:+"$HB_H1U_KRONLY"} ${HB_H1U_KC:+"$HB_H1U_KC"} ${HB_H1U_KC_B:+"$HB_H1U_KC_B"} ${DITHER_H1C_KC:+"$DITHER_H1C_KC"} 2>/dev/null | sed "s#$INPUTS/##; s#$ROOT/##" > "$OUT/env/wrapper.txt"
echo "source $SHA" >> "$OUT/env/wrapper.txt"

if [ "$MODE" = build ]; then
  "$NVCC" --version > "$OUT/env/nvcc.txt" 2>&1
  build_one() {  # ARCH NAME SRC [NVCC FLAGS]: $OUT/lib/ARCH/NAME.so, and its BUILD.txt lines (src, and flags when it has some)
    local arch=$1 name=$2 src=$3 dir="$OUT/lib/$1" deps
    shift 3
    local gen=(-gencode "arch=compute_${arch#sm_},code=$arch")
    "$NVCC" -O3 -std=c++17 "${gen[@]}" -Xptxas -v -Xcompiler -fPIC -shared "$@" -o "$dir/$name.so" "$src" > "$dir/build-$name.log" 2>&1 \
      || { echo "build $arch $name failed:"; grep -v "^ptxas info" "$dir/build-$name.log" | tail -30; return 2; }
    "$CUDA/bin/cuobjdump" --dump-resource-usage "$dir/$name.so" > "$dir/$name.res.txt" 2>&1
    "$CUDA/bin/cuobjdump" -sass "$dir/$name.so" | gzip > "$dir/$name.sass.gz"
    "$NVCC" -M -std=c++17 "${gen[@]}" "$@" "$src" > "$dir/$name.deps" 2>&1 || { cat "$dir/$name.deps"; return 2; }
    deps=$(tr ' \\' '\n\n' < "$dir/$name.deps" | grep -v ':$' | grep . | xargs -r realpath -m | while read -r f; do [ -f "$f" ] && echo "$f"; done | grep "^$ROOT/" | sed "s#^$ROOT/##" | sort -u)
    { echo "src $name $(srchash $deps) $(echo $deps)"; [ $# = 0 ] || echo "flags $name $*"; } > "$dir/$name.line"
  }
  # the dither-a-hbc arm's two (sbench.sh's builds of them): h1u_kc.so, the K split built receipts only with HB_KC_FLAGS
  # (set, even empty), and libh1c_kc.so with HB_KC_H1C_FLAGS (non-empty); HB_H1U_KC and DITHER_H1C_KC name them at run
  read -r -a kc <<< "${HB_KC_FLAGS:-}"
  read -r -a kh <<< "${HB_KC_H1C_FLAGS:-}"
  read -r -a kcb <<< "${HB_KC_FLAGS_B:-}"
  brc=0
  for arch in ${ARCH//,/ }; do
    mkdir -p "$OUT/lib/$arch"
    for lib in h1c h1f h1o; do build_one "$arch" "lib$lib" "$DITHER/$lib.cu" & done
    if [ -n "${HB_KC_FLAGS+set}" ]; then
      build_one "$arch" h1u_kc "$DITHER/h1u.cu" --expt-relaxed-constexpr -DH1U_KSPLIT=4 -DH1K_AUTO=1 -DH1K_RONLY=1 "${kc[@]}" &
    fi
    if [ -n "${HB_KC_FLAGS_B:-}" ]; then
      build_one "$arch" h1u_kcb "$DITHER/h1u.cu" --expt-relaxed-constexpr -DH1U_KSPLIT=4 -DH1K_AUTO=1 -DH1K_RONLY=1 "${kcb[@]}" &
    fi
    if [ ${#kh[@]} -gt 0 ]; then build_one "$arch" libh1c_kc "$DITHER/h1c.cu" "${kh[@]}" & fi
  done
  while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || brc=2; done
  wait || brc=2
  [ $brc = 0 ] || exit 2
  for arch in ${ARCH//,/ }; do
    dir="$OUT/lib/$arch"
    { echo "$SHA $arch"; cat "$dir"/*.line; } > "$dir/BUILD.txt"
    rm "$dir"/*.line
    grep -h -A1 "h1f_k\|h1r_units_k\|h1o_units_k\|h1c\|h1u\|h1k\|h1jr" "$dir"/*.res.txt | grep -o "Function [^:]*\|REG:[0-9]* STACK:[0-9]* SHARED:[0-9]* LOCAL:[0-9]*" | paste - - | cut -c1-150
    grep -h "spill" "$dir"/build-*.log | grep -v " 0 bytes spill stores, 0 bytes spill loads" | sort | uniq -c | head -8
  done
  exit 0
fi

[ "$MODE" = run ] || { echo "hb_served.sh: no mode $MODE" >&2; exit 2; }
[ -f "$INPUTS/BUILD.txt" ] || { echo "hb_served.sh: no inputs/BUILD.txt: copy the build's lib/$ARCH/BUILD.txt into $INPUTS" >&2; exit 2; }
[ -f "$HB/hb_sbench.py" ] || { echo "hb_served.sh: no hb_sbench.py: copy benchmarks/hash/dither/hb_sbench.py into $INPUTS" >&2; exit 2; }
case " $* " in *dither-hb*)
  [ -f "$HB/hb_side.py" ] && [ -f "${HB_H1U:-/nonexistent}" ] || {
    echo "hb_served.sh: dither-hb needs benchmarks/hash/dither/hb_side.py and a job.sh build's h1u .so in $INPUTS, and HB_H1U=ITS_NAME" >&2
    exit 2; }
  export HB_H1U ;;
esac
case " $* " in *dither-1s-hb*)
  [ -f "$HB/hb_one.py" ] && [ -f "$HB/hb_side.py" ] && [ -f "${HB_H1U:-/nonexistent}" ] || {
    echo "hb_served.sh: dither-1s-hb needs benchmarks/hash/dither/hb_one.py, hb_side.py and a job.sh build's h1u .so in $INPUTS, and HB_H1U=ITS_NAME" >&2
    exit 2; }
  export HB_H1U ;;
esac
case " $* " in *dither-1s-hbp*)
  [ -f "$HB/hb_peel.py" ] && [ -f "${HB_H1U_PEEL:-${HB_H1U:-/nonexistent}}" ] || {
    echo "hb_served.sh: dither-1s-hbp needs benchmarks/hash/dither/hb_peel.py in $INPUTS too (and HB_H1U_PEEL's .so, if set)" >&2
    exit 2; }
  [ -n "${HB_H1U_PEEL:-}" ] && export HB_H1U_PEEL ;;
esac
case " $* " in *dither-1s-hbk*)
  [ -f "$HB/hb_peel.py" ] && [ -f "${HB_H1U_KSPLIT:-/nonexistent}" ] || {
    echo "hb_served.sh: dither-1s-hbk needs benchmarks/hash/dither/hb_peel.py and an H1U_KSPLIT build's .so in $INPUTS, and HB_H1U_KSPLIT=ITS_NAME" >&2
    exit 2; }
  export HB_H1U_KSPLIT ;;
esac
cp "$INPUTS/BUILD.txt" "$OUT/env/BUILD.txt"
read -r bsha barch < "$INPUTS/BUILD.txt"
[ "$barch" = "$ARCH" ] || { echo "hb_served.sh: the libraries are $barch's, not $ARCH's" >&2; exit 2; }
for lib in libh1c libh1f libh1o ${HB_H1U_KC:+$(basename "$HB_H1U_KC" .so)} ${HB_H1U_KC_B:+$(basename "$HB_H1U_KC_B" .so)} \
           ${DITHER_H1C_KC:+$(basename "$DITHER_H1C_KC" .so)}; do
  [ -f "$INPUTS/$lib.so" ] || { echo "hb_served.sh: no inputs/$lib.so: copy the build's lib/$ARCH/$lib.so into $INPUTS" >&2; exit 2; }
  read -r _ _ h paths <<< "$(grep "^src $lib " "$INPUTS/BUILD.txt")"
  [ -n "$h" ] && [ "$(srchash $paths)" = "$h" ] || {
    echo "hb_served.sh: $lib.so was built (source $bsha) from files that differ at $SHA, or BUILD.txt has no src line for it: rebuild (HB_KC_FLAGS / HB_KC_H1C_FLAGS for the KC builds)" >&2; exit 2; }
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
# H1V3: the dither gates run its h1tile.reference(stream=True) on a few units (DITHER_H1V3_TREE), and hb_peel's
# (dither-1s-hbp and -hbk) the tree with the peel's too (HB_H1V3_PEEL_TREE); a -bG arm's gate reads credit's banded reference
# in their place. Both trees are this source's own (PouwH1Tile_v4, ungated) unless H1V3_SHA or H1V3_PEEL_SHA names a commit
h1v3_tree() {  # SHA DIR [PREFIX]: DIR/verity from the sent PREFIX-SHA.tar.gz (h1v3), else this repository's git; exit 14 without one
  local p=${3:-h1v3}
  mkdir -p "$2"
  if [ -f "$INPUTS/$p-$1.tar.gz" ]; then tar -xzf "$INPUTS/$p-$1.tar.gz" -C "$2"
  else git -C "$ROOT" archive "$1" verity 2>/dev/null | tar -x -C "$2" 2>/dev/null; fi
  [ -f "$2/verity/ml/protocols/pouw/circuit/h1tile.py" ] || { echo "hb_served: ${p^^}_NO_TREE at $1" >&2; exit 14; }
}
if [ -n "${H1V3_SHA:-}" ]; then
  h1v3_tree "$H1V3_SHA" "$S/h1v3"; export DITHER_H1V3_TREE=$S/h1v3 DITHER_H1V3_SHA=$H1V3_SHA
else
  export DITHER_H1V3_TREE=$ROOT DITHER_H1V3_SHA=${RESEARCH_SOURCE_SHA:-source}
fi
echo "h1v3 $DITHER_H1V3_SHA" >> "$OUT/env/wrapper.txt"
if [ -n "${H1V3_PEEL_SHA:-}" ]; then
  h1v3_tree "$H1V3_PEEL_SHA" "$S/h1v3p"; export HB_H1V3_PEEL_TREE=$S/h1v3p HB_H1V3_PEEL_SHA=$H1V3_PEEL_SHA
else
  export HB_H1V3_PEEL_TREE=$ROOT HB_H1V3_PEEL_SHA=${RESEARCH_SOURCE_SHA:-source}
fi
echo "h1v3_peel $HB_H1V3_PEEL_SHA" >> "$OUT/env/wrapper.txt"
# J10R's Definition (h1tile.j10r_reference) for a DITHER_SO_UNITS=j10r arm's gate (j10r_so.py), as sbench.sh's J10R_SHA
if [ -n "${J10R_SHA:-}" ]; then
  h1v3_tree "$J10R_SHA" "$S/j10r" j10r; export DITHER_J10R_TREE=$S/j10r DITHER_J10R_SHA=$J10R_SHA
  echo "j10r $J10R_SHA" >> "$OUT/env/wrapper.txt"
fi
case " $* " in *dither-1s-hbp*|*dither-1s-hbk*)
  grep -qs "^def peel_seg" "$HB_H1V3_PEEL_TREE/verity/ml/protocols/pouw/circuit/h1tile.py" || {
    echo "hb_served.sh: dither-1s-hbp needs H1V3 with the peel (peel_seg) at $HB_H1V3_PEEL_SHA" >&2
    exit 2; } ;;
esac
# the dither windows' coins are the Lean verifier of record's (pouw.verifier.coins): built before the lease, not inside it,
# under a lock, since runs of one commit share its source tree and concurrent lake builds there corrupt each other's .c/.o
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
(cd verity/core/protocols/pouw/lean/check && mkdir -p .lake && flock .lake/build.lock lake build pouw-check-difftest) \
  > "$OUT/lean-coins-build.log" 2>&1 \
  || { echo "hb_served: LEAN_BUILD_FAIL"; tail -20 "$OUT/lean-coins-build.log"; exit 13; }
# coins off the node: every port the arms' windows connect to (POUW_CHALLENGER, DITHER_CHALLENGERS'
# {arm: {"session": ID, "address": HOST:PORT, "secret_file": F, "commitment": HEX, "rounds": R}}), each its own session from
# the challenger's gate first (`coins.py open`: reserved in a dev period, opened on its stand-in root, its secret in F on this
# node, HEX the root of the coin commitment its challenger signed, which its hello must answer, R its --rounds when not
# 16384), waited for with ss -ltn
# before the lease (never a probe connect: a challenger serves the first connection whose hello it accepts); none: in-process
# (not certified)
if [ -n "${POUW_CHALLENGER:-}${POUW_CHALLENGER_SESSION:-}" ] && { [ -z "${POUW_CHALLENGER:-}" ] || [ -z "${POUW_CHALLENGER_SESSION:-}" ]; }; then
  echo "hb_served: NO_CHALLENGER: a challenger session is POUW_CHALLENGER and POUW_CHALLENGER_SESSION together"; exit 15
fi
ats=$(python3 -c 'import json, os; m = json.loads(os.environ.get("DITHER_CHALLENGERS") or "{}"); ok = isinstance(m, dict) and all(isinstance(v, dict) and {"address", "session"} <= set(v) <= {"address", "commitment", "rounds", "secret_file", "session"} and all(isinstance(x, str) and x for k, x in v.items() if k != "rounds") and type(v.get("rounds", 1)) is int and v.get("rounds", 1) > 0 for v in m.values()); print(*(v["address"] for v in m.values())) if ok else exit(1)') \
  || { echo "hb_served: NO_CHALLENGER: DITHER_CHALLENGERS is not {arm: {\"session\": ID, \"address\": HOST:PORT[, \"secret_file\": F][, \"commitment\": HEX][, \"rounds\": R]}}"; exit 15; }
ats=$(echo ${POUW_CHALLENGER:-} $ats)
for at in $ats; do
  for _ in $(seq $((${CHALLENGER_WAIT_S:-900} * 20))); do ss -Hltn "sport = :${at##*:}" | grep -q . && break; sleep 0.05; done
  ss -Hltn "sport = :${at##*:}" | grep -q . || { echo "hb_served: NO_CHALLENGER at $at"; exit 15; }
done
echo "coins ${ats:-in-process (not certified)}" | tee -a "$OUT/env/wrapper.txt"
export PATH=$(dirname "$PY"):$CUDA/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=$FI HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH=.:$ROOT:$ROOT/tools/research/src:$ROOT/tools/config:$ROOT/$DITHER HB_DITHER_DIR=$ROOT/$DITHER
NSYS=()
if [ "${DITHER_NSYS:-}" = 1 ] || [ "${DITHER_NSYS:-}" = prof ]; then
  NSYS=("$CUDA/bin/nsys" profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
        --capture-range=cudaProfilerApi --capture-range-end=stop -o "$S/gate" --force-overwrite true)
fi
cd integrations/vllm
SCRIPT=$HB/hb_sbench.py
[ -n "${HB_SCRIPT:-}" ] && SCRIPT=$ROOT/$DITHER/$HB_SCRIPT
printf '%s\n' "$PY $SCRIPT --out $OUT --scratch $S $*" > "$OUT/env/cmd.txt"
# an exact-fused or -xl arm is timed only after exactlin's kgate.py passes on this GPU, in the same lease (OUT/kgate)
case " $* " in *exact-fused*|*-xl[\ ,-]*) KGATE=1 ;; *) KGATE=0 ;; esac
export KGATE=${HB_KGATE:-$KGATE} KGATE_PY=$PY KGATE_SCRIPT=$ROOT/benchmarks/hash/exactlin/kgate.py
read -r -a lease <<< "${LEASE:---max-min 14}"
NUMA=()
for f in "$ROOT/$DITHER/numabind.py" "$INPUTS/numabind.py"; do [ -f "$f" ] && { NUMA=("$PY" "$f" "$OUT/env/numa.json"); break; }; done
if [ ${#NUMA[@]} = 0 ]; then
  export DITHER_NUMA_RECORD='{"bound": false, "node": null, "reason": "no numabind.py in the source or inputs"}'
  echo "$DITHER_NUMA_RECORD" > "$OUT/env/numa.json"
  echo "numa: unbound (no numabind.py in the source or inputs)"
fi
echo "hb_served: asking for one GPU at $(date -u +%H:%M:%SZ)"
GPU_LEASE_WHO=${HB_WHO:-dither/${RESEARCH_RUN_ID:-local}} gpu-lease 1 --wait "${lease[@]}" -- "${NUMA[@]}" bash -c '
  q=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,name,uuid
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$0/clocks.before.csv"
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -l 2 > "$0/smi.csv" 2>/dev/null &
  smi=$!
  if [ "$KGATE" = 1 ]; then
    timeout 900 "$KGATE_PY" "$KGATE_SCRIPT" --out "$0/kgate" --shapes all --models llama-3.1-8b,qwen3-8b --no-stock \
      > "$0/kgate.log" 2>&1
    g=$?
    tail -1 "$0/kgate.log"
    if [ $g -ne 0 ]; then echo "hb_served: kgate rc $g, nothing timed"; kill "$smi" 2>/dev/null; exit 4; fi
  fi
  timeout $(( ${RUN_TIMEOUT:-820} - SECONDS )) "$@"
  r=$?
  kill "$smi" 2>/dev/null
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=$q --format=csv > "$0/clocks.after.csv"
  exit $r' "$OUT" "${NSYS[@]}" "$PY" "$SCRIPT" --out "$OUT" --scratch "$S" "$@"
rc=$?
echo "hb_served: exited $rc at $(date -u +%H:%M:%SZ)"
if [ -f "$S/gate.nsys-rep" ]; then
  "$CUDA/bin/nsys" stats --report cuda_gpu_kern_sum --format csv -o "$OUT/gate" "$S/gate.nsys-rep" > "$OUT/nsys-stats.log" 2>&1 \
    || echo "hb_served: nsys stats rc $?"
  "$PY" "$ROOT/$DITHER/kernels.py" "$S/gate.sqlite" | tee "$OUT/kernels.json"
  "$PY" "$ROOT/benchmarks/hash/exactlin/steps.py" "$S/gate.sqlite" > "$OUT/steps.json" || echo "hb_served: steps.py rc $?"
  "$PY" "$ROOT/$DITHER/stepprof.py" "$S/gate.sqlite" > "$OUT/stepprof.json" \
    && "$PY" "$ROOT/$DITHER/stepprof.py" --table "$OUT/stepprof.json" | tee "$OUT/stepprof.txt" \
    || echo "hb_served: stepprof.py rc $?"
  [ "$(stat -c %s "$S/gate.nsys-rep")" -lt 209715200 ] && cp "$S/gate.nsys-rep" "$OUT/"
  [ "${HB_KEEP_SQLITE:-0}" = 1 ] && [ "$(stat -c %s "$S/gate.sqlite")" -lt 524288000 ] && gzip -c "$S/gate.sqlite" > "$OUT/gate.sqlite.gz"
fi
exit "$rc"
