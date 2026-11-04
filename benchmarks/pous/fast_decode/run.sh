#!/usr/bin/env bash
# The fast-decode prototypes on one GPU (cwd = the source tree): builds every library outside the lease, then runs bench.py
# under gpu-lease 1 --wait --max-min MAXMIN (default 20). Results go to $RESEARCH_RUN_DIR/out.
#
#   RESEARCH_RUN_DIR=DIR MAXMIN=20 GPU_LEASE_WHO=ra-fast-decode taskset -c 0-119 bash benchmarks/pous/fast_decode/run.sh
#
# The libraries: the tensor-core decode (tcsq.cu) at each width and Karatsuba depth in TC (PER:DEPTH:MINB), the NTT decode
# (nttsq.cu, P2 v4's width) unless NTT=0, the decode in use (p2_v1/p2v1.cu, CGBN with p2sq.cuh's squaring) at 16,448 bits,
# and CGBN's generic decode at every width in CGBN_WIDTHS. A library whose build fails is left out and its log kept.
# ENV_DIR holds the shared torch venv and CGBN (pod.sh's WORK/pous-bench, read-only, as p2_v1/climb.sh uses them).
# BYTES, REPS, GATE and CONTEND (space-separated library names) pass through.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
CG=$ENV_DIR/CGBN
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
[ "$(git -C "$CG" rev-parse HEAD)" = "$(cat benchmarks/pous/p2_gpu/CGBN.commit)" ] || { echo "CGBN is not at the pinned commit"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
GMPINC=$("$PY" -c 'import gmpy2, os; print(os.path.dirname(gmpy2.__file__))')
ARCH=${ARCH:-sm_120}
TC=${TC:-"16:0:8 16:1:6 16:2:6 8:0:6 8:2:6 4:0:8 4:1:8 2:0:8"}
CGBN_WIDTHS=${CGBN_WIDTHS:-"16448 8256 4160 2112"}
FD=benchmarks/pous/fast_decode
B=$(mktemp -d /tmp/fast-decode.XXXXXX); trap 'rm -rf "$B"' EXIT
NAMES=(); KINDS=(); PIDS=()
build() {   # build NAME KIND SOURCE FLAGS...
  local name=$1 kind=$2 src=$3; shift 3
  nvcc -O3 -arch="$ARCH" -std=c++17 "$@" -Xcompiler -fPIC -shared -o "$B/$name.so" "$src" -Xptxas -v > "$O/$name.ptxas" 2>&1 &
  PIDS+=($!); NAMES+=("$name"); KINDS+=("$kind")
}
for cfg in $TC; do
  IFS=: read -r per depth minb <<< "$cfg"
  build "tc-p$per-d$depth" lib $FD/tcsq.cu -DFD_PER="$per" -DFD_DEPTH="$depth" -DFD_MINB="$minb"
done
[ "${NTT:-1}" = 1 ] && build ntt-p16 lib $FD/nttsq.cu -DFD_PER=16
CGF="-DP2_TPI=32 -DXMP_WMAD -DP2_PARTS=2 -DP2_SYM -DP2_TPB=64 -DP2_LIBRARY -I $GMPINC -I $CG/include"
# shellcheck disable=SC2086
build cgbn-fast-16448 cgbn benchmarks/pous/p2_v1/p2v1.cu $CGF -DP2_BITS=16448 -DP2_FAST -DP2_KEYS4 -DP2SQ_KARA
for w in $CGBN_WIDTHS; do
  # shellcheck disable=SC2086
  build "cgbn-generic-$w" cgbn benchmarks/pous/p2_v1/p2v1.cu $CGF -DP2_BITS="$w" -DP2_KEYS4
done
ARGS=()
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    ARGS+=("--${KINDS[$i]}" "${NAMES[$i]}=$B/${NAMES[$i]}.so")
  else
    echo "build failed: ${NAMES[$i]}"; tail -20 "$O/${NAMES[$i]}.ptxas"
  fi
done
CONT=()
for c in ${CONTEND:-tc-p16-d0 cgbn-fast-16448}; do CONT+=(--contend "$c"); done
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-20}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '"$@"; rc=$?
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1; exit $rc' _ \
  "$PY" $FD/bench.py "${ARGS[@]}" "${CONT[@]}" --bytes "${BYTES:-1073741824}" --reps "${REPS:-7}" --gate "${GATE:-48}" \
  --out "$O" --result-dir "$RESEARCH_RUN_DIR"
