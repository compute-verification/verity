#!/usr/bin/env bash
# The wide word codecs' benchmark on one GPU (cwd = the source tree): builds wide.cu for each field in both squaring
# forms and wc.cu's plain kernels (native) outside the lease, checks the host build of wide.cu's arithmetic against
# wide.py, then, under one gpu-lease 1 --wait --max-min MAXMIN (default 25), wide_bench.py's gates and timings. Results go
# to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 64 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-codec-wide \
#     -- taskset -c 48-95 bash benchmarks/word_codec/wide_run.sh
#
# ENV_DIR holds the shared torch venv (benchmarks/pous/pod.sh's WORK). BENCH_ARGS passes through to wide_bench.py; LIBSET
# (default "p127s1 p255s1 p521s1": the carry-chain squaring, faster than operand scanning at every field in
# r20261012-071155-d336; add p127s0 … for both; a suffix r builds WW_ROLL=1, rN also WW_OCC=N, as in p521s1r3, and
# trailing c, u, t and w WW_CC=1, WW_B3U=1, WW_TB=2 and WW_KW=2, as in p521s1rcut) names
# the builds (p256s1 and p512s1 the full-width fields), KMAX_<bits> (defaults 4, 2, 2, 2, 3) the largest k each
# holds, and WW_DEFS extra -D flags for every build. PROFILE=1 runs, instead of the bench, wide_profile.py under
# `ncu --set full` on LIBSET's first library (PROFILE_ARGS: --shape, --batch, --k): out/profile.ncu-rep and
# out/profile-details.txt.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}  # verity's bandwidth tags and PoUS's tree, for the gates
WC=benchmarks/word_codec
LIBSET=${LIBSET:-p127s1 p255s1 p521s1}
declare -A KMAX=([127]=${KMAX_127:-4} [255]=${KMAX_255:-2} [521]=${KMAX_521:-2} [256]=${KMAX_256:-2} [512]=${KMAX_512:-3})
B=$(mktemp -d /tmp/wide-codec.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -I benchmarks/bandwidth)
declare -A PIDS=()
for lib in $LIBSET; do
  bits=${lib:1:3} sqr=${lib:5:1} suf=${lib:6} defs=""
  while [[ $suf == *[cutw] ]]; do
    [[ $suf == *c ]] && defs+=" -DWW_CC=1"
    [[ $suf == *u ]] && defs+=" -DWW_B3U=1"
    [[ $suf == *t ]] && defs+=" -DWW_TB=2"
    [[ $suf == *w ]] && defs+=" -DWW_KW=2"
    suf=${suf%?}
  done
  [[ $suf == r* ]] && defs+=" -DWW_ROLL=1" && [ -n "${suf#r}" ] && defs+=" -DWW_OCC=${suf#r}"
  # shellcheck disable=SC2086
  nvcc "${FLAGS[@]}" -DWW_BITS="$bits" -DWW_SQR="$sqr" -DWW_KMAX="${KMAX[$bits]}" $defs ${WW_DEFS:-} -o "$B/ww-$lib.so" \
    $WC/wide.cu > "$O/ww-$lib.ptxas" 2>&1 & PIDS[$lib]=$!
done
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -DWC_SQR=0 -DWC_KMAX=1 \
  -o "$B/wc-plain.so" $WC/wc.cu > "$O/wc-plain.ptxas" 2>&1 & PIDS[wc]=$!
"$PY" $WC/tests/wide_host.py --words 400 > "$O/wide-host.log" 2>&1 || { tail -30 "$O/wide-host.log"; exit 5; }
for lib in "${!PIDS[@]}"; do
  wait "${PIDS[$lib]}" || { tail -30 "$O/ww-$lib.ptxas" "$O/wc-plain.ptxas"; exit 4; }
done
for lib in $LIBSET; do
  cuobjdump -sass "$B/ww-$lib.so" | awk '/Function : /{p = ($0 ~ /chain_squareILi1E|chain_encodeILi1ELi1E|gemm_kernelILi1ELi1ELb0ELb1E/)} p' \
    > "$O/ww-$lib.sass" 2>&1
done
sha256sum "$B"/*.so > "$O/libs.sha256"
if "$PY" -c 'import pytest' 2> /dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider $WC/tests > "$O/tests.log" 2>&1 || { tail -30 "$O/tests.log"; exit 5; }
fi
export B PY WC LIBSET BENCH_ARGS=${BENCH_ARGS:-} PROFILE=${PROFILE:-} PROFILE_ARGS=${PROFILE_ARGS:-}
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-25}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  q() { nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader \
          -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1; }
  q
  LIBS=(); for lib in $LIBSET; do LIBS+=(--lib "$lib=$B/ww-$lib.so"); done
  if [ -n "${PROFILE:-}" ]; then
    lib=${LIBSET%% *}
    # shellcheck disable=SC2086
    ncu --profile-from-start off --set full --import-source no -f -o "$O/profile" \
      "$PY" $WC/wide_profile.py --lib "$lib=$B/ww-$lib.so" --out "$O" $PROFILE_ARGS > "$O/profile.log" 2>&1; rc=$?
    [ $rc = 0 ] && ncu -i "$O/profile.ncu-rep" --page details > "$O/profile-details.txt" 2>&1
  else
    # shellcheck disable=SC2086
    "$PY" $WC/wide_bench.py "${LIBS[@]}" --wc "$B/wc-plain.so" --out "$O" $BENCH_ARGS; rc=$?
  fi
  q
  exit $rc'
