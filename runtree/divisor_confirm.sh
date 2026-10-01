#!/usr/bin/env bash
# The confirming timed whole-node row for the plain-GEMM divisors (bc-2aa33ad8, 6:12 and 6:36 PM PDT 30 Sep): NVFP4 8,192^3
# takes #543's verity_nvf4_256x128_o_ew and FP8 8,192^3 #570's verity_fp8_256x128_ew only once this row passes. The harness
# is #588 (dd23c0c36) + #491 (aa4f95db9) in both trees, build bab84c16 ($IN on node 2). Each PoUW arm runs in its owner's
# tree, because the two branches' verity_pouw can't share a process (GPU 1's pearl_c_work reads scheme.device, which
# PearlC4 lacks): FP8 PearlCSm120 (GPU 1, a00db59ee) from this tree's root, NVFP4 PearlC4Nv (#580, 639128c87) from
# runtree/nvfp4, whose packages/verity/src, protocols/pouw and benchmarks/pouw are 3535b07fc's git trees.
#
#   research run ... --send ship.tar --send pearl_c4.cubin -- bash runtree/divisor_confirm.sh card|timed
#
# ship.tar: GPU 1's ship, cubin 40d5531b (r20260930-231211-2f19's inputs/ship.tar). pearl_c4.cubin: dd01ae5e (in
# r20260930-162438-25e9's inputs/pc4-ship.tar). card: gpu-lease 1, untimed, 5 reps. timed: gpu-lease 8 --timed, the
# harness on the lease's first GPU, 20 reps as the 3:41 PM PDT divisor window. Then verify.py --tier 2b on each output.
set -uo pipefail
mode=${1:?card|timed|inner}
IN=/workspace/pouw/fill-out/harness/inputs-bab84c16
ROOT=$PWD N=$PWD/runtree/nvfp4 O=$RESEARCH_RUN_DIR/confirm
NAMES=$IN/names/m32-n8192-k8192.txt
TN=3535b07fcbe656d1200fd80712bc5d47d0ed764c
reps=20; [[ $mode == card || ${CONFIRM_MODE:-} == card ]] && reps=5
G="--gpu lease --server-md $IN/node2-gpus.md --clock-label locked-2100 --finalists 3 --reps $reps"
ARM_FP8="--arm-path benchmarks/pouw/pearl_c_sm120 --arm pearlc_arm:PearlCSm120"

if [[ $mode == inner ]]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  echo "lease=${GPU_LEASE_UUID:-?} cvd=$CUDA_VISIBLE_DEVICES reps=$reps start=$(date -u +%FT%TZ)"
  rc=0
  item() {
    local name=$1 tree=$2 t0=$SECONDS r
    shift 2
    mkdir -p "$O/$name"
    (cd "$tree" && . benchmarks/pouw/harness/jobs/env.sh "$IN" &&
      "$PY" benchmarks/pouw/harness/bench.py $G "$@" --out "$O/$name/bench.json") > "$O/$name/bench.log" 2>&1
    r=$?
    echo "item $name rc=$r wall_s=$((SECONDS - t0))"
    (( r == 0 )) || rc=$r
  }
  item nvfp4 "$N" --shapes m8192-n8192-k8192 m32-n8192-k8192 --families nvfp4 --cutlass-sched nvfp4 \
    --arm-path benchmarks/pouw/pearl_c4 --arm pearl_c4_arm:PearlC4Nv
  item fp8-prefill "$ROOT" --shapes m8192-n8192-k8192 --families fp8-e4m3 --cutlass-sched fp8-e4m3 $ARM_FP8
  item fp8-decode "$ROOT" --shapes m32-n8192-k8192 --families fp8-e4m3 --cutlass-sched fp8-e4m3 \
    --lt-enumerate fp8-e4m3 --lt-enumerate-names "$NAMES" --lt-enumerate-max 140000 $ARM_FP8
  exit $rc
fi

[[ $mode == card || $mode == timed ]] || { echo "usage: $0 card|timed" >&2; exit 2; }
RD=$RESEARCH_RUN_DIR
mkdir -p "$RD/ship" "$O" && tar -xf "$RD/inputs/ship.tar" -C "$RD/ship" || exit 1
export PYTHONDONTWRITEBYTECODE=1 CONFIRM_MODE=$mode POUW_SASS_GATE_CACHE=$IN/sass-gate-cache PEARLC_SHIP=$RD/ship/ship \
  PEARLC_ROWS=/workspace/pouw/fill-out/harness/rows/$RESEARCH_RUN_ID PEARLC4_CUBIN=$RD/inputs/pearl_c4.cubin
mkdir -p "$PEARLC_ROWS"
sha256sum "$PEARLC_SHIP/pearl_c_sm120.cubin" "$PEARLC4_CUBIN" "$NAMES"
echo "40d5531b37be7360982f689c716ef34bb3986e7a836f1e5da4eb1515f2e2aa96  $PEARLC_SHIP/pearl_c_sm120.cubin
dd01ae5e5ac74ad65f62d63261ce5293b0fbf43be3d9c10e59cfd260b95f95ff  $PEARLC4_CUBIN" | sha256sum -c --quiet || exit 1
(cd "$IN/names" && grep " m32-n8192-k8192.txt\$" SHA256SUMS | sha256sum -c --quiet) || exit 1

if [[ $mode == timed ]]; then
  gpu-lease 8 --wait --timed --max-min 20 -- timeout 1140 bash "$ROOT/runtree/divisor_confirm.sh" inner
else
  gpu-lease 1 --wait --max-min 20 -- timeout 1140 bash "$ROOT/runtree/divisor_confirm.sh" inner
fi
b=$?
echo "bench rc=$b"
v=0
for spec in "nvfp4 $N $TN" "fp8-prefill $ROOT" "fp8-decode $ROOT"; do
  read -r name tree commit <<< "$spec"
  [[ -f $O/$name/bench.json ]] || { echo "verify $name: no bench.json"; v=1; continue; }
  (cd "$tree" && . benchmarks/pouw/harness/jobs/env.sh "$IN" &&
    "$PY" benchmarks/pouw/harness/verify.py "$O/$name/bench.json" --tier 2b ${commit:+--commit "$commit"}) \
    > "$O/$name/verify.log" 2>&1
  r=$?
  echo "verify $name rc=$r"
  (( r == 0 )) || v=$r
done
exit $(( b ? b : v ))
