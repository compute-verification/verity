#!/usr/bin/env bash
# Pearl-C4's first whole-model FP4 number (compute accounting, 11:03 PM PDT 30 Sep,
# note:20261001T0603Z-order-from-compute-accounting-e8ffd7f2-c066b30c-pearl-c4-llama8b-overnight): the proved NVFP4 GEMM, #548's
# sm_120 PearlC4Nv (cubin dd01ae5e), against the plain NVFP4 divisors (#570's verity_nvf4_*, CUTLASS, cuBLASLt) at each
# distinct linear of Llama-3.1-8B: q/o (n 4,096, k 4,096), k/v (1,024, 4,096), gate/up (14,336, 4,096), down (4,096, 14,336).
# Prefill at m = 8,192; decode at m = 64, Pearl-C4's smallest m, and at m = 32, where the arm pads A to 64 rows (the filler
# rows are committed, computed and hashed, and not credited). Harness and verifier are the divisor window's (runtree/nvfp4 =
# 3535b07fc's trees, build bab84c16 = $IN on node 2), with Llama-3.1-8B's linears added as panel rows (harness/shapes.py) and
# every decode panel row timed as a dependent chain (bench.py's `chained`).
#
#   research run --queue --on vy-nebius-2 --project pous --cpus 16 --mem-gb 128 --custody-r2 --custody-ttl 8h --question Q \
#     --source TREE --cwd source --send pearl_c4.cubin --declared-output ... -- bash runtree/llama8b_window.sh card|timed
#
# pearl_c4.cubin: dd01ae5e (r20260930-162438-25e9's inputs/pc4-ship.tar). card: gpu-lease 1, untimed, verify --tier 2a.
# timed: waits for WINDOW_AT (default 10:00Z, bc-c066b30c's slot), then gpu-lease 8 --timed, the harness on the lease's first
# GPU, verify --tier 2b. Both take 20 reps, bench.py's minimum. The verifies run on the CPU after the lease.
set -uo pipefail
mode=${1:?card|timed|inner}
IN=/workspace/pouw/fill-out/harness/inputs-bab84c16
ROOT=$PWD N=$PWD/runtree/nvfp4 O=$RESEARCH_RUN_DIR/llama8b
TN=3535b07fcbe656d1200fd80712bc5d47d0ed764c
reps=20
G="--gpu lease --server-md $IN/node2-gpus.md --clock-label locked-2100 --finalists 3 --reps $reps"
ARM="--families nvfp4 --cutlass-sched nvfp4 --arm-path benchmarks/pouw/pearl_c4 --arm pearl_c4_arm:PearlC4Nv"
LINEARS="n4096-k4096 n1024-k4096 n14336-k4096 n4096-k14336"
ITEMS="prefill:8192 decode64:64 decode32:32"

if [[ $mode == inner ]]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  echo "lease=${GPU_LEASE_UUID:-?} cvd=$CUDA_VISIBLE_DEVICES reps=$reps start=$(date -u +%FT%TZ)"
  rc=0
  for it in $ITEMS; do
    name=${it%%:*} m=${it##*:} t0=$SECONDS
    shapes=$(for l in $LINEARS; do printf 'm%s-%s ' "$m" "$l"; done)
    mkdir -p "$O/$name"
    (cd "$N" && . benchmarks/pouw/harness/jobs/env.sh "$IN" &&
      "$PY" benchmarks/pouw/harness/bench.py $G --shapes $shapes $ARM --out "$O/$name/bench.json") > "$O/$name/bench.log" 2>&1
    r=$?
    echo "item $name rc=$r wall_s=$((SECONDS - t0))"
    (( r == 0 )) || rc=$r
  done
  exit $rc
fi

[[ $mode == card || $mode == timed ]] || { echo "usage: $0 card|timed" >&2; exit 2; }
RD=$RESEARCH_RUN_DIR
mkdir -p "$O"
export PYTHONDONTWRITEBYTECODE=1 POUW_SASS_GATE_CACHE=$IN/sass-gate-cache PEARLC4_CUBIN=$RD/inputs/pearl_c4.cubin
sha256sum "$PEARLC4_CUBIN"
echo "dd01ae5e5ac74ad65f62d63261ce5293b0fbf43be3d9c10e59cfd260b95f95ff  $PEARLC4_CUBIN" | sha256sum -c --quiet || exit 1

if [[ $mode == timed ]]; then
  at=$(date -u -d "${WINDOW_AT:-2026-10-01T10:00:00Z}" +%s) now=$(date -u +%s)
  (( at > now )) && { echo "waiting $((at - now)) s for ${WINDOW_AT:-2026-10-01T10:00:00Z}"; sleep $((at - now)); }
  gpu-lease 8 --wait --timed --max-min 30 -- timeout 1740 bash "$ROOT/runtree/llama8b_window.sh" inner
  tier=2b
else
  gpu-lease 1 --wait --max-min 30 -- timeout 1740 bash "$ROOT/runtree/llama8b_window.sh" inner
  tier=2a
fi
b=$?
echo "bench rc=$b"
v=0
for it in $ITEMS; do
  name=${it%%:*}
  [[ -f $O/$name/bench.json ]] || { echo "verify $name: no bench.json"; v=1; continue; }
  (cd "$N" && . benchmarks/pouw/harness/jobs/env.sh "$IN" &&
    "$PY" benchmarks/pouw/harness/verify.py "$O/$name/bench.json" --tier $tier --commit "$TN") > "$O/$name/verify.log" 2>&1
  r=$?
  echo "verify $name rc=$r"
  (( r == 0 )) || v=$r
done
exit $(( b ? b : v ))
