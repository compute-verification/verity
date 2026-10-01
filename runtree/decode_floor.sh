#!/usr/bin/env bash
# Pearl-C4's decode floor on Llama-3.1-8B (plan line in lanes/pouw-fp4, 10:18Z): dev.py's per-kernel timer on #548's cubin
# (dd01ae5e) at the four linears, m 64 (decode) through m 512, to split a decode call's 0.35-0.85 ms between A's row
# commitment (commit_rows), forming, the chain, the GEMM and the leaves. Untimed: one GPU, clocks as the machine sets them, so
# its milliseconds are a breakdown, never a panel number.
#
#   research run --queue --on vy-nebius-2 --project verity --campaign pouw --gpus 1 --gpu-ids 0 --preemptible \
#     --cpus 4 --mem-gb 32 --max-min 15 --custody-r2 --custody-ttl 8h --question Q \
#     --source TREE --cwd source --send pearl_c4.cubin --declared-output 'decode-floor/*' -- bash runtree/decode_floor.sh
#
# The queue's lease holds the GPU around this workload; --gpu-ids keeps it on NUMA 0 (one of GPUs 0-3), so every host thread runs on
# PIN (default 48-91, off slot d's cores 0-47: note:20261001T0918Z-handoff-from-compute-accounting-slot-d-pinning).
set -uo pipefail
IN=/workspace/pouw/fill-out/harness/inputs-bab84c16
N=$PWD/runtree/nvfp4 O=$RESEARCH_RUN_DIR/decode-floor
CUBIN=$RESEARCH_RUN_DIR/inputs/pearl_c4.cubin
PIN=${PIN:-48-91}
mkdir -p "$O"
taskset -cp "$PIN" $$ > /dev/null || { echo "cannot pin host threads to cores $PIN" >&2; exit 1; }
echo "pinned: $(taskset -cp $$) lease=${GPU_LEASE_UUID:-?} cvd=${CUDA_VISIBLE_DEVICES:-}"
case ${CUDA_VISIBLE_DEVICES:-} in
  0|1|2|3) ;;
  *) echo "expected one GPU of 0-3 from the queue's lease, got CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'" >&2; exit 1 ;;
esac
echo "dd01ae5e5ac74ad65f62d63261ce5293b0fbf43be3d9c10e59cfd260b95f95ff  $CUBIN" | sha256sum -c --quiet || exit 1
SHAPES=""
for m in 64 128 256 512; do
  for nk in 4096x4096 1024x4096 14336x4096 4096x14336; do SHAPES+="${m}x${nk} "; done
done
cd "$N" && . benchmarks/pouw/harness/jobs/env.sh "$IN" || exit 1
timeout 780 "$PY" benchmarks/pouw/pearl_c4/dev.py time $SHAPES --reps 50 --cubin "$CUBIN" > "$O/steps.jsonl" 2> "$O/steps.err"
r=$?
echo "dev.py time rc=$r rows=$(wc -l < "$O/steps.jsonl")"
exit $r
