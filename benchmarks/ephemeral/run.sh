#!/usr/bin/env bash
# The unregistered share of a decode step's HBM reads in stock vLLM (cwd = the source tree): `steps.py` engines with dummy
# FP8 weights at Llama-3.1-8B's or 70B's shapes, under one gpu-lease 1 --wait --max-min MAXMIN (default 28), each arm under
# Nsight Compute (DRAM bytes per kernel or per CUDA graph) or timed; then `attribute.py`.  Results go to $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 32 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-ephemeral \
#     --env ARMS='70b:16:0:node 8b:8:0:node' -- taskset -c 48-95 bash benchmarks/ephemeral/run.sh
#
# ARMS: MODEL:LAYERS:MODE:HOW, MODEL 8b or 70b (the checkpoints' configs on node 2; weights are dummy), LAYERS 0 for the
# checkpoint's depth, MODE 0 (no torch.compile, full decode graphs) or c (vLLM's default compilation and fusion passes),
# HOW node (ncu per kernel), graph (ncu per CUDA graph) or time (no profiler: step times and the HBM read rate).
# BATCHES, CONTEXTS and ARM_MIN (default 12 minutes an arm) pass through.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
[ -x "$PY" ] && [ -f "$WORK/ready.json" ] || { echo "no $WORK/ready.json: run benchmarks/pouw/pearl_c_vllm/setup.sh"; exit 3; }
export PATH=/usr/local/cuda-13.0/bin:$WORK/venv312/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HOME=${HF_HOME:-/workspace/hf}
. benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
export PYTHONPATH=$PWD:$PWD/integrations/vllm CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_ENABLE_V1_MULTIPROCESSING=0
M8=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["path"])' "$WORK/ready.json")
M70=$(ls -d /workspace/hf/hub/models--unsloth--Meta-Llama-3.1-70B/snapshots/*/ | head -1)
"$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ("vllm", "torch")}))' \
  "$WORK/ready.json" > "$O/ready.json"
export NCU_BIN=${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu} PY M8 M70 ARMS=${ARMS:?} ARM_MIN=${ARM_MIN:-12}
export BATCHES=${BATCHES:-1,4,16,64} CONTEXTS=${CONTEXTS:-2048,8192,32768}
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-28}" -- bash -c '
  UUID=${GPU_LEASE_UUID%%,*}; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,memory.used --format=csv,noheader -i "$UUID" \
    > "$O/gpu.txt"
  for arm in $ARMS; do
    IFS=: read -r m layers mode how <<< "$arm"
    tag=${m}-L${layers}-${mode}-${how}
    model=$M8; [ "$m" = 70b ] && model=$M70
    A=(--model "$model" --layers "$layers" --batches "$BATCHES" --contexts "$CONTEXTS")
    [ "$mode" = c ] && A+=(--compiled)
    P=()
    case $how in
      node|graph) A+=(--profile-steps 2 --steps 12)
                  P=("$NCU_BIN" --profile-from-start off --graph-profiling "$how" --cache-control none --clock-control none
                     --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum -f -o "$O/ncu-$tag") ;;
      time) A+=(--steps 32 --read-rate) ;;
    esac
    timeout "$((ARM_MIN * 60))" "${P[@]}" "$PY" benchmarks/ephemeral/steps.py "${A[@]}" --out "$O/$tag.json" \
      > "$O/$tag.log" 2>&1
    echo "$tag $?" >> "$O/arms.rc"
  done'
for r in "$O"/ncu-*.ncu-rep; do
  [ -e "$r" ] || continue
  "$NCU_BIN" --import "$r" --csv --page raw --print-units base > "${r%.ncu-rep}.csv" 2>&1 && rm -f "$r"
done
"$PY" benchmarks/ephemeral/attribute.py "$O" > "$O/attribute.log" 2>&1; rc=$?
cat "$O/arms.rc" "$O/attribute.log"
exit $rc
