#!/usr/bin/env bash
# The receipted filler beside vLLM decode (cwd = the source tree): builds libifill.so (serve.cu) and fill_compute's
# libfill.so (its green-context split), then under one gpu-lease 1 --wait --max-min MAXMIN (default 30; GPU pins it) one
# serve.py process per arm in ARMS (default "warmup stock share:16 share:8 share:4 gc:128 gc:128:16 stock"; an arm that
# can't start with 150 s left in the lease is skipped, in arms.txt), NVML sampled by nvidia-smi every 200 ms beside them,
# and fill_compute's compare_fill.py over the arms. Results in $OUT.
#
#   research run --on vy-nebius-2 --project verity --campaign cmp-12 --source . --cwd source --mem-gb 64 \
#     --env GPU_LEASE_WHO=W11 -- bash benchmarks/filler_slack/serve.sh
#
# WORK is the e2e's vLLM environment on node 2 (benchmarks/pouw/pearl_c_vllm/setup.sh); MODEL its Llama-3.1-8B-Instruct.
# SERVE_ARGS pass through to every arm; ARM_MIN bounds one arm (default 8 minutes).
set -uxo pipefail
export O=${OUT:?}
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
[ -x "$PY" ] && [ -f "$WORK/ready.json" ] || { echo "no $WORK/ready.json: run benchmarks/pouw/pearl_c_vllm/setup.sh"; exit 3; }
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$WORK/venv312/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0f
export HF_HOME=${HF_HOME:-/workspace/hf}
. benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
export PYTHONPATH=$PWD:$PWD/integrations/vllm CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_ENABLE_V1_MULTIPROCESSING=0
D=benchmarks/filler_slack FC=benchmarks/fill_compute
B=$(mktemp -d /tmp/filler-serve.XXXXXX); trap 'rm -rf "$B"' EXIT
{ nvcc --version; git rev-parse HEAD 2>/dev/null; } > "$O/build.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -Xcompiler -fPIC -shared -o "$B/libifill.so" \
  $D/serve.cu > "$O/ptxas-ifill.txt" 2>&1 || { tail -30 "$O/ptxas-ifill.txt"; exit 4; }
nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xcompiler -fPIC -shared -o "$B/libfill.so" \
  $FC/fill.cu -ldl > "$O/ptxas-fill.txt" 2>&1 || { tail -30 "$O/ptxas-fill.txt"; exit 4; }
cat "$MODEL"/*.safetensors > /dev/null
export B PY MODEL D ARMS=${ARMS:-warmup stock share:16 share:8 share:4 gc:128 gc:128:16 stock} SERVE_ARGS=${SERVE_ARGS:-}
export ARM_MIN=${ARM_MIN:-8} MAXMIN=${MAXMIN:-30}
# shellcheck disable=SC2016
gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "$MAXMIN" -- bash -c '
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv,noheader \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader \
    -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  i=0; files=(); end=$(( $(date +%s) + MAXMIN * 60 - 45 ))
  for arm in $ARMS; do
    left=$(( end - $(date +%s) ))
    if [ "$left" -lt 150 ]; then echo "$arm skipped: ${left}s left in the lease" >> "$O/arms.txt"; continue; fi
    f="$O/arm-$i-${arm//:/-}.json"; files+=("$f")
    echo "$arm start $(date -u +%FT%T.%3NZ)" >> "$O/arms.txt"
    # shellcheck disable=SC2086
    timeout $(( left < ARM_MIN * 60 ? left : ARM_MIN * 60 )) "$PY" $D/serve.py --lib "$B/libifill.so" --fill-lib "$B/libfill.so" --model "$MODEL" \
      --arm "$arm" --out "$f" $SERVE_ARGS > "$O/arm-$i-${arm//:/-}.log" 2>&1
    echo "$arm $?" >> "$O/arms.rc"; echo "$arm end $(date -u +%FT%T.%3NZ)" >> "$O/arms.txt"; i=$((i + 1))
  done
  kill $smi
  python3 benchmarks/fill_compute/compare_fill.py "$O" "${files[@]}" > "$O/compare.txt" 2>&1; echo "compare $?" >> "$O/arms.rc"'
cat "$O/arms.rc"; cat "$O/compare.txt"
python3 $D/serve_summary.py "$O" > "$O/summary.json" 2>> "$O/errors.txt" || echo "summary rc $?" >> "$O/errors.txt"
