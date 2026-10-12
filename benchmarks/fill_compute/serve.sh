#!/usr/bin/env bash
# The filler under real serving (cwd = the source tree): builds fill.cu as run.sh does (the SASS and resource gates), then
# under one gpu-lease 1 --wait --max-min MAXMIN (default 40) one serve_fill.py process per arm in ARMS (default
# "stock gc:176 gc:176:fill gc:128 gc:128:fill stock"), and compare_fill.py's gates and points. Results go to
# $RESEARCH_RUN_DIR/out.
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 64 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-fill-compute \
#     -- taskset -c 48-95 bash benchmarks/fill_compute/serve.sh
#
# WORK is the e2e's vLLM environment on node 2 (benchmarks/pouw/pearl_c_vllm/setup.sh); MODEL its Llama-3.1-8B-Instruct.
# SERVE_ARGS pass through to every arm; ARM_MIN bounds one arm (default 8 minutes).
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
[ -x "$PY" ] && [ -f "$WORK/ready.json" ] || { echo "no $WORK/ready.json: run benchmarks/pouw/pearl_c_vllm/setup.sh"; exit 3; }
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$WORK/venv312/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0f
export HF_HOME=${HF_HOME:-/workspace/hf}
. benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
export PYTHONPATH=$PWD:$PWD/integrations/vllm CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_ENABLE_V1_MULTIPROCESSING=0
FC=benchmarks/fill_compute
B=$(mktemp -d /tmp/fill-serve.XXXXXX); trap 'rm -rf "$B"' EXIT
FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -Xcompiler -fPIC -shared)
{ nvcc --version; echo "flags: ${FLAGS[*]}"; git rev-parse HEAD 2>/dev/null; } > "$O/build.txt"
nvcc "${FLAGS[@]}" -o "$B/libfill.so" $FC/fill.cu -ldl > "$O/ptxas.txt" 2>&1 || { tail -30 "$O/ptxas.txt"; exit 4; }
cuobjdump -sass "$B/libfill.so" > "$B/fill.sass" || exit 4
cuobjdump -res-usage "$B/libfill.so" > "$O/res-usage.txt" || exit 4
python3 $FC/sass_gate.py "$O/ptxas.txt" "$B/fill.sass" "$O/res-usage.txt" "$O/sass.json" || exit 4
cat "$MODEL"/*.safetensors > /dev/null
"$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(json.dumps({k: d.get(k) for k in ("model", "vllm", "torch")}))' \
  "$WORK/ready.json" > "$O/ready.json"
export B PY MODEL FC ARMS=${ARMS:-stock gc:176 gc:176:fill gc:128 gc:128:fill stock} SERVE_ARGS=${SERVE_ARGS:-} ARM_MIN=${ARM_MIN:-8}
# shellcheck disable=SC2016
gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-40}" -- bash -c '
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  i=0; files=()
  for arm in $ARMS; do
    f="$O/arm-$i-${arm//:/-}.json"; files+=("$f")
    # shellcheck disable=SC2086
    timeout $((ARM_MIN * 60)) "$PY" $FC/serve_fill.py --lib "$B/libfill.so" --model "$MODEL" --arm "$arm" --out "$f" $SERVE_ARGS \
      > "$O/arm-$i-${arm//:/-}.log" 2>&1
    echo "$arm $?" >> "$O/arms.rc"; i=$((i + 1))
  done
  python3 $FC/compare_fill.py "$O" "${files[@]}" > "$O/compare.txt" 2>&1; echo "compare $?" >> "$O/arms.rc"'
cat "$O/arms.rc"; cat "$O/compare.txt"
