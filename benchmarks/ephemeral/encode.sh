#!/usr/bin/env bash
# ra-word-codec's kv.cu on one GPU (cwd = the source tree): `encode_rate.py` (JOB=encode, the default) or `attn_bytes.py`
# under Nsight Compute (JOB=attn). kv.cu, kv.py and kv_bench.py arrive as the run's inputs (research run --send), kv.cu is
# built at KV_FIELD = 512 outside the lease, then one gpu-lease 1 --wait.
#
#   for f in kv.cu kv.py kv_bench.py; do git show origin/cursor/ra-word-codec-10bf:benchmarks/word_codec/$f > /tmp/kv/$f; done
#   research run --on vy-nebius-2 --project verity --source . --cwd source --send /tmp/kv/kv.cu --send /tmp/kv/kv.py \
#     --send /tmp/kv/kv_bench.py --declared-output 'out/*' --env GPU_LEASE_WHO=ra-ephemeral -- bash benchmarks/ephemeral/encode.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
IN=$RESEARCH_RUN_DIR/inputs
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
export PATH=/usr/local/cuda-13.0/bin:$PATH JOB=${JOB:-encode} NCU_BIN=${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu}
B=$(mktemp -d /tmp/ephemeral-encode.XXXXXX); trap 'rm -rf "$B"' EXIT
cp "$IN"/kv.cu "$IN"/kv.py "$IN"/kv_bench.py "$B"/
sha256sum "$B"/kv.cu "$B"/kv.py "$B"/kv_bench.py > "$O/inputs.sha256"
export PY B
nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -DKV_FIELD=512 \
  -o "$B/kv-512.so" "$B/kv.cu" > "$O/kv-512.ptxas" 2>&1 || { tail -30 "$O/kv-512.ptxas"; exit 4; }
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${MAXMIN:-15}" -- bash -c '
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,clocks.mem --format=csv,noheader -i "${GPU_LEASE_UUID%%,*}" \
    > "$O/gpu.txt"
  if [ "$JOB" = attn ]; then
    "$NCU_BIN" --profile-from-start off --cache-control none --clock-control none \
      --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum -f -o "$O/ncu-attn" \
      "$PY" benchmarks/ephemeral/attn_bytes.py run --src "$B" --lib "$B/kv-512.so" --out "$O/attn.json" > "$O/attn.log" 2>&1
  else
    "$PY" benchmarks/ephemeral/encode_rate.py --src "$B" --lib "$B/kv-512.so" --out "$O/encode.json" > "$O/encode.log" 2>&1
  fi
  echo "$JOB $?" > "$O/rc"'
if [ "$JOB" = attn ] && grep -q "attn 0" "$O/rc"; then
  "$NCU_BIN" --import "$O/ncu-attn.ncu-rep" --csv --page raw --print-units base > "$O/ncu-attn.csv" 2>&1 &&
    rm -f "$O/ncu-attn.ncu-rep"
  "$PY" benchmarks/ephemeral/attn_bytes.py attribute --csv "$O/ncu-attn.csv" --out "$O/attn.json" >> "$O/attn.log" 2>&1 ||
    echo "attribute $?" >> "$O/rc"
fi
cat "$O/rc" "$O/$JOB.log"
grep -qx "$JOB 0" "$O/rc" && ! grep -q attribute "$O/rc"
