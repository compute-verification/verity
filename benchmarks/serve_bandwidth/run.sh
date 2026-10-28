#!/usr/bin/env bash
# A served mix under nsys's GPU-metrics DRAM sampling (cwd = the source tree): mix.py once for each max-batched-tokens
# of MBT (default "8192 2048"), each its own nsys session, then timeline.py over its sqlite export. Results go to
# $RESEARCH_RUN_DIR/out/mbt<N>/; a report under KEEP_MB (default 400) MB is kept there, a larger one is listed by size
# and sha256. MODEL (default the 8B), TP (1), SECS (the trace, 30 s), HZ (GPU-metrics samples a second, 10000), MIX_ARGS.
# SCRIPT=gemm_dtypes.py runs that instead, with ARGS; REDO=<a run's out/ on this node> exports and analyzes its kept
# reports again.
#
#   research run --queue --gpus 1 --project verity --source . --cwd source --mem-gb 64 --max-min 30 \
#     -- bash benchmarks/serve_bandwidth/run.sh
set -uxo pipefail
O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
export PYTHONPATH=$PWD:$PWD/integrations/vllm FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HOME=${HF_HOME:-/workspace/hf} \
  CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_ENABLE_V1_MULTIPROCESSING=0 NSYS_NVTX_PROFILER_REGISTER_ONLY=0
. benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
SB=benchmarks/serve_bandwidth
if [ "${SCRIPT:-}" = gemm_dtypes.py ]; then
  # shellcheck disable=SC2086
  exec "$PY" $SB/gemm_dtypes.py --out "$O" ${ARGS:-}
fi
if [ -n "${REDO:-}" ]; then
  # export and analyze again the reports a run kept on this node (REDO = its out/), CPU only
  B=$(mktemp -d /tmp/serve-bw.XXXXXX); trap 'rm -rf "$B"' EXIT
  rc=0
  for rep in "$REDO"/mbt*/mix*.nsys-rep; do
    src=$(dirname "$rep"); d=$O/$(basename "$src"); mkdir -p "$d"
    nsys export --type sqlite --force-overwrite true -o "$B/x.sqlite" "$rep" > "$d/export.log" 2>&1 \
      || { rc=7; tail -20 "$d/export.log"; continue; }
    "$PY" $SB/timeline.py "$B/x.sqlite" --mix "$src/mix.json" --out "$d" | tee "$d/timeline.txt" || rc=8
    rm -f "$B/x.sqlite"
  done
  exit $rc
fi
MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
cat "$MODEL"/*.safetensors > /dev/null
B=$(mktemp -d /tmp/serve-bw.XXXXXX); trap 'rm -rf "$B"' EXIT
{ nsys --version; nvidia-smi --query-gpu=index,pci.bus_id,clocks.sm,clocks.mem,power.draw --format=csv;
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; } > "$O/env.txt" 2>&1
DEVS=cuda-visible
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && DEVS=all
rc=0
for mbt in ${MBT:-8192 2048}; do
  d=$O/mbt$mbt; mkdir -p "$d"
  # shellcheck disable=SC2086
  nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --gpu-metrics-devices="$DEVS" \
    --gpu-metrics-frequency="${HZ:-10000}" --capture-range=nvtx --nvtx-capture=trace --capture-range-end=stop \
    --force-overwrite true -o "$B/mix$mbt" \
    "$PY" $SB/mix.py --out "$d" --model "$MODEL" --tp "${TP:-1}" --seconds "${SECS:-30}" \
    --max-batched-tokens "$mbt" ${MIX_ARGS:-} > "$d/mix.log" 2>&1 || { rc=$?; tail -40 "$d/mix.log"; continue; }
  nsys export --type sqlite --force-overwrite true -o "$B/mix$mbt.sqlite" "$B/mix$mbt.nsys-rep" > "$d/export.log" 2>&1 \
    || { rc=7; tail -20 "$d/export.log"; continue; }
  "$PY" $SB/timeline.py "$B/mix$mbt.sqlite" --mix "$d/mix.json" --out "$d" | tee "$d/timeline.txt" || rc=8
  sz=$(stat -c %s "$B/mix$mbt.nsys-rep")
  sha256sum "$B/mix$mbt.nsys-rep" | tee -a "$d/report.txt"; echo "bytes $sz" >> "$d/report.txt"
  [ "$sz" -le $(( ${KEEP_MB:-400} << 20 )) ] && cp "$B/mix$mbt.nsys-rep" "$d/"
  rm -f "$B/mix$mbt.sqlite"
done
exit $rc
