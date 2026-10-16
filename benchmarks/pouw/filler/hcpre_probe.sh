#!/usr/bin/env bash
# HC-PREFILL's prefill timeline probe (cmp-13, approach pouw/filler-prefill): where the M = 8,192 prefill linears (dcap's
# 32 layers of cuBLASLt int8 qkv, o, gate_up, down, as the scorer times them) leave the tensor pipe idle. Under one lease:
# nsys's per-kernel trace of the step graph (durations and gaps), then ncu on one layer's four GEMMs (launch shape,
# occupancy, tensor-pipe activity, per-SM active cycles for the wave tail). Run from the repository root
# (research run --cwd source); outputs under $OUT.
set -uo pipefail
O=${OUT:?}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
export PATH="$(dirname "$NVCC"):$PATH"
B=$(mktemp -d /tmp/hcpre-probe.XXXXXX)
trap 'rm -rf "$B"' EXIT
FL=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v)
LIBS=(-lcublasLt -lnvidia-ml -ldl)
"$NVCC" "${FL[@]}" -o "$B/dcap" benchmarks/pouw/decode_capacity/dcap.cu "${LIBS[@]}" >> "$O/build.log" 2>&1 || { echo "build failed"; exit 4; }
git rev-parse HEAD > "$O/commit.txt"
{ command -v nsys; command -v ncu; ls /opt/nvidia 2>/dev/null; ls -d /usr/local/cuda*/ 2>/dev/null; } > "$O/tools.txt" 2>&1
NSYS=$(command -v nsys || ls /usr/local/cuda-13.0/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
cat > "$B/inner.sh" <<EOF
set -uo pipefail
q="clocks.sm,clocks.max.sm,power.draw,power.limit,clocks.applications.graphics,clocks.default_applications.graphics,clocks_event_reasons.active"
smi() { nvidia-smi --query-gpu=\$q --format=csv,noheader,nounits \${GPU_LEASE_UUID:+-i \$GPU_LEASE_UUID} >> "$O/smi.csv"; }
smi
"$B/dcap" lin 8192 9 > "$O/lin.jsonl" 2>> "$O/errors.txt"; echo "lin rc \$?" >> "$O/rc.txt"
smi
if [ -n "$NSYS" ]; then
  "$NSYS" profile --trace=cuda,nvtx --cuda-graph-trace=node -o "$B/pre" --force-overwrite true "$B/dcap" lin 8192 3 \
    > "$O/nsys.log" 2>&1; echo "nsys rc \$?" >> "$O/rc.txt"
  "$NSYS" stats --report cuda_gpu_trace --format csv --output "$B/trace" "$B/pre.nsys-rep" >> "$O/nsys.log" 2>&1
  cp "$B"/trace*.csv "$O/" 2>/dev/null
fi
smi
DCAP_NOGRAPH=1 ncu --clock-control none --nvtx --nvtx-include timed/ --launch-skip 0 --launch-count 4 \
  --section LaunchStats --section Occupancy --section SpeedOfLight --section ComputeWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis \
  --metrics sm__cycles_active.avg,sm__cycles_active.max,sm__cycles_active.min,sm__cycles_elapsed.max,sm__inst_executed_pipe_tensor_op_imma.sum,sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum \
  --csv "$B/dcap" lin 8192 1 > "$O/ncu-gemm.csv" 2>> "$O/ncu-errors.txt"; echo "ncu rc \$?" >> "$O/rc.txt"
smi
EOF
gpu-lease 1 --wait --max-min "${MAXMIN:-10}" -- bash "$B/inner.sh"
echo "lease rc $?" >> "$O/rc.txt"
