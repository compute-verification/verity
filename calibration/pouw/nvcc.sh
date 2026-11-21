# Sourced by the probes' runners: NVCC, else the machine's toolkit 13.0's nvcc (`machine.py toolkit 13.0`); exits 2,
# naming the runner, where there is none or it is not executable.
NVCC=${NVCC:-$(python3 "$(dirname "${BASH_SOURCE[0]}")/../../benchmarks/kernels/machine.py" toolkit 13.0)/bin/nvcc} || { echo "$(basename "$0"): no CUDA 13.0 (machine.py, above)" >&2; exit 2; }
[ -x "$NVCC" ] || { echo "$(basename "$0"): no nvcc at $NVCC" >&2; exit 2; }
