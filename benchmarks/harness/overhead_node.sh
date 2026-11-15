#!/usr/bin/env bash
# `verity bench overhead`'s job on a GPU node, run from the shipped source by the `research run --on NODE --cwd source` that
# `verity_bench.overhead_records` launches, under `gpu-lease 1 --wait --max-min M`: it builds the PoUW harness for the card (native/build.sh at the
# card's PH_ARCH, its cache under POUW_HARNESS_BUILD on the node, so a timed run compiles only a changed harness) and runs
# `python -m verity_bench.overhead_node` on the lease's GPU, with a scratch directory outside $OUT for the Nsight report.
#
#   bash benchmarks/harness/overhead_node.sh points --card h200 [--model M] [--precision P]
#   bash benchmarks/harness/overhead_node.sh run --card h200 --scheme S --points points.json --points-art ART [--point P] \
#     [--cubin FILE --cubin-art ART]
#
# VERITY_BENCH_VENV names the Python with torch and vLLM (default /workspace/cp/venv312), CUDA the toolkit with nvcc and
# nsys (default /usr/local/cuda-13.0).
set -euo pipefail
: "${RESEARCH_RUN_DIR:?}" "${RESEARCH_RUN_ID:?}" "${GPU_LEASE_UUID:?run it on a GPU lease: gpu-lease 1 --wait -- bash $0 ...}"
CUDA=${CUDA:-/usr/local/cuda-13.0}
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
O=${OUT:-$RESEARCH_RUN_DIR/out}
mkdir -p "$O"
export PATH=$CUDA/bin:$PATH CUDA_HOME=$CUDA CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=$PWD:$PWD/tools/config:$PWD/adapters/vllm:$PWD/benchmarks/pouw:$PWD/benchmarks/harness${PYTHONPATH:+:$PYTHONPATH}
card=
for ((i = 1; i < $#; i++)); do
  if [[ ${!i} == --card ]]; then j=$((i + 1)); card=${!j}; fi
done
arch=$("$PY" -c 'import sys; from verity_bench.overhead import card; print(card(sys.argv[1]).arch)' "$card")
# sass_pins.json pins sm_120a's IEEE sequences under CUDA 12.9.1 (jobs/toolkit.sh assembles it) and sm_90a's under 13.0.1
if [[ $arch == sm_120a ]]; then
  . benchmarks/pouw/harness/jobs/toolkit.sh
  export LD_LIBRARY_PATH=$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
POUW_HARNESS_LIB=$(PH_ARCH=$arch POUW_HARNESS_BUILD=${POUW_HARNESS_BUILD:-/workspace/research/cache/pouw-harness-build} \
  bash benchmarks/pouw/harness/native/build.sh 2> "$O/harness-build.log")
export POUW_HARNESS_LIB
cp "$(dirname "$POUW_HARNESS_LIB")/build.json" "$O/harness-build.json"
sub=$1
shift
extra=()
if [[ $sub == run ]]; then
  S=$(mktemp -d)
  trap 'rm -rf "$S"' EXIT
  extra=(--scratch "$S" --nsys "$CUDA/bin/nsys" ${RESEARCH_SOURCE_SHA:+--commit "$RESEARCH_SOURCE_SHA"})
fi
"$PY" -m verity_bench.overhead_node "$sub" --uuid "$GPU_LEASE_UUID" --out "$O" --run-dir "$RESEARCH_RUN_DIR" \
  --run-id "$RESEARCH_RUN_ID" "${extra[@]}" "$@"
