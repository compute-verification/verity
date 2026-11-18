#!/usr/bin/env bash
# `verity bench overhead`'s job on a GPU node, run from the shipped source by the `research run --on NODE --cwd source` that
# `verity_bench.overhead_records` launches, under `gpu-lease 1 --wait --max-min M`: it builds the PoUW harness for the card (native/build.sh at the
# card's PH_ARCH, its cache under POUW_HARNESS_BUILD on the node, so a timed run compiles only a changed harness) and runs
# `python -m verity_bench.overhead_node` on the lease's GPU, with a private scratch directory outside $OUT (mktemp -d, 0700,
# removed when the job exits) for the Nsight report and the arm's passes, whose salts and rows never go under $OUT.
#
#   bash benchmarks/harness/overhead_node.sh points --card h200 [--model M] [--precision P]
#   bash benchmarks/harness/overhead_node.sh run --card h200 --scheme S --points points.json --points-art ART --point C|L \
#     [--cubin FILE --cubin-art ART]
#
# Run one operating point a launch (`--point C`, and `--point L` as its own launch); `--point all` times both in one
# bench.py process under nsys, in one lease.
#
# VERITY_BENCH_VENV names the Python with torch and vLLM, which the door's launch forwards; nsys is CUDA 13.0's, CUDA_HOME's
# or the run's machine's (benchmarks/kernels/machine.py toolkit 13.0), and the harness's toolkit is identity.TOOLKIT's.
set -euo pipefail
: "${RESEARCH_RUN_DIR:?}" "${RESEARCH_RUN_ID:?}" "${GPU_LEASE_UUID:?run it on a GPU lease: gpu-lease 1 --wait -- bash $0 ...}"
PY=${VERITY_BENCH_VENV:?set VERITY_BENCH_VENV to the Python with torch and vLLM on the node, which the door forwards}/bin/python
CUDA=$("$PY" benchmarks/kernels/machine.py toolkit 13.0)
O=${OUT:-$RESEARCH_RUN_DIR/out}
mkdir -p "$O"
export PATH=$CUDA/bin:$PATH CUDA_HOME=$CUDA CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=$PWD:$PWD/tools/config:$PWD/adapters/vllm:$PWD/benchmarks/pouw:$PWD/benchmarks/harness${PYTHONPATH:+:$PYTHONPATH}
card=
for ((i = 1; i < $#; i++)); do
  if [[ ${!i} == --card ]]; then j=$((i + 1)); card=${!j}; fi
done
arch=$("$PY" -c 'import sys; from verity_bench.overhead import card; print(card(sys.argv[1]).arch)' "$card")
# the harness builds with identity.TOOLKIT (native/build.sh): the machine's at that version, else jobs/toolkit.sh assembles it
harness=$("$PY" -c 'import sys; sys.path[:0] = ["benchmarks/pouw/harness"]; import identity; print(identity.TOOLKIT)')
listed=0
home=$("$PY" benchmarks/kernels/machine.py toolkit "$harness") || listed=$?
case $listed in
  0) export CUDA_HOME=$home ;;
  3) . benchmarks/pouw/harness/jobs/toolkit.sh ;;
  *) exit "$listed" ;;
esac
# the harness loads the cuBLASLt of the toolkit it links (native/build.sh, -L$CUDA_HOME/lib64), which the node's own loader
# path need not hold
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
t0=$(date -u +%s.%N)
echo "$(date -u -d "@$t0" +%FT%TZ) phase build start"
POUW_HARNESS_LIB=$(PH_ARCH=$arch POUW_HARNESS_BUILD=${POUW_HARNESS_BUILD:-${XDG_CACHE_HOME:-$HOME/.cache}/pouw-harness-build} \
  bash benchmarks/pouw/harness/native/build.sh 2> "$O/harness-build.log")
t1=$(date -u +%s.%N)
build_s=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')
echo "$(date -u -d "@$t1" +%FT%TZ) phase build end, $build_s s"
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
  --run-id "$RESEARCH_RUN_ID" --build-s "$build_s" "${extra[@]}" "$@"
