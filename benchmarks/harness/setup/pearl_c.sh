# Sourced by run_point.sh (a spec's `setup`): builds Pearl-C's sm_120 cubin from this source tree with
# verity/protocols/accounting/pouw/prover/kernels/pearl_c_sm120/build.sh (nvcc, then its SASS gate) and exports PEARLC_CUBIN, which the Pearl-C
# arm (benchmarks/pouw/pearl_c_vllm/bench_arm.py) loads. The cubin, ptxas's report, sass.json, toolchain.txt and the sources' and cubin's
# sha256 go to $RESEARCH_RUN_DIR/out/setup. Built per point, so the cubin always matches the commit's run.py.
PC_OUT=${RESEARCH_RUN_DIR:-$PWD}/out/setup/pearl_c_sm120
mkdir -p "$(dirname "$PC_OUT")"
CUDA=${CUDA_HOME:-/usr/local/cuda-13.0} bash verity/protocols/accounting/pouw/prover/kernels/pearl_c_sm120/build.sh "$PC_OUT" \
  || { echo "setup: verity/protocols/accounting/pouw/prover/kernels/pearl_c_sm120/build.sh failed" >&2; exit 4; }
sha256sum verity/protocols/accounting/pouw/prover/kernels/pearl_c_sm120/pearl_c_sm120.cu verity/protocols/accounting/pouw/prover/kernels/pearl_c_sm120/*.cuh \
  verity/protocols/accounting/pouw/prover/kernels/pearl_c_sm120/run.py "$PC_OUT/pearl_c_sm120.cubin" > "$PC_OUT/sha256.txt"
export PEARLC_CUBIN=$PC_OUT/pearl_c_sm120.cubin
