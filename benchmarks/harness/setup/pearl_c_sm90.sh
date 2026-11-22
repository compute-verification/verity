# Sourced by run_point.sh (a spec's `setup`): pearl_c.sh for an H100 or H200, building Pearl-C's sm_90a cubin from this
# source tree with verity/protocols/accounting/compute/pouw/prover/kernels/pearl_c_sm90/build.sh (nvcc, then its SASS gates) and exporting
# PEARLC_KERNEL and PEARLC_CUBIN, which the Pearl-C arm (benchmarks/pouw/pearl_c_vllm/bench_arm.py) loads. The cubin,
# ptxas's report, the gates' JSON, toolchain.txt and the sources' and cubin's sha256 go to $RESEARCH_RUN_DIR/out/setup.
# Pearl-C served as vLLM serves it, not under the window serve's batch invariance
unset VLLM_BATCH_INVARIANT
# sm_90's BF16-rows path off unless PEARLC_BF16_ROWS says otherwise: a whole capture forking each call's A tree and A
# scales races on it (BF16_RACE in benchmarks/pouw/pearl_c_vllm/e2e.py)
export PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0}
PC_OUT=${RESEARCH_RUN_DIR:-$PWD}/out/setup/pearl_c_sm90
mkdir -p "$(dirname "$PC_OUT")"
CUDA_HOME=$(python3 benchmarks/kernels/machine.py toolkit 13.0) || exit 4
PC_GENCODE=$(python3 benchmarks/kernels/machine.py gencode "$(python3 benchmarks/kernels/machine.py arch)") || exit 4
CUDA_HOME=$CUDA_HOME GENCODE=$PC_GENCODE bash verity/protocols/accounting/compute/pouw/prover/kernels/pearl_c_sm90/build.sh "$PC_OUT" \
  || { echo "setup: verity/protocols/accounting/compute/pouw/prover/kernels/pearl_c_sm90/build.sh failed" >&2; exit 4; }
# shellcheck disable=SC2046  # sources.txt's paths hold no space
sha256sum $(cat "$PC_OUT/sources.txt") verity/protocols/accounting/compute/pouw/prover/kernels/pearl_c_sm90/run.py verity/protocols/accounting/compute/pouw/prover/kernels/pearl_c_sm90/grouped.py \
  verity/protocols/accounting/compute/pouw/prover/kernels/pearl_c/driver.py "$PC_OUT/pearl_c_sm90.cubin" "$PC_OUT/debit/debit_sm90.cubin" > "$PC_OUT/sha256.txt"
export PEARLC_KERNEL=$PWD/verity/protocols/accounting/compute/pouw/prover/kernels/pearl_c_sm90 PEARLC_CUBIN=$PC_OUT/pearl_c_sm90.cubin
