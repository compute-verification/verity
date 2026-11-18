# The floored GEMM

`fgemm.cu` is the served GEMM on the exact int8 grid: the operands are noised a chunk of 32 columns at a time, the
tensor cores multiply the noised operands, and an exact peel returns the product of the codes. The directory measures
what a receipt of the running sums costs that GEMM, against cuBLASLt's int8 GEMM on the same shapes, and times nothing
until the gate passes: every arm's outputs and receipts are dumped at small shapes on poisoned output buffers, and
`ref.py` recomputes each bit on the CPU. Later kernels build on it: [`masked_gemm/`](../masked_gemm/README.md)'s
`mgemm.cu` includes `fgemm.cu`, and the edited GEMMs' references share `derive.py`'s receipts.

## Try it

Without a GPU, run the CPU scripts on made-up input:

~~~sh
uv run pytest tests/benchmarks/pouw/test_floored_gemm_scripts.py tests/benchmarks/pouw/test_floored_gemm_table.py -q
~~~

It prints `10 passed`.

On a leased RTX PRO 6000 with the CUDA 13.0 toolkit, `ncu` and `cuobjdump`, `run.sh` builds, gates and times the
arms, from the clone's root:

~~~text
research run --on MACHINE --project verity --source . --cwd clone --env GPU_LEASE_WHO=LANE \
  --send verity/protocols/resource_accounting/compute/pouw/prover/kernels/floored_gemm/run.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run.sh"'
~~~

It exits 4 when the build fails, 5 when the binary holds no IMMA and 6 when the gate fails.
