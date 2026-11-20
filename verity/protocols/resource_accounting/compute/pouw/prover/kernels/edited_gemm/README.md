# The edited GEMM

These kernels serve a linear or an attention product on dithered or edited int8 operands, with receipts of the running
sums, on the RTX PRO 6000 (sm_120), with a port to the A100 (sm_80). They are built on
[`masked_gemm/`](../masked_gemm/README.md)'s mainloop. The directory measures what serving that way costs over the
plain product, and an arm is timed only once it is right: the SASS gate reads the binary, the device gate runs every arm
on poisoned output buffers with a no-write control, and a CPU reference recomputes the dumped tiles from the inputs and
the key alone.

## Try it

Without a GPU, run the CPU reference of `cgemm.cu`'s gate:

~~~sh
uv run pytest tests/verity/protocols/resource_accounting/compute/pouw/prover/kernels/edited_gemm/test_cgemm_ref.py -q
~~~

On a leased RTX PRO 6000 with the CUDA 13.0 toolkit, `ncu` and `cuobjdump`, `benchmarks/pouw/edited_gemm/crun.sh`
builds, gates and times `cgemm.cu` (`aprun.sh` and `cpvrun.sh` the attention products), from the source tree's root:

~~~text
research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  -- bash benchmarks/pouw/edited_gemm/crun.sh
~~~

It exits 4 when the build fails and 5 when a gate fails.
