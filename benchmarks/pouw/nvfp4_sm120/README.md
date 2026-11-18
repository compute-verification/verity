# The plain NVFP4 and FP8 GEMMs on the RTX PRO 6000

A plain GEMM on sm_120a over operands that carry one scale a scale group, with no proof of work in it, written so that a
Pearl-C4 or Pearl-C kernel is timed against a baseline on the same mainloop. `mainloop_nvf4.cuh` computes one tile of
C = A Bᵀ as the chain of tensor-core atoms the silicon model states (`BLACKWELL_SM120_NVF4`; in its FP8 mode
`BLACKWELL_SM120_E4M3_M16N8K32`), so its words are the model's whatever the schedule. Operands and outputs are in the
layout CUTLASS's and cuBLASLt's GEMMs take.

These GEMMs are baselines and nothing more: they run inside the GPU harness ([`../harness/`](../harness/README.md)),
which gates and times them with every other arm, and a number of record is the harness's. Nothing in this directory is
started by itself, and nothing here commits, hashes or checks a proof.

## Try it

List the baselines `nvf4_plain.cu` adds to the harness's registry:

~~~sh
grep -o '{"verity_[a-z0-9_]*"' benchmarks/pouw/nvfp4_sm120/nvf4_plain.cu
~~~

It prints ten names, six NVFP4 (`verity_nvf4_256x128...`) and four FP8 (`verity_fp8_256x128...`).
