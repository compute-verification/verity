# nvfp4_sm120: the plain NVFP4 and FP8 GEMMs on the RTX PRO 6000

A plain GEMM on sm_120a over operands that carry one scale a scale group, with no proof of work in it, written so that a
Pearl-C4 or Pearl-C kernel is timed against a baseline on the same mainloop. `mainloop_nvf4.cuh` computes one tile of
C = A Bᵀ as the chain of tensor-core atoms the silicon model states (`BLACKWELL_SM120_NVF4`; in its FP8 mode
`BLACKWELL_SM120_E4M3_M16N8K32`), so its words are the model's whatever the schedule. Operands and outputs are in the
layout CUTLASS's and cuBLASLt's GEMMs take.

| File | What it is |
| --- | --- |
| `mainloop_nvf4.cuh` | the TMA-fed, warp-specialized mainloop, NVFP4 and FP8 |
| `nvf4_plain.cu` | the GEMMs on it as entries of the GPU harness's baseline registry; `../harness/native/build.sh` builds it into the harness's library |

## Running it

The baselines run inside the harness (`../harness/`), which gates and times them with every other arm; a number of
record is the harness's. Nothing in this directory is started by itself.

## Who reads it

The harness reads `nvf4_plain.cu`: its build compiles it, with the mainloop, into the harness's library, and
`../tests/test_harness.py` reads its text for its registry entries. Nothing here commits, hashes or checks a proof.
