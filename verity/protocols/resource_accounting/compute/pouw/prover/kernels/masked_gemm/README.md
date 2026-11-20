# The masked GEMMs

The mainloops [`edited_gemm/`](../edited_gemm/README.md)'s kernels build on: [`floored_gemm/`](../floored_gemm/README.md)'s
GEMM with its operands hidden.

| File | What it is |
|---|---|
| `mgemm.cu` | masked operands: A + R and B + S at full rank, or low-rank masks at ranks 8 to 128, as one GEMM over stacked operands; int8 and E4M3's exact integer grid |
| `lpgemm.cu` | one-side-peeled noise: A edited by a rank-32 term that a tail stage or an extended cuBLASLt call peels, B by fresh ±1 |
| `sha_noise.cuh` | noise expanded by SHA-512 in counter mode |
| `sha_pass.cuh` | the noise pass over an operand |

They are gated and timed only through the kernels that include them: `edited_gemm/egemm.cu` includes `lpgemm.cu`, and
`benchmarks/pouw/edited_gemm/crun.sh` gates `cgemm.cu` on it.
