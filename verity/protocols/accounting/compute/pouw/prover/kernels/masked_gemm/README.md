# The masked GEMMs

The mainloops [`edited_gemm/`](../edited_gemm/README.md)'s kernels build on: [`floored_gemm/`](../floored_gemm/README.md)'s
GEMM with its operands hidden.

| File | What it is |
|---|---|
| `mgemm.cu` | masked operands: A + R and B + S at full rank, or low-rank masks at ranks 8 to 128, as one GEMM over stacked operands; int8 and E4M3's exact integer grid |
| `lpgemm.cu` | one-side-peeled noise: A edited by a rank-32 term that a tail stage or an extended cuBLASLt call peels, B by fresh ±1 |
| `sha_noise.cuh` | noise expanded by SHA-512 in counter mode |
| `sha_pass.cuh` | the noise pass over an operand |

`edited_gemm/egemm.cu` includes `lpgemm.cu`. `mgemm.cu` is also timed on its own, against one plain GEMM (cuBLASLt's, or
the plain mainloop) on the same shapes: its inputs, outputs and receipts are dumped at small shapes on poisoned output
buffers, `mref.py` recomputes each bit on the CPU, and `mtable.py` prints each arm's cost beside its gate's verdict.

## Try it

Without a GPU, run the table on made-up bench and gate files:

~~~sh
uv run pytest tests/benchmarks/pouw/test_masked_gemm_tables.py -q
~~~

It prints `1 passed`.

On a leased RTX PRO 6000 with the CUDA 13.0 toolkit, `benchmarks/pouw/masked_gemm/run.sh` builds, gates and times
`mgemm.cu`'s arms, from the clone's root:

~~~text
research run --on MACHINE --project verity --source . --cwd clone --env GPU_LEASE_WHO=LANE \
  --send benchmarks/pouw/masked_gemm/run.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run.sh"'
~~~

It exits 4 when the build fails.
