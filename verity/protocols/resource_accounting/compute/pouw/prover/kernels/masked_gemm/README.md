# The masked GEMMs

Each kernel here is [`floored_gemm/`](../floored_gemm/README.md)'s GEMM with its operands hidden one way, timed
against one plain GEMM (cuBLASLt's, or the plain mainloop) on the same shapes, so the directory answers what each way of
hiding the operands costs the served GEMM. Each table prints an arm's cost beside its gate's verdict: every arm's
inputs, outputs and receipts are dumped at small shapes on poisoned output buffers, a CPU reference recomputes each bit,
and a table marks an arm whose gate failed or was not checked.

| Kernel | What is hidden, and how |
|---|---|
| `mgemm.cu` | masked operands: A + R and B + S at full rank, or low-rank masks at ranks 8 to 128, as one GEMM over stacked operands; int8 and E4M3's exact integer grid |
| `egemm.cu` | edited operands: fresh noise from {5, 7, -5, -7} added to both, as a separate pass or fused into the mainloop, with no peel |
| `sgemm.cu` | the edit with its noise expanded by SHA-512 in counter mode (`sha_noise.cuh`), over the alphabets ±1 and {-8..8} |
| `pvgemm.cu` | P V's noise: one shared edit a layer or fresh noise a head, added in a pass, in the mainloop, or where P is produced |
| `lpgemm.cu` | one-side-peeled noise: A edited by a rank-32 term that a tail stage or an extended cuBLASLt call peels, B by fresh ±1 |

## Try it

Without a GPU, run the five tables on made-up bench and gate files:

~~~sh
uv run pytest tests/benchmarks/pouw/test_masked_gemm_tables.py -q
~~~

It prints `5 passed`.

On a leased RTX PRO 6000 with the CUDA 13.0 toolkit, `run.sh` builds, gates and times the arms, from the clone's
root:

~~~text
research run --on MACHINE --project verity --source . --cwd clone --env GPU_LEASE_WHO=LANE \
  --send verity/protocols/resource_accounting/compute/pouw/prover/kernels/masked_gemm/run.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run.sh"'
~~~

It exits 4 when the build fails. [`DESIGN.md`](DESIGN.md) says what it measures for each way, the runner's variables
and what a run writes, the other jobs, and which code reads these files.
