# The bandwidth audit's work on the GPU

PoUB, the bandwidth audit (`verity/protocols/resource_accounting/memory/poub/verifier`), certifies how many reads of a
node's registered store the node made, from receipts the node computes as it reads. This directory is that work on the
GPU: the tensor-core wide-word codec (`wt.cu`, held to `wide.py`, its byte-exact reference), and one decode step at HBM
exhaustion (`step.py`). The receipts' device reader is PoUB's prover's, in
`verity/protocols/resource_accounting/memory/poub/prover/codec_tensor/`.

**Backs:** the receipts it computes on the GPU are the ones `ResourceAccounting.Guarantees.Remote.SketchBoundRun` and
`SketchBoundDeployed` bound, the memory line's sketch premise.

## Try it

~~~sh
uv run tools/check/suites.py benchmarks/codec_tensor
~~~

On the CPU, it checks `wide.py`, `sketch_receipt.py` and `sketching.py` against the audit's code and
`tests/vectors.json`, and the decode step's plan. It writes no result.

| Script | Needs | Writes under `--out` |
| --- | --- | --- |
| `bench.py` | one GPU, and `wt.cu`'s libraries (`--lib PATH`, one per field) | `codec-tensor.json` and `bench.log` |
| `step.py` | one GPU, and builds of `wt.cu`, `step.cu` and the prover's `reader.cu` | `step.json` and `step.log` |
