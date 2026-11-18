# The bandwidth audit's work on the GPU

PoUB, the bandwidth audit (`verity/protocols/resource_accounting/memory/poub/verifier`), certifies how many reads of a
node's registered store the node made, from receipts the node computes as it reads. This directory is that work on the
GPU: the tensor-core wide-word codec (`wt.cu`, held to `wide.py`, its byte-exact reference). The receipts' device
reader is PoUB's prover's, in
`verity/protocols/resource_accounting/memory/poub/prover/codec_tensor/`.

**Backs:** the receipts it computes on the GPU are the ones `ResourceAccounting.Guarantees.Remote.SketchBoundRun` and
`SketchBoundDeployed` bound, the memory line's sketch premise.

## Try it

~~~sh
uv run tools/check/suites.py benchmarks/codec_tensor
~~~

On the CPU, it checks `wide.py`, `sketch_receipt.py` and `sketching.py` against the audit's code and
`tests/vectors.json`. It writes no result.

On one GPU, as the command of a `research run`, `verity bench codec_tensor --lib PATH [--lib PATH ...] --out DIR`
(`run.py`) gates `wt.cu`'s libraries (one per field) against `wide.py`, then times them. It writes `codec-tensor.json`
and `bench.log` under `--out`, and its numbers and its gates' validation as the run's `result.json` through the
harness's kit. The code it times is `bench.py`'s, which `benchmarks/vllm_encoded`'s stack also imports.
