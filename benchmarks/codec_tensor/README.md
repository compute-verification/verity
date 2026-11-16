# benchmarks/codec_tensor

The bandwidth audit's (PoUB, `verity/protocols/resource_accounting/memory/poub/verifier`) work on the GPU, and the tensor-core wide-word
codec. `wt.cu` is the codec, a fused decode with its receipts, held to `wide.py`, its byte-exact reference. The device
reader is PoUB's prover's, in `verity/protocols/resource_accounting/memory/poub/prover/codec_tensor/`: `reader.cu` with `reader.py` gives the
audit's receipts, leaves and receipt tree, gated byte for byte against the audit's `receipts.py`; `sketch_receipt.py`
and `sketching.py` are the reference for the audit's `sketch-sha512` receipt, and `sha512.cuh` is SHA-512 for the
kernels. `step.cu` is one decode step at HBM exhaustion.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `uv run tools/check/suites.py benchmarks/codec_tensor` | The suite, on the CPU: `wide.py`, `sketch_receipt.py` and `sketching.py` against PoUB's code and `tests/vectors.json`. It writes no result. |

Nothing outside `benchmarks/` reads this directory by its path.

**Status: active.** `benchmarks/kv_filler`, `benchmarks/vllm_encoded` and `benchmarks/harness` build `wt.cu`;
`benchmarks/vllm_encoded` reads `wide.py`; and this suite holds the prover's `sketch_receipt.py` and `sketching.py` to
the audit's code. Decided by `rg codec_tensor` over
`verity/`, `benchmarks/`, `demos/`, `tests/` and `adapters/`.

**Run:** `uv run tools/check/suites.py benchmarks/codec_tensor` checks `wide.py`, `sketch_receipt.py`
and `sketching.py` against the audit's code and `tests/vectors.json`, on the CPU.

**Reads and writes:** the suite reads `benchmarks/word_codec`'s KV codec and the bandwidth audit's code; `bench.py`
writes `codec-tensor.json` and `bench.log` into its `--out`, and the reader scripts their gates and timings as
JSON. **Backs:** the receipts it computes on the GPU are the ones `ResourceAccounting.Guarantees.Remote.SketchBoundRun` and
`SketchBoundDeployed` bound, the memory line's sketch premise.
