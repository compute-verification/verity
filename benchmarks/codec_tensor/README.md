# benchmarks/codec_tensor

The bandwidth audit's (PoUB, `verity/protocols/poub/verifier`) work on the GPU, and the tensor-core wide-word
codec. `wt.cu` is the codec, a fused decode with its receipts, held to `wide.py`, its byte-exact reference. The device
reader is PoUB's prover's, in `verity/protocols/poub/prover/codec_tensor/`: `reader.cu` with `reader.py` gives the
audit's receipts, leaves and receipt tree, gated byte for byte against the audit's `receipts.py`; `sketch_receipt.py`
and `sketching.py` are the reference for the audit's `sketch-sha512` receipt, and `sha512.cuh` is SHA-512 for the
kernels. `step.cu` (`step.py`, `step.sh`) is one decode step at HBM exhaustion.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `uv run tools/verity/check/suites.py benchmarks/codec_tensor` | The suite, on the CPU: `wide.py`, `sketch_receipt.py` and `sketching.py` against PoUB's code and `tests/vectors.json`, and the decode step's plan. It writes no result. |
| `bash benchmarks/codec_tensor/step.sh`, with `MODEL` and `BATCH` | One decode step at HBM exhaustion on one GPU (`step.py`), after the codecs' tests and `reader.cu`'s gate. It writes its gates and timings as JSON to `$RESEARCH_RUN_DIR/out`. |

Nothing outside `benchmarks/` reads this directory by its path.

**Status: active.** `benchmarks/kv_filler`, `benchmarks/vllm_encoded` and `benchmarks/harness` build `wt.cu`;
`benchmarks/vllm_encoded` reads `wide.py`; and this suite holds the prover's `sketch_receipt.py` and `sketching.py` to
the audit's code. Decided by `rg codec_tensor` over
`verity/`, `benchmarks/`, `demos/`, `tests/` and `adapters/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/codec_tensor` checks `wide.py`, `sketch_receipt.py`
and `sketching.py` against the audit's code and `tests/vectors.json`, and the decode step's plan, on the CPU.

**Reads and writes:** the suite reads `benchmarks/word_codec`'s KV codec and the bandwidth audit's code; `bench.py`
writes `codec-tensor.json` and `bench.log` into its `--out`, and the step and reader scripts their gates and timings as
JSON. **Backs:** the receipts it computes on the GPU are the ones `Accounting.Guarantees.Remote.SketchBoundSession` and
`SketchBoundDeployed` bound, the memory line's sketch premise.
