# benchmarks/codec_tensor

The bandwidth audit's (PoUB, `verity/core/protocols/remote_bandwidth`) work on the GPU, and the tensor-core wide-word
codec. `wt.cu` is the codec, a fused decode with its receipts, held to `wide.py`, its byte-exact reference. `reader.cu`
(with `reader.py`, `reader.sh`) is the device reader: the audit's receipts, leaves and receipt tree, gated byte for byte
against the audit's `receipts.py`. `sketch_receipt.py` is the reference for the audit's `sketch-sha512` receipt.
`step.cu` (`step.py`, `step.sh`) is one decode step at HBM exhaustion, and `sha512.cuh` is SHA-512 for the kernels.

**Status: active.** `benchmarks/remote_session/session.cu` includes `reader.cu`; `benchmarks/kv_filler`,
`benchmarks/vllm_encoded` and `benchmarks/harness` build `wt.cu` and `reader.cu`; `benchmarks/pouw/served_h1` includes
`sha512.cuh`; and the audit's `PROTOCOL.md` and vectors test name `sketch_receipt.py`. Decided by `rg codec_tensor` over
`verity/`, `benchmarks/`, `examples/`, `tests/` and `integrations/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/codec_tensor` checks `wide.py`, `sketch_receipt.py`,
`sketching.py` and `tagging.py` against the audit's code and `tests/vectors.json`, and the decode step's plan, on the
CPU. `run.sh` builds `wt.cu` for each field and runs `bench.py` on one GPU (`RECEIPT=1` adds `reader.cu`).

**Reads and writes:** the suite reads `benchmarks/word_codec`'s KV codec and both bandwidth audits' code; `bench.py`
writes `codec-tensor.json` and `bench.log` into its `--out`, and the step and reader scripts their gates and timings as
JSON. **Backs:** the receipts it computes on the GPU are the ones `Accounting.Guarantees.Remote.SketchBoundSession` and
`SketchBoundDeployed` bound, the memory line's sketch premise.
