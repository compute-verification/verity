# benchmarks/codec_tensor

The bandwidth audit's (PoUB, `verity/protocols/poub/verifier`) work on the GPU, and the tensor-core wide-word
codec. `wt.cu` is the codec, a fused decode with its receipts, held to `wide.py`, its byte-exact reference. `reader.cu`
(with `reader.py`) is the device reader: the audit's receipts, leaves and receipt tree, gated byte for byte
against the audit's `receipts.py`. `sketch_receipt.py` is the reference for the audit's `sketch-sha512` receipt.
`step.cu` (`step.py`) is one decode step at HBM exhaustion, and `sha512.cuh` is SHA-512 for the kernels.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `uv run tools/verity/check/suites.py benchmarks/codec_tensor` | The suite, on the CPU: `wide.py`, `sketch_receipt.py` and `sketching.py` against PoUB's code and `tests/vectors.json`, and the decode step's plan. It writes no result. |
| `step.py`, with `--model`, `--batch` and a build of each of `wt.cu`, `step.cu` and `reader.cu` (`--wt-lib`, `--kv-lib`, `--reader-lib`) | One decode step at HBM exhaustion on one GPU. It writes `step.json` and `step.log` under `--out`. Nothing in the tree builds `step.cu`, whose build takes `-DKV_FIELD`, `-DKV_MASK` and `-DKV_FOLD6` at `wt.cu`'s field and codec, and `-DKV_QHEADS` at the model's query heads. |

Nothing outside `benchmarks/` reads this directory by its path. `reader.cu` leaves it inside
`benchmarks/remote_session/session.cu`'s build.

**Status: active.** `benchmarks/remote_session/session.cu` includes `reader.cu`; `benchmarks/kv_filler`,
`benchmarks/vllm_encoded` and `benchmarks/harness` build `wt.cu` and `reader.cu`; `benchmarks/pouw/served_h1` includes
`sha512.cuh`; and the audit's `PROTOCOL.md` and vectors test name `sketch_receipt.py`. Decided by `rg codec_tensor` over
`verity/`, `benchmarks/`, `demos/`, `tests/` and `adapters/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/codec_tensor` checks `wide.py`, `sketch_receipt.py`
and `sketching.py` against the audit's code and `tests/vectors.json`, and the decode step's plan, on the CPU.

**Reads and writes:** the suite reads `benchmarks/word_codec`'s KV codec and the bandwidth audit's code; `bench.py`
writes `codec-tensor.json` and `bench.log` into its `--out`, and the step and reader scripts their gates and timings as
JSON. **Backs:** the receipts it computes on the GPU are the ones `Accounting.Guarantees.Remote.SketchBoundSession` and
`SketchBoundDeployed` bound, the memory line's sketch premise.
