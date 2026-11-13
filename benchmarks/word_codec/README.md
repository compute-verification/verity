# benchmarks/word_codec

The store's Sloth codecs, each a byte-exact Python reference, a CUDA kernel and an arithmetic twin of the kernel: the
word codec (`codec.py`, `wc.cu`, `twin.py`; Sloth in one 61-bit word, which `benchmarks/vllm_encoded` serves), the KV
cache's codec `verity/word-codec/kv/v0` (`kv.py`, `kv.cu`, `kv_twin.py`, and `kv_bench.py` its benchmark), which the
bandwidth audit (PoUB, `verity/protocols/poub/verifier`) mirrors in its `codec.py`, and the wide codecs
(`wide.py`, `wide.cu`). `cuda_lib.py` is how memory accounting's benchmarks
(`codec_tensor`, `kv_filler`, `pous`, `remote_session`, `vllm_encoded` and this one) open a built CUDA library and bind
its functions' types; each library's table of them sits beside the script that loads it.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `uv run tools/verity/check/suites.py benchmarks/word_codec` | The suite, on the CPU: the references against their vectors and the twins against the references. It writes no result. |
| `bash benchmarks/word_codec/run.sh`, with `KV_FIELDS`, `KV_SCRIPT` and `KV_ARGS` as its header lists them | `kv.cu` built at each width, its tests, then its benchmark on one GPU (`kv_bench.py` unless `KV_SCRIPT` names another). It writes the gates, timings and profiles as JSON to `$RESEARCH_RUN_DIR/out`. |

Read from outside `benchmarks/`: `kv.cu`'s build, taken by its path, and `kv.py` and `kv_bench.py`, reached as
`benchmarks/vllm_encoded/paged.py`'s `K` and `KB`.

**Status: active.** `kv.cu` is the store's encode on the GPU: `benchmarks/remote_session/run.sh` builds it, the vLLM
integration calls it (`adapters/vllm/verity_vllm/protocols/remote_bandwidth/remote.py`), and `benchmarks/vllm_encoded`
and `benchmarks/kv_filler` build it. The `codec_tensor`, `kv_filler` and `vllm_encoded` suites read
its Python as inputs. Decided by `rg word_codec` over `verity/`, `benchmarks/`, `demos/`, `tests/` and `adapters/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/word_codec` checks the references against their vectors
(`tests/vectors.json`, `tests/kv_vectors.json`, `tests/wide_vectors.json`) and the twins against the references, on the
CPU. `run.sh` is `kv.cu`'s GPU benchmark.

**Reads and writes:** the GPU scripts read the built kernels and write their gates, timings and profiles as JSON in their
output directory; the suite reads only the vectors. **Backs:** the codec whose word the named assumptions
`Accounting.Assumptions.SquaringFloor` and `ChainSum` price is `kv/v0`, and the bandwidth audit's vectors test
(`verity/protocols/poub/verifier/tests/test_remote_bandwidth_vectors.py`) holds the audit's codec to it.
