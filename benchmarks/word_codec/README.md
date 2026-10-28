# benchmarks/word_codec

The store's Sloth codecs, each a byte-exact Python reference, a CUDA kernel and an arithmetic twin of the kernel: the
word codec (`codec.py`, `wc.cu`, `twin.py`; Sloth in one 61-bit word, which `benchmarks/vllm_encoded` serves), the KV
cache's codec `verity/word-codec/kv/v0` (`kv.py`, `kv.cu`, `kv_twin.py`, and `kv_bench.py` its benchmark), which the
bandwidth audit (PoUB, `verity/core/protocols/remote_bandwidth`) mirrors in its `codec.py`, and the wide codecs
(`wide.py`, `wide.cu`). `encode_rate.py` times `kv.cu`'s encode against the stated squaring ceiling, and
`model/gamma.py` is the rate audit's γ as arithmetic. `cuda_lib.py` is how memory accounting's benchmarks
(`codec_tensor`, `kv_filler`, `pous`, `remote_session`, `vllm_encoded` and this one) open a built CUDA library and bind
its functions' types; each library's table of them sits beside the script that loads it.

**Status: active.** `kv.cu` is the store's encode on the GPU: `benchmarks/remote_session/run.sh` builds it, the vLLM
integration calls it (`integrations/vllm/verity_vllm/protocols/remote_bandwidth/remote.py`), and `benchmarks/vllm_encoded`
and `benchmarks/kv_filler` build it. The `codec_tensor`, `kv_filler`, `serve_bandwidth` and `vllm_encoded` suites read
its Python as inputs. Decided by `rg word_codec` over `verity/`, `benchmarks/`, `demos/`, `tests/` and `integrations/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/word_codec` checks the references against their vectors
(`tests/vectors.json`, `tests/kv_vectors.json`, `tests/wide_vectors.json`) and the twins against the references, on the
CPU. `run.sh` is `kv.cu`'s GPU benchmark, and `encode_rate.sh` builds `kv.cu` and `benchmarks/rate_squaring/sq.cu` and
runs `encode_rate.py`.

**Reads and writes:** the GPU scripts read the built kernels and write their gates, timings and profiles as JSON in their
output directory; the suite reads only the vectors. **Backs:** the codec whose word the named assumptions
`Accounting.Assumptions.SquaringFloor` and `ChainSum` price is `kv/v0`, and the bandwidth audit's vectors test
(`verity/core/protocols/remote_bandwidth/tests/test_remote_bandwidth_vectors.py`) holds the audit's codec to it.
