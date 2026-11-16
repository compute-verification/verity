# The store's Sloth codecs

Memory accounting's store is encoded under Sloth codecs, and this directory holds them. Each codec here is a byte-exact Python reference, a CUDA kernel and an arithmetic twin of the kernel: the word codec (Sloth in one
61-bit word, which `benchmarks/vllm_encoded` serves), the KV cache's codec `verity/word-codec/kv/v0`, whose kernel is
PoUB's prover's (`verity/protocols/resource_accounting/memory/poub/prover/word_codec/`), and the wide codecs.

**Backs:** the codec whose word the named assumptions
`ResourceAccounting.Assumptions.SquaringFloor` and `ChainSum` price is `kv/v0`, and the bandwidth audit's vectors test
(`tests/verity/protocols/resource_accounting/memory/poub/verifier/test_poub_vectors.py`) holds the audit's codec to it.

## Try it

~~~sh
uv run tools/check/suites.py benchmarks/word_codec
~~~

On the CPU, it checks the references against their vectors (`tests/vectors.json`, `tests/wide_vectors.json` and the
prover's `kv_vectors.json`) and the twins against the references. It writes no result. On one GPU,
`bash benchmarks/word_codec/run.sh` builds `kv.cu` at each width, runs its tests, then its benchmark, and writes the
gates, timings and profiles as JSON to `$RESEARCH_RUN_DIR/out`; its header lists `KV_FIELDS`, `KV_SCRIPT` and `KV_ARGS`.

[`DESIGN.md`](DESIGN.md) says what each file is, how memory accounting's benchmarks load a CUDA library, and what in
the tree reads this directory.
