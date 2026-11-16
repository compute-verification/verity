# bench-instances/v1

This is the frozen input set every K=1536 proof-backend number is measured on: real and corner instances of one GEMM
coordinate under the relation `verity.circuits.targets.target.FIRST` (96 sequential Ampere `mma.sync.m16n8k16 bf16`
transitions, then rounding to BF16), each with its expected word and its provenance, and every array pinned by its sha256
in `manifest.json`. The expected words, the negatives, the index files and the seeds are committed; the other arrays
are rebuilt byte for byte and must match their digests.

## Try it

Check the committed files against the manifest:

~~~sh
uv run python -m verity_numerical.bench.instances verify --root benchmarks/numerical/bench-instances/v1
~~~

It ends with `OK`; the recheck of the built arrays is skipped until they are built. [`DESIGN.md`](DESIGN.md) has the
relation in full, each file's layout, the index files, the composition, how to rebuild the arrays and what was checked
while building them.
