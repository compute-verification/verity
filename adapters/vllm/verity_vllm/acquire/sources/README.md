# verity_vllm.acquire.sources

A source is how a Commit gets hold of a value while vLLM computes it. It is an object a Commit attaches to its
committer: it takes values from the engine while it runs and hands them to the committer during a committed step.
There are sources for FlashAttention's hidden values, for MoE, for compiled kernels and for a tensor-parallel rank's
partials.

Each module's docstring says which values its source holds and where it takes them, and `taps` attaches the sources
a Commit's manifest names. [`DESIGN.md`](DESIGN.md) lists the names other packages import from here.
