# moe_align: vLLM's MoE alignment, made a function of its inputs

Before a Marlin MoE GEMM, vLLM's `moe_align_block_size` groups the router's (token, top-k slot) pairs by expert and pads
each expert's group to a multiple of the moe block. With more than 64 experts or at least 1,024 pairs, vLLM's kernel
orders the pairs within an expert by the order its atomics run in, which varies from run to run. That order decides
which pairs share a Marlin block, and so where the GEMM's reductions are cut.

The vLLM adapter replaces the op on the Marlin path with `verity_vllm.engine.moe_align`, which writes the same buffers
with each expert's pairs in ascending pair index, with no atomics. It is always on in the adapter, and its module's
docstring defines its outputs and says how it is installed. This directory checks it on a GPU and times it against
vLLM's op.

## Try it

On a GPU node:

~~~text
research run --on NODE --project verity --source . --cwd source -- bash benchmarks/kernels/moe_align/run.sh
~~~

At DeepSeek-V4-Pro's routed experts (E = 384, top-k 6), it runs both ops on the same inputs, in one process and then in
two at once, and records whether every output of the adapter's op is the reference bit for bit (`summary.json`). It
then loads the adapter's plugin as vLLM does (`plugin.json`) and times both ops (`time.json`). It builds a venv of the
pinned vLLM wheel first if it needs one. A failed step is named in `$OUT/failed`, and the script exits 1.

`served.sh` checks and times the op inside a served engine, on DeepSeek-V4-Pro's 8 x H200 node or on fewer GPUs with
dummy weights. [`DESIGN.md`](DESIGN.md) gives its commands, what each step records, and how the op places its pairs.
