# PoUW benchmarks

These benchmarks measure what [PoUW](../../verity/protocols/resource_accounting/compute/pouw/verifier/README.md) costs
on real GPUs. Most of them are about Pearl-C's kernels on the RTX PRO 6000: a kernel is first held bit for bit to what
the verifier recomputes, and only then timed. Beside them are the served vLLM path with its audits, probes of what a
card's instructions really do, and calculators that run on a CPU.

Nothing here is part of PoUW. No code outside `benchmarks/pouw/` imports it, and it decides no verdict.
How a kernel is designed, gated, timed and landed is the kernel-engineering skill
(`.agents/skills/kernel-engineering/SKILL.md`).

## Try it

~~~sh
uv run python benchmarks/pouw/contents.py
~~~

This prints what is here as a table: each directory or file, the hardware it runs on, and what it is. It needs nothing
but a checkout. `contents.py filler` and `contents.py kernels` print the filler's tools and the GEMM kernels the same way.
A new benchmark adds its own entry under `contents/` and edits nothing else, which is why this README lists none.

`verity bench pouw NAME [ARGS]` runs one directory's runner, and with no NAME it names the directories that have one. A
runner that needs a GPU or a whole node is the command of a `research run`, as its header says.

Served results (`pearl_c_vllm/`) are measured against stock vLLM's FP8 with its CUDA graphs. The ACCEL model and the ε
vocabulary these benchmarks use are [`machine_model/`](../machine_model/README.md)'s. [`DESIGN.md`](DESIGN.md) says what
a caller may import here, which files write what, and where the filler's tools and the GEMM kernels live.
