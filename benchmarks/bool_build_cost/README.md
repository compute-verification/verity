# The Boolean Build's cost against the word Build's

A workload's circuit can be written over words or over bits. This benchmark asks how much the bit version costs: it
takes one workload's Calls as their word specializations and as their Boolean versions, and for each measures the
descriptor of one Program that calls every specialization once, and the time to compute the `Q_word_v1{16,32}`
proof-block rule of every Definition under them. The root body of a workload's Program, with its KV-cache references,
is not in this Program.

## Try it

~~~sh
verity bench bool_build_cost --workload small --part descriptor --out DIR
~~~

This needs no GPU and takes seconds. It prints the descriptor's size and digest for the word and the Boolean
version, and writes each as `DIR/<ir>.json`. The other workload, `qwen3-06b`, is Qwen3-0.6B's B1 workload with a
256-token prompt and a 32-token output; its descriptor part also runs on a VM.

The `rules` part is the heavy one. A Boolean head is tens of millions of bit gates, so run it on a machine with 16
cores and 160 GB:

~~~text
uv sync --all-packages --extra torch-cpu
uv run benchmarks/bool_build_cost/run.py --workload qwen3-06b --jobs 16 --out DIR
~~~

`run.py`'s docstring gives every option and what each file it writes holds.
