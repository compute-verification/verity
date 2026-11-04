# load_time: a large model's time to engine-ready under vLLM

`load_time.py` times a large model at tensor parallelism 8 from process start to a constructed `vllm.LLM`, and to its first
8-token generation, under each page-cache state (warm, or evicted with `posix_fadvise`), compile-cache state (cold, or warm
from an earlier configuration of the same run) and loader (vLLM's default or fastsafetensors), and once eager, without CUDA
graphs. Each configuration runs in a fresh process. It records the phases vLLM logs (weights load, torch.compile, graph
capture, engine init) and the fraction of the model's files resident in the page cache before and after (mincore), which
means something only when the job owns those files.

A served run starts its engine before it serves, so start-up is a fixed cost beside the serving it measures; this says how
much of that cost the page cache, the compile cache and the loader remove.

On a machine with 8 GPUs and vLLM installed (and fastsafetensors for its configurations), from the repository root:

~~~text
python benchmarks/load_time/load_time.py --model DIR --out OUT [--plan PLAN] [--tp 8]
~~~

`--plan` picks the configurations. `default` runs them all, and `quick` the warm page cache with a cold and a warm compile
cache, and eager. `fastsafetensors` and `fastsafetensors-restart` fill the compile cache with the default loader, then run
fastsafetensors (on a warm and a cold page cache, or twice warm); `fastsafetensors-warm` runs it once, on a cold compile
cache.

It writes `OUT/summary.json` after each configuration (the model's size, versions and GPUs, a cold read's bandwidth, and
per configuration its wall times, phases, residency and exit code) and `OUT/<config>.log` with the child's output.

Status: kept for reference. Nothing else in the repository runs it, and it has no tests.
