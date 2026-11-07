# Route capture

What a model served like production executes, at the granularity the ahead-of-time step compiler names routes in: every
engine step's launch-shape class, and the device kernels of the first steps of each class. A compiled `vllm serve` hides
most of a step from Python (CUDA graphs replay the kernels Inductor fused), so the capture reads the step's shape from
its scheduler output, the graph dispatch from the model runner, and the kernels from `torch.profiler`, which sees the
kernels a graph replays.

`run.sh` serves GLM-5.3 on eight H200s at TP8 the way production does: `vllm serve` at vLLM's defaults (compiled, CUDA
graphs full and piecewise, chunked prefill, prefix caching, async scheduling, the model runner vLLM picks), at most 256
sequences and 8,192 tokens a step, prompts up to 32k tokens. Four `vllm bench serve` clients run for about `DURATION_S`
(600 s) at once:

| Client | Prompts | Sampling | Arrivals |
| --- | --- | --- | --- |
| `sharegpt-greedy` | ShareGPT | greedy | Poisson, ramped linearly from 1 to 24 requests/s |
| `sharegpt-sampled` | ShareGPT | temperature 0.7, top-p 0.95 | Poisson, 2/s |
| `long-context` | random, 8,192 to 32,768 tokens | temperature 0.7, top-p 0.95 | Poisson, 0.1/s |
| `shared-prefix` | 16 shared 4,096-token prefixes, each with a 512-token suffix | greedy | Poisson, 1/s |

    # on the node, from a checkout of this tree, under research run (VENV's vLLM at the integration's pin)
    research run --on NODE --project verity --source CHECKOUT -- bash benchmarks/route_capture/run.sh

`run.sh`'s header lists its variables (`VENV`, `CKPT`, `SHAREGPT`, `DURATION_S`). The FlashInfer sampler stays on,
vLLM's default; if a sampled smoke request fails with it, the server restarts with it off, and `engine_env.txt` says so.

## The plugin

`route_capture_plugin.py` is a vLLM general plugin (the entry point `run.sh` writes into a site directory on
`PYTHONPATH`, the only plugin `VLLM_PLUGINS` loads). `register` runs in every process and patches the V1 and V2 model
runners when their modules are imported; `settings.py` declares its variables (`ROUTECAP_OUT` turns it on). On each
recording rank (`ROUTECAP_RANKS`, rank 0 by default) it writes into `ROUTECAP_OUT`:

| File | What it holds |
| --- | --- |
| `steps.rank<R>.jsonl` | one row per step: tokens, requests, each prefill's query length, computed tokens and prompt length, the decodes' context lengths, the requests' sampling kinds, the graph dispatch (mode, padded tokens and requests), the step's class and its trace's name when profiled |
| `kineto/rank<R>/step<N>.json.gz` | a profiled step's Chrome trace: every kernel with its grid and block, those replayed in a graph included, and the eager operators' input shapes |
| `capture_ops.rank<R>.json` | the operators run while the CUDA graphs are captured, by name and input shapes, over every captured size |
| `engine.rank<R>.json` | the runner's model, cache, scheduler, parallel and compilation configuration, and the versions |

A step's class (`key`) is its phase mix (decode, prefill or mixed), its token count rounded up to `ROUTECAP_M_EDGES` (which
hold every M where a GLM-5.3 route switches), its request count rounded up to a power of two, the prefills' attention route
(dense while the longest prefill sequence is at most the indexer's top-k, sparse past it), whether a new request hit the
prefix cache, and the sampler's mix. A rank profiles the first `ROUTECAP_PER_CLASS` steps of each class, at most
`ROUTECAP_BUDGET` in all, and past its first `ROUTECAP_WARM` only while profiling has held it for at most `ROUTECAP_DUTY`
of the run. A profiled step synchronizes the device at its start and end, so the steps around it batch a little more
than they would have. Every step gets a row; the kernels of an unprofiled step are those of its class.

After the server stops, `run.sh` keeps the compile's generated code (vLLM's and Inductor's Python, which name the ATen
nodes each fused Triton kernel computes, and Triton's kernel metadata) in `compiled-code.tar.gz`, and writes
`summary.json`:

    python benchmarks/route_capture/routes.py summary OUT/capture    # each class's steps, tokens, dispatches and kernels

The suite (`tests/`) runs the plugin on stand-ins for both model runners and torch's profiler.
