# Route capture

To prove what a served model computes, its circuit has to cover every path a served step can take. This capture
records what GLM-5.3 actually executes when served the way serving runs it: every engine step's launch-shape class
(phase mix, tokens, requests, attention route), and the device kernels of the first steps of each class, read with
`torch.profiler` because CUDA graphs hide most kernels from Python. The gap, below, lists what class G's circuits
don't yet cover. `route_capture_plugin.py`'s docstring lists the files a capture writes.

## Try it

~~~sh
uv run tools/check/suites.py benchmarks/route_capture
~~~

It runs the plugin on stand-ins for both model runners and torch's profiler, and `gap.py` on synthetic steps, on the
CPU in a few seconds.

The capture itself serves GLM-5.3 on eight H200s at TP8 under four concurrent clients for about ten minutes
(`run.sh`'s header lists its variables; [`DESIGN.md`](DESIGN.md) has the clients and the serving settings):

    # on the node, from a checkout of this tree, under research run (VENV's vLLM at the integration's pin)
    research run --on NODE --project verity --source CHECKOUT --cwd source -- bash benchmarks/route_capture/run.sh

After the server stops, `run.sh` keeps whatever the engine compiled (vLLM's and Inductor's Python, which name the ATen
nodes each fused Triton kernel computes, and Triton's kernel metadata) in `compiled-code.tar.gz`, and writes
`summary.json`:

    python benchmarks/route_capture/routes.py summary OUT/capture    # each class's steps, tokens, dispatches and kernels

## The gap

`gap.py` evaluates every captured step as the class G decoder would write its launch (`verity_vllm.program.predict.step_needs`,
which counts what `predict.dsa` calls without building the Program), and lists what class G doesn't cover:

    python benchmarks/route_capture/gap.py OUT/capture --check CIRCUIT_CHECK.json --json gap.json --md gap.md

`--check` is circuit-check's JSON report over class G's Definitions: an id it lists is registered, and passes when it has
no failures. Each selection the decoder makes reads the step's launch size (the graph dispatch's padded tokens, else its
tokens) and its rows' key counts, through the decoder's own registry calls; each Definition is the vocabulary's kind at
the statics the decoder passes, and a Definition call counts as a proof block. Where the decoder would refuse, the calls
are counted against the refusal and the evaluation goes on; the integration's tests hold the counts to the decoder's own
Program for an engine step. A row of the table is one (route, Definition or refusal, status), with the range of statics the
capture needed, whether a Definition exists at other statics, its owner (circuits for a Definition, kernels for a kernel
capture or a deterministic kernel, vllm for serving), its Definition calls a scheduled token on the recording rank, and
its share of all calls. A second table counts the tokens of steps outside class G's engine (CUDA graphs, padded
launches, prefix-cache hits, class G's limits). The JSON adds `needs`: every Definition or refusal id with its calls on
the recording rank, whether circuit-check registers it and whether it passes. vLLM's kernel warm-up runs its dummy
requests, each with every sampling feature, through the same `execute_model` before serving; the plugin records those
steps, and `gap.py` leaves the capture's leading ones out and counts them (`warmup`). FlashInfer's sampler takes a step's
unseeded requests only when one filters by top-k or top-p and none is seeded, so a seeded request keeps its whole batch
on the native sampler.

