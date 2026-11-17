# DSA top-k

The DeepSeek sparse attention (DSA) indexer picks, for each query row, the `k` keys with the largest indexer logits
(`k = 2048` for GLM-5.3 and DeepSeek-V3.2). At the adapter's vLLM pin, a row longer than `k` comes out in an order,
and with ties at the k-th logit even a set, that varies run to run: it is not a function of the logits. `dsa_topk.py`
is a vLLM plugin that rewrites those rows on the device into one canonical row, the registered selection
(`verity_vllm.definitions.registry.dsa_topk`) in ascending order, so that every row is. Its docstring defines the
canonical row and the device pass.

## Try it

~~~sh
uv run tools/check/suites.py benchmarks/dsa_topk
~~~

It tests the definition, the torch reference, the wrappers' dispatch on stand-in ops and the kept counters on the CPU;
the kernels' tests run where Triton is importable and are skipped otherwise.

`VERITY_DSA_TOPK_STEPS=DIR` keeps each step's counters on each card in `DIR/dsa_topk_steps.PID.jsonl`: a header naming
the counters, then a line a card a step, `{"step", "tokens", "device", "rows_long", ...}`, each count the step's own,
the step numbered as the step trace numbers it (a step that scheduled tokens, outside a capture and a dummy run) by a
wrapper of the model runner's `execute_model`; the drafter's forwards count in their step. At a step's start and end
the counters are copied into pinned host memory on the device's stream and the line is written once the end's copy has
landed, at a later step's end; nothing waits on the device but the process's exit.
`serve_tax` sets it beside each arm's step traces and joins the two by step (`serve_tax.topk_counts`).

## Using it

Any vLLM command at the integration's pin (a server, a replay, an in-process engine), from the pin's venv, with the
variables in the environment its engine processes inherit (`VLLM_PLUGINS` unset or naming `verity_dsa_topk`):

~~~text
python benchmarks/dsa_topk/dsa_topk.py site DIR
PYTHONPATH=DIR${PYTHONPATH:+:$PYTHONPATH} VERITY_DSA_TOPK=canonical [VERITY_DSA_TOPK_LOG=LOGDIR] COMMAND
~~~

`rows_long` above zero in a process's counters shows the plugin rewrote long rows there.

## Measuring

On a GPU node with the pin's venv (`adapters/vllm/verity_vllm/ops/pod_bootstrap.sh`'s `venv312`), `run.sh` stresses
every op at GLM-5.3's shapes, unpatched and canonical, across processes, and times the pass:

~~~text
research run --on NODE --project verity --source . --cwd source -- bash benchmarks/dsa_topk/run.sh
~~~
