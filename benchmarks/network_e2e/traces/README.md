# vLLM scheduler traces under varied traffic

Real vLLM V1 scheduler traces for network attestation's advice: what serving made unpredictable, step by step, under
Poisson traffic. Plain vLLM's OpenAI server on one GPU (an RTX PRO 6000 or an H200), with the integration's
batch-invariant eager engine (bi-eager) and FlashAttention 2 (`corpus.FA2`), or on a whole 8-GPU H200 node at TP8.
`corpus.MODELS` lists the models: `llama32_1b` (unsloth/Llama-3.2-1B, bf16), `llama31_8b_int8`
(`verity-int8/Llama-3.1-8B-INT8` with the int8_exact linears, `--quantization verity_int8`) and `glm53` (GLM-5.3,
block-FP8, at TP8 with vLLM's own kernels, at a rate given with `--rate`). A request stops on the checkpoint's
end-of-text ids (`corpus.stop_ids`), which the corpus's meta.json records as `eos_token_ids`.

Think of a trace as one load served on a fresh vLLM server with a hook in its scheduler, which writes, as JSON lines,
every request handed to the scheduler, every step's batch, the sampled tokens and every finished request. A corpus is
one plan of requests served as several traces: light, medium and saturating loads, chunked prefill and a small KV cache
that makes the scheduler preempt, and also under a hostile scheduler, with the pool cut at each execution phase's edge,
through the network attester as a relay, and under the certified tick. Comparing the traces of one plan says whether
every trace gave each request the same output ids, and where a trace first diverged.

Where a trace runs through the network attester, the network attester is a process beside the server on the serving
host, not the auditor's device on a host the developer doesn't control.

## Try it

From the tree's root, as a job on a leased GPU of a node:

~~~sh
env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/check/slot.py --by LANE -- \
  bash benchmarks/network_e2e/traces/run.sh
~~~

It needs the node's pinned vLLM venv and the C4 shard `benchmarks/ppl/datasets.json` pins; `run.sh`'s header has its
variables and resources. It writes each corpus to `OUT/corpus/<model>-<variant>/`: per trace its `steps.jsonl` and
`requests.jsonl`, the corpus's `summary.json`, and a README that says what each field is (with `JOIN=0`, `join.sh` summarizes it later on a CPU node). `corpus.py`'s docstring has
every trace, variant and subcommand, and each script's header has its command. A summarized corpus is published in the
evidence store as `vllm-trace-corpus/v1`, which `benchmarks/network_e2e/encodings/trace_corpus.py` reads.

For the 8B model, attention is vLLM's own, not the int8 attention: `verity_vllm.engine.int8_ops._requests` refuses a
request whose query rows are neither its whole window nor one row. Every prefix-cache hit and every later prefill chunk is
such a request.
