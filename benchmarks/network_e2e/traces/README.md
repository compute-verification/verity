# vLLM scheduler traces under varied traffic

Real vLLM V1 scheduler traces for network accounting's advice: what serving made unpredictable, step by step, under
Poisson traffic. Plain vLLM's OpenAI server on one RTX PRO 6000, with the integration's batch-invariant eager engine
(bi-eager). `corpus.MODELS` lists the models: `llama32_1b` (unsloth/Llama-3.2-1B, bf16, G-A's model) and
`llama31_8b_int8` (`verity-int8/Llama-3.1-8B-INT8` with the int8_exact linears, `--quantization verity_int8`).

- `vllm_trace_plugin.py`: a vLLM general plugin. In the engine core it wraps the scheduler's `schedule`,
  `update_from_output` and `_free_request`, and appends each step's scheduled requests, the model runner's batch order
  and sampled tokens, and each finished request, as JSON lines. With `VERITY_INT8_PARTS` set it also registers the
  int8_exact linears in every process the server starts.
- `traffic.py`: the request plan, which every trace serves in the same order. C4 prompts of 16 to 1024 tokens, about half
  behind one of eight shared prefixes, with `max_tokens` 64 or 256 and temperature 0. The trace's rate scales the Poisson
  arrivals.
- `corpus.py`: runs each trace in `TRACES` on a fresh server: light, medium and batch-saturating loads with whole
  prefills, chunked prefill, and an 8192-token KV cache that makes the scheduler preempt, at each model's own rates. An
  async streaming client sends each request at its arrival. `corpus.py summarize` writes the joined `steps.jsonl` and
  `requests.jsonl`, `summary.json`, and the corpus's own README, which says what each field is.
- `run.sh`: the job on a leased GPU of vy-nebius-2 (its header has the command).

For the 8B model, attention is vLLM's own, not the int8 attention: `verity_vllm.engine.int8_ops._requests` refuses a
request whose query rows are neither its whole window nor one row. Every prefix-cache hit and every later prefill chunk is
such a request.
