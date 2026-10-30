# vLLM scheduler traces under varied traffic

Real vLLM V1 scheduler traces for network accounting's advice: what serving made unpredictable, step by step, under
Poisson traffic. Plain vLLM's OpenAI server on one RTX PRO 6000, with the integration's batch-invariant eager engine
(bi-eager). `corpus.MODELS` lists the models: `llama32_1b` (unsloth/Llama-3.2-1B, bf16) and
`llama31_8b_int8` (`verity-int8/Llama-3.1-8B-INT8` with the int8_exact linears, `--quantization verity_int8`).

- `vllm_trace_plugin.py`: a vLLM general plugin. In the engine core it wraps the scheduler's `add_request`, `schedule`,
  `update_from_output` and `_free_request`, and appends each request handed to the scheduler, each step's scheduled
  requests, the model runner's batch order and sampled tokens, and each finished request, as JSON lines. With
  `VERITY_INT8_PARTS` set it also registers the int8_exact linears in every process the server starts.
- `vllm_accept_stamp.py`: an ASGI middleware the server loads (`--middleware`). It records each request's TCP accept,
  headers and body in the API server, so that `summary.arrival_order` can count, at each place along the wire, the pairs
  of requests that arrived the other way round from the engine's order.
- `traffic.py`: the request plan, which every trace serves in the same order. C4 prompts of 16 to 1024 tokens, about half
  behind one of eight shared prefixes, with `max_tokens` 64 or 256. The trace's rate scales the Poisson arrivals.
- `corpus.py`: runs each trace in `TRACES` on a fresh server: light, medium and batch-saturating loads with whole
  prefills, chunked prefill, and an 8192-token KV cache that makes the scheduler preempt, at each model's own rates. An
  async streaming client sends each request at its arrival. A corpus's variant (`corpus.VARIANTS`) sets its sampling,
  greedy (temperature 0) or seeded (temperature 0.8, top_p 0.95, top_k 50, each request's `seed` derived from its id by
  `corpus.request_seed`), and whether async scheduling is on. `corpus.py summarize` writes the joined `steps.jsonl` and
  `requests.jsonl`, `summary.json`, and the corpus's own README, which says what each field is. The summary also has
  whether every trace gave each request the same output ids, and the rows that sample past a request's last output.
  `corpus.py compare` checks the same across corpora of one plan, with the bits per request a schedule steered (log2 of
  the distinct output-id sequences) and where a trace first diverged.
- `vllm_adversarial_plugin.py`: a hostile scheduler, another general plugin. From `VERITY_ADV_SEED` it makes adversarial
  but valid choices every step: random forced preemptions, a shuffled running list and waiting queue, random hold-backs
  of waiting requests, random prefill splits, and prefix-cache hits refused at random or cut to one KV page
  (`VERITY_ADV_CACHE`). It records its choices beside the tracer's files. `corpus.HOSTILE`'s traces serve heavy's load
  under the honest scheduler and under hostile ones, which tests whether a request's outputs depend on its batch-mates.
- `corpus.WARDEN`'s traces serve heavy's and small KV's loads through the network certifier as a relay
  (`benchmarks/network_traces/warden_serve.py serve`, a process beside the server, not on the auditor host). The client
  speaks cells through the relay's shim. `summarize` checks what the relay published, and writes each relay trace
  twice more for the scheduler replay (`benchmarks/network_e2e/encodings/replay.py`'s `read_corpus_trace`): with the
  relay's admission order and with its release order as the certified ingress (`summary.warden_views`). The relay,
  the client and the server run on cores of their own (`corpus.cpu_plan`), and three probes say where a late tick or a
  late cell read came from: `warden_probe.py` (the relay's every tick, publish and collection), `host_sample.py` (the
  host's load every 100 ms) and the client's own event loop record (`corpus.drive_warden`); the summary lays each late
  tick against them (`summary.late_ticks`).
- `run.sh`: the job on a leased GPU of a node (its header has its command and resources). `join.sh`: its summaries,
  comparison and the scheduler replay's table on a CPU node, for a GPU job run with `JOIN=0`. A summarized corpus is
  published in the evidence store as `vllm-trace-corpus/v1`, which `benchmarks/network_e2e/encodings/trace_corpus.py`
  reads.
- `admit_release.py`: what admitting requests to the engine in the relay's release order would cost (serve-openai
  adding each at the bucket boundary after its release), estimated offline. The scheduler model
  (`benchmarks/network_e2e/encodings/replay.py`) runs on the trace's own step clock, once with the recorded adds and
  once with the release-order adds; it reports the added TTFT, the throughput change and the order
  bits removed. It reads traces served through the relay and traces served without it.

For the 8B model, attention is vLLM's own, not the int8 attention: `verity_vllm.engine.int8_ops._requests` refuses a
request whose query rows are neither its whole window nor one row. Every prefix-cache hit and every later prefill chunk is
such a request.
