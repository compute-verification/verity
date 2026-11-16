# vLLM scheduler traces under varied traffic

Real vLLM V1 scheduler traces for network attestation's advice: what serving made unpredictable, step by step, under
Poisson traffic, on plain vLLM's OpenAI server (`corpus.MODELS` lists the models, `corpus.TRACES` the loads and
`corpus.VARIANTS` the sampling). `vllm_trace_plugin.py` records the scheduler, `vllm_accept_stamp.py` each request's
arrival, and `traffic.py` is the request plan every trace serves. `corpus.HOSTILE`'s traces add a hostile scheduler
(`vllm_adversarial_plugin.py`), `corpus.CUT`'s cut the carried pool at execution phase edges, and
`corpus.NETWORK_CERTIFIER`'s serve through the network attester as a relay
(`verity/protocols/resource_accounting/network/attester/network_attester_serve.py serve`, a process beside the server,
not on the auditor device).

~~~bash
C=benchmarks/network_e2e/traces/corpus.py
python3 $C run --out DIR --model-id MODEL --c4 SHARD.jsonl.gz [--admit tick]   # each trace on a fresh server
python3 $C summarize DIR                     # steps.jsonl, requests.jsonl, summary.json and the corpus's README
python3 $C compare --out DIR CORPUS...       # output ids request by request across corpora of one plan
python3 $C tick TRACE_DIR                    # τ_sat and τ_free for run's --tick-from
~~~

`corpus.py run --help` has the rest of its arguments. A summarized corpus is published in the evidence store as
`vllm-trace-corpus/v1`, which `benchmarks/network_e2e/encodings/trace_corpus.py` reads.

For the 8B model, attention is vLLM's own, not the int8 attention: `verity_vllm.engine.int8_ops._requests` refuses a
request whose query rows are neither its whole window nor one row. Every prefix-cache hit and every later prefill chunk is
such a request.
