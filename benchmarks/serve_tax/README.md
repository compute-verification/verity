# The serve tax

What each change Verity makes to the production server costs on every token served, measured against stock compiled
vLLM at the integration's pin (d9105ea80) on the same node, workload and SLO. Each arm is one engine configuration of
vLLM's own OpenAI server (`vllm.entrypoints.openai.api_server`), its change installed as a vLLM general plugin, and adds
its change to the arm it names as its parent (`serve_tax.ARMS`):

| arm | its change | parent |
|---|---|---|
| `compiled` | stock vLLM: torch.compile and CUDA graphs, chunked prefill, prefix caching, async scheduling | — |
| `eager` | `--enforce-eager` | `compiled` |
| `record` | the scheduler's step trace (`verity_vllm.engine.step_trace.StepTrace`), written by a thread of its own | `compiled` |
| `release-order` | requests added in the certifier's release order (`serve_admit.ReleaseOrder`, in the engine core) | `record` |
| `tick` | admission on the certified tick (`verity_vllm.pipeline.tick_plugin`), τ calibrated on `record` | `record` |
| `tokens` | each window's tokens, 4 bytes each, committed in `rows-sha512` rows off the engine loop | `tick` |
| `kv` | the whole paged KV pool hashed in place on a side stream at each window edge | `tokens` |

`replay.py` replays one load of a `record` arm in this process under its recorded schedule (each request added before
the step it arrived before), compiled or eager, and counts the step rows and tokens that differ from the served ones.

The workload is the trace corpus's (`benchmarks/network_e2e/traces`: C4 prompts of 16 to 1,024 tokens, about 60% behind
one of eight shared prefixes, caps of 64 or 256 tokens) under Poisson arrivals, every request decoding to its cap
(`ignore_eos`) so that every arm serves the same tokens, sent by the corpus's streaming client. A job serves each arm in
one server lifetime: a warm-up, then each load REPS times, rep by rep, each from an empty prefix cache.

## Running it

On a node with the pinned vLLM venv, under `research run`, the C4 shard `benchmarks/ppl/datasets.json` pins sent with
`--send` (`research data fetch` it first):

~~~text
# stock compiled's capacity under one burst, from which the loads are set
research run --on NODE --source . --cwd source --send C4.json.gz --env CAPACITY=1 -- bash benchmarks/serve_tax/run.sh
# the arms at three loads, then the replay of the record arm's saturated load
research run --on NODE --source . --cwd source --send C4.json.gz \
  --env 'ARMS=compiled eager record release-order tick tokens kv' --env RATES=light=R1,medium=R2,saturated=R3 \
  --env REPLAY=saturated -- bash benchmarks/serve_tax/run.sh
~~~

`run.sh`'s header lists its parameters (the model, reps, the loads' span, the Commit window and the KV edge's window).
`MODEL_ID=glm53` serves GLM-5.3 at TP8 on an 8-GPU node, with the GLM-5.3 TP drivers' checkpoint; `kv` there needs the
edge in each worker process, which the plugin's model runner hook gives where vLLM loads general plugins in its workers.
The `tick`, `tokens` and `kv` arms need `verity_vllm.pipeline.tick_plugin` and the corpus client's stand-in certifier,
and refuse a tree without them. A saturated load keeps a connection open for every request queued in the server, so the
client raises its open-file limit to the hard limit first.

## What it writes

Under the job's `out/`: `plan.jsonl` (the requests), one directory per arm with its `meta.json` (the server's command,
the plugin's changes, τ, versions, each load's start), `server/` (the server's log and the plugin's records: the step
trace, each window's token roots, each KV edge's bytes and kernel time) and `<load>/rep-<r>/client.jsonl` (each request's
send, streamed chunks and tokens), then `table.md` and `table.json` (`serve_tax.py table`): per arm and load, output tokens
and requests a second, goodput at the SLO (TTFT and time per output token), TTFT and ITL at p50 and p99, each a mean over
reps with a 95% interval, the requests sent and not served, and each arm's tax, its compute per token over the
reference's at the same load less one, against `compiled` and against its parent. Below capacity the throughput is set
by the arrivals, so the tax there reads the drain and the cost shows in the latencies; a tax beside unserved requests
isn't over the same requests. `replay-{compiled,eager}/replay.json` has the replay's counts and wall time
beside the served window's.

The `kv` arm hashes the whole pool, free blocks too, at each edge: an upper bound on hashing the blocks live there, and a
measurement of the cost, not a commitment of the state (a block written after the edge before the side stream reads it
is hashed as written). With a KV window shorter than the Commit's, its tax scales down with the window's length.
