# The serve tax

What each change Verity makes to the serving engine costs on every token served, measured against stock compiled
vLLM at the integration's pin (d9105ea80) on the same node, workload and SLO. Each arm is one engine configuration of
vLLM's own OpenAI server (`vllm.entrypoints.openai.api_server`), its change installed as a vLLM general plugin, and adds
its change to the arm it names as its parent (`serve_tax.ARMS`):

| arm | its change | parent |
|---|---|---|
| `compiled` | stock vLLM: torch.compile and CUDA graphs, chunked prefill, prefix caching, async scheduling | — |
| `eager` | `--enforce-eager` | `compiled` |
| `bi-compiled` | `VLLM_BATCH_INVARIANT=1`: vLLM's batch-invariant kernels | `compiled` |
| `bi-eager` | `VLLM_BATCH_INVARIANT=1`, eager: the committing path's execution (`TargetProfile`'s `bi-eager`), nothing committed | `eager` |
| `record` | the scheduler's step trace (`verity_vllm.engine.step_trace.StepTrace`), written by a thread of its own | `compiled` |
| `release-order` | requests added in the certifier's release order (`serve_admit.ReleaseOrder`, in the engine core) | `record` |
| `tick` | admission on the certified tick (`verity_vllm.pipeline.tick_plugin`), τ calibrated on `record` | `record` |
| `tokens` | each window's tokens, 4 bytes each, committed in `rows-sha512` rows off the engine loop | `tick` |
| `kv` | the whole paged KV pool hashed in place on a side stream at each window edge | `tokens` |
| `tokens-only` | each Commit period's tokens committed as `tokens` commits them, without the tick | `record` |
| `boundary` | the boundary commitment at each replay block's edge, as a server would take it: drained, the KV pool hashed by the row-table kernels under a fresh key and copied to host memory, the scheduler's state and each rank's runner rows and generators taken and kept | `tokens-only` |
| `boundary-rows` | prefix caching off, and the two-stage audit's boundary commitment at serve: each window's finished requests committed by size class off the engine loop (`verity_vllm.pipeline.boundary_rows`), its check template and the template's instances a prompt row and an output in `BOUNDARY_ROWS` (JSON) | `record` |
| `restore-base` | seeded sampling (the corpus's `seeded`), each sampled token's logprob returned | `record` |
| `restore-pause` | a replay block's boundary state taken at each edge: the engine loop drained, the KV pool copied to host and hashed, the scheduler's, block pool's, model runner's and generators' state captured, and kept, as a server would | `restore-base` |
| `restore` | `restore-pause` with every leaf of the engine core's state dumped at each edge and each step's GPU busy time timed, for the restore's check | `restore-base` |
| `restore-unseeded` | `restore` with requests that carry no seed, as clients send them in serving | `restore` |

`replay.py` replays one load of a `record` arm in this process under its recorded schedule (each request added before
the step it arrived before), compiled or eager, and counts the step rows and tokens that differ from the served ones.
It starts from an empty engine. `restore.py` starts mid-traffic instead: it puts a restore arm's boundary state at one
edge into a fresh engine (the KV bytes into the same block ids, then the state, `boundary.restore`), replays to a later
edge, adding each request that arrived in between before the step it arrived before, and compares every step's
schedule, every sampled token and its logprob's bits, and at the far edge the KV pool's root, each block's fingerprint,
the block pool and every dumped leaf (`boundary.py`'s docstring says what an edge takes and how). `rebuild.py` starts from
less, only what a host copy of the K/V pages and the step record hold: it adds each request held at the edge to a fresh
engine as a new request with its outputs so far, copies its computed prefix's pages from the edge's KV file into blocks
the fresh engine allocates, as a KV connector's load does, and compares the replay the same way, with each request's K/V
over its computed tokens as a SHA-512 root at both edges (``--tail`` sets what a partial block holds past its prefix).
With `--at-step S --mirror-out DIR`, then `--from-mirror DIR`, it starts at an arbitrary step inside the replay block
instead, from per-step copies of the slots each step wrote (its docstring says how).

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

`run.sh`'s header lists its parameters (the model, reps, the loads' span, the Commit period and the KV edge's window).
`MODEL_ID=glm53` serves GLM-5.3 at TP8 on an 8-GPU node, with the GLM-5.3 TP drivers' checkpoint; `kv` there needs the
edge in each worker process, which the plugin's model runner hook gives where vLLM loads general plugins in its workers.
The `tick`, `tokens` and `kv` arms need `verity_vllm.pipeline.tick_plugin` and the corpus client's stand-in certifier,
and refuse a tree without them. A saturated load keeps a connection open for every request queued in the server, so the
client raises its open-file limit to the hard limit first.

On one GPU, `MODEL_ID` is `moonlight` (the default), `llama8b` or `qwen3-30b`, the last two read from their pinned
snapshots under `HF_HOME` (`serve_tax.py checkpoint --model-id ID` prints the path).

Under a deployment's traffic, the job serves the replay ladder's plan (`benchmarks/replay_ladder/prod_plan.py`: ShareGPT,
multi-turn and 8k-32k-token prompts, half of them sampled at temperature 0.7 under the public seed rule, each stopping at
its end of text or its cap), sent as `plan.jsonl` or named by `PLAN_FILE` (then no C4 shard is needed), with FlashInfer on:

~~~text
research run --on NODE --source . --cwd source --send plan.jsonl [--send fi_smoke.py] \
  --env MODEL_ID=glm53 --env 'ARMS=compiled record tokens-only boundary' --env RATES=light=R1,medium=R2,saturated=R3 \
  --env EDGES_EVERY_S=30 --env KV_COPY=first --env MAX_MODEL_LEN=40960 --env FLASHINFER=1 --env CUDA_HOME=DIR \
  --env BUDGET_MIN=M -- bash benchmarks/serve_tax/run.sh
~~~

At TP > 1 the boundary arm's engine core drains and takes its own state, and each rank takes its part where it lies
through `collective_rpc` (`boundary.rank_pool` once, `boundary.rank_take` at each edge): its KV pool hashed and, on the
ranks `KV_COPY` names, copied to pinned host memory, then its runner rows and generators, all kept in the rank's process
until the next edge. Every rank hashes under the edge's one key, so ranks that hold the same bytes give the same root, and
the edge's record says whether all did. Each load's `meta.json` has the GPUs' busy seconds over it (`gpu`: NVML's
utilization integrated), and `table.md` lays them out per thousand output tokens beside each arm's edges.

## What it writes

Under the job's `out/`: `plan.jsonl` (the requests), one directory per arm with its `meta.json` (the server's command,
the plugin's changes, τ, versions, each load's start), `server/` (the server's log and the plugin's records: the step
trace, each window's token roots, each KV edge's bytes and kernel time) and `<load>/rep-<r>/client.jsonl` (each request's
send, streamed chunks and tokens), then `table.md` and `table.json` (`serve_tax.py table`): per arm and load, output tokens
and requests a second, goodput at the SLO (TTFT and time per output token), TTFT and ITL at p50 and p99, each a mean over
reps with a 95% interval, the requests sent and not served, each load's schedule (the engine's steps, the tokens each
scheduled and the rows each sampled, from the server's `/metrics` before and after the load), and each arm's tax, its
compute per token over the reference's at the same load less one, against `compiled` and against its parent. Below capacity the throughput is set
by the arrivals, so the tax there reads the drain and the cost shows in the latencies; a tax beside unserved requests
isn't over the same requests. `replay-{compiled,eager}/replay.json` has the replay's counts, its differing requests and
its wall time beside the served window's, and `replay_tokens.jsonl.gz` its tokens. Each request decodes to its cap, so the
schedule doesn't depend on the tokens: a replay with no differing step can still sample other tokens. A mode listed twice
in `REPLAY_MODES` replays again and compares its tokens with the first replay's too, which tells a replay that differs from
serving from one that differs from itself; `REPLAY_FROM` with no `ARMS` replays an earlier job's window alone, reading
that job's run directory where it lies on the node. A replay runs at the model's GPUs as its tensor-parallel width, and
`REPLAY_CAPTURE=ID,...` takes those engine ids' logits rows and ids at every rank's sampler into the replay's `capture/`
(`replay.py --capture`), a token select's span each. `REPLAY_LOGPROBS=1` records each sampled token's logprob too
(`replay_logprobs.jsonl.gz`), and a repeated mode compares them with the first replay's bit for bit, naming each request's
first differing step: a one-ulp change anywhere in a logits row shows there even when no token flips. Every engine
`run.sh` starts, served or replayed, runs with `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`. Without it, each process times a
halved `R0_BLOCK` beside the static one for every looped reduction whose register use limits its occupancy (on
Moonlight, the residual add and RMSNorm fused into a decoder piece's output) and keeps the faster, which changes the reduction's order, so a
replay matches serving's values only when both picked the same; a window served without the pin isn't reproduced by a
replay with it. A finished run's large files (`plan.jsonl`, `client.jsonl`) are
evicted to its files artifact (`run-files.evicted.json` names it) after custody; `research data fetch ART --path GLOB
--to RUN_DIR` puts them back first.

The `kv` arm hashes the whole pool, free blocks too, at each edge: an upper bound on hashing the blocks live there, and a
measurement of the cost, not a commitment of the state (a block written after the edge before the side stream reads it
is hashed as written). With a KV window shorter than the Commit's, its tax scales down with the window's length.

A restore arm (`EDGES_S`, `SAVE_KV`, `RESTORE`, `NUM_GPU_BLOCKS` in `run.sh`'s header) takes its edges at fixed seconds
after its load's first request. Each edge first drains: the engine core schedules nothing until both batches in flight
are done and their outputs processed, so nothing is on the GPU and nothing is a placeholder; the pool is then consistent,
and its root commits it. `server/edges.<pid>.jsonl` has each edge's step, its pause split into the drain, the state's
capture, the dump, the KV copy and hash, its parts' sizes, the pool's root and the GPU's busy seconds since the previous
edge; `server/edges/` each edge's dump and block fingerprints and the served steps' outputs; `KV_DIR/<arm>/` the KV bytes
and state of the `SAVE_KV` edges, which `restore.py` reads (`run.sh` keeps `KV_DIR` only when it is given). The restore
engine runs in the job's process with the served header's configuration and stats setting, `PYTHONHASHSEED=0` and
`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`, as the served one did. `<arm>-replay/restore.json` has the comparison, the
restore's times and the replay's busy GPU-seconds beside serving's over the same steps. A few dumped leaves differ by
construction and are counted apart (`restore.by_construction`): the served engine core's process wrapper's own
attributes, and each request's `detokenize`, which the replay leaves off. A host without FlashInfer's sampler JIT runs
with `VLLM_USE_FLASHINFER_SAMPLER=0`, which leaves a seeded request's sampling as it was (it never takes FlashInfer's
path) but moves an unseeded one's to the seeded path's Gumbel noise under the seed vLLM took from numpy's generator
for it.
