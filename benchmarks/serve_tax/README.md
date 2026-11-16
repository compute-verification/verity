# The serve tax

Verity changes how the serving engine runs: it records the scheduler's steps, admits requests on the certified tick,
and commits tokens and KV state while serving. The serve tax is what each such change costs on every token served,
measured against stock compiled vLLM at the adapter's pin (d9105ea80) on the same node, workload and SLO. Each arm
below is one engine configuration of vLLM's own OpenAI server (`vllm.entrypoints.openai.api_server`), its change
installed as a vLLM general plugin, and adds its change to the arm it names as its parent (`serve_tax.ARMS`). The
serving cost is one number with a target: how much more GPU time an arm needs than stock to serve the same tokens.

## Try it

~~~sh
uv run tools/check/suites.py benchmarks/serve_tax
~~~

It tests the plugins, the boundary commitment, the replay and the restore on the CPU in seconds; the arms themselves
need a GPU node (below).

## The arms

| arm | its change | parent |
|---|---|---|
| `compiled` | stock vLLM, its defaults: CUDA graphs, with torch.compile on the architectures vLLM compiles (not those it captures breakable CUDA graphs for, GLM-5.3's among them), chunked prefill, prefix caching, async scheduling | — |
| `compiled-again` | nothing: stock in a second server lifetime, for what two starts of one engine differ by | `compiled` |
| `no-prefix-cache` | `--no-enable-prefix-caching` | `compiled` |
| `published` | the server started as vLLM's recipe for the model starts it (`Model.published`): the recipe's flags, its speculative decoding among them, the job's context limit or else the recipe's, and vLLM's own limit on the requests it runs at once in place of the runner's 256 | `compiled` |
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
| `boundary-rows` | the two-stage audit's boundary commitment at serve, vLLM's prefix cache left on: each window's finished requests committed by size class off the engine loop (`verity_vllm.pipeline.boundary_rows`), under the arm's object (`serve_tax.ROWS`) and the model's check template (`Model.check`) | `record` |
| `canonical-topk` | the canonical DSA top-k: `benchmarks/dsa_topk`'s plugin loaded and `VERITY_DSA_TOPK=canonical`, its counters kept on the node | `compiled` |
| `two-stage` | the two-stage serve's launch line: `boundary-rows` with the canonical top-k and one attention layer's history tables in the boundary commitment (`serve_tax.TABLES`), built on rank 0 and kept in a private directory on the node (`servetax_plugin`), and prefix caching off, which the tables need | `boundary-rows` |
| `two-stage-tick` | the two-stage serve's launch line admitting on the certified tick: `two-stage`'s data with the tick on, served by the corpus's client | `two-stage` |
| `tick-certifier` | `tick` with the network attester itself in front, under vLLM's client (two commands, below) | `tick` |
| `prod` | nothing: `published`'s server, as the stock arm the arms made from it are costed over | `published` |
| `prod-again` | nothing: `prod` in a second server lifetime, the noise floor of a cost over `prod` | `prod` |
| `prod-record` | the scheduler's step trace on that server, as `record` is on `compiled`; each request's prompt ids and served ids, which a replay of a pass sent by vLLM's client is given and compared with; and the canonical DSA top-k, as `two-stage` has it (its counters kept on the node), which that replay runs under too | `prod` |
| `prod-seeded` | nothing on the server: `prod`, a sampled pass's client sending each user's seed as `prod-record`'s does | `prod` |
| `prod-trace` | the step trace and each request's ids on that server: `prod-record` with the canonical top-k off; its client sends seeds, as `prod-record`'s does | `prod` |
| `restore-base` | seeded sampling (the corpus's `seeded`), each sampled token's logprob returned | `record` |
| `restore-pause` | a replay block's boundary state taken at each edge: the engine loop drained, the KV pool copied to host and hashed, the scheduler's, block pool's, model runner's and generators' state captured, and kept, as a server would | `restore-base` |
| `restore` | `restore-pause` with every leaf of the engine core's state dumped at each edge and each step's GPU busy time timed, for the restore's check | `restore-base` |
| `restore-unseeded` | `restore` with requests that carry no seed, as clients send them in serving | `restore` |

`prod` is sent none, so vLLM samples its requests as it samples any without a seed, and what a seed changes in vLLM's
sampling is in `prod-record`'s cost over `prod`, a change of ours like the record and the top-k; `prod-seeded` is
`prod` sent the seeds, the one arm sent them that keeps no record (`Arm.seeds`), which reads that change apart: never
stock, which a cost is over.

## Running it

On a node with the pinned vLLM venv, under `research run`, the C4 shard `benchmarks/ppl/datasets.json` pins sent with
`--send` (`research data fetch` it first). `SERVE_VLLM=52aeb5806` serves on vLLM v0.34.0rc1 instead, in the environment
`adapters/vllm/verity_vllm/ops/vllm_env.sh` installs beside the pin's the first time:

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
and refuse a tree without them. `tick-certifier` is two jobs: the node's (`SIDE=server`, in `run.sh`'s header),
then the certifier's host's (`serve_tax.py run --side client`), which needs no GPU. A saturated load keeps a connection open for every request queued in the server, so the
client raises its open-file limit to the hard limit first.

On one GPU, `MODEL_ID` is `moonlight` (the default), `llama8b` or `qwen3-30b`, the last two read from their pinned
snapshots under `HF_HOME` (`serve_tax.py checkpoint --model-id ID` prints the path).

Under a deployment's traffic, the job serves the replay ladder's plan (`benchmarks/replay_ladder/prod_plan.py`: ShareGPT,
multi-turn and 8k-32k-token prompts, half of them sampled at temperature 0.7 under the public seed rule, each stopping at
its end of text or its max_tokens), sent as `plan.jsonl` or named by `PLAN_FILE` (then no C4 shard is needed), with FlashInfer on:

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
