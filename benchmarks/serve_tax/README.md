# The serve tax

What each change Verity makes to the serving engine costs on every token served, measured against stock compiled
vLLM at the integration's pin (d9105ea80) on the same node, workload and SLO, and the serving cost, one number with a
target (below). Each arm is one engine configuration of
vLLM's own OpenAI server (`vllm.entrypoints.openai.api_server`), its change installed as a vLLM general plugin, and adds
its change to the arm it names as its parent (`serve_tax.ARMS`):

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
| `prod` | nothing: `published`'s server, as the stock arm the arms made from it are costed over | `published` |
| `prod-again` | nothing: `prod` in a second server lifetime, the noise floor of a cost over `prod` | `prod` |
| `prod-record` | the scheduler's step trace on that server, as `record` is on `compiled`; each request's prompt ids and served ids, which a replay of a pass sent by vLLM's client is given and compared with; and the canonical DSA top-k, as `two-stage` has it (its counters kept on the node), which that replay runs under too | `prod` |
| `restore-base` | seeded sampling (the corpus's `seeded`), each sampled token's logprob returned | `record` |
| `restore-pause` | a replay block's boundary state taken at each edge: the engine loop drained, the KV pool copied to host and hashed, the scheduler's, block pool's, model runner's and generators' state captured, and kept, as a server would | `restore-base` |
| `restore` | `restore-pause` with every leaf of the engine core's state dumped at each edge and each step's GPU busy time timed, for the restore's check | `restore-base` |
| `restore-unseeded` | `restore` with requests that carry no seed, as clients send them in serving | `restore` |

An arm is data: its flags, its environment, the vLLM general plugins its server loads and its boundary commitment's
object are all in `serve_tax.ARMS`, and the runner has no code for any one arm. The two-stage serve's launch line is
stated there alone, as the `two-stage` arm: a job that serves the two-stage audit serves that arm
(`ARMS=two-stage RATES=... PLAN_FILE=...`), so the serve and its measured cost can't differ in an engine setting.
Before an arm's server starts, the runner loads each of its plugins once in the server's environment and refuses the
arm where one doesn't load, because vLLM logs such a plugin and serves on without it.

`published` is stock vLLM under the settings its authors publish for the model, where `compiled` is stock under the
settings every other arm shares. Its data is the command of the model's recipe (`recipes/models/` in vLLM's
repository, shown at recipes.vllm.ai) for the cards the runner serves the model on: the flags that command adds
(`Model.published`) and the context limit it is served at where the job names none (`Model.published_context`). The
arm adds the flags to `compiled`'s line and drops the runner's limit of 256 requests at once, which leaves vLLM's own
for the cards. A model with no recipe here is refused.

GLM-5.3's is the recipe's command "FP8 on 8xH200 (standard)": an FP8 KV cache, speculative decoding with five draft
tokens a step from the model's own draft layer (`mtp`), and the tool-call and reasoning parsers. The arm departs from
that command in two things, each with the recipe's own alternative:

- **The front end.** The command starts vLLM's Rust front end (`VLLM_USE_RUST_FRONTEND=1`) with the tokenizer mode
  `glm5_next`, which the integration's pin does not have. The arm takes what the recipe names for such a build: the
  Python front end and `--tokenizer-mode auto`. The release the recipe names as its minimum is also later than the
  pin. A job on a build that has the Rust front end asks for the command's own with `PUBLISHED_FRONTEND=rust`: the
  server is then `vllm serve` under that variable with `--tokenizer-mode glm5_next`, and the arm's `meta.json` names
  the front end it served behind (`front_end`). Every other arm serves behind the Python one, whatever the job says.
- **The context limit.** The command names none, so vLLM takes the model's own, 1,048,576 tokens, and by the recipe's
  own note refuses to start on eight H200s, whose KV cache does not hold one request of that length. The arm serves
  at 262,144 tokens, the limit the recipe's command builder sets for these cards. A job that wants another limit
  names it (`MAX_MODEL_LEN`), for every arm it serves, so that its arms differ by their flags alone.
  `MAX_MODEL_LEN=auto` is vLLM's setting for the longest limit the cards' KV cache holds, which the engine resolves
  when it starts and prints in its log. At the integration's pin the engine then builds the model at its own limit
  and fits the cache after, so the sparse attention's table of a step is sized by the model's limit and not by the
  one resolved. The runner checks no load against `auto`: a request past the limit the engine resolved is refused by
  the server and counted as failed.

The values a recipe leaves to the operator (the requests at once, the cards' share of memory, the tokens a step) are
vLLM's own defaults on the arm's line. A job that sets any of them adds the flags to that line alone with
`PUBLISHED_ARGS=--FLAG=VALUE,...`, each flag once; the arm's `meta.json` holds the line it served (`server_argv`).

The recipe's page asks for prefix caching off under a benchmark; the arm keeps it on, as a server has it, and each
pass starts from an empty cache as every arm's does. A job that ends on `published` prints no cost: the arm is stock,
and a cost is of an arm of ours over stock.

What an arm keeps that holds values never goes under its run's directory, which is published: a boundary commitment's
classes (each class's requests' tokens and its salts), the history tables' rows and salts, and the top-k's counters.
They go to `PRIVATE/<arm>` (`--private DIR`), a directory outside every run's, the classes under
`PRIVATE/<arm>/boundary/w<window>/p<P>-o<O>/`. A serve's later stages read them, so with the corpus client such an arm
is refused without `PRIVATE`; in the serving cost's form, where nothing reads them afterwards, the arm gets a fresh
directory without it, removed when it stops. The run's directory keeps each class's public `commit.json`, a line of
`<arm>/server/boundary.PID.jsonl`.

`replay.py` replays one load of a `record` arm in this process under its recorded schedule (each request added before
the step it arrived before), compiled or eager, and counts the step rows and tokens that differ from the served ones.
It takes each request's prompt ids from the plan and its served ids from the corpus client's `client.jsonl`. vLLM's
client keeps neither, so a pass it sent (`REPLAY=agentic@4`, the pass with its rung) is replayed from an arm whose
server kept them, `prod-record` (`REPLAY_FROM=OUT/prod-record`): a line a finished request in
`<arm>/server/ids.PID.jsonl`, its engine id, prompt ids and served ids.
An arm that served on the canonical DSA top-k (`canonical-topk`, `two-stage`, `prod-record`) is replayed on it: the
replay's processes load the plugin under the arm's environment (`replay.served_under`), and keep its counters under
`--private` (`PRIVATE/replay-<mode>`) where one is given.
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

## The serving cost

One number: how much more GPU time an arm needs than stock to serve the same tokens, in percent of stock's serving
compute (lower is better), with the loads sent by vLLM's own benchmark client (`vllm bench serve` at the pin) in place
of the corpus's. `verity bench serve_tax` is `run.sh`, here with the flags of its header's last form:

~~~text
research run --on NODE --source . --cwd source -- bash benchmarks/serve_tax/run.sh --model glm53
~~~

- **A load** (`serve_tax.LOADS`) is one of the client's datasets with its flags. The number has two, and a third,
  `prefix`, has the row under it (below). `chat` is the load
  of vLLM's standing serving tests: ShareGPT's first turns, 200 prompts, each decoding its recorded answer's length.
  `long` is long context: the client's synthetic dataset at one long length (8,192 tokens in, 128 out), which stands
  until a plan of recorded long requests replaces it; the output says which it was. Its first token takes seconds on
  any engine, so its goodput counts the time per output token alone, and the engines' context limit is raised to fit
  it. A pass sends the load's prompts whatever its rung, so a rung near or past them is held for part of the pass
  only: `PASS_PROMPTS=chat=N` makes a pass of `chat` send N prompts at every rung, for a ladder that climbs past
  them. Every rung sends the same N, since passes of two sizes differ in tokens a second by their size alone, and
  the cost refuses two arms whose passes sent different numbers of prompts. Under one N the pass's draining end, its
  last requests finishing a few at a time, is a larger share of the pass the higher the rung. `PASS_PROMPTS=chat=Kx`
  sizes each rung's pass by the rung instead, K prompts for each request it holds in flight and the load's own at
  least, which makes that end the same share at every rung: for a ladder read for the rung where tokens a second
  stop rising.
- **The dataset file** of `chat` is pinned (`serve_tax.Dataset`: its repository, revision, file and SHA-512). A node
  keeps one copy, in its Hugging Face cache outside every run's directory; `serve_tax.py passes --fetch`, which
  `run.sh` runs before it holds a GPU, downloads it there where there is none, and a download or a copy whose SHA-512
  is not the pinned one is refused. No run sends the file or keeps it: an arm's `meta.json` records its path and
  SHA-512. `serve_tax.py dataset chat [--fetch]` prints the checked path, for another benchmark that reads the file.
- **A rung** is a concurrency the client holds (`--request-rate inf --max-concurrency RUNG`): it sends the next request
  when one ends, as vLLM's standing tests do. A fixed arrival rate can't carry this number: below capacity the tokens a
  second only repeat the rate, and at capacity the queue grows for as long as the pass lasts. `--loads chat=12+16+24`
  names the rungs; a load's own are a first ladder, to find where stock is saturated.
- **A pass** is a load at a rung, `<load>@<rung>`. An arm serves each pass REPS times (six by default) in one server
  lifetime, each from an empty prefix cache, every request greedy and decoding to its length (`--ignore-eos`), the same
  prompts under one seed. `<load>@<rung>/rep-<r>/result.json` is the result the client saved, and the table's figures
  for a pass are the client's own. It is the client's detailed result, with each request's prompt and output lengths,
  start, time to first token and latency in the order sent, less each request's generated text and inter-token
  times, which no run keeps: the client saves it outside the run's directory and the runner keeps the rest.
  `STOP_AT=TIME` is the time the job's GPUs are another's: an arm starts no pass it can't expect to finish by then,
  going by the same pass's longest so far, so the job ends between two passes with every pass it served, and the
  arm's `meta.json` says where it stopped (`stopped`).
- **The cost** of an arm over `compiled` at a load (`serve_tax.serving_cost`) is the arm's tax where stock is
  saturated: stock's output tokens a second over the arm's, less one. Every arm serves the same prompts, each decoded
  to its length, so both sides serve the same tokens, and at a rung where stock's GPUs have no idle time the ratio is
  the arm's extra GPU time a token. Both sides are read at one rung: of the rungs both served, the one where stock's
  output tokens a second, averaged over its reps, are highest (`serve_tax.saturated`). The cost is the benchmark kit's
  paired ratio (`verity_bench.overhead.paired`): the median over reps of stock's tokens a second over the arm's in the
  same rep, with the kit's 95% interval, the order statistics of those ratios that bound their median. Six reps are
  the fewest that have one (their least and greatest ratio); with fewer the cost has no interval and its verdict
  cannot be told. No SLO enters the cost. Each load's line gives both sides' output and prompt tokens a second at the
  rung and the prompt and output tokens a request, and the third line the GPUs an engine serves on: what a reader
  needs to turn the tokens a second into a utilisation.
- **Whether stock is saturated at that rung** is said on each load's line and in the JSON (the load's `saturated`).
  Stock's own rungs show it only where it served a rung above and no more tokens a second at any
  (`serve_tax.peak`). Otherwise the line says it is not shown, and why: stock served no rung above, or served more
  at a rung the arm did not. No verdict reads it: a job that serves stock at the one rung an earlier job found does
  not show it saturated by its own passes.
- **The cost within the SLO** (`serve_tax.within_slo`) is printed under each load's cost: stock's best goodput over
  the arm's, less one, by the same paired ratio. Goodput is the client's "Request goodput": requests a second within
  the model's SLO (TTFT and time per output token). Each side's best is at the rung where its mean over reps is
  largest, so each arm is read at its own operating point. It is the cost only where both sides meet the SLO for
  every request. No target or verdict is of it: the model's target is of GPU time at equal tokens, and the SLO
  (`Model.slo`) is the benchmark's own constant. An arm that meets the SLO for no request at any rung has no finite
  cost within it.
- **The Commit period** of an arm with a boundary commitment is 30 s in this form (`WINDOW_S`), shorter than a pass,
  so that a pass holds the commits of the requests it finishes. At a period longer than a pass the commits would fall
  between passes, where no request waits on them, and the cost would leave them out. A short period counts a period's
  fixed work more often than a long one does, so the cost errs high; the commitment's line names the period.
- **The number** is the larger of the two loads' costs, each load's cost printed beside it. Where only one load has
  run, the first line names it (`load chat alone`) and a later line says the number waits for the other: one load's
  cost is not the number. Its loads repeat no prefix, so it is labelled `no repeated prefix` and is never given
  without the row on a repeated prefix under it: where load `prefix` has not run, the first line says so in place of
  the number and the JSON's `value` is null, while each load's own cost is still printed.
- **The row on a repeated prefix** (`serve_tax.prefix_row`) is directly under the number's line and directly after
  the JSON's `run` (`prefix`): what the arm costs over stock on load `prefix`, which the number's loads say nothing
  of, since stock serves a repeated prefix from its cache and an arm with the cache off computes it again. It has no
  target and no verdict. The load is four planned requests (`serve_tax.PAIRS`): two prompts near the engines' context
  limit of 131,072 tokens, each sent again with another ending, so half of the prompt tokens repeat, all of them on
  the second and fourth request. The client's `timed_trace` dataset sends them from a file of lengths and block ids
  that the arm writes (`<arm>/prefix.trace.jsonl`; a block is 64 tokens, and two prompts share the tokens of the
  blocks whose ids they share), each prompt as the synthetic token ids the client generates for its blocks, in the
  file's order with one request in flight, so each repeat follows the request it repeats (`--loads` takes no rungs
  for it). The row is labelled `scaled down` while the load is those four requests alone: half of their prompt
  tokens repeat, which is this load's share and no serve's, where such pairs arrive among requests that repeat
  nothing.
  `--dataset prefix=FILE` serves a table of planned requests in place of those four (`serve_tax.read_plan`): a line
  a request, with its arrival time in seconds after the pass starts (`at_s`) and its prompt's and its output's
  lengths (`prompt`, `output`), and, where its prompt begins as an earlier line's does, that line counted from 0 and
  the blocks the two share (`after`, `blocks`); nothing else, so it holds no token and no text. The arm writes the
  client's file from it, and its `meta.json` records what the table plans and its SHA-256. Each request is then sent
  at its arrival time, however many are in flight (the pass is `prefix@0`), and the row says `as they arrive` in
  place of `one request in flight`. It says for each arm whether it served all that arrived, and whether stock kept
  up: stock kept up where it served all that arrived and each of its passes ended within a tenth of the time the
  requests arrive over after the last one arrived. Where stock kept up, tokens a second follows the arrivals and is
  not a cost: the row's cost is the two latencies. Where stock did not keep up the row is read as the number is, by
  tokens a second. It prints tokens a second and both latencies either way, `cost` names which of them is the cost
  and `why` says why. Each figure of such a row is its readings, one a rep, never a mean with an interval, so its
  `value` is null.
  `prefix_plan.jsonl` is the full-size table, the run of record's: the 651 requests that a serve plan of 25 minutes
  of mixed traffic at 0.4 requests a second sends within its span, of which two long ones near the context limit
  each repeat an earlier one's prompt; `prefix_plan.py` makes it from that plan's two files, which
  `prefix_plan.source.json` names.
  With one request in flight the cost is stock's output tokens a second over the arm's, less one, by the same paired
  ratio; beside it the row gives how much longer the arm's time to the first token and its 99th-percentile gap
  between tokens are, the share of the prompt tokens each side's prefix cache served by the engine's own counters
  beside the share planned to repeat, whether each side served every request, and the same for `no-prefix-cache`
  over stock where that arm served the load: what the cache alone is worth there. The load is served in a job of
  its own (`--loads prefix`), since one engine serves a job's loads and this one's context limit would be the others'
  too; that job's cost is the row alone, and `WITH=` joins it to the number's job.
- **The verdict** of a load against the model's target (`Model.target`, or `--target PERCENT`) is `meets` where the
  whole interval is at or under the target, `misses` where all of it is over, else `cannot tell`. The output says
  what the target rests on (`Model.target_basis`: `modelled` for a cost model's share, `given` for `--target`).
  `compiled-again` gives the noise floor, which reps inside one server lifetime don't see, and each load's verdict
  widens its interval by that load's floor at each end. The number's verdict follows from the loads':
  it misses where one load misses, meets only where both have run and meet, and otherwise cannot be told.
- **The target's parts.** A model's target is the sum of its parts (`Model.target_parts`), each the cost model's
  points for some of our changes (`serve_tax.CHANGES`, each read from an arm's data). GLM-5.3's are the certified
  tick, the tie order of the canonical top-k, and the snapshot bundle (the pause to copy and hash the cache at each
  edge and the carried pages' hash, with vLLM's prefix cache on). A cost is compared with the target only where its
  arm has every part's changes on and no change on that no part counts (`serve_tax.not_in_target`). Otherwise every
  verdict is `not comparable`, a fourth value, and the output names the changes that are off with their parts and
  the changes the target has no part for: the cost is not of what the target is of, so it neither meets nor misses
  it, and no target is made of the parts that are on. The `two-stage` arm leaves off admission on the certified
  tick and the edge snapshot, which need the corpus's client, and turns on four changes GLM-5.3's target has no
  part for (the step record, prefix caching off, the two-stage boundary commitment, the history tables), so its
  cost against the model's target is not comparable. `two-stage-tick` is that line with the tick on, and no cost
  here is of it: vLLM's own client can't drive a tick arm. A target given with `--target` has no parts.
- **Stock has vLLM's prefix cache on,** its default at the pin for a model that generates text, and so does
  `boundary-rows`: a request's boundary is its own prompt ids and the ids it was served, whichever KV pages its
  prefill read. `two-stage` turns it off: a history table holds its request's latent rows from position 0, and a
  request served from the cache computes none before its hit's end, so the boundary commitment refuses tables on a
  scheduler that caches prefixes. Neither load repeats a prefix
  (chat's requests are different conversations' first turns, long context's are random tokens, and the cache is
  emptied before every pass), so the number does not hold what the cache is worth on a load that does: the row on
  a repeated prefix does. `no-prefix-cache` is stock with the cache off alone, and each rep's `meta.json` records
  the prompt tokens the cache served (`schedule.prompt_tokens_cached`).
- **The arms** are by default `compiled`, `compiled-again` and `two-stage`, and the cost is of the last one named.
  `--arms all` adds one arm for each change in between (`record`, `no-prefix-cache`, `boundary-rows`) and the canonical
  top-k's own (`canonical-topk`). Each arm's cost is printed over stock and over the nearest arm above it that ran,
  at the number's load and rung, with its share of the whole for the arms from the measured one up to stock, and
  with what its step turns on: beside each such change the target's part for it, with no verdict, or `not in the
  model` for a change the target has no part for (the step record, prefix caching off, the two-stage boundary
  commitment, the history tables). A
  job of `compiled` alone prints no cost: it finds the rungs where stock is saturated and where its goodput peaks,
  which a later job's `--loads` names.
- **The boundary commitment's line** says what the measured arm's commitment did, from its own records: the finished
  requests committed, those whose commit failed, and those committed with their history tables. An arm that asks for
  tables on a tree whose boundary commitment builds none commits without them, and the line says the cost is without
  them.
- **What it leaves out.** `serve_tax.NO_ARM` lists what a served engine of ours changes that no arm turns on; the output
  prints it under the number, which counts none of it. The arms that need the corpus's client (the tick's, fed by the
  stand-in certifier, and the restore arms' seeded sampling) aren't served by vLLM's.

It writes `serving_cost.txt`, whose first line is the number with its label, interval, target and verdict and whose
next lines are the row on a repeated prefix, and `serving_cost.json`, one object whose first keys are `value`, `unit`,
`commit` and `run`, then `prefix` (the row, or null) and `label` (`not_run` names the number's loads that have not
run). `--json` prints the object last, and `serve_tax.py cost ARM_DIR... --json` prints it alone from a
finished job's arm directories, which may be of more than one job: a load served by a later job joins the number
(`WITH="OUT_DIR ..."` gives `run.sh` the earlier jobs').

The run's `result.json` (`serve_tax.result_of`, written by the kit's writer) holds each load's cost as a measurement
named by the load, and the number as one named by the model's metric id (`Model.metric`), with `ci95`, `target`,
`target_basis`, `target_parts` (each part with the changes the arm left off), `verdict` (one of the three, or `not
comparable`), `better`, `load`, and `loads`, each load's cost beside it, with its `label` and the row (`prefix`). The
row is also a measurement of its own, named `prefix`, with no target and no verdict; its `value` is null where its
requests are sent as they arrive, and its figures' readings are beside it (`cost`, `kept_up`, `why`, `tokens`,
`ttft`, `itl_p99`). A job in which a load of the
number or load `prefix` has not run writes no entry under the metric's id. `gpus.csv` names the cards the job measured on.

## The recipe's line and the agentic load

`prod` is `published`'s server, stock vLLM as its recipe for GLM-5.3 on eight H200s starts it (above), under the
name of the stock arm two arms are made from: `prod-again` is its second lifetime, and `prod-record` adds the
scheduler's step trace and keeps each request's prompt ids and served ids (`servetax_plugin.ServedIds`,
`server/ids.PID.jsonl`), since vLLM's client keeps neither and a replay of a pass is given the first and compared with
the second (`replay.window`). It serves on the canonical DSA top-k too (`serve_tax.TOPK`, `TOPK_ENV`): vLLM's own
kernels order a row of more than 2,048 keys differently run to run, and the agentic load's prompts are longer, so a
replay is to give the served tokens only where both select by the canonical rule. Its cost over `prod` holds the
top-k's with the record's. An arm is costed over the nearest stock arm above it (`serve_tax.STOCK`, `stock_of`),
so `serve_tax.py cost --arm prod-record` is over `prod`, stock at the same settings, with `prod-again`'s cost over
`prod` as its noise floor, printed beside it. The three start from one environment, the job's (`VENV`), and `cost` refuses arms that
different builds of vLLM served. Where `prod`'s server and its requests differ from the recipe's line, beside
`published`'s two departures (above), and why:

| the recipe's line | `prod` | why |
|---|---|---|
| no seed, uvicorn's access log on | `--seed 0`, `--disable-uvicorn-access-log` | every arm's (`serve_tax.BASE_ARGS`): the same engine seed on both sides, and no line a request in the server's log |
| requests as an agent's client sends them: the chat endpoint, tools, sampled as the model's own configuration says, no seed | `/v1/completions` with token ids, each request decoding to its recorded length; greedy, or under `SAMPLED=1` sampled at temperature 1.0 and top-p 0.95, each request naming its `user` and carrying no seed | every arm then serves the same number of tokens whatever it generates; the parsers are loaded and parse nothing |

Under `SAMPLED=1` the client sends `prod-record`, as it sends every arm that keeps the record, each user's seed with
the request (`serve_tax.user_seed`, `sends_seed`): its replay samples the request again under that seed. `prod` is
sent none, so vLLM samples its requests as it samples any without a seed, and what a seed changes in vLLM's sampling
is in `prod-record`'s cost over `prod`, a change of ours like the record and the top-k. The seed is the client's; a
served request need not send one. It is made from the user's name and, under `LIFETIMES`, the lifetime's number: a
pass's users have the same names in every lifetime, and a server samples much the same tokens again under one seed,
so with the name alone every lifetime of `prod-record` would serve one sample of text where `prod`'s serve a new one
each. The share of its drafts' tokens a text accepts sets how many steps it takes to serve, so that one sample's
chance would be in the cost and in no interval over the lifetimes.

The loads `agentic`, `agentic-full` and `agentic-256k` are a coding agent's recorded sessions: SemiAnalysis's AgentX
traces (`serve_tax.AGENTX`, Apache-2.0), a line a session, each request its prompt's and its output's tokens and an id
for each 64-token block of its prompt, with no token and no text. vLLM's client makes a block's tokens from its id
(its `timed_trace` dataset), so a turn's prompt begins with the tokens of the turn before it and the engine's prefix
cache serves them. The tokens are synthetic, so the share of its drafts the engine accepts is a synthetic prompt's;
`cost` prints it. A session is its own chain of requests, without the groups its subagents sent, up to the first
request that does not fit the load's context with its output: 131,072 tokens for `agentic`, the model's own limit for
`agentic-full` (`serve_tax.session_lines`). `agentic-256k` reads the publisher's own cut of the same sessions for a
context of 262,144 tokens (`serve_tax.AGENTX_256K`), the recipe's limit on these cards, in which no request passes
that context, so no session of it is cut by ours. The load is closed-loop. Its rung is its users (`--loads
agentic=4`), each a client of its own with one request in flight that sends its sessions' turns in order, each once
the turn before has ended, so no turn arrives before the prompt it repeats is computed; `PASS_PROMPTS=agentic=N` is
the turns each user sends in a pass. A pass's rate is its tokens over its seconds, from its users' first send to the
last one's last answer (`serve_tax.merged`): every arm serves the same requests and as many output tokens, so the
cost is ours's seconds for them over stock's, less one. The users end at different times, so the sum of each user's
own rate is another figure, which counts a user that ended early as if it served the whole pass; it, the users'
seconds summed and the GPUs' busy seconds are printed beside the cost as other readings of the same passes, in no
verdict. Its users' sessions must fit the engine's KV pool together, or the prefix cache evicts what a later turn
repeats: `agentic-256k` sends 12 turns a user where the others send 24, since four users' sessions of it at 24 turns pass the KV cache the
recipe's server has on eight of these cards. Sessions loads share a job's servers (`--loads agentic=4,agentic-256k`),
which no other load joins, and the job has a cost of each: `cost` prints the first's and then the others', each
written beside the first's as `serving_cost.<load>.{json,txt}`.

`cost` on such a load prints `serve_tax.session_cost`: the arm's cost over its stock arm, whose unit is a server
lifetime. A lifetime's cost is the arm's seconds for the passes both arms counted in it, summed, over stock's, less
one; the cost is the mean of the lifetimes' costs, and its 95% interval is Student's t over the lifetimes an arm. The verdict against `--target` or
`serve_tax.SESSION_TARGET` reads that interval's upper end alone: "meets" where it is under the target. One lifetime
an arm has no such interval, and its verdict is "cannot tell". Beside the cost and in no verdict it prints the cost
over the passes taken as independent (the median over them with its interval, too narrow where the lifetimes' costs
spread more than a lifetime's passes do), the spread between lifetimes and between a lifetime's passes, and the noise
floor, stock over stock a lifetime apart, saying so where the floor's own interval leaves out zero. It says the share
of prompt tokens the load plans to repeat beside the share each side's prefix cache served, and gives no verdict where
a side left a request unserved, the sides served different counts of output tokens in a rep or the two caches served
different shares in it: the sides then did not do the same work. The same work is the same requests and as many
tokens, not the same token ids: two passes of the same prompts can serve other tokens, sampled or greedy. The count of
output tokens is the same because every request is decoded to its recorded length whatever it generates, past any
end-of-text token, as vLLM's own client measures; the cost's line on the load says so, and where an arm's server kept
the served ids (`ids` in its plugin's configuration, which no stock arm has) the next line counts that arm's requests
that served one of the checkpoint's end-of-text ids before their end and went on, with the share of its output tokens
after one: text an agent's client would not have asked for. Beside each
side's output tokens a second it prints its prompt and output tokens a second a GPU, the output tokens a second a
user gets at the P90 time a token, the engine steps a pass took, and the share of its draft tokens the model accepted
with the tokens a draft gave (the engine's own counters), so a reader sees whether a cost came with more steps or
fewer accepted drafts; no figure is adjusted for either.

~~~text
bash benchmarks/serve_tax/run.sh --model glm53 --arms "prod prod-record prod-again" --loads agentic=4 --reps 6
~~~

A node drifts over a job's hours, and the passes of one server lifetime share whatever that server had of its own.
`LIFETIMES=L` serves the arms `L` times, each time a server of its own, in the arms' order and then that order turned
round, and so on, so that the two arms' lifetimes of one number are neighbours in time and the order's effect falls on
both alike; `TURNED=1` begins with the order turned round, so the last arm named, ours, is served first. `LEAVE_OUT=N`
leaves each lifetime's first `N` passes out of every figure: a lifetime's first pass after its warm-up is still
warming (compiled graphs, the prefix cache's first fill), so it is served and recorded like the rest and not counted,
and the table and the cost say how many were left out. `WARM_S=0` sends no warm-up before a lifetime's passes: a pass
that is left out warms the server as the warm-up does, which then only adds to the lifetime's time. The cost then pairs pass `r` of lifetime
`k` of its two arms (`serve_tax.lifetimes`), prints each lifetime's own cost under the cost over all of them, and
takes its noise floor from stock's own lifetimes, each second one over the one before it, where no twin of stock was
served. Two lifetimes an arm are the fewest that give the cost an interval, and so a verdict.

~~~text
LIFETIMES=6 LEAVE_OUT=1 WARM_S=0 bash benchmarks/serve_tax/run.sh --model glm53 --arms "prod prod-record" --loads agentic=4 --reps 6
~~~

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
the plugin's changes, τ, versions, each load's start, and under `settings` the integration's value settings in force in
the environment its server is started with, `verity_vllm.settings.record`: each variable set there that can change what
an engine computes, a credential by name only, and their digest), `server/` (the server's log and the plugin's records: the step
trace, each window's token roots, each KV edge's bytes and kernel time) and `<load>/rep-<r>/client.jsonl` (each request's
send, streamed chunks and tokens), then `table.md` and `table.json` (`serve_tax.py table`): per arm and load, output tokens
and requests a second, goodput at the SLO (TTFT and time per output token), TTFT and ITL at p50 and p99, each a mean over
reps with a 95% interval, the requests sent and not served, each load's schedule (the engine's steps, the tokens each
scheduled and the rows each sampled, from the server's `/metrics` before and after the load), and each arm's tax, its
compute per token over the reference's at the same load less one, against `compiled` and against its parent. Below capacity the throughput is set
by the arrivals, so the tax there reads the drain and the cost shows in the latencies; a tax beside unserved requests
isn't over the same requests. `replay-{compiled,eager}/replay.json` has the replay's counts, its differing requests,
its wall time beside the served window's and the same record of its engine's environment (`settings`), and
`replay_tokens.jsonl.gz` its tokens. Each request decodes to its cap, so the
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
