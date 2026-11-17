# vLLM at its serving settings, replayed

Does Verity's serving still reproduce when vLLM runs the way people deploy it? This benchmark serves a model at the
settings vLLM publishes for it (prefix caching on, speculative decoding on, the context limit its recipe sets for the
cards), once stock and once with our changes, then replays what ours served under its recorded schedule. It prints
what the serve turned on and three lines:

- **tokens**: whether the replay gave back every served token;
- **cost**: what our changes add to stock's serving time, as a share of serving time on one node at one load, scaled
  to nothing;
- **advice**: the bits a request that the served schedule costs as advice.

It measures nothing of its own: the serve, the load, the cost and the replay are the serving cost benchmark's
([`serve_tax/`](../serve_tax/README.md)), and this directory runs them and reads what they leave. GLM-5.3 is the one
model whose published settings it holds.

## Run it

It runs from the tree's root as the command of a `research run`, on a GPU node that the serving cost benchmark runs
on. This is the command its metrics are measured by, GLM-5.3 under a coding agent's load:

~~~text
research run --on NODE --source . --cwd source -- bash benchmarks/prod_repro/run.sh --client vllm-bench --model glm53 --again none --load agentic=4 --pass-prompts agentic=4 --lifetimes 8 --leave-out 1 --reps 2 --replay agentic@4 --sampled
~~~

The report goes to the job's `out/prod_repro.txt` and `out/prod_repro.json`. The command has four forms: serve and
replay by the corpus client, serve by vLLM's own client, read an earlier job's `out/` again with no GPU, and replay
an earlier job's pass alone. A job of the corpus client also needs the C4 shard sent with it.

~~~text
verity bench prod_repro [--model moonlight] [--stock ARM] [--again ARM|none] [--ours ARM] [--load burst] [--rate 1000]
                        [--requests 800 | --span-s S] [--reps 3] [--replay PASS|none] [--max-model-len N] [--target 1]
verity bench prod_repro --client vllm-bench --load NAME[=RUNG+RUNG...] [--reps 6] [--replay NAME@RUNG]
                        [--pass-prompts NAME=N] [--sampled] [--stop-at TIME] [--lifetimes L] [--leave-out N]
                        [--arms "ARM ARM ..."] ...
verity bench prod_repro --report OUT --load NAME [--replay PASS] [--stock ARM] [--again ARM|none] [--ours ARM]
                        [--arms "ARM ARM ..."]
verity bench prod_repro --replay-of OUT --load NAME --replay PASS [--model ...] [--ours ARM]
~~~

`run.sh`'s header says what each option does.

On Llama 3.2 1B and one H200, the job that writes its cost, its replay match and its advice, as points of the same
three metric ids, older than their other models (`record-ids` stands in for `record`: it keeps each request's ids,
which a replay under vLLM's client reads):

~~~text
research run --on NODE --source . --cwd source -- bash benchmarks/prod_repro/run.sh --client vllm-bench --model llama32-1b \
  --again none --load agentic=4 --pass-prompts agentic=4 --lifetimes 4 --leave-out 1 --reps 2 --replay agentic@4
~~~

The cost of `prod-all-on` over `prod` (`--ours prod-all-on`: the serving cost benchmark's README, "Everything on at the
recipe's line") is a point of the cost's id too, and its `parts` say in words what it has on and what it leaves off.
Each point names its `model` (the checkpoint, `repo@revision`), its `hardware` (the cards it was served on) and its
`member`, the model, the card and the arm (the cost's over its stock), which tells its line from the id's other models'
and arms'. A serve with no draft, such as the 1B's, keeps no draft, so its kept count is 0 bits and its advice is the
arrivals'.

**The method a serving cost is measured by.** Four arms, `prod`, `prod-again`, `prod-seeded` and `prod-record`, each
served in nine server lifetimes in that alternating order, each lifetime opening with a 20-second warm-up pass that
no figure counts (the benchmark's `WARM_S`) and then two counted passes (`--reps 2 --lifetimes 9 --leave-out 0`).
The cost is the mean of the nine lifetime pairs' costs with its Student's t 95% interval over them, t at the pairs
less one, and stock against stock, `prod-again` over `prod` taken the same way, is the noise floor printed beside it
and never subtracted.

## The cost priced from counts

`priced.py OUT --units TOPK_CUT.jsonl` reads a finished job's `out/` with no GPU. It prices the canonical top-k's layer
(each arm of ours over `prod-trace`, `--reference`) from what the job kept, against the top-k's device time a call on
one card (`TOPK_CUT.jsonl`, the timing of `benchmarks/dsa_topk`'s passes at the served decode shapes):

- **The rows each call reads** come from the schedule: each counted pass's decode steps in the arm's step trace, with
  the rows past the indexer's k keys and the longest of them. A decode step makes 22 calls on each card.
- **The rows whose fast selection failed** come from the model's outputs. Where a lifetime kept the top-k's counts by
  step (`dsa_topk_steps.*.jsonl` beside its step trace), each counted step's failed rows are its own, on its slowest
  card, joined to the trace by step and tokens. Else they are the counters' lifetime totals (`topk_counters.json`): a
  card's share, scaled to the counted passes' share of the lifetime's decode rows, and a lower bound by the rows a
  killed process read last. The reader places a step's failed rows in its calls three ways: packed, a request's rows
  to a call, and one row a call. Where the counts by step hold the calls a remedy failed in (`calls_bisected`,
  `calls_refined`), it reads that number in place of the placements. The bounds and the point are those placements,
  with each pass's time ranged over the distributions it was timed on.
- **Prefill steps** (a step with a row of more than a decode request's tokens) are priced only at prefill unit times,
  `--prefill-units FILE` in `TOPK_CUT.jsonl`'s form, with their own failed rows where the counts are by step. Without
  that file they are refused, and the reader gives their measured added seconds instead: ours's prefill steps less the
  reference's, from both arms' step traces, as a term marked measured.
- **What it gives:** the layer's cost in percent of the reference's counted serving time, beside the job's own measured
  layer (`prod_repro.json`'s `layers`) and the difference in points. Beside the measured layer is the same layer with
  each pass's seconds adjusted for its engine steps (each pass paired with the reference's of the same lifetime and
  rep, at the slope of ln seconds on ln steps within each arm), marked modelled. The point (`price`, which the serving
  cost's door calls too) is that layer, plus the measured prefill term where prefill was not priced, plus the job's
  measured `prod-trace` over `prod`, under `benchmarks/serving-cost-priced` with the member model, card, then arm
  (`glm-5.3-h200-prod-record`), `mark` `modeled`. It names the job's run as the run that tests it, and each measured term in its
  inputs. The reader writes it to `priced.json` alone; `verity bench serving_cost price` stores it, and nothing puts a
  priced point on the measured line `vllm/prod-serving-cost-glm53`.
- **When it writes no point:** where the counters hold no failed rows, or where the job measured no layer below.
- **What it doesn't price:** the host's time, and a failed row's own keys (each failed call is priced at its step's
  longest row).

## What is not yet true

- **Two rules of the cost's method are not yet the report's own.** The job's first lifetime pair, its servers
  started cold, is left out of the figure, and a figure from a node any card of which slowed down in a counted pass
  is named as on a throttled node, with no verdict; the report counts every pair and reads no card's clock.
- **The advice line's serve is not under the network attester.** No arm of this command is served under the
  network attester's record or the certified tick, so the arrivals are charged at what they would cost without the tick. The
  kept count is charged at its code's bits: the accept test has a Definition, and no block of a served Program calls
  it, so nothing derives the count yet.
- **The recipe's settings are GLM-5.3's alone.** For every other model the arms are the benchmark's own: speculative
  decoding off and a context limit of 4,096 tokens unless `--max-model-len` is given.
- **The context limit is not the model's own.** At the recipe's settings it is the recipe's number for the cards,
  given outright: with the limit left to it and the recipe's other flags as they are, vLLM at the benchmark's pin does
  not start this model on them (`limits.json`). The report prints the limit beside the model's own.
- **`prod` is not the recipe's line whole.** It is behind vLLM's Python front end (serve_tax's `run.sh`:
  `PUBLISHED_FRONTEND=rust` serves the recipe's Rust one on a build that has it), with the engine's seed set
  (`serve_tax.BASE_ARGS`), and its requests are token ids sent to the completions endpoint. Every engine, stock too,
  starts with Python's hash seed and Inductor's block choice pinned.
- **No circuit is compiled.** Nothing here compiles the settings' circuit ahead of serving or checks a served
  request against it.
- **The replayed window is not sampled by an audit.** It is every request of one rep, named by the command.
- **The agentic load's tokens are synthetic.** Its sessions are recorded shapes, each block of a prompt made from
  its id, so the share of its drafts the engine accepts is a synthetic prompt's. Only vLLM's client sends it; the
  corpus client's burst is independent requests with shared prefixes.
- **Only ours is replayed.** A stock arm keeps no step trace, so nothing shows that stock's own tokens come back.
