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

Each point names its `model` (the checkpoint, `repo@revision`), its `hardware` (the cards it was served on) and its
`member`, the model, the card and the arm (the cost's over its stock), which tells its line from the id's other
models' and arms'. A serve with no draft, such as the 1B's, keeps no draft, so its kept count is 0 bits and its
advice is the arrivals'.

**The method a serving cost is measured by.** Four arms, `prod`, `prod-again`, `prod-seeded` and `prod-record`, each
served in nine server lifetimes in that alternating order, each lifetime opening with a 20-second warm-up pass that
no figure counts (the benchmark's `WARM_S`) and then two counted passes (`--reps 2 --lifetimes 9 --leave-out 0`).
The cost is the mean of the nine lifetime pairs' costs with its Student's t 95% interval over them, t at the pairs
less one, and stock against stock, `prod-again` over `prod` taken the same way, is the noise floor printed beside it
and never subtracted.

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
