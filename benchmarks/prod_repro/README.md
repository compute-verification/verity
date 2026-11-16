# vLLM at its serving settings, replayed

Does Verity's serving still reproduce when vLLM runs the way people deploy it? This benchmark serves a model at the
settings vLLM publishes for it (prefix caching on, speculative decoding on, the context limit its recipe sets for the
cards), once stock and once with our changes, then replays what ours served under its recorded schedule. It prints
what the serve turned on and three lines:

- **tokens**: whether the replay gave back every served token;
- **cost**: what our changes add to stock's serving time;
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

## What is not yet true

- **The advice line's serve is not under the network attester.** No arm of this command is served under the
  certifier's record or the certified tick, so the arrivals are charged at what they would cost without the tick. The
  kept count is charged at its code's bits: the accept test has a Definition, and no block of a served Program calls
  it, so nothing derives the count yet. A serve with
  no draft (every model but GLM-5.3 here) has no line: the loader's table of a draft's advice is the one read.
- **The recipe's settings are GLM-5.3's alone.** For every other model the arms are the benchmark's own: speculative
  decoding off and a context limit of 4,096 tokens unless `--max-model-len` is given.
- **The context limit is not the model's own.** At the recipe's settings it is the recipe's number for the cards,
  given outright: with the limit left to it and the recipe's other flags as they are, vLLM at the benchmark's pin does
  not start this model on them (`limits.json`). The report prints the limit beside the model's own.
- **`prod` is not the recipe's line whole.** It is behind vLLM's Python front end, with the engine's seed set, and
  its requests are token ids sent to the completions endpoint (`benchmarks/serve_tax/README.md` lists each
  departure). Every engine, stock too, starts with Python's hash seed and Inductor's block choice pinned.
- **No circuit is compiled.** Nothing here compiles the settings' circuit ahead of serving or checks a served
  request against it.
- **The replayed window is not sampled by an audit.** It is every request of one rep, named by the command.
- **The agentic load's tokens are synthetic.** Its sessions are recorded shapes, each block of a prompt made from
  its id, so the share of its drafts the engine accepts is a synthetic prompt's. Only vLLM's client sends it; the
  corpus client's burst is independent requests with shared prefixes.
- **Only ours is replayed.** A stock arm keeps no step trace, so nothing shows that stock's own tokens come back.
