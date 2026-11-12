# vLLM at its serving settings, replayed

One command serves vLLM at the settings vLLM publishes for serving the model (prefix caching on, speculative decoding
on, the model's own context limit) with and without our changes, replays the window it served, and prints
what the serve turned on and three lines: whether the replay's tokens are the served ones, what our changes cost over
stock, and the advice a request. It measures nothing of its own. The serve, the load, the cost and the replay are the
serving cost benchmark's (`benchmarks/serve_tax/`: `run.sh`, `serve_tax.py`, `replay.py`); this directory is the runner
that calls them (`run.sh`) and the reader of what they leave (`prod_repro.py report`).

## The command

From the tree's root, as the command of a `research run` on a GPU node that the serving cost benchmark runs on (its
pinned vLLM and the model's checkpoint: `benchmarks/serve_tax/run.sh`'s header), with the C4 shard sent:

~~~text
verity bench prod_repro [--model moonlight] [--stock ARM] [--again ARM] [--ours ARM] [--load burst] [--rate 1000]
                        [--requests 800 | --span-s S] [--reps 3] [--max-model-len N] [--target 1]
verity bench prod_repro --client vllm-bench --load NAME[=RUNG+RUNG...] [--reps 6] ...
verity bench prod_repro --report OUT --load NAME [--stock ARM] [--again ARM] [--ours ARM]
~~~

- The first form serves three arms of the benchmark, each a server lifetime of its own: stock, stock a second time,
  and ours. They are the arms at those settings (`prod`, `prod-again`, `prod-record`) where `serve_tax.ARMS` has them, else
  `compiled`, `compiled-again` and `record`. The corpus client sends each the same plan (C4 prompts, about 60% of them
  behind one of eight shared prefixes) as a burst the engine can't keep up with, each rep from an empty prefix cache.
  Then `replay.py` rebuilds the engine and re-runs rep 1 of ours under its recorded schedule.
- The second form sends the load by vLLM's own client at its rungs, which gives the benchmark's paired cost. Its
  passes keep no token ids, so it has no replay.
- The third form reads an earlier job's `out/` again and needs no GPU.

The report goes to the job's `out/prod_repro.txt` and `out/prod_repro.json`. The job fails where an arm or the replay
fails, and where a replayed token or step differs from the served one: each differing request is named with its first
differing position.

## What each line means

- **model**: the model as the benchmark names it and the checkpoint the engines loaded; they are two things where
  `CHECKPOINT` names another model's directory.
- **settings**: what ours ran with, stock's beside it, each with what says so (the launch line, the engine's own
  configuration in the step trace's header, the engine's counters): prefix caching, speculative decoding, the context
  limit beside the model's own (the checkpoint's `config.json`), the KV cache's type, compiled or eager, and whether
  the settings' circuit was compiled ahead of serving. A setting the serve did not turn on is `off`; `DIFFERS` marks
  one that stock and ours do not share, and the cost line then gives its figure without a verdict.
- **tokens**: the requests of the replayed window, their served tokens, the tokens the replay kept for them and how
  many differ; the engine steps and how many have other rows than the served schedule; how many of the window's
  requests began from a prefix-cache hit and the longest hit; the mode served and replayed; and how the window was
  chosen. `reproduced` means no token and no step differs.
- **cost**: stock's output tokens a second over ours, less one: the share of stock's serving time our changes add at
  equal tokens, with its 95% interval over reps. Beside it, the same figure for stock served a second time, the noise
  floor, and whether the interval, widened by the floor, is under the target. By the corpus client the figure is
  `serve_tax.tax` and is a cost only where stock did not keep up with the arrivals; where it kept up, the line is not
  measured. By vLLM's client it is `serve_tax.serving_cost`, the paired ratio at the rung where stock serves most, which
  needs six reps for an interval.
- **advice**: the bits a request by class, each with its number of events.

A line says `not measured` with its reason where the job's files don't hold it; the last line counts the measured ones.

## What is not yet true

- **The advice line is never measured.** The advice code runs on a serve under the network certifier and the
  certified tick; this command runs neither, and the tree has no split of the bits by class.
- **Speculative decoding is off** in every arm the benchmark has, and the replay builds its engine without it.
- **The context limit is the benchmark's**, 4,096 tokens unless `--max-model-len` is given, not the model's own.
- **No circuit is compiled.** Nothing here compiles the settings' circuit ahead of serving or checks a served
  request against it.
- **The replayed window is not sampled by an audit.** It is every request of one rep, named by the command.
- **The load is not agentic.** It is the corpus's plan of independent requests with shared prefixes.
- **One serve does not give both the paired cost and a replay.** Only the corpus client's serve can be replayed, and
  only vLLM's client gives the paired cost over six reps.
- **Stock is not untouched vLLM.** The benchmark starts every engine, stock too, with Python's hash seed and
  Inductor's block choice pinned.
