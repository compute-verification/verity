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
verity bench prod_repro [--model moonlight] [--stock ARM] [--again ARM|none] [--ours ARM] [--load burst] [--rate 1000]
                        [--requests 800 | --span-s S] [--reps 3] [--replay PASS|none] [--max-model-len N] [--target 1]
verity bench prod_repro --client vllm-bench --load NAME[=RUNG+RUNG...] [--reps 6] [--replay NAME@RUNG]
                        [--pass-prompts NAME=N] [--stop-at TIME] ...
verity bench prod_repro --report OUT --load NAME [--replay PASS] [--stock ARM] [--again ARM|none] [--ours ARM]
ADVICE_TRACE=DIR [ADVICE_STOPS=ID,...] verity bench prod_repro ...
~~~

- The first form serves three arms of the benchmark, each a server lifetime of its own: stock, stock a second time,
  and ours. They are the arms at those settings (`prod`, `prod-again`, `prod-record`) where `serve_tax.ARMS` has them, else
  `compiled`, `compiled-again` and `record`. `--again none` serves stock once: the cost then has no noise floor. The
  corpus client sends each the same plan (C4 prompts, about 60% of them
  behind one of eight shared prefixes) as a burst the engine can't keep up with, each rep from an empty prefix cache.
  Then `replay.py` rebuilds the engine and re-runs rep 1 of ours under its recorded schedule (`--replay none`: no
  replay).
- The second form sends the load by vLLM's own client at its rungs, which gives the benchmark's paired cost. It
  replays nothing unless `--replay` names the pass of ours to re-run, as the benchmark names it (`NAME@RUNG`): that
  needs an arm of ours that keeps each request's prompt ids and served ids, which a pass of vLLM's client does not
  keep on its own. `--pass-prompts` and `--stop-at` are the benchmark's `PASS_PROMPTS` (the prompts a pass sends) and
  `STOP_AT` (an arm starts no pass it can't finish by then).
- The third form reads an earlier job's `out/` again and needs no GPU.
- `ADVICE_TRACE` names the trace directory of a certified serve, which the advice line is read from; no arm of the
  job is one. `ADVICE_STOPS` are the registered stop ids it is read under, EOS first; without it they are the
  `eos_token_id` of the served checkpoint's `generation_config.json`.

The report goes to the job's `out/prod_repro.txt` and `out/prod_repro.json`. The job fails where an arm or the replay
fails, where a replay was asked for and its tokens were not compared, and where a replayed token or step differs from
the served one. A job that ends early still has its report: it says which arm failed and on what, which stopped
before a pass, whether the replay ended and how its log ends; and for a replay that differs, the first token that
does and the first step.

## What each line means

- **model**: the model as the benchmark names it and the checkpoint the engines loaded; they are two things where
  `CHECKPOINT` names another model's directory.
- **what ran**: each arm from its own record (`meta.json`): the seconds its server took to come up, how many times it
  served the load and how many requests it left unfinished; `FAILED` with the record's error, `STOPPED` before the
  pass `STOP_AT` left no time for, `NOT SERVED` for an arm with no record. Then the replay: whether it ended, and where
  it did not, the last line of its log (`out/replay-<mode>.log`).
- **settings**: what ours ran with, stock's beside it, each with what says so (the launch line, the engine's own
  configuration in the step trace's header, the engine's counters): prefix caching, speculative decoding with the
  share of draft tokens accepted where the benchmark counts them, the context limit beside the model's own (the
  checkpoint's `config.json`), the KV cache's type, compiled or eager, and whether the settings' circuit was compiled
  ahead of serving. A setting the serve did not turn on is `off`; `DIFFERS` marks one that stock and ours do not
  share, and the cost line then gives its figure without a verdict. A context limit the launch line leaves to vLLM
  (`--max-model-len auto`) is the one the engine took, which only an arm that kept a step trace records: stock's is
  then `not resolved`, and two arms that both leave it to vLLM count as sharing it.
- **tokens**: the requests of the replayed window, their served tokens, the tokens the replay kept for them and how
  many differ; the engine steps and how many have other rows than the served schedule; how many of the window's
  requests began from a prefix-cache hit and the longest hit; the mode served and replayed; how the window's
  requests were sampled (greedy, or at a temperature with or without a seed); and how the window was chosen.
  `reproduced` means no token and no step differs. Where one does: the first token that differs, the one serving
  emitted in the earliest step (its request, its position, the id served and the id replayed there, and that step),
  and the first step whose rows differ, with the rows only serving ran and the rows only the replay ran.
- **cost**: stock's output tokens a second over ours, less one: the share of stock's serving time our changes add at
  equal tokens, with its 95% interval over reps. Beside it, the same figure for stock served a second time, the noise
  floor, and whether the interval, widened by the floor, is under the target. By the corpus client the figure is
  `serve_tax.tax` and is a cost only where stock did not keep up with the arrivals; where it kept up, the line is not
  measured. By vLLM's client it is `serve_tax.serving_cost`, the paired ratio at the rung where stock serves most, which
  needs six reps for an interval.
- **advice**: the bits a request of the certified serve whose trace directory `ADVICE_TRACE` names, then one line a
  class: its bits, its standing in the code and its events, `not exercised` where it has 0 bits and none of its events
  arose. The split is the tree's reader's (`prod_repro.BY_CLASS`: its `report`), whose whole row is kept in
  `out/advice_by_class.json`. It is measured only where the reader gives a table and the advice decodes
  back to the trace.

A line says `not measured` with its reason where the job's files don't hold it or can't be read; the last line counts
the measured ones and names the arms that were not served whole.

## What is not yet true

- **The advice line is not of this job's serve, and is not measured yet.** The advice code runs on a serve under the
  network certifier and the certified tick; no arm of this command is one, so the line reads another serve's trace
  directory. The tree has no reader of the bits by class, so with a directory the line says so; and the advice code
  has no field for speculative decoding, so a certified serve at these settings would be refused.
- **Speculative decoding is off** in every arm the benchmark has, and the replay builds its engine without it.
- **The context limit is the benchmark's**, 4,096 tokens unless `--max-model-len` is given, not the model's own.
- **No circuit is compiled.** Nothing here compiles the settings' circuit ahead of serving or checks a served
  request against it.
- **The replayed window is not sampled by an audit.** It is every request of one rep, named by the command.
- **The load is not agentic.** It is the corpus's plan of independent requests with shared prefixes.
- **One serve does not give both the paired cost and a replay.** Only the corpus client's serve can be replayed:
  no arm the benchmark has keeps the ids of a pass sent by vLLM's client, so `--replay NAME@RUNG` finds no window. Only
  vLLM's client gives the paired cost over six reps.
- **Stock is not untouched vLLM.** The benchmark starts every engine, stock too, with Python's hash seed and
  Inductor's block choice pinned.
