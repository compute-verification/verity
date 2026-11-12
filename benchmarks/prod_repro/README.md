# vLLM at its serving settings, replayed

One command serves vLLM at the settings vLLM publishes for serving the model (prefix caching on, speculative decoding
on, the context limit the cards hold) with and without our changes, replays the window it served, and prints
what the serve turned on and three lines: whether the replay's tokens are the served ones, what our changes cost over
stock, and the advice a request. It measures nothing of its own. The serve, the load, the cost and the replay are the
serving cost benchmark's (`benchmarks/serve_tax/`: `run.sh`, `serve_tax.py`, `replay.py`); this directory is the runner
that calls them (`run.sh`) and the reader of what they leave (`prod_repro.py report`).

## The command

From the tree's root, as the command of a `research run` on a GPU node that the serving cost benchmark runs on (its
pinned vLLM and the model's checkpoint: `benchmarks/serve_tax/run.sh`'s header), with the C4 shard sent for a job of
the corpus client:

~~~text
verity bench prod_repro [--model moonlight] [--stock ARM] [--again ARM|none] [--ours ARM] [--load burst] [--rate 1000]
                        [--requests 800 | --span-s S] [--reps 3] [--replay PASS|none] [--max-model-len N] [--target 1]
verity bench prod_repro --client vllm-bench --load NAME[=RUNG+RUNG...] [--reps 6] [--replay NAME@RUNG]
                        [--pass-prompts NAME=N] [--sampled] [--stop-at TIME] ...
verity bench prod_repro --report OUT --load NAME [--replay PASS] [--stock ARM] [--again ARM|none] [--ours ARM]
ADVICE_TRACE=DIR [ADVICE_STOPS=ID,...] verity bench prod_repro ...
~~~

- **The arms.** A job serves stock, stock a second time and ours, each a server lifetime of its own. For a model
  whose recipe the benchmark holds (GLM-5.3, `--model glm53`) they are the arms at the recipe's settings: `prod`,
  stock vLLM as its recipe starts it on these cards; `prod-again`; and `prod-record`, which adds the scheduler's step
  trace and keeps each request's prompt ids and served ids. For any other model the benchmark refuses those arms, and
  the job serves `compiled`, `compiled-again` and `record`, at the benchmark's own settings. `--again none` serves
  stock once: the cost then has no noise floor.
- **The first form** sends the load by the corpus client: every arm gets the same plan (C4 prompts, about 60% of
  them behind one of eight shared prefixes) as a burst the engine can't keep up with, each rep from an empty prefix
  cache. It then replays rep 1 of ours (`--replay none`: no replay).
- **The second form** sends the load by vLLM's own client at its rungs, which gives the benchmark's paired cost.
  `--load agentic=N` is a coding agent's recorded sessions sent by N users, each turn's prompt beginning with the
  turn before it. It replays nothing unless `--replay` names the pass of ours to re-run, as the benchmark names it
  (`NAME@RUNG`): that needs an arm of ours that keeps each request's ids (`prod-record`), which vLLM's client does
  not keep. `--pass-prompts`, `--sampled` and `--stop-at` are the benchmark's `PASS_PROMPTS` (the prompts a pass
  sends), `SAMPLED` (each request sampled at the recipe's temperature and top-p, under its user's seed) and `STOP_AT`
  (an arm starts no pass it can't finish by then; no replay starts after it either).
- **The replay** is a step of its own after the arms: the benchmark's runner once more, in its replay-only form, so
  the job leases the cards a second time (`MAX_MIN` is the job's minutes in all: the replay's lease gets what the
  arms left of it). `replay.py` rebuilds the engine from the step trace's header and re-runs
  the pass under its recorded schedule. The step runs under what ours' engine has on that the job's environment has
  not: the arm's own vLLM general plugins and environment (`prod_repro.py replay-env`; today the canonical DSA
  top-k's, for an arm that has it). They are set for the replay alone, since the benchmark builds every server's path
  from the job's and a stock arm's server must not inherit them. The plugins' site and the files they keep are in a
  directory outside the run's, removed when the step ends.
- **The third form** reads an earlier job's `out/` again and needs no GPU; where it names no arm, the job is read as
  the arms it served.
- `ADVICE_TRACE` names the trace directory of a certified serve, which the advice line is read from; no arm of the
  job is one. `ADVICE_STOPS` are the registered stop ids it is read under, EOS first; without it they are the
  `eos_token_id` of the served checkpoint's `generation_config.json`.

The report goes to the job's `out/prod_repro.txt` and `out/prod_repro.json`. The job fails where an arm or the replay
fails, where a replay was asked for and its tokens were not compared, and where a replayed token or step differs from
the served one. A job that ends early still has its report, also one whose time ended (SIGTERM: the arm stops, no
replay starts, and the report is of what was served): it says which arm failed and on what, which stopped
before a pass, whether the replay ended and how its log ends; and for a replay that differs, the first token that
does and the first step.

## What each line means

- **model**: the model as the benchmark names it and the checkpoint the engines loaded; they are two things where
  `CHECKPOINT` names another model's directory.
- **what ran**: each arm from its own record (`meta.json`): the seconds its server took to come up, how many times it
  served the load and how many requests it left unfinished; `FAILED` with the record's error, `STOPPED` before the
  pass `STOP_AT` left no time for, `NOT SERVED` for an arm with no record. Then the replay: whether it ended, and where
  it did not, the last line of its log (`out/replay-<mode>.log`); and the replay's engine (`out/replay_engine.json`):
  the plugins and environment of ours it ran under, or that ours turns nothing on in its engine beside its record.
  For the canonical DSA top-k, what its own counters say of the replay's processes and, where the job kept the arms'
  own files (`PRIVATE`), of serving's: the processes that kept counters, the calls of the indexer's top-k and the
  rows past its keys that were rewritten. The counters are lower bounds (a process writes them every few thousand
  calls and when it exits), and a replay in which no process kept any shows nothing of the top-k having run there.
- **settings**: what ours ran with, stock's beside it, each with what says so (the launch line, the engine's own
  configuration in the step trace's header, the engine's counters): prefix caching, speculative decoding with the
  share of draft tokens accepted where the benchmark counts them, the context limit beside the model's own (the
  checkpoint's `config.json`), the KV cache's type, compiled or eager, and whether the settings' circuit was compiled
  ahead of serving. A setting the serve did not turn on is `off`; `DIFFERS` marks one that stock and ours do not
  share, and the cost line then gives its figure without a verdict. Speculative decoding or a KV cache type that the
  launch line turns on and the step trace's header does not hold is said: the replay builds its engine from the
  header. A context limit the launch line leaves to vLLM (`--max-model-len auto`) is the one the engine took: the
  header's where the arm kept a step trace, else the one vLLM's own log line says it fitted (stock keeps no trace);
  `not resolved` where neither says, and two arms that both leave it to vLLM then count as sharing it. A limit under
  the model's own is printed as below it.
- **tokens**: the requests of the replayed window, their served tokens, the tokens the replay kept for them and how
  many differ; the engine steps and how many have other rows than the served schedule; how many of the window's
  requests began from a prefix-cache hit and the longest hit; the mode served and replayed; how the window's
  requests were sampled (greedy, or at a temperature with or without a seed); how the window was chosen; and, for a
  checkpoint whose `config.json` names the keys its DSA indexer's top-k keeps (`index_topk`), how many of the
  window's requests passed that many ids of prompt and output, and how many of the differing requests are among
  them: only past it does attention read the keys the top-k selects.
  `reproduced` means no token and no step differs. Where one does: the first token that differs, the one serving
  emitted in the earliest step (its request, its position, the id served and the id replayed there, and that step),
  and the first step whose rows differ, with the rows only serving ran and the rows only the replay ran.
- **cost**: stock's output tokens a second over ours, less one: the share of stock's serving time our changes add at
  equal tokens, with its 95% interval over reps. Beside it, the same figure for stock served a second time, the noise
  floor, and whether the interval, widened by the floor, is under the target. By the corpus client the figure is
  `serve_tax.tax` and is a cost only where stock did not keep up with the arrivals; where it kept up, the line is not
  measured. By vLLM's client it is `serve_tax.serving_cost`, the paired ratio at the rung where stock serves most (for
  a sessions load, at its users), which needs six reps for an interval.
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
- **The recipe's settings are GLM-5.3's alone.** For every other model the arms are the benchmark's own: speculative
  decoding off and a context limit of 4,096 tokens unless `--max-model-len` is given.
- **The context limit is not the model's own.** At the recipe's settings it is the longest the cards' KV cache
  holds, which vLLM fits when the engine starts; the report prints it beside the model's own.
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
