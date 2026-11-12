# vLLM at its serving settings, replayed

One command serves vLLM at the settings vLLM publishes for serving the model (prefix caching on, speculative decoding
on, the context limit its recipe sets for the cards) with and without our changes, replays the window it served, and
prints what the serve turned on and three lines: whether the replay's tokens are the served ones, what our changes
cost over stock, and the advice a request. It measures nothing of its own. The serve, the load, the cost and the
replay are the serving cost benchmark's (`benchmarks/serve_tax/`: `run.sh`, `serve_tax.py`, `replay.py`); this
directory is the runner that calls them (`run.sh`) and the reader of what they leave (`prod_repro.py report`).

## The command

From the tree's root, as the command of a `research run` on a GPU node that the serving cost benchmark runs on (its
pinned vLLM and the model's checkpoint: `benchmarks/serve_tax/run.sh`'s header), with the C4 shard sent for a job of
the corpus client:

~~~text
verity bench prod_repro [--model moonlight] [--stock ARM] [--again ARM|none] [--ours ARM] [--load burst] [--rate 1000]
                        [--requests 800 | --span-s S] [--reps 3] [--replay PASS|none] [--max-model-len N] [--target 1]
verity bench prod_repro --client vllm-bench --load NAME[=RUNG+RUNG...] [--reps 6] [--replay NAME@RUNG]
                        [--pass-prompts NAME=N] [--sampled] [--stop-at TIME] [--lifetimes L] [--leave-out N] ...
verity bench prod_repro --report OUT --load NAME [--replay PASS] [--stock ARM] [--again ARM|none] [--ours ARM]
verity bench prod_repro --replay-of OUT --load NAME --replay PASS [--model ...] [--ours ARM]
~~~

- **The arms.** A job serves stock, stock a second time and ours, each a server lifetime of its own. For a model
  whose recipe the benchmark holds (GLM-5.3, `--model glm53`) they are the arms at the recipe's settings: `prod`,
  stock vLLM as its recipe starts it on these cards; `prod-again`; and `prod-record`, which adds the scheduler's step
  trace and keeps each request's prompt ids and served ids. For any other model the benchmark refuses those arms, and
  the job serves `compiled`, `compiled-again` and `record`, at the benchmark's own settings. `--again none` serves
  stock once: the cost then has no noise floor. `--stock none` serves ours alone, a first start of it and its replay:
  there is then no cost, and under vLLM's client the benchmark's own cost step ends the runner with 1 for want of a
  stock arm, which the job takes as no failure where ours served its load.
- **The first form** sends the load by the corpus client: every arm gets the same plan (C4 prompts, about 60% of
  them behind one of eight shared prefixes) as a burst the engine can't keep up with, each rep from an empty prefix
  cache. It then replays rep 1 of ours (`--replay none`: no replay).
- **The second form** sends the load by vLLM's own client at its rungs, which gives the benchmark's paired cost.
  `--load agentic=N` is a coding agent's recorded sessions sent by N users, each turn's prompt beginning with the
  turn before it. It replays nothing unless `--replay` names the pass of ours to re-run, as the benchmark names it
  (`NAME@RUNG`): that needs an arm of ours that keeps each request's ids (`prod-record`), which vLLM's client does
  not keep. `--pass-prompts`, `--sampled` and `--stop-at` are the benchmark's `PASS_PROMPTS` (the prompts a pass
  sends), `SAMPLED` (each request sampled at the recipe's temperature and top-p, under its user's seed) and `STOP_AT`
  (an arm starts no pass it can't finish by then; the replay runs after the arms whatever the time, and the job's
  end is its lease's, `MAX_MIN`).
- **Server lifetimes.** `--lifetimes L` is the benchmark's `LIFETIMES`: the arms are served L times, each time a
  server of its own, in the arms' order and then that order turned round, and lifetime K of an arm is `out/<arm>.K`.
  `--leave-out N` is its `LEAVE_OUT`: each lifetime's first N passes, its warm-up, are served and recorded and left
  out of every figure. A sessions load's cost has an interval, and so a verdict, only over two lifetimes an arm or
  more. The replay re-runs the named pass of ours's first lifetime, and the report says which lifetime its window
  is of.
- **The context limit** is `--max-model-len`'s, for every arm. Where the job names none it is the benchmark's for
  each arm: at the recipe's settings the recipe's for the cards (262,144 tokens for GLM-5.3). `limits.json` says what
  is known of leaving the limit to vLLM (`--max-model-len auto`), with the run that showed it, and the report's first
  line repeats it beside the limit the arms were started at.
- **The replay** runs after the arms, under their lease: `replay.py` rebuilds the engine from the step trace's header
  and re-runs the pass under its recorded schedule. An arm that served on the canonical DSA top-k (`prod-record`) is
  replayed on it: the replay loads the arm's other vLLM plugins under the arm's environment before it builds the
  engine (`replay.served_under`), and its record says so. Such an arm keeps files on the node: `PRIVATE` names a
  directory for them outside every run's, which the corpus client needs for it; a job of vLLM's client without one
  serves all the same and keeps no counters of the top-k.
- **The third form** reads an earlier job's `out/` again and needs no GPU; where it names no arm, the job is read as
  the arms it served.
- **The fourth form** serves nothing: it copies ours's directory from an earlier job's `out/` into this job's and
  re-runs the replay of the named pass from it, under a lease taken for the replay alone. Its report is of ours alone,
  with the tokens line and no cost.
- **The advice line** is of the pass the replay re-runs, the tokens line's: what that pass's schedule costs as
  advice, by the scheduler replay's loader (`benchmarks/network_e2e/encodings/replay.py`), which reads the pass's
  records out of ours's step trace with each request's prompt ids and served ids.

The report goes to the job's `out/prod_repro.txt` and `out/prod_repro.json`. Under `research run` the run's
`result.json` names the report's two files and the replay's record among its artifacts, so a reader finds them as the
run's files; the rest of `out/` is in the run's record whole. A job with no result of the benchmark's cost (one that
served ours alone) gets one whose validation is the tokens line.

The job fails where an arm or the replay fails, where a replay was asked for and its tokens were not compared, and
where a replayed token or step differs from the served one. One kind of served step is no difference, and the report
counts and prints it: a step after a request's end, which emitted no token, whose requests had all ended in an earlier
step, and which the replay did not run (the scheduler had planned it before it learned of the end, and serving dropped
what it computed; the replay ends with its last request). A job that ends early still has its report, also one whose
time ended (SIGTERM: the arm or the replay stops, and the report is of what was served and replayed by then): it says
which arm failed and on what, which stopped before a pass, whether the replay ended and how its log ends; and for a
replay that differs, the first token that does and the first step.

## What each line means

- **the first line**: the context limit the launch lines of ours and stock name, and for a model `limits.json` says
  so of, that vLLM does not start with the limit left to it, at which settings, why, and the run that showed it.
- **model**: the model as the benchmark names it and the checkpoint the engines loaded; they are two things where
  `CHECKPOINT` names another model's directory.
- **what ran**: each arm, or each lifetime of it, from its own record (`meta.json`): the seconds its server took to
  come up, how many times it
  served the load and how many requests it left unfinished; `FAILED` with the record's error and the first exception
  its server's log names (the record's error is the last thing that failed, and the cause is above it), `STOPPED`
  before the pass `STOP_AT` left no time for, `NOT SERVED` for an arm with no record. Then the replay: whether it
  ended, and where
  it did not, the last line of its log (`out/replay-<mode>.log`); and the replay's engine, as the replay's own
  record says: the plugins and environment of ours it ran under and what set them (the replay), or that ours serves
  with no plugin beside its record.
  For the canonical DSA top-k, what its own counters say of the replay's processes and of serving's: the processes
  that kept counters, the calls of the indexer's top-k and the rows of more than its keys that were rewritten. The
  processes keep them under `PRIVATE`; the job's report reads them there and keeps the counts in
  `out/topk_counters.json`. The counts are lower bounds (a process writes them every few thousand calls and when it
  exits), and a replay in which no process kept any shows nothing of the top-k having run there.
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
  the model's own is printed as below it. The last line is the value settings in force: the adapter's record of the
  variables set in each server's environment that can change what an engine computes, which the benchmark keeps in
  each lifetime's `meta.json` and of the replay's engine in `replay.json` (`serve_tax.value_settings`). It gives how
  many are set and the record's SHA-512 for stock, ours and the replay, and names the variables ours's record
  differs from stock's in and the replay's from ours's; `out/prod_repro.json` holds every record whole, a credential
  by its name alone. It is in no verdict: ours's server is given variables stock's is not. A directory that holds no
  such record is `not recorded`.
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
- **cost**: ours's time for the same work over stock's, less one: the share of stock's serving time our changes add.
  The verdict's unit is a server lifetime: the passes of one share its server's state, so an interval over them as if
  independent is too narrow, and the estimate alone shows nothing.
  On a sessions load every figure is the benchmark's own cost's (`serve_tax.session_cost`). The cost is the mean of
  the lifetimes' costs, a lifetime's being ours's seconds for its passes over stock's, less one, a pass's seconds
  from its users' first send to the last one's last answer, with its 95% interval over the lifetimes an arm.
  `under 1%: yes` is printed only where the benchmark says that interval's upper end is under the target; `NO` where
  its lower end is over it; `not shown` otherwise, with the reason: one lifetime an arm, or one pair of lifetimes
  with a pass both arms counted where an arm stopped before the rest (no such interval), an upper end that is not
  under the target, arms that did not do the same work (a request unserved, another count of output tokens, another
  share of the prompts from the prefix cache), or settings that differ. Under it: that each request
  decoded to its recorded length, past any end-of-text token, in the benchmark's own words (vLLM's client sends each
  with its length and `ignore_eos`, in every arm); the benchmark's line counting the requests that served an
  end-of-text token before their end and went on, for a side whose server kept the served ids and whose record names
  the checkpoint's end-of-text ids (stock keeps none), or that they are not counted where no side's passes hold the
  count; each lifetime's cost and the spread between lifetimes and between a lifetime's passes; the same cost under the
  benchmark's other readings of a pass (by the seconds its GPUs were busy, by its users' seconds summed, by the sum
  of its users' own rates), in no verdict; each side's engine steps a pass and the share of its draft tokens
  accepted, which the cost is adjusted for neither of; the cost over the passes taken
  as independent, the paired ratio's median with its interval, in no verdict; and the noise floor, beside and not
  added (each lifetime pair is already two servers): stock served a second time over stock where the job
  served it, else stock's own lifetimes, each second one over the one before it, a lifetime pair its unit too, with
  a line where its own interval leaves out 0: stock then differs from stock.
  On another load the figure is the one over the passes taken as independent and has no verdict. By the corpus
  client it is `serve_tax.tax` and is a cost only where stock did not keep up with the arrivals; where it kept up,
  the line is not measured. By vLLM's client it is `serve_tax.serving_cost`, the paired ratio at the rung where
  stock serves most, which needs six reps for an interval; its noise floor is stock served a second time, measured
  the same way.
- **advice**: the bits a request that the replayed pass's schedule costs as advice, charged: the arrivals (the
  residual advice of when each request arrived, decoded back to the pass's steps; a serve under the certified tick
  sends none, and no arm here is under it) and the kept count, the drafts each row kept, at the bits of a code
  the loader builds and decodes back to the pass's counts, with its bits a row and, not sent, what the counts' entropy
  taken each alone would come to. Under it: the kept count's own figures as the loader gives them (`K`, the rows, the
  rows that kept each count, their entropy and its most, the code's bits and an ideal coder's, the bits where it is
  derived); that the kept
  count costs 0 where the audited block derives it, which rests on a Definition of the accept test that does not exist
  yet; how the pass's sampled requests were seeded, from each one's own arrival (what a sampled request with no seed
  there was sampled with is in no record: its tokens cannot be replayed); and the pass. The loader's own
  replay of the scheduler must reproduce every step of the pass: where it does not, or the serve has no draft, the
  line is not measured and says the loader's reason.

A line says `not measured` with its reason where the job's files don't hold it or can't be read; the last line counts
the measured ones and names the arms that were not served whole.

## What is not yet true

- **The advice line's serve is not under the network certifier.** No arm of this command is served under the
  certifier's record or the certified tick, so the arrivals are charged at what they would cost without the tick. The
  kept count is charged at its code's bits: no Definition holds the accept test, so nothing derives it yet. A serve with
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
