# training_slowdown

The slowdown of a training run: how much more compute the run needs when every block of its circuit holds at most c
gates and reads at most k bits beyond the registered weights θ. The package `training_slowdown` counts a run's circuit
|C| in Boolean gates, traces one training step to measure what the training formalism reads (the update's dependence,
the cuts, the step's matrix products), and gives ρ at a policy (c, k): Fit's c/(c − d) where the training fits in a
block, else the one bound's products term (the theorem of record on M, the products whose two operands both depend on
the update, given (H) and (E)), with Slowdown's token form and the sensitivities beside it. A change to what it
measures bumps `VERSION` (`training_slowdown/__init__.py`), which every record carries.

## Modules

| module | what it does |
|---|---|
| `run.py` | one job of an attack config: `Harness`, what a route's code is counted inside, the route loop (`train`) and the record's trace (`trace_step`, `retrace`); `block_c`, a start's c |
| `config.py` | an attack config: its schema, its digest and its jobs (`Job.label`), each refusal by name |
| `screen.py` | the screen of a route's source: the names a route may not contain (`FORBIDDEN`), read before the route is imported |
| `sandbox.py` | the training job's sandbox: namespaces with no network and an empty directory over each final set, over /run and over the user's tmux directory (`enter`), and what a process can still reach of them (`refusals`) |
| `routes/` | the routes a config names, and the route contract (`routes/__init__.py`); `subset`, the parameters a pattern names trained with AdamW or Muon, E the tensors in full or their deltas in codes |
| `flops.py` | `Counter`, the dispatch-mode count of every op's multiply-adds and their provenance; `guard_exits`, token data read out of torch; `flags`, work the count can't vouch for; `forward_macs_per_token` |
| `opnames.py` | the op-name sets the counter, the capture and the gate prices share |
| `gates.py` | each counted op's price in Boolean gates, from the circuits' Definitions |
| `annotate.py` | a record's maximal annotation: per step U and D_t, the depth profile and the cut; `with_structures`, its traced steps' measured quantities |
| `capture/` | one training step traced on fake tensors (`capture_step` traces any PyTorch step, `capture_fn` a function, `capture_run` a run of them): |C|, the update's dependence and the cuts (`capture.circuit`), the step's matrix products as rows (`capture.products`) and its lowering to the training-step frontend (`capture.lower`) |
| `formalism.py` | a record's quantities (`of_record`, `measure`), Fit and Slowdown at (c, k) (`at`), the one bound and its sensitivities (`bound`), the products term (`he_products`, `he_figure`) and the published ρ with what it rests on (`published`) |
| `policy.py` | a start's reference shapes and the (c, k) grid; the console's policy (`CONSOLE`), one session of the 45M reference model; `named`, the one place a config's policy name becomes (c, k) |
| `grade.py` | the grade: a config's slowdown S = p·ρ from its jobs' records and its honest config's, printed and written as `slowdown.json` (below) |
| `calculator.py` | the slowdown calculator (`verity/protocols/inference_only/rho`) as this package reads it: a fixed policy's numbers, the fleets and periods a figure is presented over, and the bound of record |
| `counterexample.py` | a run's counterexample families: their writer and the sampled bit-for-bit rebuild check |
| `codecs/` | update codes: identity, low rank, sparse mask, quantized delta, scaled quantized delta, seed replay and their composition |
| `starts.py`, `starts.toml` | the starts registry: each start by its store artifact, its files' sha256, its shape, C0 and loss, and its data's pins; `load` reads an installed start and checks it |
| `data.py` | a start's data: the training stream (`Stream`, by the job's data seed) and the held-out sets (`held_out`: the search pool and the final set), checked against the registry; the search slice from a live coin (`sample_slice`) |
| `evaluate.py` | the harness's evaluator: a checkpoint's bits per byte on the job's split, the search pool's paired with θ on the same slice |
| `model.py` | the starts' architecture, the lf dense decoder |

The reference shapes are the lf model's (`adapters/vllm/tests/program/train_step_lf.py`), which the suite checks.

## Running an attack

~~~text
verity bench training-slowdown NAME|CONFIG.toml [--dry-run] [--local [--root DIR] | --on MACHINE [--pool NAME]] \
    [--weights]
python -m training_slowdown.run --config CONFIG.toml --seed 0 --split search|final --out OUT [--starts-root DIR] \
    [--device cuda] [--keep-weights] [--state DIR]      # one job
python -m training_slowdown.run --retrace RUN_JSON --out OUT   # a record with its trace taken again
python -m training_slowdown.capture.circuit STEP.npz    # a saved step circuit's quantities
python -m training_slowdown.gates [--check]             # gates per multiply-add by op; --check compiles the Definitions
~~~

An attack is a config of `attacks/` (`config.py`'s docstring has the schema): a start, a route and its dials, the
budget, the seeds, and what the grade reads, the target gain, the policy (`console`, policy.py's rule, or a
`[policy.*]` table of `verity/protocols/inference_only/rho/data`) and the honest config. The command (`run.py` here) expands a config and its honest config into
jobs, one per seed and split, each labelled in the store `training-slowdown=<digest>/<seed>/<split>`, and runs the ones
its labels lack: on the queue (kind `training-attack`, `infra/cluster/kinds/training.toml`), on one machine with `--on`,
or here, one after another, with `--local`. It exits 0 when every job has its record, 3 while one is pending or
stopped and 1 when one failed; a failed job is never run again. A job's run gets the time its config's budget gives
it (`budget.minutes`, the job's own limit over every run it continues in; without it, the kind's). A queued job
writes nothing into the tree it runs from, which every run of its commit shares: its Python environment is the
run's own, built by `uv sync --frozen` in a directory the job makes under /tmp (`UV_PROJECT_ENVIRONMENT`), which the
job's shell removes when the job ends, stopped or not.

A stopped job continues. After each checkpoint but the last a job keeps what continuing needs in its state directory
(`--state`): the model, the optimizers' state, the stream's position, the generators, and the record so far with its
scores and its clock. Told to stop (SIGTERM: an eviction, or the end of its run's time) it first writes a `run.json`
that says so (`stopped`: "told to stop", with no step or checkpoint), then ends the route's process and leaves the
state. The command reads a stop from that record and never from the run's exit status, which is whatever stopped the
job reports (a lease out of time: `timeout`'s 124, whatever the job's own): a run that didn't end with 0 is stopped
when its record says so, and failed when it holds no record or one that doesn't. A job told to stop before its own
process is up has written none, and is failed. The command's next run starts a stopped job again, and the job
continues from the state when it is its own (the same
config digest, seed, split and `training_slowdown` VERSION, a model that hashes to the record's last checkpoint) and
refuses any other, naming what differs. The job's own process states in the record so far, for the segment that
kept the state, the SHA-512 of each of its parts (the model, the optimizers' state, the stream's position, the
generators); the job that continues hashes them again, in the state directory and as the route's process holds them
before it imports the route, and refuses a state a part of which hashes to anything else. That record lies beside
the tensors, so the command ties it to the stopped run's record on the store, which no job writes: it tells the
job the sha256 the store lists for that record's `state.json` (`--continues`), or that it found none, and the job
continues from no other state (a state on a node's disk has no such copy, and none is checked). The record lists
the segments
the job ran in, and its count is their sum: every step once, and each segment's `build`. What a stopped segment
computed after its last kept checkpoint is run again and is in no record. A state is a cache: the command keeps it
with the stopped run's output on the store, the next run fetches it on whichever machine it gets (`NODE_STATE` in
`run.py` here places it on the node's disk instead), and losing it costs a start from θ. The state directory is
hidden from the route's process as a final set is, and only the job's own process writes it. A state is at most three
times its model (the weights and two buffers of the optimizer for each), and a run holds three states at once while
it keeps one, and a fourth, the store's own copy, when it fetched the one it continues from: the command asks the
queue for that much disk from the start's parameters, and 20 GB for the rest (`run_disk_gb` in `run.py` here).

A job trains the config's start along its route and scores each checkpoint on its split. A route is a module of
`routes/` with `VERSION`, `DIALS`, `build`, `step` and `plan` (the route contract, `routes/__init__.py`); the harness
owns the loop, the batches, the checkpoints and each checkpoint's E, the weights that differ from θ encoded by the
route's plan. A checkpoint's output model, the one scored, is what E rebuilds from θ: the trained weights under a
plan that keeps them whole, and the model the kept bits give under one that keeps less. Everything a route computes
through torch is counted; the harness's own work (loading, evaluating, encoding) isn't. A route that names anything of `training_slowdown` but the codecs and the model,
or a route around the dispatcher (`screen.FORBIDDEN`), is refused, and one that uses an op the counter can't price, reads
token data out of torch, starts a thread or a process, or imports a module the screen refuses makes its run invalid
(`screen.py` has the screen).

A route's code runs in a process of its own, inside the job's sandbox (`sandbox.py`): no network interface is up and
each registered final set's directory, and the job's state directory, is empty there. So are /run and the directory
of the user's tmux server, where the sockets are through which a process has another started outside the sandbox (the
user's systemd manager, its session bus, tmux). That process stops before it imports the route if it finds
any of them reachable, or holds a capability that could undo them, as root's does, so on a machine that can't make the
sandbox a job fails with the reason and trains nothing. The
job's own process reads the held-out split and scores each checkpoint's state as the route's process hands it out;
no score goes back in. The sandbox hides those paths and nothing else of the file system: a socket at another path
stays reachable.

After its last step a job traces its route's step once, outside the count, and its record's `trace` holds what the
capture measures of it for the grade: the token axes, the forward's updated MAC nodes, and the product rows unpriced
at each k of the console's dial (`capture.products.raw_on_dial`), which `capture.products.priced` prices where they
are read. A step the capture refuses is recorded as refused, and the job stays valid.

A job's output holds no weights by default: its record, and its last checkpoint's E when E is under 50 MiB. The
command's `--weights` makes each job it runs keep its last checkpoint's whole state too.

Run the `python -m` commands from this directory. A start and its data are read where `starts.toml` says a node holds them, and checked
against it; elsewhere, `research data fetch ART --path starts/NAME/theta.safetensors` (and its `start.json`) gets a
start, and `--starts-root` points at it.

## Grading a config

~~~text
python -m training_slowdown.grade --config attacks/NAME.toml --starts starts.toml --records DIR --honest DIR \
    [--also NAME=DIR ...] [--out OUT]
~~~

A config names its `start`, its `target.gain`, its `policy`, its `honest` config and its `seeds`; each DIR holds one
config's records, `DIR/<seed>/<split>/run.json` (`split` is `search` or `final`), a traced record as its rows
(`rows.jsonl[.gz]`) where it has them. The grade reads no file of a served run. It prints one block and writes
`OUT/slowdown.json` (`slowdown/v1`):

- **the gain** each seed reached, through the fit of the starts on the start's data (loss to effective compute);
- **H**, the least counted |C| reaching the target gain: over the honest config's records, and over every config's
  given (`--also` adds configs at the same start and gain);
- **p**, the attack's |C| over H, against the honest route and against any route, each with its range over the seeds;
- **ρ**, the attack's own cost at the config's policy: `formalism.published` at (c, k) on the traced search record;
- **S = p·ρ**, with what each factor rests on, and each seed beside it: whether it reaches the gain, and its |C|;
- **the bound of record** at the same policy, what every partition of the attack's traced circuit must pay, beside S
  (what this attack pays) and never mixed into it; "not available", with what is missing, where the calculator has
  no function that gives it;
- **per period**: the run's AND gates at S and their share of one datacenter's and of the world's compute over a
  window, a week, a month and a year, with the measured, proved and scaled parts named.

A config's `policy` is a name, and `policy.named` resolves it: `console` is the rule relative to the start (c one
session of the reference model of the start's size, k the session's input ids), and any other name is a fixed policy
of the calculator's registry (`verity/protocols/inference_only/rho/data`, its `[policy.*]` tables). The block says which kind it was, and a
name that is neither is refused. p is read on the final set where the attack and its honest config each have a final
record, else on the search split, and the block says which. Without a traced search record there is no ρ and no S.

## Limits of the count

- Work outside torch's dispatcher (custom CUDA or Triton kernels, host libraries) isn't counted: the screen refuses the
  usual routes by name, and the flags show work the count can't vouch for. Torch compilation is disabled and CUDA graph
  replay refused.
- The measured quantities are one traced step's: a record's updated steps take its representative step's structure,
  scaled to their tokens. A cut counts a value at its dtype's width, and values are whole tensors.
- Bit-for-bit rebuilds need the run's kernels: families are checked on the run's device, right after it trains.

The suite: `uv run tools/verity/check/suites.py benchmarks/training_slowdown`.
