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
| `sandbox.py` | the training job's sandbox: namespaces with no network and an empty directory over each final set (`enter`), and what a process can still reach of either (`refusals`) |
| `routes/` | the routes a config names, and the route contract (`routes/__init__.py`); `subset`, the parameters a pattern names trained with the optimizer its `opt` dial names (AdamW, SGD, Lion, Adafactor, Muon, Shampoo; the ones torch lacks in `_optimizers`; under Muon and Shampoo the gains, and the embedding unless the config puts it among the matrices, train under AdamW beside it), a rate each for the matrices, the gains and the embedding, on a linear, cosine or constant schedule, E the tensors in full or their deltas in codes; `ema`, `subset`'s step with the output an exponential average of the weights, at one decay or reaching back a share of the steps so far; `lora`, the start frozen and a low-rank adapter trained beside each matrix a pattern names, the output the start with the adapters merged in |
| `flops.py` | `Counter`, the dispatch-mode count of every op's multiply-adds and their provenance; `guard_exits`, token data read out of torch; `flags`, work the count can't vouch for; `forward_macs_per_token` |
| `opnames.py` | the op-name sets the counter, the capture and the gate prices share |
| `gates.py` | each counted op's price in Boolean gates, from the circuits' Definitions |
| `annotate.py` | a record's maximal annotation: per step U and D_t, the depth profile and the cut; `with_structures`, its traced steps' measured quantities |
| `capture/` | one training step traced on fake tensors (`capture_step` traces any PyTorch step, `capture_fn` a function, `capture_steps` a step among the steps before and after it, with its step circuit, `capture_run` a run of them): |C|, the update's dependence and the cuts (`capture.circuit`), the step's matrix products as rows (`capture.products`) and its lowering to the training-step frontend (`capture.lower`) |
| `formalism.py` | a record's quantities (`of_record`, `measure`), Fit and Slowdown at (c, k) (`at`), the one bound and its sensitivities (`bound`), the products term (`he_products`, `he_figure`) and the published ρ with what it rests on (`published`) |
| `policy.py` | a start's reference shapes and the (c, k) grid; the console's policy (`CONSOLE`), one session of the 45M reference model; `named`, the one place a config's policy name becomes (c, k) |
| `grade.py` | the grade: a config's slowdown S = p·ρ from its jobs' records and its honest config's, printed and written as `slowdown.json` (below) |
| `library.py` | the library's table: each entry of `attacks/` with its step traced on a CPU, its slowdown at its policy (the calculator's bound of record on that step) and what binds, and its planned run over each fleet and period |
| `calculator.py` | the slowdown calculator (`verity/protocols/inference_only/rho`) as this package reads it: a fixed policy's numbers, the fleets and periods a figure is presented over, and the bound of record |
| `counterexample.py` | a run's counterexample families: their writer and the sampled bit-for-bit rebuild check |
| `codecs/` | update codes: identity, low rank, sparse mask, quantized delta, scaled quantized delta, seed replay and their composition |
| `starts.py`, `starts.toml` | the starts registry: each start by its store artifact, its files' sha256, its shape, C0 and loss, and its data's pins; `load` reads an installed start and checks it |
| `data.py` | a start's data: the training stream (`Stream`, by the job's data seed) and the held-out sets (`held_out`: the search pool and the final set), checked against the registry; the search slice from a live coin (`sample_slice`) |
| `evaluate.py` | the harness's evaluator: a checkpoint's bits per byte on the job's split, the search pool's paired with θ on the same slice |
| `model.py` | the starts' architecture, the lf dense decoder |

The reference shapes are the lf model's (`adapters/vllm/tests/program/train_step_lf.py`), which the suite checks.

## The library of training algorithms

~~~text
python -m training_slowdown.library [ENTRY ...] [--starts starts.toml] [--out DIR]
~~~

The library is the configs of `attacks/`. An entry is a route and its dials at a start, with the policy it is read
under: there is one kind of entry, whatever its route computes. A new entry is a config of a route
`routes/` has, or a new route (the route contract, `routes/__init__.py`) and a config of it; its config's first
comment says what it trains. Each entry is held to a reference in plain torch (`tests/test_entries.py`): its own
dials, a few steps at a small shape, and the same output model as torch's own optimizer and its own average of the
weights give, or the published rule where torch has none. An entry without a reference fails the suite.

A run that could hand out its weights or their average is two entries, `NAME` and `NAME-ema`, the same config but for
the route and the average's horizon: each is a run of its own, a frontier takes the lower of the two, and no entry
chooses between them inside a run. The suite refuses an average over a horizon with no such entry beside it, and holds
it to torch's own average of the weights that entry writes.

The standard algorithms differ in two ways the table shows apart. AdamW, SGD, Lion, Adafactor and the average of the
weights train every tensor by the same forward and backward passes, so they carry the same products with two operands
the update reaches, and differ only in the gates the optimizer itself adds to the step; gradient accumulation carries
those products in several passes, which write more outputs. Muon and Shampoo with AdamW beside them on the gains and
the embedding (`muon-whole`, `shampoo-whole`) train every tensor by those passes too: the bound is on the same
products, and the optimizer adds gates and products of its own (Muon's Newton-Schulz steps, Shampoo's factors and
roots). The rest change the passes' products. `ref-optim` (Muon) and `shampoo` leave the embedding untrained, and the
output head, whose weight is the embedding, no longer has two operands the update reaches. Frozen layers
(`last-blocks`) leave more products out, and adapters (`lora`) reach a matrix only through a rank-wide product.

The command prints one row an entry (every entry, or the ones named): the share of the start's parameters its step
trains, its slowdown under its policy and what binds. It trains nothing. It builds the entry's model at the start's
shape, traces its step on a CPU as a job does after its last step (`run.traced_step`, on fake tensors), and reads the
calculator's bound of record on the traced step (`calculator.bound`): the least slowdown of any partition of that
step into the policy's blocks, on the products whose two operands both depend on the update. `trains` is read from
the trace, the parameters the step writes over the start's: an entry that trains less charges fewer products, so its
slowdown is not its optimizer's. Where a step has dense linear algebra (Shampoo's roots), which the capture counts at
n³ MACs a matrix and not as the gates of the algorithm that computes it, the row says what share of the step's gates
that count is. Below the rows it prints each statement with what it rests on, each entry's planned run at its
slowdown (the config's steps times the traced step's gates) as a share of one datacenter's and of the world's compute
over a window, a week, a month and a year, and which parts are traced, approximated, proved, premises and scaled;
`--out` writes all of it to `DIR/library.json`. A row has no slowdown, and says why, where the capture
refuses the step, where the bound can't read it, and where the entry's E keeps less than the trained weights: a
weight then costs at most E's bits a trained weight, which only a run's record has, so the grade reads that entry.

## Running an attack

~~~text
verity bench training-slowdown NAME|CONFIG.toml [--dry-run] [--local [--root DIR] | --on MACHINE [--pool NAME]] \
    [--weights]
python -m training_slowdown.run --config CONFIG.toml --seed 0 --split search|final --out OUT [--starts-root DIR] \
    [--device cuda] [--keep-weights]                    # one job
python -m training_slowdown.run --retrace RUN_JSON --out OUT   # a record with its trace taken again
python -m training_slowdown.capture.circuit STEP.npz    # a saved step circuit's quantities
python -m training_slowdown.gates [--check]             # gates per multiply-add by op; --check compiles the Definitions
~~~

An attack is a config of `attacks/` (`config.py`'s docstring has the schema): a start, a route and its dials, the
budget, the seeds, and what the grade reads, the target gain, the policy (`console`, policy.py's rule, or a
`[policy.*]` table of `verity/protocols/inference_only/rho/data`) and the honest config. The command (`run.py` here) expands a config and its honest config into
jobs, one per seed and split, each labelled in the store `training-slowdown=<digest>/<seed>/<split>`, and runs the ones
its labels lack: on the queue (kind `training-attack`, `infra/cluster/kinds/training.toml`), on one machine with `--on`,
or here, one after another, with `--local`. It exits 0 when every job has its record, 3 while one is pending and 1 when
one failed; a failed job is never run again.

A job trains the config's start along its route and scores each checkpoint on its split. A route is a module of
`routes/` with `VERSION`, `DIALS`, `build`, `step` and `plan` (the route contract, `routes/__init__.py`; a dial its
`LATER` names is out of a config's digest while at its default, so a route gains a dial and no digest moves); the harness
owns the loop, the batches, the checkpoints and each checkpoint's E, the weights that differ from θ encoded by the
route's plan. A checkpoint's output model, the one scored, is what E rebuilds from θ: the trained weights under a
plan that keeps them whole, and the model the kept bits give under one that keeps less. Everything a route computes
through torch is counted; the harness's own work (loading, evaluating, encoding) isn't. A route that names anything of `training_slowdown` but the codecs and the model,
or a route around the dispatcher (`run.FORBIDDEN`), is refused, and one that uses an op the counter can't price, reads
token data out of torch, starts a thread or a process, or imports a module the screen refuses makes its run invalid
(`run.py`'s docstring has the screen).

A route's code runs in a process of its own, inside the job's sandbox (`sandbox.py`): no network interface is up and
each registered final set's directory is empty there. That process stops before it imports the route if it finds
either reachable, or holds a capability that could undo them, as root's does, so on a machine that can't make the
sandbox a job fails with the reason and trains nothing. The
job's own process reads the held-out split and scores each checkpoint's state as the route's process hands it out;
no score goes back in. The sandbox hides the paths the registry names and nothing else of the file system.

After its last step a job traces its route's step once, outside the count, and its record's `trace` holds what the
capture measures of it for the grade: the token axes, the forward's updated MAC nodes, and the product rows unpriced
at each k of the console's dial (`capture.products.raw_on_dial`), which `capture.products.priced` prices where they
are read. The capture takes the run's last three steps in one graph and measures the middle one (`capture_steps`),
so the measured step reads what the step before it wrote and the step after it reads what it makes; where the three
aren't the run's first steps, the first of them reads the run's own weights, not θ's (`mid_run`). That gives the
step a circuit, which the trace keeps as the capture measures it (`circuit`: the gradient's cut and a token group's
partial, the carried state, τ, a, the update's profile, and the shares of its gates in token ops and in ops an update
reaches, at the gate prices of the code that took it), with that code's version (`capture_code`). A step the capture
refuses is recorded as refused, and the job stays valid.

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
config's records as its jobs wrote them, `DIR/<seed>/<split>/run.json` (`split` is `search` or `final`). The grade
reads no file of a served run. It prints one block and writes `OUT/slowdown.json` (`slowdown/v1`):

- **the gain** each seed reached, through the fit of the starts on the start's data (loss to effective compute);
- **H**, the least counted |C| reaching the target gain: over the honest config's records, and over every config's
  given (`--also` adds configs at the same start and gain);
- **p**, the attack's |C| over H, against the honest route and against any route, each with its range over the seeds;
- **ρ**, the attack's own cost at the config's policy: `formalism.published` at (c, k) on the search record, read
  with its `trace`, whose raw product rows the grade prices with `capture.products.priced` (it holds no price);
- **the sensitivities** beside ρ and never in it, reduction and update, from the trace's `circuit`, and the version
  of the code that took the trace. A record whose trace has no `circuit` or no `capture_code` still grades: ρ reads
  the rows either way (the circuit's τ stands in for a stub, which moves ρ only where it changes how many steps a
  block's inputs cover), and the block says which of these figures are blank, why, and how the trace is taken again;
- **S = p·ρ**, with what each factor rests on, and each seed beside it: whether it reaches the gain, and its |C|;
- **the bound of record** at the same policy: the least slowdown of any partition of the attack's traced circuit, by
  the theorem's count solved for the fewest blocks (`min_cut.blocks_at_least`, times c/|C|). ρ's products term is the
  same theorem's closed form on the same work (`formalism.he_rho`), never above the solved count, so the two are lower
  bounds on one slowdown: the block prints the larger beside S, which keeps ρ's, and never mixes it in. It is "not
  available", with what is missing, where the search record gives the bound no traced step to read;
- **per period**: the run's AND gates at S and their share of one datacenter's and of the world's compute over a
  window, a week, a month and a year, with the measured, proved and scaled parts named.

A config's `policy` is a name, and `policy.named` resolves it: `console` is the rule relative to the start (c one
session of the reference model of the start's size, k the session's input ids), and any other name is a fixed policy
of the calculator's registry (`verity/protocols/inference_only/rho/data`, its `[policy.*]` tables). The block says which kind it was, and a
name that is neither is refused. p is read on the final set where the attack and its honest config each have a final
record, else on the search split, and the block says which. A search record with no trace, with one the capture
refused, or with one of a capture version the tree's `priced` doesn't read has no ρ, and then there is no S: the
reason names the job that takes the trace again at the record's commit and `config.text` (`--retrace`, above, which
takes it with the capture of the tree it runs in), or says the record has no config to re-trace from. A record's
`provenance`, where it has one (no job writes it), is passed on unread, beside the record and in what S rests on.

## Limits of the count

- Work outside torch's dispatcher (custom CUDA or Triton kernels, host libraries) isn't counted: the screen refuses the
  usual routes by name, and the flags show work the count can't vouch for. Torch compilation is disabled and CUDA graph
  replay refused.
- The measured quantities are one traced step's: a record's updated steps take its representative step's structure,
  scaled to their tokens. A cut counts a value at its dtype's width, and values are whole tensors.
- A job traces one step, one that reads an update. The steps before the run's first update read θ alone and are not
  traced, so they have no Δ_grad and Fit's check on the training holds at no (c, k): the grade's ρ is never Fit's.
  It is the products term where that exceeds 1, else 1. The reduction and update sensitivities sum the steps that
  have the traced step's cuts, and the block says so where an untraced step's token work fills two blocks or more.
- Bit-for-bit rebuilds need the run's kernels: families are checked on the run's device, right after it trains.

The suite: `uv run tools/verity/check/suites.py benchmarks/training_slowdown`.
