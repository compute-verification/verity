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
| `run.py` | one job of an attack config: `Harness`, what a route's code is counted inside, the route loop (`train`) and the record's trace taken again (`retrace`); `block_c`, a start's c |
| `traced.py` | a config's step as the capture measures it: the trace a job records (`trace_step`) and, with the capture's graph, what the library reads (`traced_step`) |
| `config.py` | an attack config: its schema, its digest and its jobs (`Job.label`), each refusal by name |
| `screen.py` | the screen of a route's source: the names a route may not contain (`FORBIDDEN`), read before the route is imported |
| `sandbox.py` | the training job's sandbox: namespaces with no network and an empty directory over each final set, over /run and over the user's tmux directory (`enter`), and what a process can still reach of them (`refusals`) |
| `routes/` | the routes a config names, and the route contract (`routes/__init__.py`); `subset`, the parameters a pattern names trained with the optimizer its `opt` dial names (AdamW, SGD, Lion, Adafactor, Muon, Shampoo; the ones torch lacks in `_optimizers`; under Muon and Shampoo the gains, and the embedding unless the config puts it among the matrices, train under AdamW beside it), a rate each for the matrices, the gains and the embedding, on a linear, cosine or constant schedule, AdamW in torch's default step or its fused kernel, E the tensors in full or their deltas in codes; `ema`, `subset`'s step with the output an exponential average of the weights, at one decay or reaching back a share of the steps so far; `lora`, the start frozen and a low-rank adapter trained beside each matrix a pattern names, the output the start with the adapters merged in; `qat`, quantization-aware training, each matrix a pattern names read as the start's plus the sparse ternary rounding of a latent that Muon trains straight through, the output the start with the last codes, handed over as codes: E is each matrix's scales and codes as one xz stream of its own, at xz's strongest preset and no other |
| `flops.py` | `Counter`, the dispatch-mode count of every op's multiply-adds and their provenance; `guard_exits`, token data read out of torch; `flags`, work the count can't vouch for; `forward_macs_per_token` |
| `opnames.py` | the op-name sets the counter, the capture and the gate prices share |
| `gates.py` | each counted op's price in Boolean gates, from the circuits' Definitions, in the dtype its record names for it (`kind_of`); a dtype no Definition prices is refused (`Unpriced`), and a record that names none is priced by `ASSUMED`. The counter writes torch's dtype, so a float32 GEMM is priced as the exact one; a row whose dtype reads `tf32` is priced as the TF32 one, which holds only for a GEMM outside the term's M, since a trace names no TF32 (`grade.priced_apart`); a record of a run with the TF32 switch on that names a GEMM float32 is refused |
| `annotate.py` | a record's maximal annotation: per step U and D_t, the depth profile and the cut |
| `capture/` | one training step traced on fake tensors (`capture_step` traces any PyTorch step, `capture_steps` a step among the steps before and after it, with its step circuit): |C|, the update's dependence and the cuts (`capture.circuit`) and the step's matrix products as rows (`capture.products`) |
| `work.py` | the work a step's record states for the theorem of record: M, the products whose two operands both depend on the update, from a trace's product rows (`priced`, `products_of_trace`, `he_of_rows`), and each reason a record states none. It computes no slowdown |
| `formalism.py` | a record's quantities (`of_record`, `measure`), Fit at (c, k) (`at`), the one bound's products term, which is the entry point's, and its sensitivities, Slowdown's token form among them (`bound`), and the published ρ with what it rests on (`published`) |
| `as_math.py` | a record that holds a fused attention kernel as one row, rewritten as the counter writes the kernel: `python -m training_slowdown.as_math SRC DST [--partial none\|SUBSET:CALLS]`. Which gradients a backward call asked for is read from the record's trace; where the trace doesn't show it the rewrite is refused unless `--partial` says, and a `--partial` the trace contradicts is refused. One case the trace leaves open: k's gradient alone in c calls leaves the products of k's with q's, or with v's, in 2c, so there the rewrite is refused naming the readings and `--partial` picks one. Such a record names no dtype and its rewritten rows name none; a fused row that names one is refused, since the counter that names dtypes writes the kernel's ops itself |
| `policy.py` | a start's reference shapes and the (c, k) grid; the console's policy (`CONSOLE`), one session of the 45M reference model; `named`, the one place a config's policy name becomes (c, k) |
| `grade.py` | the grade: a config's slowdown S = p·ρ from its jobs' records and its honest config's, printed and written as `slowdown.json` (below) |
| `library.py` | the library's table: each entry of `attacks/` with its step traced on a CPU, its slowdown at its policy (the entry point's on that step) and what binds, and its planned run over each fleet and period |
| `of_record.py`, `of_record.toml` | an entry beside its run of record: the entry's counted step set beside the run's record's (`compare`), equal, or what differs, each expected difference named by its rule; the data file names each held entry's run, the record's artifact, its path and its sha256, the gradients its attention's backward asked for where the record can't say, and the lines of the record's own program a route doesn't compute |
| `calculator.py` | the slowdown calculator (`verity/protocols/inference_only/rho`) as this package reads it, and the one module here that does: the benchmark's one slowdown (`slowdown`, below), a fixed policy's numbers, the fleets and periods a figure is presented over and the rows of β(k) |
| `counterexample.py` | a run's counterexample families: their writer and the sampled bit-for-bit rebuild check |
| `codecs/` | update codes: identity, low rank, sparse mask, quantized delta, scaled quantized delta, sparse ternary, seed replay and their composition; a codec packs a weight from its value (`encode`) or from the parts a program trained it in (`encode_trained`) |
| `starts.py`, `starts.toml` | the starts registry: each start by its store artifact, its files' sha256, its shape, C0 and loss, and its data's pins; `load` reads an installed start and checks it; `parts_read`, the parts of the stream its start.json names as read by its own training |
| `data.py` | a start's data: the training stream (`Stream`, by the job's data seed; the fresh stream, without the windows of the parts a start read, `read_windows`) and the held-out sets (`held_out`: the search pool and the final set), checked against the registry; the search slice from a live coin (`sample_slice`) |
| `evaluate.py` | the harness's evaluator: a checkpoint's bits per byte on the job's split, the search pool's paired with θ on the same slice |
| `model.py` | the starts' architecture, the lf dense decoder |

The reference shapes are the lf model's (`adapters/vllm/tests/program/train_step_lf.py`), which the suite checks.

## The one entry point

`calculator.slowdown(policy, step, run=None)` is the one place the benchmark computes a slowdown: the grade (through
`formalism.bound`) and the library's table call it, and `tests/test_one_calculator.py` fails on a second one. `policy`
is a block's `c_gates` and `k_bits`; `step` is a step's record, a job's trace (`run.trace_step`'s) or the products
stated from one (`work.products_of_trace`'s); `run` adds the run's `gates` (|C|; else the trace's own circuit's),
`tau_bits` and `code_bits`. The work and the circuit it is divided by have one price: a trace's products are priced
in the dtype its circuit counts them in, and a trace whose capture prices some in another (`work.repriced`: one of a
capture before `capture.products.PRICED_AS_TRACED_FROM`, with float32 attention products) has no slowdown over its own
circuit, with the row and the fix in `why`. It returns `slowdown`, the calculator's bound of record on the work the record states (at
least 1; None, with the reason in `why`, where the record gives none); `binds` (`products`, `none` or `unmeasured`);
`term`, the count before the floor; `bound`, the calculator's bound with its theorem, least blocks, premises and the
work it read; `figure`, the choice of that work; and `read`, what the record says of itself.

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

An entry is at a start `starts.toml` registers: `config.load` refuses another, and no entry at a registered start
stands in for a run elsewhere. The search's pilot starts (`s50`, `s100`, `m50`, `m100`, on another tokenizer) are not
registered, so the three keys of the search whose frontier runs are all there have no entry. None is an algorithm the
library lacks: each is a setting of the dials of entries here, the weights and their average.

- `fullmuon-ema`, at `s100`: the dials of `s-t20-1.1x-fullmuon` and `s-t20-1.1x-fullmuon-ema`;
- `fullmuon-fe-ema`, at `s100` and `m100`: those of `s-t20-1.1x-fullmuon-fe-wsd` and `s-t20-1.1x-fullmuon-fe-wsd-ema`;
- `muon-lr`, at `s50` and `m50`: those of `m-t20-1.1x-fullmuon-wsd` and `m-t20-1.1x-fullmuon-wsd-ema`, in two forwards
  a step.

The standard algorithms differ in two ways the table shows apart. AdamW, SGD, Lion, Adafactor and the average of the
weights train every tensor by the same forward and backward passes, so they carry the same products with two operands
the update reaches, and differ only in the gates the optimizer itself adds to the step; gradient accumulation carries
those products in several passes, which write more outputs. Muon and Shampoo with AdamW beside them on the gains and
the embedding (`muon-whole`, `shampoo-whole`) train every tensor by those passes too: the bound is on the same
products, and the optimizer adds gates and products of its own (Muon's Newton-Schulz steps, Shampoo's factors and
roots). The rest change the passes' products. `ref-optim` (Muon) and `shampoo` leave the embedding untrained, and the
output head, whose weight is the embedding, no longer has two operands the update reaches. Frozen layers
(`last-blocks`) leave more products out, and adapters (`lora`) reach a matrix only through a rank-wide product.
Quantization-aware training (`qat-top*`) trains the matrices of the last blocks and nothing else, and what one of its
steps hands the next forward is each matrix's codes, two bits an entry and a scale, where the others hand on the
weight at its width.

The command prints one row an entry (every entry, or the ones named): the share of the start's parameters its step
trains, its slowdown under its policy and what binds. It trains nothing. It builds the entry's model at the start's
shape, traces its step on a CPU as a job does after its last step (`traced.traced_step`, on fake tensors: the job's
program, since a route's step runs under bf16 autocast on a CPU as on a GPU), prints the dtypes that step's circuit
prices its matrix products in (`priced in`), and reads the
slowdown on the traced step from the one entry point (`calculator.slowdown`): the least slowdown of any partition of
that step into the policy's blocks, on the products whose two operands both depend on the update. `trains` is read from
the trace, the parameters the step writes over the start's: an entry that trains less charges fewer products, so its
slowdown is not its optimizer's. Where a step has dense linear algebra (Shampoo's roots), which the capture counts at
n³ MACs a matrix and not as the gates of the algorithm that computes it, the row says what share of the step's gates
that count is. Below the rows it prints each statement with what it rests on, each entry's planned run at its
slowdown (the config's steps times the traced step's gates) as a share of one datacenter's and of the world's compute
over a window, a week, a month and a year, and which parts are traced, approximated, proved, premises and scaled;
`--out` writes all of it to `DIR/library.json`. A row has no slowdown, and says why, where the capture
refuses the step, where the bound can't read it, and where the entry's E keeps less than the trained weights: a
weight then costs at most E's bits a trained weight, which only a run's record has, so the grade reads that entry.

## An entry beside its run of record

~~~text
python -m training_slowdown.of_record [ENTRY ...] --records DIR [--held of_record.toml] [--starts starts.toml] \
    [--out DIR]
~~~

An entry that ports a run is held to that run by its step, not by its weights, E's bytes or its loss, which only a
run has. `of_record.toml` names each such entry's run of record: the run, the store artifact holding its record, the
record's path there and its sha256; a run is named there and nowhere else. `--records DIR` holds the records, each at
`DIR/RUN/PATH`, as `research data fetch ART --to DIR/RUN --path PATH` leaves one; a record that is missing, or is
another file, is refused with that line.

For each entry (every held one, or the ones named) the command counts two steps of the entry's route under the
harness's counter, on a CPU, at the model's own initialisation and on random tokens, one row a forward, and sets the
second beside the step structure most of the record's steps have, the record's rows a forward divided out. The count
reads operand shapes and which weights have left θ, never a value, so it is the step a job of the entry takes
wherever that step is one fixed graph: the step is traced too, and an entry whose trace the capture refuses, or
whose step reads a value back, is refused. It prints, an entry:

- `updated`: the uses of an updated weight, each by its MACs a token, the updated MACs upstream of it and the bits
  it reads its weight at, which must be the record's (a route casts to bf16 on a CPU as on CUDA); this list is the
  entry's update structure;
- `log`: the step's products, each by phase, kind, op, operand shapes, count and MACs, which must be the record's;
- `attention`, for a record that holds attention's fused kernel as one row: the counter writes the kernel as the
  ops the math kernel computes attention by, so the record's rows are rewritten the same way before they are
  compared (`as_math`). Such a row doesn't say which gradients a backward call asked for. That is read from the
  record's trace where it holds one; otherwise the entry states it, `partial` in `of_record.toml` (`none`, or
  SUBSET:CALLS, as `as_math` reads `--partial`), and the line says it is the data file's word and not the record's.
  It is never assumed: an entry whose record needs it and has none is refused, naming the key. A stated ask that is
  wrong writes products the entry's step doesn't have, so it shows as a difference;
- the start and the steps, which must be the record's, and E: its plan's kinds, each of which must be one of the
  record's format's (`E_KINDS`), with its size after the two steps beside the record's last checkpoint's, and how a
  codec packs the same encoding another way than the record's program did (`E_PACKING`);
- `expected`: a difference named by the rule that covers it, with the MACs on each side. A rule fires on its exact
  case alone. The device's: the norm as elementwise ops where CUDA's fused kernel is counted at an updated gain
  alone, and not at all in a step that trains no gain (`norm`). The program's own
  arrangement of a computation the route takes too: the loss (`loss`), where the record's program means each row
  and divides the rows' sum by the rows kept and a route means over the tokens, one value and one gradient where
  the run drops no row, which the rule checks in the record's program's arguments; and the average of the weights
  (`average`), where the record's program, its arguments stating one, averages every tensor of the model in one op,
  those it never trains too, and route `ema` each tensor it trains. AdamW's kernel is no rule's: an entry of a run
  whose program asked torch for the fused kernel sets `fused`, and a step in torch's default is a difference;
- `named`: a line of the record's own program that the route rightly doesn't compute, which no rule knows: the
  entry's `record_only` in `of_record.toml` names the line, the count and MACs the record's step has of it over the
  config's, and why. A named line that the record's step does not have over the config's as named is a difference;
- `DIFFERS`: anything else.

The command exits 1 if any entry differs or is refused: one whose step the capture refuses or that reads a value
back, and one whose record's fused attention rows `as_math` can't rewrite. `--out` keeps the comparisons
(`DIR/of_record.json`) and each entry's counted record (`DIR/ENTRY/run.json`).

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
budget, the seeds, the windows it asks for where it asks, and what the grade reads, the target gain, the policy (`console`, policy.py's rule, or a
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
(`--state`): the model its route trains (under a route's `output`, `ema`'s or `lora`'s, not the output model a
checkpoint scores), the model's buffers outside its state dict, the optimizers' state, the stream's position, the
generators, and the record so far with its scores and its clock. Told to stop (SIGTERM: an eviction, or the end of
its run's time) it first writes a `run.json` that says so (`stopped`: "told to stop", with no step or checkpoint),
then ends the route's process and leaves the state. The command reads a stop from that record and never from the run's exit status, which is whatever stopped the
job reports (a lease out of time: `timeout`'s 124, whatever the job's own): a run that didn't end with 0 is stopped
when its record says so, and failed when it holds no record or one that doesn't. A job told to stop before its own
process is up has written none, and is failed. The command's next run starts a stopped job again, and the job
continues from the state when it is its own (the same
config digest, seed, split and `training_slowdown` VERSION, a model that hashes to the record's last checkpoint) and
refuses any other, naming what differs. The job's own process states in the record so far, for the segment that
kept the state, the SHA-512 of each of its parts (the model, its buffers outside its state dict, the optimizers'
state, the stream's position, the generators); the job that continues hashes them again, in the state directory and
as the route's process holds them before it imports the route, and refuses a state a part of which hashes to
anything else. That record lies beside
the tensors, so the command ties it to the stopped run's record on the store, which no job writes: it tells the
job the sha256 the store lists for that record's `state.json` (`--continues`), or that it found none, and the job
continues from no other state (a state on a node's disk has no such copy, and none is checked). The record lists
the segments
the job ran in, and its count is their sum: every step once, and each segment's `build`. What a stopped segment
computed after its last kept checkpoint is run again and is in no record. A route's `output` at the checkpoint a job
stopped at is counted outside the steps (`setup`), where an unbroken job counts it in the step after. A job with
updated MACs or tokens outside a step is invalid, so an `output` that computed on token data with an updated weight
would make a continued job invalid: it makes the unbroken job invalid too, whose last checkpoint's output is counted
after its last step, unless it computes so at an earlier checkpoint and not at the last. That is a limit, and no
route here meets it: `ema`'s output computes nothing and `lora`'s multiplies each adapter's two matrices. A state is
a cache: the command keeps it with the stopped run's output on the store, the next run fetches it on whichever machine it gets (`NODE_STATE` in
`run.py` here places it on the node's disk instead), and losing it costs a start from θ. The state directory is
hidden from the route's process as a final set is, and only the job's own process writes it. A state is at most three
times its model (the weights and two buffers of the optimizer for each), and a run holds three states at once while
it keeps one, and a fourth, the store's own copy, when it fetched the one it continues from: the command asks the
queue for that much disk from the start's parameters, and 20 GB for the rest (`run_disk_gb` in `run.py` here).

Which windows of the start's data a job is served is the harness's, and a config's to ask for, never a route's. By
default a job is served every window, in its seed's order. A config with `windows = "fresh"` is served that order
without the windows the start's own training read: every window of every part of the stream the start's `start.json`
names (`provenance.data.parts`, each part by the sha256 the stream's manifest lists it under), which is more than
the start read where its training stopped inside a part, and never less. The job refuses the config, saying what to
write where, on a start whose `start.json` names no part or names one the stream doesn't hold, and where fewer
windows are fresh than a step takes. Its record states what it was served (`windows`): the parts left out, by
sha256 with the ids of their windows, the ids of the windows of each step (a window's id is its place in the stream,
part by part in the manifest's order), and how many times the fresh windows ran out and their order started again,
so whether a job read a window its start read is a comparison of ids. A stopped job that continues is served the
windows the unbroken job is, and its record states them over every segment. `windows` is in a config's digest only
where the config asks, so a config that doesn't is the job it was, under the same label.

A job trains the config's start along its route and scores each checkpoint on its split. A route is a module of
`routes/` with `VERSION`, `DIALS`, `build`, `step` and `plan` (the route contract, `routes/__init__.py`; a dial its
`LATER` names is out of a config's digest while at its default, so a route gains a dial and no digest moves, and a
config writes one dial of a group its `EITHER` names); the routes here run their forward under bf16 autocast on every
device (`routes/subset.py`'s `step` and `routes/qat.py`'s), so a job on a CPU runs the program a job on a GPU does; the
harness
owns the loop, the batches, the checkpoints and each checkpoint's E, the weights that differ from θ encoded by the
route's plan. A checkpoint's output model, the one scored, is what E rebuilds from θ: the trained weights under a
plan that keeps them whole, and the model the kept bits give under one that keeps less. A route that trained a
weight in a codec's own form hands over those parts (`qat`, its codes), the plan's codec packs them as they are, and
the weight scored is that codec's decode of them on θ; no route brings a codec. Everything a route computes
through torch is counted; the harness's own work (loading, evaluating, encoding) isn't. The count calls a forward
product updated when it reads a weight that has left θ. A buffer of the model that θ doesn't hold (a position table, in
the state dict or not) is held at the value it has when the model is registered and checked like θ's weights: a product
with it is θ's, a buffer a step writes is an updated weight from that write on, under its own name, and one written
where the counter can't see makes the run invalid, as for a weight. |C| counts the circuit a trace of the step holds:
each product by its shape and each elementwise multiplication at one MAC an element (`flops.py`), and a fused attention
kernel as the ops the math kernel computes attention by, which are the ones a trace runs: the forward's two scalings and
two products, and in the backward the products and scalings of the gradients the step asks for. A job's record is the
same under either kernel in its ops, their shapes and their MACs, and its counted step is its trace's in MACs
(`tests/test_run.py`). Each row of a record's op
log names the dtype its op computed in, its result's (`flops.dtype_of`): its first operand's for an op that returns
nothing (an in-place list form, a fused optimizer step) and for `_scaled_mm`, whose result is cast out of the FP8 it
computes in. The op's gates are priced in that dtype, never by its phase or its class (`gates.kind_of`), and the trace's
circuit prices each node by the same rule. So a counted step is its trace's circuit in gates, attention's products and
scalings excepted: a job runs them under a fused kernel, in its operands' dtype, bf16 under the routes' autocast, and
its rows name that; a trace runs them under the math kernel, which casts half-width operands to float32, and its
circuit holds that. The counted step is under the circuit by those ops' MACs at the two dtypes' prices and by nothing
else, and a job run under the math kernel names them float32 as the trace does (`tests/test_run.py`). A dtype no
Definition prices has no price and none is assumed for it:
float64, an integer product, a list op over tensors of two dtypes. The grade refuses a record that holds such an op and
the capture a step that does (`unpriced_dtype`), each naming the op and the dtype. A fused optimizer kernel is one op,
priced at a constant per parameter (`opnames.FUSED_OPTIMIZER_MACS`, which the counter and the trace both read). Choosing
a fused kernel never lowers a step's count, at any setting of the optimizer's options (`tests/test_flops.py`): AdamW
counts 8 multiplications a parameter and Adam 7, where their unfused steps make 6 at most, and SGD and Adagrad count 4,
their unfused steps' largest (SGD's with momentum, weight decay and either Nesterov momentum or dampening; Adagrad's
foreach step with weight decay). The one exception is at most one multiplication a step: Adagrad's unfused step with a
tensor learning rate and weight decay divides the rate itself. A route that names anything of `training_slowdown` but
the codecs and the model, or a route around the dispatcher (`screen.FORBIDDEN`), is refused, and one that uses an op the
counter can't price, reads token data out of torch, starts a thread or a process, or imports a module the screen refuses
makes its run invalid (`screen.py` has the screen).

A route's code runs in a process of its own, inside the job's sandbox (`sandbox.py`): no network interface is up and
each registered final set's directory, and the job's state directory, is empty there. So are /run and the directory
of the user's tmux server, where the sockets are through which a process has another started outside the sandbox (the
user's systemd manager, its session bus, tmux). That process stops before it imports the route if it finds
any of them reachable, or holds a capability that could undo them, as root's does, so on a machine that can't make the
sandbox a job fails with the reason and trains nothing. The
job's own process reads the held-out split and scores each checkpoint's state as the route's process hands it out;
no score goes back in. Where the job's run holds a key to the store (`research run --store-read`, which a job that
continues fetches its state with), the route's process gets none of it: it is started without the names the key and
the store's configuration arrive under (those that start `AWS_`, and `RESEARCH_STORE_CONFIG`), and the directory the
configuration is in, where the key lies in a file beside it, is hidden as well. The job's own process and its shell
keep the key. The sandbox hides those paths and nothing else of the file system: a socket at another path stays
reachable.

After its last step a job traces its route's step once, outside the count, and its record's `trace` holds what the
capture measures of it for the grade: the token axes, the forward's updated MAC nodes, and the product rows unpriced
at each k of the console's dial (`capture.products.raw_on_dial`), which `capture.products.priced` prices where they
are read, each in the dtype it was traced in. The capture runs attention on torch's math path, so a step under
bf16 autocast has its attention products in float32, as that path computes them from upcast operands and as the
train-step circuit wires them; its other products are bf16. A trace of a capture before
`capture.products.PRICED_AS_TRACED_FROM` holds the same float32 products, and that capture's rule prices them in
bf16 (`capture.products.priced_dtype`); the rule goes when the stored records it reads are traced again. The
capture takes the run's last three steps in one graph and measures the middle one (`capture_steps`),
so the measured step reads what the step before it wrote and the step after it reads what it makes; where the three
aren't the run's first steps, the first of them reads the run's own weights, not θ's (`mid_run`). That gives the
step a circuit, which the trace keeps as the capture measures it (`circuit`: the gradient's cut and a token group's
partial, the carried state, τ, a, the update's profile, and the shares of its gates in token ops and in ops an update
reaches, at the gate prices of the code that took it), with that code's version (`capture_code`). A tensor an
optimizer holds and the model doesn't own (a latent, a second copy of a weight) is traced as the optimizer's, under
`optimizers.I.params.J`. A step the capture refuses is recorded as refused, with the reason, and the job stays valid:
among them a step that reads a tensor of the run it reached around its arguments, whose refusal says where to hold it.
A step that deep-copies the model, a parameter or an optimizer is refused so (`deep_copy`): the trace runs on tensors
without data, and a step that needs the weights as they were copies the state dict.

A job has its record whether or not its trace is taken. The route's process writes the record whole before the
trace and again with it, so a trace that raises, or that ends the process, leaves the count, the digests and the
scores, with the reason where the trace would be (`trace.failed`: the error, or the signal or status the process
ended with). Such a job is as valid as its count, exits as one whose trace was taken and says on stderr that it has
none; the grade reads its ρ as unmeasured, with the reason.

A job's output is its record, `OUT/run.json`, which its own process writes once, when the route's process has ended,
with the scores in. It holds no weights by default: beside the record, its last checkpoint's E when E is under
50 MiB. The command's `--weights` makes each job it runs keep its last checkpoint's whole state too.

A job on a CPU computes with oneDNN held below AMX (`ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`, `run.limit_cpu`): each of
its two processes sets the limit itself, over whatever it was started with, before torch computes anything, because
oneDNN's AMX kernels can return a wrong bf16 product on a virtual CPU and a step run twice then gives two results.
The record names the variable and the value the route's process set (`cpu_isa`); on a GPU a job sets nothing and
the field is null.

Run the `python -m` commands from this directory. A start and its data are read where `starts.toml` says a node holds them, and checked
against it; elsewhere, `research data fetch ART --path starts/NAME/theta.safetensors` (and its `start.json`) gets a
start, and `--starts-root` points at it.

## Grading a config

~~~text
python -m training_slowdown.grade --config attacks/NAME.toml --starts starts.toml --records DIR --honest DIR \
    [--also NAME=DIR ...] [--out OUT] [--as-if-assumed]
~~~

A config names its `start`, its `target.gain`, its `policy`, its `honest` config and its `seeds`; each DIR holds one
config's records as its jobs wrote them, `DIR/<seed>/<split>/run.json` (`split` is `search` or `final`). The grade
reads no file of a served run. It prints one block and writes `OUT/slowdown.json` (`slowdown/v1`):

- **the gain** each seed reached, through the fit of the starts on the start's data (loss to effective compute);
- **H**, the least counted |C| reaching the target gain: over the honest config's records, and over every config's
  given (`--also` adds configs at the same start and gain);
- **p**, the attack's |C| over H, against the honest route and against any route, each with its range over the seeds;
- **ρ**, the attack's own cost at the config's policy: `formalism.published` at (c, k) on the search record, read
  with its `trace`, whose raw product rows are priced with `capture.products.priced` (the grade holds no price);
  where Fit fails it is the one entry point's slowdown on those products;
- **the sensitivities** beside ρ and never in it, reduction and update, from the trace's `circuit`, and the version
  of the code that took the trace. A record whose trace has no `circuit` or no `capture_code` still grades: ρ reads
  the rows either way (the circuit's τ stands in for a stub, which moves ρ only where it changes how many steps a
  block's inputs cover), and the block says which of these figures are blank, why, and how the trace is taken again;
- **S = p·ρ**, with what each factor rests on, and each seed beside it: whether it reaches the gain, and its |C|;
- **the bound of record** at the same policy: the least slowdown of any partition of the attack's traced circuit, by
  the theorem's count solved for the fewest blocks, times c/|C| (the calculator's `min_cut.bound`, the `bound` of
  `calculator.slowdown`'s result). ρ's products term is this bound, so where products bind ρ is it and S is p times
  it. It is "not available", with what is missing, where the search record
  gives the bound no traced step to read;
- **per period**: the run's AND gates at S and their share of one datacenter's and of the world's compute over a
  window, a week, a month and a year, with the measured, proved and scaled parts named.

A config's `policy` is a name, and `policy.named` resolves it: `console` is the rule relative to the start (c one
session of the reference model of the start's size, k the session's input ids), and any other name is a fixed policy
of the calculator's registry (`verity/protocols/inference_only/rho/data`, its `[policy.*]` tables). The block says which kind it was, and a
name that is neither is refused. p is read on the final set where the attack and its honest config each have a final
record, else on the search split, and the block says which. A search record with no trace, with one the capture
refused, or with one of a capture version the tree's `priced` doesn't read has no ρ, and then there is no S: the
reason names the job that takes the trace again at the record's commit and `config.text` (`--retrace`, above, which
takes it with the capture of the tree it runs in), or says the record has no config to re-trace from. A record
whose trace was not taken (`trace.failed`) has none either, and its ρ is "unmeasured", not "refused", with why the
trace failed and the same job: the step was not refused, the process that traced it ended or raised. A record's
`provenance`, where it has one (no job writes it), is passed on unread, beside the record and in what S rests on.

A record is counted in the dtypes its op log names. One whose log names none is counted by the assumed dtypes
(`gates.ASSUMED`: products and forward or backward elementwise ops in bf16, optimizer and decomposition ops in
float32), and the block says of each record which it was and the device it ran on. A record whose log names a dtype
in some rows and none in others is refused, with how many rows of each kind it has: no job writes one, and it would
be counted two ways. Records of both kinds give no p
and no S, since C over H would compare a count in one pricing with a count in the other; the reason names the
records and the two ways out. One is to count the configs whose records name no dtype again, with a job of this
version. The other is `--as-if-assumed`, which counts every record by the assumed dtypes whatever its log names: its
block says so in its first two lines and begins every other line with `as-if`, and it writes
`OUT/slowdown.as-if.json` (`slowdown-as-if/v1`), never `slowdown.json`. An as-if figure is no config's slowdown.

ρ divides the work of the search record's traced products by the record's count, so the two have one price too: a
product the term's M holds is priced, in the trace, in the dtype the record's op log names for it. A record whose
trace prices one in another (`grade.priced_apart`, which matches a product to the log's ops by its shape and by
whether it is the backward's) has no ρ, and then there is no S. The reason names the product, the two dtypes and the
way out: for a product its capture's rule prices in another dtype than it was traced in, the job traced again at
capture `capture.products.PRICED_AS_TRACED_FROM` or later, which prices each product as traced; for one priced as
traced, whose traced step ran it in another dtype than the job did, the job traced again on a device of the kind the
record states. torch's autocast casts `mv`, `addmv` and `addr` on a card and not on a CPU, so a card's job whose M
holds one is refused so when it is traced again on a CPU: it is traced again on a card. A record whose log names no
dtype has none to differ from.

A record states torch's switch for TF32 matrix products as it stands when the record is written (`run.tf32`):
`torch.backends.cuda.matmul.fp32_precision`, which reads it however a job set it, or, from a torch without it,
`torch.backends.cuda.matmul.allow_tf32` and `torch.get_float32_matmul_precision()`. The block says of each record
whether it was on, off or not stated (a record written before one stated it, which is read as before). No dtype of
torch's says a product ran in TF32, so a record that states the switch on while its op log names a GEMM float32 is
refused: the reason names the ops and the way out, the job run again with the switch off. Those rows named `tf32`
would be priced as the TF32 GEMM, but only a GEMM outside the term's M can be: a trace names no TF32, so a product
of M named `tf32` is priced apart from its trace, and ρ is refused.

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

## The grader

`grader/grader.py` reads a run's training step off the Program the training-step frontend derives for it
(`step_program`, in `adapters/vllm/verity_vllm/program/frontend/train_step.py`), beside this package's trace of the
same step. From `grader/`:

~~~text
python grader.py HANDOVER --slowdown-code DIR --commit SHA [--trace STEP.products.json] --out OUT.json
~~~

`HANDOVER` is the directory `grader.write_handover` writes: `handover.json` and the step's Program at two batch shapes.
`--slowdown-code` is the directory this package sits in, which the grader loads where it sits and, in a git checkout,
holds to the commit named. It writes `OUT.json`: the step's matrix products as rows, with their multiply-adds and output
wires, the circuit's size |C| at the Boolean prices of the primitives it calls and, given `--trace`, the same rows of
this package's capture beside them. It computes no ρ. Its tests are a suite of their own:
`uv run tools/verity/check/suites.py benchmarks/training_slowdown/grader`.

## What it reads, what it writes and what loads it

- **Reads.** Each suite names what it reads outside its directory among its `inputs` (`pyproject.toml`,
  `grader/pyproject.toml`). From `verity`: the circuits' Definitions the gate prices compile, and the codec, the
  census and the Boolean prices the grader reads with. From `adapters/vllm/verity_vllm`: the frontend's training-step
  module, which derives the steps the grader reads. From the frontend's tests, by path: the lf model. From the
  slowdown calculator (`verity/protocols/inference_only/rho`), its directory put first on `sys.path`: its fleets and
  periods, its registry and its min-cut bound, which `calculator.py` reads, and with the registry the inference-only
  policy's calibrations (`verity/protocols/inference_only/nci/calibrations.json`), which the registry loads. From the
  census (`benchmarks/census/deployments.json`), by path: the reference deployment's window. A run also reads the
  program it is given, the start's θ and, where the start is named in it, the starts registry. The grade reads the
  config, the starts registry and the records it is given.
- **Writes.** The run command writes `OUT/run.json` (the count, the checkpoints, the flags, the program's and θ's
  digests); under `OUT/families/` (or the directory `--families` names), each counterexample family's directory: the
  encodings and `family.json` its program writes, and the check's `verified.json`; and under `OUT/checkpoints/`, the
  weights at the last checkpoint the program's step count plans and, with `--keep-checkpoints`, at every checkpoint.
  The grade, given `--out`, writes `OUT/slowdown.json`. `capture.circuit` and `gates` only print.
- **Loaded by.** Nothing outside this directory: `tests/test_boundaries.py` holds everything outside `benchmarks/` to
  that, and no other benchmark imports it. The grader loads the package from `--slowdown-code`, as a directory on
  `sys.path`, and refuses a second copy in one process.
