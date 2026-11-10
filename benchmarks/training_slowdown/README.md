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
| `run.py` | one job of an attack config: `Harness`, what a route's code is counted inside, and the route loop (`train`); `block_c`, a start's c |
| `config.py` | an attack config: its schema, its digest and its jobs (`Job.label`), each refusal by name |
| `routes/` | the routes a config names, and the route contract (`routes/__init__.py`); `subset`, the parameters a pattern names trained with AdamW |
| `flops.py` | `Counter`, the dispatch-mode count of every op's multiply-adds and their provenance; `guard_exits`, token data read out of torch; `flags`, work the count can't vouch for; `forward_macs_per_token` |
| `opnames.py` | the op-name sets the counter, the capture and the gate prices share |
| `gates.py` | each counted op's price in Boolean gates, from the circuits' Definitions |
| `annotate.py` | a record's maximal annotation: per step U and D_t, the depth profile and the cut; `with_structures`, its traced steps' measured quantities |
| `capture/` | one training step traced on fake tensors (`capture_step` traces any PyTorch step, `capture_fn` a function, `capture_run` a run of them): |C|, the update's dependence and the cuts (`capture.circuit`), the step's matrix products as rows (`capture.products`) and its lowering to the training-step frontend (`capture.lower`) |
| `formalism.py` | a record's quantities (`of_record`, `measure`), Fit and Slowdown at (c, k) (`at`), the one bound and its sensitivities (`bound`), the products term (`he_products`, `he_figure`) and the published ρ with what it rests on (`published`) |
| `policy.py` | a start's reference shapes and the (c, k) grid; the console's policy (`CONSOLE`), one session of the 45M reference model |
| `counterexample.py` | a run's counterexample families: their writer and the sampled bit-for-bit rebuild check |
| `codecs/` | update codes: identity, low rank, sparse mask, quantized delta, seed replay and their composition |
| `starts.py`, `starts.toml` | the starts registry: each start by its store artifact, its files' sha256, its shape, C0 and loss, and its data's pins; `load` reads an installed start and checks it |
| `data.py` | a start's data: the training stream (`Stream`, by the job's data seed) and the held-out sets (`held_out`: the search pool and the final set), checked against the registry; the search slice from a live coin (`sample_slice`) |
| `evaluate.py` | the harness's evaluator: a checkpoint's bits per byte on the job's split, the search pool's paired with θ on the same slice |
| `model.py` | the starts' architecture, the lf dense decoder |

The reference shapes are the lf model's (`integrations/vllm/tests/program/train_step_lf.py`), which the suite checks.

## Running an attack

~~~text
verity bench training-slowdown NAME|CONFIG.toml [--dry-run] [--local [--root DIR] | --on MACHINE [--pool NAME]]
python -m training_slowdown.run --config CONFIG.toml --seed 0 --split search|final --out OUT [--starts-root DIR] \
    [--device cuda]                                     # one job
python -m training_slowdown.capture.circuit STEP.npz    # a saved step circuit's quantities
python -m training_slowdown.gates [--check]             # gates per multiply-add by op; --check compiles the Definitions
~~~

An attack is a config (`config.py`'s docstring has the schema): a start, a route and its dials, the budget, the seeds,
and what the grade reads, the target gain, the policy (`console`, policy.py's rule, or a `[policy.*]` table of
`benchmarks/rho/data`) and the honest config. The command (`run.py` here) expands a config and its honest config into
jobs, one per seed and split, each labelled in the store `training-slowdown=<digest>/<seed>/<split>`, and runs the ones
its labels lack: on the queue (kind `training-attack`, `infra/cluster/kinds/training.toml`), on one machine with `--on`,
or here, one after another, with `--local`. It exits 0 when every job has its record, 3 while one is pending and 1 when
one failed; a failed job is never run again.

A job trains the config's start along its route and scores each checkpoint on its split. A route is a module of
`routes/` with `VERSION`, `DIALS`, `build`, `step` and `plan` (the route contract, `routes/__init__.py`); the harness
owns the loop, the batches, the checkpoints and each checkpoint's E, the weights that differ from θ encoded by the
route's plan and rebuilt bit for bit. Everything a route computes through torch is counted; the harness's own work
(loading, evaluating, encoding) isn't. A route that names anything of `training_slowdown` but the codecs and the model,
or a route around the dispatcher (`run.FORBIDDEN`), is refused, and one that uses an op the counter can't price, reads
token data out of torch, starts a thread or a process, or imports a module the screen refuses makes its run invalid
(`run.py`'s docstring has the screen).

Run the `python -m` commands from this directory. A start and its data are read where `starts.toml` says a node holds them, and checked
against it; elsewhere, `research data fetch ART --path starts/NAME/theta.safetensors` (and its `start.json`) gets a
start, and `--starts-root` points at it.

## Limits of the count

- Work outside torch's dispatcher (custom CUDA or Triton kernels, host libraries) isn't counted: the screen refuses the
  usual routes by name, and the flags show work the count can't vouch for. Torch compilation is disabled and CUDA graph
  replay refused.
- The measured quantities are one traced step's: a record's updated steps take its representative step's structure,
  scaled to their tokens. A cut counts a value at its dtype's width, and values are whole tensors.
- Bit-for-bit rebuilds need the run's kernels: families are checked on the run's device, right after it trains.

The suite: `uv run tools/verity/check/suites.py benchmarks/training_slowdown`.
