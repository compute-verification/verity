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
| `run.py` | `Harness`, the program's API, and the run command (below); `block_c`, a start's c |
| `flops.py` | `Counter`, the dispatch-mode count of every op's multiply-adds and their provenance; `guard_exits`, token data read out of torch; `flags`, work the count can't vouch for; `forward_macs_per_token` |
| `opnames.py` | the op-name sets the counter, the capture and the gate prices share |
| `gates.py` | each counted op's price in Boolean gates, from the circuits' Definitions |
| `annotate.py` | a record's maximal annotation: per step U and D_t, the depth profile and the cut; `with_structures`, its traced steps' measured quantities |
| `capture/` | one training step traced on fake tensors (`capture_step` traces any PyTorch step, `capture_fn` a function, `capture_run` a run of them): |C|, the update's dependence and the cuts (`capture.circuit`), the step's matrix products as rows (`capture.products`) and its lowering to the training-step frontend (`capture.lower`) |
| `formalism.py` | a record's quantities (`of_record`, `measure`), Fit and Slowdown at (c, k) (`at`), the one bound and its sensitivities (`bound`), the products term (`he_products`, `he_figure`) and the published ρ with what it rests on (`published`) |
| `policy.py` | a start's reference shapes and the (c, k) grid; the console's policy (`CONSOLE`), one session of the 45M reference model; `named`, the one place a config's policy name becomes (c, k) |
| `grade.py` | the grade: a config's slowdown S = p·ρ from its jobs' records and its honest config's, printed and written as `slowdown.json` (below) |
| `calculator.py` | the slowdown calculator (`verity/protocols/inference_only/rho`) as this package reads it: a fixed policy's numbers, the fleets and periods a figure is presented over, and the bound of record |
| `counterexample.py` | a program's counterexample families: their writer and the sampled bit-for-bit rebuild check |
| `codecs/` | update codes: identity, low rank, sparse mask, quantized delta, seed replay and their composition |
| `starts.py` | a start's entry in the starts registry the run command reads |

The reference shapes are the lf model's (`adapters/vllm/tests/program/train_step_lf.py`), which the suite checks.

## Running a program

~~~text
python -m training_slowdown.run --program train.py --starts starts.json --start small --tier 0 --seed 0 --out OUT \
    [--evaluator MODULE:FUNCTION] [--device cuda] [-- PROGRAM ARGS...]
python -m training_slowdown.starts --name small --theta THETA.pt --factory MODULE:FUNCTION [--ctx 1024] [--vocab 50304]
python -m training_slowdown.capture.circuit STEP.npz    # a saved step circuit's quantities
python -m training_slowdown.gates [--check]             # gates per multiply-add by op; --check compiles the Definitions
~~~

Run them from this directory. The program contract: a program is one Python file defining `train(te)`, `te` the
`Harness`. It loads θ into its model and registers it (`te.load`), declares its step count (`te.plan`), wraps each
training step in `with te.step():`, and may name its output model (`te.output`) and write counterexample families
(`te.family`). Everything it computes through torch is counted; the harness's own work (loading, evaluating each
checkpoint, encoding) isn't. A program that names the harness's internals or a route around the dispatcher
(`run.FORBIDDEN`) is refused, and one that uses an op the counter can't price, reads token data out of torch, starts a
thread or a process, or imports a module the screen refuses makes its run invalid (`run.py`'s docstring has the
screen).

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
