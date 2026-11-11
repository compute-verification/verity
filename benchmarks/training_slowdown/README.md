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
| `policy.py` | a start's reference shapes and the (c, k) grid; the console's policy (`CONSOLE`), one session of the 45M reference model |
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

## Limits of the count

- Work outside torch's dispatcher (custom CUDA or Triton kernels, host libraries) isn't counted: the screen refuses the
  usual routes by name, and the flags show work the count can't vouch for. Torch compilation is disabled and CUDA graph
  replay refused.
- The measured quantities are one traced step's: a record's updated steps take its representative step's structure,
  scaled to their tokens. A cut counts a value at its dtype's width, and values are whole tensors.
- Bit-for-bit rebuilds need the run's kernels: families are checked on the run's device, right after it trains.

The suite: `uv run tools/verity/check/suites.py benchmarks/training_slowdown`.
