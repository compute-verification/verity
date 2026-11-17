# Training slowdown

This is the harness of the paused attack search. An attack is one way of training a model: a route and its dials,
from a registered start. The harness trains it, counts every multiplication the run computes through torch, traces
one training step, and reads the run's slowdown at a policy off that step. The search scores a config by S:
its counted gates to a target gain over its honest config's, times that slowdown.

What it gives is the attack search's own measurement of a run, not the slowdown's definition. The slowdown is the
Glossary's. Nothing runs the attack search
today: its commands are documented as they are, and none of them is a routine step.

## Try it

The library of training algorithms traces each entry's step on a CPU, on fake tensors, and reads its slowdown under
its policy. It trains nothing. From the repository's root:

~~~sh
uv sync --all-packages --extra torch-cpu
uv run verity bench training_slowdown s-t20-1.1x-adamw --library --root DIR
~~~

It prints one line an entry: the share of the start's parameters its step trains, the dtypes its products are priced
in, its slowdown under its policy and what binds. Then it prints the statement each slowdown rests on, the planned run
as a share of one datacenter's and of the world's compute over each period, and which parts are traced, approximated,
proved, premises and scaled. torch logs tracebacks from its fake tensors on the way; the command still exits 0. Its
slowdown is the bound on one traced step at the entry's policy, with no entry trained. It runs no job and reads no
store, and in `DIR` it writes `library.json` and the metric's point, `partial`, with the entry as its `member` and its
start as its `model`, which it prints last. The library's own command (`library.py`, below) takes any entries, and
every entry of `attacks/` given none.

## What is here

- **The library**: the configs of `attacks/`, each a route and its dials at a start, with the policy it is read under.
  Each entry is held to a reference in plain torch.
- **An entry beside its run of record**: an entry that ports a run, held to that run's record by its step.
- **An attack's jobs**: a config and its honest config expanded into jobs, one per seed and split, each run where the
  store lacks its record, its route's code inside a sandbox with no network and no final set.
- **The grade**: the search's score of a config, S, p times the record's slowdown. It is neither the Glossary's slowdown
  nor the slowdown lower bound.
- **The grader** (`grader/`): a run's training step read off the Program the training-step frontend derives for it,
  beside this package's trace of the same step.

Their commands, as the search ran them:

~~~text
uv run python benchmarks/training_slowdown/training_slowdown/library.py [ENTRY ...] [--starts starts.toml] [--out DIR]
python -m training_slowdown.of_record [ENTRY ...] --records DIR [--held of_record.toml] [--starts starts.toml] \
    [--out DIR]
verity bench training-slowdown NAME|CONFIG.toml [--dry-run] [--local [--root DIR] | --on MACHINE [--pool NAME] \
    | --library [--root DIR]] [--weights]
python -m training_slowdown.run --config CONFIG.toml --seed 0 --split search|final --out OUT [--starts-root DIR] \
    [--device cuda] [--threads 8] [--keep-weights] [--state DIR]      # one job
python -m training_slowdown.run --retrace RUN_JSON --out OUT   # a record with its trace taken again
python -m training_slowdown.capture.circuit STEP.npz    # a saved step circuit's quantities
python -m training_slowdown.gates [--check]             # gates per multiply-add by op; --check compiles the Definitions
python -m training_slowdown.program_reading [--size s] [--batch 2] [--ctx 1024] [--vocab 32000] [--iterations 2] \
    [--k 65536] [--out OUT.json]                         # a step's work read off its Program, s-t20's by default
python -m training_slowdown.grade --config attacks/NAME.toml --starts starts.toml --records DIR --honest DIR \
    [--also NAME=DIR ...] [--out OUT] [--as-if-assumed]
~~~

Run `verity bench` from the repository's root and the `python -m` commands from this directory, each in the
workspace's environment, which `uv sync --all-packages --extra torch-cpu` at the root makes: `uv run verity bench ...`
and `uv run --project ../.. python -m training_slowdown.MODULE` (a bare `uv run` here makes an environment of this
directory's own, without the workspace's packages). The grader runs from `grader/`:

~~~text
python grader.py HANDOVER --slowdown-code DIR --commit SHA [--trace STEP.products.json] --out OUT.json
~~~

The suite is
`uv run tools/check/suites.py benchmarks/training_slowdown`.
