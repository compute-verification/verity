# `circuit-check`

`circuit-check` is the property checks every circuit passes whenever it is built or changed: a Definition, a subcircuit
template or one of its subcircuits, a served workload's request Program, and their Boolean lowerings. A change that adds or
modifies a circuit puts its report in the PR, and `check` runs `circuit-check --all`.

## Interface

- **Provides:** the `circuit-check` command (`circuit_check.cli:main`) and its report. `targets` (what a target string
  names) and `structural` are imported as modules too.
- **Loaded by:** `check`, which runs the command and plans it by target (`tools/verity/check/plan.py`); the benchmarks
  that stage or lower a catalog target (`benchmarks/one_stage`, `benchmarks/proof_throughput`, `benchmarks/served_units`,
  `benchmarks/unit_lowering`); Flock's pod catalog (`verity/protocols/proofs/flock/prover/pod/catalog.py`); and a migration tool under
  `adapters/vllm/tools`.
- **Loads:** every Definition module's sidecar and pins, and the module each belongs to, `verity.circuits`,
  `verity.definitions` and Flock (`verity_flock`). It imports no integration by name: a package under `adapters/`
  registers itself by a sidecar (below), and its modules that have one are then loaded like `verity`'s.

## Commands and modules

~~~sh
uv run circuit-check F32Add_v1 'RMSNormTriton_v1{N=64,EPS=1e-05}'    # Definitions, at a binding
uv run circuit-check MODULE.circuit-check.py                         # what a sidecar's bindings reach in its module
uv run circuit-check --all                                           # every registered Definition and template
uv run circuit-check --structural UNIT.tmpl TARGET                   # a template as the Definition's lowering
uv run circuit-check --structural DIR TARGET                         # a split, its nets wired to lookup slots, as it
uv run circuit-check --ambient TARGET...                             # each lowering, read under a perturbed interpreter
~~~

It exits 1 when a check fails, and names each failure (`FAIL NAME`). Redundant and dead gates are warnings with counts
(`--fail-on-warnings` makes them fail). Each realization is compared on at least 2 random vectors while the reference's work
on those the floor adds stays within 10M gate evaluations: a lowering on random vectors added after the all-edge ones, a
kernel on vectors its own sampler samples in its domain. The report counts and names the targets compared on a random vector
(`random`), on all-edge vectors only (`edge-only`) or on none (`none`), and the last two are warnings; a constant, with no
input to sample, is in none of them. A target whose every
input is unchanged is reused
from the result cache (`--no-cache` checks it now). It needs about 24 GB a process, so run it on a node through
`research run --on auto`.

| Module (`src/circuit_check/`) | What it holds |
| --- | --- |
| `checks.py` | the checks: correspondence with the reference on every realization, declared port intervals, redundant and dead gates, the partition, the Flock block a Definition needs as one unit |
| `block_held.py` | the Definitions past the Flock block that the report holds rather than fails, each with why; it only shrinks: a held one that fits, or that the whole catalog no longer checks, fails |
| `bodies.py` | a composite's evaluation and its non-constant gates, body by body |
| `targets.py` | what a target string names, and the catalog `--all` runs over |
| `sidecars.py` | each Definition module's bindings and pins, in its `<module>.circuit-check.py` and `.circuit-check.json`, and where they are found |
| `ranges.py` | a family a sidecar registers over a range of statics (`RANGES`), checked at its spots (`--ranges`), and the report's `ranges` |
| `vectors.py` | the edge and seeded random input vectors |
| `structural.py`, `structural_slots.py`, `ambient.py`, `by_calls.py` | `--structural` on a template and on a split (`verity_flock.struct_slots`), `--ambient`, and a Program root checked by its calls alone |
| `cache.py`, `units.py` | the result cache, and the catalog bindings' records, each reused while what it read is unchanged; a checker version no run has used for 72 hours is removed by the next run that uses the cache |
| `entries.py` | the shared lists' layout, one file per entry (`pins/`) |
| `cli.py` | the command line and the report (`verity-circuit-check/v1`) |

circuit-check names no circuit of the catalog it is handed (`tests/test_handed_names.py`): what it knows of one is in
the sidecars of the module that defines it. Beside a module's binding file (`<module>.circuit-check.py`) and pins file
(`.circuit-check.json`), its unkeyed file (`.circuit-check.unkeyed.json`) holds what the scheduler, the task list's guard
and the report read and no check does, so no key holds it: the peak memory, in GiB, of each of its targets that grows
past 4 GiB and of each of its bindings whose build grows past 2 GiB (`cli.heavy`, `units.built_apart`), its Definitions
past Flock's block that the report holds, with why (`block_held`), and its families' counts in the last task list that
passed. A new Definition gets a binding in its module's sidecar: the suite fails any registered Definition family the
catalog doesn't reach.

`--all` refuses a task list far larger than the last that passed before any target is checked: a family of more than
`floor` targets and more than `factor` times its count then, or the list past `factor` times its length then
(`task_list.json`, with the families' counts in their owners' unkeyed files). A growth on purpose is accepted with
`circuit-check --all --accept-task-list`, whose files the change commits.

circuit-check imports `verity` and Flock, and no integration. A package under `adapters/` registers itself by a
sidecar of its own module (`<package>/__init__.circuit-check.py`), and then its sidecars are found like `verity`'s. What
a check needs of the integration is in that sidecar: the frontend vocabularies' kinds (`_vocabulary_roots`) and the
reader of a request Program (`request_program`). A split is read the same way, by the reader the sidecar beside
`verity_flock.struct_slots` defines (`struct_slots_split`), so `structural_slots.py` imports nothing of Flock.
