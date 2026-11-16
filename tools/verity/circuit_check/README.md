# circuit-check

circuit-check is the test a circuit passes before anything uses it. It takes a circuit (a Definition, a subcircuit
template, a served workload's request Program) and runs it against its reference on edge and random inputs: every
realization of it, its kernels and its Boolean lowering included, must equal the reference bit for bit. It also checks
the circuit's declared port intervals, its redundant and dead gates, and whether it fits a Flock proof block.

A change that adds or modifies a circuit puts its report in the pull request, and `check` runs `circuit-check --all`.
A new Definition gets a binding in its module's sidecar (`<module>.circuit-check.py`): the suite fails any registered
Definition family the catalog doesn't reach.

## Try it

~~~sh
uv run circuit-check F32Add_v1
~~~

This checks one Definition, binary32 addition, in under a minute with no GPU. The last line counts the targets, the
failures and the warnings; a failure is named on its own line (`FAIL NAME`) and the command exits 1.

`circuit-check --all` checks every registered Definition and template. It needs about 24 GB a process, so run it on a
node through `research run --on auto`. `circuit-check --help` lists every other form a target takes.

## Failures, warnings and the cache

It exits 1 when a check fails, and names each failure (`FAIL NAME`). Redundant and dead gates are warnings with counts
(`--fail-on-warnings` makes them fail), as is a Definition over the partition's cost target of 5,000,000 gates
(`partition/over-cost-target`, `Options.partition_max_gates`), a served Call or not: the cost target bounds no proof, and
`block` holds what is proved to Flock's range of 2^27 bits. Each realization is compared on at least 2 random vectors while the reference's work
on those the floor adds stays within 10M gate evaluations: a lowering on random vectors added after the all-edge ones, a
kernel on vectors its own sampler samples in its domain. The report counts and names the targets compared on a random vector
(`random`), on all-edge vectors only (`edge-only`) or on none (`none`), and the last two are warnings; a constant, with no
input to sample, is in none of them. A target whose every
input is unchanged is reused
from the result cache (`--no-cache` checks it now). It needs about 24 GB a process, so run it on a node through
`research run --on auto`.

## Sidecars

circuit-check names no circuit of the catalog it is handed (`tests/tools/verity/circuit_check/test_handed_names.py`): what it knows of one is in
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
