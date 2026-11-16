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

[`DESIGN.md`](DESIGN.md) says what each warning means, how the result cache and the task list's guard work, and how
a package under `adapters/` registers its circuits by a sidecar.
