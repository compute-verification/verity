# `verity graph`

`verity graph` is Verity's check written as a graph of tasks: compile one Lean module, run one test file, check one
circuit binding, build the Flock verifier. Each task declares the files and the other tasks' outputs it reads, and its
key is a hash of exactly those. So the graph depends only on the commit, and a task's result holds while its key is
unchanged.

This package lists the graph and runs one task of it. The executor that runs the graph on the team's machines, with one
result cache, lives outside this repository and only reads what this package lists. The verified queue reads the
results. `verity tasks` is the coarser graph `check` runs today.

## Try it

~~~sh
uv run verity graph --family lean-env
~~~

This lists one family's tasks in under a second and needs nothing beyond `uv sync`. It prints a line with the count of
tasks and files and the commit, then a line for each task: its id, its environment, its key (or `reads` while the key
waits on the outputs it reads), and what it does. `verity graph --help` gives every subcommand.

## Families

| Family | Its tasks |
| --- | --- |
| `test-file` | One test file, run alone as its suite runs it. |
| `lean-module` | One Lean module compiled and replayed through the kernel. |
| `lean-env`, `lean-lock` | A Lean package's environment made and its modules replayed; the guarantees a lock holds checked against their records. |
| `circuit-binding` | One circuit-check binding. |
| `build` | The check's steps that build the Flock verifier and run what it built, and Flock's Rust. |

Each family is a lister and a runner for each of its kinds, with a table in `tasks.toml`; a new family edits nothing else.

[`DESIGN.md`](DESIGN.md) holds the contract other executors reproduce: the listing's format, the key rule, one task's
result, the events, the build family's tasks, how to write a family, and the invariants the tests hold.
