# cluster

`cluster` decides who gets the team's shared nodes. One description says what each node holds, who owns it, the shares
under which other projects borrow it, the priority classes and when each node stops. From that description and a
node's state, a pure planner decides each node's allocations: what starts, what waits, what is evicted or frozen. Every
allocation goes into a hash-chained ledger, which is the one source of lease and queue state.

It is the standard library alone, and imports neither `verity` nor `research`, so it runs on a bare node
(`tests/infra/cluster/test_boundaries.py`). A lane reaches it through `research run --queue`, which places the job with
`cluster submit`.

## Try it

~~~sh
uv run cluster validate infra/cluster/descriptions/nebius.toml
~~~

This checks the Nebius description and prints one line of what it describes, the nodes, GPUs, shares and workstreams.
It needs nothing beyond `uv sync`. `uv run cluster --help` lists every other command.

## What it does

| Command | What it does |
| --- | --- |
| `cluster plan` | One node's actions from a snapshot of its state. |
| `cluster submit` | A declared job's node and the wrappers its node's executor runs it under. |
| `cluster agent` | The planner live on a node: on node 2 it carries the plan out through `gpu-lease`, or in shadow mode records what it would have done. |
| `cluster ledger`, `cluster shadow` | The ledger read and verified; the shadow replayed over a day's logs and held to its pass bar. |
| `cluster status`, `cluster usage` | Each node's jobs and holds, and its GPU-hours by hour, read over SSH. |
| `cluster policy` | The policy node services read from outside git: what is in force, where a node differs, and the change that converges it. |

The rules the planner keeps (no GPU, CPU or lock in two allocations, owner before guest, only a strictly higher rank
evicts, and the rest) are in `plan.py`'s docstring, and random workloads are checked against every one of them on every
tick.
