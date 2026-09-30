# cluster

Cluster tooling for the machines Verity's lanes share: one description of the nodes (who owns each, the shares under
which other projects borrow, priority classes, stop times), a pure planner that decides each node's allocations, placement
across nodes, and a hash-chained allocation ledger with a failure feed and quiet-window reports. Stdlib only; it imports
neither `verity` nor `research` (`tests/test_boundaries.py`).

Nothing here runs on a node yet. `cluster plan` over a snapshot is how a shadow run compares the planner's decisions with
what a node's current tools (`gpu-lease`, node 2's fill runner, node 1's Kueue) did. A cutover is a written plan that goes
to Daniel first.

## Layout

~~~text
src/cluster/
├── cpuset.py   CPU lists in the kernel's format (0-7,16)
├── model.py    Cluster, Node, Gpu, Share, Job, Allocation; the description's TOML; job and allocation dicts
├── plan.py     plan(): one node's actions (expire, evict, freeze, start, thaw, wait, refuse); idle(): idle GPU leases
├── route.py    place(): which node's queue a job enters
├── ledger.py   the append-only, hash-chained ledger; history, quiet_report, failures
└── cli.py      cluster validate | plan | ledger verify | ledger quiet | ledger failures
descriptions/nebius.toml   vy-nebius-1 and vy-nebius-2 as their owners describe them on 30 Sep 2026
tests/          one scenario per rule (test_plan.py), and random workloads checked against every rule (test_simulation.py)
~~~

## The rules

`plan.py`'s docstring states them, and `tests/test_simulation.py` checks them on every tick of seeded random workloads over
the Nebius description and a small two-node cluster: no GPU, CPU or lock in two allocations and no memory overcommitted;
a node's reserved CPU pools (node 1's check slots) only for jobs naming them;
owner before guest, then priority, and only a strictly higher rank evicts, and only a `requeue` allocation; a quiet job
runs with everything else on its node evicted (GPU work) or frozen (CPU work); the highest-ranked blocked job reserves its
node and lower jobs backfill only around it; no job admitted past its node's admission end or its share's end, and frozen
time not counted against `max_s`; equal jobs take turns by submitter; idle GPU leases named and reclaimed from `requeue`
holders.

A lease taken by hand over SSH (a kernel inner loop, a debugger) is a `session` allocation, so the same rules see it.

## Commands

~~~text
uv run tools/check/suites.py cluster
uv run cluster validate tools/cluster/descriptions/nebius.toml
uv run cluster plan tools/cluster/descriptions/nebius.toml SNAPSHOT.json    # snapshot shape: cli.py's docstring
uv run cluster ledger verify|quiet|failures LEDGER.jsonl ...
~~~
