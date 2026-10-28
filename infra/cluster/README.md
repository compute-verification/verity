# cluster

Cluster tooling for the machines Verity's lanes share: one description of the nodes (who owns each, the shares under
which other projects borrow, priority classes, stop times), a pure planner that decides each node's allocations, placement
across nodes, and a hash-chained allocation ledger with a failure feed and quiet-window reports. Stdlib only; it imports
neither `verity` nor `research` (`tests/test_boundaries.py`).

On vy-nebius-2 the agent (`cluster agent`) runs beside today's tools. In shadow mode it observes `gpu-lease` through
`nebius2` (owner lines, wait files and `preempted.log`, reading locks from their holders' `/proc/<pid>/fdinfo`, never
through a lock), mirrors what it did into the ledger, and records what the planner would have done (`shadow.py`). In live
mode it holds `/run/gpu-lease/agent.lock` and carries the plan out: it evicts by signal and grants by writing
`grant.<pid>`, which `gpu-lease`'s agent mode (on `infra/nebius`, 49238797 in fixtures/) takes; a free `agent.lock` puts
`gpu-lease` back on its own rules. The live agent runs as `vy-cluster-agent.service` (the unit file here, run from the
copy of this directory that `research deploy` installs at `/workspace/verity-guest/cluster-agent`) with `--roll`: a new ledger segment under `live/` for each start and each UTC day, the segments one hash
chain, which `cluster ledger ... live/` reads whole. `cluster shadow replay` runs the shadow over a day's copied logs, and `cluster shadow
evaluate` is the pass bar. On vy-nebius-1, `nebius1` reads Kueue's pods and Workloads and DCGM's GPU-to-pod labels as
allocations and a queue, and `python -m cluster.nebius1 DESCRIPTION [--save|--from SNAPSHOT]` prints what the planner would
do beside Kueue. Node 1's executor under the one queue is kueue-fold's.

A lane submits with `research run --queue` (tools/research): it declares the job's class and resources, `cluster submit`
places it and names the wrappers its node's executor runs it under (`submit.py`): the machine-side runner runs in a capped
and pinned systemd scope, the job's CPU allocation, so the run's custody finishes before it is released, and a GPU job's
workload alone runs under `gpu-lease --wait` on node 2, so the GPU is booked only while it runs. On node 1 it runs under node 1's
`gpu-lease`, whose GPUs `n1_lease.py` lends from Kueue, once the description sets its `gpu_executor = "n1-lease"`; until then
such a job goes to node 2 or is refused. Only a GPU job pinned to node 1 (`--on vy-nebius-1`) goes there, since the submit path
can't see node 1's lease pool. A job names its kind (`--kind`, default `adhoc`) and research question (`--question`); a kind registered in
`kinds/<owner>.toml` supplies its shape, time and question, and the rules on kinds only warn for now. A job of an unregistered
kind gives its own time (`--max-min`) and memory (`--mem-gb`), or is refused: its scope is killed at `MemoryMax=`.

## Layout

~~~text
src/cluster/
├── cpuset.py   CPU lists in the kernel's format (0-7,16)
├── model.py    Cluster, Node, Gpu, Share, Job, Allocation; the description's TOML; job and allocation dicts
├── plan.py     plan(): one node's actions (expire, evict, freeze, start, thaw, wait, refuse); idle(): idle GPU leases
├── invariants.py  the rules as checks on a node's state, run by the simulation and by the agent every tick
├── route.py    place(): which node's queue a job enters
├── ledger.py   the append-only, hash-chained ledger; state (the one source of lease and queue state), state.json,
│               gpu-lease/usage/v1 ingest, history, quiet_report, failures, usage
├── nebius1.py  node 1's Kueue as the model sees it: pods, Workloads, DCGM's GPU-to-pod labels; read(), live and read-only
├── kueue_shadow.py  node 1 with Kueue as its only scheduler, in shadow: the lease pools', check slots' and Lean slots' demand
│               as bare Workloads in `vy-shadow`'s own cohort, and where Kueue's admission differs from what the node did
├── nebius2.py  node 2's gpu-lease as the model sees it: owner lines, wait files, preempted.log; read(), live and read-only
├── shadow.py   the planner beside gpu-lease: mirror into the ledger, plan, divergences; replay(); evaluate(), the pass bar
├── agent.py    the node agent: the tick loop (shadow or live), timed windows left alone, stop on exception or a rule broken
├── submit.py   the submit path: a declared job's placement and its node executor's wrappers (research run --queue)
├── kinds.py    the job-kind registry (kinds/<owner>.toml): a kind's shape, time and question; the held rules as warnings
├── status.py   each node's running, queued and held jobs by owner and key, GPU-h ready, holds, disk and node 1's
│               unreplayed bundles (release.py's rule), read over SSH; read-only
├── policy.py   the policy a description declares ([policy]: each key and its owner) and every key where a node's live
│               value differs, read over SSH; read-only
├── policy_apply.py  the change that converges each key that differs, as a plan, and with --apply written on the node over
│               SSH, logged in its override log and read back
└── cli.py      cluster validate | plan | submit | ledger … | agent | shadow replay | shadow evaluate | status | policy
descriptions/nebius.toml   vy-nebius-1 and vy-nebius-2 as their owners describe them on 30 Sep 2026, and their policy
descriptions/nebius-after-n1.toml   node 2 alone, staged for node 1's retirement (INF-57): it replaces nebius.toml then
kinds/<owner>.toml         each owner lane's job kinds, read from the tree a submitter ships (kinds.py's docstring)
vy-cluster-agent.service   node 2's live agent as a systemd unit: install, stop and restart in its header
tests/          one scenario per rule (test_plan.py), random workloads checked against every rule (test_simulation.py),
                and node 2's own logs and gpu-lease (fixtures/: 30 Sep 07:20-07:58Z, the live gpu-lease 0d172cf3 and
                the agent-mode gpu-lease 49238797 held for deploy)
~~~

## The rules

`plan.py`'s docstring states them, and `tests/test_simulation.py` checks them on every tick of seeded random workloads over
the Nebius description and a small two-node cluster: no GPU, CPU or lock in two allocations and no memory overcommitted;
a node's reserved CPU pools (node 1's check slots) only for jobs naming them;
owner before guest, then work within its workstream's own GPUs before borrowers (node 1's `provers` and
`deployments-gpu`, as Kueue's quotas), then priority, and only a strictly higher rank evicts, and only a `requeue`
allocation; a quiet job
runs with everything else on its node evicted (GPU work) or frozen (CPU work); the highest-ranked blocked job reserves its
node and lower jobs backfill only around it; no job admitted past its node's admission end or its share's end, and frozen
time not counted against `max_s`; equal jobs take turns by submitter; idle GPU leases named and reclaimed from `requeue`
holders.

A node's stop is machine state: its live `/etc/research/deadline`, which infra rewrites on each renewal. Node 2's agent
reads it every tick (`agent --deadline-file`), and `research run --queue` reads each candidate node's over ssh and passes it
as `cluster submit --stop NODE=EPOCH`, so a renewal places and plans without a new description. A node's `stops_at` is the
fallback for when the file can't be read, and a refusal at the admission end names which of the two it used. A share whose
`until = "stops_at"` lasts until its node's stop.

A lease taken by hand over SSH (a kernel inner loop, a debugger) is a `session` allocation, so the same rules see it. A
session may have no hard limit but an expected length (`gpu-lease` without `--max-min`), which plans backfill and is never
enforced, and a lease may be CPU-unmanaged (`cpus = 0`), as `gpu-lease`'s are.

Lease and queue state come from the ledger alone (`ledger.state`): a start takes a job out of the queue, a queued job is
bound to the live handle its request holds and is withdrawn when that drops, and the simulation checks on every tick that
the ledger replays into exactly the simulator's allocations and queue.

## Commands

~~~text
uv run tools/verity/check/suites.py cluster
uv run cluster validate infra/cluster/descriptions/nebius.toml
uv run cluster plan infra/cluster/descriptions/nebius.toml SNAPSHOT.json    # snapshot shape: cli.py's docstring
uv run cluster submit infra/cluster/descriptions/nebius.toml --run-id R --now T --project P --submitter W --max-s S [--gpus N ...]
uv run cluster ledger verify|quiet|failures LEDGER.jsonl ...
uv run cluster shadow replay infra/cluster/descriptions/nebius.toml --sampler UTIL.jsonl.gz --preempted preempted.log --out DIR
uv run cluster shadow evaluate DIR --quiet-log quiet.jsonl
uv run cluster status [--on nebius-1|nebius-2|all] [--owner OWNER] [--json]   # over SSH with ~/.ssh/research_key
python3 -m cluster kueue-shadow loop --out RECORD.jsonl --hours H   # on vy-nebius-1; `kueue-shadow down` removes it
uv run cluster policy show [--on nebius-1|nebius-2|all] [--key K] [--json]   # the policy in force: key, owner, source
uv run cluster policy diff [--on nebius-1|nebius-2|all] [--json]   # over SSH: exit 1 drift, 3 a value unread
uv run cluster policy override KEY VALUE --until 2026-10-04T10:30:00Z --by WHO --reason R [--apply]   # on KEY's node
uv run cluster policy apply [--on nebius-1|nebius-2|all] [--key K] [--json]   # the plan: each change that converges a key
uv run cluster policy apply --on nebius-2 [--key K] --apply --as WHO --reason R   # write it over SSH, log it, read it back
~~~

Policy is what node services read from outside git: node 1's lease pools (the `n1-lease` panes' environment), Kueue's
ClusterQueue quotas and stop policies, the check and Lean slot files, `gpu-lease`'s direct-run maps, the cpuset drop-in,
and node 2's booked windows and CPU sets. `[policy]` in the description declares each, with the handle who decides a
change; `cluster policy diff` names every key whose live value differs or that the description lacks, and calls a key it
couldn't read unknown, never as declared (node 2's are all unknown while a timed window is on: it reads nothing else then).
While node 2's fill runner is down (`status.txt` over 15 min old and no `fill_runner.py` process), node 2 is read as quiet
from `gpu-lease`'s own lines instead, and `diff` and `apply` print `read as quiet: WHY`: still never while a lease is
`--timed`, one pid holds every GPU, someone waits for every GPU, or a booked window is on or starts within 30 min.
A change that should end on its own is an override: `cluster policy override`, run on the key's node, records the value, who,
why and until when in that node's hash-chained log (`/workspace/research/policy/overrides.jsonl`; a dry run without
`--apply`), and `show` and `diff` use it until then. `n1_lease.py` takes its pool's `max`, `borrow` and `borrow-until` from
`cluster policy show --key vy-nebius-1.lease-pool.<pool> --json` when `$VY_POOL_POLICY` names that command; it writes the
three in force to `n1-lease.limits` in its lease directory, and `diff` reads them there for such a loop, not from its
environment.

`cluster policy apply` (`policy_apply.py`) converges the rest where it can. For each key `diff` calls drift it plans the
exact change, as a diff of each file, and with `--apply --as WHO --reason R` writes it over the same SSH path, appends one
`apply` record per key (key, value, was, who, why, the file's sha256 before and after) to the node's log, and reads the node
again to confirm each key reads as in force. Its appliers: the check and Lean slot files and `fill/cpu-sets` (the key's line,
the old one kept as a comment), `gpu-lease`'s direct-run maps (written whole: `gpu-lease` reads them whole), and the cpuset
drop-in's `Delegate=` line (then `daemon-reload`, never `systemctl revert`, and `user.slice` must keep its AllowedCPUs). On
vy-nebius-2 the drop-in is planned only, since `vy-device-cores` owns its cpusets. Lease pools, Kueue's queues, a node's
`pools` and booked windows have no applier and are skipped, saying so; an unknown key is never applied, nor an undeclared
one. The write is one command: on node 2 it starts with node 2's gate, so nothing is written while a timed window is on or
near, nor while the fill runner is down (a write needs its fresh quiet status), and under the log's lock it writes nothing unless every file and the log are byte for byte what it planned from. Exit
0 nothing left to change, 1 a change planned or a key skipped, 3 a key unknown, 4 an apply that failed or didn't read back.
