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
`gpu-lease` back on its own rules. `cluster shadow replay` runs the shadow over a day's copied logs, and `cluster shadow
evaluate` is the pass bar. On vy-nebius-1, `nebius1` reads Kueue's pods and Workloads and DCGM's GPU-to-pod labels as
allocations and a queue, and `python -m cluster.nebius1 DESCRIPTION [--save|--from SNAPSHOT]` prints what the planner would
do beside Kueue. Node 1's executor under the one queue is kueue-fold's.

A lane submits with `research run --queue` (tools/research): it declares the job's class and resources, `cluster submit`
places it and names the wrappers its node's executor runs it under (`submit.py`): the machine-side runner runs in a capped
and pinned systemd scope, the job's CPU allocation, so the run's custody finishes before it is released, and a GPU job's
workload alone runs under `gpu-lease --wait` on node 2, so the GPU is booked only while it runs. Node 1's GPUs wait for kueue-fold's executor; such a job goes to node 2 or is
refused. A job names its kind (`--kind`, default `adhoc`) and research question (`--question`); a kind registered in
`kinds/<owner>.toml` supplies its shape, time and question, and the rules on kinds only warn for now.

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
├── nebius2.py  node 2's gpu-lease as the model sees it: owner lines, wait files, preempted.log; read(), live and read-only
├── shadow.py   the planner beside gpu-lease: mirror into the ledger, plan, divergences; replay(); evaluate(), the pass bar
├── agent.py    the node agent: the tick loop (shadow or live), timed windows left alone, stop on exception or a rule broken
├── submit.py   the submit path: a declared job's placement and its node executor's wrappers (research run --queue)
├── kinds.py    the job-kind registry (kinds/<owner>.toml): a kind's shape, time and question; the held rules as warnings
├── status.py   each node's running, queued and held jobs by owner and key, GPU-h ready, holds, disk, model cache and
│               node 1's unreplayed bundles (release.py's rule), read over SSH; read-only
├── models.py   the node's model-checkpoint cache: fetch (verified, resumable), ls, use, release, evict
└── cli.py      cluster validate | plan | submit | ledger … | agent | shadow replay | shadow evaluate | status | models …
descriptions/nebius.toml   vy-nebius-1 and vy-nebius-2 as their owners describe them on 30 Sep 2026
kinds/<owner>.toml         each owner lane's job kinds, read from the tree a submitter ships (kinds.py's docstring)
models.toml                the model cache's root, expiry, max disk use, and each owner's quota per node
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

A lease taken by hand over SSH (a kernel inner loop, a debugger) is a `session` allocation, so the same rules see it. A
session may have no hard limit but an expected length (`gpu-lease` without `--max-min`), which plans backfill and is never
enforced, and a lease may be CPU-unmanaged (`cpus = 0`), as `gpu-lease`'s are.

Lease and queue state come from the ledger alone (`ledger.state`): a start takes a job out of the queue, a queued job is
bound to the live handle its request holds and is withdrawn when that drops, and the simulation checks on every tick that
the ledger replays into exactly the simulator's allocations and queue.

## Commands

~~~text
uv run tools/check/suites.py cluster
uv run cluster validate tools/cluster/descriptions/nebius.toml
uv run cluster plan tools/cluster/descriptions/nebius.toml SNAPSHOT.json    # snapshot shape: cli.py's docstring
uv run cluster submit tools/cluster/descriptions/nebius.toml --run-id R --now T --project P --submitter W --max-s S [--gpus N ...]
uv run cluster ledger verify|quiet|failures LEDGER.jsonl ...
uv run cluster shadow replay tools/cluster/descriptions/nebius.toml --sampler UTIL.jsonl.gz --preempted preempted.log --out DIR
uv run cluster shadow evaluate DIR --quiet-log quiet.jsonl
uv run cluster status [--node nebius-1|nebius-2|all] [--owner OWNER] [--json]   # over SSH with ~/.ssh/research_key
~~~

## The model cache

Each node has one cache of Hugging Face checkpoints, `/workspace/models` (`models.toml`). An entry is
`<org>--<name>@<40-hex revision>/`: the repo's files at that commit as plain files (no Hugging Face blob store, symlinks or
`.cache/`), and `.entry.json` (`verity/model-entry/v1`) with each file's sha256 and bytes, the total, owner, phase,
`declared_gb`, created, last used, expiry, and the run that fetched it. `.entry.json` is written last, atomically, so a
directory without one is incomplete and nothing reads it. The entries are the registry; there is no other database.

~~~text
cluster models fetch REPO@REV --owner O --phase P --declared-gb N [--expires-h 72] [--include GLOB ...] [--out DIR]
cluster models ls [--json] [--owner O]        # entries by owner, GB, last use, expiry; incomplete downloads
cluster models use REPO@REV                   # a job's first step: bumps last use and expiry, prints the entry's directory
cluster models release REPO@REV --owner O     # the owner deletes its entry now
cluster models evict [--dry-run]              # as root (sudo -n): expired entries and day-old incomplete downloads
~~~

Every command also takes `--root`, `--config` and `--now ISO`. Refusals exit 2 for a malformed request and 3 for a rule,
with one line; a failed download or checksum exits 1.

**Fetch** takes a pinned revision or nothing: a branch or tag exits 2. It reads the file list and checksums from
`/api/models/<repo>/revision/<rev>?blobs=true` and refuses (exit 3) when a file has no checksum, when the listed total is
over `--declared-gb`, when the owner's entries on the node would pass its quota with this one (circuits 500 GB, anyone
else 200 GB), or when the filesystem would pass `max_use_pct` (75%) afterwards. Other fetches still running count
against both. The 75% stays under `research disk-hold`'s 80%, so a fetch never holds other lanes' runs. Files stream into
`.tmp/<entry>-<pid>/` and are hashed as they are written: an LFS file's sha256 against `lfs.sha256`, any other file's git
blob sha1 against its blob id. A mismatch deletes the download; an interrupted fetch leaves it for the next one to resume
with HTTP Range. A per-entry `flock` makes a second fetch of the same model wait and then find the entry. Fetching an entry
that is already cached only bumps it, and only its owner may do that. Fetch is stdlib Python. It sends the token at
`HF_TOKEN_PATH` (default `/etc/verity/hf-token`) to huggingface.co alone, never to the CDN, and never prints it. Without
the token, a gated repo exits 3 naming that path.

**Lifecycle.** A job calls `use` when it starts, so the entry expires `expires_h` (72 h) after its last use. The owner
`release`s it when its phase ends; otherwise `evict` deletes it once it has expired. Neither deletes an entry that any
process holds open or mapped, as `/proc/*/fd` and `/proc/*/maps` show it by device and inode. Only root sees every
process, so `evict` without root deletes nothing (`--dry-run` says what it would delete). Nothing schedules `evict` yet.
`cluster status` prints each node's cache in one line, for example `models: 3 entries, 142.0 GB (circuits 120.3/500 GB,
infra 21.7/200 GB), next expiry 2026-10-03T19:00Z`.

**Asking for a model.** The `model.fetch` kind (`kinds/infra.toml`: 4 CPUs, 8 GiB, 240 min) runs on the node that will
read the model, so pin it with `--on`, and pass the same size to `--disk-gb` and `--declared-gb`:

~~~bash
research run --queue --kind model.fetch --source . --project verity --on vy-nebius-2 --disk-gb 17 \
  --question "Is Qwen/Qwen3-8B@<rev> on vy-nebius-2, verified?" --cwd source --declared-output 'out/*' -- \
  bash -c 'PYTHONPATH=tools/cluster/src python3 -m cluster models fetch Qwen/Qwen3-8B@<rev> --owner circuits \
    --phase B --declared-gb 17 --out "$RESEARCH_RUN_DIR/out"'
~~~

The job that reads the model then runs `MODEL=$(PYTHONPATH=tools/cluster/src python3 -m cluster models use REPO@REV)` and
loads from `$MODEL` with `HF_HUB_OFFLINE=1`.
