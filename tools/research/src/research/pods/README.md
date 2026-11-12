# `research.pods`: the machines

The machines `research` runs work on: the registry that names them, the lease and the budget that bound each one, how
to reach one, and the files the Nebius nodes are set up from. How a run reaches a machine is in
`tools/research/README.md` ("Remote runs"); this file says what is here and what code outside this directory loads
from it.

## `research pods`

`research.cli` hands every `research pods` verb to `part.py`'s `main`, which routes it:

~~~text
research pods registry [--json]             every machine `research run --on` accepts, and where its entry comes from
research pods register NAME --host ADDR --project P    bind a name to an ssh machine; refused unless it answers over ssh
research pods unregister NAME               drop the name
research pods ssh TARGET [-- CMD ...]       reach a machine by its name
research pods sync TARGET [SRC]             put a worktree's files on it (what git would show, nothing ignored)
research pods lease TARGET                  its lease line
research pods extend TARGET --hours H       the lease ends no sooner than now + H; it is never shortened
research pods check-part TARGET [--json]    whether it is the reference part; exit 3 when it is not
research pods guard status                  the budgets guard: `guard` runs it, `guard stop` ends it
research pods unpreserved TARGET            the Attempts recorded there that the store has not preserved
research pods drain TARGET                  terminate it only when nothing on it is unpreserved
~~~

`create` makes a RunPod pod with its lease, and any other verb (`list`, `terminate`) goes to `runpod.py` unchanged.
`create` needs `--max-hours` and a budget line that covers the name, and makes at most one live pod per `--name`, so a
retry loop can't duplicate pods: exit 4 when one exists (`--allow-duplicate` overrides), 5 for RunPod's explicit
no-stock reply, the one failure to answer with another spec, 6 when no budget line lets it, and 2 for any other failure.

## What bounds a pod

Nothing that bounds spend depends on an agent VM, which can be suspended at any moment, or on one process staying
alive. Each module's docstring is its contract.

- **A lease on every pod** (`lease.py`, `sh/lease.sh`). `create --max-hours H` arms it once sshd answers and
  terminates a pod it could not arm. At the expiry the idle guard ends a pod whose runs all have custody, and the
  lease loop ends it after a grace whatever is unfetched, with the pod's own key. `research run --on` extends the
  lease to cover the run's `--timeout`; `extend` never shortens it.
- **One budget file** (`budgets.py`): `budgets.toml`, where each line is a name prefix with a spending limit (total,
  per rolling day, or both), an expiry, a longest pod age and who approved it. `create` refuses unless the guard
  polled lately and holds a live, untripped line that covers the name.
- **One guard** (`guard.py`) on the control pod re-reads the file from the notes' origin every poll. It terminates the
  pods of a tripped line, every project pod at the account's balance floor, a pod past its line's age, and a project
  pod no line covers. It never trips on the clock and never fails open: it decides from memory before it records, a
  failed read keeps the last value, and `--detach` runs it under a supervisor that restarts it after a crash.

## Modules

| File | What it holds | Loaded from outside `research` by |
|---|---|---|
| `registry.py` | The machine registry: one TOML file per machine under `machines.d/` in the notes repository. | `infra/ops` (a released node has no files to read), `benchmarks/numerical` (a pod's id) |
| `runpod.py` | RunPod's API, which now hosts only the control pod: create, list, get, ssh, terminate. | `benchmarks/numerical`, `tools/verity/check`'s tests |
| `budgets.py` | `budgets.toml`, every spending limit, at the notes repository's root; `gate` is what `create` asks. | `infra/control` (the guard's host) |
| `connect.py` | `ssh` and `sync`; `ssh_base` is the ssh command for a name, looked up at call time. | `infra/control` |
| `part.py` | The `research pods` router, `create`, and the reference-part check (`check-part`; `create --require-reference-part`). | |
| `lease.py` | The lease every pod carries from creation, so it ends whether or not anything else is alive. | |
| `guard.py` | The one budgets guard: it enforces every line of `budgets.toml`, re-read each poll. | |
| `drain.py` | Teardown gated on preservation (`unpreserved`, `drain`). | |
| `workproc.py` | What counts as work on a pod: one definition for the notes' pod probe and `sh/pod_guard.sh`. | |

`sh/` holds what runs on the machine itself:

- `sh/gpu_lease.sh`, installed as `gpu-lease`: `gpu-lease N [--wait] -- CMD` holds N GPUs for CMD, and `gpu-lease
  status` says who holds which. `benchmarks/served_units` links it for its window.
- `sh/pod_guard.sh`, the guard a job launcher sources on the pod: the pod ends itself after an idle window with no
  job process, or when its job chain finishes. `adapters/vllm`'s pipeline falls back to this copy when the pod has
  none of its own.
- `sh/lease.sh`, the loop that ends the pod once its lease has expired.

## `nebius/`

What the Nebius nodes run, and what brings one up (`launch.sh`, `vm_setup.sh`, `teardown.sh`). `nebius/deploy.toml` is
the manifest of every file a node has installed: `research deploy drift|install` (`infra/ops`) reads it, so a file
added here reaches a node only through a line there.

| File | What it is |
|---|---|
| `nebius/check.py` | The bootstrap check: whether a node is the machine paid for (driver, GPUs, memory, CPUs, data disk and its read rate) with its lease and `gpu-lease` live. It runs on the node, under `research run --on NODE`; its docstring has the line. |
| `nebius/slot_only.py` | On a node marked for it, stops (or renices) each run that did not come through a check slot. `tools/verity/check` reads the pools' names from it, and `benchmarks/harness`'s tests its `through_slot`. |
| `nebius/wake.py` | The control pod's side of `research run --on`'s wake: it starts a stopped node for an agent that holds no cloud key, only one the registry names and a budget line covers. |
| `nebius/fleet.py` | The fleet's usage record: `sample`, `report`, `instances`, `quota`, `panels`. |
| `nebius/node_ops.py` | A shared node's OOM guard, status page and alerts. |
| `nebius/oom_order.py` | Orders the kernel OOM killer's picks by each run's memory against its reservation. |
| `nebius/sky/` | Node 1's Kueue queue: `nebius/sky/kueue.yaml` (the queues and their quotas, which `cluster_up.sh` applies), `nebius/sky/jobs/` (one template per kind of job, `nebius/sky/jobs/config-run.yaml` among them), `nebius/sky/job_tree.sh` (the tree a job runs from, one copy per content), `nebius/sky/vllm_bootstrap.sh` (a job's vLLM bootstrap, skipped once it has passed), and `nebius/sky/store_evict.sh` with `nebius/sky/store_evict_src.conf` (the timer that frees a node's disk of local copies the remote store holds). |
| `nebius/challenger/` | The auditor host's setup and units for the challenger and the network certifier's rehearsal, which `verity/composite/device/live_challenger` reads. `nebius/verifier/` is the same for the verifier's host. |
| `nebius/lockout/` | Taking root away from the nodes' operator account; it has its own README. |

`tools/research`'s tests read these files by path, so a rename here is a rename there.
