# research

Shared tooling for executing, observing, and coordinating computational work
during development. One installable package, one CLI (`research`), stdlib
only. It runs commands and records what happened; it imports nothing from
Verity's packages (enforced by `tests/test_cli.py::test_no_packages_import`).

This file is for people changing the tool; its users read the `team` and
`writing-runs` skills.

Its tests are the `research` suite: `uv run tools/check/suites.py tools/research`.

## Layout and provenance

~~~text
src/research/
├── cli.py          the `research` command: every subcommand starts here, and its docstring lists them
├── at_main.py      runs origin/main's research when the checkout's is only behind it; an agent VM's `research` starts here
├── runs.py         run ids, the local registry (~/.research/runs or $RESEARCH_RUNS), read-time summaries
├── record.py       job.json v0.2 = the harness's v0.1 + tool, run, workload_fingerprint.expected, cost; v0.1 stays readable
├── result.py       the result.json contract (schema, run_id, fingerprint, validation, measurements, artifacts)
├── remote.py       --on: machines file, tool/source snapshots, request, claim-by-run-id, exclusive lock, fetch, preservation manifest
├── disk.py         --disk-gb admission (--queue): held on the machine while the disk would pass 80%, released at 75%
├── cores.py        --cpus (--queue): a queued run's whole cores, claimed on the machine
├── mem.py          --mem-gb admission: cgroup limit - in use - live /workspace/ramlock reservations; MemoryNotAvailable
├── auto.py         --on auto: a CPU node's research pool or node 2's queue, and when the run should start there
├── jobuser.py      --job-user: the workload as an unprivileged user in a sandboxed system unit; it makes no namespace but those
│                   --job-namespaces lists (of user, mnt, net), where the workload holds capabilities
├── env.py          conditions(): the one baseline of an attempt's conditions (source, runtime, hardware, execution); ONE SKU->family map
├── merge.py, queue.py, vqueue.py   landing: the merge gate, the merge queue and the verified queue, which
│                   tools/verity/landq replays
├── policy.py       the landing's policy over tools/check/policy.toml: what is off the merge gate and what may stay red
├── checkview.py    `research status RUN` on a check run; tools/check's live view reads its targets
├── goals.py        `research goals`: the Goals page (the writing-goals skill)
├── linear.py       asks as Linear issues, and the audit the control pod's top watch runs
├── response.py     `research msg linear response`: the minutes each lane takes to answer a Linear comment that rings it
├── flags.py        renamed(): an old flag kept as a hidden alias of its new name; infra/ops uses it
├── store/          the evidence store (`research data`); README.md there is the contract and names its modules
├── telemetry/      the recorded-attempt harness (below)
├── jobs/           the Job queue's client side (below)
└── pods/           the machines: registry, leases, budgets and guard, the control pod's door (door.py), the Nebius nodes' files; README.md there
tests/              pure-Python; the machine side of --on is exercised locally (launch_request is a plain function);
                    waits.py: waiting without a clock, for any suite's tests; test_pythonpath.py: a shipped tree's import roots
tests/fixtures/     job_v0.1.json: a record as the harness alone writes it; run_v0.2/: a finished run dir with a `store` section (state.json -> status.json at test time)
~~~

`research run` pre-writes `job.json` (the harness keeps an existing one) and
calls the harness's `run()` directly: one execution engine, one record.

## Credentials

The laptop's files come first and, where they exist, always win:
`~/.runpod/config.toml` (API key), `~/.runpod/ssh/runpodctl-ssh-key` (+ its
`.pub` and `veritor-campaign-known_hosts`), `~/.research/store.toml` (the
store's remote). Where a file is absent (a cloud VM with only the Cursor
secrets) the environment stands in:

| laptop file | environment | notes |
|---|---|---|
| `~/.runpod/config.toml` | `RUNPOD_API_KEY` | default GraphQL URL |
| `~/.runpod/ssh/runpodctl-ssh-key` | `RUNPOD_SSH_KEY_B64` | decoded once to a mode-600 file under `$XDG_STATE_HOME/research/runpod/` (default `~/.local/state`), known_hosts beside it; public key via `ssh-keygen -y` |
| `~/.research/store.toml` | `R2_ENDPOINT`, `R2_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` [, `AWS_SESSION_TOKEN`] | only when `$RESEARCH_STORE_CONFIG` is unset; `R2_BUCKET` must be the store's bucket (the one `tools/research/store.pod.toml` names) |

No value is ever printed.

On a cloud VM a long-lived shell can lose the injected secrets, which the
tmux server's global environment keeps while its session `vm-secrets` lives
(`tools/agent-guard/kept_session.py`; `research doctor` says when it is gone).
Every `research` command first sets from it each secret the shell lacks of
those `$CLOUD_AGENT_INJECTED_SECRET_NAMES` names (`secret_hint.recover`; not
under a test, so a secret a test unset stays unset). If one is still missing,
the store, `research queue ready`, node ssh and the Nebius scripts fail and
say `rerun from a fresh tmux login shell` (`secret_hint.py`). A shell can
lose that list too, so a store that has used the remote and now finds none
configured fails `research data label` and `put` the same way
(`LocalStore.remote_lost`).

`research doctor [--role lane|coordinator|infra] [--json]`: can this VM do its job (secret names, status-only
probes, ssh, `uv sync --check`, disk, clock); each finding names its fix (exit 1); it never prints a value.

## Commands

~~~text
uv sync                                  # from the verity root
uv run pytest                            # all tests

research run [--project P] [--stage S] [--timeout T] [--expect K=V] [--require-result] [--env K=V] [--cwd DIR] [harness flags] -- CMD...
research run [--tool NAME|module:NAME] [--scratch NAME]... [--campaign C] [...] -- CMD...      # evidence store, below
research run --on MACHINE --project P [--source GITDIR] [--send FILE] [--env K=V] [--env-host MACHINE] [--exclusive] [--mem-gb N [--mem-wait S]] [--preflight TOOL] [--wait [--wait-max S]] [...] -- CMD...
                                         # --wait: 20 min at most, then exit 79 (research/waitcap.py)
research run --on MACHINE [--custody-ttl 6h] ...    # the default: the machine puts the attempt + every run file on the store's remote (store README §10.1)
research run --on MACHINE --no-custody-r2 ...       # opt out: the records stay on the machine until `research fetch --all`
research run --queue --source GITDIR --project P [--kind K] [--question Q] [--mem-gb G] [--disk-gb N] [--max-min M | --timeout T] [--gpus N]
             [--gpu-model M] [--cpus N] [--class work|bench|...] [--phase cpu|gpu] [--preemptible] [--quiet] [--on NODE] -- CMD...   # the one queue (infra/cluster): placed, then an --on run in its CPU allocation, a GPU leased around the workload alone
research data custody RUN [--publish] | --triage                 # a run's custody on the remote, from any machine; the runs root by custody state
research run-preserve PATH [--kind K] [--meta JSON]              # in an --on run's workload: its runner preserves one file of $OUT now (§10.1)
research fetch RUN [--all]               # records only, or every file sha256-verified with preserved.json written last
research status [RUN] [--refresh] [--all] [--json]   # under each active run, a `mem` line per cgroup it executes in (scopemem.py):
                                         # current, peak, limit, growth from its own samples, NEAR LIMIT within 10%; remote: with RUN or --refresh
research inspect RUN [--json|--py-spy]   # fetch, status RUN and inspect take a run another VM launched (a full id): found on
                                         # --on NAME, else on each ssh machine but the auditor devices, and its records fetched here
research data SUBCOMMAND ...             # the store: put/show/fetch/select/... and lookup / pull / attempt publish
research pods SUBCOMMAND ...             # the machines: registry, ssh, sync, leases, the budgets guard (src/research/pods/README.md)
research tele SUBCOMMAND ...             # the harness's own surface (below)
research jobs add|list|show|cancel|events ...   # the Job queue (below)
research goals show|set|confirm|done|remove|replace|priority|audit|blockers ...   # the Goals page (goals.py); blockers ranks
                                         # the open blockers by what waits on each (goals_blockers.py)
research velocity [--line|--json]  # the velocity score from records, never a gate (velocity.py)
research compare ORIGINAL... REPLAY [--expect E] [--json]   # a replay's verdict against the runs it replays (compare.py)
research digest --as LANE   # what changed for a lane since its last digest (digest.py)
research merge REF [--into main] [--tool check] [-m MSG] [--push] [--dry-run]   # the merge gate (merge.py)
research merge --train A B C [--on MACHINE --project P] [--command CMD] [--push]  # several refs, one checked candidate
research merge --train --prepare BRANCH [--onto BASE] [--allow-removal PATH] N@SHA ...  # build, lint, push a train
research merge --sweep [--into main] [--remote origin] [--dry-run]       # close the PRs that landed, retarget their children
research merge --mirror [SHA]  # main or SHA to node 1's bare repo
research merge --canary [--tool check] [--json]                    # the gates' daily self-test, in scratch stores
research queue vqueue|status|ready|review|eject|open|edit|link  # the verified queue onto `main` (vqueue_loop.py, queue.py)
research queue judge RUN [--by LANE]      # the mark `queue ready --on` writes, from its quick-tier run that ended after its caller died
research queue stats [--json] [--short] [--rounds] [--until TS]  # queue_table.py; --log FILE: queue_stats.py
research lean-times ...  # Lean tasks' p95
research run-times [--hours 12] [--on MACHINE,...] [--json]  # each run's wait to start, setup and run time, by kind, machine and pool (run_times.py)
research git old2new|remap|check|build ...  # after a history rewrite (commitmap.py, below)
research msg send|close|read|list|names|traffic|owners|handles|subscriptions|ask-daniel|groups sync [--dry-run]   # Slack (slack.py, below)
~~~

After a history rewrite, an accepted `commit-map/v1` (labelled `commit_map=accepted`) lets an attempt of an old commit count for
its image when the tree is unchanged, and the gate refuses a REF holding a commit the rewrite replaced. `research git remap`,
`old2new SHA`, `check REF` and `build PRE POST` move clones onto it, name images, guard and build the map (`commitmap.py`).

`research merge` is how `main` receives a merge: in the target's checkout it runs `git merge --no-ff REF` only when the store
(refreshed from a usable remote first) holds an attempt of the tool, default `check`, with `state` done, rc 0,
and a clean source at REF's exact commit, and when the target's tip is an ancestor of REF, so the merged tree is the checked one.
The attempt's own record and result must agree with the target's declaration: its command (`Tool.command`, in a check slot or
not, plus check.py's own flags only), a result kind it produces (`check/v1`, never `lean-fast/v1`), a passed verdict and no step
listed twice. Anything else, unreadable included, passes no gate, and a refusal exits 3. The merge commit carries a `Check:
<attempt id>` trailer; `research` only reads the attempts of the repository's tool (`tools/check`). `--canary` checks, in scratch
stores, that every gate refuses each bad attempt in `CANARY` and admits a good one.

`research merge --train` (several refs, one checked candidate, bisected on a failure), `--train --prepare` (the
lander's train build, checked by `tools/check/train.sh BRANCH MACHINE`), the sweep of landed pull requests, the mirror
to node 1, and a tool's `Tool.merge_requires` and `Tool.after_land`: `merge.py`'s docstring states each.
`check`'s declaration of them is `tools/check/tool.py`.

`research queue` is the merge queue. Its loop, the verified queue (`research queue vqueue`, ci's: `vqueue_loop.py`,
policy `vqueue.py`), merges the PRs ready to land onto `main` as candidates (`queue/<sha12>`, one `Queue: #N` merge each),
checks each (graph2's tasks by default), and fast-forwards `main` to one with no new failure (`--gate verdict`: AGENTS.md). A PR is ready to land when it's open against `main` (drafts too), names its owner (a `Lane {name}` line, else
its Cursor agent id), has no hold label, is marked ready at its head by that owner (`research queue ready PR --mark --by
LANE`, or `--on auto` to run the quick tier first, or a `ready` label on `pr:{n}@{head}`), and has a reviewer's land verdict
there (`research queue review`), or changes nothing drive.toml's `lowrisk_excludes` names, or is a move's PR
the loop certifies. The mark's preflight refuses a lock merge dropping an entry the merged tree's statements need
(`lock_drops`). `research queue status` says what each PR waits for.

The workload receives `RESEARCH_RUN_ID`, `RESEARCH_RUN_DIR` and `OUT` (its `out/`, uploaded in full) and may write
`$RESEARCH_RUN_DIR/result.json` (see `result.py` for the envelope). A local workload runs in its run directory unless `--cwd`
says otherwise (`--cwd source` = the invoking directory, `clone` = a clone of it), so a relative `result.json` lands there and
never in the checkout; `--env K=V` is set for the workload only and recorded as the attempt's `params.env`, the values of
secret-named keys, of keys ending in `_HOST` and of `FLEET_JUMP` redacted. A brief names a machine, never its address:
`--env-host vy-challenger-1` gives a job `VY_CHALLENGER_1_HOST` from the register, `eval "$(research pods env vy-challenger-1)"` a shell.
Every job also starts with `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` (`result.env_defaults`, in `params.env` too) unless its
`--env`, its command or the runner's environment names it: oneDNN's AMX kernel returns a preempted bf16 product wrong now and
then. Four facts stay separate in `inspect`: exited, result well-formed, validation reported passed, preserved.
`--require-result` makes exit 0 without a valid result exit 65.

### Remote runs

`~/.research/machines.toml` (or `$RESEARCH_MACHINES`) lists the machines a
project may use:

~~~toml
[machines.vy-control]
provider = "runpod"
pod_id = "9tnzjcc6iygyv0"
project = "verity"
root = "/workspace/research"
python = "python3"
~~~

The pod registry (`src/research/pods/registry.py`) is the same table as one file per machine,
`machines.d/<name>.toml` in the notes repo, so creating a pod never means
hand-merging a shared file: `research pods create --name N --max-hours H ... --register
--project P`, or `research pods register N --pod-id ID --project P` for an
existing pod. `register` and `run --on` use one directory: `$RESEARCH_MACHINES_D`,
else `~/.research/notes/machines.d` (`none` turns it off, and `register` refuses).
The two sources merge, and a name they bind to different pods is refused.
`research pods registry` lists the merged table, and `registry import
--apply` seeds `machines.d` from machines.toml.

A machine that isn't a RunPod pod is `provider = "ssh"`: `host`, and optionally `port` (22), `user` (root) and
`rate_usd_per_hour` (the attempts' cost rate), reached with the same research key. `research pods register NAME --host ADDR
--user U --rate R --project P` writes it, refusing a host where `hostname` doesn't answer over ssh; `research pods ssh NAME`
reaches it. The Nebius nodes are such machines (`src/research/pods/nebius/`): `launch.sh` brings one up (`--dry-run`
needs no credentials), `teardown.sh` stops or deletes it, `n1_replacement.sh` plans n1's replacement, `check.py` is the
bootstrap check, `prefetch.py` fills the weight cache, and the lease is `src/research/pods/sh/lease.sh` (`LEASE_MACHINE_ID`, a stop command).
Runs share a node's GPUs through `gpu-lease N -- CMD` (`src/research/pods/sh/gpu_lease.sh`: a flock per GPU, `CUDA_VISIBLE_DEVICES` set).
`--cores CPUS` pins CMD to those CPUs and holds the leases to the machine's CPU quota: the script's header says how.

What runs on the Nebius nodes outside a run (infra's loops and the files they read) is listed in `src/research/pods/nebius/deploy.toml`,
each with its repository file and owner (a Slack handle), and goes there only through `research deploy install NODE --by WHO
[PATH ...] --apply` (else a dry run) from a pushed commit (`main`, or a PR's head for a hotfix).
Each loop is a systemd unit (`units/` beside the manifest, settings in `env/`), never tmux.
`research deploy drift`, `roll` and `attic` (what a node runs that isn't main's, a file made from the register of
machines, and a stray's way out), and where a replaced file's backup goes: `infra/ops/verity_ops/deploy.py`'s docstring.
A Lean build kept from node 1's credentials (the Lean cache's daily one) runs in `vy-lean-sandbox` (`lean_sandbox.py`).

Monitors (`research monitors list|check|run|relay`, `research/monitors.py`) wake their owners: node 1's `vy-monitors.timer`
writes an alert note per incident and per clear, and the relay posts them in `#agent-alerts`.

An unlisted or foreign machine is refused before anything is contacted.

How a run is shipped to the machine and admitted there is stated in the module that does it:

| What | Where it is stated |
| --- | --- |
| the shipped tool's identity check (`ResearchVersionMismatch`), the source ship and its `READY.json` (`SourceNotReady`), the git transport (`RESEARCH_SHIP=stream` skips it) | `src/research/remote.py` |
| the workload's directory (`--cwd source`, `--cwd clone`) and its `PYTHONPATH` (`research env --source TREE` prints it) | `remote.py`, `clone.py` |
| the claim by run id, a re-delivered launch, `--exclusive` | `remote.py` |
| `research cancel RUN --reason TEXT`, a queued run's holds and their withdrawal (`Withdrawn`) | `cli.py`, `withdraw.py`, `telemetry/scope.py` |
| extraction mtimes, and installed binaries' provenance (`BINARY.provenance.json`, job.json `run.binaries`) | `remote.py`, `binprov.py` |
| `--mem-gb` admission (`MemoryNotAvailable`), `research mem` | `mem.py` |
| a tool's preflight (`Tool.preflight`, `PreflightFailed`) and `--preflight TOOL`; `check`'s is `tools/check/preflight.py` | `remote.py` |

Never SIGKILL the runner: one killed during its custody upload leaves the run without custody and its minted key on disk.

With `--no-custody-r2`, the only copy of the evidence is on the machine until `fetch --all` completes. Provisioning is a
separate, explicit act; `--on` never creates a pod.

### Standing campaign targets: `research keeper`

`research keeper set|stop|status NAME --on NODE` keeps N chunks of a command leased on a Nebius node until a deadline, so no agent
VM stays awake to relaunch them (`src/research/pods/nebius/keeper.py`).

### Pods bound themselves: leases, one budget file, one guard

Nothing that bounds spend depends on an agent VM, which can be suspended at any moment: every pod carries a lease, and one
guard on the control pod enforces `budgets.toml` (`src/research/pods/README.md`).

### The harness: `research tele`

`src/research/telemetry/` is the recorded-attempt harness, standard library only; `research tele SUBCOMMAND` is `python3 -m
research.telemetry SUBCOMMAND` (`telemetry/cli.py`, whose docstring lists them). Each record of an attempt has its module:
`job.json` and `status.json`, its identity and state, with secret-named values redacted (`telemetry/job.py`); `events.jsonl`,
through one append helper (`events.py`); `resources.jsonl`, the sampler beside the workload (`sample.py`, reading `cgroup.py`
and `procs.py`); `failure.json`, how it ended (`classify.py`); and `RULES.md`, rendered from a policy file
(`telemetry/policy.py`). `run.py` is the harness itself: it starts the sampler and the workload, enforces the stage timeout and
records the end. `cancel.py` writes a cancel's intent before its signal and its result after, `scope.py` ends a run's processes
by their cgroup scope (what `research cancel` and a timeout stop), `pyspy.py` dumps a stalled run's Python stacks, and
`source_identity.py` records which source tree each imported package resolves to. Outside `research`, `infra/ops` reads
`telemetry.job`'s secret-name pattern, and the vLLM integration's tests import `source_identity`.

`telemetry/fixtures/run_fixtures.sh` checks the classifier against the real kernel, which `tests/test_telemetry.py` feeds
synthetic evidence: each case (`oom_own`, `timeout`, `guardian`, `bystander`, `forkshare`, `writer`, `sampler_killed`,
`sampler_denied`, `truncated`, `spans`) runs a workload under `tele run`, the memory hog, shared mapping, buffered writer and
span emitter beside it (`hog.py`, `forkshare.py`, `writer.py`, `spans.py`), and writes what `failure.json` concluded beside the
case's expectation to `OUT/SUMMARY.txt`. Nothing starts it. Run it by hand after a change to how an attempt's end is classified
or its cgroup read (`classify.py`, `cgroup.py`, `sample.py`), as root in a Linux pod of its own whose cgroup (v2) has a memory
limit, since its cases drive that cgroup to an OOM kill: `bash src/research/telemetry/fixtures/run_fixtures.sh OUT [CASE...]`.

### The Job queue: `research jobs`

`src/research/jobs/` is the client side of the Job queue, the control app's `/api/jobs`: batch, unattended work on pods, where a
job is added, claimed by a worker under a lease, renewed while it runs and completed with an outcome.

~~~text
research jobs add merge-check --commit REF      # or `lean-regen --commit REF --package DIR`, or `audit-main`
research jobs list [--state S] [--kind K] [--json]    research jobs show ID    research jobs cancel ID
research worker --dispatch --pods A,B [--once]  # claims jobs and runs each with `research run --on` on one of the pods
~~~

`research.jobs.model` is the queue's reference semantics; `vectors.json` beside it pins every rule as request and response
cases, which the site runs too. `research.jobs.client` talks to the site or to the model, with `$JOBS_TOKEN` for `research
jobs` and `$JOBS_WORKER_TOKEN` for the worker, neither ever printed. `jobs/cli.py` holds both commands, `jobs/dispatch.py` the
worker, and `jobs/trains.py` the model of the trains the service builds, checks and lands, whose gate `research merge --canary`
tests. Only `research` imports the package.

The disk floor: `research run` and `research fetch --all` refuse (exit 3) while the runs root's filesystem has less than
`$RESEARCH_DISK_FLOOR_GB` free (default 1.5; `0` turns it off), naming the eviction that frees it. `research data custody RUN
--publish` removes the run's files of 1 MiB and more once custody holds (`--keep-local` keeps them); `run-files.evicted.json`
names the fetch that restores them. `research data evict --src-dir DIR` (every node's unit) also deletes each
shipped source tree that is exactly a commit on a ref of the node's `git/verity.git`, unused and unchanged for 1 h; any doubt
keeps them all (`store/evict.py`). Its other passes: `store/leftovers.py`.

### research data

`src/research/store/` is the evidence store; its `README.md` is the contract
(four objects: Tool, Derivation, Attempt, Artifact; two relations: Replica,
Label).
Local root `~/.research/store` (or `$RESEARCH_STORE`); an optional remote in
`~/.research/store.toml` (or `$RESEARCH_STORE_CONFIG`: `fs` directory or
`s3`/R2 via stdlib SigV4). The sqlite catalog is disposable (`reindex`).

~~~text
research data put --kind proof/v1 --meta '{"backend":"sp1"}' --file proof.bin [--preserve]   # -> art:<hex>
research data show ART|RUN|DRV [--json]      research data where ART        research data fetch ART [--to DIR]
research data verify ART                     # PRESERVED = remote size + full read-back sha256; exit 1 otherwise
research data select --kind K --where meta.k=v --label k=v --tool T --derivation D --attempt R [--json|--ids]
research data snapshot --name N --members art:a,art:b | --select <select flags>   # dataset-snapshot/v1; identity = content
research data label ART verified accepted --by red-team-1 [--ref RUN]              research data labels ART [--history]
research data reindex [--remote]             research data sql "SELECT ..."       # read-only connection
~~~

Python: `research.store.local.LocalStore`, its methods listed in the contract.

### The evidence store: `--tool`, `--scratch`, and the publish step

What `research run` does:

- Before launch it collects `research.env.conditions()` (source tree, runtime,
  hardware, execution; `None` when unknown), resolves the harness `--input`
  paths to content identity, and, with `--tool NAME` (registry in
  `store/tools_registry.py`: `tc_probe`) or `--tool module:NAME`,
  loads the `Tool` declaration, parses the workload argv into params, replaces
  path-valued key params by the content they point at, hashes the declared
  closure, and computes the derivation id and the `reuse_key` (`None` on a
  dirty tree). Without `--tool` the derivation is the whole redacted argv and
  the `reuse_key` is `None`. All of it goes into `job.json` as the optional
  `store` section.
  An unknown tool exits 2 before anything runs; any other failure in this step
  is recorded in `store.error` and the workload still runs.
- `--scratch NAME` (repeatable) creates `<store>/scratch/<reuse_key>/NAME/`
  (`run-<run id>` without a reuse key) and exports `TRITON_CACHE_DIR` for
  `triton`, `RESEARCH_SCRATCH_<NAME>` otherwise. Cold per commit, on purpose.
- `--campaign C` (default `$RESEARCH_CAMPAIGN`) is recorded in the conditions
  and exported to the workload.
- After the workload exits, whatever its status, `store.attempt.publish`
  turns a valid `result.json` into an artifact (kind `result.kind`, default
  `bench-result/v1`), its `artifacts` + `measurement_files` into one
  `run-files/v1` tree, `events.jsonl` / `resources.jsonl` into `telemetry/v1`,
  and writes the attempt manifest (write-once). One line reports it:
  `research: attempt published: N outputs drv=... reuse=...`. A publish failure
  never changes the run's exit status: it is written to
  `store_publish_error.json` and `research data attempt publish RUN` retries;
  a run whose attempt names its run record (run custody, `--run-record`)
  retries with `research data custody RUN --publish`, since an attempt is
  write-once and `attempt publish` leaves the record out (it refuses a run with
  run custody) (also the way to publish old run dirs; `--all --runs-dir DIR` sweeps a runs
  root and prints `published / already present / refused (not terminal) /
  errors` with one line per error class). A configured-but-unusable remote
  (credential env vars unset here) does not stop a local publish: the attempt
  is written, `preserve_error` records why nothing was pushed.
- Repo replicas: `artifacts.json` maps `art:` ids to tracked files
  registered with `research data put --kind fixture/v1 --file PATH`; `where`
  lists the `repo` replica when the file still hashes to its id.
- With `--on`, the flags travel to the machine, which runs the same code in
  the shipped source tree; a `git archive` has no `.git`, so the control side
  sends the commit it shipped (clean) as `--source-identity`. The machine publishes
  to its own `<root>/store`; `research data pull --from MACHINE RUN --project P`
  copies the attempt and its outputs (sha256-verified) into the local store.
- `research data lookup --tool T [--param k=v]... [--condition k=v]...
  [--source-tree SHA | --any-source] [-- ARGV...]` answers "have we done this?"

### Retention of node data: `research keep`, `research retention`

`src/research/retention.py`: nothing under a node's `/workspace` is deleted without a record that allows it. A record
(`research/retention/v0`) is a sidecar, `<dir>/.retention.json` or `<file>.retention.json`. It covers everything beneath its
path. A directory its owner made read-only gets its record beside it, `<dir>.retention.json`, and
gc leaves a tree whole while any directory in it is read-only. A run dir whose `.custody` says `preserved: true` and which has no sidecar has an implicit record: owner its lane, else its
campaign, else `runner`; `preserved` its run record; `expires` 48 h after `verified_utc` (`RUN_EXPIRY`); priority `normal`. Nothing
is written for it, and a sidecar wins.

A finished or runner-dead run that custody did not preserve gets a low-priority retention record that expires 7 days after the run
directory's last write, so the collector takes it, its only copy (`UNPRESERVED_RUN_EXPIRY`). So does a refused launch.
One running, queued or in a slot line stays; one whose clone stands waits (`scratch/<run id>` goes first).
To keep one: `research keep RUNDIR ...` before the 7 days are over, or `research data custody RUN --publish` if it ran.
gc takes them only with `--unpreserved-runs`; it and the report list them apart.

A launcher's copy of a run on another machine (a directory named as the run's id that holds the launcher's `remote.json`) has no
record, except under gc's `--launcher-copies`: there it expires 48 hours after its last write, and goes only once the remote holds
the run's attempt, ended, and a run record with the copy's files (the rule is in `retention.py`'s docstring). A pass under
`--launcher-copies` deletes launchers' copies and nothing else: any other path it would take is listed as left. `research keep
COPY ...` keeps one longer. The unit `vy-launcher-copies-gc` runs it hourly over root's runs root on vy-nebius-cpu-7.

~~~text
research keep PATH --owner @LANE --expires DATE|never|7d --priority P [--regenerate CMD] [--art ART | --preserve] [--why T] [--by URL]
research keep show PATH    # the record covering PATH, explicit or implicit, as JSON
research retention report [--root /workspace] [--depth 2] [--json]
research retention gc [--root /workspace] [--force] [--high-pct 80] [--low-pct 75] [--max-paths 200] [--jobs-dir D]... [--store S] [--unpreserved-runs] [--launcher-copies] [--json] [--apply]
research retention rm PATH... --approved-by @OWNER --ref REF [--by URL] [--root /workspace] [--jobs-dir D]... [--json] [--apply]
~~~

`--preserve` puts the path in the store as `node-data/v1` and pushes it PRESERVED (the store README's §7) before recording its id;
a failed push records nothing. The report lists every recorded
path and every area in the top `--depth` levels with no record beneath it (`unowned`).

Policy `default` (`default/v0`) acts only over `--high-pct` of blocks or inodes, or with `--force`. It deletes
the lowest priority, then the oldest expiry, until the filesystem is under `--low-pct`, at most `--max-paths` paths a pass. A
path qualifies only when its record is valid, of this node, expired, `low` or `normal`, and preserved or regenerable (or an
unpreserved run's, or a launcher's copy's), it holds no nested record, and no file in `/workspace/pouw/fill/queue`, `/workspace/pouw/fill/running` or a
`--jobs-dir` names it. Just before `--apply` deletes a path, its record and the job scripts are read again. A preserved path also needs its artifact to hold every regular file
now under it at its size (but for the uncovered run below), and its stored copy `verified`, `expired` or `withdrawn` on the remote; one that can't be asked about
fails the pass (`retention.STORED`). A preserved run's marker must still vouch for the run dir (the store README's §10.1). Without
`--apply` gc deletes nothing and reads no remote; it prints every deletion and skip with its reason.

A symlink or other entry that isn't a regular file holds no bytes, so the artifact need not hold it (custody never stores one).
A preserved run whose run record lacks files custody did not store (it trimmed them past its limit: `uncovered`) holds their only
copy, as a never-stored run does: gc deletes it, under `--unpreserved-runs`, 7 days after custody verified the run (the marker's
`verified_utc`; never the last write, which the store eviction moves), once the remote holds the rest, and lists it with that date
until then, with how to keep it: `research keep RUNDIR --owner @LANE --expires DATE --priority P`, or a custody that declares the
files (`--declared-output`); a run with a keep record never goes this way. Each deleted run's line says how much of it was
stored nowhere else, and the pass's summary counts these runs apart from the never-stored and the covered ones.

`rm` is the one way to delete data you don't own on a node, for a cleanup the policy doesn't reach. It deletes a path only when a
record covers it (its own or an ancestor's), and that record and every record beneath it (sidecars and implicit ones) name the
`--approved-by` handle as owner and are not `keep`. No job script may name the path, and nothing in it may be read-only or a mount. Each
decision of an `--apply` (delete, refuse, failed) is a `research/retention-log/v1` line in `/workspace/research/retention/deletions.jsonl`
with `--by` (default `$RESEARCH_AGENT_URL`), the approver and `--ref`.

### Slack

`research msg` (`slack.py`) is the coordination API on Slack: a handle posts with
`send` (`--to` names lead it and are woken; `--kind announce` for an announcement, `--kind fyi` or `--quiet` for a note that
needs no reply) and closes a thread it was asked in with `close` (a status line, ✅, or ❌ with `--declined WHY`). The
conventions (only handle holders are on Slack, who subscribes to what, the norms) are the `using-slack` skill, and
`.agents/skills/using-slack/registry.json` names the handles (user groups, and the paths each `owns`), the channels and the
humans the commands check against. It speaks the Web API over `urllib` with `$SLACK_BOT_TOKEN`, never prints a token, and
`$SLACK_API_URL` points it elsewhere: `tests/test_slack.py` runs every subcommand against a fake Web API on a local port.
`--dry-run` sends nothing and prints each request. Every handle posts through the one bot under its own username, so the
username is a label: `verify-author` proves a post is Verity's bot, a listed alerting bot or a named human, not which handle
wrote it. The same bot can place each reaction on a message only once, which is why an announcement's addressees answer with
status lines rather than reactions. `groups sync` is a plan until `--apply`, which then writes the groups' ids into the
registry.

The bot needs `chat:write`, `chat:write.customize` (the handle as the username), `groups:history` (reading threads in private
channels; `channels:history` for a public one), `reactions:write` (`close`), and for `groups sync` `usergroups:read` and
`usergroups:write`. It must be a member of both channels (`chat:write.public` does not reach private channels), and Cursor's
Slack app must be in both too: Cursor delivers the wakes, this bot posts.

Routing: each handle holder subscribes to its own inbox thread, not to the top level. The sender rings the doorbells of
the names a post addresses right after posting, by the rule in `route_targets` (`slack.py`), which
`tests/slack_routing_vectors.json` pins. Only posts by Verity's bot (the registry's `bot_id`) and by the named humans route:
another app's bot posting under a handle's name rings no one.

### Notes and evidence over MCP

`research notes mcp` (`notes_mcp.py`) serves a notes clone and the evidence catalog to an agent's MCP client over stdio; it hosts
nothing and reads no credential. Its tools: `search_notes` (full text ranked, or the newest notes, by lane, kind and time; each
hit says how many of its cited ids the catalog holds), `read_note` (by path or id, in any form notes cite each other by, a path from
before a move into a month folder included), `evidence` (for a run id, an `art:` prefix or `note:<id>`: what
the catalog holds, and the notes citing it) and `sql` (read-only, over the notes index and the catalog attached as `ev`). The
index (`notes_index.py`) catches up with the clone before every call, re-reading only the notes that changed; `research notes
index` builds it, or answers `--search` / `--evidence` in a shell. The catalog is the local store's `catalog.sqlite`, or
`--catalog` names a copy made by `research data catalog --compact`: the same tables without the whole manifests, for a machine
that queries evidence without keeping a store. The notes root defaults to `$RESEARCH_NOTES` or `~/.research/notes`. In Cursor,
`.cursor/mcp.json`:

~~~json
{"mcpServers": {"research-notes": {"command": "uv", "args": ["--directory", "/workspace", "run", "research", "notes", "mcp"]}}}
~~~

### Notes stamped ahead of the clock

`research notes sync` (`notes_sync.py`; `checkpoint`, `bind` and `relaunch` run it with `RESEARCH_NOTES_SYNC=1`) leaves out any
lane note it would add or change whose filename stamp (a new path) or front matter `created:` (a new value) is more than 5 minutes
ahead of the clock, syncs the rest, and exits 5. Each note left out gets one line, from the steward's watch pass too, every
pass until the note is renamed or the clock passes its stamp:

~~~text
SYNC-REFUSED lanes/x/20260928T0700Z-report-x.md: stamp 20260928T0700Z is 18 min ahead; use 20260928T0642Z (date -u +%Y%m%dT%H%MZ)
~~~

Once pushed, it looks up the `art:` ids and run ids the pushed lines cite on the store's remote, which is what every reader
resolves, and warns, never refuses, for each note citing one the remote lacks. A run counts once its attempt, or a label on it
(`custody waived`), is there: each `SYNC-WARN` line names the command that puts it there. `RESEARCH_NOTES_CITE_CHECK=0`
turns it off.

### The steward off the laptop

`research notes watch` is the steward: liveness, reaping, deadlines,
budgets and scheduled runs (`notes.py`). Three flags let it run on any
always-on host (`steward.py`):

- `--liveness git` (or `RESEARCH_LIVENESS=git`): a notes file's time is
  its newest notes commit, not its mtime, so a fresh clone gives the same
  rows as the laptop. A lane's branch signal is origin's `lane/*` tip
  (`--fetch` refreshes it), and the local worktree isn't read. `relaunch`
  bases the successor on origin's tip when the dead lane's machine is gone.
- `--lease [store|PATH]`: only the holder of `steward/lease.json` in the
  store's bucket reaps, routes, renders, keeps `steward-state.json` and
  commits the notes. The lease is changed by conditional PUT only. A second
  copy refuses (exit 2) and names the holder; with `--standby` it watches
  read-only and takes over when the lease frees.
- `research notes service systemd|launchd` prints the service definition
  (restart on exit and after a reboot; `watch --lease --standby`).

Its schedule is `<notes root>/steward.toml`, re-read every pass: `[[run]]` entries, each a recorded run launched on a machine once a UTC day at or after `at`, or once an ISO week with
`weekday` (`run_lines`):

~~~toml
[[run]]
name = "lean-upstream"                       # its lines' name, and the key of its state
at = "10:30Z"
weekday = "mon"                              # weekly; daily without it
source = "/workspace/steward/verity"         # the tree it is launched from: its cwd, and its workspace on PYTHONPATH
args = ["--on", "vy-lean-small", "--project", "verity", "--source", ".", "--cwd", "source", "--tool", "lean_upstream", "--",
        "python", "tools/verity/lean/upstream.py", "security/Proofs", "--fetch", "--rev", "origin/main"]
~~~

`args` are `research run`'s and must put the run on a machine (`--on`). The steward runs `research run --id <run id> <args>`,
which returns once the machine has started the workload detached, so a pass waits for the launch (at most 10 minutes), never
for the run. It prints `LAUNCHED <name> <run id> on <machine>`, or `RUN-FAILED <name> <run id>: <why>`, and the next launch is
the next period's; an invalid entry is one `RUN-FAILED <name>: <why>` line. The run's Attempt records what ran and how it
ended (`research inspect <run id>`). The Lean runs it schedules are declared in `tools/verity/lean/tool.py`: `lean_audit`
(`check.py --all --build --fresh`, nightly on `main`) and `lean_upstream` (the entry above, weekly).

### A lane's older notes

`research notes archive LANE` (`notes_archive.py`) moves the lane's notes stamped more than `--older-days` (3) days ago into
`lanes/<lane>/<YYYY-MM>/` (`--by day`: `<YYYY-MM>/<DD>/`) in one commit, which `research notes sync` pushes; without `--apply` it
prints the plan. It keeps reports, unstamped names, anything untracked or changed since the last commit, and mail the lane's
`inbox` has not shown (all mail, in a clone without the lane's `.inbox-seen`, which git ignores: run it where the lane reads its
inbox); it overwrites nothing, and a failure puts every file back. `inbox`, `status` and `mail` read only a lane's own folder, so
that folder stays its working set; `git log --follow` finds a moved note's history.

### Approaches: check, then claim

The registry of what has been tried, what is live and what was killed and why (`approaches.py`; the notes' `kb/onboarding.md`
is the how-to). An approach is a record in the evidence store, kind `approach/v1` with meta `{campaign, slug}`, so its `art:` id
is a function of its name `<campaign>/<slug>`. Its title, hypothesis, owner lane, status (`live | parked | killed | superseded |
merged`), reason and evidence are labels in the vocabulary's `approach` group. A run joins one with `research data label RUN
approach <campaign>/<slug> --by LANE`.

~~~text
research notes approaches [--campaign C] [--status S] [--grep WORDS] [--refresh]   # the registry
research notes claim C/SLUG --lane L [--title T --hypothesis H --type scheme|route|attack] [--cite TOKEN]... [--reopen WHY]
research notes approach C/SLUG [STATUS --reason R] --by L [--cite TOKEN]... [--uncite TOKEN]... [--superseded-by C/SLUG] [--owner L] [--at UTC]
research notes approaches render | check                                           # campaigns/<c>/APPROACHES.md; the lint
~~~

- **`claim` refuses** (exit 2, nothing written) an approach that is live under another lane, and names the owner. It also
  refuses a killed, superseded or merged approach without `--reopen WHY`. It writes through to the remote and re-reads it, exiting
  3 when a rival claim landed first.
- **`approach --at`** records work done before the registry existed. History is append-only.
- **`approach` without a status** changes only cites (`--cite`, `--uncite`).
- **The public view** is one generated file per campaign, `campaigns/<c>/APPROACHES.md` in the notes. The steward's
  `[approaches]` rule in `steward.toml` (`store`, `every_min`) refreshes and renders it. The campaign's `BRIEF.md` front matter
  `registry: titles | full` sets whether hypotheses and reasons are shown, since the notes repo is public.
