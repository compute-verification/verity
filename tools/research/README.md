# research

Shared tooling for executing, observing, and coordinating computational work
during development. One installable package, one CLI (`research`), stdlib
only. It runs commands and records what happened; it imports nothing from
Verity's packages (enforced by `tests/test_cli.py::test_no_packages_import`).

The user-facing procedure (when to record a run, how to find one, how to hand
off) will be a repo-level skill, once it exists. This file is for people changing
the tool.

## Layout and provenance

~~~text
src/research/
├── cli.py          research run | fetch | status | inspect | tele | data | launch-request
├── runs.py         run ids, the local registry (~/.research/runs or $RESEARCH_RUNS), read-time summaries
├── record.py       job.json v0.2 = the harness's v0.1 + tool, run, workload_fingerprint.expected, cost; v0.1 stays readable
├── result.py       the result.json contract (schema, run_id, fingerprint, validation, measurements, artifacts)
├── remote.py       --on: machines file, tool/source snapshots, request, claim-by-run-id, exclusive lock, fetch, preservation manifest
├── disk.py         --disk-gb admission (--queue): held on the machine while the disk would pass 80%, released at 75%
├── mem.py          --mem-gb admission: cgroup limit - in use - live /workspace/ramlock reservations; MemoryNotAvailable
├── auto.py         --on auto: a CPU node's research pool (cpu-1, -3, -5 to -9) or node 2's queue
├── jobuser.py      --job-user: the workload as an unprivileged user in a sandboxed system unit
├── env.py          conditions(): the one baseline of an attempt's conditions (source, runtime, hardware, execution); ONE SKU->family map
├── store/          the evidence store (`research data`; README.md there is the contract): tool, tools_registry, derivation, attempt,
│                   cli_prov (provenance) + canon, ids, model, local, index, remote, remote_s3, kinds, cli (storage)
├── telemetry/      the recorded-attempt harness: job.json, status.json, events.jsonl, resources.jsonl, failure.json
└── pods/           RunPod lifecycle (runpod.py), leases (lease.py), budgets.toml and its guard (budgets.py, guard.py), pod-side sh/
tests/              pure-Python; the machine side of --on is exercised locally (launch_request is a plain function)
tests/fixtures/     job_v0.1.json: a record as the harness alone writes it; run_v0.2/: a finished run dir with a `store` section (state.json -> status.json at test time)
~~~

The `telemetry` and `pods` packages were promoted from `veritor` `zk/campaign/` at
commit `df7fbea3d` (branch `lane/r17-mprof`, 2026-09-21) and have since been
edited here like the rest of the package.

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

No value is ever printed. `research data mint-credential` already read the
same variables.

On a cloud VM a long-lived shell can lose the injected secrets that a fresh
tmux login shell still has. Every `research` command first sets each secret
named in `$CLOUD_AGENT_INJECTED_SECRET_NAMES` and missing from the shell from
the tmux server's global environment, where that login shell gets them
(`secret_hint.recover`); one run under a test doesn't, so a secret the test
unset stays unset. If one is still missing, the store, `research queue
ready`, node ssh and the Nebius scripts fail and say `rerun from a fresh tmux
login shell` (`secret_hint.py`). A shell can lose that list too, so a store
that has used the remote and now finds none configured fails `research data
label` and `put` the same way (`LocalStore.remote_lost`).

`research doctor [--role lane|coordinator|infra] [--json]`: can this VM do its job (secret names, status-only
probes, ssh, `uv sync --check`, disk, clock); each finding names its fix (exit 1); it never prints a value.

## Commands

~~~text
uv sync                                  # from the verity root
uv run pytest                            # all tests

research run [--project P] [--stage S] [--timeout T] [--expect K=V] [--require-result] [--env K=V] [--cwd DIR] [harness flags] -- CMD...
research run [--tool NAME|module:NAME] [--scratch NAME]... [--campaign C] [...] -- CMD...      # evidence store, below
research run --on MACHINE --project P [--source GITDIR] [--send FILE] [--env K=V] [--exclusive] [--mem-gb N [--mem-wait S]] [--preflight TOOL] [--wait [--wait-max S]] [...] -- CMD...
                                         # --wait: 20 min at most, then exit 79 (research/waitcap.py)
research run --on MACHINE [--custody-ttl 6h] ...    # the default: the machine puts the attempt + every run file on the store's remote (store README §10.1)
research run --on MACHINE --no-custody-r2 ...       # opt out: the records stay on the machine until `research fetch --all`
research run --queue --source GITDIR --project P [--kind K] [--question Q] [--mem-gb G] [--disk-gb N] [--max-min M | --timeout T] [--gpus N]
             [--gpu-model M] [--cpus N] [--class work|bench|...] [--phase cpu|gpu] [--preemptible] [--quiet] [--on NODE] -- CMD...   # the one queue (infra/cluster): placed, then an --on run in its CPU allocation, a GPU leased around the workload alone
research data custody RUN [--publish] | --triage                 # a run's custody on the remote, from any machine; the runs root by custody state
research fetch RUN [--all]               # records only, or every file sha256-verified with preserved.json written last
research status [RUN] [--refresh] [--all] [--json]   # under each active run, a `mem` line per cgroup it executes in (scopemem.py):
                                         # current, peak, cap, growth from its own samples, NEAR CAP within 10%; remote: with RUN or --refresh
research inspect RUN [--json|--py-spy]   # fetch, status RUN and inspect take a run another VM launched (a full id): found on
                                         # --on NAME, else on each ssh machine, and its records fetched here
research data SUBCOMMAND ...             # the store: put/show/fetch/select/... and lookup / pull / attempt publish
research pods create --name N --max-hours H ... | list | terminate | check-part    # RunPod lifecycle (pods/runpod.py, pods/part.py)
research pods extend TARGET --hours H | lease TARGET                               # the pod's lease (pods/lease.py)
research pods guard [--budgets NOTES|FILE] [--detach] | status | stop               # the one budgets guard (pods/guard.py, pods/budgets.py)
research pods register|unregister|registry ...                    # the pod registry, one file per machine (pods/registry.py)
research pods ssh TARGET [--print] [-- CMD...]                    # TARGET = machines.toml name | pod id | pod name (pods/connect.py)
research pods sync TARGET [SRC] [--dest DIR] [--exclude G]...     # the worktree's git-visible files -> DIR (default /workspace/src)
research tele SUBCOMMAND ...             # the harness's own surface
research data SUBCOMMAND ...             # the evidence store (below)
research merge REF [--into main] [--tool check] [-m MSG] [--push] [--dry-run]   # the merge gate (merge.py)
research merge --train A B C [--on MACHINE --project P] [--command CMD] [--push]  # several refs, one checked candidate
research merge --train --prepare BRANCH [--onto BASE] [--allow-removal PATH] N@SHA ...  # build, lint, push a train
research merge --sweep [--into main] [--remote origin] [--dry-run]       # close the PRs that landed, retarget their children
research merge --mirror [SHA]                                           # main or SHA to node 1's bare repo
research merge --canary [--tool check] [--json]                    # the gates' daily self-test, in scratch stores
research queue status|sync|land|eject|drive|ready|open [--dry-run]  # the merge queue on `next` (queue.py)
research queue judge RUN [--by LANE]      # the mark `queue ready --on` writes, from its quick-tier run that ended after its caller died
research queue stats --log FILE [--hours 24] [--until T] [--json]  # where landed PRs' time went (queue_stats.py)
research git old2new|remap|check|build ...                              # after a history rewrite (commitmap.py, below)
research msg send|close|read|list|names|traffic|owners|handles|subscriptions|ask-daniel|groups sync [--dry-run]   # Slack (slack.py, below)
~~~

After a history rewrite, an accepted `commit-map/v1` (labelled `commit_map=accepted`) lets an attempt of an old commit count for
its image when the tree is unchanged, and the gate refuses a REF holding a commit the rewrite replaced. `research git remap`,
`old2new SHA`, `check REF` and `build PRE POST` move clones onto it, name images, guard and build the map (`commitmap.py`).

`research merge` is how `main` receives a merge: in the target's checkout it runs `git merge --no-ff REF` only when the store
(refreshed from the remote first, when one is usable) holds an attempt of the tool, default `check`, with `state` done, rc 0,
and a clean source at REF's exact commit, and when the target's tip is an ancestor of REF, so the merged tree is the checked one.
The attempt's own record and result must agree with the target's declaration: its command (`Tool.command`, in a check slot or
not, plus check.py's own flags only), a result kind it produces (`check/v1`, never `lean-fast/v1`), a passed verdict and no step
listed twice. Anything else, unreadable included, passes no gate, and a refusal exits 3. The merge commit carries a `Check:
<attempt id>` trailer; `research` only reads the attempts of the repository's tool (`tools/verity/check`). `--canary` checks, in scratch
stores, that every gate refuses each bad attempt in `CANARY` and admits a good one.

`research merge --train A B C` merges the refs in order onto the target's tip in a scratch worktree, one `--no-ff` merge each
(so its commits are the train's prefixes), records the tool on the last (`research run --tool`, `--on` a machine), and when it
passes fast-forwards the target to that exact tree. On a failure it bisects the prefixes to the first ref whose merge breaks it
and fast-forwards to the longest prefix that passed.

`research merge --train --prepare BRANCH [--onto BASE] N@SHA ...` is the lander's train build: the same chain, in a scratch
worktree, onto the remote's `--into` or a train ahead's prep branch (`--onto`). An `N@SHA` or `BRANCH@SHA` whose head moved is
refused, and the build stops at the first ref that doesn't merge. check's preflight (`check.py --preflight`, the markdown caps
among its tests) runs on the tip, which is pushed as BRANCH, never forced. Nothing is recorded or landed:
`tools/verity/check/train.sh BRANCH MACHINE` checks it (`auto`: the node
`check.py --slots` picks), and `research merge BRANCH` lands it.

After a merge or a train lands, `research merge` sweeps the pull requests (`gh`; `--sweep` does only that, `--dry-run` prints
it). From the remote's copy of the target, the truth rather than the checkout, it closes each open pull request whose head is on
it, naming the commit, retargets each whose base branch's tip is on it (stacks GitHub never closes), and deletes a closed one's
head branch unless an open one uses it. With `--push`, which whoever lands a train passes, a merge that lands is
pushed, never forced, before the sweep, and a refused push sweeps nothing and exits 1; without it, run `research merge --sweep`
after `git push`. No remote skips the sweep, and its failure leaves the exit code alone. Then `main` goes over ssh to node 1's
bare repo (120 s; `RESEARCH_MERGE_MIRROR`: host, empty skips); a failure logs the `--mirror` rerun.

A tool's declaration may name result steps a merge needs by path (`Tool.merge_requires`, `{step: [path prefixes]}`, read from
the target's checkout): when the merge's diff touches a prefix, only an attempt whose `result.json` `steps` show that step
`passed` admits it; `!` patterns leave paths out (`!README.md`: every file of that name under the prefixes). `check` requires
`lean-agreement` for `verity/{ml,core/service}/flock/`, less README.md and PROTOCOL.md; `--train --send FILE` ships what such a
step needs. It may also name what runs after a landing (`Tool.after_land`, an argv): once `--push` has pushed, it starts detached
in the checkout with the landed commit appended, and can't change the exit code; `check`'s is its `post_train.py`, the vLLM
regression's manifest rebuilds, which label the landing's check attempt.

Since 12 Oct `research queue drive` moves `main`; the lander handles only what drive refuses.

`research queue` is the merge queue: `next` is `main` plus every admitted PR, one `--no-ff` merge commit each, and `land`
fast-forwards `main` to the newest commit on `next` that the same gate accepts. A PR is admitted when it's open (drafts too),
names its owner (a `Lane {name}` line, else its Cursor agent id), has no hold label, and is marked ready at its head by that
owner (`research queue ready PR --mark --by LANE`, or `--on auto` to run the quick tier first, or a `ready` label on
`pr:{n}@{head}`). No PR waits on a person's approval or on a per-head grant. The branch holds the queue's state (each
merge commit's `PR:`, `Head:` and `Owner:` trailers); a sync keeps the prefix that still matches and merges the rest again, and a
conflicting or ejected PR's owner gets one handoff per head. `--shadow` builds `queue-shadow`, never moves `main` and tells no one.

The workload receives `RESEARCH_RUN_ID`, `RESEARCH_RUN_DIR` and `OUT` (its `out/`, uploaded in full) and may write
`$RESEARCH_RUN_DIR/result.json` (see `result.py` for the envelope). A local
workload runs in its run directory unless `--cwd` says otherwise (`--cwd source` = the invoking directory, `clone` = a clone of it), so a relative
`result.json` lands there and never in the checkout; `--env K=V` is set for the
workload only and recorded as the attempt's `params.env` (values of
secret-named keys redacted). Four facts
stay separate in `inspect`: exited, result well-formed, validation reported
passed, preserved. `--require-result` makes exit 0 without a valid result
exit 65.

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
existing pod. `register` refuses a pod id the RunPod account doesn't know and
a pod whose name isn't `N`. `register` and `run --on` use one directory: `$RESEARCH_MACHINES_D`,
else `~/.research/notes/machines.d` (`none` turns it off, and `register` refuses).
The two sources merge, and a name they bind to different pods is refused.
`create` makes at most one live pod per `--name`, so a retry loop can't duplicate pods. It refuses when a live pod of that
name exists (exit 4, `--allow-duplicate` overrides). After a failed create call it uses a pod that appeared under the name
anyway. Only RunPod's explicit no-stock reply is exit 5, the one failure to answer with another spec; any other failure is
exit 2.
`research pods registry` lists the merged table, and `registry import
--apply` seeds `machines.d` from machines.toml.

A machine that isn't a RunPod pod is `provider = "ssh"`: `host`, and optionally `port` (22), `user` (root) and
`rate_usd_per_hour` (the attempts' cost rate), reached with the same research key. `research pods register NAME --host ADDR
--user U --rate R --project P` writes it, refusing a host where `hostname` doesn't answer over ssh; `research pods ssh NAME`
reaches it. The Nebius nodes are such machines (`src/research/pods/nebius/`): `launch.sh` brings one up (`--dry-run`
needs no credentials), `teardown.sh` stops or deletes it, `n1_replacement.sh` plans n1's replacement, `check.py` is the
bootstrap check, `prefetch.py` fills the weight cache, and the lease is `sh/lease.sh` (`LEASE_MACHINE_ID`, a stop command).
Runs share a node's GPUs through `gpu-lease N -- CMD` (`sh/gpu_lease.sh`: a flock per GPU, `CUDA_VISIBLE_DEVICES` set).

What runs on the Nebius nodes outside a run (infra's loops and the files they read) is listed in `src/research/pods/nebius/deploy.toml`,
each with its repository file and owner (a Slack handle), and goes there only through `research deploy install NODE --by WHO
[PATH ...] --apply` (else a dry run) from a pushed commit (`main`, or a PR's head for a hotfix). Each loop is a systemd unit
(`units/` beside the manifest, settings in `env/`), never tmux; `install` restarts a running service whose unit, env file
or running file changed, once the node is `quiet`. A `hold` entry is reported, never installed.
`research deploy drift` reads each node and names every file that isn't main's (`stale`, `ahead`, `unknown`,
`missing`), every process running a file replaced since it started (`restart`), and every file in a managed directory
that no entry lists (`stray`); the hourly preflight probe posts its findings whenever they change (`ops deploy`). A new
loop or script goes into the repository and the manifest before it runs; a backup of a replaced file is
`.deploy-prev/`, which `install` keeps, never a `*.bak-*` beside it. A stray leaves through `research deploy attic NODE PATH ...
--by WHO --apply` (`--backups` for every backup), which moves it under the record's directory, recorded, so `sudo mv` restores it.
A Lean build kept from node 1's credentials (the Lean cache's daily one) runs in `vy-lean-sandbox` (`lean_sandbox.py`).

Monitors (`research monitors list|check|run|relay`, `research/monitors.py`) wake their owners: node 1's `vy-monitors.timer`
writes an alert note per incident and per clear, and the relay posts them in `#agent-alerts`.

An unlisted or foreign machine is refused before anything is contacted. The
tool ships as a content-addressed source snapshot under `<root>/tool/` and
runs via `PYTHONPATH`; no pip, no credentials on the machine. The machine
checks that the `research` package that answers is the one shipped (its content
identity): `launch-request` checks before it claims the run, and the runner
checks again before the workload starts. On a mismatch the run is refused
with `ResearchVersionMismatch`, naming the running package's path. A package
too old to know the check fails closed. The one exception: a `--source`
tree's own older package runs unverified, and the launcher says so. job.json
records `run.remote.tool_running`. `launched` waits for the runner's job.json; a
runner that exits first is refused (exit 3). `--source` ships
`git archive HEAD` of a clean checkout to `<root>/src/<sha>/` (a dirty tree is
refused, not mislabeled). The ship is verified before anything may run from
it: the archive lands in `<sha>.partial-<nonce>/`, its sha256 is compared on
both ends and the extracted file set against the archive's manifest, then the
directory is renamed into place carrying `READY.json` (identity, archive
sha256, file count, launcher). A failed or truncated transfer removes its
partial directory and the run is refused (`SourceNotReady`); the machine
refuses to execute from a source directory without a `READY.json` naming the
requested identity. The attempt records `execution.remote.source_sha`
and `source_archive_sha256`.

The commit travels as git objects when it can: the launcher `git push`es it
into the machine's bare repository `<root>/git/verity.git` through its own ssh
argv (`remote.git_push_url`, an `ext::` URL; no credential anywhere), so only
the objects the machine lacks cross the uplink (the first push to a new pod
carries the history). The machine checks the commit's tree hash, extracts
`git archive` of it, and lists every file's sha256, which must equal the
launcher's archive manifest (`READY.json` `verified:
git-push+tree+file-sha256`, `transport: git`; job.json
`run.remote.source_transport`). A machine without git, or any failure of that
path, prints one line and ships the archive stream as above; nothing is ever
marked READY by a failed git ship. `RESEARCH_SHIP=stream` skips git. The shipped tree is the runner's source root, not
the workload's cwd: the workload runs in its own run directory, as locally,
because every run of one commit shares `<root>/src/<sha>/` and relative outputs
used to race there. `--cwd source` (or `source/SUB`) runs it in the tree, and
`--cwd clone` (`clone/SUB`), for a workload that writes there, in its own clone
of it, removed when it exits (`research/clone.py`); job.json records `run.cwd` and `run.cwd_basis`. The
runner and the workload get the tree's `PYTHONPATH` derived from its root
`pyproject.toml` workspace members and each member's build config
(`src/`, `python/`, flat), with a user `--env PYTHONPATH` appended; job.json
records it (`run.remote.pythonpath`). `research env --source TREE` prints the
same line for a local checkout. The machine claims the run directory by run id and
starts the runner detached; re-delivering a launch after an ssh drop finds
the claim and reports state. The runner's argv carries the workload's command, so a
`pkill -f` aimed at the workload matches it too: it logs SIGTERM/SIGHUP to
launcher.log and keeps supervising, so the run still records its end and
publishes. Stop a run with `research cancel RUN --reason TEXT` from any VM: SIGTERM to its workload's process group
(status.json `pgid`), its cgroup scopes (`cluster-<run>`, `gpu-lease-<pid>`) and its processes outside the group,
recorded first as a manual cancel (`CANCELLED_MANUAL`). 20 s later the runner SIGKILLs what is left, recorded in status.json
`scope_end`; so does a stage timeout (`telemetry.scope`); if the runner already logged the workload's exit or is gone, the
cancel does. A run still in `disk-hold` or `cpu-hold` is withdrawn: its hold ends as a manual cancel and
refused.json says `Withdrawn` (`withdraw.py`). Never SIGKILL the runner: one killed during
its custody upload leaves the run without custody and its minted key on disk. `--exclusive` refuses while any run is live, and
any launch while an exclusive holder is.

The shipped tree is extracted with extraction mtimes (`tar -m`), not the commit time `git archive` stamps: a build dir shared
across commits (Cargo's `/workspace/sp1-target`) judges freshness by mtime, and with commit times an older commit shipped after a
newer build looked built and ran the newer binary. Installed binaries (`/workspace/bin`, or `$RESEARCH_BIN_DIRS`) carry their
own provenance: a build runs `python -m research.binprov record BINARY --source TREE` after installing (the SP1 and Ligero
bootstraps do), writing `BINARY.provenance.json` (sha256, source commit/tree/dirty), and every run records each binary's sha256
with its provenance `recorded` / `stale` (replaced since) / `missing` in job.json `run.binaries` (`binprov.py`).

`--mem-gb N` is memory admission on the machine (`mem.py`). A pod's processes
share one memory cgroup, and `free` describes the host (vyv-v2cpu3: 946 GB
host, 256 GB cgroup). The machine admits the run only if the cgroup limit, minus
what is in use (page cache excluded), minus the not-yet-used part of every live
reservation in the fleet's `/workspace/ramlock/*.json` (machines.toml
`ramlock = DIR` overrides it), is at least N. They count against the machine's
outermost limited cgroup (a node's user slice), not the run's. A reservation is live unless its
pid is gone or its run is terminal. The run then writes
`research-<run id>.json` there, and the runner removes it at exit. Otherwise the
run is refused with `MemoryNotAvailable`, and `refused.json` gives every number.
`--mem-wait S` polls for up to S seconds instead. job.json records the
admission (`run.remote.mem`) and the attempt records
`execution.mem_gb_reserved`. `research mem` prints the reading on the machine.

A tool can declare a preflight (`Tool.preflight`: a Python script in the tree
and its args), which says whether a machine is prepared for it. When a
`--tool` run's shipped tree declares one, `launch-request` runs it there,
in the tree, with the runner's environment, before anything else is
admitted. `RESEARCH_PREFLIGHT_INPUTS` names the run's `--send` files and,
with custody, `RESEARCH_PREFLIGHT_CUSTODY` the staged key's directory. It
runs under the interpreter `research` itself runs on, since it may be
checking what else is installed. On exit 0 the run starts, and job.json
records the report (`run.remote.preflight`). Any other exit refuses the run
(`PreflightFailed`, exit 3, `refused.json`), with the script's stderr as the
reason. Nothing starts and no attempt is recorded, so a refusal is an error,
never a verdict, and `research merge --train` stops rather than bisecting. A
declaration that exists but cannot be read also refuses. `--preflight TOOL`
runs TOOL's preflight before a run of something else, such as a Lean record
regeneration on a check pod, and refuses if the tree declares none. `check`
declares `tools/verity/check/preflight.py`: the tools `pod_setup.sh` installs, the
Lean dependencies (warm, or restorable from the store, never GitHub), one
store read, `avx512f` when the run carries an upstream build that needs it,
and free disk.

Limitations, stated: until `fetch --all` completes, the only copy of the
evidence is on the machine. Provisioning is a separate, explicit act; `--on` never creates a pod.

### Standing campaign targets: `research keeper`

`research keeper set|stop|status NAME --on NODE` keeps N chunks of a command leased on a Nebius node until a deadline, so no agent
VM stays awake to relaunch them (`src/research/pods/nebius/keeper.py`).

### Pods bound themselves: leases, one budget file, one guard

Nothing that bounds spend may depend on an agent VM, which can be suspended at any moment, or on one guard process staying alive.
The code is in `src/research/pods/`.

- **A lease on every pod** (`lease.py`, `sh/lease.sh`). `research pods create --max-hours H` puts the expiry in the pod's
  environment, then, once sshd answers, arms the lease loop over ssh with the lease file and starts the idle guard (`pod_guard.sh
  daemon`: `--first-idle-min`, default 20, before the pod's first run; `--idle-min 0` arms none, for a pool pod kept up idle);
  `--boot-lease` also runs the loop from a RunPod image's start
  command, before sshd (opt-in until a live run shows sshd comes up with it). At the
  expiry the idle guard ends a pod whose runs all have custody; the lease loop ends it after a 15-minute grace whatever is unfetched,
  with the pod's own key (GraphQL `podTerminate`). A pod that cannot be armed is terminated by `create`. `research run --on` extends
  the lease to cover the run's `--timeout` plus 15 minutes; `research pods extend TARGET --hours H` extends it by hand (never
  shortens it), `research pods lease TARGET` shows it.
- **One budget file** (`budgets.py`): `budgets.toml` at the notes repo's root; each line is a prefix with `cap_usd` (total),
  `cap_usd_per_day` (any rolling 24 hours; its trip clears as the window moves) or both, `expires`, `max_pod_hours` and `by`. `research pods create` reads the guard's state from its host (one ssh call) and refuses (exit 6) unless
  the guard polled in the last 3 minutes and its budgets have a live, untripped line covering the name with `max_pod_hours` at least
  `--max-hours`.
- **One guard** (`research pods guard`, `guard.py`; its docstring is the contract) on the control pod re-reads `budgets.toml`
  from the notes origin every poll: per-line caps (a tripped line's pods terminated, new creates refused until the cap is raised),
  the account balance floor, each line's max pod age, and project pods no line covers. It never trips on the clock. It never fails
  open: it decides and terminates from memory before it records, every write is best effort, a failed read keeps the last value,
  and `--detach` runs it under a supervisor that restarts it after a crash.

The disk floor: `research run` and `research fetch --all` refuse (exit 3) while the runs root's filesystem has less than
`$RESEARCH_DISK_FLOOR_GB` free (default 1.5; `0` turns it off), naming the eviction that frees it. `research data custody RUN
--publish` removes the run's files of 1 MiB and more once custody holds (`--keep-local` keeps them); `run-files.evicted.json`
names the fetch that restores them. `research data evict --src-dir DIR` (node 1's hourly unit) also deletes each
shipped source tree that is exactly a commit on a ref of the node's `git/verity.git`, unused and unchanged for 2 h; any doubt
keeps them all (`store/evict.py`).

### research data

`src/research/store/` is the evidence store; its `README.md` is the contract
(four objects: Tool, Derivation, Attempt, Artifact; two relations: Replica,
Label). Artifacts are immutable, typed, content-addressed manifests with an
optional file or tree payload; the id is the sha256 of the canonical manifest.
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

Python: `research.store.local.LocalStore`, its methods listed in the contract. Nothing is
called PRESERVED without a remote; nothing in the store is ever overwritten.

### The evidence store: `--tool`, `--scratch`, and the publish step

`src/research/store/README.md` is the contract. What `research run` does:

- Before launch it collects `research.env.conditions()` (source tree, runtime,
  hardware, execution; `None` when unknown), resolves the harness `--input`
  paths to content identity, and, with `--tool NAME` (registry in
  `store/tools_registry.py`: `tc_probe`, `bench_vu`) or `--tool module:NAME`,
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
  `store_publish_error.json` and `research data attempt publish RUN` retries
  (also the way to publish old run dirs; `--all --runs-dir DIR` sweeps a runs
  root and prints `published / already present / refused (not terminal) /
  errors` with one line per error class). A configured-but-unusable remote
  (credential env vars unset here) does not stop a local publish: the attempt
  is written, `preserve_error` records why nothing was pushed.
- The r20 campaign's numbers live in the store: the frozen ledger was imported
  once (labels on the attempt when the run is in the store, else on a
  `ledger-row/v1` artifact; red-team verdicts as separate assertions).
- Repo replicas: `fixtures/artifacts.json` maps `art:` ids to tracked files
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
(`research/retention/v0`) is a sidecar, `<dir>/.retention.json` or `<file>.retention.json`, so it travels with its data: `path`
(relative to the root), `node`, `owner` (a lane handle), `by`, `created`, `expires` (UTC, or null for never), `priority`
(`low|normal|high|keep`), `preserved` (an art id or null), `regenerate` (a command or null), `why`. It covers everything beneath its
path. A directory its owner made read-only gets its record beside it, `<dir>.retention.json`. A directory with both is invalid, and
gc leaves a tree whole while any directory in it is read-only. A run dir whose `.custody` says `preserved: true` and which has no sidecar has an implicit record: owner its lane, else its
campaign, else `runner`; `preserved` its run record; `expires` 48 h after `verified_utc` (`RUN_EXPIRY`); priority `normal`. Nothing
is written for it, and a sidecar wins.

~~~text
research keep PATH --owner @LANE --expires DATE|never|7d --priority P [--regenerate CMD] [--art ART | --preserve] [--why T] [--by URL]
research keep show PATH                      # the record covering PATH, explicit or implicit, as JSON
research retention report [--root /workspace] [--depth 2] [--json]
research retention gc [--root /workspace] [--force] [--high-pct 80] [--low-pct 75] [--jobs-dir D]... [--json] [--apply]
research retention rm PATH... --approved-by @OWNER --ref REF [--by URL] [--root /workspace] [--jobs-dir D]... [--json] [--apply]
~~~

`--preserve` puts the path in the store as `node-data/v1` and pushes it PRESERVED (the store README's §7) before recording its id;
a failed push records nothing. The report walks with `os.scandir` and `lstat`, never into a recorded path. It lists every recorded
path and every area in the top `--depth` levels with no record beneath it (`unowned`): unowned and expired first, then totals.

Policy `default` (`default/v0`) acts only over `--high-pct` (vy-disk-guard's 80%) or with `--force`. It deletes the lowest priority,
then the oldest expiry, until the filesystem is under `--low-pct` (75%). A path qualifies only when its record is valid, of this node,
expired, `low` or `normal`, and preserved or regenerable, it holds no nested record, and no file in `/workspace/pouw/fill/queue`,
`/workspace/pouw/fill/running` or a `--jobs-dir` names it. Just before `--apply` deletes a path, its record and the job scripts are
read again. A preserved path also needs its artifact to hold every file now under it at its size, so files custody trimmed never
count as preserved, and `preserved.check` in head mode must find the artifact PRESERVED. An implicit record's marker must still vouch
for the run dir (the store README's §10.1). A path without a record is never deleted. Without `--apply` gc deletes nothing and reads
no remote; either way it prints every deletion and every skip with its reason. Its named failure: on 1 Oct node 2 held 737 GiB of
unowned, unpreserved bitsets that nobody could decide to delete.

`rm` is the one way to delete data you don't own on a node, for a cleanup the policy doesn't reach. It deletes a path only when a
record covers it (its own or an ancestor's), and that record and every record beneath it (sidecars and implicit ones) name the
`--approved-by` handle as owner and are not `keep`. No job script may name the path, and nothing in it may be read-only or a mount. A
path no record covers has nobody to say yes and is never deleted. The records are read again just before `--apply` deletes. Each
decision of an `--apply` (delete, refuse, failed) is a `research/retention-log/v1` line in `/workspace/research/retention/deletions.jsonl`
with `--by` (default `$RESEARCH_AGENT_URL`), the approver and `--ref`. Its named failure: on 3 Oct a disk cleanup ran a bare `rm -rf`
on four FP8 passes (about 74 GB each) under `pouw/mvp-e2e`. Their record named @compute-accounting as owner until 7 Oct, but another
lane had approved the deletion, so the passes were lost. Here that `rm` is refused with the true owner's name.

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
(`custody waived`), is there. `RESEARCH_NOTES_CITE_CHECK=0` turns it off.

~~~text
SYNC-WARN lanes/x/20260929T0600Z-report-x.md: cites r20260928-042319-9aac, which the evidence store does not have (a run is in the store once its attempt is on the remote: `research data custody <run> --publish` where its run dir is, else `research data label <run> custody waived --by <you>`)
~~~

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
        "python", "tools/verity/lean/upstream.py", "Security/Proofs", "--fetch", "--rev", "origin/main"]
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
