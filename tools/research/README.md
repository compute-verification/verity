# research

Shared tooling for executing, observing, and coordinating computational work
during development. One installable package, one CLI (`research`), stdlib
only. It runs commands and records what happened; it imports nothing from
Verity's `packages/` (enforced by `tests/test_cli.py::test_no_packages_import`).

The user-facing procedure (when to record a run, how to find one, how to hand
off) is the repo-level skill at `.agents/skills/research/SKILL.md`, once it
exists. This file is for people changing the tool.

## Layout and provenance

~~~text
src/research/
├── cli.py          research run | fetch | status | inspect | tele | data | launch-request
├── store/          the evidence store (`research data`): README.md is the spec; canon, ids, model, local, index, remote, remote_s3, kinds, cli
├── runs.py         run ids, the local registry (~/.research/runs or $RESEARCH_RUNS), read-time summaries
├── record.py       job.json v0.2 = the harness's v0.1 + tool, run, workload_fingerprint.expected, cost; v0.1 stays readable
├── result.py       the result.json contract (schema, run_id, fingerprint, validation, measurements, artifacts)
├── remote.py       --on: machines file, tool/source snapshots, request, claim-by-run-id, exclusive lock, fetch, preservation manifest
├── mem.py          --mem-gb admission: cgroup limit - in use - live /workspace/ramlock reservations; MemoryNotAvailable
├── env.py          conditions(): the one baseline of an attempt's conditions (source, runtime, hardware, execution); ONE SKU->family map
├── store/          the evidence store (README.md there is the contract): tool, tools_registry, derivation, attempt, cli_prov (provenance)
│                   + canon, local, index, remote, kinds, cli (storage)
├── telemetry/      the recorded-attempt harness: job.json, status.json, events.jsonl, resources.jsonl, failure.json
└── pods/           RunPod lifecycle (runpod.py), budget cap, memory guard, and pod-side shell guards under sh/
tests/              pure-Python; the machine side of --on is exercised locally (launch_request is a plain function)
tests/fixtures/     job_v0.1.json: a record as the harness alone writes it; run_v0.2/: a finished run dir with a `store` section (state.json -> status.json at test time)
~~~

`telemetry/` and `pods/` were promoted from `veritor` `zk/campaign/` at
commit `df7fbea3d` (branch `lane/r17-mprof`, 2026-09-21) and have since been
edited here like the rest of the package. `pods/sh/deadline.sh` and
`rearm.sh` came from the r17 deadman pod (`/root/dm`), where they had never
been committed. The diffs in `telemetry/patches/` are unapplied cancel-intent
integrations against veritor's launcher scripts.

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

## Commands

~~~text
uv sync                                  # from the verity root
uv run pytest                            # all tests

research run [--project P] [--stage S] [--timeout T] [--expect K=V] [--require-result] [--env K=V] [--cwd DIR] [harness flags] -- CMD...
research run [--tool NAME|module:NAME] [--scratch NAME]... [--campaign C] [...] -- CMD...      # evidence store, below
research run --on MACHINE --project P [--source GITDIR] [--send FILE] [--env K=V] [--exclusive] [--mem-gb N [--mem-wait S]] [...] -- CMD...
research fetch RUN [--all]               # records only, or every file sha256-verified with preserved.json written last
research status [--refresh] [--all] [--json]
research inspect RUN [--json]
research data SUBCOMMAND ...             # the store: put/show/fetch/select/... and lookup / pull / attempt publish
research pods create|list|terminate|check-part ...                # RunPod lifecycle (pods/runpod.py, pods/part.py)
research pods ssh TARGET [--print] [-- CMD...]                    # TARGET = machines.toml name | pod id | pod name (pods/connect.py)
research pods sync TARGET [SRC] [--dest DIR] [--exclude G]...     # the worktree's git-visible files -> DIR (default /workspace/src)
research tele SUBCOMMAND ...             # the harness's own surface
research data SUBCOMMAND ...             # the evidence store (below)
~~~

The workload receives `RESEARCH_RUN_ID` and `RESEARCH_RUN_DIR` and may write
`$RESEARCH_RUN_DIR/result.json` (see `result.py` for the envelope). A local
workload runs in its run directory unless `--cwd` says otherwise (`--cwd source` = the invoking directory), so a relative
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

An unlisted or foreign machine is refused before anything is contacted. The
tool ships as a content-addressed source snapshot under `<root>/tool/` and
runs via `PYTHONPATH`; no pip, no credentials on the machine. The machine
checks that the `research` package that answers is the one shipped (its content
identity): `launch-request` checks before it claims the run, and the runner
checks again before the workload starts. On a mismatch the run is refused
with `ResearchVersionMismatch`, naming the running package's path. A package
too old to know the check fails closed. The one exception: a `--source`
tree's own older package runs unverified, and the launcher says so. job.json
records `run.remote.tool_running`. `--source` ships
`git archive HEAD` of a clean checkout to `<root>/src/<sha>/` (a dirty tree is
refused, not mislabeled). The ship is verified before anything may run from
it: the archive lands in `<sha>.partial-<nonce>/`, its sha256 is compared on
both ends and the extracted file set against the archive's manifest, then the
directory is renamed into place carrying `READY.json` (identity, archive
sha256, file count, launcher). A failed or truncated transfer removes its
partial directory and the run is refused (`SourceNotReady`); the machine
refuses to execute from a source directory without a `READY.json` naming the
requested identity (the legacy `.complete` marker is verified per file once,
then adopted or quarantined). The attempt records `execution.remote.source_sha`
and `source_archive_sha256`. The shipped tree is the runner's source root, not
the workload's cwd: the workload runs in its own run directory, as locally,
because every run of one commit shares `<root>/src/<sha>/` and relative outputs
used to race there. `--cwd source` (or `source/SUB`) is the explicit opt-in to
run inside the tree; job.json records `run.cwd` and `run.cwd_basis`. The
runner and the workload get the tree's `PYTHONPATH` derived from its root
`pyproject.toml` workspace members and each member's build config
(`src/`, `python/`, flat), with a user `--env PYTHONPATH` appended; job.json
records it (`run.remote.pythonpath`). `research env --source TREE` prints the
same line for a local checkout. The machine claims the run directory by run id and
starts the runner detached; re-delivering a launch after an ssh drop finds
the claim and reports state. `--exclusive` refuses if any run is live, and
any launch refuses if an exclusive holder is live.

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
`ramlock = DIR` overrides it), is at least N. A reservation is live unless its
pid is gone or its run is terminal. The run then writes
`research-<run id>.json` there, and the runner removes it at exit. Otherwise the
run is refused with `MemoryNotAvailable`, and `refused.json` gives every number.
`--mem-wait S` polls for up to S seconds instead. job.json records the
admission (`run.remote.mem`) and the attempt records
`execution.mem_gb_reserved`. `research mem` prints the reading on the machine.

Limitations, stated: until `fetch --all` completes, the only copy of the
evidence is on the machine. Budget enforcement is `research pods guard`
(`pods/guard.py`; its docstring is the contract), not this command: scoped to
a pod-name prefix, sticky trips on deadline / cap / balance floor that
terminate every scoped pod (the guard's host last), and `--rate-max` shedding
that terminates the newest scoped pods first and never the guard's host. It
replaces the loops of `pods/budget_cap.py` and `pods/sh/deadline.sh`, which
stay verbatim until the fleet switches over. Provisioning is a separate,
explicit act; `--on` never creates a pod.

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

Python: `research.store.local.LocalStore` (`put_artifact`, `get_manifest`,
`fetch`, `where`, `put_attempt`, `get_attempt`, `add_label`, `labels`,
`snapshot`, `verify`, `push`, `reindex`, `.index`, `.remote`). Nothing is
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
- The r20 campaign's numbers live in the store. `python -m
  verity_numerical.bench.ledger_to_store [--store ROOT] [--dry-run]` imports
  the frozen ledger once (labels on the attempt when the run is in the store,
  else on a `ledger-row/v1` artifact; red-team verdicts as separate assertions;
  idempotent). `python -m verity_numerical.bench.store_tables --diff-ledger
  [--proof-class-by 'red-team-*']` rebuilds the like-for-like table from store
  queries alone and diffs it row by row against `decision_tables.py`.
- Repo replicas: `fixtures/artifacts.json` maps `art:` ids to tracked files
  registered with `research data put --kind fixture/v1 --file PATH`; `where`
  lists the `repo` replica when the file still hashes to its id.
- With `--on`, the flags travel to the machine, which runs the same code in
  the shipped source tree; a `git archive` has no `.git`, so the control side
  sends the commit/tree it knew as `--source-identity`. The machine publishes
  to its own `<root>/store`; `research data pull --from MACHINE RUN --project P`
  copies the attempt and its outputs (sha256-verified) into the local store.
- `research data lookup --tool T [--param k=v]... [--condition k=v]...
  [--source-tree SHA | --any-source] [-- ARGV...]` answers "have we done this?"

## Not yet built, in order

1. `research compare <a> <b>`: descriptive diff of two run records
   (source, fingerprints expected/observed, environment, measurements with
   ratios labeled as treatment/baseline, validation, preservation). It shows
   differences; the workload's contract decides which ones matter.
2. `research status` reading the project's budget ledger with its
   observation time (no second spend counter).
3. Reconcile the final `lane/r17-mprof` telemetry revision against
   `df7fbea3d` once the r17 campaign is frozen; port, do not overwrite.
4. Milestone: one real SP1 case through `--on`, then the seven-stage
   `campaign_h100.sh` replaced end to end: the stage sequence stays a
   benchmark-owned script; `paired_run.py` writes `result.json`; `research`
   supplies targeting, shipping, identity, telemetry, retrieval. No SP1
   branch inside `research`.
