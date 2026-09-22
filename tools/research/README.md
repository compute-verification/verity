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
├── cli.py          research run | fetch | status | inspect | tele | launch-request
├── runs.py         run ids, the local registry (~/.research/runs or $RESEARCH_RUNS), read-time summaries
├── record.py       job.json v0.2 = the harness's v0.1 + tool, run, workload_fingerprint.expected, cost; v0.1 stays readable
├── result.py       the result.json contract (schema, run_id, fingerprint, validation, measurements, artifacts)
├── remote.py       --on: machines file, tool/source snapshots, request, claim-by-run-id, exclusive lock, fetch, preservation manifest
├── env.py          conditions(): the one baseline of an attempt's conditions (source, runtime, hardware, execution); ONE SKU->family map
├── store/          the evidence store (README.md there is the contract): tool, tools_registry, derivation, attempt, cli_prov (provenance)
│                   + canon, local, index, remote, kinds, cli (storage)
├── telemetry/      the recorded-attempt harness: job.json, status.json, events.jsonl, resources.jsonl, failure.json  (unmodified upstream)
└── pods/           RunPod lifecycle (runpod.py), budget cap, memory guard, and pod-side shell guards under sh/           (unmodified upstream)
tests/              pure-Python; the machine side of --on is exercised locally (launch_request is a plain function)
tests/fixtures/     job_v0.1.json: a record as the harness alone writes it; run_v0.2/: a finished run dir with a `store` section
~~~

`telemetry/` and `pods/` were promoted verbatim from `veritor` `zk/campaign/`
at commit `df7fbea3d` (branch `lane/r17-mprof`, 2026-09-21) with import paths
adjusted and nothing else; `research` never edits them, so reconciling a
newer upstream revision is a directory diff. `pods/sh/deadline.sh` and
`rearm.sh` came from the r17 deadman pod (`/root/dm`), where they had never
been committed. The diffs in `telemetry/patches/` are unapplied cancel-intent
integrations against veritor's launcher scripts.

`research run` pre-writes `job.json` (the harness keeps an existing one) and
calls the harness's `run()` directly: one execution engine, one record.

## Commands

~~~text
uv sync                                  # from the verity root
uv run pytest                            # all tests

research run [--project P] [--stage S] [--timeout T] [--expect K=V] [--require-result] [harness flags] -- CMD...
research run [--tool NAME|module:NAME] [--scratch NAME]... [--campaign C] [...] -- CMD...      # evidence store, below
research run --on MACHINE --project P [--source GITDIR] [--send FILE] [--env K=V] [--exclusive] [...] -- CMD...
research fetch RUN [--all]               # records only, or every file sha256-verified with preserved.json written last
research status [--refresh] [--all] [--json]
research inspect RUN [--json]
research data SUBCOMMAND ...             # the store: put/show/fetch/select/... and lookup / pull / attempt publish
research tele SUBCOMMAND ...             # the harness's own surface
~~~

The workload receives `RESEARCH_RUN_ID` and `RESEARCH_RUN_DIR` and may write
`$RESEARCH_RUN_DIR/result.json` (see `result.py` for the envelope). Four facts
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
runs via `PYTHONPATH`; no pip, no credentials on the machine. `--source` ships
`git archive HEAD` of a clean checkout to `<root>/src/<sha>/` (a dirty tree is
refused, not mislabeled). The machine claims the run directory by run id and
starts the runner detached; re-delivering a launch after an ssh drop finds
the claim and reports state. `--exclusive` refuses if any run is live, and
any launch refuses if an exclusive holder is live.

Limitations, stated: until `fetch --all` completes, the only copy of the
evidence is on the machine. Budget enforcement is the project's deadman
(`pods/budget_cap.py`), not this command. Provisioning is a separate,
explicit act; `--on` never creates a pod.

### The evidence store: `--tool`, `--scratch`, and the publish step

`src/research/store/README.md` is the contract. What `research run` does:

- Before launch it collects `research.env.conditions()` (source tree, runtime,
  hardware, execution; `None` when unknown), resolves the harness `--input`
  paths to content identity, and, with `--tool NAME` (registry in
  `store/tools_registry.py`: `tc_probe`, `bench_vu`) or `--tool module:NAME`,
  loads the `Tool` declaration, parses the workload argv into params, replaces
  path-valued key params by the content they point at, hashes the declared
  closure, and computes the derivation id and the `reuse_key` (`None` on a
  dirty tree). All of it goes into `job.json` as the optional `store` section.
  An unknown tool exits 2 before anything runs; any other failure in this step
  is recorded in `store.error` and the workload still runs.
- `--scratch NAME` (repeatable) creates `<store>/scratch/<reuse_key>/NAME/`
  (`dirty-<run id>` on a dirty tree) and exports `TRITON_CACHE_DIR` for
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
  (also the way to publish old run dirs; `--all` sweeps a runs root).
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
