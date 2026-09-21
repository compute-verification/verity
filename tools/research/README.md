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
├── telemetry/      the recorded-attempt harness: job.json, status.json, events.jsonl, resources.jsonl, failure.json  (unmodified upstream)
└── pods/           RunPod lifecycle (runpod.py), budget cap, memory guard, and pod-side shell guards under sh/           (unmodified upstream)
tests/              pure-Python; the machine side of --on is exercised locally (launch_request is a plain function)
tests/fixtures/     job_v0.1.json: a record as the harness alone writes it
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
research run --on MACHINE --project P [--source GITDIR] [--send FILE] [--env K=V] [--exclusive] [...] -- CMD...
research fetch RUN [--all]               # records only, or every file sha256-verified with preserved.json written last
research status [--refresh] [--all] [--json]
research inspect RUN [--json]
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
