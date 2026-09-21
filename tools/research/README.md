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
├── cli.py          research run | status | inspect | tele
├── runs.py         run ids, the local registry (~/.research/runs or $RESEARCH_RUNS), summaries
├── telemetry/      the recorded-attempt harness: job.json, status.json, events.jsonl, resources.jsonl, failure.json
└── pods/           RunPod lifecycle (runpod.py), budget cap, memory guard, and pod-side shell guards under sh/
tests/              33 pure-Python tests (29 promoted with telemetry, 4 for the CLI); Linux-only fixtures in telemetry/fixtures/
~~~

`telemetry/` and `pods/` were promoted verbatim from `veritor` `zk/campaign/`
at commit `df7fbea3d` (branch `lane/r17-mprof`, 2026-09-21) with import paths
adjusted and nothing else. `pods/sh/deadline.sh` and `rearm.sh` came from the
r17 deadman pod (`/root/dm`), where they had never been committed. The diffs
in `telemetry/patches/` are unapplied cancel-intent integrations against
veritor's launcher scripts; they document what the guards still owe the
harness.

The pod-side scripts (`budget_cap.py`, `mem_guard.py`, `deadman.sh`,
`pod_guard.sh`) are copied to pods as plain files and import `runpod.py` as a
sibling via `sys.path`; keep them runnable without the package installed.

## Commands

~~~text
uv sync                                  # from the verity root
uv run pytest                            # all tests
uv run research run [--project P] [--stage S] [--timeout T] [harness flags] -- CMD...
uv run research status [--all] [--json]
uv run research inspect <run-id | unique prefix | dir> [--json]
uv run research tele <subcommand>        # the full harness surface: init, event, sample, cancel-intent, classify, ...
~~~

`research run` allocates an id (`r20260921-172041-9cfe`), creates the run
directory, and hands off to the harness, which tees stdout/stderr, samples
resources in a sidecar, enforces `--timeout` with a recorded cancel intent
(SIGTERM, grace, SIGKILL), classifies the outcome, and exits with the
workload's status. Anything the harness accepts passes through
(`--tree-sha`, `--workload-digest`, `--input name=path`, `--declared-output`,
`--policy`, `--no-sampler`, ...).

A run directory is self-describing; reading one needs no service and no
credentials.

## Not yet built, in order

1. Run-record extensions: a structured `workload_fingerprint` object (today
   `computation.workload_digest` is one string) and `cost` (provider, machine
   type, GPU, start/stop, the billing rate actually used, derived cost).
   Schema bump to `telemetry/job/v0.2`, readable alongside v0.1.
2. `<run-dir>/result.json`: the language-neutral contract by which a
   workload (Rust SP1 host, Python, shell) reports application-level results
   and validation. Files, not stdout markers.
3. `research run --on <machine>`: ship the tree (`git archive`), check the
   project budget, source `pod_guard.sh`, run under the harness on the pod,
   retrieve the run directory. Assembled from `pods/` and the launcher
   pattern in veritor's `zk/campaign/campaign*.sh`; no new provisioning path.
4. `research compare <a> <b>`: structural diff of two run records
   (conditions, fingerprints, measurements). Semantic comparability is the
   workload skill's call, not this tool's.
5. Milestone: replace `veritor/zk/campaign/campaign_h100.sh` end to end with
   `research run --on <pod> -- <inner command>` plus a `result.json` from
   `paired_run.py`, losing nothing it records today (commit, ELF digest, vk
   hash, binary hashes, watchdog, budget check, rsync-back). If that needs
   SP1-specific escape hatches in `research`, the abstraction is wrong.

Upstream is still moving: `lane/r17-mprof` may add commits until the r17
campaign closes. Re-promote from its final revision before step 3; the copy
is a directory, so this is mechanical.
