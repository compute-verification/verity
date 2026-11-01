# `check`

`check` is the repository's one check. `verity check` runs `check.py` here, on each landing's tip before `main` moves there, and on any
commit you ask it to. The jobs it runs are data in `ci.toml`: each has an owner, a part, a group and what its failure means.

| Part | Jobs |
| --- | --- |
| `tests` | `pytest` (every test suite, through `suites.py`), `lean-suites` (the suites that read the Lean verifier's build), `ratchet-gate` (the vLLM integration's ratchet gate on each merge above the check's base: on a train, it names the PR whose merge grows a lint allowlist) |
| `circuits` | `flock-circuit-build`, `flock-rust-tests` (C-Flock's Rust), `circuit-check` (the property checks every circuit passes) |
| `lean` | `lean-build` and `lean-unit-cut` (the Lean verifier and its partition check), `lean-audit` (the Lean check of every Lake package, `tools/verity/lean/`), `lean-changed` (the early Lean check on a merge, which halts the run), and `lean-agreement` (the cross-check against upstream's Rust verifier, when a change touches C-Flock) |

`verity check tests|circuits|lean` runs only those parts and what their jobs need. `check.py`'s docstring lists the rest: shards,
`--deferred STEP,...`, `--plan` and `--slots`.

A job with `gate = false` in `ci.toml` is off the merge gate: a recorded check and its shards leave it out, and `research
merge` requires it of no check. Today those are `circuit-check` and `ratchet-gate`. Each runs after landing, in the run that
checks `checked`'s tip, and doesn't hold `main`. A red `circuit-check` reverts its culprit at once and lands no PR touching its
inputs until it passes; a red `ratchet-gate` reverts its culprit as any other after-landing red does. `circuit-check` goes back
on the gate once `circuit-check --all` fits inside the rest of the check; `ratchet-gate` is in no shard plan, and reports
without blocking a merge.

## Suites and reuse

A suite is a directory whose `pyproject.toml` configures pytest. Each runs in its own process from its own directory. A pass is
kept under a hash of its inputs and reused while that hash is unchanged. The inputs are the suite's directory, the workspace
packages it depends on, what it resolves to in `uv.lock`, the paths it lists under `[tool.verity.tests] inputs`, its fixtures,
the environment (Python, the platform, the base image, the `VERITY_*` and `FLOCK_*` variables), and the runner itself.
Markdown is left out unless a suite lists it. The read guard (`guard/verity_suite_guard.py`) fails a test that opens a
repository file outside its suite's inputs, so a reused pass never hides a change to something a test read. In a
`parallel = true` suite each passing test is also kept on its own, under its module's key.

Passes travel between machines as verdict packs (`verdicts.py`). The Lean audit and `circuit-check` reuse their own verdicts
the same way, each keyed on what it reads. A nightly uncached run must agree with the cached ones. A disagreement means some
part reads something it doesn't declare, and the fix goes in the check.

`suites.py`'s docstring is the full contract. The testing skill (`.agents/skills/testing/SKILL.md`) has the rules every test
follows.

## Running it

Builds and tests run on the cluster's nodes, not on an agent VM. Ship the checkout with `research run` and sync its tree first:

~~~sh
R="research run --on auto --cwd source -- bash -lc"
$R "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py flock"            # one suite
$R "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py --changed origin/main --quick"
$R "uv sync -q --all-packages --extra torch-cpu && uv run --extra torch-cpu verity check"               # the whole check
uv run tools/verity/check/suites.py --list                # each suite's inputs and whether a pass is cached (reads only)
uv run tools/verity/check/suites.py --explain SUITE       # why a suite would run
~~~

`research run` returns once the job starts, and `research status RUN` checks back on it. To record a check of a clean
commit as an Attempt, run `uv run python tools/verity/check/check.py --record --on vy-nebius-cpu-1 --pool research` from
the checkout.

## Tasks (`tasks/v1`)

The check is also a graph of tasks, the whole contract between Verity and an executor. `verity tasks` (`tasks.py`) has two
commands:

~~~sh
verity tasks list --json [--kind after-landing|preflight] [--changed BASE] [--circuit-shards M] [--gather | --costs FILE]
verity tasks run ID [--out FILE] [--work DIR] [--circuit-shards M]
~~~

`list` prints the graph of the checked-out tree, uncommitted files included. `--kind after-landing` (the default) is every job
in `ci.toml`, as drive's after-landing run checks it; `--kind preflight` is the jobs on the merge gate, a PR's preflight.
`--changed BASE` keeps the tasks whose inputs the change from BASE's merge-base with HEAD touches, by `suites.py --changed`'s
rule for a suite and by input paths for the rest, and the tasks they depend on. `--circuit-shards M` splits circuit-check into M
tasks (default: its ci.toml `shards`). The document is `{"schema": "tasks/v1", "commit", "tree", "kind", "circuit_shards",
"changed": {"base": the merge-base, "files": how many files changed} or null, "costs": [notes], "tasks": [...]}`, and each
task is:

| Field | What it is |
| --- | --- |
| `id` | the shard planner's part (`shards.py`): `pytest:SUITE` or `lean-suites:SUITE` (`suites.AFTER_LEAN`, which read the verifier's build), `lean-audit:PKG`, `circuit-check:K/M`, or a job's name |
| `kind` | `suite`, `lean-package`, `circuit-check`, `build` (`lean-build`, `flock-circuit-build`) or `step` (another job, run whole: `flock-rust-tests`, `lean-unit-cut`, `lean-agreement`, `lean-changed`, `ratchet-gate`) |
| `argv`, `cwd` | the command `check.py` builds for that part alone (`check.groups`), with `<tree>`, `<cache>`, `<out>` and `<python>` for the tree, the check's cache, the run's output directory and the interpreter; `argv` is null where check can't run it on this tree |
| `reason` | null, or why `argv` is null: one starting `skipped:` is a skip, any other a failure |
| `inputs` | `{"closure", "paths"}`: paths and globs from the root that cover what the task reads, and the function that lists them (`suites.inputs_of`, with the runner's files; `lean_audit.inputs`; circuit-check's workspace closure). `["."]` is the whole tree, for a job that declares none |
| `needs` | the artifacts it reads that another task makes: `lean-build:verity/core/service/flock/.lake/build`, `flock-circuit-build:$FLOCK_CHECKOUT` |
| `outputs` | the artifacts it makes (a build's) |
| `deps` | the task ids that must end before it starts (its job's ci.toml `needs` and `after`) |
| `beside` | the tasks that build what it builds (a spec's audit and the packages its `proved_in` names), so best on one worker; no order |
| `resources` | `{cores, mem_gb, disk_gb}` it asks of a worker: 32, 128, 60 for a Lake package's audit; 1, `suites.TEST_PROCESS_GB`, 10 for a suite that runs in one process; 16, 64, 20 for the rest |
| `cost_s` | its measured seconds (`check.measure`: this machine's caches, `--gather` the newest recorded checks' too, `--costs` a `{id: seconds}` over them), null when none |
| `key` | SHA-512 over its id, argv, cwd and what it reads; null when it has none |
| `unkeyed` | null, or why `key` is null |

A task's key hashes what a suite's key hashes (`suites.components`: each input's git object id, its fixtures, the
resolved lock entries and installed versions, Python, the platform, the base image and the `VERITY_*`/`FLOCK_*` variables) under
the environment check gives that step; a Lake package's, `lean_audit.components` (its content and its path dependencies' and
provers', the auditor's files, the toolchain pin) with Python and the base image; a job in `check.STEP_KEYS`, that step's key.
So keys are of the machine that lists them: list on a worker. These have no key yet, and run whenever they're in the graph: the
builds (no declared inputs, read guard or keyed outputs), circuit-check's slices (it keys each target, from a task list only a
run builds), and the jobs that keep no verdict (`lean-unit-cut`, `lean-changed`, `ratchet-gate`). The preflight is no task: a
worker's `uv sync --locked` refuses a lock that isn't current, each suite fetches the fixtures it reads, and the repository's
suite holds the lints.

`run ID` runs that task here as `check.py` runs that part in a shard of its own (`check.run` with only that part, without the
preflight), its `deps` beside it until artifacts are shared, and writes `{"schema", "id", "verdict": passed|failed|skipped,
"key", "seconds", "outputs": {"dir", "files", "artifacts"}, "steps": [check's step records]}` to stdout or `--out`; the steps'
progress goes to stderr. It exits 0 for passed or skipped, 1 for failed.

## Landing

PRs land through the merge queue (`research queue drive`): its flags are `drive.toml` and its admission rules are
`queue.toml`. Before a PR lands, the landing gates `drive.toml` names (`gates/markdown_caps.py` among them) judge its
merge. Reviewed PRs land on `checked` (`land_into`); after each landing, one run of the full check checks `checked`'s tip
with every job, those off the gate too (`train.sh --deferred`), and `main` moves only to a commit whose run passed every step
on the gate, where `post_train.py` runs the vLLM regression's rebuilds. A red job off the gate reverts its culprit on `checked`
but doesn't hold `main`. The team skill's "Landing" is the procedure. A pure move that `move_check.py` certifies
(`verity check move BASE HEAD`) lands ahead of the queue.
