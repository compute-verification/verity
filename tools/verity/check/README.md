# `check`

`check` is the repository's one check. `verity check` runs `check.py` here, on each candidate before `main` moves to it, and on any
commit you ask it to. The jobs it runs are data in `ci.toml`: each has an owner, a part, a group and what its failure means.

| Part | Jobs |
| --- | --- |
| `tests` | `pytest` (every test suite, through `suites.py`), `lean-suites` (the suites that read the Lean verifier's build), `ratchet-gate` (the vLLM integration's ratchet gate on each merge above the check's base: in a candidate, it names the PR whose merge grows a lint allowlist) |
| `circuits` | `flock-circuit-build`, `flock-rust-tests` (Flock's Rust), `circuit-check` (the property checks every circuit passes) |
| `lean` | `lean-build` and `lean-unit-cut` (the Lean verifier and its partition check), `lean-audit` (the Lean check of every Lake package, `tools/verity/lean/`), and `lean-agreement` (the cross-check against upstream's Rust verifier, when a change touches Flock) |

`verity check tests|circuits|lean` runs only those parts and what their jobs need. `check.py`'s docstring lists the rest: shards,
`--deferred STEP,...`, `--plan` and `--slots`.

The runners (`suites.py`, `circuit-check`, `check.py`'s steps) report real verdicts and nothing else: a test, target or step
that failed is failed, whoever is fixing it. What gates and what may stay red is the landing policy, `policy.toml`
(`research.policy`), whose allowances the verified queue reads at a candidate's base, so no candidate's own allowance covers
its own failure. An allowance there (`[[allow]]`: an id, its
owner, when it expires in UTC, and why) covers one suite, one test or one circuit-check failure: the verified queue takes
such a failure as not red (no suspect, no ejection, no hold on `main`) until it expires. The quick tier applies `main`'s
allowances the same way.

A job in `policy.toml`'s `off_gate` is off the merge gate: a recorded check and its shards leave it out, and `research
merge` requires it of no check. Today those are `circuit-check` and `ratchet-gate`. Each still runs in a candidate's check,
and its red holds nothing; under the loop's `--gate check`, a red that the run beside the candidate's base didn't have rings
its owners. Under the loop's `--gate graph2`, `ratchet-gate` runs with
`lean-agreement` in a run of their own beside the executor's check of the candidate (`research.vqueue_slim`), which waits
for that run's preflights and `lean-agreement`, not its `ratchet-gate`. A red `ratchet-gate` (drive.toml's `warn_steps`
too) names the merges its log fails, whose lanes fix it forward in their next PR; a candidate's `ratchet-gate` judges only
the merges above its base. `circuit-check` goes back on the gate once `circuit-check --all` fits inside the rest of the
check; `ratchet-gate` is in no shard plan, and reports without blocking a merge.

## Suites and reuse

A suite is a directory whose `pyproject.toml` configures pytest. Each runs in its own process from its own directory. A pass is
kept under a hash of its inputs and reused while that hash is unchanged. The inputs are the suite's directory, the workspace
packages it depends on, what it resolves to in `uv.lock`, the paths it lists under `[tool.verity.tests] inputs`, its fixtures,
the environment (Python, the platform, the base image, the `VERITY_*` and `FLOCK_*` variables but those `settings.py` declares
`where`), and the runner itself.
Markdown is left out unless a suite lists it. The read guard (`guard/verity_suite_guard.py`) fails a test that opens a
repository file outside its suite's inputs, so a reused pass never hides a change to something a test read. In a
`parallel = true` suite each passing test is also kept on its own, under its module's key.

A long suite runs in parts by test file: `suites.py SUITE --files K/M` (or `SUITE:K/M`, `SUITE:1,3/M`) runs part K of M,
its test files packed longest first by their recorded seconds (`suites.partition`; a file with no record counts the median), so
a tree and a timing record give one partition, named by its digest. A part's key is the suite's with the other parts' test
files left out (`part_key`), so an edit to one file reruns its part alone, and the join of a suite's parts (`combine_parts`:
every K once, of one partition, all passed) is the suite's verdict. The check splits a suite measured past `shards.SPLIT_S`
(15 min; `--split-s`) into ceil(seconds / threshold) parts, and its shards partition by the timings the plan carries.
A suite whose long tests share few files declares `[tool.verity.tests] parts = P` instead and runs in P parts by test id
(`suites.id_part_of`): part K collects every file and runs the tests whose node id hashes to K (sha256, as circuit-check
shards its targets), so the parts are one on every machine whatever its timing record. A part's key is the suite's with the
part, and its parts join as a split suite's do. `ci.toml`'s plans name a lean suite's parts (`lean-suites:SUITE:K/P`, as
`lean-audit:PKG` names a package) to spread it over shards, each of which runs `lean-build` beside them, and a check that holds
every part runs the suite whole.

Passes travel between machines as verdict packs (`verdicts.py`). The Lean audit and `circuit-check` reuse their own verdicts
the same way, each keyed on what it reads. A nightly uncached run, which compiles what it tests and first imports every pass
the commit's recorded checks hold, must agree with them. A disagreement means some part reads something it doesn't declare,
and the fix goes in the check. The nightly also runs the opt-in tests `check.py`'s `OPT_IN` names, by name in runs of their
own with their variables set, which every other check leaves skipped.

`suites.py`'s docstring is the full contract. The testing skill (`.agents/skills/testing/SKILL.md`) has the rules every test
follows.

## Modules

| File | What it is |
| --- | --- |
| `check.py` | `check` itself: every step, its shards, `--deferred`, `--record` |
| `ci.toml`, `declaration.py` | the jobs `check` runs, and their reader |
| `suites.py` | every test suite, each run in its own process, with passes cached |
| `guard/` | the read guard, the pytest plugin `suites.py` loads into every suite |
| `verdicts.py` | passes carried between machines as one verdict pack |
| `common.py` | the helpers `check.py` and the Lean check share, apart so the Lean check's key covers them |
| `plan.py` | `check.py --plan`: what `check` would reuse at a commit and what it would run |
| `spans.py` | the spans a check run writes beside its steps, which the dashboard reads |
| `tool.py`, `preflight.py` | the check's store declaration, and whether a machine is prepared for it |
| `pod_setup.sh` | prepares a check pod once, so every pod of the pool computes the same verdict keys |
| `slot.py`, `lean_slot.py` | a command in one of a node's merge-check or Lean check slots |
| `train.sh` | the verified queue's recorded check of a candidate, and its records run |
| `post_train.py` | the vLLM reference regression's manifest rebuilds and the fast rows of record's neutrality after a landing |
| `lean_prebuild.py` | a Lean change built before the candidate that holds it: the audit of a ready mark's merge (`tools/verity/lean/warm.py`) as a `research run` in a machine's research pool, off the check slots |
| `drive.toml`, `queue.toml` | the merge queue's flags and its admission rules |
| `gates/markdown_caps.py` | the warn gate naming a change that puts a Markdown file over its cap |
| `move_check.py`, `anchors.py` | the certificate of a pure move, and that every path a file reaches from its own place survives it |
| `merge_map.py` | git's merge driver for `fixtures/artifacts.json` |
| `merge_rows.py` | git's merge driver for Markdown: two changes to one table's rows merge row by row |
| `pins.py` | every file that records a Lean source's digest, with the command that regenerates it |
| `mutants.py` | deletes each check a verifier source runs, one at a time, and names each deletion its tests miss |

## Running it

Builds and tests run on the cluster's nodes, not on an agent VM. Ship the checkout with `research run` and sync its tree first:

~~~sh
R="research run --on auto --cwd source -- bash -lc"
$R "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py flock"            # one suite
$R "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py --changed $(git fetch -q origin main && git merge-base HEAD FETCH_HEAD) --quick"
$R "uv sync -q --all-packages --extra torch-cpu && uv run --extra torch-cpu verity check"               # the whole check
uv run tools/verity/check/suites.py --list                # each suite's inputs and whether a pass is cached (reads only)
uv run tools/verity/check/suites.py --explain SUITE       # why a suite would run
~~~

`research run` returns once the job starts, and `research status RUN` checks back on it. To record a check of a clean
commit as an Attempt, run `uv run python tools/verity/check/check.py --record --on auto` from the checkout.

## Tasks (`tasks/v1`)

The check is also a graph of tasks, the whole contract between Verity and an executor. `verity tasks` (`tasks.py`) has two
commands:

~~~sh
verity tasks list --json [--kind after-landing|preflight] [--changed BASE] [--circuit-shards M] [--gather | --costs FILE]
                         [--split-s S]
verity tasks run ID [--out FILE] [--work DIR] [--graph art:ID|FILE | [--circuit-shards M] [--timings FILE]]
verity tasks costs --graph art:ID|FILE [--graph ...] [--seconds FILE] [--out FILE]
~~~

`list` prints the graph of the checked-out tree, uncommitted files included. `--kind after-landing` (the default) is every job
in `ci.toml`, as a candidate's check runs it; `--kind preflight` is the jobs on the merge gate, a PR's preflight.
`--changed BASE` keeps the tasks whose inputs the change from BASE's merge-base with HEAD touches, by `suites.py --changed`'s
rule for a suite and by input paths for the rest, and the tasks they depend on. `--circuit-shards M` splits circuit-check into M
tasks (default: its ci.toml `shards`, or ceil(seconds / S) when `--costs` gives `circuit-check` more seconds than those cover).
A suite that declares `[tool.verity.tests] parts = P` is listed in its P parts by test id, each at 1/P of its cost; another whose
cost is past `--split-s` seconds (default `shards.SPLIT_S`, 15 min) in M = ceil(cost / S) parts by test file (at most one per
file), each priced at its files' share of the suite's cost. The lister's own caches and `--gather`'s
recorded checks measure a suite on other machines than the executor's worker slots, where it can take several times as long, so
the executor lists with `--costs` from `costs` over the graph it ran last; then each part is about S of the executor's seconds.
The
document is `{"schema": "tasks/v1", "commit", "tree", "kind", "circuit_shards", "split_s", "splits": {suite: {"m", "seconds",
"digest", "shares", "files", "timings"}}, "changed": {"base": the merge-base, "files": how many files changed} or null, "costs":
[notes], "tasks": [...]}`, where `splits` names each split suite's partition (its digest, each part's share and files, and the
timing record it is of), and each task is:

| Field | What it is |
| --- | --- |
| `id` | the shard planner's part (`shards.py`): `pytest:SUITE` or `lean-suites:SUITE` (`suites.AFTER_LEAN`, which read the verifier's build), `pytest:SUITE:K/M` for part K of a split suite (by test file, or by test id where it declares `parts`), `lean-audit:PKG`, `circuit-check:K/M`, or a job's name |
| `kind` | `suite`, `lean-package`, `circuit-check`, `build` (`lean-build`, `flock-circuit-build`) or `step` (another job, run whole: `flock-rust-tests`, `lean-unit-cut`, `lean-agreement`, `ratchet-gate`) |
| `argv`, `cwd` | the command `check.py` builds for that part alone (`check.groups`), with `<tree>`, `<cache>`, `<out>` and `<python>` for the tree, the check's cache, the run's output directory and the interpreter; `argv` is null where check can't run it on this tree |
| `reason` | null, or why `argv` is null: one starting `skipped:` is a skip, any other a failure |
| `inputs` | `{"closure", "paths"}`: paths and globs from the root that cover what the task reads, and the function that lists them (`suites.inputs_of`, with the runner's files; `lean_audit.inputs`; circuit-check's workspace closure). `["."]` is the whole tree, for a job that declares none |
| `needs` | the artifacts it reads that another task makes: `lean-build:verity/protocols/flock/verifier/.lake/build`, `flock-circuit-build:$FLOCK_CHECKOUT` |
| `outputs` | the artifacts it makes (a build's) |
| `deps` | the task ids that must end before it starts (its job's ci.toml `needs` and `after`) |
| `beside` | the tasks that build what it builds (a spec's audit and the packages its `proved_in` names), so best on one worker; no order |
| `resources` | `{cores, mem_gb, disk_gb}` it asks of a worker: 32, 128, 60 for a Lake package's audit; 1, `suites.TEST_PROCESS_GB`, 10 for a suite that runs in one process; 16, 64, 20 for the rest |
| `cost_s` | its measured seconds (`check.measure`: this machine's caches, `--gather` the newest recorded checks' too, `--costs` a `{id: seconds}` over them), null when none |
| `key` | SHA-512 over its id, argv, cwd and what it reads; null when it has none |
| `unkeyed` | null, or why `key` is null |

A task's key hashes what a suite's key hashes (`suites.components`: each input's git object id, its fixtures, the
resolved lock entries and installed versions, Python, the platform, the base image and the `VERITY_*`/`FLOCK_*` variables) under
the environment check gives that step, a split suite's part `suites.part_key` (its own test files and the suite's shared
inputs), and its argv pins the partition (`SUITE:K/M@digest`, which `suites.py` refuses on a machine whose record partitions
otherwise); a Lake package's, `lean_audit.components` (its content and its path dependencies' and
provers', the auditor's files, the toolchain pin) with Python and the base image; a job in `check.STEP_KEYS`, the git ids of its
`inputs` paths (which cover what that step's key reads of the tree) under the environment a suite's key reads. What a step's
key reads of the machine (the kernel release, `rustc -vV`, the slot's paths) check's own cache keys where the task runs, so a
lister and a worker on other slots or hosts list one key. So keys are of the machine that lists them only as a suite's are
(Python, the platform, the base image): list on a worker. These have no key yet, and run whenever they're in the graph: the
builds (no declared inputs, read guard or keyed outputs), circuit-check's slices (it keys each target, from a task list only a
run builds), and the jobs that keep no verdict (`lean-unit-cut`, `ratchet-gate`). The preflight is no task: a
worker's `uv sync --locked` refuses a lock that isn't current, each suite fetches the fixtures it reads, and the repository's
suite holds the lints.

`run ID` runs that task here as `check.py` runs that part in a shard of its own (`check.run` with only that part, without the
preflight), its `deps` beside it until artifacts are shared, and writes `{"schema", "id", "verdict": passed|failed|skipped,
"key", "seconds", "outputs": {"dir", "files", "artifacts"}, "steps": [check's step records]}` to stdout or `--out`; the steps'
progress goes to stderr. It exits 0 for passed or skipped, 1 for failed, 2 on a usage error or a refused graph.

`--graph art:ID|FILE` names the graph ID was listed in: the executor's `tasks-graph/v1` artifact, fetched from the evidence
store, or a file holding `list --json`'s document. The run takes the graph's `splits` (each suite's M, partition digest and
timings) and `circuit_shards` instead of deciding them from this machine's record, so any id the graph lists, a suite listed
whole or a part `SUITE:K/M`, runs the listed files under the listed key on any worker; a key that still differs (another
tree or environment) is named on stderr. It refuses a graph whose `commit` isn't this tree's HEAD or that doesn't list ID.
Without `--graph`, the tasks are this machine's: a split suite's part runs on the partition `--timings` gives (`{suite: the
listing's splits timings}`), else on this machine's timing record. A suite's verdict is the join of its parts'
(`suites.combine_parts`).

`costs` writes `{task id: seconds}`, what the executor took for the tasks of a graph it ran (`--graph`, read as `run` reads
one, of any commit), as `list --costs` reads it. Each task is priced from its newest `pass` verdict label on `task:<key>` (an
unkeyed task's key is its check's, `research.executor.keyed`), whose `ref` is the run that earned it: the sum of the task's own
steps' seconds in that run's `out/task-result.json` (not a dep's build run beside it), else the run's wall seconds, which also
count its wait for a slot. `--seconds FILE` (`{task id: seconds}`, as the executor's queue records them) is taken over the
store's. A split suite's parts are summed back to `JOB:SUITE` and circuit-check's slices to `circuit-check`, each over the
share of the job the priced parts cover (the graph's partition's shares; a slice 1/M), so the next listing cuts them anew. A
failed task and one the executor didn't run are unpriced, and stderr names them. Given several graphs, each id gets its most.
Circuit-check assigns its targets to shards by the hash of each one's id and keeps each target's pass under its own key, so a
changed shard count reruns no target.

A recorded check publishes its live record (`live.py`) on the store's remote as each step ends and every few minutes, so
it can be followed from any machine with the store's key, without its node: `verity ci ls` lists the recent checks, running
and ended, and `verity ci inspect RUN` shows a check's steps with their state, elapsed time, an ETA from each step's median
over recent recorded checks, and its reds so far. What isn't recorded shows as unknown, never as passed.

## Landing

PRs land through the verified queue (`research queue vqueue`): its flags are `drive.toml` and its admission rules are
`queue.toml`. A PR's ready mark runs the landing gates `drive.toml` names on its head, and its warn gates
(`gates/markdown_caps.py`) too, whose failure only warns. The loop merges the marked PRs that are reviewed or low-risk onto
`main` as a candidate, checks it (by default graph2's tasks, with `lean-agreement` and `ratchet-gate` run beside them; under
`--gate check`, every job, those off the gate too, in shards: `train.sh --deferred`), and
moves `main` only to a candidate with no new failure, each gating task passed or failing on its base too; `post_train.py`
then runs the vLLM regression's rebuilds and fast neutrality tier. A failure the candidate's base shares blames no PR; any
other's suspects are the PRs whose change touches what it names: a lone one is ejected, several are bisected. A red job off
the gate holds nothing. The team skill's "Landing" is the procedure. A layout move's PR (`Move: tools/move/MAP.toml` in its
body) lands on its ready mark alone once the queue's rerun of its map is certified (`tools/move/README.md`);
`move_check.py` certifies a pure move (`verity check move BASE HEAD`).

## The public boundary

The public monorepo is exported one way from this repository. `public_boundary.toml` is the decided list of what it holds.
`public_boundary.py` reads what each public file needs (Lean imports, Lake requires, Python imports and dependencies, anchored
paths, commands) and fails on an edge into a file the export doesn't copy that `public_boundary_allowed.txt` doesn't list; the
list only shrinks (`tests/test_public_boundary.py`). `public_export.py DEST` is the export's dry run: the public set as one
commit of fresh history, its manifests pruned to what it copies, each Lean lock stating only what it copies
(`tools/verity/lean/public_lock.py`), the leftovers a public reader shouldn't meet counted by area, and the third-party pins. A
Lean library's root file it doesn't copy, an aggregate importing what isn't public, it writes as the imports of the modules
below it that it does copy, so the library builds whole, and a module only files it leaves out import it lists as a
root of the package's lock, so the Lean check builds it. It
never rewords a lock: one whose kept text holds lane text, or names a declaration, a module or a path it leaves out, refuses
the export by file, section and entry until the lock is reworded, unless the boundary lists it as known residue
(`[[residue]]`); with `--build` (a node's work: `research run`), each Lake package's `lake build`, the workspace's
`uv sync` and the licenses of what they fetched. It publishes nothing.

## Settings

Every environment variable check reads is declared once in `settings.py`, with its type, default, doc and role: `value` (hashed
into the run's `result.json` under `settings`), `where` (telemetry there, with the reason it changes no verdict) or `debug`
(refused in a recorded run, and by `train.sh`). `check.py` and `suites.py` refuse to start with a `CHECK_*` or `LEAN_SLOT*`
variable it doesn't declare. Reads outside it are held by `tools/verity/check/tests/test_settings.py` to a list that only shrinks.
