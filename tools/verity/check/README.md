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
the environment (Python, the platform, the base image, the `VERITY_*` and `FLOCK_*` variables but those `settings.py` declares
`where`), and the runner itself.
Markdown is left out unless a suite lists it. The read guard (`guard/verity_suite_guard.py`) fails a test that opens a
repository file outside its suite's inputs, so a reused pass never hides a change to something a test read. In a
`parallel = true` suite each passing test is also kept on its own, under its module's key.

Passes travel between machines as verdict packs (`verdicts.py`). The Lean audit and `circuit-check` reuse their own verdicts
the same way, each keyed on what it reads. A nightly uncached run must agree with the cached ones. A disagreement means some
part reads something it doesn't declare, and the fix goes in the check.

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
| `ops.py` | the one way the check and its tests reach research (see [Running from another repository](#running-from-another-repository)) |
| `tool.py`, `preflight.py` | the check's store declaration, and whether a machine is prepared for it |
| `pod_setup.sh` | prepares a check pod once, so every pod of the pool computes the same verdict keys |
| `slot.py`, `lean_slot.py` | a command in one of a node's merge-check or Lean check slots |
| `train.sh` | the lander's recorded check of a train |
| `post_train.py` | the vLLM reference regression's manifest rebuilds after a landing |
| `drive.toml`, `queue.toml` | the merge queue's flags and its admission rules |
| `gates/markdown_caps.py` | the landing gate holding Markdown to its caps |
| `move_check.py`, `anchors.py` | the certificate of a pure move, and that every path a file reaches from its own place survives it |
| `merge_map.py` | git's merge driver for `fixtures/artifacts.json` |
| `pins.py` | every file that records a Lean source's digest, with the command that regenerates it |
| `mutants.py` | deletes each check a verifier source runs, one at a time, and names each deletion its tests miss |

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
commit as an Attempt, run `uv run python tools/verity/check/check.py --record --on auto` from the checkout.

A recorded check publishes its live record (`live.py`) on the store's remote as each step ends and every few minutes, so
it can be followed from any machine with the store's key, without its node: `verity ci ls` lists the recent checks, running
and ended, and `verity ci inspect RUN` shows a check's steps with their state, elapsed time, an ETA from each step's median
over recent recorded checks, and its reds so far. What isn't recorded shows as unknown, never as passed.

## Landing

PRs land through the merge queue (`research queue drive`): its flags are `drive.toml` and its admission rules are
`queue.toml`. Before a PR lands, the landing gates `drive.toml` names (`gates/markdown_caps.py` among them) judge its
merge. Reviewed PRs land on `checked` (`land_into`); after each landing, one run of the full check checks `checked`'s tip
with every job, those off the gate too (`train.sh --deferred`), and `main` moves only to a commit whose run passed every step
on the gate, where `post_train.py` runs the vLLM regression's rebuilds. A red job off the gate reverts its culprit on `checked`
but doesn't hold `main`. The team skill's "Landing" is the procedure. A pure move that `move_check.py` certifies
(`verity check move BASE HEAD`) lands ahead of the queue.

## Settings

Every environment variable check reads is declared once in `settings.py`, with its type, default, doc and role: `value` (hashed
into the run's `result.json` under `settings`), `where` (telemetry there, with the reason it changes no verdict) or `debug`
(refused in a recorded run, and by `train.sh`). `check.py` and `suites.py` refuse to start with a `CHECK_*` or `LEAN_SLOT*`
variable it doesn't declare. Reads outside it are held by `tools/verity/check/tests/test_settings.py` to a list that only shrinks.

## Running from another repository

The check and research are to move out of verity into the team's ops repository. Three seams let the check run from there
against a verity checkout, so that the move changes paths alone:

- research is reached only through `ops.py`, which names each module and attribute the check and its tests take from it
  (`tests/test_ops.py` fails on any other import of research here).
- `ci.toml`, `drive.toml` and `queue.toml` sit at `declaration.HOME` from the root of the repository they configure, and
  `research.queue.CHECK_HOME` names the same directory. Each reader takes that root (`declaration.load(root)`,
  `queue.flags_at(repo, ref)`, `merge.declared_jobs(repo, commit)`) and never reads the working directory.
- A suite's key holds the runner (`suites.RUNNER`) as each file's git blob id, read where the runner is (`runner_ids`), under
  its name at `tools/verity/check/`. Today's keys are unchanged, and a runner elsewhere with the same bytes earns and reuses
  the same passes. A runner change moves every key, but `--changed` sees it only while the runner is in the tree it checks.
  `check.py`'s step keys hold none of the check's own files.

Still bound to the layout: `ROOT` is the runner's own `parents[3]`, and `--plan` plans a commit with that commit's
`tools/verity/check/check.py`.

### Landing in the ops repository

A change to the ops repository lands on its owner's review once the ops repository's own tests pass. Those are the check's
suite and research's, run by its runner on its own tree. No verity check gates it. Drive runs the ops repository's `main`, so
a landed change applies at drive's next tick. A runner change re-keys every verity suite, so the next verity check reruns
them all, and its full check of `checked` is what judges the change against verity. A red there that the change caused is
reverted in the ops repository.
