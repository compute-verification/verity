# `check`

`check` is the repository's one check. `verity check` runs `check.py` here, on `main` after each landing, and on any commit
you ask it to. The jobs it runs are data in `ci.toml`: each has an owner, a part, a group and what its failure means.

| Part | Jobs |
| --- | --- |
| `tests` | `pytest` (every test suite, through `suites.py`), `lean-suites` (the suites that read the Lean verifier's build) |
| `circuits` | `flock-circuit-build`, `flock-rust-tests` (C-Flock's Rust), `circuit-check` (the property checks every circuit passes) |
| `lean` | `lean-build` and `lean-unit-cut` (the Lean verifier and its partition check), `lean-audit` (the Lean check of every Lake package, `tools/verity/lean/`), `lean-changed` (the early Lean check on a merge, which halts the run), and `lean-agreement` (the cross-check against upstream's Rust verifier, when a change touches C-Flock) |

`verity check tests|circuits|lean` runs only those parts and what their jobs need. `check.py`'s docstring lists the rest: shards,
`--deferred STEP,...`, `--plan` and `--slots`.

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

## Landing

PRs land through the merge queue (`research queue drive`): its flags are `drive.toml` and its admission rules are
`queue.toml`. Before a PR lands, the landing gates `drive.toml` names (`gates/markdown_caps.py` among them) judge its
merge. After each landing, one run checks `main`'s tip (`train.sh --deferred`), and `post_train.py` runs the vLLM
regression's rebuilds. AGENTS.md "Checking and merging" is the procedure. A pure move that `move_check.py` certifies
(`verity check move BASE HEAD`) lands ahead of the queue.
