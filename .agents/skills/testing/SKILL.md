---
name: testing
description: >
  How Verity's tests are organized, run and cached, and the rules every test follows. Each package's tests live in its own
  tests/ and run as one suite in their own process (`uv run tools/verity/check/suites.py`), with passes cached by input and a guard
  that fails a suite reading files outside its declared inputs ([tool.verity.tests] inputs). Covers the `slow` mark and
  --quick, opt-in heavy checks, fixtures, determinism, and dummy credentials (GitGuardian). Use when writing, moving or running
  tests, adding a package or test directory, a suite fails with "read files outside its inputs", a test is slow or flaky, or
  a test needs fixtures, credentials, a GPU or a pod.
---

# Testing

A **suite** is a directory whose `pyproject.toml` configures pytest (`[tool.pytest.ini_options]`, `testpaths = ["tests"]`):
every workspace package, and the root's, which checks invariants of the whole tree. `tools/verity/check/suites.py` runs
each suite in its own process from its own directory, and `check` runs the same runner as its pytest step.

## Run tests

Suites run remotely through `research run --on auto` (the team skill's "Machines"; `research/auto.py` says where a job went). Give
it `--cwd source`, so the job runs in the shipped tree rather than its run directory, and `uv sync` that tree first:
`suites.py` refuses another checkout's environment. `research status RUN` checks back on it.

~~~sh
R="research run --on auto --cwd source -- bash -lc"
S="uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py"
$R "$S"                         # every suite whose inputs changed since it last passed there
$R "$S flock verity"            # these suites (a package name, a directory, or a unique part of one)
$R "$S flock -- -k tanh -x"     # arguments after -- go to pytest
$R "$S --quick"                 # without the tests marked slow
$R "$S --changed $(git rev-parse origin/main) --quick"   # only the suites your change touches: the pre-mark run (the team skill)
$R "$S --fresh"                 # ignore the cache
uv run tools/verity/check/suites.py --list                    # each suite's inputs, and whether a pass is cached (local: reads only)
~~~

To record a check of your clean commit as an Attempt, the whole check or some of its steps (`--deferred lean-agreement`),
launch `uv run python tools/verity/check/check.py --record --on vy-nebius-cpu-1 --pool research [--deferred STEP,...]` from
your checkout. It starts its own `research run` in a slot of that lane pool (the CPU nodes' check slots take train checks only),
so inside a `research run` it is refused, naming this command.

Locally, while editing, one named test file or `-k` selection of under about a minute and 1 GB is fine:
`cd verity/ml/flock && uv run pytest tests/test_layouts.py -k typed`. It doesn't cache and doesn't load the read guard. The shell
hook refuses a local pytest that names no file or `-k`; `AGENT_GUARD_LOCAL_OK=1` is the opt-in.

A pass is cached under a hash of the suite's inputs (its directory, its workspace dependencies, the paths it declares,
the lock and installed distributions, the environment), uncommitted edits included; a passing test is also cached alone,
under its module's key, which leaves out the test modules it can't reach (`suites.py`'s docstring has both keys). So a
test depends on another test module only by importing or naming it, never through what that module does at import. The
nightly `audit-main` job (`check.py --no-cache`) names any cached pass a fresh run contradicts. `suites.py --explain
SUITE` says why a suite would run, and `lean_audit.py --explain PKG` the same for a Lean package.

## Where a test goes

- **In the tests/ of the package it tests**, mirroring its modules (`verity/ml/circuits/tests/` tests
  `verity.ml.circuits`); never in another package's suite or the root `tests/`, which is only for whole-tree invariants.
- **A new package** joins the workspace (`[tool.uv.workspace] members` in the root pyproject) with its siblings' pytest
  table; a directory with tests but no importable package is a virtual member (`[tool.uv] package = false`), like
  `tools/verity/check`. A test file no suite runs fails `tests/test_repository.py`.

## What a suite may read

The runner's guard (`tools/verity/check/guard/verity_suite_guard.py`) fails a suite whose tests open a repository file
outside its inputs, in every process of the suite (pytest-xdist workers and `conftest.py` imports included), so a pass is
reused only while everything its tests read is unchanged. When it fails:

- **The package imports another at run time:** add it to `[project] dependencies`, with a `[tool.uv.sources]` workspace
  entry; a dependency's own dependency groups don't come with it. A dependency that lists `[tool.verity.tests] exports`
  (research does) is those paths of it: add a file importing code reads there.
- **Only the tests need it** (a cross-check against another package, a shared fixture, the census): list it under
  `[tool.verity.tests] inputs` in the suite's pyproject.toml, with a comment saying why, as narrowly as a directory or
  file allows. A listed workspace package is its directory alone, not its dependencies, and every listed path adds the
  suite to the quick tier of each change under it.

A Python a test starts is guarded like the test: declare the file its failure names, or give the test a stand-in.
Programs the guard can't trace (bash, git, a Python run with `-I`, `-E` or `-S` or off the suite's PYTHONPATH) fail
nothing, so list what they read, and list a path a test only checks the existence of (the guard sees opens, not
`stat()`). A test that builds or runs a Lake package lists its lakefile, and its suite goes in `AFTER_LEAN`
(`tools/verity/check/suites.py`), after `check`'s lean-build step. Locate files from `__file__` (`Path(__file__).parent /
"fixtures"`), never the working directory, and never lean on another suite's side effects: every suite runs alone.

## Keep it fast

- **A test over about 5 s gets `@pytest.mark.slow`**, which `--quick` and `-m "not slow"` skip and `check` runs; the
  runner lists slower tests without the mark. A suite's quick tier should finish in about a minute: make slow tests
  smaller before marking more.
- **Minutes-long or machine-bound checks are opt-in**, skipping unless a variable asks (`VERITY_POUS_SLOW`,
  `FLOCK_IR_SLOW`, `VERITY_BENCH_FULL`, `VERITY_REGRESSION`). Name it `VERITY_{PACKAGE}_...`, and say in the skip reason
  how long it takes.
- **A check far past ten minutes cold** stays out of `check` and runs as a recorded run on the change that needs it:
  integrations/vllm's heavy circuit-neutrality tier, `verity-vllm neutrality --tier heavy` on a CPU pod (vy-nebius-cpu-2;
  `integrations/vllm/verity_vllm/pipeline/neutrality.py`), on a change under `integrations/vllm/verity_vllm/program/`,
  `integrations/vllm/verity_vllm/engine/` or its `pipeline/build.py` meant to leave the circuits alone.
- **GPU, pod and vLLM tests skip themselves** without the hardware or runtime; their real runs are recorded runs on a pod.
- **Skip on a third-party package, never on a module of this checkout** (the guard fails that skip):
  `pytest.importorskip("torch")`, then import ours plainly.
- **A test that needs the evidence store** skips where it can't reach it with a reason containing `the evidence store is
  not reachable` (a recorded `check`, `VERITY_STORE_REQUIRED=1`, fails that skip). Inside `check` and `research run --on`
  the store is reachable only through the run's custody key, so fetch through a helper that hands it to the store CLI
  (integrations/vllm's `store_io`, C-Flock's `verifier/ci.py` `fetch`).
- **A suite over a few minutes declares `parallel = true`** under `[tool.verity.tests]`, once it passes `suites.py SUITE
  --fresh` a few times, and its tests then spread over pytest-xdist workers: they must run in any order and process, and
  session and module fixtures are built once per worker, so keep them cheap. A suite whose workers are big declares
  `worker_gb`, one worker's peak.
- In `check` a suite still running after `SUITE_TIMEOUT_S` (an hour) fails as timed out; one that honestly needs longer
  declares `timeout_s` under `[tool.verity.tests]`, with a comment giving its longest pass.

## Write tests that mean something

- **Deterministic:** seeded randomness, no ordering dependence, no network unless an opt-in variable asks, and no
  wall-clock time (AGENTS.md "Tests"): wait on the event (a child's exit, a line of its output, a FIFO the test holds
  open: `tools/research/tests/waits.py`), and inject the clock the code measures with (`monkeypatch.setattr(mod,
  "clock", ...)`). No result may depend on load, a shared machine or what holds it (GPU leases, check slots, pods), ports
  and sockets, a race between processes, or an outside service. `suites.py` reports a test that passed and failed under
  one key as `nondeterministic`: it is deleted, not fixed or rerun.
- **Isolated:** write only under `tmp_path`, and change the environment, `sys.path` or a module only through
  `monkeypatch`; the runner fails a run whose tests changed a repository file, naming it.
- **Exact:** compare bit-exact code's words, digests and pinned vectors, not approximations; a negative checks that the
  code rejects by name, not merely that it raises.
- **Statistical floors from a sweep:** a bound on a sampled quantity (a rate, a spread, a count over random draws) comes
  from a sweep over many seeds, with margin past the worst seed's value; its docstring names the sweep (seeds and worst
  value).
- **Fixtures:** small ones sit in the suite's own `fixtures/`, shared ones under the root `fixtures/`, registered in
  `fixtures/artifacts.json`. One over the 256 KiB cap is registered there and gitignored instead: `research data put
  --kind fixture/v1 --file|--tree PATH --preserve`, its id and path into the registry and its path into `.gitignore`.
  The control pod exports `next`'s registry every 5 minutes, so the fixture needs no ask. `suites.py` fetches it before the suites that read it (`research data fetch-fixtures` by hand, as before
  `cargo test`), checks every byte against its id, and keys the suite on the id. A verifier's fixtures are the prover's
  bytes, never regenerated in its test; when a test can recompute an expected output, pin its digest, not the file.

## Dummy credentials

No real credential enters the tree. A dummy is short and says what it is (`"dummy"`, `"not-to-be-copied"`). A realistic
one, to test redaction or scanning, is split into adjacent string literals no scanner reads as one (`"hf_"
"abcdefghijklmnopqrstuvwxyz0123"`); build environment blobs from a dict, never `NAME_KEY=value` in one literal.
`tests/test_repository.py::test_no_secret_like_literals` enforces this before GitGuardian's PR check, whose exclusions
and false-positive rulings live in its dashboard, not the repository.

## Before a PR

The suites your change touches pass with no read outside their inputs (the pre-mark run, the team skill's "Landing"), and a new or
changed circuit passes `circuit-check` (AGENTS.md "Circuits").
