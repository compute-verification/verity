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

A **suite** is a directory whose `pyproject.toml` configures pytest (`[tool.pytest.ini_options]`, `testpaths = ["tests"]`).
Every workspace package is one, and the repository's own suite at the root checks invariants of the whole tree
(`tests/test_repository.py`). `tools/verity/check/suites.py` runs each suite in its own process from its own directory, and `check`
runs the same runner as its pytest step.

## Run tests

~~~sh
uv run tools/verity/check/suites.py                        # every suite whose inputs changed since it last passed on this machine
uv run tools/verity/check/suites.py flock verity           # these suites (a package name, a directory, or a unique part of one)
uv run tools/verity/check/suites.py flock -- -k tanh -x    # arguments after -- go to pytest
uv run tools/verity/check/suites.py --quick                # without the tests marked slow
uv run tools/verity/check/suites.py --fresh                # ignore the cache
uv run tools/verity/check/suites.py --list                 # each suite's inputs, and whether a pass is cached
~~~

Inside one package, plain pytest works too: `cd verity/ml/flock && uv run pytest -m "not slow" -k tanh`. It doesn't cache and
doesn't load the read guard.

A pass is cached under a hash of the suite's inputs: its directory, the directories of the workspace packages it depends on,
the paths it declares, the workspace's metadata (every `pyproject.toml` and `uv.lock`), the installed distributions, Python,
the platform, whether `lake`, `cargo` and `nvidia-smi` are on the PATH, and every `VERITY_*` and `FLOCK_*` variable.
Uncommitted edits count. Only passes are cached, in `~/.cache/verity/tests`.

In a `parallel = true` suite each passing test is also cached on its own, even when the suite failed, under its module's key:
the suite's key less the sibling test modules it can't reach (by import, or by name in a string). A run after a fix to one test
module re-runs that module and whatever reaches it, and counts the rest as `reused`. So a test depends on another test module
only by importing or naming it, never through what that module does at import (a top-level call puts it in every key). `--fresh`
reuses nothing; the nightly `audit-main` job runs `check.py --no-cache` on `main` and names any cached pass it contradicts.
`uv run tools/verity/check/suites.py --explain SUITE` says why a suite would run (the input, dependency or variable that moved since the
nearest pass), and `lean_audit.py --explain PKG` the same for a Lean package. A test that passed and failed under one key is
reported `nondeterministic`.

## Where a test goes

- **In the tests/ of the package it tests**, mirroring the package's modules (`verity/ml/circuits/tests/` tests `verity.ml.circuits`).
  Never in another package's suite, and never in the root `tests/`, which is only for invariants of the whole repository.
- **A new package** joins the workspace (`[tool.uv.workspace] members` in the root pyproject) with the pytest table its
  siblings have. A directory with tests but no importable package is a virtual member (`[tool.uv] package = false`), like
  `tools/verity/check` and `tools/verity/lean`.
- `tests/test_repository.py::test_every_test_file_is_in_a_suite` fails for a test file that no suite runs.

## What a suite may read

The runner's guard (`tools/verity/check/guard/verity_suite_guard.py`) fails a suite whose tests open a repository file outside its
inputs, in every process of the suite, including pytest-xdist workers and `conftest.py` imports. That
keeps the cache sound: a pass is only reused while everything the tests read is unchanged. When it fails:

- **The package imports another at run time:** add it to `[project] dependencies` (with a `[tool.uv.sources]` workspace entry).
  A dependency's own dependency groups don't come with it. A dependency that lists `[tool.verity.tests] exports` (research
  does) is those paths of it: when the failure names another of its files that importing code reads, add the file there.
- **Only the tests need it,** such as a cross-check against another package, a shared fixture or the census: list the path
  under `[tool.verity.tests] inputs` in the suite's pyproject.toml, with a comment saying why. List what the tests read, as
  narrowly as a directory or a file allows: a listed workspace package is its directory alone, not its dependencies, and
  every listed path adds the suite to the quick tier of each change under it.

A Python a test starts is guarded like the test, and the failure names the test and the file the process read: declare the
file, or give the test a stand-in for it. Other programs (bash, git, a Python run with `-I`, `-E` or `-S`, or with a PYTHONPATH
that replaces the suite's) aren't traced, and nothing fails when they read outside the inputs, so list what they read; a Python
they start on the suite's PYTHONPATH is traced again. The guard sees opens, not `exists()` or `stat()`, so list a path a test
only checks the existence of too. Locate files from `__file__`
(`Path(__file__).parent / "fixtures"`), never from the working directory. Never lean on another suite's side effects, such as
a `sys.path` entry another test file added: every suite runs alone.

## Keep it fast

- **A test over about 5 s gets `@pytest.mark.slow`.** `--quick` and `-m "not slow"` skip it, and `check` runs it. The runner
  lists tests that took longer without the mark.
- **Minutes-long or machine-bound checks are opt-in:** they skip unless a variable asks for them, as `VERITY_POUS_SLOW`,
  `FLOCK_IR_SLOW`, `VERITY_BENCH_FULL` and `VERITY_REGRESSION` do. Name the variable `VERITY_{PACKAGE}_...` and say in the skip
  reason how long it takes.
- **GPU, pod and vLLM tests skip themselves** when the hardware or the runtime is absent. Their real runs are recorded runs
  on a pod (`research run --on ...`), never a laptop.
- **A test that needs the evidence store** skips where it can't reach it with a reason containing `the evidence store is not
  reachable`. Under a recorded `check` (`VERITY_STORE_REQUIRED=1`) the guard fails that skip instead, so `check` never passes
  without it. Inside `check` the store is reachable only through the run's custody key (`$CHECK_STORE_CUSTODY`), so fetch
  through a helper that hands it to the store CLI: integrations/vllm's `store_io`, or C-Flock's `verifier/ci.py` `fetch`.
- A suite's quick tier should finish in about a minute. When it doesn't, make the slow tests smaller before marking more.
- **A suite over a few minutes declares `parallel = true`** under `[tool.verity.tests]`. The runner then spreads its tests over
  the `--jobs` slots with pytest-xdist (`--dist worksteal`), one parallel suite at a time, beside the other suites on the few
  slots they need to finish when their longest does. C-Flock does
  (637 s in one process, 209 s on four). Its tests must then run in any order and in any process. Session and module
  fixtures are built once per worker, so keep them cheap. Declare it only after the suite passes
  `uv run tools/verity/check/suites.py SUITE --fresh` a few times.
- In `check`, a suite still running after an hour (`SUITE_TIMEOUT_S`) has its process group killed and fails as timed out. A
  suite that honestly needs longer declares `timeout_s` under `[tool.verity.tests]`, with a comment giving its longest pass.
  The bound is flat, not a multiple of a suite's measured time: a suite slowed by memory load on a shared node would then
  fail falsely and cost a full rerun, while a hung suite costs at most an hour of one slot (Daniel left the choice to
  velocity, 7 Oct 10:29 PM PDT).

## Write tests that mean something

- **Deterministic:** seeded randomness, no ordering dependence, no network unless an opt-in variable asks, and no wall-clock
  time: `tests/test_no_wall_clock.py` rejects sleeps, clock readings, timeouts and sleeping workloads in every suite. Wait on
  the event instead (a child's exit, a line of its output, a FIFO the test holds open: `tools/research/tests/waits.py`), and
  inject the clock or timer that the code under test measures time with (`monkeypatch.setattr(mod, "clock", ...)`).
  Don't write a test whose result can depend on load, a shared machine or what holds it (GPU leases, check slots, pods),
  ports and sockets, a race between processes, or an outside service. A test that flakes (it passed and failed under one
  key, which `suites.py` reports as `nondeterministic`) is deleted, not fixed or rerun.
- **Isolated:** write only under `tmp_path`; change the environment, `sys.path` or a module only through `monkeypatch`. The
  runner fails a run in which the tests changed a file of the repository, and names the files.
- **Exact:** for bit-exact code, compare words, digests and pinned vectors, not approximations. A negative checks that the
  code rejects by name, not merely that it raises.
- **Statistical floors from a sweep:** a bound a test asserts on a sampled quantity (a rate, a spread, a count over random
  draws) is set from a sweep over many seeds, with margin past the worst seed's value, never from one run's. Its docstring
  names the sweep (seeds and worst value), so a later change of seed doesn't turn the test flaky.
- **Small fixtures** sit beside the tests, in a `fixtures/` directory of the suite's own. Shared ones go under the root `fixtures/`, registered in
  `fixtures/artifacts.json`. `tests/test_repository.py` caps a tracked file at 256 KiB (source code aside). A larger
  fixture is registered there and gitignored instead: `research data put --kind fixture/v1 --file|--tree PATH --preserve`,
  add the id and path to the registry and the path to `.gitignore`, then `research data export`. `suites.py` fetches it
  before the suites that read it (`research data fetch-fixtures` by hand, e.g. before `cargo test`), checks every byte
  against its id, and keys the suite on the id. A verifier's fixtures are the prover's bytes, never regenerated in its test.
  When a test can recompute an expected output, pin its digest rather than the file.

## Dummy credentials

No real credential ever enters the tree. A dummy is short and says what it is: `"dummy"`, `"not-to-be-copied"`. A test that
needs a realistic one, to test redaction or scanning, splits it into adjacent string literals, which Python joins and no
scanner reads as one: `"hf_" "abcdefghijklmnopqrstuvwxyz0123"`. Build environment blobs from a dict rather than writing
`NAME_KEY=value` inside one literal.

`tests/test_repository.py::test_no_secret_like_literals` enforces this before GitGuardian's PR check sees it. GitGuardian
ignores in-repo configuration: its exclusions and false-positive rulings are set in its dashboard.

## Before a PR

- `uv run tools/verity/check/suites.py` passes, and no suite reads outside its inputs.
- A new or changed circuit passes `circuit-check` (AGENTS.md "Circuits").
- The merge itself needs a passing `check` of the exact commit (AGENTS.md "Checking and merging").
