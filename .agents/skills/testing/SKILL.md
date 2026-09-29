---
name: testing
description: >
  How Verity's tests are organized, run and cached, and the rules every test follows. Each package's tests live in its own
  tests/ and run as one suite in their own process (`uv run tools/check/suites.py`), with passes cached by input and a guard
  that fails a suite reading files outside its declared inputs ([tool.verity.tests] inputs). Covers the `slow` mark and
  --quick, opt-in heavy checks, fixtures, determinism, and dummy credentials (GitGuardian). Use when writing, moving or running
  tests, adding a package or test directory, a suite fails with "read files outside its inputs", a test is slow or flaky, or
  a test needs fixtures, credentials, a GPU or a pod.
---

# Testing

A **suite** is a directory whose `pyproject.toml` configures pytest (`[tool.pytest.ini_options]`, `testpaths = ["tests"]`).
Every workspace package is one, and the repository's own suite at the root checks invariants of the whole tree
(`tests/test_repository.py`). `tools/check/suites.py` runs each suite in its own process from its own directory, and `check`
runs the same runner as its pytest step.

## Run tests

~~~sh
uv run tools/check/suites.py                        # every suite whose inputs changed since it last passed on this machine
uv run tools/check/suites.py flock verity           # these suites (a package name, a directory, or a unique part of one)
uv run tools/check/suites.py flock -- -k tanh -x    # arguments after -- go to pytest
uv run tools/check/suites.py --quick                # without the tests marked slow
uv run tools/check/suites.py --fresh                # ignore the cache
uv run tools/check/suites.py --list                 # each suite's inputs, and whether a pass is cached
~~~

Inside one package, plain pytest works too: `cd backends/flock && uv run pytest -m "not slow" -k tanh`. It doesn't cache and
doesn't load the read guard.

A pass is cached under a hash of the suite's inputs: its directory, the directories of the workspace packages it depends on,
the paths it declares, the workspace's metadata (every `pyproject.toml` and `uv.lock`), the installed distributions, Python,
the platform, whether `lake`, `cargo` and `nvidia-smi` are on the PATH, and every `VERITY_*` and `FLOCK_*` variable.
Uncommitted edits count. Only passes are cached, in `~/.cache/verity/tests`.

## Where a test goes

- **In the tests/ of the package it tests**, mirroring the package's modules (`packages/verity/tests/ir/` tests `verity.ir`).
  Never in another package's suite, and never in the root `tests/`, which is only for invariants of the whole repository.
- **A new package** joins the workspace (`[tool.uv.workspace] members` in the root pyproject) with the pytest table its
  siblings have. A directory with tests but no importable package is a virtual member (`[tool.uv] package = false`), like
  `tools/check` and `tools/lean`.
- `tests/test_repository.py::test_every_test_file_is_in_a_suite` fails for a test file that no suite runs.

## What a suite may read

The runner's guard (`tools/check/suite_guard.py`) fails a suite whose tests open a repository file outside its inputs. That
keeps the cache sound: a pass is only reused while everything the tests read is unchanged. When it fails:

- **The package imports another at run time:** add it to `[project] dependencies` (with a `[tool.uv.sources]` workspace entry).
- **Only the tests need it,** such as a cross-check against another package, a shared fixture or the census: list the path
  under `[tool.verity.tests] inputs` in the suite's pyproject.toml, with a comment saying why. A listed workspace package
  brings its own dependencies.

Programs a test starts as a subprocess aren't traced, so list what they read. Locate files from `__file__`
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
- A suite's quick tier should finish in about a minute. When it doesn't, make the slow tests smaller before marking more.

## Write tests that mean something

- **Deterministic:** seeded randomness, no wall-clock or ordering dependence, no network unless an opt-in variable asks.
- **Isolated:** write only under `tmp_path`; change the environment, `sys.path` or a module only through `monkeypatch`. The
  runner fails a run in which the tests changed a file of the repository, and names the files.
- **Exact:** for bit-exact code, compare words, digests and pinned vectors, not approximations. A negative checks that the
  code rejects by name, not merely that it raises.
- **Small fixtures** sit beside the tests (`tests/fixtures/`). Shared ones go under the root `fixtures/`, registered in
  `fixtures/artifacts.json`. Large evidence lives in the evidence store, not in git: `tests/test_repository.py` caps a
  tracked file at 5 MiB.

## Dummy credentials

No real credential ever enters the tree. A dummy is short and says what it is: `"dummy"`, `"not-to-be-copied"`. A test that
needs a realistic one, to test redaction or scanning, splits it into adjacent string literals, which Python joins and no
scanner reads as one: `"hf_" "abcdefghijklmnopqrstuvwxyz0123"`. Build environment blobs from a dict rather than writing
`NAME_KEY=value` inside one literal.

`tests/test_repository.py::test_no_secret_like_literals` enforces this before GitGuardian's PR check sees it. GitGuardian
ignores in-repo configuration: its exclusions and false-positive rulings are set in its dashboard.

## Before a PR

- `uv run tools/check/suites.py` passes, and no suite reads outside its inputs.
- A new or changed circuit passes `circuit-check` (AGENTS.md "Circuits").
- The merge itself needs a passing `check` of the exact commit (AGENTS.md "Checking and merging").
