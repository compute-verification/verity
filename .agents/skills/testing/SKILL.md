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

Suites run remotely through `research run --on auto` (AGENTS.md "Tests"). Its launch says where the job went, why, and
when it should start (`research inspect RUN` says it again on the VM that launched it), and warns when that is over 10
minutes away (`research/auto.py`). A machine that can start no run takes no job: the launch passes over it for the next,
saying which and why. Give it `--cwd source`, so the job runs in the shipped tree rather than its run directory, and
`uv sync` that tree first: `suites.py` refuses another checkout's environment. `research status RUN` checks back on it.

~~~sh
R="research run --on auto --cwd source -- bash -lc"
S="uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py"
$R "$S"                         # every suite whose inputs changed since it last passed there
$R "$S flock verity"            # these suites (a package name, a directory, or a unique part of one)
$R "$S flock -- -k tanh -x"     # arguments after -- go to pytest
$R "$S --quick"                 # without the tests marked slow
python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD   # here, in a second, before the pre-mark run: the no-growth lints on your change
$R "$S --changed $(git fetch -q origin main && git merge-base HEAD FETCH_HEAD) --quick"   # the pre-mark run; the mark's tier runs it on the head merged onto main
$R "$S --fresh"                 # ignore the cache, Lake's artifact cache too: what the suites test is compiled
uv run tools/verity/check/suites.py --list                    # each suite's inputs, and whether a pass is cached (local: reads only)
~~~

To record a check of your clean commit as an Attempt, the whole check (every job on the merge gate: circuit-check and
ratchet-gate, off it, only with `--deferred circuit-check,ratchet-gate`) or some of its steps (`--deferred lean-agreement`),
launch `uv run python tools/verity/check/check.py --record --on auto --pool research [--deferred STEP,...]` from your
checkout. It starts its own `research run` in a slot of the research pool `research run --on auto` would spill to (the CPU
nodes' check slots take train checks only), so inside a `research run` it is refused, naming this command.

Locally, while editing, one named test file or `-k` selection of under about a minute and 1 GB is fine:
`cd verity/protocols/flock/prover && uv run pytest tests/test_layouts.py -k round_trip`. It doesn't cache and doesn't load the read guard. The shell
hook refuses a local pytest that names no file or `-k`; `AGENT_GUARD_LOCAL_OK=1` is the opt-in.

A pass is cached under a hash of the suite's inputs (its directory, its workspace dependencies, the paths it declares,
the lock and installed distributions, the environment), uncommitted edits included; a passing test is also cached alone,
under its module's key, which leaves out the test modules it can't reach (`suites.py`'s docstring has both keys). So a
test depends on another test module only by importing or naming it, never through what that module does at import. The
nightly `audit-main` job (`check.py --no-cache`) names any cached pass a fresh run contradicts, among every pass the commit's
recorded checks hold, which it imports first. `suites.py --explain
SUITE` says why a suite would run, and `lean_audit.py --explain PKG` the same for a Lean package.

A failed test is reported failed, whoever is fixing it: no runner tolerates a red. To let one stand while its fix lands, add
an allowance to `tools/verity/check/policy.toml` (`suite NAME::TEST`, the owner's handle, a UTC expiry and a reason) in a
reviewed change: the verified queue and the quick tier take that failure as not red until it expires.

## Where a test goes

- **In the tests/ of the package it tests**, mirroring its modules (`verity/circuits/tests/` tests
  `verity.circuits`); never in another package's suite or the root `tests/`, which is only for whole-tree invariants.
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
  suite to the quick tier of each change under it and keys each of its test files in the graph, Python with its imports.
- **One test file reads what its source doesn't name** (a tree it walks, a module it loads by a computed name, what a
  program it starts reads): the graph's test-file family keys each file on its own inputs, so add a
  `[[tool.verity.tests.reads]]` entry with `tests` (the files, from the suite's directory) and `paths` (what they read,
  Python included). An entry naming no test file or no file of the tree is refused.

A Python a test starts is guarded like the test: declare the file its failure names, or give the test a stand-in.
Programs the guard can't trace (bash, git, a Python run with `-I`, `-E` or `-S` or off the suite's PYTHONPATH) fail
nothing, so list what they read, and list a path a test only checks the existence of (the guard sees opens, not
`stat()`). The graph's test-file runner fails a file that starts one unless its suite lists it under `[tool.verity.tests]
untraced`, which keys it on the suite's whole tree. A test that builds or runs a Lake package lists its lakefile, and its suite goes in `AFTER_LEAN`
(`tools/verity/check/suites.py`), after `check`'s lean-build step. Locate files from `__file__` (`Path(__file__).parent /
"fixtures"`), never the working directory, and never lean on another suite's side effects: every suite runs alone.

## Keep it fast

- **A test over about 5 s gets `@pytest.mark.slow`**, which `--quick` and `-m "not slow"` skip and `check` runs; the
  runner lists slower tests without the mark. A suite's quick tier should finish in about a minute: make slow tests
  smaller before marking more. The pre-mark run (`--changed REV --quick`, also the tier `research queue ready` runs before a
  mark) keeps the slow tests of each test module the change adds or modifies, or that imports one or a changed helper by
  name; those of the code under test run in `check`.
- **Minutes-long or machine-bound checks are opt-in**, skipping unless a variable asks (`VERITY_POUS_SLOW`,
  `FLOCK_IR_SLOW`, `VERITY_BENCH_FULL`, `VERITY_REGRESSION`, which `check`, the quick tier and `suites.py` inside a
  `research run` set). Name it `VERITY_{PACKAGE}_...`, and say in the skip reason how long it takes. A test that leaves
  `check` this way but must still run goes in `check.py`'s `OPT_IN`, which the nightly cold check (`--no-cache`) runs by
  name with its variable set; `tests/test_opt_in_tests.py` holds it to its mark.
- **A check far past ten minutes cold** stays out of `check` and runs as a recorded run on the change that needs it:
  adapters/vllm's heavy circuit-neutrality tier, `verity-vllm neutrality --tier heavy` on a 256-CPU node (`research
  pods list`; `adapters/vllm/verity_vllm/pipeline/neutrality.py`), on a change under `adapters/vllm/verity_vllm/program/`,
  `adapters/vllm/verity_vllm/engine/` or its `pipeline/build.py` meant to leave the circuits alone. Its fast tier runs
  after each landing (`post_train.py`), and a row it finds moved rings @vllm: a PR that moves one on purpose re-pins it.
- **GPU, pod and vLLM tests skip themselves** without the hardware or runtime; their real runs are recorded runs on a pod.
- **Skip on a third-party package, never on a module of this checkout** (the guard fails that skip):
  `pytest.importorskip("torch")`, then import ours plainly.
- **A test that needs the evidence store** skips where it can't reach it with a reason containing `the evidence store is
  not reachable` (a recorded `check`, `VERITY_STORE_REQUIRED=1`, fails that skip). Inside `check` and `research run --on`
  the store is reachable only through the run's custody key, so fetch through a helper that hands it to the store CLI
  (adapters/vllm's `store_io`, Flock's `agreement/ci.py` `fetch`).
- **A suite over a few minutes declares `parallel = true`** under `[tool.verity.tests]`, once it passes `suites.py SUITE
  --fresh` a few times, and its tests then spread over pytest-xdist workers: they must run in any order and process, and
  session and module fixtures are built once per worker, so keep them cheap. A suite whose workers hold more than
  `suites.py`'s `TEST_PROCESS_GB` declares `worker_gb`, one worker's peak.
- **A suite that alone makes a check's shard long declares `parts = P`** under `[tool.verity.tests]`: it runs as P parts
  by test id (`suites.py SUITE:K/P`, each collecting every file and running the tests whose node id hashes to K), the
  shard planner spreads them over shards (a lean suite's, `lean-suites:SUITE:K/P`, are named in `ci.toml`'s plans), and
  check reports the suite as one, passed only when every part did. Each part builds its own fixtures, so a suite whose
  cost is in shared fixtures gains little.
- In `check` a suite still running after `SUITE_TIMEOUT_S` (an hour) fails as timed out; one that honestly needs longer
  declares `timeout_s` under `[tool.verity.tests]`, with a comment giving its longest pass.

## Write tests that mean something

- **Deterministic:** seeded randomness, no ordering dependence, no network unless an opt-in variable asks, and no
  wall-clock time (AGENTS.md "Tests"): wait on the event (a child's exit, a line of its output, a FIFO the test holds
  open: `tools/research/tests/waits.py`), and inject the clock the code measures with (`monkeypatch.setattr(mod,
  "clock", ...)`). No result may depend on load, a shared machine or what holds it (GPU leases, check slots, pods), ports
  and sockets, a race between processes, or an outside service. `suites.py` reports a test that passed and failed under
  one key as `nondeterministic`: it is deleted, not fixed or rerun. On a CPU with AMX (a cloud VM), oneDNN's AMX kernel
  now and then returns a preempted bf16 matrix product wrong, so the runner starts every suite's process, and `research
  run` every job, with `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` unless the environment already names it (`ONEDNN_ISA`). A
  script run outside both sets it itself, before torch's first primitive, which is when oneDNN reads it.
- **Isolated:** write only under `tmp_path`, and change the environment, `sys.path` or a module only through
  `monkeypatch`; the runner fails a run whose tests changed a repository file, naming it.
- **Exact:** compare bit-exact code's words, digests and pinned vectors, not approximations; a negative checks that the
  code rejects by name, not merely that it raises.
- **Statistical floors from a sweep:** a bound on a sampled quantity (a rate, a spread, a count over random samples) comes
  from a sweep over many seeds, with margin past the worst seed's value; its docstring names the sweep (seeds and worst
  value).
- **Fixtures:** small ones sit in the suite's own `fixtures/`, shared ones under the root `fixtures/`, registered in
  `fixtures/artifacts.json`. One over the 256 KiB cap is registered there and gitignored instead: `research data put
  --kind fixture/v1 --file|--tree PATH --preserve`, its id and path into the registry and its path into `.gitignore`.
  So is a benchmark's build input that only a rebuild of its data reads (the selection a stream is built from),
  whatever its size.
  The control pod exports `main`'s registry every 5 minutes, so the fixture needs no ask. `suites.py` fetches it before the suites that read it (`research data fetch-fixtures` by hand, as before
  `cargo test`), checks every byte against its id, and keys the suite on the id. A verifier's fixtures are the prover's
  bytes, never regenerated in its test; when a test can recompute an expected output, pin its digest, not the file. A
  fixture its test module rewrites from the tree (``python MODULE --write``) is a `Writer` in `tools/verity/check/pins.py`,
  rewritten by `uv run python tools/verity/check/pins.py --write PATH`, which the test's failure names (`tests/test_pins.py`).

## Dummy credentials

No real credential enters the tree. A dummy is short and says what it is (`"dummy"`, `"not-to-be-copied"`). A realistic
one, to test redaction or scanning, is split into adjacent string literals no scanner reads as one (`"hf_"
"abcdefghijklmnopqrstuvwxyz0123"`); build environment blobs from a dict, never `NAME_KEY=value` in one literal.
`tests/test_repository.py::test_no_secret_like_literals` enforces this before GitGuardian's PR check, whose exclusions
and false-positive rulings live in its dashboard, not the repository.

## Before a PR

The suites your change touches pass with no read outside their inputs (the pre-mark run above), and a new or
changed circuit passes `circuit-check` (AGENTS.md "Circuits").
