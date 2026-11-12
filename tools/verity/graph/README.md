# `verity graph`

The fine-grained check: Verity's checks as a graph of tasks that depends only on the commit. An executor runs any part of it
with one result cache, one artifact store and one event stream; the verified queue only reads results. This package is Verity's side:
the listing (graph/v2), one task's run (result/v1) and a serial reference executor. `verity tasks` (`tools/verity/check/
tasks.py`, tasks/v1) is the coarse graph `check` runs today.

~~~sh
verity graph [--json [--packed]] [--family F ...] [--root DIR]
                                                       # the graph; --json prints its canonical bytes, --packed packed (below)
verity graph unpack [FILE]                             # the canonical bytes of the graph/v2 a packed listing packs
verity graph run ID --work DIR [--graph FILE]          # run task ID in DIR, print its result/v1 (exit 0 pass, 1 fail, 3 error);
                                                       # FILE: a listing of this checkout, packed or not, or ID's part of
                                                       # it: no lister runs
verity graph check --local [--family F ...] [--work DIR] [--retries N] [--events FILE]
                                                       # every task, serially, in order, uncached (exit 0: all passed);
                                                       # events/v1 to FILE (default DIR/events.jsonl), DIR/results.json
~~~

`--family F` keeps family F's tasks and every task they read, transitively. Usage errors and a tree that makes no graph exit 2.

## Rules

1. Keys come from declared inputs, not traced reads. A sandbox enforces the declaration: an undeclared read is an error.
2. Keys include repository paths. A pure move reruns what it moved, once.
3. Runners report facts. What gates, what may stay red and what a red triggers is the verified queue's configuration, never a runner's.
4. `tasks.toml` holds families (tens of entries), not one row per task.
5. No `owner` field anywhere in the graph, results or events: who is paged is the verified queue's policy, from paths. The listing
   refuses one in a table or a task, and `contract.result_errors` in a result.

## Layout

| path | what |
| --- | --- |
| `verity_graph/contract.py` | schema names, `canonical`, `file_digest`, `env_digest`, `key`, `result_errors` |
| `verity_graph/graph.py` | the listing: `tasks.toml`, the tree's files, each family's tasks completed and checked, digests, keys; a task's part (`part`, `given`); the packed listing (`pack`, `unpack`) |
| `verity_graph/families/<family>.py` | a family's lister, `tasks(root: Path, files: list[str]) -> list[dict]` |
| `verity_graph/runners/<kind>.py` | a kind's runner, `run(task: dict, work: Path) -> dict` |
| `verity_graph/runners/__init__.py` | dispatch by kind, `call` (a command, logged and bounded by the task's timeout) and `execute` |
| `verity_graph/local.py` | the serial reference executor |
| `verity_graph/families/build.py` and its kinds' runners (`runners/lean_build.py`, `lean_unit_cut.py`, `lean_agreement.py`, `lean_suites.py`, `flock_circuit_build.py`, `flock_rust_tests.py`) | the build family (below) |
| `tasks.toml` | `[slot]`, `[env.<name>]` and `[family.<name>]` tables |
| `tests/key_vectors.json` | the vectors Verity and the executor both reproduce |

In a module's name a family's or kind's `-` is written `_`: family `test-file` is `families/test_file.py`, kind
`circuit-binding` is `runners/circuit_binding.py`. Both registries find their modules themselves, so a new family adds its
lister, its kinds' runners, its tests and its own `tasks.toml` table, and edits nothing else.

## graph/v2

~~~json
{"schema": "verity-graph/v2", "commit": "<sha>",
 "envs": {"py": {"spec": ["pyproject.toml", "uv.lock"], "digest": "sha512:<hex>"}, "lean": {...}, "rust": {...}},
 "files": {"<path>": "sha512:<hex>"},
 "tasks": [{"id": "lean-module:security/Proofs/Flock/Claims/Recursion/AccTies.lean", "kind": "lean-module",
            "about": "compile and kernel-replay one module", "env": "lean", "argv": ["verity", "graph", "run", "<id>"],
            "vars": {}, "inputs": ["security/Proofs/Flock/Claims/Recursion/AccTies.lean", "..."],
            "reads": [{"task": "lean-module:...", "output": "olean"}],
            "outputs": [{"name": "olean", "path": "out/AccTies.olean"}],
            "resources": {"cores": 1, "mem_gb": 6, "disk_gb": 1}, "timeout_s": 900, "priority": 251, "key": null}]}
~~~

- The same bytes on any machine for one commit: no timings, store reads, environment or machine state. `--json` prints
  `canonical(doc)` and a newline.
- `commit` is HEAD's when the tree is clean, else `null`: a dirty tree's graph is never taken for a commit's.
- `files` holds every input of any task and every env's spec files, once, each `file_digest`. The tree's files are git's,
  tracked and untracked-unignored (`git ls-files -co --exclude-standard`), as POSIX paths from the root.
- An env's `spec` is its `tasks.toml` globs resolved over the tree's files, sorted. `py`: `uv.lock` and the root
  `pyproject.toml`; `lean`: every `lean-toolchain` and `lake-manifest.json`; `rust`: `verity/protocols/flock/prover/check_build.sh`, which
  pins the Rust toolchain and the Flock commit.
- Tasks are sorted by id. An id is `<kind>:<name>`, unique in the graph.
- `argv` defaults to `["verity", "graph", "run", "<id>"]`; `vars` is `{name: string}`. Both are in the key, so a lister puts in
  `vars` what a run depends on of the tree beyond its inputs' bytes: the circuit-binding family's `lookups` is the digest of
  the names, not the bytes, of the paths where its runs can look (every path under circuit-check's declared inputs, and the
  module names at the root, each workspace member's import-path entry and each directory above a declared input: the var
  `places`), so a path added, removed or renamed there reruns them and an edit reruns none. Its runner judges each run by
  the run's own report: a file read outside the task's inputs, a probe or listing outside `places`, or a library git
  doesn't have is an undeclared read.
- `inputs` are files of the tree, sorted. The core adds the kind's runner (`runners/<kind>.py` and `runners/__init__.py`) to
  every task's inputs, so a runner's change reruns every task of its kind.
- `reads` name another task's output; `outputs` are files, by relative path in the task's work directory.
- `key` is set when the task reads no artifact; otherwise it is `null`, and the executor computes it once the reads' digests
  exist.
- `priority` is how many reads the longest chain of them through the task makes, the tasks that wait on it one after another
  and those it waits on so (`graph.priorities`): the Lean modules under a lock and the lock itself have the graph's highest,
  and a task that reads nothing and that nothing reads has 0. Of the queued tasks of submissions it ranks alike, an executor
  claims one of the highest `priority` first, ahead of any order of its own (`timeout_s`, ids), so a chain never waits behind
  the tasks off it. In a task's graph (`graph.task_graph`), which holds none of the tasks that read it, a priority counts the
  chains that graph holds.

### Packed

Most of a listing's bytes are paths and ids it repeats: the test files of a suite share most of their thousands of inputs,
and a Lean module reads the `olean` and `axioms` of each of the hundreds of modules its compile loads.
`verity graph --json --packed` prints the listing as `verity-graph/v2-packed` (`graph.pack`), the same document but for
each task's `inputs` and `reads`: its inputs are runs of indices into `files`' paths in order, and its reads, under each
output name it reads, runs of indices into `tasks`. A run is two integers, how far past the previous run's end it starts
(the first, past 0) and how many consecutive indices it holds, so `[0, 3, 2, 1]` is indices 0, 1, 2 and 5. Unpacking
(`graph.unpack`, `verity graph unpack`) gives back the graph/v2 document, its canonical bytes `verity graph --json`'s: each
task's inputs in order and its reads sorted by task and output, as graph/v2 has them. `verity graph run --graph` takes a
packed listing as it takes graph/v2.

## The key rule

~~~text
key = "sha512:" + hex(SHA-512(b"verity-key/v1\n" + canonical({
        "env": <env digest>, "kind": ..., "argv": [...], "vars": {...},
        "inputs":  sorted [[path, file digest], ...],
        "reads":   sorted [[task id, output name, output digest], ...],
        "outputs": sorted [[name, path], ...]})))
canonical(x) = json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
file digest  = "sha512:" + hex(SHA-512(file bytes))
env digest   = "sha512:" + hex(SHA-512(b"verity-env/v1\n" + canonical(sorted [[path, file digest], ...])))
~~~

Sorting is by code point (Python's string order, which is UTF-8's byte order). `id`, `about`, `resources`, `timeout_s` and
`priority` stay out of the key. A read enters by its output's digest, so an artifact rebuilt to the same bytes leaves its
readers' keys unchanged (early cutoff). `contract.key(task, file_digests, read_digests, env_digest)` takes `read_digests` as
{read task: its result's `outputs`}. `tests/key_vectors.json` covers inputs alone, outputs, reads of one and of two tasks,
unicode paths, escapes in vars and argv, another env, and the fields the key leaves out; `python tests/test_contract.py
--write` rewrites it from the cases in that file, which is a change to the contract.

## Running one task

The executor makes a fresh work directory, puts each artifact the task reads there at the path its producer's output
declares (two reads or a read and an output that would land on one path are refused when listing), and runs the task's
argv with `--work DIR` appended, from the checkout's root. An executor lists the graph once and appends `--graph FILE` too,
FILE holding that listing or the task's part of it, `graph.part(doc, id)`: the task, its env, the digests of its inputs and
its env's spec files, and of each task it reads the id and the outputs it reads. A part grows with its task's own reads, not
with the tree or with what lies under the task. Given FILE, `verity graph run` runs no lister (`graph.given`): it takes FILE
when it holds the task, the outputs it reads and its env, and each file the task's key reads (its inputs and its env's spec
files) has here the digest FILE names, whatever commit FILE names, so a checkout whose bytes differ refuses it (exit 2) rather
than key a result from another tree's digests. Without FILE it finds the task with `graph.task_graph`, which runs the
family's lister but completes, digests and keys only the task and what it reads. Then it runs the task's kind's runner
through `runners.execute` and prints result/v1 as one canonical line:

~~~json
{"schema": "verity-result/v1", "task": "<id>", "key": "sha512:<hex>", "status": "pass|fail|error",
 "seconds": 41.2, "cpu_s": 39.8, "peak_rss_gb": 3.1, "outputs": {"olean": "sha512:<hex>"},
 "undeclared_reads": [], "detail": {}, "log": "sha512:<hex>"}
~~~

`execute` computes the key from the reads in the work directory (a read missing there is an error), gives the runner the
task's `timeout_s`, measures wall seconds, CPU seconds and peak memory (`getrusage` of the process and its children), digests
each declared output present, and the log `task.log`, where `runners.call` writes each command and its output. A pass
missing a declared output is a fail; a task past its timeout is a fail; a runner that raises, returns a field of the
core's, or reports an undeclared read is an error.

`error`: the task couldn't run (worker died, disk full, over its memory ceiling, an undeclared read); never a result, and the
executor retries it. `fail`: a result, never retried.

An artifact is its bytes, without mode bits: a runner that runs one it reads makes its copy executable. Finding the task
reads this package, `tasks.toml` and what its family's lister reads; a sandbox holds the runner's commands to the task's
inputs, not that.

## events/v1

The executor writes one JSON line per event; the dashboard reads only these and the graph.

~~~json
{"schema": "verity-events/v1", "ts": "...", "submission": "...", "purpose": "after-landing|ready|lean-check|nightly|adhoc",
 "commit": "<sha>", "task": "<id>", "key": "...", "event": "queued|hit|started|finished|errored|cancelled",
 "worker": "...", "wait_s": 0, "seconds": 0, "status": "...", "peak_rss_gb": 0, "hit_from": "<submission>"}
~~~

`verity graph check --local` writes them too, to its work directory's `events.jsonl` or `--events FILE` (`purpose` adhoc,
`worker` local, never `hit`: it has no cache; a task whose read didn't pass is `cancelled`).

## The reference executor

`verity graph check --local` lists the graph once and runs it serially in `graph.order` (every task after the tasks it reads,
ties by id), each task in a fresh work directory under `--work` with its reads copied from the run's artifact store (by
digest), as its argv with `--work DIR --graph PART`, PART its part of that listing written beside DIR, so that no task's run
lists again, on a dirty working tree too. It takes a result only when it is result/v1 of that task, its key is the one
computed here from the reads' digests, it reports no undeclared read, and each output's bytes are the digest it names;
anything else is an error. An error is retried `--retries` times (default 1) and never becomes a result; a fail is never
retried; a task whose read didn't pass is cancelled. Each task's last result/v1 (its pass or fail, else its last error) is in
`--work`'s `results.json`.

## The build family

The check's steps that build the Flock verifier and run what it built, and Flock's Rust, as `tools/verity/check/check.py`
runs them, one task a step:

| task | env | runs | inputs | outputs |
| --- | --- | --- | --- | --- |
| `lean-build:verity/protocols/flock/verifier` | lean | `lake build` there (`suites.LAKE_BUILD`) | the files of the package and of the packages of the tree it requires by path, but each one's `lean-audit.json` | each `lean_exe` of its `defaultTargets`, at `.lake/build/bin/<exe>`; `lake-build`, each of those packages' `.lake/build` but the executables, in one archive |
| `lean-unit-cut:verity/protocols/flock/agreement/unit_cut_agree.py` | py | `unit_cut_agree.py --lean VERIFY --out WORK/lean-unit-cut` | the script and its import closure (the test-file family's `Imports`) | `lean-unit-cut/result.json`, the cuts and how many agreed |
| `lean-agreement:verity/protocols/flock/agreement` | py | `check.py --lean-agreement WORK UPSTREAM INPUTS VERIFY` | that directory, and what `check.py` reads as it starts (`families/build.py`'s `CHECK_READS`) | `flock-agreement/ci/sets.json`, ci.py's verdicts per set |
| `lean-suites:SUITE`, `lean-suites:SUITE:K/M` | py | `suites.py SUITE --lean-built --json ... --timeouts --jobs N --mem-gb G` on each suite of `suites.AFTER_LEAN`, or with `--files K/M` on each part of one `families/build.py`'s `PARTS` splits by test file, or of one that declares `parts`, by test id; a parallel suite on the family's slot, another in one process | the whole tree, as tasks/v1 keys a job that declares no inputs: those suites build and run Lake packages, which no tracer follows | `suites-lean.json`, suites.py's report |
| `flock-circuit-build:verity/protocols/flock/prover/live` | rust | `verity/protocols/flock/prover/check_build.sh` | the script, its patches and the live crate (`check.rust_tests_inputs`) | none: `cargo check` builds nothing a reader runs |
| `flock-rust-tests:verity/protocols/flock/prover/live` | rust | `check_build.sh test`, cargo's jobs and test threads at the task's cores | as flock-circuit-build's | none |

`lean-build`'s outputs sit at their paths in a built tree, where the suites that run them look (the Flock suite's
`artifacts`); `lean-unit-cut` and `lean-agreement` read `flock-verify` and get its path as their var `verify`, and
`lean-suites` reads every one and puts it at that path in the tree before suites.py runs (its var `bin`), and unpacks
`lake-build` into the tree and runs `lake build` there under check's lock (its var `tree`), once a tree (the stamp
`.lake/graph-build.sha512`, which lean-build writes in its own), so the tree is as check's lean-build step leaves it: the
suites' own `lake build`s of the verifier, at once in every test process, find every module built, where on the executables
alone each built the whole verifier in one `.lake` and the links broke. A part's task
(its var `files`) hands suites.py a timing record with none of the suite's files (`--timings`), so part K is every M-th test
file by path from the K-th, the same wherever it runs, and the M parts hold each file once. A lean-suites fail names each
failed suite with its failed tests (`detail.failed`), which the verified queue names, allows and blames as that suite's red. A
run that ran nothing is no pass: ci.py skipping every set of the agreement, a partition check that compared no cut, a suite
suites.py didn't run. Each is an error, so no result. The agreement fetches the upstream
build and inputs `upstream.json` pins (`research data fetch`, as ci.py fetches what it pins) and holds them to the pin's
sha256, so the store and its client are no input. Those fetches and ci.py's read the store's remote that the runner's
environment names: a `research run` job on a node with no store config of its own gets one with `--store-read`, and without
one the task is an error that says so. Each runner reports a fail only for the tree's fault and an error for the
machine's: a toolchain that didn't set up, a pin that didn't fetch, check.py's `cannot run here`, a command a signal killed,
and for `check_build.sh` any failing exit whose log shows no refused patch and no crate that didn't compile. The verified
queue's loop counts no lean-agreement task's result: it takes lean-agreement, and ratchet-gate, which no family lists, from a
train.sh run beside the executor's check (`research.landing.SLIM_STEPS`, `research.vqueue_slim`).

## Writing a family

A family's lister returns graph/v2 tasks without `priority` and `key`, for the tree at `root`, reading nothing but files under
`root` (its `files` is the tree's file list). The core fills what a task leaves out from its table, `about` from the table's
template over the task's `id`, `kind`, `name` (the id after `<kind>:`) and `vars`, and refuses a field outside graph/v2, an
input that isn't a file of the tree, a read of a task or output that doesn't exist, a cycle, a path two things land on, and
resources (a table's or a task's) past `[slot]`, the largest slot an executor leases a task, which no slot would ever take.
`graph.covered(root, family, files)` is the files the table's `files` globs cover. Test it on a toy checkout
(`tests/conftest.py`'s `toy_tree`, which writes a `tasks.toml`, runner stubs and files into a fresh git repository), and the
whole tree's invariants (`tests/test_invariants.py`) cover it once it lists.

A runner returns `{"status": "pass"|"fail"|"error", "detail": {...}, "undeclared_reads": [...]}` (the last two optional) and
runs its commands through `runners.call(argv, work, cwd)`, which logs them and stops them at the task's timeout.

## Invariants

Each but 5 has a test: `test_invariants.py` on the whole tree (slow), `test_graph.py` and `test_local.py` on a toy checkout.

1. `verity graph` gives the same bytes in two checkouts of one commit at different paths (two `git worktree`s, listed from
   different working directories and environments).
2. It opens nothing outside the tree but the interpreter's own files (an audit hook on `open`, `os.listdir`, `os.scandir`).
3. Keys reproduce `tests/key_vectors.json`.
4. No `owner` field.
5. It lists the whole graph in under 60 CPU seconds of one core, as the tree is and with its Lean families at their size once
   its modules are `module` files (`tests/ported_listing.py`: of each module a task reads, `olean` and `ir`; seven outputs a
   module). It is measured on an idle machine (`verity graph --json` and `ported_listing.py` under `getrusage`), not in a
   suite: on a node at load 160 the same listing takes twice the CPU seconds it takes idle (72 against 36), so a suite run
   beside other work fails it whatever the tree.
6. Errors never become results; a fail is never retried.
7. The reference executor lists once: no task's run calls a lister, on a clean tree or a dirty one.
