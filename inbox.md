[2026-11-12T21:05Z] bcec8d460356
THREADS, Thu 12 Nov, 1:05 PM PST.

**Draft PR #9482 is open and unmarked at head `bc81aeba94`, which is my branch merged with `main`; the suite and the grader's suite both pass there in full.**

- PR: https://github.com/compute-verification/verity/pull/9482. Branch `cursor/training-threads-45e7`, linked to TRN-148. (RAN `research queue open --draft`)
- The suite, run at that head: `uv run tools/verity/check/suites.py benchmarks/training_slowdown benchmarks/training_slowdown/grader`, slow tests included. (RAN `r20261112-194924-43b8`, vy-nebius-cpu-10; `source_sha` `bc81aeba94`) Its last lines:
  - `passed  benchmarks/training_slowdown 577 passed in 472 s; 577 outcomes recorded`
  - `passed  benchmarks/training_slowdown/grader 28 passed, 3 skipped in 36 s; 28 outcomes recorded`
  - `2 suites: 2 passed (0 cached), 0 failed; 8.5 min of test time`
- An earlier run at `6efd76f877`, which differs only by a line rewrap, gave the same result. (RAN `r20261112-192512-c608`)
- Ruff and the root lints that read these files pass on the VM: `test_no_wall_clock`, `test_doc_paths`, `test_retired_terms`, `test_module_strings`, `test_cli_conventions`, `test_bookkeeping` and `test_refusal_remediations`. None of their warnings names a changed file. (RAN locally)

Built as decided, with the two additions from my first message (READ: the PR diff):

- The job's own process passes `--threads` to the route's process.
- `joined` refuses a segment at another count.

The fixture:

- `one_thread` holds `run.THREADS` at 1 for the session.
- A new autouse `back_to_one_thread` sets 1 after each test.
- The slow resume test runs its jobs at `--threads 2`. It checks that a restart at the default is refused with the `--threads 2` message and leaves the state, and that the job then continues at 2 to the unbroken job's record, `threads` included.

Handoff note: `training-20261112T2104Z-handoff-THREADS` (synced).

Done: everything in the brief. Left, in order: your read and mark of #9482. No run is in flight.

A possible follow-up, which I did not build (INFERRED, from READ of `benchmarks/training_slowdown/run.py` `job_args`): the node launcher passes no `--threads`, so every queued job runs at the default, and its restarts hold to the default too.

