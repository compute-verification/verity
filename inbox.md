[2026-11-13T04:11Z] d9baf721ea3d
THREADS, Thu 12 Nov, 8:11 PM PST — TRN-153 draft

**Draft [#9608](https://github.com/compute-verification/verity/pull/9608) is open at head `14b6b448755a83cf7231ef1842c7a25b43d35302`, and the suites it touches passed on a node.**

- **Head.** `14b6b448755a83cf7231ef1842c7a25b43d35302`, branch `cursor/trn153-cpu-attention-45e7`, from main `1e4d5c993c`. Two commits: the change, then `VERSION` 4.26.0 last.
- **Run.** RAN `r20261113-032842-571b` (vy-nebius-cpu-14, `suites.py --changed 1e4d5c993c08…`). Its last line: "4 suites: 4 passed (0 cached), 0 failed; 30.2 min of test time". By suite:
  - `benchmarks/training_slowdown`: 600 passed;
  - `benchmarks/training_slowdown/grader`: 28 passed, 3 skipped;
  - `repository`: 392 passed;
  - `verity-graph`: 246 passed.
- **The six points, as built:**
  - The pin: `run.cpu_attention`, entered in `main` on a CPU after `limit_cpu` and `set_threads`, so it holds in both processes.
  - The guard: `flops.CPU_FUSED_ATTENTION` feeds the record's `unrepeatable_ops`, and `Harness.so_far` makes the run invalid. Its message names the fix: "leave the kernel to the job (no `sdpa_kernel` in the route), or run the job at `--threads 1`".
    - It holds above one thread only. At one thread it would make `test_a_jobs_trace_and_its_counter_agree_on_the_traced_step`, which asserts a fused-kernel run valid, fail: an extra rule.
  - The record: `cpu_attention` sits beside `cpu_isa`.
  - The statements: the three are rewritten as constraints. No number, run id, machine name, date or issue id is in the tree.
  - `VERSION`: 4.26.0, the number left after #9039's 4.24.0 and the grow route's 4.25.0. To be renumbered when I merge `main` after those two land.
  - Tests: two new tests. `test_resume`'s two-thread test stays.
- **One addition.** I deleted `test_resume.py`'s `ON_A_NODE` and `differ`. They told a reader to blame this CPU's fused bf16 kernel when a continued job's digests differ, and a job on a CPU no longer runs that kernel. The assertion now says only "the continued job's checkpoints aren't the unbroken job's".
- **Also RAN, locally:**
  - The new and neighbouring tests by name: 9 passed.
  - Mutations: without the pin, the first new test fails; with the guard off, the second fails.
  - `ruff check` on the changed files passes.
- **The body** lists RAN apart from READ, and says the Intel figures come from one VM because no Intel node was reachable.
- **What I could not do:**
  - No Intel node was reachable, so no Intel node checked the pin or the guard.
  - `research queue open` warned that TRN-153 carries no label naming a live goal. I left the label alone, since I post nothing to Linear.
  - The PR is not marked.

