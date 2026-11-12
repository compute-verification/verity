[2026-11-12T22:28Z] 7ad9db5975ac
THREADS, Thu 12 Nov, 2:30 PM PST.

**The must-fix is in: a script whose shell wrote the stop record now ends 143, never 0. #9505's head is `893b17bf62f9034341ae0128a7fc8df1c3cdd7f4`; it is still a draft, and its body lists the new test.**

- Suite, RAN `r20261112-221652-ab0c` (`benchmarks/training_slowdown`, slow tests included): "1 suites: 1 passed (0 cached), 0 failed; 8.3 min of test time" (580 passed).
- The new test is case `quits` of `test_job_shell::test_a_queued_jobs_environment_is_its_runs_own_under_tmp_and_gone_when_it_ends`: a `uv` that exits 0 on SIGTERM while the shell waits on it.
  - With the fix: status 143, the record is `stop_record`, nothing after `uv` started, and `store_state` reads the run as stopped.
  - Fix removed (RAN locally): `assert 0 == 143`. With that assertion also skipped, `store_state` gives `assert 'done' == 'stopped'`.
- The `ENV` line, changed: the trap's record branch now ends `... > "$OUT/run.json"; [ "$status" -ne 0 ] || status=143; }}; exit $status`. Its comment says that status 0 with a record is still the job's own record of its end (`store_state`).
- Re-wrapped, words unchanged: the three README lines and `run.py`'s docstring line that I had lengthened. The docstring's exit codes also lacked the 4 for a stall; that is added.
- `main` moved to `783b526ab6`, but nothing under `benchmarks/training_slowdown`, so I didn't merge it.
- What you got wrong: nothing. One note: `/tmp` on my VM was cleared between turns and the checkout was back on `main`, so I re-checked out the branch before editing.

