[2026-11-12T21:49Z] 5b49f0be0cde
THREADS, Thu 12 Nov, 1:52 PM PST.

**#9505 is open as a draft at `7a51fd2d89`: a job stopped before it is up is continued, not failed. Each change has a test that fails without it, and the suite passed on a node with slow tests included (RAN `r20261112-213657-bfd0`: "benchmarks/training_slowdown 579 passed in 509 s"; "1 suites: 1 passed (0 cached), 0 failed; 8.5 min of test time").** `main` is still `5f0c008aa0`, so the head is already merged with it. Hand-off note: `training-20261112T2138Z-handoff-THREADS-stop-before-up` (synced). It has the sizing as I sent it to you.

Each change, with the test that holds it. "Fails without" was RAN locally: I removed each change in turn and ran its tests.

1. The shell writes the stop record. `ENV`'s TERM trap writes `stop_record(seed, split)` to `$OUT/run.json` when the job wrote none, then exits with the stopped command's status, so nothing after a stop starts.
   - Held by: `test_job_shell` cases `sync` and `fetch` (the record equals `stop_record`; status 143) and `ends` (no record).
   - Case `job` holds that the job's own record (here with a step) is not overwritten. That fails without the `[ -e ]` guard.
2. The state is fetched into `$(dirname "$OUT")/fetching` and moved into `$OUT/state` once `research data fetch` exits 0; the EXIT trap removes the fetch directory.
   - Your ruling 3: that command resolves every selected file and checks each blob against the manifest's sha256 before it writes any, and exits nonzero otherwise. So exit 0 is the digest check, and I added none (READ `tools/research/src/research/store/local.py`, `fetch` and `_resolve_blob`).
   - Held by: `test_job_shell` case `fetch`. A research stopped mid-fetch, with `state.json` written and no tensors, leaves no `out/state`.
3. A stopped job goes on from the latest state any of its runs kept.
   - Held by: `test_command::test_a_run_stopped_before_its_job_was_up_loses_no_earlier_state`.
4. The bound, as you ruled: `stalled` (exit 4), not failed.
   - The count is the trailing runs that kept no newer state. At a nonzero multiple of `STALLS = 3` the label is stalled, and the note names the three runs, why, and that `--continue-stalled` continues it. Because the count is a multiple, each use allows three more such stops with no record kept of the use.
   - A new state resets the count. A failed label is never run, with the flag too.
   - Held by: `test_command::test_a_job_stopped_every_time_before_its_next_checkpoint_stalls_until_told_to_continue`. Removing the bound fails it, and so does never continuing a stall.
5. `--threads N` comes from the continued state's `state.json` `record.threads`.
   - Held by: the item-3 test (`--threads 3`).
   - It is inert until #9482 lands: no record on `main` states `threads`, so nothing is passed, and the job on `main` lacks the flag. Either PR can land first.

The rule that a crash that is not a stop stays a failure is held by `test_a_stop_is_read_from_the_runs_record_and_never_from_its_exit_status` (unchanged). A finished job's record is untouched: the trap writes only when no `run.json` exists.

Left:

1. `research queue open` warned that TRN-42's labels name no goal `training/stopped-job-resumes` (its live one is `training/second-swarm-results`). I kept the line as you asked; label TRN-42, or tell me to change the line.
2. Untested: a command that exits 0 on TERM, where the trap's `exit $status` is what stops the chain. Not built: a stall bound under `NODE_STATE`, which is off by default.
3. Ruff and the root lints are clean. The only warning is `test_doc_paths`'s, all of it outside this package.

No run of mine is in flight.

