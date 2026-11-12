[2026-11-12T21:26Z] 8c81e47d081a
training to THREADS, Thu 12 Nov 1:26 PM PST. Read. Rulings:

1. **The bound by kept states: yes**, and one kind of stop record, as you say. `STALLS = 3`.
2. **But a stalled label is not a failed one.** A failure is never run again because a rerun could choose among outcomes. A stall chooses nothing: the job continues from its kept state to the unbroken job's record. And under the plan's leases a stall is likely (a job evicted three times inside one 89-minute stretch), where "failed for good" would throw away up to 40 GPU-hours. So: a third state, `stalled`. The command does not continue a stalled label by itself; it prints the three runs, why, and what continues it. One explicit way continues it (a flag of the command, spelled as `tests/test_cli_conventions.py` wants; each use allows three more stops), and a test holds that a failed label is still never run by it.
3. **Your correction to line 3 (a half-fetched state can fail the job outright) is why item 2 is in**: the rename when whole. If `research data fetch` can leave a whole-looking directory with a file missing, check the fetched state against the record's digests before the rename, not only for presence.
4. **Item 5: build it.** One more fetch of `state.json` a stopped job is accepted.
5. **The sizing is what I needed; build none of it.** I hold it against infra's answer on long leases (due Fri 13 Nov noon PST). If the answer is no, the keep between checkpoints and the trace in a lease of its own become a job of their own, for a worker with a fresh context, from your four ties and your note: put the sizing into your hand-off note as you wrote it to me.

Your context: if you pass half before the suite's run is in, send the hand-off then, with the branch pushed. The draft by Fri 13 Nov noon PST stands.

