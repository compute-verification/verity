[2026-11-13T03:06Z] 2e6f534260ad
@training to THREADS, Thu 12 Nov, 7:08 PM PST — TRN-153: option 2 with the guard

**Ruled: option 2 with the guard.** Points 1 and 2 are accepted as you sent them; nothing goes to Daniel tonight, since no record behind a figure is a CPU job.

Build it as you described, with these six points fixed:

1. **The pin.** On a CPU, `main` enters the math kernel before either process computes anything, beside `limit_cpu`. Both processes, since the job's own process scores checkpoints.
2. **The guard.** The counter makes a run invalid when the fused CPU attention op, forward or backward, runs with more than one thread held. Its message names the fix. It holds at one thread too only if that costs no extra rule; otherwise as you wrote it.
3. **The record** names the kernel beside `cpu_isa`.
4. **The three statements** are rewritten as constraints, each stated as itself. No measured number, run id or machine name goes into the tree: what you measured goes in the PR's body and the note.
5. **`VERSION`.** A minor bump, as the draft's last commit, so it can be renumbered. #9039 holds 4.24.0 and the grow route's change takes 4.25.0; yours takes the next free number when you merge `main`. Draft against `main` now and merge once after those two land.
6. **Tests.** Your two new ones, and `test_resume`'s two-thread test stays. If it ever fails, it is deleted, not rerun.

Not in this change: refusing a CPU job's record at the grade. That is a question I put to Daniel tonight; leave it.

Draft by Fri 13 Nov, 6 PM PST, as planned: the PR as a draft, the suites the change touches on a node, and one message to me with the head and the run ids. Say in the body, apart from each other, what you ran and what you read, and that the Intel figures come from one VM because no Intel node was reachable.

