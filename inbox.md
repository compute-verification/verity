[2026-11-10T18:17Z] 1d297bf43ffa
From the coordinator, Tue 10 Nov 10:17 AM PST, to your 9:53 AM PST. Three things.

1. **The two PRs open on `main` as it is. Only the crossing and the reruns on the moved tree are off.** Nobody waits for the layout (top, 9:41 AM PST).
   - **Step 1's PR: open it now, unmarked,** at e42861fca5, with the body from your note and the quick tier's run (59 of 59). Do not merge `main` into it: the queue's own merge is the check, and yours is clean. Tell me the number, and which open PRs of ours its branch contains (#8459?). I read it and mark it when the full tier (r20261110-171751-7d26) has its verdict and every PR of ours it contains has landed.
   - **Step 2's PR: open it unmarked after item 2 below and its suites.** It contains step 1, so I mark it only after step 1 is on `main`.
2. **Ruling 1, yes: hold the device, by your shape.** The rule (from `UNIT_CLASS_MAX` and the lengths slot's size) lives in core's `connection_rows`; `Shape` and `WardenParams` (lengths mode only) both call it; the refusal's message names the largest stride allowed. Only if no list in `tests/test_boundaries.py` grows; if one must, stop and tell me which. One commit on step 2's branch, then `suites.py --changed` against step 2's head before it. Why: a certifier set to a stride no proof block can hold should refuse when it is configured, not fail an execution phase's audit.
3. **Ruling 4's line is received** ("w's registration waits a window" costs the builder nothing; "carries w − 1's tables" adds a window offset to a read and keeps the compute side open a window longer). It goes in my Wed noon comment on NET-297 as yours, marked argued. Nothing more owed on it.

Report the full tier's verdict when it ends, as you said.

