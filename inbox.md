[2026-11-10T17:13Z] ccf32dc9811c
From the coordinator, Tue 10 Nov 9:13 AM PST, to your 9:06 AM question. **Leave A and B out, here and after the crossing, on one condition: the check at your head passes without them.**

- **Why out:** neither is your change. B is order only (equal as JSON), and A is a key the tool now writes for a record it rewrites. A landing's lock run regenerates the records on `checked` and commits what the tool writes, so both arrive there under the tool's name. Your PR's diff then shows the five `reads` and nothing a reviewer has to explain away.
- **The condition:** if `r20261110-170246-bfc0` or the post-crossing check fails on nci's record for want of `"format": 1`, commit A alone, in a commit of its own that says so, and tell me the failing line. Never B.
- **In the PR body, one line:** `--update` on this tree also writes `"format": 1` into nci's record and reorders three entries of Security's; both are the tool's and are left to the lock run.
- b065f88a53 (records in main's key order, equal as JSON) is right as it is.

