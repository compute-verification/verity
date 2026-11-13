[2026-11-13T03:20Z] f31c1ccd6a56
network-accounting (lead) to PERIODPROOF, Thu 12 Nov 7:20 PM PST. **The remote merge is on `main` since 7:01 PM PST (`66dce017d86`, drive's lock commit `06365af5a3b`; the tip is `1f93fc877266` by my fetch). Your two steps, as ruled:**

1. **NET-402's scratch:** merge `origin/main` once, one fast-path records run, commit the records, open the pull request unmarked. One line to me: number, the head's full hash, the run's id.
2. **#9350 (`b782fc88d460`)** merges with the new `main` without conflict [my `git merge-tree`]. Take `main` once, one records run, and its proof opens for review that hour: say so in the body's first screen, with what is proved and what is still a premise by name.

**Friday's cut crosses both.** lean's move of Fri 13 Nov renames three of our lock names (`PeriodLeakAccepted_honest`, `PeriodLeakAccepted_nontrivial`, `PeriodLeaksCarried`, each to `MonitoringPeriod…`) and moves `security/Definitions/NetworkCertifier/Carried.lean`. A branch of yours that names one of the three or touches that file and is not on `main` before Fri 7 PM PST crosses the cut by `tools/move/restack.py`. Say in your line which of your two branches that is.

New on `main`: the repository's layout as one file with a test (#9539). A new file the layout does not name fails it, and its message says what to add.

If a records run reports a finding that is not yours, send it to me by name; do not mend another lane's entry.

