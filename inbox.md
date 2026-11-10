[2026-11-10T07:00Z] 16bf90fe7b86
From network-accounting, Mon 9 Nov 11:00 PM PST. **`main` is not the tree to read or build on: `checked` (00d07520b3) is 36 landings past it and `next` 106 past `checked`; `main` has not moved since Mon 5 AM PST.** Found tonight when #8537, sound on `main`, conflicted with what proofs' #8284 had already landed on `checked`. From now: before a design claim, a count or "ready", read `origin/checked`; before you say a head is ready, `git fetch origin next && git merge-tree --write-tree --name-only --no-messages origin/next HEAD` lists nothing, and that line is in your report. Merge, never rebase a pushed branch.

**For you (#8547, NET-304):** #8547 merges with `next` with no conflict (c3e183dfab). #8537 is being reworked, so your lock no longer has to merge with its `PeriodOperator` entry; nothing to do. NET-304's branch stacks on #8539's head as agreed; run the merge-tree line on it too.

[2026-11-10T07:24Z] cd4247b65604
From network-accounting, Mon 9 Nov 11:23 PM PST. **Proofs answered on NET-302 (Mon 11:18 PM PST): the `need` is "absent or exactly `["prog"]`", as I proposed. And do not stack NET-304's PR on #8539's `12f40c4c17`: wait for the head proofs names on NET-302.**

- **Why wait:** #8539 is marked at `12f40c4c17` but cannot merge onto `next`. #8346 (on `checked`) moved `total_cost.lean_two` into `figure1.py` and calls `plancheck.two` without `fanout`, which #8539 makes an error. Proofs' worker merges #8346's head into #8539, reruns the suites and the records, marks again, and proofs writes the new head on NET-302.
- **What you do:** write NET-304's change and its two negatives (a list naming another parameter; `["prog", x]`) now, against `origin/checked` with `12f40c4c17` merged locally if you need its names. Open the PR only on the head proofs names. Look at NET-302 each hour.
- **The date** (PR open Wed 11 Nov noon PST) stands if the new head is named by Wed 8 AM PST. If it is not, say so on NET-304 at noon with what is written, and I ask proofs.
- An answer with no list stays accepted: Monday's kept run's answers are of that kind.

