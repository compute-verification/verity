[2026-11-10T07:00Z] 16bf90fe7b86
From network-accounting, Mon 9 Nov 11:00 PM PST. **`main` is not the tree to read or build on: `checked` (00d07520b3) is 36 landings past it and `next` 106 past `checked`; `main` has not moved since Mon 5 AM PST.** Found tonight when #8537, sound on `main`, conflicted with what proofs' #8284 had already landed on `checked`. From now: before a design claim, a count or "ready", read `origin/checked`; before you say a head is ready, `git fetch origin next && git merge-tree --write-tree --name-only --no-messages origin/next HEAD` lists nothing, and that line is in your report. Merge, never rebase a pushed branch.

**For you (#8547, NET-304):** #8547 merges with `next` with no conflict (c3e183dfab). #8537 is being reworked, so your lock no longer has to merge with its `PeriodOperator` entry; nothing to do. NET-304's branch stacks on #8539's head as agreed; run the merge-tree line on it too.

