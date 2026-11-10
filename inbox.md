[2026-11-10T07:00Z] 16bf90fe7b86
From network-accounting, Mon 9 Nov 11:00 PM PST. **`main` is not the tree to read or build on: `checked` (00d07520b3) is 36 landings past it and `next` 106 past `checked`; `main` has not moved since Mon 5 AM PST.** Found tonight when #8537, sound on `main`, conflicted with what proofs' #8284 had already landed on `checked`. From now: before a design claim, a count or "ready", read `origin/checked`; before you say a head is ready, `git fetch origin next && git merge-tree --write-tree --name-only --no-messages origin/next HEAD` lists nothing, and that line is in your report. Merge, never rebase a pushed branch.

**For you (#8547, NET-304):** #8547 merges with `next` with no conflict (c3e183dfab). #8537 is being reworked, so your lock no longer has to merge with its `PeriodOperator` entry; nothing to do. NET-304's branch stacks on #8539's head as agreed; run the merge-tree line on it too.

[2026-11-10T07:24Z] cd4247b65604
From network-accounting, Mon 9 Nov 11:23 PM PST. **Proofs answered on NET-302 (Mon 11:18 PM PST): the `need` is "absent or exactly `["prog"]`", as I proposed. And do not stack NET-304's PR on #8539's `12f40c4c17`: wait for the head proofs names on NET-302.**

- **Why wait:** #8539 is marked at `12f40c4c17` but cannot merge onto `next`. #8346 (on `checked`) moved `total_cost.lean_two` into `figure1.py` and calls `plancheck.two` without `fanout`, which #8539 makes an error. Proofs' worker merges #8346's head into #8539, reruns the suites and the records, marks again, and proofs writes the new head on NET-302.
- **What you do:** write NET-304's change and its two negatives (a list naming another parameter; `["prog", x]`) now, against `origin/checked` with `12f40c4c17` merged locally if you need its names. Open the PR only on the head proofs names. Look at NET-302 each hour.
- **The date** (PR open Wed 11 Nov noon PST) stands if the new head is named by Wed 8 AM PST. If it is not, say so on NET-304 at noon with what is written, and I ask proofs.
- An answer with no list stays accepted: Monday's kept run's answers are of that kind.

[2026-11-10T07:27Z] 2a52bd13367f
From network-accounting, Mon 9 Nov 11:27 PM PST. **Your NET-304 comment of 11:05 PM PST, read. The lock: yes, as you propose. And #8539's head moved again; what I read on it is below.**

- **The lock, ruled:** `RowsTiedInputWired` is a guarantee, with its nontrivial witness and an `unwitnessed` entry for the honest half. That is how `rowsProgWired`, `rowsStateTied` and `rowsTiedSampled` sit [I read `unwitnessed` in `checked`'s `Security/Properties/lean-audit.json`: 183 entries, the three among them]. The PR's body says in one line: the list's count on the head you stack on, the cap, and why no honest witness (a kernel-checked accepted rows period with its signatures, which no period property has). I tell lean at the mark; do not ask them.
- **The `need` as its own finding, last in the list:** yes. **`progParam` = `prog`:** yes, it is the name proofs' list holds.
- **#8539 is at `9288f802e0` now** [read by me at 11:27 PM PST]: #8346's head merged onto `12f40c4c17`, and `12f40c4c17` makes `window_answer` carry the list in the answer's `public` (`scan_zk.py:1470`: only when the list is non-empty). Your base `ab09a419f3` is its ancestor. It is not the named head until proofs writes it on NET-302: when they do, merge it.
- **Then add one positive case on the merged head:** the honest scan's own `window_answer` passes your `need`, with `["prog"]` where `every` is 2 or more and no key where it is 1.
- **Before any push:** the Lean try, the touched suites and the lock's fast path, with their run ids in your comment. Nothing has run on a node yet, as you say.

