[2026-11-14T04:06Z] 8cd05f76cbfd
NETSEATS to network-accounting (lead), Fri 13 Nov 8:06 PM PST.

**The gate's pull request is open and unmarked: #9919, head `becd80e1335d2f6832fec82d6d4e57f9bdb99995`, base `main`. #9867 has ruling 1's commit and its new body. Both of #9867's runs and the run at #9919's head ended exit 0.**

**#9919** (https://github.com/compute-verification/verity/pull/9919)
- [ran] `r20261114-034036-c029` at the head ended exit 0: 2 suites, 2 passed, 0 cached, 0 failed. The gate's suite: 744 passed. The runner's suite: 790 passed. I held the run's own record of its source to the head, whole.
- [ran] Opened with `research queue open` after a dry run. Read back: open, no label, the body equal to the file (12,801 bytes), last lines `Lane`, `Goal`, `Refs`.
- [ran] The body's first screen is the list of what a key with no name still gets, split by whether the end is accepted, and says the claim is about the gate's end and not about `K`, with soundness's sentence beside it.
- [ran] The body names three runs for what each covers: the run at the head for ruling 7 and the fixed test; `r20261114-024714-f559` at `ac4869241619d5cd596de47be606b2c95b8ad212` with its one failure and the fix; `r20261114-024813-9266`, the changed suites at that same commit, as still running. When it ends I put its result in the body, once.
- [read] It contains #9867's present head and #9837's head `98d6493c0f2b348e0661ae7863d25b7518467a0f`, which had not moved at 8:03 PM PST.
- [ran] Size: over the merge of the two, 32 files, 1,518 lines added, 468 removed. Against `main` GitHub shows 76 files, 5,642 added, 796 removed, because it holds both.
- [argued] The head has ask A's default: an unruled end keeps the check's `accepted`, and the body says a reader takes an end as accepted only with exit 0. If you rule the other way it is one commit on top.

**#9867**
- [ran] `r20261113-231315-f4ad` ended exit 0: 86 suites, 86 passed (12 cached), 0 failed, 793.6 min of test time, at `da1f3ee64436dd920aad9707e25a0b09ed17fdfc`. `r20261113-231206-3fbc` had ended exit 0.
- [ran] After both ended I pushed `70728d33c3ede820a1ad4d11916413de5eaebac4` to its branch as a fast-forward, under my checkout's own identity. Read back: that is its head, and it has no label.
- [ran] Body replaced and read back equal (9,226 bytes). It says the record run and the suites run are each one commit behind the head, and why the record's result stands: the record's three files under the two lists hold 0 lines with the key, of 93, 62 and 62.
- [ran] No pool run is at its exact head, and I launched none. The two changed test files pass there on my machine: 48 passed.

**Next.** Slice 2, `job.json` first, when you date it. I start reading for it at my next wake.

**Least sure of:** ask A, as before. And #9867's last commit has no pool run of its own: its evidence is the local 48 and the runner's suite passing at #9919's head, where the same test file is changed further.
**Did not look at:** whether #9760's head moved under #9867.

Running: `r20261114-024813-9266` (the changed suites at `ac4869241619…`).
TIMER: one, `netseats-changed-suites-end`, about 11:20 PM PST Fri 13 Nov.

