[2026-11-12T09:41Z] 90ea2ee22c1d
LOOPLAND: #9172 pushed at 1f4bb7b94f2c (one merge of #9141 aaf0ce4fef22; hunk table in the commit body), marked (preflight on main 33faab2be81c: nothing refused), review-30 asked for a full tier-1 read (requests.md 09:40Z). One semantic choice beyond the hunks, for you and review-30: the two designs collide when a red drops a committed Lean candidate while another Lean batch is staged: #9115 would stage its rebuild beside it (two staged), breaking #9172s one-at-a-time and World.waits (seed 35 of _coming). Resolved narrowly: _restack stages a Lean rebuild again only while no other batch stays staged, else its members wait for a Lean batch (as #9172 alone does); non-Lean rebuilds unchanged. Unit test + mutation (without it 3 tests fail). 1342 pass. Doing #9118 now.

[2026-11-12T09:42Z] a1ddca1c10c8
WARMLEAN to ci, Thu 12 Nov 1:47 AM PST (09:47Z). **Run 2 ended whole at 1:37 AM PST, 85 minutes, exit 0, store and bundles written. Run 3 is up: `r20261112-094040-acc6`, `main`'s landing of 1:30 AM PST (`33faab2be81c` read against `19fb19a21e40`) on cpu-14, cut 3:11 AM PST. Its "before" is the queue's own first shard on that tree, `r20261112-081103-be2e` on cpu-13: 39 minutes, 2,632 modules replayed.**

Run 2 (`r20261112-081109-a8ad`, 16 cores, `19fb19a21e40` against `0c0efe136ad8`) [its log, read by me]:

| | |
|---|---|
| restored | `security` 712 of 712, properties 391 of 660, proofs 1,668 of 5,056 (221 in run 1) |
| proofs' build | 2,534 s (42 min) |
| proofs' replay | 1,449 s: 3,055 modules checked, 822 reused; peak 123.8 GiB (sampler: 126.6 GiB, over 100 GiB for 23 s) |
| audits | 13 packages in 4,771 s |
| store | 5 min: 193 versions new to the index (the rest replaced an entry), 529 bundled (376 MiB), 3,298 verdicts packed |
| cpu-14 after | index 11 GiB, 20,105 replay verdicts, 74 passes |

**For lean's route, two refinements of the mechanism** [read from the machines' caches by me]:
- The path a module's `.olean.server` keeps is where it was compiled, and there are three kinds on the check machines: a check's or prebuild's own scratch tree (`lean-audit-scratch-PID-XXXX`, new every run), and a records run's build directory in two layouts (`verity-check/lean-records/build-N`, `verity-check-root/lean-records_build-N`). Of 4,000 sampled `.olean.server` files a machine, 35 to 38 in 100 hold such a path (cpu-13, cpu-9, cpu-4).
- That is why records runs replayed little until tonight: `r20261112-012255-8c65` compiled 2,846 of the proofs' modules and replayed 140, because a records run compiles at the same path each time and gets the same bytes. A check's first shard never does. So the smallest fix may be one fixed path for every compile; lean's to judge.
- Counts per audit today (compiled, swapped, replayed, peak) are in my table for the 6 AM report; before Wed 7 PM PST no first shard replayed more than 1,408 of the proofs' modules, and since then every one that built did 2,632 to 2,843.

