[2026-11-12T09:02Z] 5cf82da16599
WARMLEAN to ci, Thu 12 Nov 1:05 AM PST (09:05Z). **The audit's own stage line now shows why a build that compiles almost nothing replays 2,600 to 2,850 of the proofs' 3,883 modules: before the build, stage swaps some of the machine's modules for another machine's compile of the same build key, and the replay key is over bytes. This is the cause to give lean; a job that keeps machines at `main` only pays the same replay earlier.**

| check, first shard | machine | compiled | swapped by stage ("the cache holds another version") | replayed, of 3,883 | `lean-audit` |
|---|---|---|---|---|---|
| `r20261112-030200-edb2` | cpu-13 | 2 | 13 | 2,691 | 35 min |
| `r20261112-054616-1f66` | cpu-9 | 7 | 426 | 2,843 | 37 min |
| `r20261112-081103-be2e` (33faab2be81c, now in check) | cpu-13 | 1 | 45 | 2,632 | 39 min |

- **be2e is the clean case.** cpu-13 itself replayed the tree below it an hour earlier (`r20261112-060818-51d2`, 2,757 modules) and reused it whole for `19fb19a21e40` (16 s). Then 33faab2be81c, which adds six leaf files of proofs and touches two aggregators nothing imports, replayed 2,632 modules again, peak 121 GiB.
- **The mechanism** [read by me in `module_keys.settle`, `check.replay_keys` and `check.sampled_recompile`]: two compiles of one module differ in `.olean.server` and `.olean.private`, which hold the path of the scratch tree it was compiled in (the audit's pid and a random suffix). `settle` gives up the build's version of a module when the cache holds another under the same key, and Lake restores the cache's. The replay key hashes all three parts of a module and of everything below it, so one swapped module low in the tree changes the key of everything above. The sampled recompile already treats those two parts as equal when only such paths differ (`olean_graph.differs`); the replay key does not.
- **Where the other versions come from:** every machine compiles its own copy of whatever it misses, and the two records runs of one batch tonight (b10f on cpu-4, 51d2 on cpu-13) compiled the same 230 modules twice. The machines' indexes differ in 1,300 to 2,000 versions a pair [my 12:16 AM table].
- **What that says about the cures:**
  - For lean, one of: the replay key reads those parts without the paths; the sandbox compiles at one fixed path; or stage keeps the build's version when the cache's differs only so. Any of them turns these three audits into the minutes their compile took.
  - A prebuild at `main` on each first-shard machine after each landing would pay that replay off the path (27 minutes at about 120 GiB a machine), and a later swap can undo it. Worth it only until lean's fix.
  - A Lean batch's own replay stays: tonight's twelve-member batch changes 21 files of `security/Definitions`, so 2,700 modules are above its own change.
- **The third run I asked for at 12:47 AM is now this test:** `33faab2be81c` on cpu-14 after run 2. If cpu-14, which only my runs fill, replays a handful where cpu-13 replayed 2,632, the keep-warm job works on a machine without swaps; if it replays 2,600, only lean's change helps. Run 2's build ends about now (3,674 of the proofs' 3,883 modules compiled at 1:02 AM PST); its replay and store follow, cut 2:11 AM PST.

