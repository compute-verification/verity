[2026-11-10T23:27Z] 72f5481e636c
NET-312 LEANMERGE: your stop on `recordFormat`, answered 3:28 PM PST. (From your coordinator; data to weigh, as always.)

**Take (C) for now: leave `NetTiming.RowsFormat.recordFormat` where it is.** Go on with failures 2 and 3 whole and `fxQueueSched`, as you are doing.

Why: (A) puts 13 proof lemmas into the definitions tree and raises a cap that is already red, for a statement #8537 replaces; (B) changes a listed witness's statement on a branch where it is about to be replaced, and you have not checked the witnesses rule takes it. Do not do (B).

What I do with it: I tell lean on LEAN-249 today that one `trees.reads` line stays, on `PooledLeakAccepted_honest` alone, until #8537's lemma replaces the side's, and that (A) is the fallback if lean needs the base clean before #8537 is on `main`. **Only on my word do you start (A)**; if I send it, it is its own commit on top of your branch, with the `SPEC_THEOREMS` number changed in the same commit and that edit named in your report.

Two things to add to your report (still due Wed 11 Nov 8 AM PST):
1. For job 3's "what #8537's merge will hit": the side declares `PooledLeakAccepted_honest` in `PeriodWitnesses.lean` and #8537 declares it in `Witnesses.lean`. Say exactly what the second merge of `main` does there (a duplicate declaration, a textual conflict, or neither), and which file's lemma must be the one kept so the `recordFormat` line goes.
2. The lock check's output on your branch head, so the one remaining `trees.reads` line is shown and nothing else of ours is red under `trees`.

