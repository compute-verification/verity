[2026-11-10T23:17Z] 1a1488d19ea3
NET-312 LEANMERGE, turn 2, a stop on one toy (3:20 PM PST). The rest of Job 1 goes on meanwhile; nothing is pushed yet.

**The toy that cannot move as it is: `NetTiming.RowsFormat.recordFormat`** (`Security/Proofs/NetworkCertifier/RecordRows.lean` L199-204; the same on `main`). [read]
- What it needs: it is a `RecordFormat`, whose field `read_record` is a proof (`Definitions/NetworkCertifier/Tap.lean` L40), and its body fills that field with the proofs-tree theorem `NetTiming.RowsFormat.read_record` (L185-195). That theorem stands on 12 more lemmas of the same file (`certifierLane_length`, `certifierLane_fst`, `certifierCode_fst`, `getElem?_filter_countP_take`, `countP_range_div`, `filterMap_flatMap_ite`, `filterMap_bucketFrames`, `lanesOfRows_bucketFrames`, `length_bucketFrames`, `filterMap_range_getElem?`, `filterMap_finRange_getElem?`, `RowsFormat.frames_record`): 13 theorems, L32-195, Mathlib only, no ArkLib. [read]
- The witness that reads it: `NetTiming.Witness.PooledLeakAccepted_honest` alone (the side's, `PeriodWitnesses.lean` L30-37), in `(RowsFormat.recordFormat fxRows fxRows_wf 4 1).fits s`. [read: the lock run's facts, `r20261110-193753-4779`]

Three ways, yours to pick:
- (A) move the 13 theorems with it into `Definitions/NetworkCertifier/RecordRows.lean`. `tests/test_lean_packages.py`'s `test_the_spec_states_no_new_lemma` allows it in words ("a theorem a definition's proof field needs stays, and raises the cap in this change"), so `SPEC_THEOREMS` goes up by 13 on a cap already red, and that is an edit outside "Lean paths only". [read]
- (B) restate the witness's first conjunct as `fxRows.Fits 4 1 s`, which is what `.fits` of that format is by `rfl`. It then reads the definitions tree only, but its `type_hash` moves, and I have not checked that the witnesses rule still takes it as `hfit` at the instance. [argued]
- (C) leave it. #8537's `PooledLeakAccepted_honest` (`6b793802ff24`, `Witnesses.lean` L159) reads no rows format, no `fxQueueSched` and no `fxRows`, so the day it replaces the side's lemma this line is gone with no move; until then the lock keeps one `trees.reads` line, on this one statement. [read]

What I am doing without waiting: failures 2 and 3 whole (18 definitions of `WitnessToys`: the 14 named and `fxFrames`, `oneRow`, `fxCode`, `fxFrame`, which their bodies read; `obs0`, `obs1`), and `fxQueueSched` of failure 1, into `Definitions/NetworkCertifier/{Witness,RecordRows}.lean`. With `recordFormat` left in place, failure 1 will read `recordFormat` alone until your word. I lean to (C) if #8537 is on `main` before the crossing, else (A).

