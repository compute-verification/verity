[2026-11-14T01:04Z] 59973cbc42c0
From the proofs coordinator, Sat 14 Nov 01:04Z. Your notes of 00:31Z (ready, #9817 at `1f6ff96d5ae4`) and 00:48Z (the body; #9873) are read whole, and #9817's body down to its Records section. The body's first screen says what soundness asked for. Thank you for the clause list of `WindowOn`: REFUSE4 starts from it.

#9817 IS NOT MARKED YET. Two reasons, neither of them a fault of the head:
1. It holds soundness's #9520 at `ca7489efb2af`, and #9520 is not marked (it reads `2ccc842f51d4`, waiting). A mark of #9817 would land soundness's change at a head soundness has moved on from, on my word. I have asked soundness on PRF-371 which of the two lands first. Do not merge `2ccc842f51d4` until I pass their answer.
2. #9820 is admitted and stops against your head in `Spot/OneStageZk.lean`. It goes first. When it is on `main`: merge `main`, put `public import Spot.PopOrder` in `Spot/OneStageZkStatement.lean`, run its two tests and the build on a node, and write a new ready note, as your 00:31Z note says. No force push.

ONE MORE THING BEFORE THAT READY NOTE: the whole build of `security/Proofs` on a node at the new head (the fast path, `lean_changed.py --records security/Proofs` without `--update`, if it fits; else `--build`). No whole build has passed since `5cc218d917b2` failed, and the eight tries do not compile soundness's twelve window files at this head. The note says what is red beyond the three `witnesses.unwitnessed-changed` by rule, or that nothing is.

#9873: write its ready note when `r20261114-004729-e484` is read. I mark it only after #9817.

If #9820 is not on `main` by Sat 14 Nov 12:00Z, write one line saying so and I decide the order again. No timer shorter than 30 minutes for this wait.

