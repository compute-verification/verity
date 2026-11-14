[2026-11-14T19:33Z] baa05df024ab
## Coordinator, Sat 14 Nov 19:33Z: MEMBERS2: two later merges in OneStageZk

From the proofs lead: for your mend of `Spot.OneStageZk`, two merges touched that area after the failed build and may matter: #9821 at `a681119c7ad` (its three declarations were placed after `membersOf` in `OneStageZkStatement.lean`, since `architecture` split `OneStageZk.lean`) and #10049's merge at 19:24Z ("the public file's header and rows held for every member"). Mend on the current tip, not the failed build's.

