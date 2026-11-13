[2026-11-13T02:05Z] 4f28f29454cd
Daniel asked at 5:52 PM PST for the exact two-stage statement, its error term and what it quantifies existentially. I answered from `SpotKSTwo`, `RTwo` and your `Leg.lean` and `Claim.lean` at `99da2b918923`, and told him this:
at Friday's shape (8 outer blocks, all drawn; 4 of 64 inner blocks per cell) one incorrect inner block in each of m+1 outer blocks is missed with probability (15/16)^(m+1). So `CapsAt dd Pt m δ` holds only at δ ≥ (15/16)^(m+1): 0.94 at m = 0, 0.60 at m = 7. The claim counted in outer blocks (`UnitsAreReplay`) says almost nothing there. The count of inner blocks does: at δ = 1/2, at most 10 of 10,240.
So the corollary you put after Friday (your 00:59Z entry: `CellCapsAt` beside `CapsAt`, the vocabulary's inner blocks mapped onto the cells in place of `UnitsAreReplay`) comes forward, as a second statement beside the outer-block one, replacing nothing:
1. State `CellCapsAt`, the map and the corollary in `not_rTwo_of_wrong`'s and `holds_of_rTwo`'s shape, in `Claim.lean` or a file beside it. Prove the bookkeeping if it is under an hour.
2. It does not displace the coupling's pieces or the form's pull request #9574. If it would, say so and stop at the statements.
3. Which of the two the top-level target reads is Daniel's call; I recommended the inner blocks. Do not change `TwoLeg` or `UnitsAreReplay` until he answers.
Entry with the statements as written, and whether `Claim.lean` has compiled (I could not find a run that says so).

