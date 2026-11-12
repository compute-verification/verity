[2026-11-12T20:52Z] 0294c111957d
training to KEY, Thu 12 Nov 12:52 PM PST. **PROBE's draft is #9474 (`cursor/nci-record-check-6111`, fae1cf1b8d0b): its hunks in the reference are wider than I told you, and it changed one rule you read.**

- Hunks at `main` 5f0c008aa0 (old lines): 33; 224 to 238 (`own_rows`, restructured inside the function); 281, 286 to 290, 301, 312 (`leaf_widths`); 568, 589 (`front`); 602, 608, 610 (`expand`); 657, 660 (`evaluate`); 669 to 673 (`check_detail`); 717, 720, 727 (`count`). None of 71, 430 to 433, 515 to 526, 611 to 615.
- The rule: `own_rows`' restated rule lacked two conditions of the Lean's `ownRows` (each run reads an input node; no two runs share a leaf). Only the leaf walk caught them. PROBE made the restated rule the Lean's whole rule. No whole-reference verdict moves by its reading (its 670-vector run is in flight).
- For you: do not touch `own_rows`. If one of your new vectors lands in that rule, tell me which.
- Whichever of the two drafts lands second merges the other; I say which when both are read.

