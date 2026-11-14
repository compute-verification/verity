[2026-11-14T21:17Z] 2af23951e577
## Coordinator, Sat 14 Nov 21:17Z: STAGECOINS: the six label errors are yours

From the proofs lead: good trace on `Spot/Ranks.lean` (WINDOWREG has the ring now; lean is told). The six "challenges are of the family 'positions'" `ValueError`s: their branch's worker is gone, so they are YOURS to mend: the calls at `test_two_stage_second.py` lines 372, 443 and 655 pass `FC.Challenger("first")` with no label; give them `label=TS.family_of(layout)` as `fd71e0c8bdc` gave its own tests' calls, after checking that the label is what the test means (the family the layout's challenges are of), not just what makes the check pass. Push with your next commit and list it.

