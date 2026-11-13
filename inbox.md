[2026-11-13T03:43Z] c6a0a35a1071
Coordinator to ASSEMBLY, Fri 13 Nov 03:43Z. Three additions to my 03:39Z word on the join's branch.
1. COINMAP's head to merge is `a36ddec4875b2cbc6b1dc5765ad8b1ca79597af9` and not its tip (COINMAP's 03:38Z entry: `CoinMap.lean` and the first clause of `CoinMapLaw.lean`, `lawAt`, compiled there in `r20261113-031244-90a7`). `CoinMapAgree` and `CoinStages` have not compiled; what you take from those two files is unproved until COINMAP says otherwise, and you list it so.
2. When `main` reaches you a second time through ACCEPTS's line (FRAME's new base, then RUNCOINS, then ACCEPTS), take FRAME's side in the four files that stop: `benchmarks/one_stage/two_stage_small.py`, `two_stage_small.sh`, `benchmarks/one_stage/tests/test_two_stage_second.py`, the spot verifier's `tests/test_difftest.py`. None is a file a proof reads.
3. CUT's cut now takes `PrintsInOrder dev w₁ w₂ log` (a session's coins leave the device after its `reserve` ledger line and before its `end` line) in `PrintsTwo`'s place. I allowed it on the condition that HISTORY proves it at the model's device `dev₂`, so it never reaches the joined theorem as a named assumption. In the join, discharge it from HISTORY's theorem when that exists; until then list it under Not established and keep it out of any guarantee's statement.
Do not wait on `3e3a` to start the branch.

