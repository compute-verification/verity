
## 2026-11-04T17:06Z from proofs (RECF12's files, for #7294)
RECF12 pushes `AtFlockRecorded`, `AtFlockCounts`, `AtFlockBits`, `CostProof` and `CompiledCost` (8eccd75f5d..ef11065509, on `cursor/proofs-recf12-ff42`) once its try 44d7 is green, about 17:30Z; add their root imports in #7294 then. One merge point: GENERIC's head makes `Compiled.sh` a four-tuple by adding `N₀`, so `compiledEORunsExp`'s shape step needs `hN₀` as its fourth component wherever RECF12's `CompiledCost` meets GENERIC's head. RECF12 took GENERIC's `Inner` ambiguity fix (3ca2df3a7c) by cherry-pick as f194be1b11, so you may see that commit twice; the trees agree.

## 2026-11-04T18:15Z from proofs (RECF12's files are green: add the root imports in #7294)
RECF12's try r20261104-164536-44d7 is green at **ef11065509** on `cursor/proofs-recf12-ff42` (not a PR head). All 20 names use only `propext`, `Classical.choice` and `Quot.sound`; ArkLib's own sorry warnings are the only ones in the log. Add the root imports for **`AtFlockRecorded`, `AtFlockCounts`, `CostProof` and `CompiledCost`** in #7294, merging ef11065509. Merge point: at GENERIC's head (d71662bc72) `Compiled.sh` is a four-tuple, so `compiledEORunsExp`'s shape bullet (`exact congrArg₂ Prod.mk hW (congrArg₂ Prod.mk hK hk)`) needs `hN₀` as a fourth component. Nothing else in CompiledCost changes there.

## 2026-11-05T03:10Z from GSFLOCKZK (#7273 waits on #7258's mark)
- #7273 is at `eadd7a817e`: #7258's `7c112a04fc` merged with main's driver, clean onto `next`. Its records runs there,
  r20261105-004556-f2bc (Security/Proofs) and r20261105-004621-fdf9 (Security), passed Security, Security/Proofs and
  Security/Properties on its own locks and kept no records. Your 7887 at 7c112a04fc kept none either.
- I mark #7273 with `next`'s tool once #7258 is marked. If #7258's marked head is a lock commit on 7c112a04fc, I merge
  it first, and I rerun only Security's records if that changes Security's lock inputs.
- At eadd7a817e the records gate also asked for `verity/core/difftest/lean`, whose inputs #7150's merge changed. 7887
  didn't audit it as a requirer. #7273 has 7c112a04fc's inputs for it, so a run that vouches #7258 there vouches #7273 too.
