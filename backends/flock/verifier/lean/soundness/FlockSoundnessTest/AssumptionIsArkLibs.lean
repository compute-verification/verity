import FlockSoundness.Assumptions

/-!
The named assumption is literally ArkLib's pinned statement: ArkLib's admitted theorem proves it
by `exact`, with no rewriting. (This test depends on ArkLib's `sorry`, which is the point: it
checks the statement, not a proof. Nothing in `FlockSoundness` imports it.)
-/

example : FlockSoundness.Assumptions.BCHKS25Thm46 :=
  fun domain k δ hk hδ h => CodingTheory.rs_mcaError_le_in_johnson_range domain k δ hk hδ h
