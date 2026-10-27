import Proofs.Accounting.TS.Session.Guarantees

/-!
# The served layer is sound through its C-Flock sessions (the proof)

`Accounting.SecurityProofs.TS.SessionSoundAt` is `Flock.SoundAtBridge.soundAt_of_sessions` on the served layer `pf`,
with `ServedRootsLinked`'s event as the event off the sessions' bad events. It needs neither `RelOfRows` nor
`OpenedPlurAt` on every path: the named premises state them only on the accepted sessions whose drawn units are right,
and every other unsound proof is `RowsSession.BadU`.

This module imports `Proofs.Flock.SoundAtBridge`, whose imports make `X` a token (ArkLib's `notation3 "X"`), so
`Proofs/Accounting.lean` imports it rather than `Proofs/Accounting/TS.lean`, whose theorems bind `X`.
-/

open Accounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg in
/-- **A window of proofs is sound at `δs + δ` through its C-Flock sessions**
(`Accounting.Guarantees.TS.SessionSoundAt`). An unsound proof is accepted, so flock-verify accepted its session
(`AcceptedLive`), and its relation fails at its opening. If every drawn unit is right at its owned reads,
`ServedChecksHold` gives the relation at the session's value layer, so that layer isn't the opening:
`ServedRootsLinked`'s event, at most `δs`. Otherwise some drawn unit is wrong or off, `RowsSession.BadU`, at most
`n · bound ≤ δ` by `Flock.SecurityProofs.EndToEndRegisteredRows` (`RowsSession.prob_badU_le`). -/
theorem Accounting.SecurityProofs.TS.SessionSoundAt : Accounting.Guarantees.TS.SessionSoundAt := by
  intro Ω _ S nπ V pf sess out rows δs δ hCR hAcc hLaw hChk hLink hδ hδ0
  refine soundAt_of_sessions pf (Out := fun s : RowsSession => s.Out) (fun s => s.G)
    (fun s => s.τ) (fun s => s.BadU) sess out
    (fun x w i => LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) ∧
      (∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u) ∧
      rows (sess x w i) (sess x w i).Xv ≠ pf.opened x w i) δs δ hLaw hLink
    (fun x w i => (RowsSession.prob_badU_le Flock.SecurityProofs.EndToEndRegisteredRows _ (hCR x w i)).trans
      (hδ x w i)) hδ0 ?_
  rintro x w i ⟨hA, hR⟩
  have hL := hAcc x w i hA
  by_cases hD : ∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u
  · refine Or.inl ⟨hL, hD, fun hs => hR ?_⟩
    rw [← hs]
    exact hChk x w i hL hD
  · refine Or.inr ⟨hL, ?_⟩
    by_contra hno
    exact hD fun u hu =>
      ⟨fun hw => hno ⟨u, hu, Or.inl hw⟩, Classical.byContradiction fun hr => hno ⟨u, hu, Or.inr hr⟩⟩
