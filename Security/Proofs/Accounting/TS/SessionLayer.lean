import Proofs.Accounting.TS.Session.Guarantees

/-!
# A window's proofs read through their C-Flock sessions: the bridge's premises on our side

`Flock.Guarantees.SoundAtBridge` (#2909) takes four premises on the proof layer's side. Two of them say how the layer
reads its sessions, and `Accounting.TS.onSessions` meets them by definition: `AcceptedLive`
(`onSessions_acceptedLive`) and `OpenedPlurAt` at `δo = 0` (`onSessions_openedPlur`). Its relation and rightness are
`pf`'s, so `Pins` and `RelOfRows` are `pf`'s (`onSessions_pins`, `onSessions_relOfRows`).

`onSessions_soundAt` is the bridge on it, with `WindowLaw` and `RelOfRows` as hypotheses.
`Accounting.SecurityProofs.TS.SessionSoundAt` takes them from the named premises instead: `WindowLaw` as
`Accounting.Assumptions.TS.WindowLaw`, and `RelOfRows`, for a relation that is a check at committed served leaves, as
`ServedChecksHold` and `ServedRootsLinked`. An unsound proof is then either one whose session's drawn units are all
right, and its served leaves off the committed trees' (`ServedRootsLinked`'s event, at most `δs`), or one that
`RowsSession.BadU` catches (`EndToEndRegisteredRows`, `n · bound ≤ δ`).

This module and `Proofs.Accounting.TS.Session` import `Proofs.Flock.SoundAtBridge`, whose imports make `X` a token
(ArkLib's `notation3 "X"`), so `Proofs/Accounting.lean` imports this module rather than `Proofs/Accounting/TS.lean`,
whose theorems bind `X`.
-/

namespace Accounting.SecurityProofs.TS.SessionLayer

open Accounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

variable {Ω : Type} {S nπ : ℕ} {V : Type}
variable (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
  (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V)

/-- **`AcceptedLive` by definition.** -/
theorem onSessions_acceptedLive [Fintype Ω] : AcceptedLive (onSessions pf sess out rows) sess out :=
  fun _ _ _ h => h.2

/-- **`OpenedPlurAt` at `0` by definition.** -/
theorem onSessions_openedPlur [Fintype Ω] : OpenedPlurAt (onSessions pf sess out rows) sess rows 0 :=
  openedPlur_zero (onSessions pf sess out rows) sess rows fun _ _ _ _ => rfl

/-- The relations pin on the session layer exactly when they pin on `pf`. -/
theorem onSessions_pins : (onSessions pf sess out rows).Pins ↔ pf.Pins := Iff.rfl

/-- `RelOfRows` on the session layer is `RelOfRows` on `pf`. -/
theorem onSessions_relOfRows [Fintype Ω] :
    RelOfRows (onSessions pf sess out rows) sess out rows ↔ RelOfRows pf sess out rows :=
  Iff.rfl

/-- **The bridge on the session layer**: with `AcceptedLive` and `OpenedPlurAt` met by definition,
`EndToEndRegisteredRows` gives `SoundAt δ` from its named assumptions at every session, the window's law, `RelOfRows`
on `pf`, and `n · bound ≤ δ` at every session a path runs. -/
theorem onSessions_soundAt (hE : Flock.Guarantees.EndToEndRegisteredRows) [Fintype Ω] (δ : ℝ)
    (hCR : ∀ x w i, (sess x w i).CR) (hLaw : WindowLaw sess out) (hRel : RelOfRows pf sess out rows)
    (hδ : ∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) (hδ0 : 0 ≤ δ) :
    (onSessions pf sess out rows).SoundAt δ := by
  have h := soundAtBridge_of_rows hE (onSessions pf sess out rows) sess out rows 0 δ hCR hLaw
    (onSessions_acceptedLive pf sess out rows) (onSessions_openedPlur pf sess out rows) hRel hδ hδ0
  rwa [zero_add] at h

end Accounting.SecurityProofs.TS.SessionLayer

open Accounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg in
/-- **A window read through its C-Flock sessions is sound at `δs + δ`** (`Accounting.Guarantees.TS.SessionSoundAt`):
`soundAt_of_sessions` with `ServedRootsLinked`'s event as the event off the sessions' bad events. An unsound proof is
accepted with `pf`'s relation failing at its session's value layer. If every drawn unit is right at its owned reads,
`ServedChecksHold` gives the check there, so the relation's Definition leaves only the served leaves off the committed
trees', `ServedRootsLinked`'s event; otherwise some drawn unit is wrong or off, `RowsSession.BadU`, at most
`n · bound ≤ δ` by `EndToEndRegisteredRows` (`RowsSession.prob_badU_le`). -/
theorem Accounting.SecurityProofs.TS.SessionSoundAt : Accounting.Guarantees.TS.SessionSoundAt := by
  intro Ω _ S nπ V pf sess out rows chk served δs δ hE hCR hLaw hChk hLink hRel hδ hδ0
  refine soundAt_of_sessions (onSessions pf sess out rows) (Out := fun s : RowsSession => s.Out) (fun s => s.G)
    (fun s => s.τ) (fun s => s.BadU) sess out
    (fun x w i => LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i) ∧
      (∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u) ∧
      ¬ served x w i (rows (sess x w i) (sess x w i).Xv)) δs δ hLaw hLink
    (fun x w i => (RowsSession.prob_badU_le hE _ (hCR x w i)).trans (hδ x w i)) hδ0 ?_
  rintro x w i ⟨⟨_, hL⟩, hR⟩
  by_cases hD : ∀ u ∈ (out x w i).2.1, ¬ (sess x w i).WrongAt u ∧ (sess x w i).RowsAt (out x w i) u
  · exact Or.inl ⟨hL, hD, fun hs => hR (hRel x w i _ (hChk x w i hL hD) hs)⟩
  · refine Or.inr ⟨hL, ?_⟩
    by_contra hno
    exact hD fun u hu =>
      ⟨fun hw => hno ⟨u, hu, Or.inl hw⟩, Classical.byContradiction fun hr => hno ⟨u, hu, Or.inr hr⟩⟩
