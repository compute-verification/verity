module

public import Proofs.Flock.SpotFlock.RunsAs.Steps

/-!
# What a stage's check of challenges with no code says

Each stage of an accepted two-stage run got no code from `Run.coinsCode` on the bytes it read (`accepted_steps`). Where
it read any, the verifier holds a key, a window and a family, the record has a digest and the challenger's check finds
no reason under them (`coinsCode_read`); where it read none, it carries no evidence (`coinsCode_empty`).

`RunCoinsFromDevice` (`RunsAs/Statement.lean`) is proved from `coinsCode_read` in
`Proofs.Properties.Challenger.RunCoins`, beside the proof it reads of the challenger's sample
(`Proofs.Properties.Challenger.Drawn`), which this file does not import.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- **A check of challenges with no code on bytes read**: where `coinsCode` gives no code on a non-empty read, the
verifier holds a key, a window and a family, the record has a digest, and the challenger's check finds no reason
under them after it. -/
theorem coinsCode_read {key : Option Spot.Challenger.ChallengerKey} {main coins record : Json}
    {window family : Option String} {read : ByteArray}
    (h : Spot.Run.coinsCode key main coins record window family read = none) (hr : 0 < read.size) :
    ∃ k w label after, key = some k ∧ window = some w ∧ family = some label ∧
      Spot.Registration.digest record = .ok after ∧
      Spot.Challenger.coinsReason k main coins w label after read = none := by
  have hne : read.isEmpty = false := by
    unfold ByteArray.isEmpty
    simp only [beq_eq_false_iff_ne]
    omega
  unfold Spot.Run.coinsCode at h
  rw [hne] at h
  cases hd : Spot.Registration.digest record with
  | error e => cases key <;> cases window <;> cases family <;> simp [hd] at h
  | ok after =>
    cases key with
    | none => cases window <;> cases family <;> simp at h
    | some k =>
      cases window with
      | none => cases family <;> simp at h
      | some w =>
        cases family with
        | none => simp at h
        | some label => exact ⟨k, w, label, after, rfl, rfl, rfl, rfl, by simpa [hd] using h⟩

/-- **A check of challenges with no code on no byte** carries no evidence. -/
theorem coinsCode_empty {key : Option Spot.Challenger.ChallengerKey} {main coins record : Json}
    {window family : Option String} {read : ByteArray}
    (h : Spot.Run.coinsCode key main coins record window family read = none) (hr : read.size = 0) :
    coins = Json.null := by
  have he : read.isEmpty = true := by
    unfold ByteArray.isEmpty
    simp only [beq_iff_eq]
    exact hr
  unfold Spot.Run.coinsCode at h
  rw [he] at h
  cases coins <;> first | rfl | simp at h

end SpotFlock.RunsAs
