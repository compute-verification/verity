module

public import Proofs.Flock.SpotFlock.RunsAs.Steps

/-! What a stage's check of challenges with no code says (`coinsCode_read`, `coinsCode_empty`). -/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

theorem coinsCode_read {key : Option Spot.Challenger.ChallengerKey} {main coins record : Json}
    {window family : Option String} {read : ByteArray}
    (h : Spot.Run.coinsCode key main coins record window family read = none) (hr : 0 < read.size) :
    ∃ k w label after, key = some k ∧ window = some w ∧ family = some label ∧
      Spot.Docket.digest record = .ok after ∧
      Spot.Challenger.coinsReason k main coins w label after read = none := by
  have hne : read.isEmpty = false := by
    unfold ByteArray.isEmpty
    simp only [beq_eq_false_iff_ne]
    omega
  unfold Spot.Run.coinsCode at h
  rw [hne] at h
  cases hd : Spot.Docket.digest record with
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
